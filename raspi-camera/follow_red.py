#!/usr/bin/env python3
"""
Red object follower - runs on Pi Zero W.
Detects red objects and turns toward them using motors.

Usage:
    python3 follow_red.py

Requires:
    - picamera2
    - pigpio daemon running (sudo systemctl start pigpiod)
    - OpenCV (cv2)
"""

import cv2
import numpy as np
import pigpio
import time
import signal
import sys
import os
import argparse

# GPIO pin assignments (same as motor_test.py)
LEFT_MOTOR = 18   # Pin 12
RIGHT_MOTOR = 13  # Pin 33

# Motor direction (set to True if motor runs backward)
LEFT_INVERTED = True
RIGHT_INVERTED = True

# PWM values (microseconds)
NEUTRAL = 1500
TURN_SPEED = 110  # Amount to add/subtract from neutral for turning

# Color tracking settings (red in HSV)
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])
MIN_AREA = 1000  # Minimum blob area to track

# Control parameters
CENTER_DEADZONE = 50  # Pixels from center to ignore (considered "centered")
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
LOST_TURN_TIMEOUT = 0.15  # Seconds to keep turning after losing red
LOST_FWD_TIMEOUT = 0.5    # Seconds to keep forward after losing red

# Debug output directory
DEBUG_DIR = "/tmp/follow_red_debug"


class MotorController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpio daemon. Run: sudo systemctl start pigpiod")
        self.stop()
        print("Motors initialized")

    def _apply_inversion(self, left_us, right_us):
        """Apply motor inversion."""
        if LEFT_INVERTED:
            left_us = 3000 - left_us
        if RIGHT_INVERTED:
            right_us = 3000 - right_us
        return left_us, right_us

    def set_motors(self, left_us, right_us):
        """Set motor speeds in microseconds (1000-2000)."""
        left_us, right_us = self._apply_inversion(left_us, right_us)
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, left_us)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, right_us)

    def stop(self):
        """Stop both motors."""
        self.set_motors(NEUTRAL, NEUTRAL)

    def turn_left(self, speed=TURN_SPEED):
        """Turn left: right forward, left reverse."""
        self.set_motors(NEUTRAL - speed, NEUTRAL + speed)

    def turn_right(self, speed=TURN_SPEED):
        """Turn right: left forward, right reverse."""
        self.set_motors(NEUTRAL + speed, NEUTRAL - speed)

    def forward(self, speed=TURN_SPEED):
        """Go forward."""
        self.set_motors(NEUTRAL + speed, NEUTRAL + speed)

    def cleanup(self):
        """Stop motors and release GPIO."""
        self.stop()
        time.sleep(0.1)
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, 0)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, 0)
        self.pi.stop()


def detect_red(frame):
    """
    Detect red blobs in frame.
    Returns (detected, cx, cy, area, mask) where cx,cy is centroid position.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red (two ranges because red wraps around hue 0/180)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            # Get centroid
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, cx, cy, area, mask

    return False, 0, 0, 0, mask


def save_debug_frame(frame, mask, state, detected, cx, cy, frame_center):
    """Save annotated debug frame and mask to disk."""
    debug = frame.copy()

    # Draw center line
    cv2.line(debug, (frame_center, 0), (frame_center, frame.shape[0]), (100, 100, 100), 1)

    # Draw deadzone
    cv2.line(debug, (frame_center - CENTER_DEADZONE, 0),
             (frame_center - CENTER_DEADZONE, frame.shape[0]), (50, 50, 50), 1)
    cv2.line(debug, (frame_center + CENTER_DEADZONE, 0),
             (frame_center + CENTER_DEADZONE, frame.shape[0]), (50, 50, 50), 1)

    if detected:
        # Draw crosshair on red object
        cv2.circle(debug, (cx, cy), 10, (0, 255, 0), 2)
        cv2.line(debug, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
        cv2.line(debug, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)

    # Add state label
    color = (0, 255, 0) if state == "CENTER" else (0, 255, 255) if detected else (0, 0, 255)
    cv2.putText(debug, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save frame and mask
    cv2.imwrite(os.path.join(DEBUG_DIR, "frame.jpg"), debug)
    cv2.imwrite(os.path.join(DEBUG_DIR, "mask.jpg"), mask)


def main():
    parser = argparse.ArgumentParser(description="Red object follower")
    parser.add_argument("--debug", action="store_true",
                       help="Save debug frames to /tmp/follow_red_debug/")
    parser.add_argument("--no-motors", action="store_true",
                       help="Disable motors (camera + detection only)")
    args = parser.parse_args()

    print("Red Object Follower")
    print("=" * 40)
    print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    if args.debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug frames: {DEBUG_DIR}/frame.jpg and mask.jpg")
    if args.no_motors:
        print("Motors DISABLED (detection only)")
    print("Press Ctrl+C to quit")
    print()

    # Initialize camera
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        print("Camera started")
    except Exception as e:
        print(f"Error starting camera: {e}")
        return

    # Initialize motors
    motors = None
    if not args.no_motors:
        try:
            motors = MotorController()
        except RuntimeError as e:
            print(f"Error: {e}")
            camera.stop()
            return

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        if motors:
            motors.cleanup()
        camera.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    frame_center = FRAME_WIDTH // 2
    last_state = None
    last_seen_time = 0  # Time red was last detected
    last_direction = None  # Last turn direction when red was seen
    fps_time = time.time()
    fps_count = 0

    print("Searching for red objects...")
    print()

    try:
        while True:
            # Capture frame
            # NOTE: picamera2 "RGB888" actually outputs BGR (OpenCV convention)
            frame = camera.capture_array()

            # Detect red
            detected, cx, cy, area, mask = detect_red(frame)

            # Decide action based on red position
            if detected:
                last_seen_time = time.time()
                offset = cx - frame_center

                if offset < -CENTER_DEADZONE:
                    state = "LEFT"
                    last_direction = "LEFT"
                    # Proportional: turn faster when further from center
                    max_offset = frame_center - CENTER_DEADZONE
                    ratio = min(1.0, abs(offset) / max_offset)
                    speed = int(TURN_SPEED * (0.5 + 0.5 * ratio))
                    if motors:
                        motors.turn_left(speed)
                elif offset > CENTER_DEADZONE:
                    state = "RIGHT"
                    last_direction = "RIGHT"
                    max_offset = frame_center - CENTER_DEADZONE
                    ratio = min(1.0, abs(offset) / max_offset)
                    speed = int(TURN_SPEED * (0.5 + 0.5 * ratio))
                    if motors:
                        motors.turn_right(speed)
                else:
                    state = "CENTER"
                    last_direction = "FORWARD"
                    if motors:
                        motors.forward()
            elif last_direction and (
                (last_direction == "FORWARD" and time.time() - last_seen_time < LOST_FWD_TIMEOUT) or
                (last_direction != "FORWARD" and time.time() - last_seen_time < LOST_TURN_TIMEOUT)):
                # Recently lost red - keep turning same direction
                state = f"HOLD {last_direction}"
                if motors:
                    if last_direction == "LEFT":
                        motors.turn_left()
                    elif last_direction == "RIGHT":
                        motors.turn_right()
                    else:
                        motors.forward()
            else:
                state = "SEARCHING"
                last_direction = None
                if motors:
                    motors.stop()

            # Print state changes
            if state != last_state:
                if detected:
                    print(f"[{state}] Red at ({cx}, {cy}), area={area}")
                elif "HOLD" in state:
                    print(f"[{state}] Lost red, holding direction")
                else:
                    print(f"[{state}] No red detected")
                last_state = state

            # Save debug frames (overwrites each time so you can scp the latest)
            if args.debug:
                save_debug_frame(frame, mask, state, detected, cx, cy, frame_center)

            # FPS tracking
            fps_count += 1
            if time.time() - fps_time >= 5.0:
                fps = fps_count / 5.0
                print(f"FPS: {fps:.1f}")
                fps_count = 0
                fps_time = time.time()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        if motors:
            motors.cleanup()
        camera.stop()
        print("Done")


if __name__ == "__main__":
    main()
