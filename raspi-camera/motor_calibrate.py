#!/usr/bin/env python3
"""
Motor calibration script - measure turn angles and distances.

Runs timed motor pulses for manual observation and measurement.
Use the results to determine how long to run motors for specific
turns (90°, 180°) and distances.

Requires pigpio daemon: sudo systemctl start pigpiod

Usage:
    python3 motor_calibrate.py
"""

import pigpio
import time
import sys

# GPIO pin assignments
LEFT_MOTOR = 18   # Pin 12
RIGHT_MOTOR = 13  # Pin 33

# Motor direction
LEFT_INVERTED = True
RIGHT_INVERTED = True

# PWM
NEUTRAL = 1500
DEFAULT_SPEED = 110  # Same as follow_red.py


class MotorController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpio daemon. Run: sudo systemctl start pigpiod")
        self.stop()

    def set_motors(self, left_us, right_us):
        if LEFT_INVERTED:
            left_us = 3000 - left_us
        if RIGHT_INVERTED:
            right_us = 3000 - right_us
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, left_us)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, right_us)

    def stop(self):
        self.set_motors(NEUTRAL, NEUTRAL)

    def turn_left(self, speed=DEFAULT_SPEED):
        self.set_motors(NEUTRAL - speed, NEUTRAL + speed)

    def turn_right(self, speed=DEFAULT_SPEED):
        self.set_motors(NEUTRAL + speed, NEUTRAL - speed)

    def forward(self, speed=DEFAULT_SPEED):
        self.set_motors(NEUTRAL + speed, NEUTRAL + speed)

    def reverse(self, speed=DEFAULT_SPEED):
        self.set_motors(NEUTRAL - speed, NEUTRAL - speed)

    def cleanup(self):
        self.stop()
        time.sleep(0.1)
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, 0)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, 0)
        self.pi.stop()


def countdown(seconds=3):
    """Countdown before running a test."""
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO!")


def run_test(motors, action, speed, duration):
    """Run a single motor test."""
    action_fn = {
        "left": motors.turn_left,
        "right": motors.turn_right,
        "forward": motors.forward,
        "reverse": motors.reverse,
    }[action]

    countdown()
    action_fn(speed)
    time.sleep(duration)
    motors.stop()
    print(f"  STOP - ran {action} at speed {speed} for {duration:.2f}s")


def turn_calibration(motors):
    """Run turn tests at different durations."""
    print()
    print("TURN CALIBRATION")
    print("=" * 40)
    print("The robot will turn in place at each duration.")
    print("Observe how many degrees it rotates each time.")
    print()

    speed = DEFAULT_SPEED
    durations = [0.25, 0.5, 0.75, 1.0]

    for direction in ["right", "left"]:
        print(f"--- Turn {direction.upper()} tests (speed={speed}) ---")
        for dur in durations:
            input(f"\n  Press Enter to turn {direction} for {dur}s...")
            run_test(motors, direction, speed, dur)
            result = input(f"  How many degrees did it turn? (or press Enter to skip): ").strip()
            if result:
                deg_per_sec = float(result) / dur
                print(f"  => {deg_per_sec:.1f} degrees/sec at speed {speed}")
                time_for_90 = 90.0 / deg_per_sec
                time_for_180 = 180.0 / deg_per_sec
                print(f"  => 90° would take {time_for_90:.2f}s, 180° would take {time_for_180:.2f}s")
        print()


def forward_calibration(motors):
    """Run forward tests at different durations."""
    print()
    print("FORWARD CALIBRATION")
    print("=" * 40)
    print("The robot will drive forward at each duration.")
    print("Measure how far it travels (in cm or inches).")
    print()

    speed = DEFAULT_SPEED
    durations = [0.5, 1.0, 1.5, 2.0]

    print(f"--- Forward tests (speed={speed}) ---")
    for dur in durations:
        input(f"\n  Press Enter to drive forward for {dur}s...")
        run_test(motors, "forward", speed, dur)
        result = input(f"  How far did it go? (e.g. '15cm' or '6in', or Enter to skip): ").strip()
        if result:
            # Parse number from input
            num = ''.join(c for c in result if c.isdigit() or c == '.')
            if num:
                dist = float(num)
                dist_per_sec = dist / dur
                unit = ''.join(c for c in result if c.isalpha()) or 'units'
                print(f"  => {dist_per_sec:.1f} {unit}/sec at speed {speed}")
    print()


def custom_test(motors):
    """Run custom speed/duration/direction combos."""
    print()
    print("CUSTOM TEST")
    print("=" * 40)
    print("Enter custom parameters to fine-tune.")
    print("Type 'done' to return to menu.")
    print()

    while True:
        print("Actions: left, right, forward, reverse")
        action = input("Action (or 'done'): ").strip().lower()
        if action == 'done':
            break
        if action not in ('left', 'right', 'forward', 'reverse'):
            print("Invalid action")
            continue

        try:
            speed = input(f"Speed [{DEFAULT_SPEED}]: ").strip()
            speed = int(speed) if speed else DEFAULT_SPEED

            duration = input("Duration in seconds [0.5]: ").strip()
            duration = float(duration) if duration else 0.5
        except ValueError:
            print("Invalid number")
            continue

        run_test(motors, action, speed, duration)
        print()


def main():
    print("Motor Calibration")
    print("=" * 40)
    print()

    try:
        motors = MotorController()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Motors initialized. Place the robot on a flat surface.")
    print()

    try:
        while True:
            print("Menu:")
            print("  1. Turn calibration (preset durations)")
            print("  2. Forward calibration (preset durations)")
            print("  3. Custom test (specify action/speed/duration)")
            print("  q. Quit")
            print()

            choice = input("Choice: ").strip().lower()

            if choice == '1':
                turn_calibration(motors)
            elif choice == '2':
                forward_calibration(motors)
            elif choice == '3':
                custom_test(motors)
            elif choice == 'q':
                break
            else:
                print("Invalid choice")
                print()

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up...")
        motors.cleanup()
        print("Done")


if __name__ == "__main__":
    main()
