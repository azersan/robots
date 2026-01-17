#!/usr/bin/env python3
"""
Local CV processing - runs on your Mac.
Pulls video from Pi and does all processing locally.

Usage:
    python3 local_cv.py              # Use default Pi address
    python3 local_cv.py 192.168.4.80 # Specify Pi address
"""

import cv2
import numpy as np
import time
import sys

# Pi stream URL
PI_HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.80"
STREAM_URL = f"http://{PI_HOST}:8080/stream"

# Color tracking settings (same as before)
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])
MIN_AREA = 500  # For 320x240 stream


def process_frame(frame):
    """Detect red blobs and return annotated frame with tracking info."""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tracking info to return
    tracking = {"detected": False, "cx": 0, "cy": 0, "radius": 0}

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            tracking["detected"] = True

            # Get bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(largest)
            tracking["radius"] = radius

            # Get centroid
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                tracking["cx"] = cx
                tracking["cy"] = cy

                # Draw circle and center
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw center line and direction hint
                frame_center = frame.shape[1] // 2
                cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]),
                        (100, 100, 100), 1)

                # Direction indicator
                if cx < frame_center - 50:
                    direction = "<< LEFT"
                    dir_color = (0, 255, 255)
                elif cx > frame_center + 50:
                    direction = "RIGHT >>"
                    dir_color = (0, 255, 255)
                else:
                    direction = "CENTER"
                    dir_color = (0, 255, 0)

                cv2.putText(frame, direction, (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)

    # Add mask preview
    mask_small = cv2.resize(mask, (160, 120))
    mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_color, (0, 0), (159, 119), (0, 255, 0), 2)
    frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = mask_color

    return frame, mask, tracking


def main():
    print(f"Connecting to Pi stream at {STREAM_URL}")
    print("Press 'q' to quit, 's' to save screenshot")
    print()

    # Open video stream
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print(f"ERROR: Could not connect to {STREAM_URL}")
        print("Make sure the Pi is running stream_raw.py")
        return

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        # Process frame
        frame, mask, tracking = process_frame(frame)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Draw FPS and status
        status = "TRACKING" if tracking["detected"] else "SEARCHING"
        status_color = (0, 255, 0) if tracking["detected"] else (0, 100, 255)

        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Upscale for display (stream is 320x240 for performance)
        display = cv2.resize(frame, (640, 480))
        cv2.imshow("Pi Camera - Local CV", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
