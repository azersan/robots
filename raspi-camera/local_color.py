#!/usr/bin/env python3
"""
Color tracking with local CV processing.
Supports local webcam or remote Pi stream.

Usage:
    python3 local_color.py              # Default: Pi stream
    python3 local_color.py --local      # Use Mac webcam
    python3 local_color.py --source IP  # Pi at specific IP
"""

import cv2
import numpy as np
import time
import video_source
import red_detect

MIN_AREA = 1500  # Larger for 640x480


def process_frame(frame):
    """Detect red blobs and return annotated frame with tracking info."""
    mask, blob = red_detect.largest_red_blob(frame, MIN_AREA)

    # Tracking info to return
    tracking = {"detected": False, "cx": 0, "cy": 0, "radius": 0, "area": 0}

    if blob:
        tracking.update(detected=True, cx=blob["cx"], cy=blob["cy"],
                        radius=blob["radius"], area=blob["area"])

        # Draw circle and center
        cv2.circle(frame, (int(blob["x"]), int(blob["y"])),
                   int(blob["radius"]), (0, 255, 0), 2)
        cv2.circle(frame, (blob["cx"], blob["cy"]), 5, (0, 255, 0), -1)

        # Draw center line
        frame_center = frame.shape[1] // 2
        cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]),
                (100, 100, 100), 1)

    return frame, mask, tracking


def create_side_panel(mask, tracking, fps, frame_height):
    """Create an info panel showing mask and stats."""
    # Create panel
    panel_width = 250
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Add mask preview at top (scaled to fit panel width)
    mask_w = panel_width - 20  # 10px margin on each side
    mask_h = int(mask_w * mask.shape[0] / mask.shape[1])
    mask_scaled = cv2.resize(mask, (mask_w, mask_h))
    mask_color = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_color, (0, 0), (mask_w-1, mask_h-1), (0, 255, 0), 1)

    # Place mask preview
    panel[10:10+mask_h, 10:10+mask_w] = mask_color

    # Add label
    cv2.putText(panel, "MASK VIEW", (10, 10 + mask_h + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Add stats
    y_offset = 10 + mask_h + 55

    # FPS
    cv2.putText(panel, f"FPS: {fps}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 30

    # Status
    if tracking["detected"]:
        status = "TRACKING"
        status_color = (0, 255, 0)
    else:
        status = "SEARCHING"
        status_color = (0, 100, 255)

    cv2.putText(panel, f"Status: {status}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    y_offset += 30

    if tracking["detected"]:
        # Position
        cv2.putText(panel, f"Pos: ({tracking['cx']}, {tracking['cy']})", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30

        # Direction
        frame_center = 320  # Half of 640
        if tracking["cx"] < frame_center - 50:
            direction = "<< LEFT"
            dir_color = (0, 255, 255)
        elif tracking["cx"] > frame_center + 50:
            direction = "RIGHT >>"
            dir_color = (0, 255, 255)
        else:
            direction = "CENTER"
            dir_color = (0, 255, 0)

        cv2.putText(panel, f"Dir: {direction}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dir_color, 1)
        y_offset += 30

        # Size (rough distance indicator)
        cv2.putText(panel, f"Radius: {int(tracking['radius'])}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return panel


def main():
    # Parse video source arguments
    args = video_source.parse_args(description="Color tracking")

    # Open video source
    cap = video_source.get_capture(args)
    source_desc = video_source.get_source_description(args)
    print("Press 'q' to quit, 's' to save screenshot, 'r' to reconnect")
    print()

    if not cap.isOpened():
        print(f"ERROR: Could not open video source")
        if not video_source.is_local(args):
            print("Make sure the Pi is running robot_server.py")
        return

    print("Connected! Stream should appear shortly...")

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            time.sleep(1)
            cap = video_source.reconnect(args, cap)
            continue

        # Process frame
        frame, mask, tracking = process_frame(frame)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(mask, tracking, fps, frame.shape[0])

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow(f"Color Tracking - {source_desc}", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting...")
            cap = video_source.reconnect(args, cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
