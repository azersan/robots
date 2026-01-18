#!/usr/bin/env python3
"""
Body pose detection and gesture recognition using MediaPipe.
Supports local webcam or remote Pi stream.

Usage:
    python3 local_pose.py              # Default: Pi stream
    python3 local_pose.py --local      # Use Mac webcam
    python3 local_pose.py --source IP  # Pi at specific IP
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import video_source

# Model path
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# Gesture detection thresholds
ARM_RAISED_THRESHOLD = 0.15  # How much higher than shoulder to count as "raised"
ARM_OUT_THRESHOLD = 0.15     # How far out to the side to count as "out"

# Gesture log
gesture_log = []  # List of (timestamp, gesture) tuples
last_gesture = None
last_logged_gesture = None
last_log_time = 0
gesture_flash_until = 0  # Time until which to show flash effect
MIN_LOG_INTERVAL = 2.0  # Minimum seconds between logging the same gesture

# Landmark indices (MediaPipe Pose has 33 landmarks)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
NOSE = 0


def download_model():
    """Download the pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")


def log_gesture(gesture):
    """Log a gesture if it's different or enough time has passed."""
    global last_gesture, last_logged_gesture, last_log_time, gesture_flash_until, gesture_log

    # Skip non-actionable states
    if gesture in ["NO PERSON", "PARTIAL VIEW", "READY"]:
        last_gesture = gesture
        return False

    now = time.time()

    # Only log if:
    # 1. Different from last logged gesture, OR
    # 2. Same gesture but enough time has passed (treat as new occurrence)
    should_log = False
    if gesture != last_logged_gesture:
        should_log = True
    elif now - last_log_time >= MIN_LOG_INTERVAL:
        # Same gesture repeated after cooldown - could log, but skip to avoid spam
        pass

    if should_log:
        timestamp = time.strftime("%H:%M:%S")
        gesture_log.append((timestamp, gesture))
        # Keep only last 10 entries
        if len(gesture_log) > 10:
            gesture_log.pop(0)
        last_logged_gesture = gesture
        last_log_time = now
        gesture_flash_until = now + 0.5  # Flash for 0.5 seconds
        print(f"[{timestamp}] Gesture: {gesture}")

    last_gesture = gesture
    return should_log


def detect_gesture(landmarks):
    """Analyze pose landmarks and return detected gesture."""
    if not landmarks or len(landmarks) < 17:
        return "NO PERSON", (128, 128, 128)

    # Get key landmarks (normalized 0-1 coordinates)
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]

    # Check visibility (skip if key points aren't visible)
    min_visibility = 0.5
    if (left_shoulder.visibility < min_visibility or
        right_shoulder.visibility < min_visibility):
        return "PARTIAL VIEW", (128, 128, 128)

    # Arm raised detection (wrist above shoulder by threshold)
    left_arm_raised = left_wrist.y < (left_shoulder.y - ARM_RAISED_THRESHOLD)
    right_arm_raised = right_wrist.y < (right_shoulder.y - ARM_RAISED_THRESHOLD)

    # Arm out to side detection (wrist far from shoulder)
    left_arm_out = left_wrist.x < (left_shoulder.x - ARM_OUT_THRESHOLD)
    right_arm_out = right_wrist.x > (right_shoulder.x + ARM_OUT_THRESHOLD)

    # Gesture logic
    if left_arm_raised and right_arm_raised:
        return "STOP", (0, 0, 255)  # Red - both arms up

    if left_arm_raised and left_arm_out and not right_arm_raised:
        return "TURN LEFT", (0, 255, 255)  # Yellow

    if right_arm_raised and right_arm_out and not left_arm_raised:
        return "TURN RIGHT", (0, 255, 255)  # Yellow

    if left_arm_out and not left_arm_raised:
        return "POINT LEFT", (255, 255, 0)  # Cyan

    if right_arm_out and not right_arm_raised:
        return "POINT RIGHT", (255, 255, 0)  # Cyan

    # Default - arms down, neutral pose
    return "READY", (0, 255, 0)  # Green


def draw_landmarks(frame, landmarks):
    """Draw pose landmarks and connections on frame."""
    if not landmarks:
        return frame

    h, w = frame.shape[:2]

    # Define connections (pairs of landmark indices)
    connections = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 24),  # hips
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

    # Draw landmarks
    for i, lm in enumerate(landmarks):
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    return frame


def create_side_panel(gesture, gesture_color, fps, inference_ms, frame_height, is_flashing):
    """Create info panel showing gesture and stats."""
    panel_width = 280
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y_offset = 30

    # Title
    cv2.putText(panel, "Pose Detection", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 45

    # Current gesture - large and prominent with flash effect
    if is_flashing:
        # Draw flash background
        cv2.rectangle(panel, (5, y_offset - 25), (panel_width - 5, y_offset + 10),
                     gesture_color, -1)
        text_color = (0, 0, 0)  # Black text on colored background
    else:
        text_color = gesture_color

    cv2.putText(panel, gesture, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += 45

    # Stats
    cv2.putText(panel, f"FPS: {fps}  Inf: {inference_ms:.0f}ms", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 35

    # Gesture log
    cv2.putText(panel, "Gesture Log:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    y_offset += 22

    # Show recent gestures (newest first)
    for timestamp, logged_gesture in reversed(gesture_log[-8:]):
        cv2.putText(panel, f"{timestamp} {logged_gesture}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y_offset += 18

    # Controls at bottom
    y_offset = frame_height - 80
    cv2.putText(panel, "Gesture Guide:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 18
    cv2.putText(panel, "Arms up=STOP  Arm out=POINT", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    y_offset += 25
    cv2.putText(panel, "r-reconnect s-screenshot q-quit", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return panel


def main():
    # Parse video source arguments
    args = video_source.parse_args(description="Body pose detection")

    # Download model if needed
    download_model()

    print("Initializing pose detection...")

    # Create pose landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Open video source
    cap = video_source.get_capture(args)
    source_desc = video_source.get_source_description(args)
    print("Press 'q' to quit, 's' to save screenshot, 'r' to reconnect")
    print()

    if not cap.isOpened():
        print(f"ERROR: Could not open video source")
        if not video_source.is_local(args):
            print("Make sure the Pi is running stream_h264.py")
        return

    print("Connected! Try some gestures:")
    print("  - Raise both arms = STOP")
    print("  - Left arm up and out = TURN LEFT")
    print("  - Right arm up and out = TURN RIGHT")
    print()

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0
    inference_ms = 0
    frame_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            time.sleep(1)
            cap = video_source.reconnect(args, cap)
            continue

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run pose detection
        inference_start = time.time()
        frame_timestamp += 33  # ~30fps in milliseconds
        results = landmarker.detect_for_video(mp_image, frame_timestamp)
        inference_ms = (time.time() - inference_start) * 1000

        # Get landmarks and detect gesture
        landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
        gesture, gesture_color = detect_gesture(landmarks)

        # Log gesture if it's new
        log_gesture(gesture)

        # Check if we should flash
        is_flashing = time.time() < gesture_flash_until

        # Draw pose skeleton on frame
        frame = draw_landmarks(frame, landmarks)

        # Draw gesture text on frame (with flash effect)
        if is_flashing:
            # Draw background rectangle for flash
            cv2.rectangle(frame, (10, 15), (300, 60), gesture_color, -1)
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        else:
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, gesture_color, 3)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(gesture, gesture_color, fps, inference_ms, frame.shape[0], is_flashing)

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow(f"Pose Detection - {source_desc}", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"pose_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting...")
            cap = video_source.reconnect(args, cap)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
