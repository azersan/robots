#!/usr/bin/env python3
"""
Hand gesture detection using MediaPipe.
Supports local webcam or remote Pi stream.

Usage:
    python3 local_hands.py              # Default: Pi stream
    python3 local_hands.py --local      # Use Mac webcam
    python3 local_hands.py --source IP  # Pi at specific IP

Capture mode (for building test cases):
    python3 local_hands.py --local --capture
    # Press 1-7 to select expected gesture, 'c' to capture
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import json
import video_source
from gesture_hands import (
    detect_hand_gesture, landmarks_to_dict, GESTURE_NAMES
)

# Model path
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Gesture log
gesture_log = []
last_logged_gesture = None
last_log_time = 0
gesture_flash_until = 0
MIN_LOG_INTERVAL = 2.0

# Capture mode state
capture_selected_gesture = None
capture_count = 0


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")


def log_gesture(gesture):
    """Log a gesture if it's different from the last one."""
    global last_logged_gesture, last_log_time, gesture_flash_until, gesture_log

    # Skip non-actionable states
    if gesture in ["NO HAND"]:
        return False

    now = time.time()

    # Only log if different from last logged gesture
    if gesture != last_logged_gesture:
        timestamp = time.strftime("%H:%M:%S")
        gesture_log.append((timestamp, gesture))
        if len(gesture_log) > 10:
            gesture_log.pop(0)
        last_logged_gesture = gesture
        last_log_time = now
        gesture_flash_until = now + 0.5
        print(f"[{timestamp}] Hand Gesture: {gesture}")
        return True

    return False


def save_test_case(landmarks, handedness, frame, expected_gesture):
    """Save a test case for evaluation."""
    global capture_count

    # Create test data directory if needed
    test_dir = os.path.join(os.path.dirname(__file__), "test_data", "hands")
    os.makedirs(test_dir, exist_ok=True)

    # Generate case ID
    gesture_slug = expected_gesture.lower().replace(" ", "_")
    case_id = f"{gesture_slug}_{capture_count:03d}"
    case_dir = os.path.join(test_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # Save screenshot
    screenshot_path = os.path.join(case_dir, "screenshot.jpg")
    cv2.imwrite(screenshot_path, frame)

    # Save case data
    case_data = {
        "id": case_id,
        "expected_gesture": expected_gesture,
        "handedness": handedness,
        "landmarks": landmarks_to_dict(landmarks),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    case_path = os.path.join(case_dir, "case.json")
    with open(case_path, 'w') as f:
        json.dump(case_data, f, indent=2)

    capture_count += 1
    print(f"Saved test case: {case_id}")
    return case_id


def draw_hand_landmarks(frame, landmarks, handedness):
    """Draw hand landmarks and connections on frame."""
    if not landmarks:
        return frame

    h, w = frame.shape[:2]

    # Connections for hand
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]

    # Color based on handedness
    color = (0, 255, 0) if handedness == "Right" else (255, 0, 0)

    # Draw connections
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(frame, pt1, pt2, color, 2)

    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    return frame


def draw_pointing_line(frame, landmarks):
    """Draw a line showing pointing direction when POINTING gesture detected."""
    if not landmarks or len(landmarks) < 21:
        return frame

    h, w = frame.shape[:2]

    # Index finger landmarks: MCP(5) -> PIP(6) -> DIP(7) -> TIP(8)
    # Use PIP to TIP for direction (more stable than MCP to TIP)
    pip = landmarks[6]  # Index PIP (middle joint)
    tip = landmarks[8]  # Index TIP

    # Calculate direction vector
    dx = tip.x - pip.x
    dy = tip.y - pip.y

    # Normalize and extend the line
    length = (dx**2 + dy**2)**0.5
    if length < 0.01:  # Avoid division by zero
        return frame

    # Extend line by 2x frame width for visibility
    extend = 2.0 / length
    end_x = tip.x + dx * extend
    end_y = tip.y + dy * extend

    # Convert to pixel coordinates
    pt1 = (int(tip.x * w), int(tip.y * h))
    pt2 = (int(end_x * w), int(end_y * h))

    # Draw the pointing line (cyan, thin)
    cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw a small circle at the fingertip
    cv2.circle(frame, pt1, 6, (255, 255, 0), -1)

    return frame


def create_side_panel(gesture, gesture_color, fps, inference_ms, frame_height, is_flashing, num_hands, capture_mode=False):
    """Create info panel showing gesture and stats."""
    panel_width = 280
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y_offset = 30

    # Title
    title = "CAPTURE MODE" if capture_mode else "Hand Detection"
    title_color = (0, 200, 255) if capture_mode else (255, 255, 255)
    cv2.putText(panel, title, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
    y_offset += 30

    cv2.putText(panel, f"Hands: {num_hands}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 35

    # Current gesture with flash effect
    if is_flashing:
        cv2.rectangle(panel, (5, y_offset - 25), (panel_width - 5, y_offset + 10),
                     gesture_color, -1)
        text_color = (0, 0, 0)
    else:
        text_color = gesture_color

    cv2.putText(panel, gesture, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += 45

    # Stats
    cv2.putText(panel, f"FPS: {fps}  Inf: {inference_ms:.0f}ms", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 35

    if capture_mode:
        # Capture mode: show gesture selection menu
        cv2.putText(panel, "Select gesture (1-8):", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y_offset += 22

        for i, name in enumerate(GESTURE_NAMES):
            key = i + 1
            is_selected = (capture_selected_gesture == name)
            if is_selected:
                cv2.rectangle(panel, (5, y_offset - 14), (panel_width - 5, y_offset + 4),
                             (0, 100, 0), -1)
            color = (0, 255, 0) if is_selected else (180, 180, 180)
            cv2.putText(panel, f"{key}. {name}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18

        y_offset += 10
        if capture_selected_gesture:
            cv2.putText(panel, f"Ready: {capture_selected_gesture}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            cv2.putText(panel, "Press 'c' to capture", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            cv2.putText(panel, "Select a gesture first", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        y_offset += 30
        cv2.putText(panel, f"Captured: {capture_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    else:
        # Normal mode: show gesture log
        cv2.putText(panel, "Gesture Log:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        y_offset += 22

        for timestamp, logged_gesture in reversed(gesture_log[-6:]):
            cv2.putText(panel, f"{timestamp} {logged_gesture}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_offset += 18

    # Gesture guide at bottom
    y_offset = frame_height - 120
    cv2.putText(panel, "Gestures:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 20

    gestures = [
        "Fist = FIST",
        "Thumb up = THUMBS UP",
        "Point = POINTING",
        "V sign = PEACE",
        "Open hand = OPEN PALM",
    ]
    for text in gestures:
        cv2.putText(panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        y_offset += 16

    y_offset += 5
    if capture_mode:
        cv2.putText(panel, "c-capture r-reconnect q-quit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    else:
        cv2.putText(panel, "r-reconnect s-screenshot q-quit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return panel


def main():
    global capture_selected_gesture

    # Parse video source arguments with capture mode
    parser = video_source.create_parser(description="Hand gesture detection")
    parser.add_argument('--capture', action='store_true',
                       help='Enable test case capture mode')
    args = video_source.parse_args(parser=parser)

    if args.capture:
        print("CAPTURE MODE: Press 1-8 to select gesture, 'c' to capture")

    # Download model if needed
    download_model()

    print("Initializing hand detection...")

    # Create hand landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

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

    print("Connected! Try some hand gestures:")
    print("  - Fist, Thumbs up, Pointing, Peace sign, Open palm")
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

        # Run hand detection
        inference_start = time.time()
        frame_timestamp += 33
        results = landmarker.detect_for_video(mp_image, frame_timestamp)
        inference_ms = (time.time() - inference_start) * 1000

        # Process detected hands
        gesture = "NO HAND"
        gesture_color = (128, 128, 128)
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        current_landmarks = None
        current_handedness = "Unknown"

        if results.hand_landmarks:
            # Use the first detected hand
            current_landmarks = results.hand_landmarks[0]
            current_handedness = results.handedness[0][0].category_name if results.handedness else "Unknown"

            gesture, gesture_color = detect_hand_gesture(current_landmarks, current_handedness)

            # Draw all detected hands
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                hand_name = results.handedness[i][0].category_name if results.handedness else "Unknown"
                frame = draw_hand_landmarks(frame, hand_landmarks, hand_name)

            # Draw pointing line if POINTING gesture detected
            if gesture == "POINTING":
                frame = draw_pointing_line(frame, current_landmarks)

        # Log gesture
        log_gesture(gesture)

        # Check flash
        is_flashing = time.time() < gesture_flash_until

        # Draw gesture text on frame
        if is_flashing:
            cv2.rectangle(frame, (10, 15), (250, 60), gesture_color, -1)
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        else:
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, gesture_color, 2)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(gesture, gesture_color, fps, inference_ms,
                                  frame.shape[0], is_flashing, num_hands,
                                  capture_mode=args.capture)

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow(f"Hand Detection - {source_desc}", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"hand_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting...")
            cap = video_source.reconnect(args, cap)
        elif args.capture and ord('1') <= key <= ord('8'):
            # Select gesture for capture
            idx = key - ord('1')
            if idx < len(GESTURE_NAMES):
                capture_selected_gesture = GESTURE_NAMES[idx]
                print(f"Selected: {capture_selected_gesture}")
        elif args.capture and key == ord('c'):
            # Capture current frame
            if capture_selected_gesture and current_landmarks:
                save_test_case(current_landmarks, current_handedness, frame, capture_selected_gesture)
            elif not capture_selected_gesture:
                print("Select a gesture first (1-8)")
            else:
                print("No hand detected - show your hand and try again")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
