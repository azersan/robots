"""
Hand gesture detection logic - pure Python, no OpenCV/MediaPipe dependencies.
Used by local_hands.py and eval_hands.py for testable gesture recognition.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class Landmark:
    """Normalized hand landmark (mirrors MediaPipe structure)."""
    x: float  # 0-1, left to right
    y: float  # 0-1, top to bottom
    z: float = 0.0  # relative depth
    presence: float = 1.0
    visibility: float = 1.0


# Hand landmark indices (21 points per hand)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5   # Base of index finger
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# Recognized gestures (for capture mode menu)
GESTURE_NAMES = [
    'THUMBS UP',
    'FIST',
    'POINTING',
    'PEACE',
    'OPEN PALM',
    'ROCK ON',
    'CALL ME',
    'FOUR',
]


def is_finger_extended(landmarks, tip_idx: int, mcp_idx: int) -> bool:
    """Check if a finger is extended (tip above mcp in y)."""
    return landmarks[tip_idx].y < landmarks[mcp_idx].y


def is_thumb_extended(landmarks) -> bool:
    """Check if thumb is extended (away from palm)."""
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    # Thumb is extended if tip is far from index mcp horizontally
    return abs(thumb_tip.x - index_mcp.x) > 0.1


def detect_hand_gesture(landmarks, handedness: str) -> Tuple[str, Tuple[int, int, int]]:
    """
    Analyze hand landmarks and return detected gesture.

    Args:
        landmarks: List of 21 Landmark objects (or MediaPipe landmarks)
        handedness: "Left" or "Right"

    Returns:
        Tuple of (gesture_name, bgr_color)
    """
    if not landmarks or len(landmarks) < 21:
        return "NO HAND", (128, 128, 128)

    # Check which fingers are extended
    thumb_up = is_thumb_extended(landmarks)
    index_up = is_finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle_up = is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_up = is_finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky_up = is_finger_extended(landmarks, PINKY_TIP, PINKY_MCP)

    fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

    # Gesture recognition
    if fingers_up == 0 and not thumb_up:
        return "FIST", (0, 0, 255)  # Red

    if thumb_up and fingers_up == 0:
        return "THUMBS UP", (0, 255, 0)  # Green

    if index_up and fingers_up == 1 and not thumb_up:
        return "POINTING", (255, 255, 0)  # Cyan

    if index_up and middle_up and fingers_up == 2 and not thumb_up:
        return "PEACE", (255, 0, 255)  # Magenta

    if fingers_up == 4 and thumb_up:
        return "OPEN PALM", (0, 255, 255)  # Yellow - stop signal

    if fingers_up == 4 and not thumb_up:
        return "FOUR", (255, 165, 0)  # Orange

    if pinky_up and index_up and not middle_up and not ring_up:
        return "ROCK ON", (128, 0, 255)  # Purple

    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "CALL ME", (0, 200, 200)  # Teal

    # Count fingers
    count = fingers_up + (1 if thumb_up else 0)
    return f"{count} FINGERS", (200, 200, 200)


def landmarks_to_dict(landmarks) -> List[Dict[str, float]]:
    """
    Convert list of landmarks to serializable dict format.

    Works with both MediaPipe NormalizedLandmark objects and our Landmark dataclass.
    """
    result = []
    for lm in landmarks:
        entry = {
            'x': float(lm.x),
            'y': float(lm.y),
            'z': float(lm.z) if hasattr(lm, 'z') else 0.0,
        }
        # Include presence/visibility if available
        if hasattr(lm, 'presence'):
            entry['presence'] = float(lm.presence)
        if hasattr(lm, 'visibility'):
            entry['visibility'] = float(lm.visibility)
        result.append(entry)
    return result


def dict_to_landmarks(data: List[Dict[str, float]]) -> List[Landmark]:
    """Convert serialized dict format back to Landmark objects."""
    return [
        Landmark(
            x=d['x'],
            y=d['y'],
            z=d.get('z', 0.0),
            presence=d.get('presence', 1.0),
            visibility=d.get('visibility', 1.0)
        )
        for d in data
    ]
