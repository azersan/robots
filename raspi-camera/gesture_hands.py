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

# Additional joints for straightness detection
INDEX_PIP = 6
INDEX_DIP = 7
MIDDLE_PIP = 10
MIDDLE_DIP = 11
RING_PIP = 14
RING_DIP = 15
PINKY_PIP = 18
PINKY_DIP = 19

# Finger joint mappings: (MCP, PIP, DIP, TIP)
FINGER_JOINTS = {
    'index': (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    'middle': (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    'ring': (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    'pinky': (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Recognized gestures (for capture mode menu)
GESTURE_NAMES = [
    'THUMBS UP',
    'FIST',
    'POINTING',
    'PEACE',
    'OPEN PALM',
    'ROCK ON',
    'CALL ME',
    'NONE',  # For testing false positives - hand visible but no specific gesture
]


def _distance(lm1, lm2) -> float:
    """Calculate 2D distance between two landmarks."""
    return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5


def get_straightness_ratio(landmarks, mcp: int, pip: int, dip: int, tip: int) -> float:
    """Get the straightness ratio for a finger (0-1, higher = straighter).

    Compares direct MCP→TIP distance to sum of joint segments.
    Returns 1.0 for perfectly straight, lower values for bent fingers.
    """
    direct = _distance(landmarks[mcp], landmarks[tip])
    segments = (
        _distance(landmarks[mcp], landmarks[pip]) +
        _distance(landmarks[pip], landmarks[dip]) +
        _distance(landmarks[dip], landmarks[tip])
    )
    if segments < 0.01:
        return 0.0
    return direct / segments


def is_finger_straight(landmarks, mcp: int, pip: int, dip: int, tip: int) -> bool:
    """Check if a finger is straightened by comparing direct distance to segment sum.

    This is direction-agnostic: works for pointing up, down, forward, etc.
    A straight finger has ratio close to 1.0, a bent finger has lower ratio.
    """
    # Direct distance from MCP to TIP
    direct = _distance(landmarks[mcp], landmarks[tip])

    # Sum of segments: MCP->PIP + PIP->DIP + DIP->TIP
    segments = (
        _distance(landmarks[mcp], landmarks[pip]) +
        _distance(landmarks[pip], landmarks[dip]) +
        _distance(landmarks[dip], landmarks[tip])
    )

    if segments < 0.01:  # Avoid division by zero
        return False

    ratio = direct / segments
    return ratio > 0.9  # Threshold: 0.9 = straight finger


def get_finger_extension(landmarks, tip_idx: int) -> Tuple[bool, float]:
    """Check if a finger is extended and return confidence.

    Returns (is_extended, confidence) where confidence is 0-1.
    Higher confidence means the finger state is clearly extended or clearly curled.
    """
    STRAIGHT_THRESH = 0.9
    CURLED_THRESH = 0.75

    finger_map = {
        INDEX_TIP: 'index',
        MIDDLE_TIP: 'middle',
        RING_TIP: 'ring',
        PINKY_TIP: 'pinky',
    }

    finger = finger_map.get(tip_idx)
    if finger and finger in FINGER_JOINTS:
        mcp, pip, dip, tip = FINGER_JOINTS[finger]
        ratio = get_straightness_ratio(landmarks, mcp, pip, dip, tip)

        is_extended = ratio > STRAIGHT_THRESH

        # Confidence based on how far from the ambiguous zone
        if ratio > STRAIGHT_THRESH:
            # Clearly extended: confidence increases with ratio
            margin = (ratio - STRAIGHT_THRESH) / (1.0 - STRAIGHT_THRESH)
            confidence = min(1.0, 0.5 + margin * 0.5)
        elif ratio < CURLED_THRESH:
            # Clearly curled: confidence increases as ratio decreases
            margin = (CURLED_THRESH - ratio) / CURLED_THRESH
            confidence = min(1.0, 0.5 + margin * 0.5)
        else:
            # Ambiguous zone (0.75-0.9): low confidence
            # Lowest at midpoint (0.825)
            midpoint = (STRAIGHT_THRESH + CURLED_THRESH) / 2
            dist_from_mid = abs(ratio - midpoint) / (STRAIGHT_THRESH - midpoint)
            confidence = 0.2 + dist_from_mid * 0.3  # 0.2 to 0.5

        return is_extended, confidence

    # Fallback - shouldn't happen for standard fingers
    return False, 0.5


def is_finger_extended(landmarks, tip_idx: int, mcp_idx: int) -> bool:
    """Check if a finger is extended using straightness detection.

    Uses joint alignment to detect extension regardless of pointing direction.
    Falls back to y-position check for additional robustness.
    """
    # Map tip index to finger name
    finger_map = {
        INDEX_TIP: 'index',
        MIDDLE_TIP: 'middle',
        RING_TIP: 'ring',
        PINKY_TIP: 'pinky',
    }

    finger = finger_map.get(tip_idx)
    if finger and finger in FINGER_JOINTS:
        mcp, pip, dip, tip = FINGER_JOINTS[finger]
        return is_finger_straight(landmarks, mcp, pip, dip, tip)

    # Fallback to original y-position check
    diff = landmarks[mcp_idx].y - landmarks[tip_idx].y
    return diff > 0.03


def get_thumb_extension(landmarks) -> Tuple[bool, float]:
    """Check if thumb is extended and return confidence.

    Returns (is_extended, confidence) where confidence is 0-1.
    """
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]

    # Vertical distance: positive means thumb is above index MCP
    vert_dist = index_mcp.y - thumb_tip.y
    horiz_dist = abs(thumb_tip.x - index_mcp.x)

    # Horizontal extension threshold
    HORIZ_THRESH = 0.1
    VERT_THRESH = 0.1

    # Check horizontal: thumb out to the side, but not curled under palm
    horiz_extended = horiz_dist > HORIZ_THRESH and vert_dist >= -0.02
    # Check vertical: thumb up (tip clearly above index MCP)
    vert_extended = vert_dist > VERT_THRESH

    is_extended = horiz_extended or vert_extended

    # Compute confidence based on distance from threshold
    if is_extended:
        # How far above threshold?
        if vert_extended:
            margin = (vert_dist - VERT_THRESH) / VERT_THRESH  # Normalized margin
        else:
            margin = (horiz_dist - HORIZ_THRESH) / HORIZ_THRESH
        confidence = min(1.0, 0.5 + margin)  # 0.5 at threshold, 1.0 at 2x threshold
    else:
        # How far below threshold?
        max_dist = max(horiz_dist, vert_dist)
        if max_dist < 0.05:
            confidence = 1.0  # Clearly not extended
        else:
            margin = (HORIZ_THRESH - max_dist) / HORIZ_THRESH
            confidence = min(1.0, 0.5 + margin)

    return is_extended, confidence


def is_thumb_extended(landmarks) -> bool:
    """Check if thumb is extended (away from palm).

    Checks both horizontal (thumb out to side) and vertical (thumbs up/down).
    For horizontal extension, thumb must not be curled under the palm.
    """
    extended, _ = get_thumb_extension(landmarks)
    return extended


# Minimum confidence threshold for gesture detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

# Minimum finger spread for OPEN PALM (distinguishes from relaxed hand)
MIN_FINGER_SPREAD = 0.052


def get_finger_spread(landmarks) -> float:
    """Calculate average distance between adjacent fingertips.

    Higher values indicate fingers are spread apart (deliberate gesture).
    Lower values indicate fingers are close together (relaxed hand).
    """
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    total = 0
    for i in range(len(tips) - 1):
        total += _distance(landmarks[tips[i]], landmarks[tips[i + 1]])
    return total / 3  # Average of 3 gaps


def detect_hand_gesture(landmarks, handedness: str,
                        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
                        ) -> Tuple[str, Tuple[int, int, int]]:
    """
    Analyze hand landmarks and return detected gesture.

    Args:
        landmarks: List of 21 Landmark objects (or MediaPipe landmarks)
        handedness: "Left" or "Right"
        confidence_threshold: Minimum confidence to return a gesture (0-1).
                             Below this, returns "UNKNOWN".

    Returns:
        Tuple of (gesture_name, bgr_color)
    """
    if not landmarks or len(landmarks) < 21:
        return "NO HAND", (128, 128, 128)

    # Get finger states with confidence
    thumb_up, thumb_conf = get_thumb_extension(landmarks)
    index_up, index_conf = get_finger_extension(landmarks, INDEX_TIP)
    middle_up, middle_conf = get_finger_extension(landmarks, MIDDLE_TIP)
    ring_up, ring_conf = get_finger_extension(landmarks, RING_TIP)
    pinky_up, pinky_conf = get_finger_extension(landmarks, PINKY_TIP)

    fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

    # Helper to compute gesture confidence from relevant finger confidences
    def gesture_confidence(*confs):
        return sum(confs) / len(confs) if confs else 0

    # Gesture recognition with confidence
    gesture = None
    color = (200, 200, 200)
    confidence = 0.0

    if fingers_up == 0 and not thumb_up:
        gesture = "FIST"
        color = (0, 0, 255)  # Red
        confidence = gesture_confidence(thumb_conf, index_conf, middle_conf, ring_conf, pinky_conf)

    elif thumb_up and fingers_up == 0:
        gesture = "THUMBS UP"
        color = (0, 255, 0)  # Green
        confidence = gesture_confidence(thumb_conf, index_conf, middle_conf, ring_conf, pinky_conf)

    elif index_up and fingers_up == 1 and not thumb_up:
        gesture = "POINTING"
        color = (255, 255, 0)  # Cyan
        confidence = gesture_confidence(thumb_conf, index_conf, middle_conf, ring_conf, pinky_conf)

    elif index_up and middle_up and fingers_up == 2 and not thumb_up:
        gesture = "PEACE"
        color = (255, 0, 255)  # Magenta
        confidence = gesture_confidence(thumb_conf, index_conf, middle_conf, ring_conf, pinky_conf)

    elif fingers_up == 4:
        # Check if fingers are spread (not just a relaxed open hand)
        spread = get_finger_spread(landmarks)
        if spread >= MIN_FINGER_SPREAD:
            gesture = "OPEN PALM"
            color = (0, 255, 255)  # Yellow
            confidence = gesture_confidence(index_conf, middle_conf, ring_conf, pinky_conf)

    elif pinky_up and index_up and not middle_up and not ring_up:
        gesture = "ROCK ON"
        color = (128, 0, 255)  # Purple
        confidence = gesture_confidence(index_conf, middle_conf, ring_conf, pinky_conf)

    elif thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        gesture = "CALL ME"
        color = (0, 200, 200)  # Teal
        confidence = gesture_confidence(thumb_conf, index_conf, middle_conf, ring_conf, pinky_conf)

    # Check confidence threshold
    if gesture and confidence >= confidence_threshold:
        return gesture, color

    # Low confidence or no specific gesture matched
    if confidence_threshold > 0 and (gesture is None or confidence < confidence_threshold):
        return "UNKNOWN", (128, 128, 128)

    # Fallback: count fingers (no confidence threshold for this)
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
            'z': float(lm.z) if hasattr(lm, 'z') and lm.z is not None else 0.0,
        }
        # Include presence/visibility if available and not None
        if hasattr(lm, 'presence') and lm.presence is not None:
            entry['presence'] = float(lm.presence)
        if hasattr(lm, 'visibility') and lm.visibility is not None:
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
