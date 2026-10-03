"""
Shared red-object detection used by the color tracking apps.

Red wraps around the HSV hue circle (0 and 180), so detection needs two
ranges OR'd together. Tune the ranges here and every consumer picks them up.
"""

import cv2
import numpy as np

# Red color range in HSV (red wraps around 0, so we need two ranges)
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])


def red_mask(frame_bgr):
    """Return a cleaned-up binary mask of red pixels in a BGR frame."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def largest_red_blob(frame_bgr, min_area):
    """Find the largest red blob in a BGR frame.

    Returns (mask, blob) where blob is None if nothing big enough was found,
    otherwise a dict with cx, cy (centroid), x, y, radius (enclosing circle),
    and area.
    """
    mask = red_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= min_area:
        return mask, None

    ((x, y), radius) = cv2.minEnclosingCircle(largest)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return mask, None

    return mask, {
        "cx": int(M["m10"] / M["m00"]),
        "cy": int(M["m01"] / M["m00"]),
        "x": x,
        "y": y,
        "radius": radius,
        "area": area,
    }
