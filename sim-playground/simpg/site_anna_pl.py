"""Site layout for the Anna Pl driveway run, traced from a satellite screenshot.

All coordinates were picked by hand off a 1072x1236 px screenshot (north up)
with the route drawn on it. They are converted to sim meters with:

    x_m = (px - ORIGIN_PX[0]) * M_PER_PX      (east)
    y_m = (ORIGIN_PX[1] - py) * M_PER_PX      (north)

so the origin is the start point (S) by the garage. M_PER_PX is an estimate
from house/road widths; measure the real route length (Google Maps "Measure
distance") and adjust it so route_length() matches.
"""

import math

M_PER_PX = 0.075
ORIGIN_PX = (395.0, 140.0)  # S: in front of the garage at #11

# Red route line, start (S) to bins (E).
ROUTE_PX = [
    (395, 140),
    (440, 165),
    (490, 215),
    (540, 270),
    (600, 335),
    (680, 410),
    (750, 455),
    (790, 510),
    (815, 590),
    (830, 700),
    (845, 820),
    (860, 940),
    (870, 1050),
    (865, 1100),
    (800, 1118),
]

# Where the bins sit at the bottom of Anna Pl, and which way they face (deg, 0 = east).
BINS_PX = [(778, 1128), (752, 1134)]
BIN_YAW_DEG = 90.0

# Paved surfaces as polylines with a width (m). Visual only; physics is a flat plane for now.
PAVEMENT = [
    # House #11 driveway, garage apron down to the lane.
    {"width": 5.0, "px": [(370, 130), (440, 165), (540, 270), (680, 410), (760, 470)]},
    # Anna Pl lane, from the bend at the top right down to the street.
    {"width": 5.5, "px": [(560, 0), (700, 120), (750, 300), (775, 470), (810, 600), (835, 760),
                          (855, 940), (865, 1100)]},
    # The street at the bottom (Anna Pl continues west).
    {"width": 7.0, "px": [(1000, 1120), (800, 1130), (640, 1150), (480, 1236)]},
]

# Houses as (center_px, size_px (w, h), rotation deg, height m). Rough footprints.
HOUSES = [
    {"center": (300, 300), "size": (230, 260), "rot": 28.0, "height": 8.0},   # #11
    {"center": (250, 720), "size": (250, 300), "rot": 22.0, "height": 8.0},   # #10
]

# Tree clusters: (center_px, radius_px). Wooded strip between the driveway and lane,
# plus the woods on the east side of the lane.
TREE_CLUSTERS = [
    ((560, 420), 70), ((640, 560), 90), ((610, 720), 80), ((640, 960), 110),
    ((940, 400), 120), ((960, 700), 130), ((960, 1000), 110), ((470, 60), 50),
    ((80, 330), 70), ((200, 1060), 60),
]


def px_to_m(p):
    return ((p[0] - ORIGIN_PX[0]) * M_PER_PX, (ORIGIN_PX[1] - p[1]) * M_PER_PX)


def route_m():
    return [px_to_m(p) for p in ROUTE_PX]


def bins_m():
    return [px_to_m(p) for p in BINS_PX]


def route_length():
    pts = route_m()
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def start_pose():
    """(x, y, yaw) at S, facing along the first route segment."""
    (x0, y0), (x1, y1) = route_m()[:2]
    return x0, y0, math.atan2(y1 - y0, x1 - x0)


if __name__ == "__main__":
    print(f"route length ~ {route_length():.1f} m over {len(ROUTE_PX) - 1} segments")
    print("bins at", [tuple(round(v, 1) for v in b) for b in bins_m()])
