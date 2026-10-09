#!/usr/bin/env python3
"""
Save what the garage LED clock (AWTRIX) is showing right now as a PNG.

    python tools/clock_screen.py out.png [--after SECONDS] [--host 192.168.4.37]

Reads /api/v1/display/screen (52x16 RGB ints) and draws each pixel as a
square so the image is readable. Handy for checking that messages fit.
"""

import argparse
import json
import time
import urllib.request

import cv2
import numpy as np


def grab(host):
    with urllib.request.urlopen(f"http://{host}/api/v1/display/screen", timeout=5) as r:
        d = json.load(r)
    px = np.array(d["pixels"], dtype=np.uint32).reshape(d["height"], d["width"])
    rgb = np.stack([(px >> 16) & 255, (px >> 8) & 255, px & 255], axis=-1).astype(np.uint8)
    return rgb


def render(rgb, scale=16, gap=2):
    h, w, _ = rgb.shape
    img = np.full((h * scale, w * scale, 3), 25, np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = (int(v) for v in rgb[y, x])
            cv2.rectangle(img, (x * scale + gap, y * scale + gap),
                          ((x + 1) * scale - gap, (y + 1) * scale - gap), (b, g, r), -1)
    return img


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--host", default="192.168.4.37")
    p.add_argument("--after", type=float, default=0, help="wait this long before grabbing")
    a = p.parse_args()
    time.sleep(a.after)
    cv2.imwrite(a.out, render(grab(a.host)))
