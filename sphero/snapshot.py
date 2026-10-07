"""Grab one frame from the webcam and save it, so Claude can look at it."""
import argparse
import time

import cv2

p = argparse.ArgumentParser()
p.add_argument("-c", "--camera", type=int, default=0, help="camera index")
p.add_argument("-o", "--out", default="snapshot.jpg")
args = p.parse_args()

cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Let auto-exposure settle before keeping a frame.
t0 = time.time()
ok, frame = False, None
while time.time() - t0 < 1.5:
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"camera {args.camera}: no frame")
cv2.imwrite(args.out, frame)
print(f"saved {args.out} {frame.shape[1]}x{frame.shape[0]}")
