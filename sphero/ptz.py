"""Read / set the webcam's pan, tilt and zoom (Insta360 Link 2 is a UVC PTZ camera).

  python ptz.py                 # print current values
  python ptz.py --tilt -10      # nudge tilt (relative, in camera units)
  python ptz.py --tilt-abs 0    # set absolute
"""
import argparse
import time

import cv2

PROPS = {"pan": cv2.CAP_PROP_PAN, "tilt": cv2.CAP_PROP_TILT, "zoom": cv2.CAP_PROP_ZOOM}


def read(cap):
    return {k: cap.get(v) for k, v in PROPS.items()}


def nudge(cap, name, delta=None, absolute=None):
    prop = PROPS[name]
    cur = cap.get(prop)
    target = absolute if absolute is not None else cur + delta
    ok = cap.set(prop, target)
    return cur, target, ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--camera", type=int, default=0)
    for k in PROPS:
        p.add_argument(f"--{k}", type=float, help="relative change")
        p.add_argument(f"--{k}-abs", type=float, help="absolute value")
    p.add_argument("-o", "--out", help="save a frame after moving")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.read()
    print("before:", read(cap))
    for k in PROPS:
        rel, ab = getattr(args, k), getattr(args, f"{k}_abs")
        if rel is not None or ab is not None:
            print(k, nudge(cap, k, rel, ab))
    t0 = time.time()
    while time.time() - t0 < 2.0:  # let the gimbal move / exposure settle
        ok, frame = cap.read()
    print("after: ", read(cap))
    if args.out and ok:
        cv2.imwrite(args.out, frame)
        print("saved", args.out)
    cap.release()
