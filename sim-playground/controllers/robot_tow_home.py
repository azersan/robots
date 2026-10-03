"""Camera-only tow-home controller for the anna_pl scene (robot API on port 8643 only).

Pipeline
  1. FOLLOW   : drive down the driveway, turn right onto the lane, follow it to the street.
                Pavement vs grass from color (green chroma); steering picks the bearing whose
                ground corridor stays clear of grass longest, for several corridor widths
                (wider corridors only exist near the middle of the pavement -> centering).
                Pose is dead-reckoned from commanded speed + compass and the path is logged.
  2. STAGE    : find the blue recycling cart at the end of the lane, estimate where its back
                face is, drive into the strip of street between the lawn and the carts and
                stop ~1.3 m behind the cart, facing it.
  3. LATCH    : visual servo on the yellow latch bar, creep in, try the latch every cm.
  4. TOW HOME : with the cart blocking the camera, retrace the logged outbound path in
                reverse (pure pursuit on the dead-reckoned pose) back to the start.

usage: python robot_tow_home.py [--seed N] [--no-reset] [--debug] [--realtime-after]
"""
import argparse
import io
import json
import math
import os
import sys
import time

import httpx
import numpy as np
from PIL import Image

BASE = "http://localhost:8643/robot"
W, H, F, CAM_H, PITCH = 320, 240, 171.4, 0.24, math.radians(6.8)
CAM_AHEAD = 0.45     # camera ahead of the wheel axle (rotation centre), measured
HOOK_AHEAD = 0.03    # hook ahead of camera
DT_FRAMES = 3        # 0.05 s per control tick
DT = DT_FRAMES / 60.0
DBG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "robot", "run")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def hvec(h_deg):
    """unit vector (east, north) for a compass heading"""
    r = math.radians(h_deg)
    return np.array([math.sin(r), math.cos(r)])


def heading_of(vec):
    return math.degrees(math.atan2(vec[0], vec[1])) % 360.0


# ----------------------------------------------------------------------------- camera geometry
def ground_point(u, v, h=CAM_H):
    x = (np.asarray(u, float) + 0.5 - W / 2) / F
    y = -(np.asarray(v, float) + 0.5 - H / 2) / F
    fc = y * math.sin(PITCH) + math.cos(PITCH)
    uc = y * math.cos(PITCH) - math.sin(PITCH)
    t = np.where(uc < -1e-4, h / np.maximum(-uc, 1e-4), np.nan)
    return t * fc, -t * x


_uu, _vv = np.meshgrid(np.arange(W), np.arange(H))
GF, GL = ground_point(_uu, _vv)
GF = np.nan_to_num(GF, nan=0.0); GL = np.nan_to_num(GL, nan=0.0)
GR = np.hypot(GF, GL)
GB = np.degrees(np.arctan2(GL, GF))
V0 = 104
NB, BSTEP, RSTEP, NR = 61, 2.0, 0.25, 36
BEARINGS = np.arange(NB) * BSTEP - 60.0
_rows = slice(V0, H)
_bidx = np.clip(np.round((GB[_rows] + 60) / BSTEP).astype(int), 0, NB - 1)
_ridx = np.clip((GR[_rows] / RSTEP).astype(int), 0, NR - 1)
_ok = (GR[_rows] > 0) & (GR[_rows] < NR * RSTEP) & (np.abs(GB[_rows]) <= 61)
_flat = (_bidx * NR + _ridx)[_ok]
_cnt = np.bincount(_flat, minlength=NB * NR).reshape(NB, NR)
_RR = (np.arange(NR) + 0.5) * RSTEP


def grass_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return ((g - b) / (r + g + b + 30.0) > 0.09) & (g > r - 6)


def polar_grass(gm):
    gs = np.bincount(_flat, weights=gm[_rows][_ok].astype(float), minlength=NB * NR).reshape(NB, NR)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(_cnt > 0, gs / np.maximum(_cnt, 1), 0.0)


def free_ranges(pf, half_w, thr=0.35, rmax=8.0):
    bad = (pf > thr).astype(np.int32)
    cs = np.vstack([np.zeros((1, NR), np.int32), np.cumsum(bad, axis=0)])
    idx = np.arange(NB)
    hit = np.zeros((NB, NR), bool)
    for j in range(NR):
        d = int(math.degrees(math.atan2(half_w, _RR[j])) / BSTEP + 0.5)
        lo = np.clip(idx - d, 0, NB); hi = np.clip(idx + d + 1, 0, NB)
        hit[:, j] = (cs[hi, j] - cs[lo, j]) > 0
    first = np.where(hit.any(1), hit.argmax(1), NR)
    return np.minimum(first * RSTEP, rmax)


# ----------------------------------------------------------------------------- object detection
def blue_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (b > r + 25) & (b > g + 15) & (r < 70) & (g < 90)


def yellow_mask(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > b + 40) & (g > b + 28) & (r > 50) & (r >= g - 5)


def blue_blob(im):
    """largest blue column-run; returns dict with u range, bottom row, ground point of bottom centre"""
    bm = blue_mask(im)
    bm[:40] = bm[:40]  # keep
    colcnt = bm.sum(0)
    cols = colcnt > 2
    if cols.sum() < 2:
        return None
    # split into runs of columns
    runs, start = [], None
    for u in range(W + 1):
        c = cols[u] if u < W else False
        if c and start is None:
            start = u
        if not c and start is not None:
            if u - start >= 2:
                runs.append((start, u - 1))
            start = None
    if not runs:
        return None
    best = max(runs, key=lambda r: colcnt[r[0]:r[1] + 1].sum())
    u0, u1 = best
    sub = bm[:, u0:u1 + 1]
    bottoms = np.array([np.nonzero(sub[:, i])[0].max() if sub[:, i].any() else -1 for i in range(sub.shape[1])])
    vb = int(np.percentile(bottoms[bottoms >= 0], 80))
    uc = (u0 + u1) / 2.0
    f, l = ground_point(uc, vb + 1)
    return dict(u0=u0, u1=u1, uc=uc, vb=vb, fwd=float(f), left=float(l), npx=int(sub.sum()))


def find_bar(im, u_hint=None):
    """yellow latch bar: returns centre pixel, width, slope, ground position (h=0.08 m)."""
    ym = yellow_mask(im)
    ym[:95] = False
    vs, us = np.nonzero(ym)
    if len(us) < 5:
        return None
    # column clusters
    colc = np.bincount(us, minlength=W) > 0
    runs, start = [], None
    for u in range(W + 1):
        c = colc[u] if u < W else False
        if c and start is None:
            start = u
        if not c and start is not None:
            runs.append((start, u - 1)); start = None
    # merge runs separated by small gaps
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 6:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    if u_hint is None:
        u_hint = W / 2
    cand = [r for r in merged if r[1] - r[0] >= 2]
    if not cand:
        return None
    r0 = min(cand, key=lambda r: 0 if r[0] <= u_hint <= r[1] else min(abs(r[0] - u_hint), abs(r[1] - u_hint)))
    sel = (us >= r0[0]) & (us <= r0[1])
    vs, us = vs[sel], us[sel]
    rows = np.bincount(vs, minlength=H)
    vb = int(np.argmax(rows))
    band = np.abs(vs - vb) <= max(2, rows[vb] // 40)
    ub = us[band]
    u0, u1 = float(ub.min()), float(ub.max())
    uc = (u0 + u1) / 2
    # slope: mean row of the bar's top edge in left vs right thirds
    span = u1 - u0
    slope = 0.0
    if span > 20:
        lt = vs[(us < u0 + span / 4) & band]; rt = vs[(us > u1 - span / 4) & band]
        if len(lt) and len(rt):
            slope = (np.mean(rt) - np.mean(lt)) / (span * 0.75)
    f, l = ground_point(uc, vb, CAM_H - 0.08)
    fl, ll = ground_point(u0, vb, CAM_H - 0.08)
    fr_, lr = ground_point(u1, vb, CAM_H - 0.08)
    return dict(u=uc, v=vb, u0=u0, u1=u1, w=span, slope=slope, fwd=float(f), left=float(l),
                width_m=float(abs(ll - lr)), npx=int(len(us)))


# ----------------------------------------------------------------------------- robot client
class Bot:
    def __init__(self, debug=False):
        self.c = httpx.Client(timeout=30)
        self.debug = debug
        self.x = 0.0; self.y = 0.0
        self.s = None
        self.path = []          # outbound path (x, y, heading)
        self.logging = False
        self.hf = None
        self.k = 0
        self.wall0 = time.time()
        if debug:
            os.makedirs(DBG_DIR, exist_ok=True)

    def _req(self, method, path, **kw):
        for attempt in range(60):
            try:
                r = self.c.request(method, BASE + path, **kw)
                r.raise_for_status()
                return r
            except Exception as e:  # gateway busy: wait and retry
                if attempt % 10 == 0:
                    print("request failed (%s), retrying" % e, file=sys.stderr)
                time.sleep(2)
        raise RuntimeError("robot API not responding")

    def sensors(self):
        self.s = self._req("GET", "/sensors").json()
        return self.s

    def reset(self, seed):
        self._req("POST", "/run", json={"running": False})
        self.s = self._req("POST", "/reset", json={"scene": "anna_pl", "randomize": True, "seed": seed}).json()
        self.x = self.y = 0.0
        return self.s

    def lockstep(self):
        self._req("POST", "/run", json={"running": False})

    def img(self, cam="front"):
        r = self._req("GET", "/camera/%s.png" % cam)
        self.last_png = r.content
        im = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.float32)
        if cam == "front" and self.debug and self.logging and self.s is not None:
            self._accum(im)
        return im

    # ---- debug map (dead-reckoned bird's-eye view of what the camera classified) ----
    MRES, MX0, MY0, MW, MH = 0.1, -20.0, 20.0, 800, 1200

    def _accum(self, im):
        if not hasattr(self, "_mg"):
            self._mg = np.zeros((self.MH, self.MW)); self._mc = np.zeros((self.MH, self.MW))
            self._msel = (np.arange(H)[:, None] >= V0) & (GR > 0) & (GR < 6.0)
        m = self._msel
        h = math.radians(self.hd)
        cam = self.pos() + CAM_AHEAD * hvec(self.hd)
        fw, lf = GF[m], GL[m]
        ex = cam[0] + fw * math.sin(h) - lf * math.cos(h)
        ny = cam[1] + fw * math.cos(h) + lf * math.sin(h)
        ix = ((ex - self.MX0) / self.MRES).astype(int); iy = ((self.MY0 - ny) / self.MRES).astype(int)
        ok = (ix >= 0) & (ix < self.MW) & (iy >= 0) & (iy < self.MH)
        wt = 1.0 / (0.3 + GR[m][ok]) ** 2
        np.add.at(self._mg, (iy[ok], ix[ok]), grass_mask(im)[m][ok] * wt)
        np.add.at(self._mc, (iy[ok], ix[ok]), wt)

    def save_map(self, name, trace=()):
        if not hasattr(self, "_mg"):
            return
        g = np.where(self._mc > 0, self._mg / np.maximum(self._mc, 1e-9), -1)
        img = np.full((self.MH, self.MW, 3), 255, np.uint8)
        img[g >= 0.5] = (60, 150, 60)
        img[(g >= 0) & (g < 0.5)] = (150, 150, 150)
        def put(pts, col):
            for (x, y) in pts:
                ix = int((x - self.MX0) / self.MRES); iy = int((self.MY0 - y) / self.MRES)
                if 0 <= ix < self.MW and 0 <= iy < self.MH:
                    img[max(0, iy - 1):iy + 1, max(0, ix - 1):ix + 1] = col
        put([p[:2] for p in self.path], (220, 0, 0))
        put(trace, (0, 0, 230))
        Image.fromarray(img).save(os.path.join(DBG_DIR, name + ".png"))

    def save(self, name, cam="front"):
        if self.debug:
            with open(os.path.join(DBG_DIR, name + ".png"), "wb") as f:
                f.write(self._req("GET", "/camera/%s.png" % cam).content)

    @property
    def hd(self):
        return self.s["compass_deg"]

    def drive(self, v, w, frames=DT_FRAMES):
        self._req("POST", "/drive", json={"v": float(v), "w": float(w)})
        h0 = self.s["compass_deg"] if self.s else None
        self.s = self._req("POST", "/step", json={"frames": int(frames)}).json()
        h1 = self.s["compass_deg"]
        hm = h1 if h0 is None else (h0 + wrap180(h1 - h0) / 2.0)
        d = v * frames / 60.0
        self.x += d * math.sin(math.radians(hm))
        self.y += d * math.cos(math.radians(hm))
        # filtered heading (compass noise is ~2 deg per reading)
        if self.hf is None:
            self.hf = h1
        else:
            pred = self.hf - math.degrees(w * frames / 60.0)
            self.hf = (pred + 0.3 * wrap180(h1 - pred)) % 360
        if self.logging and (not self.path or math.hypot(self.x - self.path[-1][0], self.y - self.path[-1][1]) >= 0.2):
            self.path.append((self.x, self.y, h1))
        self.k += 1
        return self.s

    def stop(self, frames=3):
        return self.drive(0, 0, frames)

    def latch(self, engage=True):
        return self._req("POST", "/latch", json={"engage": engage}).json()["latched"]

    def pos(self):
        return np.array([self.x, self.y])

    def avg_heading(self, n=6):
        hs = []
        for _ in range(n):
            hs.append(self.stop(1)["compass_deg"])
        h0 = hs[0]
        return (h0 + np.mean([wrap180(h - h0) for h in hs])) % 360

    # ---- primitive manoeuvres (dead reckoning) ----
    def turn_to(self, h_target, wmax=0.8, tol=1.5, timeout=20.0):
        t0 = self.s["time"]
        self.hf = None
        self.stop(1)
        while self.s["time"] - t0 < timeout:
            e = wrap180(h_target - self.hf)
            if abs(e) < tol:
                break
            w = float(np.clip(-math.radians(e) * 2.5, -wmax, wmax))
            if abs(w) < 0.12:
                w = math.copysign(0.12, w)
            self.drive(0, w)
        self.stop()

    def drive_to(self, target, v=0.6, tol=0.05, reverse=False, timeout=60.0):
        """straight-line drive (forward or reverse) to a world point with heading hold"""
        t0 = self.s["time"]
        target = np.asarray(target, float)
        while self.s["time"] - t0 < timeout:
            d = target - self.pos()
            dist = float(np.hypot(*d))
            want = heading_of(d)
            if reverse:
                want = (want + 180) % 360
            e = wrap180(want - self.hd)
            fwd_comp = float(np.dot(d, hvec(self.hd))) * (-1 if reverse else 1)
            if dist < tol or fwd_comp < 0.0:
                break
            if abs(e) > 25 and dist > 0.3:
                self.turn_to(want, tol=3.0)
                continue
            sp = min(v, max(0.08, dist * 1.2))
            w = float(np.clip(-math.radians(e) * 2.0, -0.8, 0.8)) if dist > 0.15 else 0.0
            self.drive(-sp if reverse else sp, w)
        self.stop()


# ----------------------------------------------------------------------------- phase 1
_NEAR_ROWS = slice(int(np.argmin(np.abs(GF[:, W // 2] - 0.8))), int(np.argmin(np.abs(GF[:, W // 2] - 0.3))) + 1)


def follow_route(bot, log):
    """driveway -> right onto lane -> lane to the street. Returns when the bins are close."""
    s_len = 0.0
    last = bot.pos().copy()
    bot.path = [(bot.x, bot.y, bot.hd)]
    bot.logging = True
    lane_heads = []
    widths = [2.2, 1.6, 1.1, 0.7, 0.4]
    wts = [0.7, 1.0, 1.0, 1.0, 1.5]
    min_clear = 9.0
    bin_hits = []
    sel = np.abs(BEARINGS) <= 40
    t_start = bot.s["time"]
    while True:
        im = bot.img()
        gm = grass_mask(im)
        pf = polar_grass(gm)
        score = np.zeros(NB)
        frs = []
        for hw, wt in zip(widths, wts):
            fr = free_ranges(pf, hw)
            frs.append(fr)
            score += wt * np.minimum(fr, 6.0) / 6.0
        # route preference: driveway heads ~135, after ~30 m the lane heads ~172
        des = 135.0 if s_len < 30.0 else 174.0
        pref = float(np.clip(wrap180(des - bot.hd), -40, 40))  # + = right in compass terms
        pref_b = -pref  # bearing convention + = left
        score -= 0.5 * np.abs(BEARINGS - pref_b) / 40.0
        score[~sel] = -99
        i = int(np.argmax(score))
        tb = BEARINGS[i]
        fr_n = frs[-1][i]
        v = 1.0 if fr_n > 3.0 else (0.5 if fr_n > 1.5 else 0.25)
        w = float(np.clip(math.radians(tb) * 1.6, -1.0, 1.0))
        # nearest grass in the strip 0.3-0.8 m ahead (for the log: how close we run to the edge)
        near = gm[_NEAR_ROWS]
        if near.any():
            c_now = float(np.min(np.abs(GL[_NEAR_ROWS][near])))
            if bot.debug and c_now < 0.45 and len([f for f in os.listdir(DBG_DIR) if f.startswith("near_")]) < 12:
                bot.save("near_%04d" % bot.k)
            min_clear = min(min_clear, c_now)
        bot.drive(v, w)
        p = bot.pos()
        s_len += float(np.hypot(*(p - last)))
        last = p.copy()
        if 45 < s_len < 68:
            lane_heads.append(bot.hd)
        if bot.k % 40 == 0:
            log("follow t=%.1f s=%.1f x=%.1f y=%.1f hd=%.0f tb=%+d fr=%.1f score=%.2f clear=%.2f"
                % (bot.s["time"], s_len, bot.x, bot.y, bot.hd, tb, fr_n, score[i], min_clear))
            min_clear = 9.0
            if bot.debug and bot.k % 200 == 0:
                bot.save("follow_%04d" % bot.k)
        # look for the bins at the end of the lane
        if s_len > 62:
            bb = blue_blob(im)
            if bb is not None and bb["fwd"] > 0:
                # world position of the cart's bottom-centre
                cam = p + CAM_AHEAD * hvec(bot.hd)
                hdv = hvec(bot.hd); left = hvec(bot.hd - 90)
                wp = cam + bb["fwd"] * hdv + bb["left"] * left
                rng = math.hypot(bb["fwd"], bb["left"])
                brg = math.degrees(math.atan2(bb["left"], bb["fwd"]))
                bin_hits.append((rng, wp))
                if len(bin_hits) % 10 == 1:
                    log("  cart seen: range %.1f bearing %.0f world %s" % (rng, brg, np.round(wp, 1)))
                if rng < 9.0 or brg < -32:
                    break
        if s_len > 110 or bot.s["time"] - t_start > 200:
            raise RuntimeError("did not find the bins")
    lane_h = (lane_heads[0] + np.mean([wrap180(h - lane_heads[0]) for h in lane_heads])) % 360 if lane_heads else 174.0
    near = [wp for (r, wp) in bin_hits if r < 12.0] or [bin_hits[-1][1]]
    return s_len, lane_h, np.mean(near[-10:], axis=0)


# ----------------------------------------------------------------------------- phase 2
def estimate_back_face(bot, n_hat, frames=4):
    """from the current pose (cart in view), estimate the cart's back-face bottom centre in world.
    Uses the bottom edge of the blue (+dark wheels) blob projected on the ground."""
    pts_all = []
    for _ in range(frames):
        im = bot.img()
        bm = blue_mask(im)
        dark = im.max(2) < 45
        bb = blue_blob(im)
        if bb is None:
            bot.stop(2); continue
        u0, u1 = bb["u0"], bb["u1"]
        cam = bot.pos() + CAM_AHEAD * hvec(bot.hd)
        hdv = hvec(bot.hd); left = hvec(bot.hd - 90)
        for u in range(u0, u1 + 1):
            col = np.nonzero(bm[:, u])[0]
            if len(col) == 0:
                continue
            vb = col.max()
            # extend through dark wheel pixels directly below
            while vb + 1 < H and dark[vb + 1, u] and vb - col.max() < 4:
                vb += 1
            f, l = ground_point(u, vb + 1)
            if not np.isfinite(f):
                continue
            pts_all.append(cam + float(f) * hdv + float(l) * left)
        bot.stop(2)
    if len(pts_all) < 5:
        return None
    P = np.array(pts_all)
    e_hat = np.array([n_hat[1], -n_hat[0]])  # n rotated clockwise 90 (to the right of n)
    pn = P @ n_hat; pe = P @ e_hat
    back = pn > np.max(pn) - 0.18
    ce = 0.5 * (np.percentile(pe[back], 5) + np.percentile(pe[back], 95))
    width = np.percentile(pe[back], 95) - np.percentile(pe[back], 5)
    cn = np.percentile(pn[back], 70)
    return cn * n_hat + ce * e_hat, float(width)


_LAWN_ROWS = list(range(int(np.argmin(np.abs(GF[:, W // 2] - 6.0))), int(np.argmin(np.abs(GF[:, W // 2] - 0.6))) + 1, 2))


def lawn_right(im):
    """lateral position of the grass edge on the right: distance (m) from the robot centre line to
    the left boundary of the grass region that reaches the right image border. Negative means the
    edge is left of our centre line (we are heading along the lawn itself)."""
    gm = grass_mask(im)
    ds = []
    for v in _LAWN_ROWS:
        row = gm[v]
        if row[-8:].mean() < 0.8:          # grass must reach the right border on this row
            continue
        k = W - 1
        while k > 0 and (row[k] or row[max(0, k - 3):k].any()):
            k -= 1
        ds.append(-GL[v, min(W - 1, k + 1)])
    return float(np.median(ds)) if len(ds) >= 3 else None


def measure_lawn(bot, n=3):
    vals = []
    for _ in range(n):
        r = lawn_right(bot.img())
        if r is not None:
            vals.append(r)
        bot.stop(2)
    return float(np.median(vals)) if vals else None


LAWN_CLEAR = 1.05   # axle distance from the lawn edge while in the strip behind the carts


def stage(bot, lane_h, bin_wp, log):
    """drive into the strip of street between the lawn and the carts, stop abeam the blue
    cart and turn to face it."""
    n_hat = hvec(lane_h + 180.0)   # cart backs face up the lane
    e_hat = np.array([n_hat[1], -n_hat[0]])
    est = estimate_back_face(bot, n_hat)
    face = est[0] if est is not None else bin_wp
    log("back face est %s (blob est %s)" % (np.round(face, 2), np.round(bin_wp, 2)))
    s_axle = face + (0.06 + 1.5) * n_hat     # conservative row: the lawn is ~2.8 m behind the carts
    # point on our lane track abeam the staging point
    p = bot.pos(); d = hvec(lane_h)
    A = np.array([[d[0], -e_hat[0]], [d[1], -e_hat[1]]])
    a, b = np.linalg.solve(A, s_axle - p)
    M = p + a * d
    log("stage: pos %s M %s S %s (a=%.1f b=%.1f)" % (np.round(p, 2), np.round(M, 2), np.round(s_axle, 2), a, b))
    if a > 0.2:
        bot.turn_to(lane_h, tol=3)
        bot.drive_to(M, v=0.6)
    bot.M = [bot.x, bot.y]
    west_h = (lane_h + 90.0) % 360
    bot.turn_to(west_h, tol=2)
    bot.save("stage_M")
    est = estimate_back_face(bot, n_hat)
    if est is not None and np.hypot(*(est[0] - face)) < 1.5:
        face = est[0]
        log("back face est2 %s" % np.round(face, 2))
    # get onto the row LAWN_CLEAR south of the lawn edge before heading into the strip
    dl = measure_lawn(bot)
    if dl is not None and abs(dl - LAWN_CLEAR) > 0.1:
        sh = dl - LAWN_CLEAR                       # + = move right (towards the lawn)
        log("  lawn %.2f m right at the lane mouth: side-step %+.2f m" % (dl, sh))
        bot.turn_to(heading_of(n_hat if sh > 0 else -n_hat), tol=2)
        bot.drive_to(bot.pos() + sh * n_hat, v=0.4)
        bot.turn_to(west_h, tol=2)
    for leg in range(2):
        dl = measure_lawn(bot)
        along = float((bot.pos() - face) @ e_hat)        # how far east of the cart we are
        shift = (dl - LAWN_CLEAR) if dl is not None else float((s_axle - bot.pos()) @ n_hat)
        if dl is None:
            log("  lawn edge not seen, using cart estimate")
        frac = 0.5 if leg == 0 else 1.0
        tgt = bot.pos() - frac * along * e_hat + shift * n_hat * (0.8 if leg == 0 else 1.0)
        log("  leg %d: lawn %.2f m right, %.2f m to go, shift %.2f" % (leg, dl if dl is not None else -1, along, shift))
        bot.drive_to(tgt, v=0.4)
        if leg == 0:
            bot.turn_to(west_h, tol=2)
    bot.turn_to(lane_h, tol=2)
    bot.S = [bot.x, bot.y]
    bot.west_h = west_h
    bot.save("stage_S")
    return n_hat


# ----------------------------------------------------------------------------- phase 3
def latch_on(bot, lane_h, log, max_tries=4):
    bot.logging = False
    for attempt in range(max_tries):
        realign(bot, lane_h, log)
        ok = approach_and_latch(bot, lane_h, log)
        if ok:
            return True
        log("latch attempt %d failed, backing out" % attempt)
        bot.latch(False)
        # back straight out ~1 m
        p0 = bot.pos().copy()
        while np.hypot(*(bot.pos() - p0)) < 0.9:
            bot.drive(-0.2, 0)
        bot.stop()
    return False


def measure_bar(bot, n=3):
    res = []
    hd = bot.avg_heading(12)
    for _ in range(n):
        im = bot.img()
        bb = blue_blob(im)
        bar = find_bar(im, bb["uc"] if bb else None)
        if bar is not None and bar["w"] > 8:
            cam = bot.pos() + CAM_AHEAD * hvec(hd)
            wp = cam + bar["fwd"] * hvec(hd) + bar["left"] * hvec(hd - 90)
            res.append((wp, bar))
        bot.stop(2)
    if not res:
        return None, None
    wp = np.median(np.array([r[0] for r in res]), axis=0)
    return wp, res[-1][1]


def realign(bot, lane_h, log, tol=0.08):
    """facing the cart from ~1-1.5 m: shift sideways (turn 90, drive, turn back) until the
    axle is on the bar's normal line, then point straight at the bar."""
    n_hat = hvec(lane_h + 180)
    e_hat = np.array([n_hat[1], -n_hat[0]])
    for it in range(3):
        wp, bar = measure_bar(bot)
        if wp is None:
            im = bot.img(); bb = blue_blob(im)
            if bb is None:
                log("realign: no cart in view"); bot.turn_to(lane_h, tol=2); return
            ang = math.degrees(math.atan2(bb["left"], bb["fwd"]))
            bot.turn_to(bot.hd - ang, tol=1.5)
            continue
        lat = float((bot.pos() - wp) @ e_hat)      # + = we are east of the bar line
        dist = float((bot.pos() - wp) @ n_hat) - CAM_AHEAD
        log("realign %d: camera %.2f m behind bar, lateral %+.3f m" % (it, dist, lat))
        if abs(lat) < tol:
            break
        h = (lane_h + 90) % 360 if lat > 0 else (lane_h - 90) % 360
        bot.turn_to(h, tol=1.5)
        bot.drive_to(bot.pos() - lat * e_hat, v=0.15, tol=0.01)
        bot.turn_to(lane_h, tol=1.5)
    wp, bar = measure_bar(bot)
    if bar is not None:
        ang = math.degrees(math.atan2(bar["left"], bar["fwd"]))
        if abs(ang) > 1.0:
            bot.turn_to(bot.hd - ang, tol=1.0)


def approach_and_latch(bot, lane_h, log):
    last = None
    u_hint = None
    measured = False
    while True:
        im = bot.img()
        bb = blue_blob(im)
        if u_hint is None and bb is not None:
            u_hint = bb["uc"]
        bar = find_bar(im, u_hint)
        if bar is None:
            if last is None:
                # aim at the cart itself
                if bb is None:
                    log("no cart in view"); return False
                ang = math.atan2(bb["left"], bb["fwd"])
                bot.drive(0.12, float(np.clip(ang * 1.5, -0.4, 0.4)))
                continue
            break
        if bar["v"] > 222 or bar["fwd"] < 0.21:
            last = bar
            break
        u_hint = bar["u"]
        last = bar
        # orientation check once at ~0.6 m
        if not measured and bar["fwd"] < 0.65 and bar["w"] > 40:
            measured = True
            # slope -> depth difference across bar: dv/dd = f*0.16/d^2
            dvdd = F * 0.16 / bar["fwd"] ** 2
            dd = bar["slope"] * bar["w"] / dvdd           # right minus left depth (m) over bar span
            yaw = math.degrees(math.atan2(dd, bar["width_m"]))
            log("bar at %.2f m: lateral %.3f yaw %.1f deg (u=%.0f w=%.0f)" % (bar["fwd"], bar["left"], yaw, bar["u"], bar["w"]))
        ang = math.atan2(bar["left"], bar["fwd"])
        v = 0.18 if bar["fwd"] > 0.6 else 0.1
        bot.drive(v, float(np.clip(ang * 1.5, -0.35, 0.35)))
        if bar["fwd"] < 0.4 and bot.k % 3 == 0:
            bot.stop(1)
            if bot.latch(True):
                log("latched while approaching (bar %.2f m)" % bar["fwd"])
                return True
    log("blind phase, last bar fwd %.2f left %.3f" % (last["fwd"], last["left"]))
    bot.save("blind_start")
    d = 0.0
    while d < 0.22:
        bot.drive(0.1, 0, 6); bot.stop(2); d += 0.01
        if bot.latch(True):
            log("latched after blind %.2f m" % d)
            return True
    return False


# ----------------------------------------------------------------------------- phase 4
def steer_bearing(im, pref_b, pref_w=0.5, widths=(2.2, 1.6, 1.1, 0.7, 0.4), wts=(0.7, 1.0, 1.0, 1.0, 1.5)):
    """pavement-following bearing for one camera image (camera frame, + = camera's left)."""
    pf = polar_grass(grass_mask(im))
    score = np.zeros(NB)
    fr_last = None
    for hw, wt in zip(widths, wts):
        fr_last = free_ranges(pf, hw)
        score += wt * np.minimum(fr_last, 6.0) / 6.0
    score -= pref_w * np.abs(BEARINGS - pref_b) / 40.0
    score[np.abs(BEARINGS) > 40] = -99
    i = int(np.argmax(score))
    return float(BEARINGS[i]), float(fr_last[i])


def tow_home(bot, log, speed=0.5, look=2.0):
    """pull the cart home by reversing: steer from the REAR camera (pavement follower), with the
    route (which branch to take) given by the logged outbound path, retraced backwards."""
    bot.logging = False
    t0 = bot.s["time"]
    # 1. straight back out to the staging row (1 m clear of the gray cart's swing), then pivot
    #    clockwise so the cart swings round to the west and we face along the strip
    if getattr(bot, "S", None) is not None:
        bot.drive_to(np.array(bot.S), v=0.3, reverse=True)
        bot.turn_to(bot.west_h, wmax=0.5, tol=3)
    pts = np.array([p[:2] for p in bot.path])[::-1]
    j = 0
    while j < len(pts) - 1 and np.hypot(*(pts[j] - bot.pos())) < 0.5:
        j += 1
    tow_trace = []
    min_clear, n_near = 9.0, 0
    while True:
        while j < len(pts) - 1 and np.hypot(*(pts[j] - bot.pos())) < look:
            j += 1
        tgt = pts[j]
        d = float(np.hypot(*(tgt - bot.pos())))
        back_h = (bot.hd + 180) % 360
        e = wrap180(heading_of(tgt - bot.pos()) - back_h)   # + = route goes clockwise of travel
        if j == len(pts) - 1:
            seg = pts[-1] - pts[max(0, len(pts) - 8)]
            seg = seg / max(1e-6, float(np.hypot(*seg)))
            if d < 0.15 or float(np.dot(pts[-1] - bot.pos(), seg)) < 0.05:
                break
        pref_b = float(np.clip(-e, -40, 40))                # rear camera: + = its left = CCW of travel
        remaining = d + 0.2 * (len(pts) - 1 - j)
        im = bot.img("rear")
        if remaining < 3.0:
            # last few metres: keep straight (the start is straight in front of the garage)
            tb, fr = steer_bearing(im, 0.0, pref_w=1.5)
            tb = float(np.clip(tb, -6, 6))
        else:
            tb, fr = steer_bearing(im, pref_b, pref_w=1.0)
            if abs(e) > 50:                                  # sharp corner of the route: mostly pivot
                tb = pref_b
        if j == len(pts) - 1:
            v = -min(speed, max(0.08, d))
        elif abs(tb) < 12 and fr > 2.0:
            v = -speed
        elif abs(tb) < 30:
            v = -0.25
        else:
            v = -0.08
        w = float(np.clip(math.radians(tb) * 1.6, -0.7, 0.7))
        gnear = grass_mask(im)[_NEAR_ROWS]
        if gnear.any():
            c_now = float(np.min(np.abs(GL[_NEAR_ROWS][gnear])))
            min_clear = min(min_clear, c_now)
            if bot.debug and c_now < 0.4 and n_near < 15:
                bot.save("tnear_%05d" % bot.k, "rear"); n_near += 1
        bot.drive(v, w)
        tow_trace.append((bot.x, bot.y))
        if bot.k % 100 == 0:
            log("tow t=%.1f x=%.1f y=%.1f hd=%.0f j=%d/%d e=%+.0f tb=%+.0f fr=%.1f latched=%s"
                % (bot.s["time"], bot.x, bot.y, bot.hd, j, len(pts), e, tb, fr, bot.s["latched"]))
            if bot.debug and bot.k % 400 == 0:
                bot.save("tow_rear_%05d" % bot.k, "rear")
        if bot.s["time"] - t0 > 500:
            log("tow timeout"); break
    bot.stop()
    log("tow finished: closest grass seen 0.3-0.8 m behind the tail: %.2f m to the side" % min_clear)
    bot.save("home_rear", "rear")
    return tow_trace


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="dev: continue from tmp/robot/run/state.json")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--stop-after", default=None, help="follow|stage|latch")
    ap.add_argument("--realtime-after", action="store_true")
    ap.add_argument("--verify", action="store_true", help="dev: reference views at start, unlatch+look at end")
    a = ap.parse_args()
    bot = Bot(debug=a.debug)
    wall0 = time.time()

    def log(msg):
        print("[%6.1fs wall] %s" % (time.time() - wall0, msg), flush=True)

    state_file = os.path.join(DBG_DIR, "state.json")

    def dump(phase, **kw):
        if a.debug:
            json.dump(dict(phase=phase, x=bot.x, y=bot.y, path=bot.path, M=getattr(bot, "M", None),
                       S=getattr(bot, "S", None), west_h=getattr(bot, "west_h", None), **kw),
                      open(state_file, "w"))

    if a.resume:
        st = json.load(open(state_file))
        bot.lockstep(); bot.sensors()
        bot.x, bot.y, bot.path = st["x"], st["y"], st["path"]
        bot.M = st.get("M"); bot.S = st.get("S"); bot.west_h = st.get("west_h")
        lane_h, bin_wp = st["lane_h"], np.array(st["bin_wp"])
        log("resumed after %s at t=%.1f" % (st["phase"], bot.s["time"]))
        done = st["phase"]
    else:
        bot.reset(a.seed if a.seed is not None else 0)
        bot.stop()
        bot.save("start")
        log("start: heading %.1f" % bot.hd)
        if a.verify:   # dev: reference views at the start
            h0 = bot.avg_heading(10)
            for dh in (-70, 70, 180):
                bot.turn_to(h0 + dh, tol=1.5); bot.save("ref_%+d" % dh)
            bot.turn_to(h0, tol=1.5)
            bot.x = bot.y = 0.0
        s_len, lane_h, bin_wp = follow_route(bot, log)
        log("reached the bins: path %.1f m, lane heading %.1f, cart near %s, t=%.1f" % (s_len, lane_h, np.round(bin_wp, 2), bot.s["time"]))
        bot.stop()
        done = "follow"
        dump("follow", lane_h=lane_h, bin_wp=list(bin_wp))
    if a.stop_after == "follow":
        return
    if done == "follow":
        stage(bot, lane_h, bin_wp, log)
        bot.save_map("map_stage")
        done = "stage"
        dump("stage", lane_h=lane_h, bin_wp=list(bin_wp))
    if a.stop_after == "stage":
        return
    if done == "stage":
        ok = latch_on(bot, lane_h, log)
        log("latched=%s t=%.1f" % (ok, bot.s["time"]))
        dump("latch", lane_h=lane_h, bin_wp=list(bin_wp))
        if not ok or a.stop_after == "latch":
            bot.stop()
            return
    trace = tow_home(bot, log)
    bot.save_map("map", trace)
    bot.stop()
    s = bot.sensors()
    log("done: t=%.1f latched=%s dr pos (%.2f, %.2f) hd=%.0f" % (s["time"], s["latched"], bot.x, bot.y, s["compass_deg"]))
    if a.verify:   # dev only: drop the cart and look around to compare with the start views
        h0 = bot.path[0][2]
        bot.latch(False)
        for dh in (-70, 70, 180):
            bot.turn_to(h0 + dh, tol=1.5); bot.save("end_%+d" % dh)
    if a.realtime_after:
        bot._req("POST", "/run", json={"running": True})


if __name__ == "__main__":
    main()
