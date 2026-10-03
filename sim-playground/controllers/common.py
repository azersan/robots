"""Small helpers shared by controllers: a gateway client and path following."""

import math

import httpx


class Gateway:
    def __init__(self, url="http://localhost:8642"):
        self.c = httpx.Client(base_url=url, timeout=30.0)

    def get(self, path, **params):
        r = self.c.get(path, params=params)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.content

    def post(self, path, **body):
        r = self.c.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def info(self):
        return self.get("/api/info")

    def state(self):
        return self.get("/api/state")

    def reset(self, **kw):
        return self.post("/api/reset", **kw)

    def lockstep(self):
        self.post("/api/run", running=False)

    def realtime(self, speed=1.0):
        self.post("/api/run", running=True, speed=speed)

    def step(self, frames):
        return self.post("/api/step", frames=frames)

    def drive(self, v, w):
        return self.post("/api/drive", v=v, w=w)

    def latch(self, engage=True):
        return self.post("/api/latch", engage=engage)

    def camera(self, name, fmt="png"):
        return self.get(f"/api/camera/{name}.{fmt}")

    def depth(self, name):
        return self.get(f"/api/camera/{name}/depth.png")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def closest_index(path, p):
    return min(range(len(path)), key=lambda i: math.dist(path[i], p))


def path_length_from(path, i, p):
    """Remaining distance along path from point p (projected near index i) to the end."""
    total = math.dist(p, path[min(i + 1, len(path) - 1)])
    total += sum(math.dist(a, b) for a, b in zip(path[i + 1:], path[i + 2:]))
    return total


def lookahead_point(path, p, lookahead):
    """Pure-pursuit target: first point on the polyline at least `lookahead` ahead of p's projection."""
    best_i, best_t, best_d = 0, 0.0, math.inf
    for i, (a, b) in enumerate(zip(path, path[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy or 1e-9
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
        d = math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1])
        if d < best_d:
            best_i, best_t, best_d = i, t, d
    remaining = lookahead
    i, t = best_i, best_t
    while i < len(path) - 1:
        a, b = path[i], path[i + 1]
        seg = math.dist(a, b)
        left = seg * (1 - t)
        if left >= remaining:
            f = t + remaining / (seg or 1e-9)
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), best_i, best_d
        remaining -= left
        i, t = i + 1, 0.0
    return path[-1], best_i, best_d


def pure_pursuit(pose, target, v):
    """Yaw rate that arcs from pose (x, y, yaw) through target at speed v."""
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    L2 = dx * dx + dy * dy or 1e-9
    return v * 2.0 * local_y / L2
