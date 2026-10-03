"""Sim Gateway: HTTP API in front of the simulator.

    python -m simpg.gateway [--scene flat|anna_pl] [--port 8642] [--paused]

Two ways to drive time:
  * real time: the gateway steps the sim itself (POST /api/run {"running": true})
  * lockstep:  pause it and call POST /api/step {"frames": n} from your controller

Open http://localhost:8642/ for the web viewer.
"""

import argparse
import io
import os
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from .sim import CAMERAS, FPS, SCENES, Sim

HERE = os.path.dirname(__file__)


class Gateway:
    def __init__(self, scene, seed=0, running=True):
        self.lock = threading.RLock()
        self.sim = Sim(scene, seed)
        self.running = running
        self.speed = 1.0
        self.sim_fps = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        next_t = time.perf_counter()
        count, window = 0, time.perf_counter()
        while not self._stop:
            if not self.running:
                time.sleep(0.01)
                next_t = time.perf_counter()
                continue
            with self.lock:
                self.sim.step(1)
            count += 1
            now = time.perf_counter()
            if now - window >= 1.0:
                self.sim_fps, count, window = count / (now - window), 0, now
            next_t += 1.0 / (FPS * self.speed)
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:  # fell behind (e.g. a slow render); don't try to catch up
                next_t = time.perf_counter()


class ResetReq(BaseModel):
    scene: str | None = None
    seed: int | None = None
    randomize: bool = False


class StepReq(BaseModel):
    frames: int = 1


class RunReq(BaseModel):
    running: bool = True
    speed: float | None = None


class DriveReq(BaseModel):
    v: float = 0.0  # m/s, + forward
    w: float = 0.0  # rad/s, + left


class LatchReq(BaseModel):
    engage: bool = True


class ConfigReq(BaseModel):
    cmd_timeout: float | None = None


def create_app(gw: Gateway) -> FastAPI:
    app = FastAPI(title="Sim Gateway")

    @app.get("/")
    def viewer():
        return FileResponse(os.path.join(HERE, "viewer.html"))

    @app.get("/api/info")
    def info():
        with gw.lock:
            out = gw.sim.info()
        out.update(running=gw.running, speed=gw.speed, cmd_timeout=gw.sim.cmd_timeout)
        return out

    @app.get("/api/state")
    def state():
        with gw.lock:
            out = gw.sim.get_state()
        out.update(running=gw.running, sim_fps=round(gw.sim_fps, 1))
        return out

    @app.post("/api/reset")
    def reset(req: ResetReq):
        with gw.lock:
            if req.scene and req.scene != gw.sim.scene_name:
                if req.scene not in SCENES:
                    raise HTTPException(404, f"unknown scene {req.scene!r}; have {list(SCENES)}")
                timeout = gw.sim.cmd_timeout
                gw.sim = Sim(req.scene, req.seed or 0)
                gw.sim.cmd_timeout = timeout
                if req.randomize:
                    gw.sim.reset(randomize=True)
            else:
                gw.sim.reset(seed=req.seed, randomize=req.randomize)
            return gw.sim.get_state()

    @app.post("/api/step")
    def step(req: StepReq):
        if gw.running:
            raise HTTPException(409, "sim is running in real time; POST /api/run {\"running\": false} first")
        with gw.lock:
            gw.sim.step(max(0, min(req.frames, 6000)))
            return gw.sim.get_state()

    @app.post("/api/run")
    def run(req: RunReq):
        gw.running = req.running
        if req.speed:
            gw.speed = max(0.05, min(req.speed, 20.0))
        return {"running": gw.running, "speed": gw.speed}

    @app.post("/api/drive")
    def drive(req: DriveReq):
        with gw.lock:
            return gw.sim.set_drive(req.v, req.w)

    @app.post("/api/latch")
    def latch(req: LatchReq):
        with gw.lock:
            return gw.sim.latch(req.engage)

    @app.post("/api/config")
    def config(req: ConfigReq):
        with gw.lock:
            if req.cmd_timeout is not None:
                gw.sim.cmd_timeout = max(0.0, req.cmd_timeout)
            return {"cmd_timeout": gw.sim.cmd_timeout}

    @app.get("/api/camera/{name}.{fmt}")
    def camera(name: str, fmt: str, quality: int = 80):
        if name not in CAMERAS:
            raise HTTPException(404, f"unknown camera {name!r}; have {list(CAMERAS)}")
        with gw.lock:
            rgb, _ = gw.sim.render(name)
        return _encode(rgb, fmt, quality)

    @app.get("/api/camera/{name}/depth.png")
    def depth(name: str):
        """16-bit PNG, millimeters; 0 = no hit."""
        if name not in CAMERAS:
            raise HTTPException(404, f"unknown camera {name!r}")
        with gw.lock:
            _, d = gw.sim.render(name)
        mm = np.where(d > 0, np.clip(d * 1000.0, 0, 65535), 0).astype(np.uint16)
        return _encode(mm, "png")

    return app


def _encode(arr, fmt, quality=80):
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        return Response(buf.getvalue(), media_type="image/jpeg")
    if fmt == "png":
        Image.fromarray(arr).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")
    raise HTTPException(400, "fmt must be png or jpg")


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="flat", choices=list(SCENES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--paused", action="store_true", help="start in lockstep mode")
    args = ap.parse_args()

    gw = Gateway(args.scene, args.seed, running=not args.paused)
    print(f"Sim Gateway: scene={args.scene} -> http://localhost:{args.port}/")
    uvicorn.run(create_app(gw), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
