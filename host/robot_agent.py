"""Laptop agent: connects the robot (or the MuJoCo sim) to the Cloudflare robot-router Worker.

It opens an OUTBOUND WebSocket to the Worker (works behind venue Wi-Fi, no open ports), announces the
skills from config.toml [skills.*], and then:
  snapshot -> sends one JPEG from the same camera the policy uses
  run      -> runs the skill (real: robot.py rollout / replay, sim: policy or motion in MuJoCo)
  stop     -> stops the running skill
and reports status + an after-snapshot when a skill finishes. While a skill runs it streams what the
dashboard's Live panel shows: `log` lines (inference, phases) and, in the sim, ~8 Hz `frame`s (camera JPEG +
joint state + the commanded joint targets).

    .venv\\Scripts\\python host\\robot_agent.py --url https://robot-router.<you>.workers.dev --token <ROBOT_TOKEN>
    .venv\\Scripts\\python host\\robot_agent.py --url http://127.0.0.1:8787 --token dev --sim --show
"""

import argparse
import asyncio
import base64
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot import load_config  # noqa: E402 - config.toml + this laptop's config.local.toml (ports, camera index)

CONFIG = load_config()
SKILLS = CONFIG.get("skills", {})


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FRAME_HZ = 8  # live stream rate to the dashboard (each frame ~25 KB)
PREVIEW_HZ = 5  # idle camera / arm preview rate


def jpeg_b64(rgb: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


class Backend:
    """Runs jobs one at a time on its own thread (MuJoCo's GL context must stay on one thread). Between jobs it
    calls idle() ~10x a second, which streams the camera preview / arm positions when they are switched on."""

    PREVIEW_MAX_S = 600  # the preview switches itself off after 10 min, so a forgotten tab doesn't stream all day

    def __init__(self, send):
        self.send = send  # thread-safe: send(dict)
        self.jobs: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = False
        self.camera_on = False
        self.camera_since = 0.0
        self.next_preview = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def skills(self) -> list[dict]:
        raise NotImplementedError

    def log(self, text: str) -> None:
        print("  ", text, flush=True)
        self.send({"type": "log", "text": text})

    def report_hw(self) -> None:
        self.send({"type": "hw", **self.hw_status()})

    def submit(self, msg: dict) -> None:
        if msg["type"] == "stop":
            self.stop_flag.set()
            self.on_stop()
        elif msg["type"] == "snapshot" and self.busy:
            self.send({"type": "snapshot", "id": msg["id"], "jpeg": self.busy_snapshot()})
        elif msg["type"] == "hw" and self.busy:
            self.log(f"'{msg.get('action')}' ignored: a skill is running")
        else:
            self.jobs.put(msg)

    def _loop(self) -> None:
        self.setup()
        self.report_hw()
        while True:
            try:
                msg = self.jobs.get(timeout=0.1)
            except queue.Empty:
                try:
                    if self.camera_on and time.perf_counter() - self.camera_since > self.PREVIEW_MAX_S:
                        self.hw("camera_off")
                        self.log("camera preview switched off after 10 min")
                        self.report_hw()
                    elif time.perf_counter() >= self.next_preview:
                        self.next_preview = time.perf_counter() + 1 / PREVIEW_HZ
                        self.idle()
                except Exception as e:  # e.g. USB unplugged: switch the preview off instead of looping on errors
                    self.log(f"preview error: {e}")
                    self.hw("camera_off")
                    self.hw("arm_off")
                    self.report_hw()
                continue
            try:
                if msg["type"] == "snapshot":
                    self.send({"type": "snapshot", "id": msg["id"], "jpeg": self.snapshot()})
                elif msg["type"] == "hw":
                    self.hw(str(msg.get("action", "")))
                    self.report_hw()
                elif msg["type"] == "run":
                    self.busy = True
                    self.stop_flag.clear()
                    self.send({"type": "status", "state": "running", "detail": f"running {msg['skill']}"})
                    ok, detail = self.run(msg["skill"])
                    self.busy = False
                    self.send({"type": "done", "skill": msg["skill"], "ok": ok, "detail": detail, "jpeg": self.snapshot()})
            except Exception as e:  # report instead of dying - the stage demo must keep going
                self.busy = False
                if msg["type"] == "hw":
                    self.log(f"{msg.get('action')} failed: {e}")
                    self.report_hw()
                else:
                    self.send({"type": "done", "skill": msg.get("skill", "?"), "ok": False, "detail": f"error: {e}"})

    def preview(self, rgb: np.ndarray | None, state: list[float] | None, label: str) -> None:
        jpeg = jpeg_b64(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA), 70) if rgb is not None else None
        self.send({"type": "frame", "label": label, "jpeg": jpeg, "joints": JOINTS, "state": state, "target": None})

    # hooks
    def setup(self) -> None: ...
    def snapshot(self) -> str | None: ...
    def busy_snapshot(self) -> str | None: return None
    def run(self, skill: str) -> tuple[bool, str]: ...
    def on_stop(self) -> None: ...
    def hw(self, action: str) -> None: ...
    def hw_status(self) -> dict: ...
    def idle(self) -> None: ...


class CameraReader:
    """Holds a webcam open and keeps only its newest frame (a reader thread, so the preview never lags)."""

    def __init__(self, cam: dict):
        self.cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {cam['index']} did not open - another program (teleop, rollout) using it?")
        self.frame: np.ndarray | None = None
        self.fps = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        t0 = time.perf_counter()
        while self.frame is None and time.perf_counter() - t0 < 4:
            time.sleep(0.05)
        if self.frame is None:
            self.close()
            raise RuntimeError(f"camera {cam['index']} opened but sends no frames")

    def _run(self) -> None:
        n, t0 = 0, time.perf_counter()
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame, n = f, n + 1
            else:
                time.sleep(0.01)
            if time.perf_counter() - t0 >= 2:
                self.fps, n, t0 = n / (time.perf_counter() - t0), 0, time.perf_counter()
        self.cap.release()

    def rgb(self) -> np.ndarray | None:
        return None if self.frame is None else self.frame[:, :, ::-1]

    def close(self) -> None:
        self.running = False
        self.thread.join(2)  # the device must be free before lerobot opens it


class RealBackend(Backend):
    """Real SO-101: skills run as robot.py subprocesses (mirrored live by host/live_hook.py). While idle the
    dashboard can switch on a camera preview and a READ-ONLY follower connection (positions only, torque untouched)."""

    def __init__(self, send):
        self.proc: subprocess.Popen | None = None
        self.cam = next(iter(CONFIG.get("cameras", {}).values()), {"index": 0, "width": 640, "height": 480})
        self.reader: CameraReader | None = None
        self.bus = None  # lerobot FeetechMotorsBus while the arm is connected
        self.calibrated = False
        self.arm_error = ""
        self.camera_error = ""
        self.live_jpeg: str | None = None
        super().__init__(send)

    def skills(self):
        return [{"name": n, "description": s["description"]} for n, s in SKILLS.items()
                if "policy" in s or "replay_dataset" in s]

    # ---- devices

    def hw(self, action):
        if action == "camera_on" and not self.reader:
            self.camera_error = ""
            try:
                self.reader = CameraReader(self.cam)
            except RuntimeError as e:
                self.camera_error = str(e)
                raise
            self.camera_on, self.camera_since = True, time.perf_counter()
            self.log(f"camera {self.cam['index']} on")
        elif action == "camera_off":
            if self.reader:
                self.reader.close()
                self.log("camera off")
            self.reader, self.camera_on = None, False
        elif action == "arm_on" and not self.bus:
            self.arm_error = ""
            try:
                self._arm_connect()
            except Exception as e:
                self.arm_error = next((ln.strip() for ln in str(e).splitlines() if ln.strip()), repr(e))[:200]
                raise
        elif action == "arm_off":
            if self.bus:
                self.bus.disconnect(disable_torque=False)  # leave torque as it was: the arm must not drop
                self.log("arm disconnected")
            self.bus = None

    def _arm_connect(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        f = CONFIG["follower"]
        robot = SO101Follower(SO101FollowerConfig(port=f["port"], id=f["id"], calibration_dir=ROOT / "calibration"))
        robot.bus.connect()  # opens the port and pings the 6 motors; no configure(), no torque change
        self.bus = robot.bus
        self.calibrated = bool(robot.calibration) and robot.bus.is_calibrated
        if not robot.calibration:
            self.log(f"arm on {f['port']}: no calibration/{f['id']}.json - run robot.py calibrate follower")
        elif not self.calibrated:
            self.log(f"arm on {f['port']}: calibration file does not match the motors - recalibrate")
        else:
            self.log(f"arm connected on {f['port']} (read-only: positions, torque untouched)")

    def _arm_state(self) -> list[float] | None:
        if not (self.bus and self.calibrated):
            return None
        vals = self.bus.sync_read("Present_Position", num_retry=2)
        return [round(float(vals[j]), 1) for j in JOINTS]

    def hw_status(self):
        f = CONFIG["follower"]
        if self.bus:
            arm = {"state": "on" if self.calibrated else "warn",
                   "detail": f"{f['port']} · " + ("calibrated · read-only" if self.calibrated else "not calibrated")}
        else:
            arm = {"state": "error" if self.arm_error else "off", "detail": self.arm_error or f"{f['port']} · not connected"}
        c = self.cam
        if self.reader:
            cam = {"state": "on", "detail": f"index {c['index']} · {c['width']}x{c['height']}"}
        else:
            cam = {"state": "error" if self.camera_error else "off",
                   "detail": self.camera_error or f"index {c['index']} · off"}
        return {"arm": arm, "camera": cam}

    def idle(self):
        if self.reader or self.bus:
            self.preview(self.reader.rgb() if self.reader else None, self._arm_state(), "live preview")

    # ---- skills

    def snapshot(self):
        if self.reader and self.reader.frame is not None:
            return jpeg_b64(self.reader.rgb())
        # camera not switched on: open it just for this frame (only while no skill runs - lerobot holds it then)
        cap = cv2.VideoCapture(self.cam["index"], cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam["height"])
        frame = None
        for _ in range(5):  # first frames are often dark while auto-exposure settles
            ok, f = cap.read()
            frame = f if ok else frame
        cap.release()
        return jpeg_b64(frame[:, :, ::-1]) if frame is not None else None

    def busy_snapshot(self):
        return self.live_jpeg

    def run(self, skill):
        s = SKILLS[skill]
        py = [sys.executable, str(ROOT / "robot.py")]
        if "policy" in s:
            cmd = py + ["rollout", "--policy", s["policy"], "--duration", str(s.get("duration_s", 25))]
        else:
            cmd = py + ["replay", "--dataset", s["replay_dataset"], "--episode", str(s.get("replay_episode", 0))]
        # lerobot needs the camera and the COM port to itself (Windows: one process per device)
        had = (self.reader is not None, self.bus is not None)
        if any(had):
            self.log("releasing camera / arm port for lerobot")
            self.hw("camera_off")
            self.hw("arm_off")
            self.report_hw()
        print(">", " ".join(cmd), flush=True)
        self.log("run: " + " ".join(cmd[1:]))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONUTF8": "1", "ROBOT_LIVE": "1"})
        t0, last = time.perf_counter(), 0.0
        self.live_jpeg = None
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("@@LIVE "):  # camera + joints + targets from inside lerobot (host/live_hook.py)
                try:
                    m = json.loads(line[7:])
                except ValueError:
                    continue
                self.live_jpeg = m.get("jpeg") or self.live_jpeg
                self.send({"type": "frame", "t": round(time.perf_counter() - t0, 2), "joints": JOINTS, **m})
            elif line and (time.perf_counter() - last > 0.5 or "error" in line.lower()):
                last = time.perf_counter()  # lerobot's own log, rate-limited so progress bars don't flood the socket
                self.log(line[:300])
        code = self.proc.wait()
        self.proc = None
        for on, action in zip(had, ("camera_on", "arm_on")):
            if on:
                try:
                    self.hw(action)
                except Exception as e:
                    self.log(f"could not reopen after the run: {e}")
        self.report_hw()
        if self.stop_flag.is_set():
            return False, "stopped"
        return code == 0, f"{skill} finished (exit {code})"

    def on_stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()  # e-stop is still the follower's power plug


class SimBackend(Backend):
    """MuJoCo sim stands in for the real arm, so the whole Cloudflare chain can be tested without hardware."""

    def __init__(self, send, show: bool):
        self.show = show
        self.last = None
        self.t0 = 0.0
        self.next_frame = 0.0
        super().__init__(send)

    def skills(self):
        return [{"name": n, "description": s["description"]} for n, s in SKILLS.items()
                if "sim_policy" in s or "sim_motion" in s]

    def setup(self):
        sys.path.insert(0, str(ROOT / "sim"))
        from so101_sim.env import HOME, SO101PickEnv

        self.HOME = HOME
        self.env = SO101PickEnv("pen")
        self.seed = 1000
        self.last = self.env.reset(seed=self.seed).image
        self.policies = {}

    def _frame(self, obs, target=None):
        self.last = obs.image
        if self.show:
            cv2.imshow("SO-101 sim (robot_agent)", obs.image[:, :, ::-1])
            cv2.waitKey(1)
        now = time.perf_counter()
        if now >= self.next_frame:  # live view for the dashboard
            self.next_frame = now + 1 / FRAME_HZ
            small = cv2.resize(obs.image, (320, 240), interpolation=cv2.INTER_AREA)
            self.send({"type": "frame", "t": round(now - self.t0, 2), "jpeg": jpeg_b64(small, 70),
                       "joints": JOINTS, "state": [round(float(v), 1) for v in obs.state],
                       "target": None if target is None else [round(float(v), 1) for v in target]})

    def snapshot(self):
        return jpeg_b64(self.env.render())

    def hw(self, action):
        if action == "camera_on":
            self.camera_on, self.camera_since = True, time.perf_counter()
            self.log("sim camera preview on")
        elif action == "camera_off":
            self.camera_on = False
            self.log("sim camera preview off")
        elif action in ("arm_on", "arm_off"):
            self.log("the sim arm is always connected")

    def hw_status(self):
        return {"arm": {"state": "on", "detail": "MuJoCo sim · always connected"},
                "camera": {"state": "on" if self.camera_on else "off", "detail": "sim camera · 640x480"}}

    def idle(self):
        if self.camera_on:
            obs = self.env.observe()
            self.preview(obs.image, [round(float(v), 1) for v in obs.state], "live preview (sim)")

    def busy_snapshot(self):
        return jpeg_b64(self.last) if self.last is not None else None

    def _policy(self, path: str):
        if path not in self.policies:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg = PreTrainedConfig.from_pretrained(path)
            cfg.device = dev.type
            pol = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(dev).eval()
            pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                                 preprocessor_overrides={"device_processor": {"device": dev.type}})
            self.policies[path] = (pol, pre, post, dev)
        return self.policies[path]

    def run(self, skill):
        s = SKILLS[skill]
        motion = s.get("sim_motion")
        self.t0 = time.perf_counter()
        self.next_frame = 0.0
        if motion == "reset":
            self.seed += 1
            self._frame(self.env.reset(seed=self.seed))
            self.log(f"scene reset (seed {self.seed})")
            return True, "new pen placed"
        if motion == "wave":
            sys.path.insert(0, str(ROOT / "host"))
            from leader_to_sim import NAMES, wave

            self.log("scripted motion: 6 s wave, joint targets from a sine pattern (no AI model)")
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < 6 and not self.stop_flag.is_set():
                target = np.array([wave(t)[n] for n in NAMES], np.float32)
                self._frame(self.env.step(target), target)
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - t0 - t)))
            self.log("returning to home pose")
            for _ in range(30):
                self._frame(self.env.step(self.HOME), self.HOME)
            return True, "waved"

        from lerobot.common.control_utils import predict_action

        # resolve checkpoints/last (a junction) so a newer checkpoint is picked up without restarting the agent
        path = os.path.realpath(ROOT / s["sim_policy"])
        if not Path(path).exists():
            return False, f"sim policy not trained yet: {s['sim_policy']}"
        if path not in self.policies:
            self.log(f"loading policy {Path(path).relative_to(ROOT) if Path(path).is_relative_to(ROOT) else path}")
        pol, pre, post, dev = self._policy(path)
        pol.reset(); pre.reset(); post.reset()
        c = pol.config
        self.log(f"{c.type.upper()} policy on {dev.type}: camera 640x480 + 6 joint angles in -> "
                 f"chunk of {getattr(c, 'chunk_size', '?')} future joint targets out, executes {getattr(c, 'n_action_steps', '?')} of them at 30 Hz")
        ensemble = getattr(c, "temporal_ensemble_coeff", None) is not None  # then it infers every step
        obs = self.env.observe()
        lifted = False
        for i in range(int(s.get("duration_s", 8) * 30)):
            if self.stop_flag.is_set():
                return False, "stopped"
            t = time.perf_counter()
            infer = i % 30 == 0 if ensemble else len(getattr(pol, "_action_queue", ())) == 0
            act = predict_action({"observation.state": obs.state, "observation.images.front": obs.image},
                                 pol, dev, pre, post, use_amp=False, task=s["description"], robot_type="so101_follower_sim")
            target = act.cpu().numpy().reshape(-1)
            if infer:
                ms = (time.perf_counter() - t) * 1000
                self.log(f"step {i}: inference {ms:.0f} ms - state [{', '.join(f'{v:.0f}' for v in obs.state)}] "
                         f"-> next target [{', '.join(f'{v:.0f}' for v in target)}]")
            obs = self.env.step(target)
            self._frame(obs, target)
            if not lifted and self.env.is_success():
                self.log(f"step {i}: pen is off the table")
            lifted |= self.env.is_success()
            time.sleep(max(0, 1 / 30 - (time.perf_counter() - t)))
        return lifted, "pen lifted" if lifted else "pen not lifted"


async def main_async(a) -> None:
    import websockets

    url = a.url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + f"/api/agent?token={a.token}"
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    send = lambda m: loop.call_soon_threadsafe(outbox.put_nowait, m)  # noqa: E731 - called from backend thread
    backend = SimBackend(send, a.show) if a.sim else RealBackend(send)
    mode = "sim" if a.sim else "real"

    while True:
        try:
            async with websockets.connect(url, max_size=8 * 2**20, ping_interval=20) as ws:
                await ws.send(json.dumps({"type": "hello", "mode": mode, "skills": backend.skills(), **backend.hw_status()}))
                print(f"connected ({mode}) - skills: {[s['name'] for s in backend.skills()]}", flush=True)

                async def pump():
                    while True:
                        await ws.send(json.dumps(await outbox.get()))

                pump_task = asyncio.create_task(pump())
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        print("<", msg.get("type"), msg.get("skill", ""), flush=True)
                        backend.submit(msg)
                finally:
                    pump_task.cancel()
        except Exception as e:
            print(f"connection lost ({e.__class__.__name__}: {e}) - retrying in 3 s", flush=True)
            await asyncio.sleep(3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="Worker URL, e.g. https://robot-router.<you>.workers.dev")
    p.add_argument("--token", required=True, help="the ROBOT_TOKEN secret")
    p.add_argument("--sim", action="store_true", help="drive the MuJoCo sim instead of the real arm")
    p.add_argument("--show", action="store_true", help="(sim) show the sim camera in a window")
    a = p.parse_args()
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
