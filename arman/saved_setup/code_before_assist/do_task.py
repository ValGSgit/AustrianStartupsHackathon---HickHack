"""One command: check everything, let MolmoAct2 (on the Lightning GPU) do the task, bring the robot home.

    python do_task.py "move the arm towards the white pawn, grasp it, lift it up, and drop it into the cup"
    python do_task.py                     # the task in settings.env (MOLMO_TASK)
    python do_task.py --seconds 90 "..."  # run longer (default 60 s; Ctrl+C stops early)
    python do_task.py --check             # only run the checks and save camera previews
    python do_task.py --fake              # no robot/cameras: test the whole chain with a fake robot

Run it in the Terminal app (macOS only lets Terminal use the cameras).
Cameras: FRONT_CAM (phone, top) is required; SIDE_CAM (USB webcam, side) is optional, see settings.env.
Stopping (time up, Ctrl+C, or the safety guard): the robot holds its pose, eases to 'home', goes limp.
"""

import argparse
import functools
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config

HERE = Path(__file__).parent
S = config.S
STUDIO = S.get("LIGHTNING_SSH", "")
REMOTE_DIR = "/teamspace/studios/this_studio/molmo"
# The model on the Studio: the base model, or a fine-tuned one (MOLMO_CHECKPOINT in settings.env).
CHECKPOINT = S.get("MOLMO_CHECKPOINT") or f"{REMOTE_DIR}/MolmoAct2-SO100_101-LeRobot"
PORT = 8080
SIZES = [(640, 360), (640, 480), (1280, 720)]  # tried in this order if the configured size fails
BLOCKED_DIFF, BLOCKED_SECONDS = 25.0, 1.5
SEND_SIZE = 378  # MolmoAct2's own image size (its processor resizes every image to 378x378): full detail,
                 # and as JPEG still ~15x lighter than a raw camera frame


def ok(msg: str) -> None:
    print(f"  OK    {msg}", flush=True)


def fail(msg: str, fix: str) -> None:
    print(f"  FAIL  {msg}\n        -> {fix}", flush=True)
    sys.exit(1)


def ssh(cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", STUDIO, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


# ---------------------------------------------------------------- checks

def check_studio() -> None:
    print("\n1. Lightning Studio (MolmoAct2 on the GPU)")
    if not STUDIO:
        fail("LIGHTNING_SSH is not set", "add LIGHTNING_SSH=s_...@ssh.lightning.ai to settings.env")
    out = ssh("echo REACHABLE")
    if "REACHABLE" not in out:
        fail(f"cannot reach the Studio ({out.splitlines()[-1] if out else 'no answer'})",
             "start the Studio on lightning.ai (GPU: L40S); if it says 'permission denied', run Lightning's SSH setup command again")
    ok("Studio reachable")
    # Open the tunnel first: it is a lasting SSH connection, and some Studios pause while none is open.
    if subprocess.run(["nc", "-z", "127.0.0.1", str(PORT)], capture_output=True).returncode != 0:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                        "-f", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}", STUDIO], capture_output=True, timeout=30)
    import hashlib

    local_md5 = hashlib.md5((HERE / "cloud/molmo_server.py").read_bytes()).hexdigest()
    if local_md5 not in ssh(f"md5sum {REMOTE_DIR}/molmo_server.py"):
        print("        server code changed, restarting the model server with the new version", flush=True)
        ssh("pkill -f '[m]olmo_server'; sleep 2")
    if "READY" in ssh(f"pgrep -f '[m]olmo_server' >/dev/null && grep -q 'MODEL READY' {REMOTE_DIR}/server.log && echo READY") \
            and "SAME" not in ssh(f"grep -q 'checkpoint={CHECKPOINT}$' {REMOTE_DIR}/server.log && echo SAME"):
        print("        a different model is loaded, restarting the model server with the requested one", flush=True)
        ssh("pkill -f '[m]olmo_server'; sleep 2")
    if "READY" not in ssh(f"pgrep -f '[m]olmo_server' >/dev/null && grep -q 'MODEL READY' {REMOTE_DIR}/server.log && echo READY"):
        print("        model server not running, starting it (2-6 minutes: loading 22 GB into the GPU)...", flush=True)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(HERE / "cloud/molmo_server.py"),
                        f"{STUDIO}:{REMOTE_DIR}/"], capture_output=True, timeout=60)  # always the latest server code
        ssh(f"cd {REMOTE_DIR} && source .venv/bin/activate && "
            f"(nohup python -u molmo_server.py --port={PORT} --checkpoint={CHECKPOINT} > server.log 2>&1 &)")
        for _ in range(144):  # up to 12 min: after the Studio wakes from sleep, the disk is slow at first
            time.sleep(5)
            log = ssh(f"grep -E 'MODEL READY|Traceback|Error' {REMOTE_DIR}/server.log | tail -2")
            if "MODEL READY" in log:
                break
            if "Traceback" in log or "Error" in log:
                fail(f"model server crashed: {log}", "tell Claude (log: ~/molmo/server.log on the Studio)")
        else:
            fail("model server did not get ready within 12 minutes", "tell Claude")
    ok("MolmoAct2 loaded and ready on the GPU")

    import grpc
    from lerobot.transport import services_pb2, services_pb2_grpc

    try:
        ch = grpc.insecure_channel(f"127.0.0.1:{PORT}")
        grpc.channel_ready_future(ch).result(timeout=15)
        t = time.perf_counter()
        services_pb2_grpc.AsyncInferenceStub(ch).Ready(services_pb2.Empty())
        ch.close()
    except Exception as e:  # noqa: BLE001
        fail(f"secure tunnel to the model server does not answer ({e})",
             "close old tunnels: pkill -f 'ssh.*-L 8080' and run again")
    ok(f"secure tunnel to the model server works ({(time.perf_counter() - t) * 1000:.0f} ms)")


def check_robot() -> None:
    print("\n2. Robot")
    import glob

    port = S["FOLLOWER_PORT"]
    if port not in glob.glob("/dev/tty.usbmodem*"):
        fail(f"robot USB not found ({port})", "plug in the robot's USB cable and its 12V power supply")
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=port, motors={j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
                                              for i, j in enumerate(config.JOINTS)})
    bus.connect(handshake=False)
    try:
        found = bus.broadcast_ping() or {}
        missing = [j for i, j in enumerate(config.JOINTS) if i + 1 not in found]
        if missing:
            fail(f"motors not answering: {missing}", "check the robot's power supply and 3-pin cables (python motors.py)")
        volts = min(bus.read("Present_Voltage", j, normalize=False, num_retry=3) for j in config.JOINTS) / 10
    finally:
        bus.disconnect(disable_torque=False)
    if volts < 10:
        fail(f"robot motors get only {volts:.1f} V", "the robot needs its 12V power supply (never 12V on the joystick)")
    ok(f"all 6 motors answer, {volts:.1f} V")
    robot = config.make_robot()  # writes the saved calibration into the motors if they lost it
    robot.connect()
    robot.bus.disable_torque()
    robot.disconnect()
    ok("calibration loaded in the motors")


def open_camera(index: int, size: tuple[int, int]):
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    for w, h in [size] + [s for s in SIZES if s != size]:
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, width=w, height=h, fps=30))
        try:
            cam.connect()
            frame = cam.read()
            return (w, h), frame, cam
        except Exception:  # noqa: BLE001
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
    return None, None, None


def check_cameras() -> dict[str, tuple[int, int, int]]:
    """Returns {slot: (index, width, height)} for the cameras that work. Saves a preview of each.
    FRONT_CAM / SIDE_CAM in settings.env: a number, 'iphone' or 'webcam' (found automatically), or empty."""
    print("\n3. Cameras")
    import cv2

    found = config.identify_cameras()
    print(f"        found: {found or 'no cameras'}")
    wanted = {"cam0": ("FRONT", "top camera", True), "cam1": ("SIDE", "side camera", False)}
    result = {}
    for slot, (prefix, label, required) in wanted.items():
        setting = S.get(f"{prefix}_CAM", "").strip().lower()
        if setting == "":
            if required:
                fail(f"{prefix}_CAM not set", f"set {prefix}_CAM in settings.env (iphone, webcam or a number)")
            print(f"  --    no {label} set (optional; {slot} gets a copy of cam0)")
            continue
        if setting.isdigit():
            idx = int(setting)
        elif setting in found:
            idx = found[setting]
        else:
            fail(f"{label}: no '{setting}' camera found (found: {list(found) or 'none'})",
                 "iPhone: Bluetooth + Wi-Fi on, locked, near the Mac (or by cable). Webcam: is it plugged in?")
        size = (int(S.get(f"{prefix}_WIDTH") or 640), int(S.get(f"{prefix}_HEIGHT") or 480))
        got, frame, cam = open_camera(idx, size)
        if got is None:
            fail(f"{label} (camera {idx}) does not open", "allow camera access for Terminal; check with python cam_test.py")
        cam.disconnect()
        preview = HERE / f"outputs/do_task_{slot}.jpg"
        preview.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(preview), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        ok(f"{label}: {setting} = camera {idx}, {got[0]}x{got[1]}, preview {preview.relative_to(HERE)}")
        result[slot] = (idx, *got)
    if len(result) == 2 and result["cam0"][0] == result["cam1"][0]:
        fail("top and side camera are the same camera", "set FRONT_CAM and SIDE_CAM to different cameras")
    return result


# ---------------------------------------------------------------- the run

def shrink_images(robot) -> None:
    """Send camera images as 378x378 JPEGs (the size MolmoAct2's processor resizes to, same stretch) instead
    of raw full-size frames. The server (cloud/molmo_server.py) unpacks them."""
    import cv2

    feats = dict(robot.observation_features)
    cams = [k for k, v in feats.items() if isinstance(v, tuple)]
    robot.__dict__["observation_features"] = {**feats, **{k: (SEND_SIZE, SEND_SIZE, 3) for k in cams}}
    observe = robot.get_observation

    def get_observation():
        obs = observe()
        for k in cams:
            small = cv2.resize(np.asarray(obs[k]), (SEND_SIZE, SEND_SIZE), interpolation=cv2.INTER_AREA)
            obs[k] = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        return obs

    robot.get_observation = get_observation


class FakeRobot:
    """Stand-in for the arm + cameras, so the whole chain can be tested without hardware."""

    name = "so101_follower"
    is_connected = True
    cameras = {}

    def __init__(self, cams):
        self.cams = cams
        self.state = {f"{j}.pos": v for j, v in zip(config.JOINTS, [0.0, -97.0, 95.0, 75.0, 0.0, 2.0])}
        self.actions = 0

    @functools.cached_property
    def action_features(self):
        return {f"{j}.pos": float for j in config.JOINTS}

    @functools.cached_property
    def observation_features(self):
        return {**self.action_features, **{s: (h, w, 3) for s, (_, w, h) in self.cams.items()}}

    def connect(self, calibrate=True):
        pass

    def disconnect(self):
        print(f"  [fake robot] performed {self.actions} movements, final pose "
              + ", ".join(f"{k.split('.')[0]}={v:.0f}" for k, v in self.state.items()))

    def get_observation(self):
        # A smooth test picture (like a real camera image, unlike random noise, which JPEG cannot compress).
        y, x = np.mgrid[0:1:complex(0, 360), 0:1:complex(0, 640)]
        base = (np.stack([x, y, (x + y) / 2], axis=-1) * 255).astype(np.uint8)
        imgs = {s: cv2.resize(base, (w, h)) for s, (_, w, h) in self.cams.items()}
        return {**self.state, **imgs}

    def send_action(self, action):
        self.state.update(action)
        self.actions += 1
        return action


def guard(robot, client) -> None:
    """Stop the run if a joint cannot reach its commanded position (something blocks the arm) or a motor errors."""
    last_cmd, since = {}, [None]
    send, observe = robot.send_action, robot.get_observation

    def send_action(action):
        try:
            sent = send(action)
        except Exception as e:  # noqa: BLE001
            print(f"\n  STOP: a motor reported a problem ({e}). Something may be blocking the arm.", flush=True)
            client.shutdown_event.set()
            raise
        last_cmd.update(sent)
        return sent

    def get_observation():
        obs = observe()
        stuck = [k for k, v in last_cmd.items() if k not in ("gripper.pos", "wrist_roll.pos")
                 and k in obs and abs(v - obs[k]) > BLOCKED_DIFF]
        if not stuck:
            since[0] = None
        elif since[0] is None:
            since[0] = time.time()
        elif time.time() - since[0] > BLOCKED_SECONDS:
            print(f"\n  STOP: the robot cannot reach its target ({', '.join(k.split('.')[0] for k in stuck)}).",
                  flush=True)
            client.shutdown_event.set()
        return obs

    robot.send_action, robot.get_observation = send_action, get_observation


def background_uploads(client) -> None:
    """LeRobot's control loop waits for every observation upload (one internet round trip, ~0.2 s to Lightning)
    before executing the next movement, which makes the arm stutter. Upload in a background thread instead;
    while one upload is still on its way, newer ones are skipped (the model only needs the latest)."""
    send, busy = client.send_observation, threading.Lock()

    def worker(obs):
        try:
            send(obs)
        finally:
            busy.release()

    def send_observation(obs):
        if not busy.acquire(blocking=False):
            return False
        threading.Thread(target=worker, args=(obs,), daemon=True).start()
        return True

    client.send_observation = send_observation


def run(task: str, seconds: float, cams: dict, fake: bool) -> None:
    from lerobot.async_inference import robot_client as rc
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    robot_cfg = SO101FollowerConfig(
        port=S["FOLLOWER_PORT"], id=S["FOLLOWER_ID"], max_relative_target=15.0,
        disable_torque_on_disconnect=False,  # hold the pose when the run ends; we bring it home after
        cameras={slot: OpenCVCameraConfig(index_or_path=i, width=w, height=h, fps=30) for slot, (i, w, h) in cams.items()},
    )
    make = rc.make_robot_from_config
    def build(cfg):
        if fake:
            robot = FakeRobot(cams)
        else:
            config._retry_torque_writes()
            robot = config._sync_calibration_on_connect(make(cfg))
        shrink_images(robot)
        return robot

    rc.make_robot_from_config = build

    cfg = RobotClientConfig(
        policy_type="molmoact2", pretrained_name_or_path=CHECKPOINT, robot=robot_cfg,
        actions_per_chunk=30, task=task, server_address=f"127.0.0.1:{PORT}",
        policy_device="cuda", chunk_size_threshold=0.7,
    )
    print(f"\n4. Running for {seconds:.0f} s (Ctrl+C stops early)\n   task: \"{task}\"", flush=True)
    client = rc.RobotClient(cfg)
    guard(client.robot, client)
    background_uploads(client)
    if not client.start():
        client.stop()
        fail("the model server refused the robot", "tell Claude (server log: ~/molmo/server.log)")
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    timer = threading.Timer(seconds, client.shutdown_event.set)
    timer.start()
    t0 = time.time()
    try:
        client.control_loop(task=task)
    except KeyboardInterrupt:
        print("\n   stopped with Ctrl+C", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n   stopped: {e}", flush=True)
    finally:
        timer.cancel()
        client.shutdown_event.set()
        try:
            client.stop()
        except Exception as e:  # noqa: BLE001
            print(f"   (robot disconnect: {e})")
        receiver.join(timeout=5)
        print(f"   ran {time.time() - t0:.0f} s", flush=True)


def go_home(fake: bool) -> None:
    print("\n5. Back to the resting pose")
    if fake:
        ok("(fake robot, nothing to move)")
        return
    import poses

    robot = config.make_robot()
    robot.connect()
    poses.release(robot)
    ok("robot at rest, motors off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="*")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--check", action="store_true", help="only run the checks")
    ap.add_argument("--fake", action="store_true", help="fake robot and cameras (tests the cloud chain)")
    args = ap.parse_args()
    task = " ".join(args.task) or S.get("MOLMO_TASK") or "pick up the pawn and drop it into the cup"
    if not shutil.which("nc"):
        fail("the 'nc' tool is missing", "tell Claude")

    check_studio()
    if args.fake:
        cams = {"cam0": (0, 640, 360)} | ({"cam1": (1, 640, 480)} if S.get("SIDE_CAM") else {})
        print("\n2./3. Robot and cameras: FAKE (test mode)")
    else:
        check_robot()
        cams = check_cameras()
    if args.check:
        print("\nAll checks passed. Ready to run.")
        return
    try:
        run(task, args.seconds, cams, args.fake)
    finally:
        go_home(args.fake)


if __name__ == "__main__":
    main()
