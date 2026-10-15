"""One command: check everything, let MolmoAct2 (on the Lightning GPU) do the task, bring the robot home.

    python do_task.py --pick "paper ball" --into box     # builds the task sentence the way the model was trained
    python do_task.py --pick pen --onto notebook
    python do_task.py "put the duck and the chess queen into the cup"   # several objects: one after another
    python do_task.py "move the arm towards the white pawn, grasp it, lift it up, and drop it into the cup"
    python do_task.py                     # the task in settings.env (MOLMO_TASK)
    python do_task.py --seconds 90 ...    # allow longer (default 60 s; Ctrl+C stops early)
    python do_task.py --check             # only run the checks and save camera previews
    python do_task.py --fake              # no robot/cameras: test the whole chain with a fake robot

Several objects in one instruction are split into one model sentence each (tasks.py) and done in turn; --seconds
is per object. Under the team dashboard's robot agent (ROBOT_LIVE=1) both cameras stream to its Live panel
(live_view.py). Run it in the Terminal app (macOS only lets Terminal use the cameras).
Cameras: FRONT_CAM (phone, top) is required; SIDE_CAM (USB webcam, side) is optional, see settings.env.
Help for the model (assist.py, switch off in settings.env): once the gripper holds the object, the code carries
it to the taught drop pose (python teach_drop.py cup) and opens; after a missed grab it opens, lifts and lets the
model try again; firmer grip; motor temperature watch. Every run is logged in outputs/runs/.
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
import live_view
from tasks import split_tasks

HERE = Path(__file__).parent
S = config.S
STUDIO = S.get("LIGHTNING_SSH", "")
REMOTE_DIR = "/teamspace/studios/this_studio/molmo"
# The model on the Studio: the base model, or a fine-tuned one (MOLMO_CHECKPOINT in settings.env).
CHECKPOINT = S.get("MOLMO_CHECKPOINT") or f"{REMOTE_DIR}/MolmoAct2-SO100_101-LeRobot"
PORT = 8080
SIZES = [(640, 360), (640, 480), (1280, 720)]  # tried in this order if the configured size fails
SEND_SIZE = int(__import__('os').environ.get('SEND_SIZE', 378))  # MolmoAct2's own image size (its processor resizes every image to 378x378): full detail,
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

CLOUD_FILES = ["molmo_server.py", "test_molmo.py", "train_molmo.sh"]  # cloud/ on the Mac -> ~/molmo on the Studio


def md5_local(name: str) -> str:
    import hashlib

    return hashlib.md5((HERE / "cloud" / name).read_bytes()).hexdigest()


def copy_to_studio(names: list[str]) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *[str(HERE / "cloud" / n) for n in names],
                    f"{STUDIO}:{REMOTE_DIR}/"], capture_output=True, timeout=120)


def update_studio_code() -> None:
    """Before the model loads: bring every file in cloud/ on the Studio up to date with this Mac, and restart
    the model server if its code changed (a running server keeps using the code it started with)."""
    remote = ssh(f"cd {REMOTE_DIR} && md5sum {' '.join(CLOUD_FILES)} 2>/dev/null")
    changed = [n for n in CLOUD_FILES if md5_local(n) not in remote]
    if not changed:
        ok("cloud code on the Studio is up to date")
        return
    if "molmo_server.py" in changed:
        print("        model server code changed: stopping the old server, it restarts with the new code", flush=True)
        ssh("pkill -f '[m]olmo_server'; sleep 2")
    copy_to_studio(changed)
    remote = ssh(f"cd {REMOTE_DIR} && md5sum {' '.join(changed)}")
    still = [n for n in changed if md5_local(n) not in remote]
    if still:
        fail(f"could not update {still} on the Studio", "check the Studio is running, then run again")
    ok(f"updated on the Studio: {', '.join(changed)}")


def check_studio_health() -> None:
    """Same LeRobot version as this Mac, model files present, GPU not busy with training."""
    import importlib.metadata

    info = ssh(f"cd {REMOTE_DIR} && source .venv/bin/activate && "
               "python -c 'import importlib.metadata as m; print(\"LEROBOT\", m.version(\"lerobot\"))'; "
               f"test -f {CHECKPOINT}/config.json && echo CHECKPOINT_OK; "
               "pgrep -f '[l]erobot-train' >/dev/null && echo TRAINING_RUNNING; "
               "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=90)
    if "CHECKPOINT_OK" not in info:
        fail(f"model not found on the Studio: {CHECKPOINT}",
             "MOLMO_CHECKPOINT in settings.env: empty = base model, or the exact path train_molmo.sh printed")
    if "TRAINING_RUNNING" in info:
        fail("a training run is using the GPU right now",
             f"wait for it (ssh {STUDIO} 'tail ~/molmo/train.log') or stop it: ssh {STUDIO} \"pkill -f lerobot-train\"")
    mac = importlib.metadata.version("lerobot")
    studio = next((line.split()[1] for line in info.splitlines() if line.startswith("LEROBOT ")), "unknown")
    if studio != mac:
        print(f"  WARN  LeRobot {studio} on the Studio, {mac} on this Mac: they should match (tell Claude if the run fails)")
    gpu = next((line for line in info.splitlines() if "MiB" in line), "GPU not found")
    ok(f"Studio ready: {gpu.strip()}, LeRobot {studio}, model {Path(CHECKPOINT).name}")


def check_studio() -> None:
    print("\n1. Lightning Studio (MolmoAct2 on the GPU)")
    if not STUDIO:
        fail("LIGHTNING_SSH is not set", "add LIGHTNING_SSH=s_...@ssh.lightning.ai to settings.env")
    out = ssh("echo REACHABLE")
    if "REACHABLE" not in out:
        if "not running" in out:
            fail("the 'arman1' Studio is stopped",
                 "on lightning.ai open Studio 'arman1', press Start, wait until it shows Running (GPU L4), run again")
        fail(f"cannot reach the Studio ({out.splitlines()[-1] if out else 'no answer'})",
             "start the 'arman1' Studio on lightning.ai (GPU: L4); if it says 'permission denied', run Lightning's SSH setup command again")
    ok("Studio reachable")
    # Open the tunnel first: it is a lasting SSH connection, and some Studios pause while none is open.
    if subprocess.run(["nc", "-z", "127.0.0.1", str(PORT)], capture_output=True).returncode != 0:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                        "-f", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}", STUDIO], capture_output=True, timeout=30)
    update_studio_code()
    check_studio_health()
    if "READY" in ssh(f"pgrep -f '[m]olmo_server' >/dev/null && grep -q 'MODEL READY' {REMOTE_DIR}/server.log && echo READY") \
            and "SAME" not in ssh(f"grep -q 'checkpoint={CHECKPOINT}$' {REMOTE_DIR}/server.log && echo SAME"):
        print("        a different model is loaded, restarting the model server with the requested one", flush=True)
        ssh("pkill -f '[m]olmo_server'; sleep 2")
    state = ssh(f"pgrep -f '[m]olmo_server' >/dev/null && "
                f"(grep -q 'MODEL READY' {REMOTE_DIR}/server.log && echo READY || echo LOADING) || echo OFF")
    if "READY" not in state:
        if "LOADING" in state:
            print("        the model server is still loading from an earlier start, waiting for it...", flush=True)
        else:
            print("        model server not running, starting it (2-6 minutes: loading 22 GB into the GPU)...", flush=True)
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
        temps = {j: bus.read("Present_Temperature", j, normalize=False, num_retry=3) for j in config.JOINTS}
    finally:
        bus.disconnect(disable_torque=False)
    if volts < 10:
        fail(f"robot motors get only {volts:.1f} V", "the robot needs its 12V power supply (never 12V on the joystick)")
    hottest = max(temps, key=temps.get)
    if temps[hottest] >= 55:
        fail(f"the {hottest} motor is hot ({temps[hottest]} degrees C)", "switch the robot off for 10 minutes to cool down")
    ok(f"all 6 motors answer, {volts:.1f} V, hottest motor {temps[hottest]} degrees C ({hottest})")
    robot = config.make_robot()  # writes the saved calibration into the motors if they lost it
    robot.connect()
    robot.bus.disable_torque()
    robot.disconnect()
    ok("calibration loaded in the motors")


def open_camera(index: int | str, size: tuple[int, int]):
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    if index == "realsense":
        from lerobot.cameras.zmq import ZMQCamera

        cam = ZMQCamera(config.camera_config("realsense", *size))
        try:
            cam.connect()
            return (640, 480), cam.read(), cam
        except Exception:  # noqa: BLE001
            return None, None, None
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
    wanted = {"cam0": ("FRONT", "top camera"), "cam1": ("SIDE", "side camera")}
    result, problems = {}, {}
    for slot, (prefix, label) in wanted.items():
        setting = S.get(f"{prefix}_CAM", "").strip().lower()
        if setting == "":
            problems[slot] = (f"{prefix}_CAM not set", f"set {prefix}_CAM in settings.env (iphone, webcam or a number)")
            continue
        if setting.isdigit():
            idx = int(setting)
        elif setting == "realsense":
            idx = "realsense"
        elif setting in found:
            idx = found[setting]
        else:
            problems[slot] = (f"{label}: no '{setting}' camera found (found: {list(found) or 'none'})",
                              "iPhone: Bluetooth + Wi-Fi on, locked, near the Mac (or by cable). Webcam: is it "
                              "plugged in? RealSense: use the name realsense (colour, via realsense_bridge.py)")
            continue
        if any(idx == r[0] for r in result.values()):
            problems[slot] = ("top and side camera are the same camera", "set FRONT_CAM and SIDE_CAM to different cameras")
            continue
        size = (int(S.get(f"{prefix}_WIDTH") or 640), int(S.get(f"{prefix}_HEIGHT") or 480))
        got, frame, cam = open_camera(idx, size)
        if got is None:
            problems[slot] = ((f"{label}: no picture from the RealSense",
                               "start it first in another Terminal tab: sudo .venv/bin/python realsense_bridge.py")
                              if idx == "realsense" else
                              (f"{label} (camera {idx}) does not open",
                               "allow camera access for Terminal; check with python cam_test.py"))
            continue
        cam.disconnect()
        preview = HERE / f"outputs/do_task_{slot}.jpg"
        preview.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(preview), cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        ok(f"{label}: {setting} = camera {idx}, {got[0]}x{got[1]}, preview {preview.relative_to(HERE)}")
        result[slot] = (idx, *got)
        CAMERA_NAMES[slot] = setting
    if "cam0" not in result and "cam1" in result:  # one camera is enough: the side camera becomes the model's cam0
        print(f"  --    {problems['cam0'][0]}: using the side camera as the only camera")
        result = {"cam0": result["cam1"]}
        CAMERA_NAMES["cam0"] = CAMERA_NAMES.pop("cam1")
    elif "cam0" not in result:
        fail(*problems["cam0"])
    elif "cam1" in problems and S.get("SIDE_CAM", "").strip():
        print(f"  --    {problems['cam1'][0]} (optional, continuing with the top camera only)")
    elif "cam1" not in result:
        print("  --    no side camera set (optional; cam1 gets a copy of cam0)")
    return result


CAMERA_NAMES: dict[str, str] = {}  # slot -> the settings.env name (iphone, webcam), shown on the live view


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


def background_uploads(client) -> None:
    """LeRobot's control loop waits for every observation upload (one internet round trip, ~0.2 s to Lightning)
    before executing the next movement, which makes the arm stutter. Upload in a background thread instead;
    while one upload is still on its way, newer ones are skipped (the model only needs the latest)."""
    send, busy = client.send_observation, threading.Lock()

    def worker(obs):
        try:
            send(obs)
        except RuntimeError:
            if client.running:  # an upload still on its way when a try ends: nothing to do
                raise
        finally:
            busy.release()

    def send_observation(obs):
        if not busy.acquire(blocking=False):
            return False
        threading.Thread(target=worker, args=(obs,), daemon=True).start()
        return True

    client.send_observation = send_observation


RESULTS = {
    "time": "time is up", "ctrl-c": "stopped with Ctrl+C", "blocked": "stopped: the arm was blocked",
    "hot": "stopped: a motor got hot", "motor-error": "stopped: a motor reported a problem",
    "error": "stopped after an error", "missed": "missed the object (no retries left)",
    "dropped": "the object slipped out (no retries left)",
}


def model_turn(rc, cfg, sup, task: str, seconds: float, attempt: int, stop_on: set[str]) -> str:
    """Let the model drive until the time is up or the grab watch sees something in `stop_on`.
    The robot stays connected (and holds its pose) afterwards."""
    client = rc.RobotClient(cfg)  # reuses the connected robot, see run()
    sup.begin(client, attempt, stop_on)
    background_uploads(client)
    if not client.start():
        client.channel.close()
        sup.finish()
        return "refused"
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    timer = threading.Timer(max(1.0, seconds), sup.end, args=("time",))
    timer.start()
    try:
        client.control_loop(task=task)
    except KeyboardInterrupt:
        sup.end("ctrl-c")
    except Exception as e:  # noqa: BLE001
        print(f"\n   stopped: {e}", flush=True)
        sup.end("error")
    finally:
        timer.cancel()
        client.shutdown_event.set()
        client.channel.close()
        receiver.join(timeout=5)
        sup.finish()
    return sup.outcome or "time"


def run(task: str, seconds: float, cams: dict, fake: bool, drop_name: str | None) -> str:
    from lerobot.async_inference import robot_client as rc
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    import assist
    import poses

    robot_cfg = SO101FollowerConfig(
        port=S["FOLLOWER_PORT"], id=S["FOLLOWER_ID"], max_relative_target=15.0,
        disable_torque_on_disconnect=False,  # we bring it home and switch it off ourselves
        cameras={slot: config.camera_config(i, w, h) for slot, (i, w, h) in cams.items()},
    )
    # One connection for the whole run. Every new connection makes LeRobot switch all motors off for a moment
    # to write its settings: the arm sagged (and dropped what it held) before going home.
    if fake:
        robot = FakeRobot(cams)
    else:
        config._retry_torque_writes()
        robot = config._sync_calibration_on_connect(rc.make_robot_from_config(robot_cfg))
    live_view.install(robot)  # before shrink_images: the dashboard gets the full camera pictures
    shrink_images(robot)
    robot.connect()
    robot.connect = lambda calibrate=True: None  # the model client "connects" it again on every try
    rc.make_robot_from_config = lambda cfg: robot
    # Ending a try closes the connection to the model server on purpose; LeRobot reports that as an error.
    rc.RobotClient.logger.addFilter(lambda r: "Channel closed" not in r.getMessage())
    sup = assist.Supervisor(robot)
    live_view.watch(sup)

    print("\n4. Settings for this run")
    if not fake:
        try:
            ok(f"grip: {assist.set_grip_strength(robot)}")
        except Exception as e:  # noqa: BLE001
            print(f"  --    grip: could not change it ({e}); using LeRobot's setting")
    home = poses.load_poses().get("home")
    drop_key, drop = assist.drop_pose_for(task, drop_name)
    auto_drop = assist.AUTO_DROP and drop is not None and home is not None
    if auto_drop:
        ok(f"auto drop: when the gripper holds something, the code carries it to '{drop_key}' and opens")
    elif assist.AUTO_DROP:
        why = "no 'home' pose taught" if home is None else (
            f"'{drop_key}' not taught yet: python teach_drop.py {drop_key[6:]}" if drop_key
            else "no taught drop pose matches this task (teach one: python teach_drop.py cup)")
        print(f"  --    auto drop off for this run ({why}); the model drops by itself")
    retries = assist.RETRIES if home is not None else 0
    ok(f"retries after a missed grab: {retries}")

    cfg = RobotClientConfig(
        policy_type="molmoact2", pretrained_name_or_path=CHECKPOINT, robot=robot_cfg,
        actions_per_chunk=30, task=task, server_address=f"127.0.0.1:{PORT}",
        policy_device="cuda", chunk_size_threshold=0.7,
    )
    print(f"\n5. Running for up to {seconds:.0f} s (Ctrl+C stops early)\n   task: \"{task}\"", flush=True)
    t0 = time.time()
    deadline = t0 + seconds
    attempt, result = 0, None
    stop_on = ({"holding"} if auto_drop else set()) | ({"missed"} if retries else set())
    try:
        while True:
            attempt += 1
            if attempt > 1:
                print(f"\n   try {attempt}", flush=True)
            outcome = model_turn(rc, cfg, sup, task, deadline - time.time(), attempt, stop_on)
            if outcome == "refused":
                fail("the model server refused the robot", "tell Claude (server log: ~/molmo/server.log)")
            if outcome == "holding":
                outcome = assist.carry(robot, sup, drop, home, open_to=sup.open_level)
                if outcome == "done":
                    result = f"success: carried the object to '{drop_key}' and let go"
                    break
            if outcome in ("missed", "dropped") and retries > 0 and deadline - time.time() > 8:
                retries -= 1
                assist.reopen(robot, sup, home, open_to=sup.open_level)
                continue
            result = RESULTS.get(outcome, outcome)
            if outcome in ("missed", "dropped") and retries > 0:
                result = result.replace("no retries left", "no time left for another try")
            break
    except KeyboardInterrupt:
        result = "stopped with Ctrl+C"
    finally:
        go_home(robot, fake, sup)
        log = sup.save(result or "stopped")
        print(f"\n   RESULT: {result} ({time.time() - t0:.0f} s, {attempt} {'try' if attempt == 1 else 'tries'})")
        shown = log.relative_to(HERE) if log.is_relative_to(HERE) else log
        print(f"   run log: {shown} (log.csv, events.txt, camera pictures)", flush=True)
    return result or "stopped"


def go_home(robot, fake: bool, sup=None) -> None:
    """Ease to 'home' over the open connection, then switch the motors off. If that connection is broken
    (a motor error), connect again and do it from scratch."""
    import assist
    import poses

    print("\n6. Back to the resting pose")
    if sup is not None:
        sup.phase = "home"
    home = poses.load_poses().get("home")
    try:
        if home:
            assist.move(robot, home, 2.0)
        if not fake:
            robot.bus.disable_torque()
        robot.disconnect()
        ok("robot at rest, motors off")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  (going home over the open connection failed: {e}; connecting again)")
    if fake:
        return
    try:
        robot.disconnect()  # free the USB port for the new connection
    except Exception:  # noqa: BLE001
        pass
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
    ap.add_argument("--pick", help='the object, e.g. "paper ball"')
    ap.add_argument("--into", help="container to drop it into, e.g. box")
    ap.add_argument("--onto", help="surface to put it on, e.g. notebook")
    ap.add_argument("--drop", help="taught drop pose (teach_drop.py NAME); default: from --into/--onto or the task text")
    args = ap.parse_args()
    target = args.into or args.onto
    if args.pick:
        if not target:
            fail("--pick needs --into or --onto", 'e.g. python do_task.py --pick "paper ball" --into box')
        # The wording of the model's own example: "move the arm towards the lemon, grasp it, lift it up, and drop
        # it into the red bowl". The model lowercases and strips punctuation itself.
        how = "drop it into the" if args.into else "put it on the"
        task = f"move the arm towards the {args.pick}, grasp it, lift it up, and {how} {target}"
    else:
        task = " ".join(args.task) or S.get("MOLMO_TASK") or "pick up the pawn and drop it into the cup"
    drop_name = args.drop or (target.split()[-1] if target else None)
    todo = split_tasks(task) or [task]
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
        for i, t in enumerate(todo, 1):
            print(f"   task {i}/{len(todo)}: \"{t}\"")
        return
    if args.fake:
        CAMERA_NAMES.update({slot: "fake camera" for slot in cams})
    if len(todo) == 1:
        live_view.set_label("MolmoAct2", CAMERA_NAMES)
        live_view.final(f"done: {run(todo[0], args.seconds, cams, args.fake, drop_name)}")
        return
    print(f"\n{len(todo)} objects, one after another:", *(f"\n   {i}. {t}" for i, t in enumerate(todo, 1)), flush=True)
    results = []
    for i, t in enumerate(todo, 1):
        what = object_name(t)
        print(f"\n========== object {i}/{len(todo)}: {what} ==========", flush=True)
        live_view.set_label(f"{what} ({i}/{len(todo)})", CAMERA_NAMES)
        results.append((what, run(t, args.seconds, cams, args.fake, drop_name)))
        if results[-1][1].startswith("stopped"):  # Ctrl+C / STOP / a guard: don't start the next object
            break
    done = sum(r.startswith("success") for _, r in results)
    summary = "; ".join(f"{w}: {r}" for w, r in results)
    head = "success" if done == len(todo) else f"{done}/{len(todo)} done"
    print(f"\n   RESULT: {head}: {done}/{len(todo)} objects ({summary})", flush=True)
    live_view.final(f"{head}: {done}/{len(todo)} objects")


def object_name(task: str) -> str:
    """'move the arm towards the duck, grasp it, ...' -> 'duck'"""
    return task.split("towards the ", 1)[-1].split(",", 1)[0] if "towards the " in task else task


if __name__ == "__main__":
    main()
