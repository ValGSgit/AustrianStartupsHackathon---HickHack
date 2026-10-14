"""Run this first when the robot is plugged in. Checks everything before you start. Never moves the arm.

    python check.py
"""

import glob
import sys
from pathlib import Path

try:
    import lerobot  # noqa: F401
except ImportError:
    sys.exit("LeRobot not found. Run first:  source .venv/bin/activate")

import config

S = config.S
ok = True


def report(passed: bool, msg: str, fix: str = "") -> None:
    global ok
    ok &= passed
    print(("  OK   " if passed else "  FAIL ") + msg + ("" if passed else f"\n         -> {fix}"))


print("\n1. settings.env")
for key in ("FOLLOWER_PORT", "LEADER_PORT", "HF_USER"):
    report("CHANGE_ME" not in S.get(key, "CHANGE_ME"), f"{key} = {S.get(key)}", "fill it in settings.env")
report(S.get("FOLLOWER_PORT") != S.get("LEADER_PORT"), "follower and leader ports differ",
       "each arm has its own port; run lerobot-find-port for each")

print("\n2. Hugging Face login (needed to upload recordings and train in the cloud)")
try:
    from huggingface_hub import whoami

    user = whoami()["name"]
    report(user == S.get("HF_USER"), f"logged in as {user}",
           f"HF_USER in settings.env is '{S.get('HF_USER')}' but you are logged in as '{user}'")
except Exception:
    report(False, "not logged in", "run: hf auth login   (paste a WRITE token from huggingface.co/settings/tokens)")

print("\n3. USB ports")
ports = glob.glob("/dev/tty.usbmodem*")
print(f"       found: {ports or 'none'}")
for key in ("FOLLOWER_PORT", "LEADER_PORT"):
    report(S.get(key) in ports, f"{key} is plugged in", "plug in the arm AND its power supply, then run lerobot-find-port")

print("\n4. Calibration files (made by lerobot-calibrate)")
calib = Path.home() / ".cache/huggingface/lerobot/calibration"
for path, what in ((calib / "robots/so_follower" / f"{S.get('FOLLOWER_ID')}.json", "follower"),
                   (calib / "teleoperators/so_leader" / f"{S.get('LEADER_ID')}.json", "leader")):
    report(path.exists(), f"{what} calibrated ({path.name})",
           f"run lerobot-calibrate for the {what} with id={path.stem} (README section 0)")

print("\n5. Cameras")
import cv2

for key in ("FRONT_CAM", "WRIST_CAM"):
    idx = S.get(key, "")
    if not idx:
        print(f"  --   {key} not set (skipped)")
        continue
    cap = cv2.VideoCapture(int(idx))
    frame = None
    for _ in range(10):  # some cameras return black/empty frames for the first reads
        got, frame = cap.read()
    cap.release()
    if frame is not None:
        report(True, f"{key}={idx} gives a {frame.shape[1]}x{frame.shape[0]} image")
        if frame.mean() < 10:
            print("         (image is almost black: lens cap, or wrong camera?)")
    else:
        report(False, f"{key}={idx} opens",
               "run lerobot-find-cameras opencv; allow camera access for your terminal app in System Settings > Privacy > Camera")
print("       Note: index 0 is often the MacBook's own webcam. Check the pictures from lerobot-find-cameras.")

print("\n6. Follower arm (reads joints, does not move)")
if S.get("FOLLOWER_PORT") in ports:
    try:
        robot = config.make_robot()
        robot.connect(calibrate=False)
        joints = {k.split(".")[0]: round(v, 1) for k, v in robot.get_observation().items() if k.endswith(".pos")}
        robot.disconnect()
        report(True, f"joints: {joints}")
    except Exception as e:
        report(False, f"connect failed: {e}", "check power supply, 3-pin cables, and calibration (step 4)")
else:
    print("  --   skipped (port missing)")

print("\n7. Saved poses")
from pick_place import taught_spots
from poses import load_poses

poses = load_poses()
spots = taught_spots(poses)
report("home" in poses and "above_cup" in poses and bool(spots),
       f"home/above_cup taught, spots taught: {spots}", "python poses.py teach  (see README)")

print("\nAll good. Go!" if ok else "\nFix the FAIL lines above, then run again.")
sys.exit(0 if ok else 1)
