"""One launcher for the whole SO-101 workflow: setup -> teleop -> record -> train -> rollout.

Every command is built from config.toml, so ports, calibration ids and camera
keys/resolution/fps are identical between recording and running the policy.

Usage (from the project folder):
    .venv\\Scripts\\python robot.py check
    .venv\\Scripts\\python robot.py <command> --help
Add --dry-run before the command to print the lerobot command instead of running it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config() -> dict:
    # config.local.toml (gitignored) overrides config.toml per laptop: COM ports, camera index, GPU settings.
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    local = ROOT / "config.local.toml"
    if local.exists():
        _merge(cfg, tomllib.loads(local.read_text(encoding="utf-8")))
    return cfg


CONFIG = load_config()
SCRIPTS = Path(sys.executable).parent  # .venv/Scripts holds the lerobot-*.exe entry points
DATA_DIR = ROOT / "data"
# Calibration lives in the repo (calibration/<id>.json), not in each laptop's HF cache,
# so every laptop maps joints exactly like the one the dataset was recorded with.
CALIB_DIR = ROOT / "calibration"
OUTPUTS_DIR = ROOT / "outputs" / "train"

DRY_RUN = False


# ---------------------------------------------------------------- helpers

def run(tool: str, *args: str) -> None:
    if tool == "lerobot-train":
        # Windows-safe wrapper: replaces the `checkpoints/last` symlink (needs admin) with a junction.
        cmd = [sys.executable, str(ROOT / "train_win.py"), *args]
    elif os.environ.get("ROBOT_LIVE") == "1" and tool in ("lerobot-rollout", "lerobot-replay"):
        # host/robot_agent.py: same tool, but the follower's camera/joints/targets are mirrored for the dashboard
        cmd = [sys.executable, str(ROOT / "host" / "live_hook.py"), tool, *args]
    else:
        exe = SCRIPTS / f"{tool}.exe"
        cmd = [str(exe if exe.exists() else tool), *args]
    print(">", " ".join(cmd), flush=True)
    if DRY_RUN:
        return
    # lerobot prints non-cp1252 characters; without UTF-8 mode it crashes on Windows consoles.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    sys.exit(subprocess.call(cmd, env=env))


def hf_repo(name: str, local_ok: bool = False) -> str:
    user = CONFIG["hf"]["user"]
    if user == "CHANGE_ME":
        if local_ok:  # local-only data (e.g. sim datasets with --no-push) doesn't need a Hub account
            return f"local/{name}"
        sys.exit("Set [hf] user in config.toml first (your Hugging Face username).")
    return f"{user}/{name}"


def cameras_arg() -> str:
    cams = CONFIG.get("cameras", {})
    parts = [
        f"{name}: {{type: opencv, index_or_path: {c['index']}, width: {c['width']}, "
        f"height: {c['height']}, fps: {c['fps']}}}"
        for name, c in cams.items()
    ]
    return "{" + ", ".join(parts) + "}"


def robot_args(with_cameras: bool = True) -> list[str]:
    f = CONFIG["follower"]
    args = [
        f"--robot.type={f['type']}", f"--robot.port={f['port']}", f"--robot.id={f['id']}",
        f"--robot.calibration_dir={CALIB_DIR}",
    ]
    if with_cameras and CONFIG.get("cameras"):
        args.append(f"--robot.cameras={cameras_arg()}")
    return args


def teleop_args() -> list[str]:
    l = CONFIG["leader"]
    return [
        f"--teleop.type={l['type']}", f"--teleop.port={l['port']}", f"--teleop.id={l['id']}",
        f"--teleop.calibration_dir={CALIB_DIR}",
    ]


def dataset_root(name: str) -> Path:
    return DATA_DIR / name


def local_policy_dir(name: str) -> Path:
    return OUTPUTS_DIR / name / "checkpoints" / "last" / "pretrained_model"


def smolvla_rename_map() -> str:
    """SmolVLA's base model names its cameras camera1..3; ours are named in config.toml ([cameras.front] ...).
    The same map is needed when fine-tuning and at rollout, so the checkpoint keeps the base model's names."""
    names = list(CONFIG.get("cameras", {}))[:3]
    return json.dumps({f"observation.images.{n}": f"observation.images.camera{i + 1}" for i, n in enumerate(names)})


def policy_config(path: str) -> dict:
    """config.json of a local checkpoint or Hub policy ({} if it can't be read)."""
    try:
        local = Path(path) / "config.json"
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(path, "config.json")).read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- commands

def cmd_check(_):
    import serial.tools.list_ports
    import torch

    print(f"python   {sys.version.split()[0]}")
    print(f"torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu      {props.name}  {props.total_memory / 2**30:.1f} GiB")
    ports = list(serial.tools.list_ports.comports())
    print("serial   " + (", ".join(f"{p.device} ({p.description})" for p in ports) or "none found"))
    print(f"follower {CONFIG['follower']['port']}   leader {CONFIG['leader']['port']}   (from config.toml)")
    print(f"cameras  {cameras_arg()}")
    print(f"hf user  {CONFIG['hf']['user']}")


def cmd_find_port(_):
    run("lerobot-find-port")


def cmd_find_cameras(_):
    run("lerobot-find-cameras", "opencv")


def cmd_setup_motors(a):
    # Only needed for brand-new / re-used motors: rewrites motor ids, one motor connected at a time.
    if a.arm == "follower":
        f = CONFIG["follower"]
        run("lerobot-setup-motors", f"--robot.type={f['type']}", f"--robot.port={f['port']}")
    else:
        l = CONFIG["leader"]
        run("lerobot-setup-motors", f"--teleop.type={l['type']}", f"--teleop.port={l['port']}")


def cmd_positions(a):
    """Print joint values in LeRobot units - the same numbers datasets store and policies output."""
    import time

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    l, f = CONFIG["leader"], CONFIG["follower"]
    devices = [("leader", SO101Leader(SO101LeaderConfig(port=l["port"], id=l["id"], calibration_dir=CALIB_DIR)))]
    if a.follower:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # keep torque on when we disconnect, so the follower holds its pose instead of dropping
        devices.append(("follower", SO101Follower(SO101FollowerConfig(
            port=f["port"], id=f["id"], calibration_dir=CALIB_DIR, disable_torque_on_disconnect=False))))
    for label, d in devices:
        d.connect(calibrate=False)
        if not d.is_calibrated:
            sys.exit(f"{label}: motors don't match calibration/{CONFIG[label]['id']}.json - run: robot.py restore-calibration {label}")
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    print("            " + " ".join(f"{n[:12]:>13}" for n in names))
    try:
        while True:
            for label, d in devices:
                vals = d.get_action() if label == "leader" else d.get_observation()
                print(f"{label:>10}: " + " ".join(f"{vals[n + '.pos']:13.1f}" for n in names), flush=True)
            if not a.watch:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        for _, d in devices:
            d.disconnect()


def _arm(label: str):
    """SO101 follower/leader object for config.toml's port + id, with calibration/<id>.json loaded (not connected)."""
    c = CONFIG[label]
    if label == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        return SO101Follower(SO101FollowerConfig(port=c["port"], id=c["id"], calibration_dir=CALIB_DIR))
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    return SO101Leader(SO101LeaderConfig(port=c["port"], id=c["id"], calibration_dir=CALIB_DIR))


def sync_calibration(label: str, write: bool, force: bool = False) -> bool:
    """Make the motors use calibration/<id>.json again. True if they match (already, or after writing).

    Every calibration (ours, another team's, arman/calibrate.py) is stored IN the motors (homing offset + limits).
    After someone else recalibrated, lerobot-* asks "Press ENTER to use provided calibration file" on connect,
    which hangs piped/background runs (agent, stage). Writing our file back restores the exact joint mapping our
    dataset and policy use, as long as the motors were not physically re-mounted (checked via the end stops).
    """
    dev = _arm(label)
    if not dev.calibration:
        sys.exit(f"no calibration/{CONFIG[label]['id']}.json - calibrate the {label} first")
    bus = dev.bus
    bus.connect()
    try:
        motors, ours = bus.read_calibration(), dev.calibration
        diff = [m for m in ours if (ours[m].homing_offset, ours[m].range_min, ours[m].range_max)
                != (motors[m].homing_offset, motors[m].range_min, motors[m].range_max)]
        if not diff:
            print(f"{label}: motors match calibration/{dev.id}.json")
            return True
        # Present = Actual - Homing_Offset, so range + offset is the end stop in raw encoder ticks: the same for
        # any recalibration of the same physical joint (within sweep accuracy), whatever the middle pose was.
        print(f"{label}: motors hold a DIFFERENT calibration (someone else calibrated this arm)")
        print(f"  {'joint':14} {'file off/min/max':>20} {'motors off/min/max':>20} {'end-stop shift':>15}")
        worst = 0
        for m in diff:
            o, n = ours[m], motors[m]
            shift = max(abs((o.range_min + o.homing_offset - n.range_min - n.homing_offset + 2048) % 4096 - 2048),
                        abs((o.range_max + o.homing_offset - n.range_max - n.homing_offset + 2048) % 4096 - 2048))
            if m != "wrist_roll":  # full-turn joint: no end stops to compare
                worst = max(worst, shift)
            print(f"  {m:14} {o.homing_offset:6}/{o.range_min:5}/{o.range_max:5}  "
                  f"{n.homing_offset:6}/{n.range_min:5}/{n.range_max:5}  {shift:12} ticks")
        if worst > 200 and not force:
            print(f"  end stops moved up to {worst} ticks (~{worst * 360 // 4096} deg): a different arm or a re-mounted "
                  f"motor horn? Check the COM port. If it is our arm, re-run with --force, then compare the rest "
                  f"pose with `robot.py positions --follower`.")
            return False
        if not write:
            print(f"  -> fix: robot.py restore-calibration {label}")
            return False
        # EEPROM writes need torque off (disable_torque also unlocks). The arm must rest on the table.
        bus.disable_torque()
        bus.write_calibration(ours)
        # The goal register is in the old frame; set it to where the arm is now so the next torque-on can't jump.
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False), normalize=False)
        ok = bus.is_calibrated
        print(f"{label}: wrote calibration/{dev.id}.json to the motors -> " + ("OK" if ok else "STILL DIFFERENT"))
        return ok
    finally:
        bus.disconnect(disable_torque=False)


def ensure_calibration(labels: tuple[str, ...]) -> None:
    """Before a lerobot-* run: fail with a clear fix instead of lerobot's interactive calibration prompt."""
    if DRY_RUN or os.environ.get("ROBOT_SKIP_CALIB_CHECK") == "1":
        return
    bad = []
    for label in labels:
        try:
            if not sync_calibration(label, write=False):
                bad.append(label)
        except Exception as e:  # port busy / arm unpowered: let lerobot report it the usual way
            print(f"{label}: calibration check skipped ({type(e).__name__}: {e})")
    if bad:
        sys.exit(f"Run: robot.py restore-calibration {' '.join(bad)}   (arm resting on the table)")


def cmd_restore_calibration(a):
    if not a.check:
        # torque goes off only for arms that differ, but a raised follower would drop onto the table
        input("Torque is switched off while writing (the arm goes limp). Put the arm(s) in the REST pose, then ENTER...")
    ok = all([sync_calibration(label, write=not a.check, force=a.force) for label in a.arms or ["follower", "leader"]])
    if ok and not a.check:
        print("Done. Proof it reached EEPROM: power-cycle the arm, then `robot.py restore-calibration --check` "
              "(should say 'match'). Then `robot.py positions --follower` at the rest pose.")
    sys.exit(0 if ok else 1)


def cmd_calibrate(a):
    if a.arm == "follower":
        run("lerobot-calibrate", *robot_args(with_cameras=False))
    else:
        run("lerobot-calibrate", *teleop_args())


def cmd_teleop(a):
    ensure_calibration(("follower", "leader"))
    run(
        "lerobot-teleoperate",
        *robot_args(with_cameras=not a.no_cameras),
        *teleop_args(),
        f"--display_data={str(not a.no_cameras).lower()}",
    )


def cmd_record(a):
    name = a.dataset or CONFIG["hf"]["dataset"]
    root = dataset_root(name)
    if root.exists() and not a.resume:
        sys.exit(
            f"{root} already exists. Use --resume to add episodes "
            f"(--episodes = how many MORE to record), or delete the folder to start over."
        )
    ensure_calibration(("follower", "leader"))
    t = CONFIG["task"]
    run(
        "lerobot-record",
        *robot_args(),
        *teleop_args(),
        f"--dataset.repo_id={hf_repo(name, local_ok=a.no_push)}",
        f"--dataset.root={root}",
        f"--dataset.num_episodes={a.episodes}",
        f"--dataset.single_task={t['description']}",
        f"--dataset.episode_time_s={t['episode_time_s']}",
        f"--dataset.reset_time_s={t['reset_time_s']}",
        f"--dataset.fps={t['fps']}",
        f"--dataset.push_to_hub={str(not a.no_push).lower()}",
        "--dataset.private=true",
        f"--display_data={str(not a.no_display).lower()}",
        f"--resume={str(a.resume).lower()}",
    )


def cmd_replay(a):
    name = a.dataset or CONFIG["hf"]["dataset"]
    ensure_calibration(("follower",))
    run(
        "lerobot-replay",
        *robot_args(with_cameras=False),
        f"--dataset.repo_id={hf_repo(name, local_ok=True)}",
        f"--dataset.root={dataset_root(name)}",
        f"--dataset.episode={a.episode}",
    )


def cmd_train(a):
    tr = CONFIG["train"]
    dataset = a.dataset or CONFIG["hf"]["dataset"]
    if a.type == "smolvla":
        return train_smolvla(a, dataset)
    policy = a.policy or CONFIG["hf"]["policy"]
    out = OUTPUTS_DIR / policy

    if a.resume:
        cfg = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        extra = [f"--steps={a.steps}"] if a.steps else []  # --steps raises the total to train longer
        run("lerobot-train", f"--config_path={cfg}", "--resume=true", *extra)
        return

    if out.exists():
        sys.exit(f"{out} already exists. Use --resume to continue it, or pick another --policy name.")
    run(
        "lerobot-train",
        f"--dataset.repo_id={hf_repo(dataset, local_ok=a.no_push)}",
        f"--dataset.root={dataset_root(dataset)}",
        "--policy.type=act",
        "--policy.device=cuda",
        f"--policy.use_amp={str(tr['use_amp']).lower()}",
        f"--policy.repo_id={hf_repo(policy, local_ok=a.no_push)}",
        f"--policy.push_to_hub={str(not a.no_push).lower()}",
        "--policy.private=true",
        f"--steps={a.steps or tr['steps']}",
        f"--batch_size={a.batch_size or tr['batch_size']}",
        f"--save_freq={tr['save_freq']}",
        f"--num_workers={tr['num_workers']}",
        "--log_freq=100",
        f"--output_dir={out}",
        f"--job_name={policy}",
        f"--wandb.enable={str(a.wandb).lower()}",
    )


def train_smolvla(a, dataset: str):
    """Fine-tune SmolVLA from [train.smolvla] base on the same recordings ACT uses. --cloud runs it on HF Jobs."""
    sv = CONFIG["train"].get("smolvla", {})
    policy = a.policy or f"smolvla_{dataset}"
    out = OUTPUTS_DIR / policy
    if a.resume:
        cfg = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        run("lerobot-train", f"--config_path={cfg}", "--resume=true", *([f"--steps={a.steps}"] if a.steps else []))
        return
    if out.exists() and not a.cloud:
        sys.exit(f"{out} already exists. Use --resume to continue it, or pick another --policy name.")
    if a.cloud:  # the job runs on a Hugging Face GPU: dataset in, model out, both via the Hub
        where = [f"--dataset.repo_id={hf_repo(dataset)}", f"--job.target={a.cloud}",
                 f"--job.timeout={sv.get('cloud_timeout', '6h')}", "--policy.push_to_hub=true"]
        batch = a.batch_size or sv.get("cloud_batch_size", 64)
    else:
        where = [f"--dataset.repo_id={hf_repo(dataset, local_ok=a.no_push)}", f"--dataset.root={dataset_root(dataset)}",
                 f"--policy.push_to_hub={str(not a.no_push).lower()}", f"--num_workers={CONFIG['train']['num_workers']}"]
        batch = a.batch_size or sv.get("batch_size", 4)
    base = sv.get("base", "lerobot/smolvla_base")
    if not a.cloud and not Path(base).exists():
        # Windows: lerobot turns "lerobot/smolvla_base" into "lerobot\smolvla_base" and fails, so pass a local copy
        from huggingface_hub import snapshot_download

        base = snapshot_download(base)
    run(
        "lerobot-train",
        f"--policy.path={base}",
        f"--rename_map={smolvla_rename_map()}",
        *where,
        "--policy.device=cuda",
        f"--policy.repo_id={hf_repo(policy, local_ok=a.no_push and not a.cloud)}",
        "--policy.private=true",
        f"--steps={a.steps or sv.get('steps', 20000)}",
        f"--batch_size={batch}",
        f"--save_freq={CONFIG['train']['save_freq']}",
        "--log_freq=100",
        f"--output_dir={out}",
        f"--job_name={policy}",
        f"--wandb.enable={str(a.wandb).lower()}",
    )


def cmd_rollout(a):
    policy = a.policy or CONFIG["hf"]["policy"]
    local = local_policy_dir(policy)
    if Path(policy).exists():
        path = str(Path(policy).resolve())
    elif local.exists():
        path = str(local)
    else:
        path = hf_repo(policy)  # download from the Hub
    print(f"policy: {path}")
    extra = []
    pc = policy_config(path)
    if pc.get("type") in ("smolvla", "pi0", "pi05"):
        # ~1 s per chunk on the GTX 1650: real-time chunking computes the next chunk while this one runs
        extra.append("--inference.type=rtc")
        if any(k.startswith("observation.images.camera") for k in pc.get("input_features", {})):
            extra.append(f"--rename_map={smolvla_rename_map()}")
    ensure_calibration(("follower",))
    run(
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={path}",
        *robot_args(),
        *extra,
        f"--task={a.task or CONFIG['task']['description']}",
        f"--duration={a.duration}",
        f"--display_data={str(a.display).lower()}",
        *([f"--robot.max_relative_target={float(a.max_step)}"] if a.max_step else []),
    )


# ---------------------------------------------------------------- CLI

def main():
    global DRY_RUN
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the lerobot command instead of running it")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="show python/torch/GPU, serial ports and config").set_defaults(fn=cmd_check)
    sub.add_parser("find-port", help="identify which COM port belongs to which arm").set_defaults(fn=cmd_find_port)
    sub.add_parser("find-cameras", help="list OpenCV camera indexes and save test frames").set_defaults(fn=cmd_find_cameras)

    s = sub.add_parser("setup-motors", help="(new motors only) assign motor ids one by one")
    s.add_argument("arm", choices=["follower", "leader"])
    s.set_defaults(fn=cmd_setup_motors)

    s = sub.add_parser("calibrate", help="calibrate an arm (once per arm, per laptop)")
    s.add_argument("arm", choices=["follower", "leader"])
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("restore-calibration",
                       help="write calibration/*.json back into the motors (after another team calibrated the arm)")
    s.add_argument("arms", nargs="*", choices=["follower", "leader"], help="default: both")
    s.add_argument("--check", action="store_true", help="only compare, write nothing")
    s.add_argument("--force", action="store_true", help="write even if the end stops moved a lot")
    s.set_defaults(fn=cmd_restore_calibration)

    s = sub.add_parser("positions", help="print joint values in LeRobot units (check calibration / directions)")
    s.add_argument("--follower", action="store_true", help="also read the follower (it keeps holding its pose)")
    s.add_argument("--watch", action="store_true", help="keep printing 4x per second until Ctrl-C")
    s.set_defaults(fn=cmd_positions)

    s = sub.add_parser("teleop", help="drive the follower with the leader arm")
    s.add_argument("--no-cameras", action="store_true", help="skip cameras and the rerun viewer")
    s.set_defaults(fn=cmd_teleop)

    s = sub.add_parser("record", help="record teleop demonstrations into a LeRobot dataset")
    s.add_argument("--episodes", type=int, default=10, help="episodes to record in this session")
    s.add_argument("--dataset", help="dataset name (default: [hf] dataset)")
    s.add_argument("--resume", action="store_true", help="append episodes to an existing dataset")
    s.add_argument("--no-push", action="store_true", help="don't upload to the Hub at the end")
    s.add_argument("--no-display", action="store_true", help="don't open the rerun viewer")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("replay", help="replay a recorded episode on the follower (stage fallback)")
    s.add_argument("--episode", type=int, default=0)
    s.add_argument("--dataset")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("train", help="train an ACT (local GPU) or SmolVLA (cloud or local) policy")
    s.add_argument("--type", choices=["act", "smolvla"], default="act",
                   help="act: small, one task, trains on the GTX 1650. smolvla: fine-tunes lerobot/smolvla_base, "
                        "follows the task text ([train.smolvla])")
    s.add_argument("--cloud", metavar="FLAVOR", nargs="?", const="__config__",
                   help="(smolvla) train on a Hugging Face GPU via HF Jobs, e.g. a100-large (paid, needs `hf auth login`); "
                        "no value = [train.smolvla] cloud")
    s.add_argument("--dataset")
    s.add_argument("--policy", help="output/policy name (default: [hf] policy)")
    s.add_argument("--steps", type=int)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    s.add_argument("--no-push", action="store_true", help="don't upload the trained policy to the Hub")
    s.add_argument("--wandb", action="store_true", help="log to Weights & Biases (run `wandb login` first)")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("rollout", help="run a trained policy on the robot")
    s.add_argument("--policy", help="policy name, local checkpoint folder, or Hub repo id")
    s.add_argument("--duration", type=int, default=60, help="seconds to run")
    s.add_argument("--task", help="task text for the policy (default: [task] description). SmolVLA follows it")
    s.add_argument("--display", action="store_true", help="show cameras in rerun (costs some fps)")
    s.add_argument("--max-step", type=float, metavar="N",
                   help="safety: cap each joint's move per 30 Hz step to N position units (-100..100 scale); "
                        "2 = ~60 units/s, good for a first supervised rollout")
    s.set_defaults(fn=cmd_rollout)

    a = p.parse_args()
    DRY_RUN = a.dry_run
    if getattr(a, "cloud", None) == "__config__":
        a.cloud = CONFIG["train"].get("smolvla", {}).get("cloud", "a100-large")
    a.fn(a)


if __name__ == "__main__":
    main()
