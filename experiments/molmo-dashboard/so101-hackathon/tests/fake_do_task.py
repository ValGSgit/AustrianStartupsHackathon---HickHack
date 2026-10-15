"""do_task.py with a scripted fake model server and a simulated arm + cameras: the whole dashboard flow (agent ->
do_task -> live view -> RESULT) without the Lightning GPU, the robot or the cameras.

    ROBOT_LIVE=1 ../.venv/bin/python fake_do_task.py --seconds 25 "put the duck and the chess queen into the cup"

Same arguments as do_task.py (--fake is added). Writes only inside tests/ (poses.json, grip.json, outputs/runs).
The simulated gripper misses on its first try and holds the object from the second (offline_run_test.SimRobot).
The camera pictures are drawn: a table with a duck, a queen and a cup, and the gripper where the arm points.
"""

import json
import sys
import threading
from concurrent import futures
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import grpc  # noqa: E402
import numpy as np  # noqa: E402
import offline_run_test as sim  # noqa: E402 - fake model server + simulated gripper
from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.transport import services_pb2_grpc  # noqa: E402

do_task, assist, poses = sim.do_task, sim.assist, sim.poses
placed: list[str] = []  # objects already in the cup (one per finished object run)


class DrawnRobot(sim.SimRobot):
    def get_observation(self):
        obs = super().get_observation()
        pan, lift = self.state["shoulder_pan.pos"], self.state["shoulder_lift.pos"]
        for slot, (_, w, h) in self.cams.items():
            img = np.full((h, w, 3), (200, 185, 160), np.uint8)  # table
            cv2.circle(img, (int(w * .78), int(h * .55)), int(h * .12), (240, 240, 245), -1)  # cup
            cv2.putText(img, "cup", (int(w * .74), int(h * .75)), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 60, 60), 2)
            for name, (x, y), col in (("duck", (.25, .6), (250, 210, 40)), ("queen", (.45, .7), (30, 30, 30))):
                if name not in placed:
                    cv2.circle(img, (int(w * x), int(h * y)), int(h * .06), col, -1)
                    cv2.putText(img, name, (int(w * x) - 20, int(h * y) + 35), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 60, 60), 1)
            gx, gy = int(w * (.5 + pan / 200)), int(h * (.35 - lift / 400))
            cv2.circle(img, (gx, gy), 9, (200, 40, 40), 3)  # gripper
            if slot != "cam0":
                img = cv2.flip(img, 1)
            obs[slot] = img
        return obs


def main() -> None:
    taught = json.loads((sim.PROJECT / "poses.json").read_text())
    taught["above_cup"] = {**taught["home"], "shoulder_pan.pos": 40.0, "shoulder_lift.pos": -40.0, "elbow_flex.pos": 40.0}
    (HERE / "poses.json").write_text(json.dumps(taught))
    poses.POSES_FILE = HERE / "poses.json"
    assist.HERE, assist.GRIP_FILE = HERE, HERE / "grip.json"
    do_task.PORT = sim.PORT
    do_task.FakeRobot = DrawnRobot
    do_task.check_studio = lambda: print("\n1. Lightning Studio: FAKE model server on this Mac (test)", flush=True)
    run = do_task.run

    def run_and_place(task, *a):
        result = run(task, *a)
        if result.startswith("success"):
            placed.append(do_task.object_name(task).split()[-1])
        return result

    do_task.run = run_and_place

    server = sim.FakeModelServer(PolicyServerConfig(host="127.0.0.1", port=sim.PORT))
    g = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, g)
    g.add_insecure_port(f"127.0.0.1:{sim.PORT}")
    g.start()
    sys.argv = ["do_task.py", *sys.argv[1:], "--fake"]
    try:
        do_task.main()
    finally:
        g.stop(0)


if __name__ == "__main__":
    main()
    threading.Timer(1, lambda: None).start()
