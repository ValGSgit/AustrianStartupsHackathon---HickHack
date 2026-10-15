"""Offline end-to-end test of do_task.run + assist.py: a scripted fake model server on localhost and a simulated
robot whose gripper stops on an object. No GPU, no hardware.

    cd tests && ../.venv/bin/python offline_run_test.py <scenario>   (writes only inside tests/)
      retry_then_carry   1st try: no object -> missed -> reopen; 2nd try: object -> holding -> carry -> success
      old_behaviour      AUTO_DROP=0, RETRIES=0: the model runs until the time is up, no take-over
      no_drop_pose       AUTO_DROP=1 but no taught drop pose: model only, retries still work
"""

import json
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

PROJECT = Path("/Users/arman/Documents/workplace/so101-hackathon")
HERE = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

import grpc  # noqa: E402
import torch  # noqa: E402
from lerobot.async_inference import policy_server as _ps  # noqa: E402
from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402

import assist  # noqa: E402
import config  # noqa: E402
import do_task  # noqa: E402
import poses  # noqa: E402

SCENARIO = sys.argv[1]
PORT = 8097
_ps.observations_similar = lambda *a, **k: False


class FakeModelServer(_ps.PolicyServer):
    """Real LeRobot server plumbing, scripted 'model': reach, open the gripper, close it."""

    sessions = 0

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        import pickle

        specs = pickle.loads(request.data)  # nosec
        self.device, self.policy_type = specs.device, specs.policy_type
        self.lerobot_features, self.actions_per_chunk = specs.lerobot_features, specs.actions_per_chunk
        FakeModelServer.sessions += 1
        self.session_start = time.time()
        return services_pb2.Empty()

    def _predict_action_chunk(self, observation_t):
        obs = observation_t.get_observation()
        self.last_processed_obs = observation_t
        tau0 = time.time() - self.session_start
        chunk = []
        for k in range(self.actions_per_chunk):
            tau = tau0 + k / 30
            reach = min(1.0, tau / 1.5)
            joints = [10 * reach, -94 + 90 * reach, 97 - 100 * reach, 69 - 40 * reach, 0.0]
            if tau < 0.5:
                g = 2.0
            elif tau < 1.5:
                g = 2 + 38 * (tau - 0.5)
            elif tau < 2.0:
                g = 40.0
            elif tau < 2.6:
                g = 40 - 35 * (tau - 2.0) / 0.6
            else:
                g = 5.0
            chunk.append(torch.tensor(joints + [g]))
        return self._time_action_chunk(observation_t.get_timestamp(), chunk, observation_t.get_timestep())


class SimRobot(do_task.FakeRobot):
    """Arm joints go where they are told; the gripper follows with a lag and stops on the object (22 wide).
    No object during the first opening of the gripper, an object from the second opening on."""

    WIDTH = 22.0

    def __init__(self, cams):
        super().__init__(cams)
        self.grip = self.state["gripper.pos"]
        self.openings, self.was_closed = 0, True

    @property
    def has_object(self):
        return self.openings >= 2

    def send_action(self, action):
        self.state.update(action)
        self.actions += 1
        cmd = action["gripper.pos"]
        if cmd > 30 and self.was_closed:
            self.openings, self.was_closed = self.openings + 1, False
        if cmd < 10:
            self.was_closed = True
        target = max(cmd, self.WIDTH) if (self.has_object and cmd < self.WIDTH and self.grip >= self.WIDTH - 1) else cmd
        self.grip += (target - self.grip) * 0.35
        return action

    def measured(self):
        return {**self.state, "gripper.pos": self.grip}

    def get_observation(self):
        obs = super().get_observation()
        obs["gripper.pos"] = self.grip
        return obs


def main():
    # Isolated files: never touch the real poses.json, grip.json or outputs/.
    taught = json.loads((PROJECT / "poses.json").read_text())
    if SCENARIO != "no_drop_pose":
        taught["above_cup"] = {**taught["home"], "shoulder_pan.pos": 40.0, "shoulder_lift.pos": -40.0, "elbow_flex.pos": 40.0}
    (HERE / "poses.json").write_text(json.dumps(taught))
    poses.POSES_FILE = HERE / "poses.json"
    assist.HERE, assist.GRIP_FILE = HERE, HERE / "grip.json"
    if SCENARIO == "old_behaviour":
        assist.AUTO_DROP, assist.RETRIES = False, 0
    do_task.PORT = PORT
    do_task.FakeRobot = SimRobot

    server = FakeModelServer(PolicyServerConfig(host="127.0.0.1", port=PORT))
    g = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, g)
    g.add_insecure_port(f"127.0.0.1:{PORT}")
    g.start()

    seconds = 12 if SCENARIO == "old_behaviour" else 25
    t0 = time.time()
    do_task.run("move the arm towards the paper ball, grasp it, lift it up, and drop it into the cup", seconds,
                {"cam0": (0, 640, 360), "cam1": (1, 640, 480)}, True, None)
    print(f"\nTEST {SCENARIO}: finished in {time.time() - t0:.1f} s, model sessions: {FakeModelServer.sessions}")
    runs = sorted((HERE / "outputs/runs").iterdir())
    print((runs[-1] / "events.txt").read_text())
    print("files:", sorted(p.name for p in runs[-1].iterdir()))
    g.stop(0)


if __name__ == "__main__":
    main()
    threading.Timer(1, lambda: None).start()
