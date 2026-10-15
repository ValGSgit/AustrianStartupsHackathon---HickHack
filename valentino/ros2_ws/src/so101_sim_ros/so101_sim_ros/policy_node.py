"""Runs a trained LeRobot policy (e.g. ACT) against ROS topics: camera + joint states in, joint commands out.

Same inference path as `lerobot-rollout` (LeRobot's predict_action + the policy's saved processors).
Here it drives the sim; the identical topic interface could drive a real arm through a ROS bridge.

    ros2 run so101_sim_ros policy_node --ros-args -p policy_path:=/ws/outputs/train/act_sim_pick_pen/checkpoints/last/pretrained_model
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

from so101_sim.env import FPS, JOINTS, lerobot_to_rad, rad_to_lerobot


class PolicyNode(Node):
    def __init__(self):
        super().__init__("so101_policy")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("task", "Pick up the pen")
        self.declare_parameter("device", "cpu")
        path = self.get_parameter("policy_path").value
        if not path:
            raise SystemExit("policy_path parameter is required")
        self.task = self.get_parameter("task").value

        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.predict_action = predict_action
        self.device = torch.device(self.get_parameter("device").value)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = self.device.type
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )
        self.get_logger().info(f"loaded {cfg.type} policy from {path} on {self.device}")

        self.state = None
        self.image = None
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Image, "/so101/camera/front/image_raw", self.on_image, 2)
        self.pub = self.create_publisher(JointState, "/so101/joint_commands", 10)
        self.create_service(Trigger, "/so101/policy_reset", self.on_reset)
        self.create_timer(1.0 / FPS, self.step)

    def on_joints(self, msg: JointState) -> None:
        q = np.zeros(6)
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                q[JOINTS.index(name)] = pos
        self.state = rad_to_lerobot(q)

    def on_image(self, msg: Image) -> None:
        self.image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def on_reset(self, _req, res):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        res.success, res.message = True, "policy state cleared"
        return res

    def step(self) -> None:
        if self.state is None or self.image is None:
            return
        action = self.predict_action(
            {"observation.state": self.state, "observation.images.front": self.image},
            self.policy, self.device, self.pre, self.post, use_amp=False, task=self.task,
            robot_type="so101_follower_sim",
        )
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = [float(x) for x in lerobot_to_rad(action.cpu().numpy().reshape(-1))]  # drop batch dim
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
