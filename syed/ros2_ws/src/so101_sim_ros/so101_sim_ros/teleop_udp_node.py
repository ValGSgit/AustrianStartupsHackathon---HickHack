"""Leader arm -> sim bridge. Receives LeRobot-style actions over UDP and republishes them as ROS commands.

USB devices can't easily reach a Docker container on Windows, so the leader arm is read on the
host by `host/leader_to_sim.py`, which sends one JSON datagram per tick:
    {"shoulder_pan.pos": 12.3, ..., "wrist_roll.pos": -40.0, "gripper.pos": 35.0}
(real-robot LeRobot units: arm -100..100, gripper 0..100). This node converts to radians and publishes
/so101/joint_commands.
"""

import json
import socket

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from so101_sim.env import JOINTS, MOTOR_NAMES, lerobot_to_rad


class TeleopUdpNode(Node):
    def __init__(self):
        super().__init__("so101_teleop_udp")
        self.declare_parameter("port", 5005)
        port = int(self.get_parameter("port").value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.pub = self.create_publisher(JointState, "/so101/joint_commands", 10)
        self.received = 0
        self.create_timer(1.0 / 100, self.poll)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"waiting for leader-arm packets on udp/{port}")

    def poll(self) -> None:
        latest = None
        while True:  # drain the socket, keep only the newest packet
            try:
                latest, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
        if latest is None:
            return
        try:
            action = json.loads(latest)
            values = [float(action[k]) for k in MOTOR_NAMES]
        except (ValueError, KeyError) as e:
            self.get_logger().warning(f"bad packet: {e}")
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = [float(x) for x in lerobot_to_rad(values)]
        self.pub.publish(msg)
        self.received += 1

    def report(self) -> None:
        self.get_logger().info(f"{self.received} leader packets in the last 5 s")
        self.received = 0


def main() -> None:
    rclpy.init()
    node = TeleopUdpNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
