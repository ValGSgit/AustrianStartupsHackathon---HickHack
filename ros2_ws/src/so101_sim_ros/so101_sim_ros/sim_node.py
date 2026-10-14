"""MuJoCo SO-101 simulation as a ROS 2 node.

Publishes
  /joint_states                          sensor_msgs/JointState      radians (gripper joint in rad too), 30 Hz
  /so101/camera/front/image_raw          sensor_msgs/Image (rgb8)    640x480, `camera_hz`
  /so101/camera/front/image_raw/compressed  sensor_msgs/CompressedImage (jpeg), for Foxglove
  /so101/object_pose                     geometry_msgs/PoseStamped
  /so101/lifted                          std_msgs/Bool               object > 4 cm above the table
Subscribes
  /so101/joint_commands                  sensor_msgs/JointState      position targets in radians, by joint name
Services
  /so101/reset                           std_srvs/Trigger            new random object pose, arm to home

Parameter `autoplay:=true` runs the scripted pick expert in a loop (good for checking the setup).
"""

import threading
import time

import cv2
import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from so101_sim.env import CAM_H, CAM_W, FPS, HOME, JOINTS, SO101PickEnv, lerobot_to_rad, rad_to_lerobot
from so101_sim.expert import plan_pick


class SimNode(Node):
    def __init__(self):
        super().__init__("so101_sim")
        self.declare_parameter("obj", "pen")
        self.declare_parameter("camera_hz", 15.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("autoplay", False)
        obj = self.get_parameter("obj").value
        self.seed = int(self.get_parameter("seed").value)
        self.autoplay = bool(self.get_parameter("autoplay").value)

        self.env = SO101PickEnv(obj, render=False)  # physics only; the camera thread renders
        self.env.reset(seed=self.seed)
        self.lock = threading.Lock()
        self.snapshot = self.env.data.qpos.copy()  # rendering only needs positions
        self.snapshot_fresh = False
        self.target = HOME.copy()  # LeRobot units
        self.script: list[np.ndarray] = []

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_img = self.create_publisher(Image, "/so101/camera/front/image_raw", 2)
        self.pub_jpg = self.create_publisher(CompressedImage, "/so101/camera/front/image_raw/compressed", 2)
        self.pub_obj = self.create_publisher(PoseStamped, "/so101/object_pose", 10)
        self.pub_lift = self.create_publisher(Bool, "/so101/lifted", 10)
        self.create_subscription(JointState, "/so101/joint_commands", self.on_command, 10)
        self.create_service(Trigger, "/so101/reset", self.on_reset)

        self.camera_period = 1.0 / float(self.get_parameter("camera_hz").value)
        self.running = True
        # Rendering runs in its own thread with its own GL context, so slow CPU rendering in Docker
        # never stalls the 30 Hz physics/joint_states loop.
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()
        self.create_timer(1.0 / FPS, self.step)
        self.get_logger().info(f"SO-101 sim ready: obj={obj}, 30 Hz physics, camera up to "
                               f"{1 / self.camera_period:.0f} Hz, autoplay={self.autoplay}")

    def on_command(self, msg: JointState) -> None:
        q = lerobot_to_rad(self.target)  # current target in rad, then overwrite named joints
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                q[JOINTS.index(name)] = pos
        self.target = rad_to_lerobot(q)

    def on_reset(self, _req, res):
        self.seed += 1
        self.env.reset(seed=self.seed)
        self.target = HOME.copy()
        self.script = []
        res.success, res.message = True, f"reset with seed {self.seed}"
        return res

    def step(self) -> None:
        if self.autoplay:
            if not self.script:
                self.seed += 1
                self.env.reset(seed=self.seed)
                self.script = plan_pick(self.env, np.random.default_rng(self.seed))
            self.target = self.script.pop(0)

        self.env.step(self.target, render=False)
        with self.lock:
            self.snapshot[:] = self.env.data.qpos
            self.snapshot_fresh = True
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = JOINTS
        js.position = [float(x) for x in self.env.data.qpos[:6]]
        js.velocity = [float(x) for x in self.env.data.qvel[:6]]
        self.pub_js.publish(js)

        pos, yaw = self.env.object_pose()
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = now, "world"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in pos)
        ps.pose.orientation.w, ps.pose.orientation.z = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        self.pub_obj.publish(ps)
        self.pub_lift.publish(Bool(data=bool(self.env.is_success())))

    def camera_loop(self) -> None:
        renderer = mujoco.Renderer(self.env.model, CAM_H, CAM_W)  # GL context belongs to this thread
        scene_data = mujoco.MjData(self.env.model)
        while self.running:
            t0 = time.perf_counter()
            with self.lock:
                fresh = self.snapshot_fresh
                if fresh:
                    scene_data.qpos[:] = self.snapshot
                    self.snapshot_fresh = False
            if fresh:
                mujoco.mj_forward(self.env.model, scene_data)  # also places cameras (mj_kinematics does not)
                renderer.update_scene(scene_data, camera="front")
                self.publish_image(renderer.render())
            time.sleep(max(0.0, self.camera_period - (time.perf_counter() - t0)))
        renderer.close()

    def publish_image(self, rgb: np.ndarray) -> None:
        now = self.get_clock().now().to_msg()
        img = Image()
        img.header.stamp, img.header.frame_id = now, "front_camera"
        img.height, img.width = rgb.shape[:2]
        img.encoding, img.step = "rgb8", rgb.shape[1] * 3
        img.data = rgb.tobytes()  # no cv_bridge: it pins numpy 1.x and clashes with LeRobot
        self.pub_img.publish(img)
        ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            c = CompressedImage()
            c.header = img.header
            c.format, c.data = "jpeg", jpg.tobytes()
            self.pub_jpg.publish(c)


def main() -> None:
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.cam_thread.join(timeout=2)
        node.destroy_node()
        rclpy.try_shutdown()
