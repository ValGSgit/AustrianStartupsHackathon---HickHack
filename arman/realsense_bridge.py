"""Makes the RealSense D435i's COLOUR camera usable on the Mac.

macOS only shows the RealSense's grey infrared image to normal apps. Its colour camera needs Intel's own
library, and on macOS 12+ that library needs admin rights (sudo) to reach the USB device. This small program
runs with sudo, reads the colour image and passes it on (only inside this Mac, port 5555) to do_task.py /
record.py, which run normally. Set FRONT_CAM=realsense or SIDE_CAM=realsense in settings.env.

    sudo .venv/bin/python realsense_bridge.py           # leave it running in its own Terminal tab
    sudo .venv/bin/python realsense_bridge.py --test    # grab one picture: outputs/realsense_test.jpg

Plug the RealSense into a USB-C port directly (USB 3), not through a hub. Ctrl+C stops it.
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

PORT = 5555
NAME = "realsense"
WIDTH, HEIGHT, FPS = 640, 480, 30


def start():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit("No RealSense found. Plugged in directly (USB 3)? Started with sudo?")
    dev = devices[0]
    print(f"Found {dev.get_info(rs.camera_info.name)}, firmware {dev.get_info(rs.camera_info.firmware_version)}, "
          f"USB {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    pipe.start(cfg)
    for _ in range(30):  # let auto exposure settle
        pipe.wait_for_frames(5000)
    return pipe


def main() -> None:
    pipe = start()
    if "--test" in sys.argv:
        frame = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
        out = Path(__file__).parent / "outputs/realsense_test.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {out} ({frame.shape[1]}x{frame.shape[0]}, colour)")
        pipe.stop()
        return
    sock = zmq.Context().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{PORT}")
    print(f"Sending the colour image on 127.0.0.1:{PORT}. Leave this running; Ctrl+C stops it.")
    sent, t0 = 0, time.time()
    try:
        while True:
            frame = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
            # Encoded as-is (RGB), exactly like LeRobot's own image server: its camera reader decodes back to RGB.
            jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
            msg = {"timestamps": {NAME: time.time()}, "images": {NAME: base64.b64encode(jpg).decode()}}
            sock.send_string(json.dumps(msg), zmq.NOBLOCK)
            sent += 1
            if time.time() - t0 > 10:
                print(f"  {sent / (time.time() - t0):.0f} pictures/s", flush=True)
                sent, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
