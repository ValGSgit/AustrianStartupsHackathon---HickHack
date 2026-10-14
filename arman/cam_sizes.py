"""Which picture sizes does camera N really give? Run in the Terminal app:  python cam_sizes.py 1"""
import sys

import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for w, h in [(640, 480), (640, 360), (1280, 720), (1920, 1080)]:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, f = False, None
    for _ in range(15):
        ok, f = cap.read()
    got = f"{f.shape[1]}x{f.shape[0]}" if ok else "no image"
    print(f"asked {w}x{h} -> got {got} at {cap.get(cv2.CAP_PROP_FPS):.0f} fps", flush=True)
    if ok:
        cv2.imwrite(f"outputs/cam_test/phone_{w}x{h}.jpg", f)
    cap.release()
