"""Save one photo from each camera (0 to 4) to outputs/cam_test/. Run in the Terminal app."""
import time
from pathlib import Path

import cv2

out = Path(__file__).parent / "outputs/cam_test"
out.mkdir(parents=True, exist_ok=True)
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    # Ask for what LeRobot will ask for (settings in scripts/_common.sh), then report what we really get.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(idx, "opened:", cap.isOpened(), "fps:", cap.get(cv2.CAP_PROP_FPS), flush=True)
    frame, t = None, time.time()
    while cap.isOpened() and time.time() - t < 3:
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        print(idx, "no image", flush=True)
        continue
    cv2.imwrite(str(out / f"camera_{idx}.jpg"), frame)
    print(idx, "saved", frame.shape[1], "x", frame.shape[0], "brightness", round(float(frame.mean())), flush=True)
