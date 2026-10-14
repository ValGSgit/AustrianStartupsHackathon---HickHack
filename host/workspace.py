"""Where things are on the table, in robot coordinates, from ONE camera image - whatever the camera angle.

Objects lie on a printed A4 sheet ("the mat") with an ArUco marker in each corner. Every look() finds the markers
and fits camera pixel -> table plane, so the camera can be moved or bumped between commands. Where the sheet lies
relative to the robot is measured once (`python host/skills.py teach-sheet`: put the gripper tip on two markers)
and stored in calibration/workspace.json.

Robot frame (same as the MuJoCo model): origin under the shoulder_pan axis on the table, x forward, y to the robot's
left, z up, metres. Sheet frame: same axes, origin at the sheet centre; on the printout the robot sits at the
BOTTOM edge (image up = +x, image left = +y).

    python host/workspace.py sheet           # -> outputs/workspace_sheet.png, print at 100 % on A4 landscape
    python host/workspace.py look [image]    # detect markers + objects in a photo (or camera index from config)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHEET_JSON = ROOT / "calibration" / "workspace.json"

# A4 landscape. Long side (297 mm) runs along robot y, short side (210 mm) along robot x.
SHEET_X, SHEET_Y = 0.210, 0.297
MARKER = 0.040  # printed marker side (m), black border included
MARGIN = 0.008  # paper edge -> marker
DICT = cv2.aruco.DICT_4X4_50
# Marker centres in the sheet frame: id -> (x, y). 0 = far left, 1 = far right, 2 = near right, 3 = near left.
_cx = SHEET_X / 2 - MARGIN - MARKER / 2
_cy = SHEET_Y / 2 - MARGIN - MARKER / 2
MARKERS = {0: (_cx, _cy), 1: (_cx, -_cy), 2: (-_cx, -_cy), 3: (-_cx, _cy)}
# Lettered target boxes printed on the sheet (sheet frame centre x, y; side BOX_SIDE). Far row A B C, near row
# D X Z, each left to right as the robot sees it. A 4.5 cm band across the middle stays free for objects.
BOX_SIDE = 0.045
BOXES = {"A": (0.045, 0.065), "B": (0.045, 0.0), "C": (0.045, -0.065),
         "D": (-0.045, 0.065), "X": (-0.045, 0.0), "Z": (-0.045, -0.065)}
BOX_GREY = 185  # light enough that the object detector (< 72 % of paper brightness) ignores the print
# Default placement: sheet centre 20 cm in front of the shoulder, square to the robot (near edge at 9.5 cm).
DEFAULT_POSE = {"x": 0.20, "y": 0.0, "yaw": 0.0}
MM = 1e-3


def marker_corners_sheet(mid: int) -> np.ndarray:
    """Corners of marker `mid` in the sheet frame, in OpenCV's order (TL, TR, BR, BL of the printed image)."""
    x, y = MARKERS[mid]
    h = MARKER / 2
    # printed image: up = +x, left = +y  ->  TL = (+x, +y), TR = (+x, -y), BR = (-x, -y), BL = (-x, +y)
    return np.array([(x + h, y + h), (x + h, y - h), (x - h, y - h), (x - h, y + h)], np.float64)


def sheet_image(px_per_m: float = 300 / 0.0254) -> np.ndarray:
    """The printable sheet (white, 4 markers, a few cm ticks), at 300 dpi by default."""
    W, H = round(SHEET_Y * px_per_m), round(SHEET_X * px_per_m)
    img = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    side = round(MARKER * px_per_m)
    for mid in MARKERS:
        tl = marker_corners_sheet(mid)[0]
        col, row = round(W / 2 - tl[1] * px_per_m), round(H / 2 - tl[0] * px_per_m)
        img[row:row + side, col:col + side] = cv2.aruco.generateImageMarker(d, mid, side)
        cv2.putText(img, str(mid), (col + side + 10, row + side // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    # light-grey ticks every 5 cm (centre cross) help placing objects; too faint to look like objects
    for k in range(-2, 3):
        c, r = round(W / 2 + k * 0.05 * px_per_m), round(H / 2 + k * 0.05 * px_per_m)
        cv2.line(img, (c, H // 2 - 15), (c, H // 2 + 15), 225, 2)
        if abs(k) < 2:
            cv2.line(img, (W // 2 - 15, r), (W // 2 + 15, r), 225, 2)
    for letter, (bx, by) in BOXES.items():
        h = BOX_SIDE / 2
        c0, r0 = round(W / 2 - (by + h) * px_per_m), round(H / 2 - (bx + h) * px_per_m)
        c1, r1 = round(W / 2 - (by - h) * px_per_m), round(H / 2 - (bx - h) * px_per_m)
        t = max(2, round(0.0008 * px_per_m))
        cv2.rectangle(img, (c0, r0), (c1, r1), BOX_GREY, t)
        scale = BOX_SIDE * px_per_m / 60
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 * t)
        cv2.putText(img, letter, ((c0 + c1 - tw) // 2, (r0 + r1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, BOX_GREY, 2 * t)
    cv2.putText(img, "ROBOT THIS SIDE  -  print at 100% (marker squares = 40 mm)", (round(W * 0.28), H - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 160, 2)
    return img


# ---------------------------------------------------------------- placement of the sheet relative to the robot

def load_pose() -> dict:
    try:
        return {**DEFAULT_POSE, **json.loads(SHEET_JSON.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULT_POSE)


def sheet_to_robot(pts: np.ndarray, pose: dict) -> np.ndarray:
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.stack([pose["x"] + c * pts[:, 0] - s * pts[:, 1], pose["y"] + s * pts[:, 0] + c * pts[:, 1]], 1)


def box_polygon(letter: str, pose: dict | None = None) -> np.ndarray:
    """Corners (4, 2) of a printed box in the robot frame."""
    bx, by = BOXES[letter]
    h = BOX_SIDE / 2
    return sheet_to_robot([(bx + h, by + h), (bx + h, by - h), (bx - h, by - h), (bx - h, by + h)], pose or load_pose())


def box_centre(letter: str, pose: dict | None = None) -> tuple[float, float]:
    (x, y), = sheet_to_robot([BOXES[letter]], pose or load_pose())
    return float(x), float(y)


def pose_from_touches(touched: dict[int, tuple[float, float]]) -> dict:
    """Sheet pose from the robot-frame positions of >= 2 marker centres (rigid 2D fit)."""
    ids = sorted(touched)
    if len(ids) < 2:
        raise ValueError("touch at least 2 markers")
    A = np.array([MARKERS[i] for i in ids])
    B = np.array([touched[i] for i in ids])
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    yaw = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    t = cb - np.array([[c, -s], [s, c]]) @ ca
    pose = {"x": float(t[0]), "y": float(t[1]), "yaw": float(yaw)}
    resid = np.linalg.norm(sheet_to_robot(A, pose) - B, axis=1)
    pose["touch_error_mm"] = round(float(resid.max() * 1000), 1)
    return pose


# ---------------------------------------------------------------- perception

@dataclass
class Obj:
    id: int
    x: float  # robot frame, m (centre)
    y: float
    yaw: float  # rad, direction of the long side (-pi/2..pi/2); 0 = pointing away from the robot
    length: float  # m, long side
    width: float  # m, short side
    color: str
    shape: str  # elongated | square | round | blob
    px: tuple[int, int]  # centre in the camera image (for drawing / labels)

    def describe(self) -> str:
        return (f"#{self.id}: {self.color} {self.shape} object, {self.length * 100:.0f} x {self.width * 100:.1f} cm, "
                f"at x={self.x * 100:.0f} cm (forward), y={self.y * 100:.0f} cm (left+), "
                f"long side turned {math.degrees(self.yaw):.0f} deg from straight ahead")

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Scene:
    ok: bool
    detail: str
    markers: list[int]
    objects: list[Obj]
    H: np.ndarray | None  # camera pixel -> robot (x, y) on the table plane
    annotated: np.ndarray  # BGR image with markers, sheet outline and numbered objects
    camera: tuple | None = None  # (K, rvec, tvec) robot frame -> image, for drawing 3D paths (cv2.projectPoints)
    image: np.ndarray | None = None  # the raw frame

    def project(self, pts3: np.ndarray) -> np.ndarray | None:
        """Robot-frame 3D points (m) -> image pixels; None without a camera estimate."""
        if self.camera is None:
            return None
        K, rv, tv = self.camera
        px, _ = cv2.projectPoints(np.asarray(pts3, np.float64).reshape(-1, 1, 3), rv, tv, K, np.zeros(5))
        return px.reshape(-1, 2)

    def text(self) -> str:
        if not self.ok:
            return f"(no table view: {self.detail})"
        return "\n".join(o.describe() for o in self.objects) or "no objects on the sheet"


_detector = None


def _aruco():
    global _detector
    if _detector is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
    return _detector


def _color_name(bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]) * 2, int(hsv[1]), int(hsv[2])
    if v < 60:
        return "black"
    if s < 50:
        return "white" if v > 190 else "grey"
    for limit, name in [(15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (255, "blue"),
                        (290, "purple"), (335, "pink"), (360, "red")]:
        if h < limit:
            return name
    return "red"


def camera_model(img_pts: np.ndarray, rob_pts: np.ndarray, size: tuple[int, int]) -> tuple | None:
    """Camera centre in the robot frame (m) from the marker corners: one-view calibration with only the focal
    length free (webcams: centred principal point, square pixels, distortion ignored). Used for parallax."""
    w, h = size
    obj = np.column_stack([rob_pts, np.zeros(len(rob_pts))]).astype(np.float32)
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    try:
        _, K, _, rv, tv = cv2.calibrateCamera([obj], [img_pts.astype(np.float32)], (w, h), K, np.zeros(5), flags=flags)
    except cv2.error:
        return None
    R, _ = cv2.Rodrigues(rv[0])
    c = (-R.T @ tv[0]).reshape(3)
    return (c, (K, rv[0], tv[0])) if c[2] > 0.05 else None


def look(bgr: np.ndarray, pose: dict | None = None, min_area_mm2: float = 40.0) -> Scene:
    """Markers -> homography -> objects on the sheet. bgr = camera frame (OpenCV channel order)."""
    pose = pose or load_pose()
    out = bgr.copy()
    corners, ids, _ = _aruco().detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    found = [] if ids is None else [int(i) for i in ids.reshape(-1) if int(i) in MARKERS]
    if not found:
        cv2.putText(out, "no sheet markers visible", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return Scene(False, "no ArUco markers of the sheet visible - is the sheet in view and lit?", [], [], None, out)
    img_pts, rob_pts = [], []
    for c, i in zip(corners, ids.reshape(-1)):
        if int(i) in MARKERS:
            img_pts.append(c.reshape(4, 2))
            rob_pts.append(sheet_to_robot(marker_corners_sheet(int(i)), pose))
    img_pts, rob_pts = np.concatenate(img_pts), np.concatenate(rob_pts)
    H, _ = cv2.findHomography(img_pts, rob_pts)
    if H is None:
        return Scene(False, "homography failed", found, [], None, out)
    cv2.aruco.drawDetectedMarkers(out, corners, ids)
    model = camera_model(img_pts, rob_pts, (bgr.shape[1], bgr.shape[0])) if len(found) >= 2 else None
    cam = model[0] if model else None

    # Rectified top-down view of the sheet, 1 px = 1 mm: pixel (col, row) <-> sheet (x, y)
    W, Hh = round(SHEET_Y / MM), round(SHEET_X / MM)
    sheet_px = np.array([[W / 2 - y / MM, Hh / 2 - x / MM] for x, y in [(SHEET_X / 2, SHEET_Y / 2),
                         (SHEET_X / 2, -SHEET_Y / 2), (-SHEET_X / 2, -SHEET_Y / 2), (-SHEET_X / 2, SHEET_Y / 2)]])
    corners_rob = sheet_to_robot([(SHEET_X / 2, SHEET_Y / 2), (SHEET_X / 2, -SHEET_Y / 2),
                                  (-SHEET_X / 2, -SHEET_Y / 2), (-SHEET_X / 2, SHEET_Y / 2)], pose)
    Hinv = np.linalg.inv(H)
    corners_img = cv2.perspectiveTransform(corners_rob.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    cv2.polylines(out, [corners_img.astype(np.int32)], True, (0, 200, 255), 2)
    to_top = cv2.getPerspectiveTransform(corners_img.astype(np.float32), sheet_px.astype(np.float32))
    top = cv2.warpPerspective(bgr, to_top, (W, Hh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    valid = cv2.warpPerspective(np.full(bgr.shape[:2], 255, np.uint8), to_top, (W, Hh)) > 0

    # Objects = whatever is clearly not white paper. Paper brightness is measured locally (lighting gradients).
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2].astype(np.float32)
    paper = cv2.medianBlur(cv2.resize(hsv[:, :, 2], (W // 8, Hh // 8), interpolation=cv2.INTER_AREA), 9)
    paper = cv2.resize(paper, (W, Hh)).astype(np.float32)
    mask = ((gray < paper * 0.72) | (hsv[:, :, 1] > 70)) & valid
    m = round((MARGIN + MARKER + 0.004) / MM)  # ignore the marker corners and the paper edge
    edge = round(0.006 / MM)
    mask[:edge, :] = mask[-edge:, :] = False
    mask[:, :edge] = mask[:, -edge:] = False
    for r0, c0 in [(0, 0), (0, W - m), (Hh - m, 0), (Hh - m, W - m)]:
        mask[r0:r0 + m, c0:c0 + m] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)

    objs: list[Obj] = []
    for k in sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA]):
        if stats[k, cv2.CC_STAT_AREA] < min_area_mm2 or len(objs) >= 9:
            continue
        pts = np.column_stack(np.nonzero(lab == k))[:, ::-1].astype(np.float32)  # (col, row)
        (cc, cr), (w, h), ang = cv2.minAreaRect(pts)
        L, Wd = max(w, h) * MM, min(w, h) * MM
        # long-side direction in rectified px -> sheet frame (col = -y, row = -x)
        a = math.radians(ang if w >= h else ang + 90)
        d_sheet = np.array([-math.sin(a), -math.cos(a)])  # (dx, dy) in the sheet frame
        sx, sy = (Hh / 2 - cr) * MM, (W / 2 - cc) * MM
        (rx, ry), = sheet_to_robot([(sx, sy)], pose)
        if cam is not None:
            # Parallax: the camera sees the object's top and near side, which land on the table plane further
            # from the camera than the footprint. Pull the centre back toward the camera for a point at half the
            # object's height (height ~ true width; the blob is ~1.5x wider than the object).
            hh = min(Wd * 0.65, 0.04) / 2
            f = (cam[2] - hh) / cam[2]
            rx, ry = cam[0] + (rx - cam[0]) * f, cam[1] + (ry - cam[1]) * f
        yaw = math.atan2(d_sheet[1], d_sheet[0]) + pose["yaw"]
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        fill = stats[k, cv2.CC_STAT_AREA] / max(w * h, 1)
        shape = "elongated" if L > 2.5 * Wd else ("round" if fill < 0.85 and L < 1.3 * Wd else
                                                   "square" if L < 1.3 * Wd else "blob")
        color = _color_name(np.median(top[lab == k], axis=0))
        p_img = cv2.perspectiveTransform(np.array([[[rx, ry]]], np.float64), Hinv).reshape(2)
        objs.append(Obj(len(objs) + 1, float(rx), float(ry), float(yaw), float(L), float(Wd), color, shape,
                        (int(p_img[0]), int(p_img[1]))))
    for letter in BOXES:  # printed target boxes, as the robot believes they are
        poly = cv2.perspectiveTransform(box_polygon(letter, pose).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        cv2.polylines(out, [poly.astype(np.int32)], True, (200, 120, 0), 1)
        c = poly.mean(0).astype(int)
        cv2.putText(out, letter, (c[0] - 6, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 120, 0), 1)
    for o in objs:  # numbered labels: the vision model names objects by these numbers
        cv2.circle(out, o.px, 16, (255, 255, 255), -1)
        cv2.circle(out, o.px, 16, (0, 0, 0), 2)
        cv2.putText(out, str(o.id), (o.px[0] - 8, o.px[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cam_txt = f", camera at ({cam[0] * 100:.0f}, {cam[1] * 100:.0f}, {cam[2] * 100:.0f}) cm" if cam is not None else ""
    return Scene(True, f"{len(found)}/4 markers{cam_txt}", found, objs, H, out, model[1] if model else None, bgr)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        out = ROOT / "outputs" / "workspace_sheet.png"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), sheet_image(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # 300 dpi: print at 100 % (no "fit to page"); the marker squares must measure 40 mm
        print(f"wrote {out}  (A4 landscape, 300 dpi, print at 100 %; check a marker measures 40 mm)")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "look":
        if len(sys.argv) > 2:
            img = cv2.imread(sys.argv[2])
        else:
            sys.path.insert(0, str(ROOT))
            from robot import CONFIG

            cam = CONFIG["cameras"]["front"]
            cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            for _ in range(10):
                ok, img = cap.read()
            cap.release()
        s = look(img)
        print(s.detail)
        print(s.text())
        out = ROOT / "outputs" / "workspace_look.jpg"
        cv2.imwrite(str(out), s.annotated)
        print(f"annotated -> {out}")
        return
    print(__doc__)


if __name__ == "__main__":
    main()
