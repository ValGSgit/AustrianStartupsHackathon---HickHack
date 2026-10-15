"""Camera placement diagrams for the SO-101 + MolmoAct2 guide (all distances in cm)."""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge

OUT = sys.argv[1]
BLUE, ORANGE, GREY, GREEN, RED = "#1f6fb2", "#e07b00", "#888888", "#2e8b57", "#c0392b"


def fov(ax, x, y, heading_deg, half_deg, length, color):
    a1, a2 = math.radians(heading_deg - half_deg), math.radians(heading_deg + half_deg)
    ax.add_patch(Polygon([(x, y), (x + length * math.cos(a1), y + length * math.sin(a1)),
                          (x + length * math.cos(a2), y + length * math.sin(a2))],
                         closed=True, color=color, alpha=0.12, lw=0))
    for a in (a1, a2):
        ax.plot([x, x + length * math.cos(a)], [y, y + length * math.sin(a)], color=color, lw=0.8, ls="--")


# ---------------------------------------------------------------- 1. top view
fig, ax = plt.subplots(figsize=(8.2, 7.2))
ax.add_patch(Rectangle((-45, -25), 110, 105, color="#f4f1ea", zorder=0))  # table
ax.text(-43, 76, "TABLE (plain, light colour, nothing else on it)", fontsize=8, color=GREY)
ax.add_patch(Wedge((0, 0), 35, -10, 190, color=GREEN, alpha=0.07))
ax.add_patch(Wedge((0, 0), 35, -10, 190, fill=False, color=GREEN, lw=1, ls=":"))
ax.text(-41, 24, "comfortable\nreach ~35 cm", fontsize=7.5, color=GREEN)
ax.add_patch(FancyBboxPatch((-7, -7), 14, 14, boxstyle="round,pad=0.5", color="#222222"))
ax.text(0, -12.5, "ROBOT BASE\n(clamped, faces forward = up)", ha="center", va="top", fontsize=8)
ax.annotate("", xy=(0, 14), xytext=(0, 7.5), arrowprops=dict(arrowstyle="->", color="#222222", lw=1.5))
ax.add_patch(Rectangle((-14, 14), 20, 13, fill=False, color=RED, lw=1.5, ls="--"))
ax.text(-4, 28.5, "OBJECT ZONE: 15-27 cm in front", ha="center", fontsize=8, color=RED)
ax.add_patch(Circle((-8, 20), 1.6, color="#222222"))
ax.add_patch(Circle((-1, 22.5), 1.6, color="#222222"))
ax.add_patch(Circle((15, 20), 7, color="#d9d9d9", ec=GREY))
ax.text(15, 20, "BOWL / BOX\nwide + low", ha="center", va="center", fontsize=7)
# cam0: phone in front, facing the robot
cx0, cy0 = 4, 62
fov(ax, cx0, cy0, -95, 33, 95, BLUE)
ax.add_patch(Rectangle((cx0 - 4, cy0 - 1.5), 8, 3.5, color=BLUE))
ax.text(cx0 + 6, cy0 + 4, "CAM 0 = iPhone (top / front)\n60 cm in front of the base,\n50 cm above the table,\ntilted ~50° down, facing the robot",
        fontsize=8, color=BLUE, va="bottom")
# cam1: MacBook on the right side, facing left
cx1, cy1 = 52, 21
fov(ax, cx1, cy1, 180, 30, 80, ORANGE)
ax.add_patch(Rectangle((cx1 - 1.2, cy1 - 8), 2.4, 16, color=ORANGE))
ax.text(cx1 - 1, cy1 - 24, "CAM 1 = MacBook webcam (side)\nlaptop 40-45 cm to the side of\nthe object zone, lid camera ~22 cm high,\nlid tilted so it looks ~20° down",
        fontsize=8, color=ORANGE, ha="right", va="top")
ax.text(-43, -22, "You and other people: BEHIND cam 0 or behind the laptop, never behind the robot "
        "(the cameras would see you move).", fontsize=7.5, color=RED)
ax.set_xlim(-46, 66), ax.set_ylim(-26, 82)
ax.set_aspect("equal"), ax.set_xlabel("cm"), ax.set_ylabel("cm")
ax.set_title("1. Top view: where everything goes", fontsize=12, loc="left")
fig.tight_layout(), fig.savefig(f"{OUT}/top_view.png", dpi=200), plt.close(fig)


# ---------------------------------------------------------------- 2. side views
def elevation(ax, title, cam_x, cam_h, target_x, target_h, vfov, color, label, notes, robot_x=0, flip=False):
    ax.plot([-30, 75], [0, 0], color="#555555", lw=2)
    ax.text(-28, -4, "table", fontsize=8, color=GREY, zorder=4)
    # robot (side silhouette)
    rx = robot_x
    ax.add_patch(Rectangle((rx - 5, 0), 10, 5, color="#222222"))
    ax.plot([rx, rx + 2, rx + 14, rx + 20, rx + 22], [5, 19, 22, 14, 6], color="#222222", lw=4, solid_capstyle="round")
    ax.text(rx, -4, "robot", ha="center", fontsize=8, zorder=4)
    ax.add_patch(Rectangle((target_x - 1.5, 0), 3, 5, color="#222222"))
    ax.text(target_x, -4, "object", ha="center", fontsize=8, color=RED, zorder=4)
    pitch = math.degrees(math.atan2(cam_h - target_h, abs(cam_x - target_x)))
    heading = 180 + pitch if cam_x > target_x else -pitch
    length = math.hypot(cam_x - target_x, cam_h - target_h) * 1.6
    fov(ax, cam_x, cam_h, heading if cam_x > target_x else -pitch, vfov / 2, length, color)
    ax.add_patch(Rectangle((-30, -16), 105, 16, color="white", zorder=2))
    ax.plot([-30, 75], [0, 0], color="#555555", lw=2, zorder=3)
    ax.plot([cam_x, target_x], [cam_h, target_h], color=color, lw=1.2)
    ax.add_patch(Rectangle((cam_x - 2.5, cam_h - 1.5), 5, 3, color=color))
    ax.plot([cam_x, cam_x], [0, cam_h], color=color, lw=0.8, ls=":")
    ax.text(cam_x + 1.5, cam_h / 2, f"{cam_h:.0f} cm\nhigh", fontsize=8, color=color)
    ax.annotate("", xy=(target_x, -9), xytext=(cam_x, -9), arrowprops=dict(arrowstyle="<->", color=color), zorder=4)
    ax.text((cam_x + target_x) / 2, -13.5, f"{abs(cam_x - target_x):.0f} cm to the object zone", ha="center", fontsize=8, color=color, zorder=4)
    ax.text(cam_x - 1 if cam_x > target_x else cam_x + 1, cam_h + 3, f"{label}\ntilt ~{pitch:.0f}° down",
            fontsize=8.5, color=color, ha="right" if cam_x > target_x else "left")
    ax.text(-28, 46, notes, fontsize=7.8, va="top", bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.set_xlim(-30, 75), ax.set_ylim(-16, 62), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")


fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 8.6))
elevation(a1, "2a. Cam 0 (iPhone) seen from the side", cam_x=60, cam_h=50, target_x=20, target_h=4, vfov=40,
          color=BLUE, label="iPhone, landscape", robot_x=0,
          notes="Must show: the whole robot incl. its base, the object, the bowl/box,\n"
                "and the gripper over the full path, with ~5 cm margin around it.\n"
                "Not straight down (90°) and not from behind the robot.")
elevation(a2, "2b. Cam 1 (MacBook webcam) seen from the front", cam_x=45, cam_h=22, target_x=3, target_h=4, vfov=45,
          color=ORANGE, label="MacBook lid camera", robot_x=-18,
          notes="Must show: the table surface line, the object and the bowl/box at table level,\n"
                "and the gripper from ~25 cm above the object down to the table.\n"
                "This view gives the HEIGHT information the top camera lacks.")
fig.tight_layout(), fig.savefig(f"{OUT}/side_views.png", dpi=200), plt.close(fig)
print("diagrams ok")
