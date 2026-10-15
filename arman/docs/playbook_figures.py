"""Figure for the task playbook: what happens during one do_task.py run.  python3 docs/playbook_figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "figures"
BLUE, GREEN, ORANGE, GREY, RED = "#dce8f5", "#dff2e3", "#fde9d2", "#eeeeee", "#f8dcdc"

fig, ax = plt.subplots(figsize=(8.4, 9.6))
ax.set_xlim(0, 100), ax.set_ylim(0, 112), ax.axis("off")


def box(x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=color, ec="#555555", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, weight="bold" if bold else "normal",
            wrap=True)


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=8, color="#333333", style="italic")


box(22, 102, 56, 7, "python do_task.py --pick \"paper ball\" --into box", GREY, bold=True)
box(22, 89, 56, 8.5, "1. Studio: copy any changed cloud code, restart the server if\nits code changed, check LeRobot version, model files, GPU free", BLUE)
box(22, 77.5, 56, 7, "Load MolmoAct2 on the L4 (2-6 min the first time, then reused)", BLUE)
box(22, 66, 56, 7, "2.-3. Robot (12 V, motor temperature, calibration) and\ncameras (previews saved in outputs/)", BLUE)
box(22, 54.5, 56, 7, "4.-5. Firm grip on. The MODEL drives the arm\n(30 moves per plan, a new plan every 0.3 s)", GREEN, bold=True)
for y1, y2 in ((102, 97.5), (89, 84.5), (77.5, 73), (66, 61.5)):
    arrow(50, y1, 50, y2)

box(2, 36, 30, 10, "Gripper closed on NOTHING\n(missed)", ORANGE)
box(68, 36, 30, 10, "Gripper stopped part way:\nHOLDING something", GREEN)
box(36, 38, 28, 6, "time up / blocked /\nmotor hot / Ctrl+C", RED)
arrow(40, 54.5, 17, 46.5)
arrow(60, 54.5, 83, 46.5)
arrow(50, 54.5, 50, 44.5)

box(2, 19, 30, 10, "CODE: open, lift a little,\nnew try with a fresh plan\n(up to RETRIES times)", ORANGE)
arrow(17, 36, 17, 29.5)
ax.annotate("", xy=(22, 57.5), xytext=(4, 24), arrowprops=dict(arrowstyle="->", color="#b36b00", lw=1.1,
                                                                  connectionstyle="arc3,rad=-0.45"))
ax.text(-1, 50, "try again", fontsize=8, color="#b36b00", style="italic")

box(68, 17, 30, 12, "CODE: squeeze, lift ~8-10 cm,\ncarry to the taught pose\n'above_box', open\n(AUTO_DROP)", GREEN)
arrow(83, 36, 83, 29.5)
ax.text(70, 31.5, "slips out on the way:\nretry", fontsize=7.6, color="#b36b00", style="italic")

box(22, 2, 56, 8, "6. Ease to 'home', motors off (same connection, no sag)\nRESULT + log in outputs/runs/<time>/", BLUE, bold=True)
arrow(50, 38, 50, 10.6)
arrow(83, 17, 70, 10.6)

fig.tight_layout()
fig.savefig(OUT / "run_flow.png", dpi=200)
print("run_flow.png ok")
