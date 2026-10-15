"""Builds the SO-101 + MolmoAct2 setup and improvement guide."""
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

G, OUT, REF, OUTS = sys.argv[1:5]  # guide dir, output pdf, reference frames dir, project outputs dir

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1f3b5a"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1f3b5a"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, alignment=TA_LEFT)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=11, textColor=colors.HexColor("#444444"))
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.6, leading=11.2)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")


def p(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(path).getSize()
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


def photo_pair(a, b, cap_a, cap_b, w=6.6):
    t = Table([[img(a, w), img(b, w)], [p(cap_a, SM), p(cap_b, SM)]], colWidths=[w * cm + 0.3 * cm] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return t


story = []
story += [p("SO-101 + MolmoAct2: camera setup, improvements and fine-tuning plan", H1),
          p("Arman's SO-101 hackathon setup, 14 October 2026. Every recommendation below is based on the MolmoAct2 "
            "paper, the official LeRobot MolmoAct2 documentation, Hugging Face's own SO-101 reference dataset, or our "
            "own tests on this robot. Where something is an estimate rather than a measurement, it says so.", SM),
          Spacer(1, 6),
          table([["Section", "What it covers"],
                 ["1. Camera setup", "Exact positions for the iPhone (top) and the MacBook webcam (side), what each must see, drawings and example photos."],
                 ["2. Improvements made", "Code changes made today and why. The biggest: the model was seeing a blurry image."],
                 ["3. Fine-tuning plan", "Is it easy, which mode fits the L4, how many demos, steps, hours and rounds."],
                 ["4. Before you connect", "Short checklist and the exact commands."]], [4.2 * cm, 12.6 * cm]),
          Spacer(1, 8)]

# ---------------------------------------------------------------- 1
story += [p("1. Camera setup", H1),
          p("<b>Why two cameras, and why these positions.</b> The official LeRobot MolmoAct2 example for SO-100/101 uses a "
            "<b>top</b> and a <b>side</b> camera at 640x480, mapped to the model's two camera slots cam0 and cam1. "
            "Hugging Face's own SO-101 reference dataset (lerobot/svla_so101_pickplace) uses the same pair, called "
            "\"up\" and \"side\". The MolmoAct2 paper evaluated SO-100/101 with a wrist camera plus one outside camera; "
            "a wrist camera would be the best second camera, but with an iPhone and a MacBook, top + side is the closest "
            "proven setup. The top view shows <i>where</i> things are; the low side view shows <i>how high</i> the "
            "gripper is above the object, which a single top view cannot show well."),
          Spacer(1, 4), img(f"{G}/top_view.png", 16.5), PageBreak(),
          img(f"{G}/side_views.png", 15.5), PageBreak(),
          p("Official reference setup (Hugging Face, lerobot/svla_so101_pickplace)", H2),
          photo_pair(f"{REF}/ref_up.jpg", f"{REF}/ref_side.jpg",
                     "<b>\"up\" camera (= our cam0, iPhone).</b> In front of the robot, raised, looking down at roughly "
                     "45-50°. Whole robot, object and container in view. Plain white table and wall.",
                     "<b>\"side\" camera (= our cam1, MacBook).</b> Low, close to table level, 30-40 cm away. You can "
                     "see exactly how high the gripper is above the red block. Low, flat container (tray outline)."),
          p("Our photos from today", H2),
          photo_pair(f"{OUTS}/probe_1.jpg", f"{OUTS}/probe_0.jpg",
                     "<b>MacBook webcam: good basis.</b> Pawn and cup clearly visible on a white table. Fix: a person "
                     "stands behind the robot (movement in the background confuses the model; stand behind the "
                     "camera instead); the cup is a tall stack (use one low, wide bowl or box).",
                     "<b>iPhone: unusable as it was.</b> Completely out of focus: it was too close to the table or the "
                     "lens was covered. Mount it 50-70 cm away from the work area, landscape, lens clean."),
          p("What each camera must cover", H2),
          table([["", "Cam 0: iPhone (top / front)", "Cam 1: MacBook webcam (side)"],
                 ["Position", "60 cm in front of the robot base, 50 cm above the table, facing the robot",
                  "40-45 cm to the side of the object zone, laptop on the table (lid camera ~22 cm high)"],
                 ["Angle", "tilted ~50° down; centre of the picture = object zone; never straight down, never from behind the robot",
                  "lid tilted so it looks ~20° down at the object zone"],
                 ["Must be in view", "whole robot incl. base; object; bowl/box; the gripper's whole path; ~5 cm margin",
                  "table surface; object and bowl/box; gripper from ~25 cm above the object down to the table"],
                 ["Background", "plain; no people or moving things", "plain wall or a box; not you"],
                 ["Fixing", "phone holder or tape; never touch it during or between runs",
                  "laptop must not move; mark its feet with tape"],
                 ["Format", "landscape, 640x360 (what the iPhone delivers)", "640x480"]],
                [3.0 * cm, 6.9 * cm, 6.9 * cm]),
          ]
story += [KeepTogether([p("Scene rules that matter as much as the cameras", H2)] + bullets([
    "<b>Light:</b> bright, even, from above or from the cameras' side. Your test showed it: with a flashlight it worked "
    "better. Avoid windows behind the scene (backlight) and hard moving shadows.",
    "<b>Objects:</b> one object at a time, a strong colour against the table (black pawn, red can); 15-27 cm in front "
    "of the base. The model's SO-100/101 test tasks were like \"block in box\", \"pen on notebook\".",
    "<b>Container:</b> low and wide (bowl, box, tray). A tall narrow cup is the hardest target.",
    "<b>Wording:</b> keep the style the model was trained with: \"move the arm towards the X, grasp it, lift it up, and "
    "drop it into the Y\". Name colours (\"black pawn\", \"white bowl\").",
    "<b>People:</b> stay behind cam 0 or behind the laptop. Keep a hand near the robot's power plug, not in view."]))]
story += [Spacer(1, 12)]

# ---------------------------------------------------------------- 2
story += [p("2. Improvements made today (before reconnecting)", H1),
          table([["#", "Change", "Why (evidence)", "Status"],
                 ["1", "<b>Full image resolution</b>: images are now sent and kept at 378x378 instead of being squeezed to 224x224.",
                  "MolmoAct2's image processor resizes every image to 378x378 (processor_config.json: crop_mode \"resize\", size 378). "
                  "LeRobot's remote server first shrank images to the 224x224 written in the checkpoint config, so the model "
                  "received a blurry, upscaled copy: ~35% fewer pixels per side than in training. Likely a real cause of "
                  "imprecise grasps.",
                  "Image path tested on the Mac (decode, size, scaling, camera slot). First GPU run will confirm."],
                 ["2", "Smooth motion: small JPEG uploads, uploads in the background, planning on every request, model kept loaded.",
                  "Measured on the L4: 131 -> 647 movements per 25 s (about 26 of 30 possible per second). "
                  "Planning time 0.19 s per second of motion.", "Tested (done earlier today)"],
                 ["3", "Cameras found automatically by type (iPhone = 16:9, webcam = 4:3 colour), with a longer wake-up wait for the iPhone over Wi-Fi.",
                  "macOS renumbered the cameras several times today; the iPhone needs a few seconds to deliver.", "Tested"],
                 ["4", "Recording now includes the side camera (front + side), so training data matches the two-camera setup.",
                  "Training and running must use the same cameras.", "Tested with a fake robot"],
                 ["5", "Fine-tuning scripts (Mac + Studio) and switching to a fine-tuned model with one setting (MOLMO_CHECKPOINT). "
                  "The server restarts by itself when the code or the model changes.",
                  "Official LeRobot MolmoAct2 docs: start from the LeRobot checkpoint (keeps its SO-101 joint-frame correction), "
                  "action-expert-only or LoRA for small datasets, bf16, gradient checkpointing, 8 flow timesteps.",
                  "Command validated on the Mac up to weight loading; real training needs the Studio"],
                 ["6", "Server wait raised to 12 minutes.", "After the Studio wakes from sleep its disk is slow: loading took 523 s once.", "Done"]],
                [0.6 * cm, 5.0 * cm, 7.6 * cm, 3.6 * cm]),
          Spacer(1, 6),
          p("<b>What code cannot fix.</b> The base model has never seen your table, objects or exact calibration. "
            "Zero-shot it scored 56.7 on average in the authors' SO-100/101 tests (33.3 on \"block in box\", 86.7 on "
            "\"pen on notebook\"), with two cameras. Sometimes working is therefore the expected zero-shot result; the "
            "reliable way to \"much better movement\" for one task is fine-tuning on your own demonstrations (section 3)."),
          PageBreak()]

# ---------------------------------------------------------------- 3
story += [p("3. Fine-tuning on the L4: is it easy, how much, how long", H1),
          p("<b>Is it easy?</b> Yes, technically: LeRobot supports fine-tuning MolmoAct2 directly, and the scripts are ready. "
            "The real work is recording good demonstrations. Three commands: record on the Mac, train on the Studio, "
            "point do_task.py at the new model."),
          p("Which training mode fits the L4 (24 GB)", H2),
          table([["Mode", "What is trained", "Memory (official, batch 8 / 16)", "Fits L4?"],
                 ["Action expert only <b>(start here)</b>", "the part that turns what the model sees into movements",
                  "16.5 / 18.3 GB", "Yes, batch 16"],
                 ["LoRA on vision-language", "small adapters in the 'seeing' part + the action part", "20.2 / 26.8 GB",
                  "Yes at batch 8 only"],
                 ["Full fine-tune", "everything", "48.3 / 49.8 GB", "No (needs 80 GB GPU)"]],
                [4.2 * cm, 5.4 * cm, 4.2 * cm, 3.0 * cm]),
          p("Official guidance: for real datasets under 200 demonstrations, batch 16-32 with action-expert-only or LoRA, "
            "always keeping the action expert trainable (\"crucial for model performance\"). Measured on an H100 by the "
            "LeRobot team; the L4 has the memory but is much slower.", SM),
          p("Data: how many demonstrations", H2)]
story += bullets([
    "<b>50 demonstrations per task</b> to start: 5 object positions x 10 each, recorded with the joystick through "
    "record.py (both cameras). This is the standard LeRobot starting point (their docs: \"at least 50 episodes, "
    "10 per location\"; the SmolVLA docs found 25 not enough).",
    "<b>Consistency beats quantity:</b> same approach every time (from above, grasp low, lift straight up, over the "
    "container, open). Smooth, calm, no hesitations.",
    "Each demo 10-20 s. Recording 50 takes about <b>60-90 minutes</b> including resets.",
    "Add <b>20-30 more</b> later, targeted at the positions where the model fails."])
story += [p("Training: steps, time, rounds", H2),
          table([["", "Round 1", "Round 2", "Round 3 (if needed)"],
                 ["Data", "50 demos", "+20-30 demos of failure cases", "+20 demos or switch to LoRA"],
                 ["Training", "action expert only, batch 16, 3,000 steps", "train again on all data, 3,000 steps",
                  "LoRA, batch 8, 3,000 steps"],
                 ["Time on L4 (estimate)", "~2-4 h", "~2-4 h", "~3-5 h"],
                 ["Test", "20 tries, same positions + 5 new", "same test", "same test"],
                 ["Typical result*", "clearly better than zero-shot", "~70-90% on the fixed task", "if vision was the problem"]],
                [3.4 * cm, 4.5 * cm, 4.5 * cm, 4.4 * cm]),
          p("* Estimates, not measurements. The time per training step on an L4 is not published; the script prints the "
            "real speed in the first minutes, and the estimate gets replaced by a measurement then. The authors' own "
            "real-robot fine-tunes used 8 H100 GPUs and 50,000 updates for several tasks at once; one fixed task in one "
            "scene needs far less. Typical LeRobot single-task results after 50-100 good demos are in the 70-90% range, "
            "but nothing guarantees it for this model on this table.", SM),
          Spacer(1, 4),
          p("<b>Overall: about 2-3 rounds, 1-2 days elapsed, roughly 6-12 GPU hours on the L4.</b> Keep checkpoints every "
            "500 steps and test the 1,500 / 3,000-step ones: the last one is not always the best."),
          p("Practical points", H2)]
story += bullets([
    "<b>Turn off auto-sleep for the Studio while training</b> (Lightning studio settings). Today the Studio paused "
    "whenever no connection was open.",
    "The model server and training share the GPU: the training script stops the server; do_task.py restarts it.",
    "<b>Backup:</b> ACT trains on the same recordings in ~1-2 h on the L4 and runs cool on the Mac.",
    "<b>Never move the cameras or recalibrate</b> between recording and testing. If you must, record again."])
story += [Spacer(1, 12)]

# ---------------------------------------------------------------- 4
story += [p("4. Before you connect: checklist and commands", H1)]
story += bullets([
    "Robot: 12V supply on the robot, 5V on the joystick (never 12V on the joystick). USB plugged in. Arm at rest.",
    "Mac: Bluetooth ON (the iPhone camera needs it), Wi-Fi on. The RealSense unplugged.",
    "iPhone: 60 cm in front of the robot, 50 cm high, tilted ~50° down, landscape, locked, lens clean, in focus.",
    "MacBook: 40-45 cm to the side of the object zone, lid tilted so its camera looks ~20° down. Feet marked with tape.",
    "Scene: one object, one low wide container, plain table, even light, nobody behind the robot.",
    "Lightning: start Studio arman1 (L4). First start after sleep: model loading can take up to ~9 minutes."])
story += [Spacer(1, 6),
          table([["Step", "Command (in the so101-hackathon folder, Terminal app)"],
                 ["Check everything + camera previews", "python do_task.py --check"],
                 ["Run a task (zero-shot)", "python do_task.py \"move the arm towards the black pawn, grasp it, lift it up, and drop it into the bowl\""],
                 ["Record demos for fine-tuning", "python record.py 10   (repeat 5x)"],
                 ["Fine-tune on the Studio", "scripts/train_molmo.sh   (or: scripts/train_molmo.sh lora 3000)"],
                 ["Use the fine-tuned model", "set MOLMO_CHECKPOINT=.../checkpoints/003000/pretrained_model in settings.env, then do_task.py as usual"]],
                [5.2 * cm, 11.6 * cm]),
          Spacer(1, 10),
          p("Sources", H2)]
story += bullets([
    "MolmoAct2 paper (arXiv 2605.02881): SO-100/101 zero-shot results, camera setup (wrist + external), fine-tuning compute.",
    "LeRobot MolmoAct2 documentation (huggingface.co/docs/lerobot/molmoact2): top + side example, memory table, fine-tuning modes, joint-frame transform.",
    "allenai/MolmoAct2-SO100_101 processor_config.json: image resize to 378x378.",
    "Hugging Face dataset lerobot/svla_so101_pickplace: reference up + side camera views.",
    "LeRobot docs (il_robots, smolvla): 50 episodes, 10 per location; 25 episodes not enough.",
    "Our own tests on this robot and the L4 Studio (14 October 2026)."])

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.6 * cm, title="SO-101 + MolmoAct2 guide", author="Arman")
doc.build(story)
print("PDF written:", OUT)
