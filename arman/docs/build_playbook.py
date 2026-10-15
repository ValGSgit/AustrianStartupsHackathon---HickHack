"""Builds docs/SO101_Task_Playbook.pdf.   python3 docs/playbook_figures.py && python3 docs/build_playbook.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DOCS = Path(__file__).parent
FIG = DOCS / "figures"
OUT = DOCS / "SO101_Task_Playbook.pdf"

ss = getSampleStyleSheet()
NAVY = colors.HexColor("#1f3b5a")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceBefore=4, spaceAfter=8, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5, textColor=NAVY)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, alignment=TA_LEFT)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=11, textColor=colors.HexColor("#444444"))
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.5, leading=11.1)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("CODE", parent=B, fontName="Courier", fontSize=8.6, leading=11.4, leftIndent=8,
                      backColor=colors.HexColor("#f3f3f3"), borderPadding=(3, 4, 3, 4), spaceBefore=3, spaceAfter=5)
NOTE = ParagraphStyle("NOTE", parent=B, backColor=colors.HexColor("#fff6e0"), borderPadding=(5, 6, 5, 6),
                      borderColor=colors.HexColor("#e0b84c"), borderWidth=0.6, spaceBefore=4, spaceAfter=8)


def p(t, st=B):
    return Paragraph(t, st)


def code(*lines):
    return Paragraph("<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for l in lines), CODE)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True, shade=None):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    for row, col in (shade or {}).items():
        style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor(col)))
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


GREEN, YELLOW, RED = "#e6f4e8", "#fff6dc", "#fbe3e3"
story = []

# ---------------------------------------------------------------- cover
story += [
    p("SO-101 + MolmoAct2: task playbook and setup checklist", H1),
    p("Arman's SO-101 hackathon setup, 14 October 2026. This is the second guide. The first one, "
      "<b>docs/SO101_MolmoAct2_Guide.pdf</b>, has the exact camera positions with drawings and the fine-tuning plan. "
      "This one covers: which tasks to do, how to set everything up in the right order, what the new code does for "
      "you, and what to do when something goes wrong. Facts come from the MolmoAct2 paper, the model's Hugging Face "
      "page, the official LeRobot docs, the Feetech motor documentation and our own tests; estimates are marked.", SM),
    Spacer(1, 6),
    table([["Section", "What it covers"],
           ["1. Status", "What is ready, what was tested and how, what can only be tested with the robot."],
           ["2. Your documents and files", "Every PDF, command and settings line you need, in one list."],
           ["3. Setup, step by step", "Plug-and-play order from power to first task, with the exact commands."],
           ["4. The best tasks", "What the model was trained and tested on, a ranked list for a weak gripper, demo combos."],
           ["5. What the code does for you", "Automatic cloud update, carry-to-container, retry, firm grip, logs."],
           ["6. How to improve a lot", "The improvements ranked by impact: what you do and what the code does."],
           ["7. RealSense D435i", "Whether to use it and how."],
           ["8. Training on your own demos", "Short version of the fine-tuning plan."],
           ["9. When something goes wrong", "Messages you may see and what to do."]], [4.3, 12.5]),
    Spacer(1, 10),
    p("<b>The short version.</b> Start the arman1 Studio on lightning.ai, plug in the robot (12 V) and the cameras, "
      "then in the Terminal app: <font name='Courier'>python do_task.py --check</font>, "
      "<font name='Courier'>python teach_drop.py box</font>, and "
      "<font name='Courier'>python do_task.py --pick \"paper ball\" --into box</font>. "
      "Use soft or long-flat objects and a wide, low container.", NOTE),
]

# ---------------------------------------------------------------- 1 status
story += [
    PageBreak(), p("1. Status: is everything ready?", H1),
    p("Yes for everything that can be tested without the robot, the cameras and the GPU. The table says exactly what "
      "was tested and how. The parts marked yellow can only be confirmed on the real hardware; the first real run "
      "checks them automatically and writes a log that Claude can read to fine-tune the numbers."),
    Spacer(1, 4),
    table([["Part", "State", "How it was tested"],
           ["Cloud update before every run (code sync, restart if changed, version/model/GPU checks)", "Ready, tested",
            "Against a simulated Studio folder: old files replaced, server restarted only when its code changed, missing model caught."],
           ["Whole run: model turns, grab watch, carry, retry, going home", "Ready, tested",
            "Offline end-to-end test: real LeRobot client and server code, scripted fake model, simulated gripper that stops on an object. "
            "Three scenarios pass repeatedly (tests/offline_run_test.py)."],
           ["Firm grip setting and motor temperature stop", "Ready, tested", "Against a fake motor bus: writes, restore to normal, hot-motor stop."],
           ["RealSense colour camera path", "Ready, tested without the camera",
            "A fake sender pushed a pure red picture through the same route; it arrived red (colours not swapped)."],
           ["Sharper images (378 x 378 instead of 224)", "Ready, needs the GPU",
            "Unit-tested on the Mac; confirmed on the next --check with the Studio running."],
           ["Grab-detection numbers (when 'holding' vs 'missed')", "Educated guesses",
            "Need one real run; outputs/runs/ shows the real gripper numbers. grip_test.py measures them directly."],
           ["RealSense on this Mac", "Needs the camera plugged in", "sudo .venv/bin/python realsense_bridge.py --test"],
           ["Drop pose for each container", "You teach it once", "python teach_drop.py box (cup, bowl, notebook...)"]],
          [5.6, 3.0, 8.2], shade={1: GREEN, 2: GREEN, 3: GREEN, 4: GREEN, 5: YELLOW, 6: YELLOW, 7: YELLOW, 8: YELLOW}),
    Spacer(1, 6),
    p("<b>Safety kept from before:</b> 15-degree limit per move, stop if a joint is blocked for 1.5 s, stop on a motor "
      "error or a motor at 60 degrees C, hold the pose when a run ends and ease home before the motors go limp. "
      "Never 12 V on the joystick. Emergency stop: unplug the robot's power supply."),
    p("<b>To get the old behaviour back:</b> in settings.env set AUTO_DROP=0, RETRIES=0, GRIP_STRENGTH=normal. "
      "The complete code before these changes is in saved_setup/code_before_assist/."),
]

# ---------------------------------------------------------------- 2 documents
story += [
    PageBreak(), p("2. Your documents and files", H1),
    p("Everything lives in <b>workplace/so101-hackathon/</b>. Run every command from that folder in the "
      "<b>Terminal app</b> (macOS only lets Terminal use the cameras), after <font name='Courier'>source .venv/bin/activate</font>."),
    p("Documents", H2),
    table([["File", "Use it for"],
           ["docs/SO101_Task_Playbook.pdf", "This guide: tasks, setup order, what the code does, troubleshooting."],
           ["docs/SO101_MolmoAct2_Guide.pdf", "Camera positions with drawings and photos, the image fix, the full fine-tuning plan."],
           ["settings.env", "All settings: ports, cameras, task, Studio address, model, AUTO_DROP, RETRIES, GRIP_STRENGTH."],
           ["outputs/runs/&lt;time&gt;/", "One folder per run: events.txt (what happened + RESULT), log.csv (every move), camera pictures."],
           ["outputs/do_task_cam0.jpg, cam1.jpg", "What the cameras see, saved by every check."]], [5.6, 11.2]),
    p("Commands", H2),
    table([["Command", "What it does"],
           ["python do_task.py --check", "Updates the cloud code, loads the model, checks robot and cameras, saves camera previews. Moves nothing."],
           ["python do_task.py --pick X --into Y", "Runs a task (also --onto Y). Builds the sentence the way the model was trained."],
           ["python do_task.py \"full sentence\"", "Runs a task with your own wording. Add --seconds 90 to allow longer, --drop box to choose the drop pose."],
           ["python do_task.py --fake", "Tests the whole cloud chain with a fake robot (Studio must be running)."],
           ["python teach_drop.py box", "With the joystick: teaches 'above_box', where the code drops things. Once per container."],
           ["python grip_test.py above_box", "Measures the gripper empty and with each object (saves grip.json); lets you feel the grip."],
           ["sudo .venv/bin/python realsense_bridge.py", "Gives the RealSense colour picture to the other programs (own Terminal tab). Add --test for one photo."],
           ["python joystick.py", "Drive the robot with the joystick (for checking, not needed for tasks)."],
           ["python record.py 10", "Record 10 demonstrations with the joystick for training."],
           ["scripts/train_molmo.sh", "Copy the demos to the Studio and train there (see section 8)."],
           ["scripts/save_setup.sh", "Back up calibrations, offsets, poses and settings (restore: add 'restore')."],
           ["cd tests && ../.venv/bin/python offline_run_test.py retry_then_carry", "Re-run the offline test after any code change (no hardware)."]],
          [6.3, 10.5]),
    p("Settings you may change (settings.env)", H2),
    table([["Line", "Meaning"],
           ["FRONT_CAM=iphone", "Top camera. iphone, webcam, realsense or a number."],
           ["SIDE_CAM=webcam", "Side camera. Recommended: realsense (see section 7) or webcam."],
           ["MOLMO_TASK=\"...\"", "Task used when you run do_task.py without a task."],
           ["MOLMO_CHECKPOINT=", "Empty = base model. After training: the path train_molmo.sh printed."],
           ["AUTO_DROP=1", "Code carries a held object to the taught drop pose. 0 = model does everything."],
           ["RETRIES=2", "New tries after a missed grab. 0 = none."],
           ["GRIP_STRENGTH=strong", "Gripper keeps more force while squeezing. normal = LeRobot's setting."]], [5.6, 11.2]),
]

# ---------------------------------------------------------------- 3 setup
story += [
    PageBreak(), p("3. Setup, step by step", H1),
    p("A. Before the hackathon table (no hardware needed)", H2),
    *bullets([
        "<b>Rubber on the gripper.</b> Rubber bands around both fingers, or a strip of bike inner tube, anti-slip mat "
        "or hot-glue bumps on the inner faces. Rubber holds roughly 2 to 4 times better than smooth plastic with the same squeeze. 10 minutes.",
        "<b>Objects:</b> pick from section 4 (soft or long-flat, under about 100 g, at most two-thirds of the gripper opening).",
        "<b>Container:</b> wide and low (box, bowl, basket, plate, notebook). No tall narrow cup.",
        "<b>Tape marks:</b> a square where the object starts (15 to 27 cm in front of the robot base) and a mark for the container.",
    ]),
    p("B. Power and cables", H2),
    *bullets([
        "Robot (follower): its <b>12 V</b> supply. Joystick (leader): <b>5 V</b> only, never 12 V.",
        "Both USB cables into the Mac. The check reads the voltage and refuses to run below 10 V.",
    ]),
    p("C. Cameras", H2),
    *bullets([
        "<b>Top (FRONT_CAM=iphone):</b> iPhone 60 cm in front of the robot, 50 cm above the table, tilted about 50 degrees down, "
        "landscape, in focus. Mac Bluetooth and Wi-Fi ON, iPhone locked and near the Mac.",
        "<b>Side (SIDE_CAM=realsense or webcam):</b> 40 to 45 cm to the side of the object area, lens about 20 cm high, looking "
        "about 20 degrees down. For the RealSense, first start the bridge in its own Terminal tab (it asks for your Mac password):",
    ]),
    code("sudo .venv/bin/python realsense_bridge.py"),
    *bullets(["Nobody standing behind the robot in view. Even light (a lamp from the side helps). Drawings: next page."]),
    p("D. The cloud (Lightning AI)", H2),
    *bullets([
        "On lightning.ai open the Studio <b>arman1</b>, press <b>Start</b>, wait until it shows <b>Running</b> (GPU L4).",
        "Nothing else to do there. do_task.py updates the cloud code by itself <b>before</b> loading the model (section 5). "
        "Never paste Lightning's setup command (it contains your private key) anywhere except your own Terminal.",
        "When you are finished for the day, press <b>Stop</b> on the Studio to save credits.",
    ]),
    p("E. Check, teach, run", H2),
    code("source .venv/bin/activate",
         "python do_task.py --check          # 2-6 min the first time (model loading)",
         "open outputs/do_task_cam0.jpg outputs/do_task_cam1.jpg   # do the views match the drawings?",
         "python teach_drop.py box           # once per container, with the joystick",
         "python grip_test.py above_box      # optional, 2 min: measures your gripper",
         "python do_task.py --pick \"paper ball\" --into box"),
    p("After each run the Terminal shows <b>RESULT</b> and the log folder. If it failed, tell Claude the folder name: "
      "the log shows exactly where (missed, slipped, blocked) and the camera pictures show what the model saw."),
    PageBreak(), p("Camera positions (from the first guide)", H2),
    img(FIG / "top_view.png", 14.5), img(FIG / "side_views.png", 13.0),
]

# ---------------------------------------------------------------- 4 tasks
story += [
    PageBreak(), p("4. The best tasks", H1),
    p("4.1 What the model was trained and tested on", H2),
    p("<b>Training:</b> the SO-100/101 version of MolmoAct2 learned from about 38,000 demonstrations in 1,222 public "
      "SO-100/101 datasets that people uploaded to Hugging Face, mostly pick-and-place. The paper does not list them one by one."),
    p("<b>Tested by its makers on an SO-100 arm, without extra training</b> (15 tries per task; the score gives part credit: "
      "0.25 for reaching, 0.5 for picking up, 1.0 for placing, so these are not pure success rates):"),
    table([["Task", "MolmoAct2", "pi0 (comparison)", "What it tells us"],
           ["Pen onto a notebook", "86.7", "80.0", "Long thin object lying flat + big flat target: best case."],
           ["Tissues into a basket", "73.3", "20.0", "Soft objects forgive a weak, imprecise grip."],
           ["Fork onto a plate", "70.0", "30.0", "Long flat object, big low target."],
           ["Block into a box", "33.3", "90.0", "Hard small objects are its weak spot."],
           ["Stacking blocks", "20.0", "6.7", "Precise placing: avoid."]], [4.0, 2.1, 2.6, 8.1],
          shade={1: GREEN, 2: GREEN, 3: GREEN, 4: RED, 5: RED}),
    Spacer(1, 4),
    p("<b>The model page's own example:</b> \"move the arm towards the lemon, grasp it, lift it up, and drop it into the red "
      "bowl\" (two cameras: top and side). do_task.py builds exactly this kind of sentence from --pick and --into."),
    p("<b>A warning from someone else's test:</b> a person who tested this model offline on SO-100 setups it had never seen "
      "found its movements look right but land in the wrong place on the table; it did well only on setups that were in its "
      "training data. That is why the code now takes over the carry, and why training on your own demos (section 8) is the big jump."),
    p("4.2 Ranked tasks for your gripper", H2),
    table([["#", "Task", "Command", "Why"],
           ["1", "Pen or marker onto a notebook", "--pick pen --onto notebook", "Best tested score. Grip it in the middle."],
           ["2", "Paper ball into a box", "--pick \"paper ball\" --into box", "Squishes, cannot roll far, big target."],
           ["3", "Tissue or napkin into a basket", "--pick tissue --into basket", "Second-best tested task; soft."],
           ["4", "Fork or spoon onto a plate", "--pick fork --onto plate", "Tested 70; flat and long."],
           ["5", "Lemon, lime or plastic fruit into a bowl", "--pick lemon --into bowl", "The model page's own example."],
           ["6", "Sponge into a bowl or box", "--pick sponge --into box", "Soft, light, easy to see."],
           ["7", "Small plush toy into a basket", "--pick \"plush toy\" --into basket", "Soft; bright colour helps."],
           ["8", "Sock or cloth into a basket", "--pick sock --into basket", "Soft; no exact grip needed."],
           ["9", "Rubber duck into a wide bowl", "--pick \"rubber duck\" --into bowl", "Light; use a wide bowl, not a cup."],
           ["10", "Wrapped candy into a bowl", "--pick candy --into bowl", "Light, flat."],
           ["11", "Sticky-note pad onto a notebook", "--pick \"sticky notes\" --onto notebook", "Flat, parallel sides."],
           ["12", "Eraser or glue stick into a pencil box", "--pick eraser --into \"pencil box\"", "Open, wide box."],
           ["13", "Toy car onto a plate", "--pick \"toy car\" --onto plate", "Flat sides."]],
          [0.7, 4.6, 5.6, 5.9], shade={i: GREEN for i in range(1, 14)}),
    Spacer(1, 4),
    table([["Medium (works sometimes)", "Why it is harder"],
           ["Cube or Lego block into a box or bowl", "Common in uploaded demos, but only 33 in the paper's test; hard and small."],
           ["Ball into a bowl", "Rolls away when dropped badly or pushed."],
           ["Dice or bottle cap into a box", "Small; thin objects may read like 'closed on nothing' (grip_test.py shows it)."],
           ["Chess pawn into a WIDE bowl", "Small, round, tips over. Into a cup is much harder."]], [6.0, 10.8],
          shade={i: YELLOW for i in range(1, 5)}),
    Spacer(1, 4),
    table([["Avoid for the demo", "Why"],
           ["Stacking anything", "20 in the paper's test."],
           ["Tall narrow cups as the target", "The drop has to be precise; a wide container forgives."],
           ["Red Bull can", "Shiny (confuses cameras) and heavy for this gripper (about 270 g full)."],
           ["Picking up a cup or glass", "Big, hard, see-through or shiny."],
           ["Two steps in one sentence", "The model was trained on one pick-and-place per sentence."]], [6.0, 10.8],
          shade={i: RED for i in range(1, 6)}),
    p("4.3 Demo combos that look impressive", H2),
    p("Several easy commands in a row look like one smart task. Teach each container once (teach_drop.py)."),
    code("# \"Clean the desk\"",
         "python do_task.py --pick pen --onto notebook",
         "python do_task.py --pick \"paper ball\" --into box",
         "python do_task.py --pick fork --onto plate",
         "",
         "# \"Breakfast\"",
         "python do_task.py --pick lemon --into bowl",
         "python do_task.py --pick spoon --onto plate"),
    p("4.4 Wording rules", H2),
    *bullets([
        "Use the trained pattern (--pick/--into does it): \"move the arm towards the X, grasp it, lift it up, and drop it into the Y\".",
        "Name colours when there is more than one similar thing: \"the red block\", \"the white pawn\".",
        "One object, one target per command. Keep the same words every time for the same task.",
        "The container word picks the drop pose: --into box uses 'above_box', --into \"red bowl\" uses 'above_bowl'.",
    ]),
]

# ---------------------------------------------------------------- 5 what the code does
story += [
    PageBreak(), p("5. What the code does for you", H1),
    img(FIG / "run_flow.png", 13.2),
    p("5.1 The cloud is updated before the model loads", H2),
    *bullets([
        "Every run first compares the cloud code files (<b>cloud/</b>) on this Mac with the Studio (by checksum) and copies what changed, then "
        "checks the copy arrived. If the model server's own code changed, the old server is stopped and started again with the new code.",
        "It then checks: the same LeRobot version on both sides (0.6.1), the model files exist on the Studio, and no training run is using the GPU.",
        "Only then is the model loaded (2 to 6 minutes the first time; later runs reuse the loaded model and start in seconds). "
        "If a server is still loading from an earlier start, it waits for it instead of starting over.",
    ]),
    p("5.2 Help during the task (assist.py)", H2),
    table([["Feature", "What happens", "Switch off"],
           ["Grab watch", "Compares where the model tells the gripper to go with where it really is: holding, missed, slipped out, let go.", "always on (only watches)"],
           ["Carry (AUTO_DROP)", "Once holding: squeeze fully, lift about 8 to 10 cm, move to the taught 'above_&lt;container&gt;', open. The container never moves, so this half always works, and slow smooth moves keep a weak grip from shaking the object loose.", "AUTO_DROP=0"],
           ["Retry (RETRIES)", "Closed on nothing: open, lift a little, start a fresh try. Also after the object slips out during the carry.", "RETRIES=0"],
           ["Firm grip", "LeRobot lets the gripper drop to the motor's 20% holding force after squeezing for 2 s. 'strong' keeps 35%; the maximum (50%) is unchanged. The original value is saved and restored by 'normal'.", "GRIP_STRENGTH=normal"],
           ["Motor heat", "The check refuses to start above 55 degrees C; a run stops at 60 (motors cut out at about 70).", "always on"],
           ["One connection", "Before, going home reconnected the robot, which switches all motors off for a moment: the arm sagged and dropped what it held. Now one connection for the whole run.", "always on"],
           ["Run log", "outputs/runs/&lt;time&gt;/: events.txt, log.csv, camera pictures per try.", "always on"]],
          [3.1, 10.6, 3.1]),
]

# ---------------------------------------------------------------- 6 improvements
story += [
    PageBreak(), p("6. How to improve a lot (biggest first)", H1),
    table([["#", "Improvement", "You do", "Code does", "Expected effect"],
           ["1", "Train on your own demos", "50 joystick demos (5 spots x 10), about 1 to 1.5 h", "scripts/train_molmo.sh trains on the L4", "Biggest: it learns your table, cameras and gripper. 70-90% on one fixed task is typical, not guaranteed."],
           ["2", "Code carries and drops", "Teach the drop pose once", "Carry after a good grab (done)", "Removes the failures after a good grab."],
           ["3", "Retry missed grabs", "Nothing", "Done (RETRIES=2)", "A miss no longer wastes the run."],
           ["4", "Easy task first", "Paper ball / tissue / pen", "--pick/--into wording (done)", "Probably the quickest gain, no training."],
           ["5", "Rubber on the gripper", "10 minutes with rubber bands", "Firm grip setting (done)", "Fewer slips, especially smooth objects."],
           ["6", "Same scene every time", "Tape marks, same lamp, empty table", "Nothing", "The model reacts to everything it sees."],
           ["7", "Cameras as in the drawings", "Place, focus, check previews", "378 x 378 images (done)", "Better distance judgement, fewer missed grabs."]],
          [0.6, 3.2, 4.1, 3.9, 5.0]),
    Spacer(1, 6),
    p("<b>Why the gripper is weak, in one line:</b> the motor's overload protection (set by LeRobot to protect the motor) cuts the "
      "force after 2 s of squeezing, and smooth plastic fingers slide. Rubber fixes the sliding; GRIP_STRENGTH=strong softens the cut."),
]

# ---------------------------------------------------------------- 7 realsense
story += [
    PageBreak(), p("7. RealSense D435i", H1),
    *bullets([
        "<b>Works, with one extra step.</b> On a Mac, normal apps only get its grey infrared picture. Its colour camera needs "
        "Intel's library, which on macOS 12 and newer needs admin rights (sudo). realsense_bridge.py does that and passes the "
        "colour picture to the other programs, only inside this Mac.",
        "<b>Depth does not help this model:</b> this MolmoAct2 version uses only the colour pictures (its depth option is switched off).",
        "<b>Sharpness does not decide it:</b> every picture is shrunk to 378 x 378 before the model sees it.",
        "<b>The real advantage is placement:</b> it is a separate camera on a cable, so it can stand exactly where the drawings say. "
        "The MacBook webcam forces the whole laptop into that spot. The model page's example also used two RealSense cameras (top + side).",
        "<b>Recommendation:</b> RealSense as the <b>side</b> camera (SIDE_CAM=realsense), iPhone on top. If the iPhone keeps dropping out "
        "or looking blurry, put the RealSense on top instead. Plug it straight into a USB-C port (USB 3), not a hub.",
    ]),
    code("sudo .venv/bin/python realsense_bridge.py --test     # one colour photo: outputs/realsense_test.jpg",
         "sudo .venv/bin/python realsense_bridge.py            # leave running in its own tab, then run do_task.py"),
]

# ---------------------------------------------------------------- 8 training
story += [
    p("8. Training on your own demos (short version)", H1),
    *bullets([
        "<b>Record:</b> python record.py 10, five times, moving the object to 5 spots (10 demos each), same task sentence, same cameras, same light. "
        "Smooth, decisive demos: grip low on the object, lift, carry, open over the container.",
        "<b>Train:</b> scripts/train_molmo.sh (action-expert mode, fits the L4). Turn off auto-sleep on the Studio while it trains. "
        "One round of 3,000 steps: about 2 to 4 hours on the L4 (estimate; the real speed shows in the first minutes).",
        "<b>Use it:</b> set MOLMO_CHECKPOINT in settings.env to the path it prints; do_task.py restarts the server with the new model.",
        "<b>Rounds:</b> expect 2 to 3 rounds (record, train, test 20 tries, add demos where it fails): 1 to 2 days, about 6 to 12 GPU hours. "
        "Full plan in docs/SO101_MolmoAct2_Guide.pdf.",
    ]),
]

# ---------------------------------------------------------------- 9 troubleshooting
story += [
    PageBreak(), p("9. When something goes wrong", H1),
    table([["You see", "Do this"],
           ["the 'arman1' Studio is stopped", "lightning.ai: open arman1, Start, wait for Running, run again."],
           ["Permission denied (publickey)", "Run Lightning's SSH setup command again yourself (it contains your private key)."],
           ["a training run is using the GPU", "Wait for training to finish or stop it (the message shows the command)."],
           ["model not found on the Studio", "MOLMO_CHECKPOINT in settings.env: empty for the base model, or the exact trained path."],
           ["model server crashed", "Tell Claude; the log is ~/molmo/server.log on the Studio."],
           ["robot motors get only X V", "Plug in the robot's 12 V supply (never 12 V on the joystick)."],
           ["motor is hot / STOP: motor too hot", "Switch the robot off for 10 minutes."],
           ["no 'iphone' camera found", "Mac Bluetooth and Wi-Fi on, iPhone locked and close; wait a few seconds; run again."],
           ["no picture from the RealSense", "Start sudo .venv/bin/python realsense_bridge.py in another tab first."],
           ["auto drop off for this run", "Teach the drop pose it names: python teach_drop.py box."],
           ["STOP: the robot cannot reach its target", "Something blocks the arm. Clear it; check the cable is not caught."],
           ["missed again and again", "Object position, light and camera views first; then send Claude the outputs/runs folder."],
           ["holding detected but nothing in the gripper", "Thin object or the jaws touched the table: run grip_test.py and send the log to Claude."],
           ["object slips during the carry", "Rubber on the fingers; lighter object; GRIP_STRENGTH=strong."]], [6.0, 10.8]),
    Spacer(1, 10),
    p("Sources", H2),
    *bullets([
        "MolmoAct2 paper, arXiv 2605.02881 (SO-100/101 zero-shot table, training data): arxiv.org/html/2605.02881",
        "Model page: huggingface.co/allenai/MolmoAct2-SO100_101 (example task, two cameras, depth disabled)",
        "Community out-of-the-box test: mcobzarenco-fontaine-blog.static.hf.space/posts/2026-08-10-molmoact2-oob-results.html",
        "Feetech STS memory table (overload and holding torque): aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis",
        "Intel RealSense macOS installation (sudo on macOS 12+): dev.realsenseai.com/?p=40689",
        "pyrealsense2 for macOS: github.com/cansik/pyrealsense2-macosx",
    ]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.6 * cm, title="SO-101 + MolmoAct2 task playbook", author="Claude for Arman")


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.9 * cm, 1.0 * cm, "SO-101 + MolmoAct2 task playbook, 14 Oct 2026")
    canvas.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"built {OUT}")
