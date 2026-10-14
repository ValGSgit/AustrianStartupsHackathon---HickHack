"""Shell game stage runner: big status screen, robot shuffles, judge guesses, camera verifies.

    .venv\\Scripts\\python shellgame\\show.py                        # SPACE = play, 1-4 = guess, ESC = stop (arm holds), q = quit
    .venv\\Scripts\\python shellgame\\show.py --moves 5 --speed 1.3 --fullscreen
    .venv\\Scripts\\python shellgame\\show.py --fake --no-vision --moves 3 --auto   # no hardware, one game, no keys

The robot runs in a worker thread; the window stays live. ESC sets a stop flag: the next servo command is not sent
and the arm holds where it is (torque stays on). q goes home slowly and switches the motors off.
Each game is logged to evals/shell_log.csv.
"""

import argparse
import csv
import datetime as dt
import queue
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import arm as A  # noqa: E402
import stage  # noqa: E402
from stage import AMBER, FG, GREEN, RED  # noqa: E402

LOG = HERE.parent / "evals" / "shell_log.csv"


class Stopped(Exception):
    pass


class Show:
    def __init__(self, a):
        self.a = a
        self.stop = threading.Event()
        self.keys: queue.Queue = queue.Queue()   # keys the worker waits for (guess, SPACE on pause)
        self.ui = ("Ready", FG, "SPACE = play   q = quit")
        self.frame = None                        # latest camera frame (main thread writes)
        self.frame_n = 0
        self.worker: threading.Thread | None = None

    # ------------------------------------------------ helpers used by the worker
    def set(self, title, color, sub=""):
        self.ui = (title, color, sub)

    def say(self, text):
        print(f"[say] {text}", flush=True)
        try:
            stage.say(text)
        except Exception as e:  # speech is nice to have, never fatal
            print(f"(speech failed: {e})")

    def fresh_frame(self, timeout=2.0):
        """A frame taken after now (the arm has stopped moving), or None."""
        n, t0 = self.frame_n, time.time()
        while self.frame_n < n + 2 and time.time() - t0 < timeout:
            time.sleep(0.03)
        return self.frame

    def wait_key(self, allowed):
        if self.a.auto and " " in allowed:  # testing: the pause continues by itself
            time.sleep(1)
            return " "
        while True:
            if self.stop.is_set():
                raise Stopped
            try:
                k = self.keys.get(timeout=0.1)
            except queue.Empty:
                continue
            if k in allowed:
                return k

    # ------------------------------------------------ the game (worker thread)
    def check_cups(self, s, problems_log):
        import game as G
        import motions as M
        import vision as V
        M.go_ready(self.arm, self.poses)  # out of the camera's view
        for attempt in (1, 2):
            frame = self.fresh_frame()
            if frame is None:
                ok, problems = False, ["no camera frame"]
            else:
                ok, problems = V.check_layout(frame, self.setup, G.empty_slot(s))
            if ok:
                return
            problems_log += problems
            print("camera:", "; ".join(problems), flush=True)
            states = V.slot_states(frame, self.setup) if frame is not None else []
            off = [i for i, st in enumerate(states, 1) if i != G.empty_slot(s) and st["off_center"]]
            if attempt == 2 or not off:  # a missing or extra cup cannot be fixed by a nudge
                break
            self.set("Fixing a cup", AMBER, "\n".join(problems[:3]))
            self.say("One cup is off its mark. Fixing it.")
            for i in off:
                M.nudge(self.arm, self.poses, i)
            M.go_ready(self.arm, self.poses)
        self.set("Fix the cups", RED, "\n".join(problems[:3]) + "\nput them on their marks - SPACE to continue")
        self.say("Please fix the cups")
        self.wait_key({" "})

    def play(self):
        import game as G
        import motions as M
        a, rng = self.a, self.rng
        problems: list[str] = []
        s = G.new_state(rng.choice([1, 2, 3]))
        moves = G.random_moves(s, a.moves, rng)
        start = G.ball_slot(s)
        self.set("Watch the ball", AMBER, f"the ball is under cup {start}")
        self.say(f"Watch closely. The ball is under cup {start}.")
        M.go_ready(self.arm, self.poses)
        M.show_ball(self.arm, self.poses, start)
        for k, move in enumerate(moves, 1):
            self.set(f"Shuffling {k}/{len(moves)}", AMBER, "keep your eyes on the ball")
            M.move_cup(self.arm, self.poses, *move)
            s = G.apply(s, move)
            print(f"move {k}: {move[0]} -> {move[1]}   ball in slot {G.ball_slot(s)}", flush=True)
            if self.vision:
                self.check_cups(s, problems)
        M.go_ready(self.arm, self.poses)
        ball = G.ball_slot(s)
        empty = G.empty_slot(s)
        cups = [i for i in range(1, 5) if i != empty]
        self.set("Which cup?", FG, f"press {', '.join(map(str, cups))}")
        self.say("Which cup is the ball under?")
        guess = 1 if a.auto else int(self.wait_key({str(i) for i in cups}))
        self.set(f"You said cup {guess}", AMBER, "let's see...")
        verified, seen = self.reveal(ball)
        if guess == ball:
            line = "You got me."
        else:
            line = f"I told you. It was under cup {ball}."
        if verified is None:
            self.set(f"Cup {ball}", AMBER, f"{line}\n(no camera check)")
        elif verified:
            self.set("VERIFIED", GREEN, f"ball under cup {ball} - camera saw it\n{line}")
        else:
            self.set("MISSED", RED, f"camera did not see the ball under cup {ball}\n{line}")
            line += " But my camera did not see the ball."
        self.say(line)
        log(a, ball, guess, verified, problems + ([] if verified is not False else [f"ball not seen ({seen} px)"]))

    def reveal(self, slot):
        """Lift the cup; meanwhile watch for the ball near that slot. Returns (verified or None, max ball px)."""
        import motions as M
        import vision as V
        if not self.vision:
            M.reveal(self.arm, self.poses, slot)
            return None, 0
        done, err = threading.Event(), []

        def run():
            try:
                M.reveal(self.arm, self.poses, slot)
            except Exception as e:
                err.append(e)
            finally:
                done.set()
        threading.Thread(target=run, daemon=True).start()
        hits, best, last_n = 0, 0, -1
        while not done.is_set():
            if self.frame_n != last_n and self.frame is not None:
                last_n = self.frame_n
                ok, info = V.ball_visible(self.frame, self.setup, slot)
                hits, best = hits + ok, max(best, info["px"])
            time.sleep(0.03)
        if err:
            raise err[0]
        return hits >= 3, best

    def run_game(self):
        try:
            self.play()
            self.result = "done"
        except Stopped:
            self.set("STOPPED", RED, "arm holds its pose\nSPACE = new game   q = quit (arm goes home, motors off)")
            self.say("Stopped")
            self.result = "stopped"
        except Exception as e:
            print(f"game error: {e!r}", flush=True)
            self.set("ERROR", RED, f"{str(e)[:60]}\nSPACE = new game   q = quit")
            self.result = "error"

    # ------------------------------------------------ main thread: window, camera, keys
    def main(self):
        a = self.a
        A.SPEED = a.speed
        import motions as M
        self.poses = A.load_poses()
        if a.fake and M.missing_poses(self.poses):
            self.poses = M.fake_poses()
            print("fake run: using generated poses (poses.json missing or incomplete)")
        missing = M.missing_poses(self.poses)
        if missing:
            sys.exit(f"poses not taught: {', '.join(missing)}\nrun: .venv\\Scripts\\python shellgame\\teach.py teach")
        cap = None
        self.vision = not a.no_vision and not a.mock_camera
        if self.vision:
            import selfcheck
            import vision as V
            self.setup = V.load_setup()
            cap = selfcheck.open_camera()
        seed = a.seed if a.seed is not None else random.randrange(10**6)
        self.rng = random.Random(seed)
        print(f"seed {seed}  speed {a.speed}  moves {a.moves}  vision {'on' if self.vision else 'off'}")

        self.arm = A.connect(fake=a.fake)
        send = self.arm.send_action

        def guarded(action):  # ESC: stop issuing commands, the arm holds its last goal
            if self.stop.is_set():
                raise Stopped
            if a.fake:
                time.sleep(1 / A.FPS / 4)  # fake: 4x real time, so the screen can be watched
            return send(action)
        self.arm.send_action = guarded

        cv2.namedWindow(stage.WIN, cv2.WINDOW_NORMAL)
        if a.fullscreen:
            cv2.setWindowProperty(stage.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        auto_started, auto_done_at = False, None
        try:
            while True:
                if cap is not None:
                    ok, f = cap.read()
                    if ok:
                        self.frame, self.frame_n = f, self.frame_n + 1
                img = self.frame
                if a.mock_camera:
                    img = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(img, "mock camera", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
                if img is not None and self.vision:
                    import vision as V
                    img = V.annotate(img, self.setup, V.slot_states(img, self.setup))
                title, color, sub = self.ui
                k = stage.show(stage.screen(title, color, sub, img), 30)
                busy = self.worker is not None and self.worker.is_alive()
                if a.auto and not auto_started:
                    k, auto_started = ord(" "), True
                if a.auto and auto_started and not busy and self.worker is not None:
                    auto_done_at = auto_done_at or time.time()
                    if time.time() - auto_done_at > 1.5:
                        break
                if k == 27 and busy:
                    self.stop.set()
                    self.set("STOPPING", RED, "arm holds")
                elif k == ord("q"):
                    self.stop.set()
                    if self.worker:
                        self.worker.join(5)
                    break
                elif k == ord(" ") and not busy:
                    self.stop.clear()
                    self.keys = queue.Queue()
                    self.worker = threading.Thread(target=self.run_game, daemon=True)
                    self.worker.start()
                elif k != 255 and busy:
                    self.keys.put(chr(k))
        finally:
            self.stop.clear()
            self.arm.send_action = send
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print("going home, motors off...", flush=True)
            A.release(self.arm)
        return getattr(self, "result", None)


def log(a, ball, guess, verified, problems):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "moves", "speed", "ball_slot", "guess", "verified", "problems"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), a.moves, a.speed, ball, guess,
                    "" if verified is None else int(verified), " | ".join(dict.fromkeys(problems))])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="no arm (FakeArm), generated poses if poses.json is incomplete")
    p.add_argument("--no-vision", action="store_true", help="no camera: no layout checks, no ball verification")
    p.add_argument("--mock-camera", action="store_true", help="black frame on screen instead of the camera (no checks)")
    p.add_argument("--moves", type=int, default=7)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--auto", action="store_true", help="testing: auto SPACE, guess 1, exit after one game")
    result = Show(p.parse_args()).main()
    print(f"result: {result}")


if __name__ == "__main__":
    main()
