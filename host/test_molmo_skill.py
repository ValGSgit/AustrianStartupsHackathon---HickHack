"""Hardware-free test of the molmo_pick skill: a fake Worker (speaks the RobotSession WebSocket protocol) and a
stub do_task.py stand in for Cloudflare, the Lightning GPU, the arm and the cameras.

    python host/test_molmo_skill.py        # needs: websockets, numpy, opencv (the agent's own imports)

Checks: the skill is announced only when [molmo] dir is set, enabled = false hides a skill, the router's task
text reaches do_task.py, success comes from its RESULT line, STOP sends Ctrl+C (SIGINT) so it can go home.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUB = r'''
import json, signal, sys, time
from pathlib import Path
here = Path(__file__).parent
args = sys.argv[1:]
(here / "args.json").write_text(json.dumps(args))
def on_int(*_):
    print("6. Back to the resting pose", flush=True)
    (here / "went_home").write_text("yes")
    print("   RESULT: stopped with Ctrl+C (1 s, 1 try)", flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, on_int)
print("  OK    Studio reachable (stub)", flush=True)
task = args[-1] if not args[-1].replace(".", "").isdigit() else ""
if "slow" in task:
    time.sleep(0.7)  # the agent rate-limits log lines to one per 0.5 s
    print("stub: driving", flush=True)
    time.sleep(60)
if "fail" in task:
    print("  FAIL  robot not found\n        -> plug in the USB cable", flush=True)
    sys.exit(1)
print("\n   RESULT: success: carried the object to 'above_cup' and let go (12 s, 1 try)", flush=True)
'''

TASK = "move the arm towards the white pawn, grasp it, lift it up, and drop it into the cup"


def load_agent(molmo: Path):
    import robot

    cfg = robot.load_config()
    cfg["molmo"] = {"dir": str(molmo), "python": sys.executable}
    cfg["skills"]["wave"]["enabled"] = False
    robot.load_config = lambda: cfg  # robot_agent reads its config at import
    sys.path.insert(0, str(ROOT / "host"))
    import robot_agent

    robot_agent.RealBackend.snapshot = lambda self: None  # no camera in a test (and no camera permission)
    return robot_agent


async def fake_worker(ra, molmo: Path) -> list[str]:
    import websockets

    results: list[str] = []
    done = asyncio.get_running_loop().create_future()

    async def until(ws, kind, pred=lambda m: True):
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == kind and pred(m):
                return m

    async def handler(ws):
        hello = json.loads(await ws.recv())
        names = [s["name"] for s in hello["skills"]]
        assert "molmo_pick" in names, names
        assert next(s for s in hello["skills"] if s["name"] == "molmo_pick")["instruction"] is True
        assert "wave" not in names, "enabled = false must hide the skill"
        results.append(f"hello: {names}")

        await ws.send(json.dumps({"type": "run", "skill": "molmo_pick", "task": TASK}))
        d = await until(ws, "done")
        args = json.loads((molmo / "args.json").read_text())
        assert args == ["--seconds", "60", TASK], args
        assert d["ok"] is True and d["detail"].startswith("success"), d
        results.append(f"run: ok={d['ok']} detail={d['detail']!r} args={args}")

        await ws.send(json.dumps({"type": "run", "skill": "molmo_pick", "task": "fail please"}))
        d = await until(ws, "done")
        assert d["ok"] is False and "exit 1" in d["detail"], d
        results.append(f"failed check: ok={d['ok']} detail={d['detail']!r}")

        await ws.send(json.dumps({"type": "run", "skill": "molmo_pick", "task": "slow task"}))
        await until(ws, "log", lambda m: "driving" in m["text"])
        await ws.send(json.dumps({"type": "stop"}))
        d = await until(ws, "done")
        assert d["ok"] is False and d["detail"] == "stopped", d
        assert (molmo / "went_home").exists(), "STOP must reach do_task.py as SIGINT"
        results.append(f"stop: ok={d['ok']} detail={d['detail']!r} went_home=yes")
        done.set_result(True)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        a = type("A", (), {"url": f"http://127.0.0.1:{port}", "token": "t", "sim": False, "show": False})
        agent = asyncio.create_task(ra.main_async(a))
        try:
            await asyncio.wait_for(done, 60)
        finally:
            agent.cancel()
    return results


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        molmo = Path(tmp)
        (molmo / "do_task.py").write_text(STUB)
        ra = load_agent(molmo)

        for line in asyncio.run(fake_worker(ra, molmo)):
            print("PASS", line)

        ra.MOLMO = {"dir": ""}
        names = [s["name"] for s in ra.RealBackend.skills(object.__new__(ra.RealBackend))]
        assert "molmo_pick" not in names, names
        print("PASS no [molmo] dir: molmo_pick not announced", names)


if __name__ == "__main__":
    main()
