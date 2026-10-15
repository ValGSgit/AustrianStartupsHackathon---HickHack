"""Download the official SO-101 MuJoCo model (TheRobotStudio/SO-ARM100, Apache-2.0) at a pinned commit.

    python sim/fetch_model.py            -> sim/model/so101_new_calib.xml + sim/model/assets/*.stl
    python sim/fetch_model.py --lowpoly  -> also a decimated copy (needs: pip install trimesh fast-simplification)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"  # 2026-10-06
REPO = "TheRobotStudio/SO-ARM100"
SRC = "Simulation/SO101"
FILES = ["so101_new_calib.xml"]
DEST = Path(__file__).resolve().parent / "model"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def make_lowpoly(reduction: float = 0.92) -> None:
    """Decimated meshes for fast CPU rendering (Docker). Collision uses convex hulls, so physics is unchanged."""
    import fast_simplification
    import numpy as np
    import trimesh

    src, dst = DEST / "assets", DEST / "assets_lowpoly"
    dst.mkdir(exist_ok=True)
    before = after = 0
    for f in sorted(src.glob("*.stl")):
        m = trimesh.load(f, force="mesh")
        v, faces = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces, target_reduction=reduction)
        trimesh.Trimesh(v, faces).export(dst / f.name)
        before, after = before + len(m.faces), after + len(faces)
    xml = (DEST / "so101_new_calib.xml").read_text().replace('meshdir="assets"', 'meshdir="assets_lowpoly"')
    (DEST / "so101_new_calib_lowpoly.xml").write_text(xml)
    print(f"low-poly meshes: {before} -> {after} triangles")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowpoly", action="store_true", help="also build decimated meshes for fast CPU rendering")
    args = ap.parse_args()
    raw = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{SRC}"
    listing = json.loads(get(f"https://api.github.com/repos/{REPO}/contents/{SRC}/assets?ref={COMMIT}"))
    meshes = [f"assets/{e['name']}" for e in listing if e["name"].endswith(".stl")]
    for rel in FILES + meshes:
        out = DEST / rel
        if out.exists() and out.stat().st_size > 0:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(get(f"{raw}/{rel}"))
        print("fetched", rel, file=sys.stderr)
    print(f"SO-101 model ready in {DEST} ({len(meshes)} meshes, commit {COMMIT[:8]})")
    if args.lowpoly:
        make_lowpoly()


if __name__ == "__main__":
    main()
