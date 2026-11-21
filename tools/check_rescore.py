"""Check that a code change leaves the scores of saved runs unchanged.

    python tools/check_rescore.py [runs] [--against HEAD]

Rescores every saved run (a folder with meta.json and result.json under ``runs``) twice, with the
code at git revision ``--against`` (default HEAD) and with the working tree, and compares the two
result.json files as parsed JSON. Saved runs are never touched: each run's logs (meta.json and the
*.jsonl streams, not the frame bank) are copied to a temporary folder first. Prints each run whose
scores differ, with the first differing keys, and exits 1 if any did. Runs that one side cannot
rescore (e.g. a terrain file that is gone) are listed as errors.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, List

REPO = Path(__file__).resolve().parents[1]

SCORER = """
import json, sys, traceback
sys.path.insert(0, sys.argv[1])
from vivid2.runner import score_run_dir
for d in json.load(open(sys.argv[2], encoding="utf-8")):
    try:
        score_run_dir(d)
    except Exception as e:
        open(d + "/error.txt", "w", encoding="utf-8").write(repr(e))
"""


def diff(a: Any, b: Any, path: str = "", out: List[str] = None, limit: int = 5) -> List[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'only before' if k not in b else 'only after'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} -> {str(b)[:60]}")
    return out


def _copy_logs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and (f.suffix == ".jsonl" or f.name == "meta.json"):
            shutil.copy2(f, dst / f.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="?", default=str(REPO / "runs"))
    ap.add_argument("--against", default="HEAD", help="git revision to compare with (default HEAD)")
    args = ap.parse_args(argv)
    dirs = sorted(p.parent for p in Path(args.runs).rglob("result.json") if (p.parent / "meta.json").exists())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "base"
        archive = subprocess.run(["git", "-C", str(REPO), "archive", args.against, "src"], capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(base)
        sides = {"before": base / "src", "after": REPO / "src"}
        lists = {}
        for side in sides:
            run_dirs = []
            for i, d in enumerate(dirs):
                _copy_logs(d, tmp / side / f"r{i}")
                run_dirs.append(str(tmp / side / f"r{i}"))
            lists[side] = tmp / f"{side}.json"
            lists[side].write_text(json.dumps(run_dirs), encoding="utf-8")
        for side, src in sides.items():
            subprocess.run([sys.executable, "-c", SCORER, str(src), str(lists[side])], check=True, cwd=tmp)
        same = changed = errors = 0
        for i, d in enumerate(dirs):
            a, b = tmp / "before" / f"r{i}", tmp / "after" / f"r{i}"
            err = [(s, (p / "error.txt").read_text(encoding="utf-8")) for s, p in (("before", a), ("after", b))
                   if (p / "error.txt").exists()]
            if err:
                errors += 1
                print(f"ERROR {d}: " + "; ".join(f"{s}: {e}" for s, e in err))
                continue
            ra = json.loads((a / "result.json").read_text(encoding="utf-8"))
            rb = json.loads((b / "result.json").read_text(encoding="utf-8"))
            if ra == rb:
                same += 1
            else:
                changed += 1
                print(f"CHANGED {d}: " + "; ".join(diff(ra, rb)))
    print(f"{len(dirs)} runs: {same} identical, {changed} changed, {errors} errors (vs {args.against})")
    return 1 if changed or errors else 0


if __name__ == "__main__":
    sys.exit(main())
