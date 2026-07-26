#!/usr/bin/env python
"""Convert recipe .py files (jupytext percent) to executed .ipynb notebooks.

Usage:
    python scripts/build_notebooks.py                       # all recipes
    python scripts/build_notebooks.py 00_fundamentals/01_quickstart [...]
    python scripts/build_notebooks.py --no-execute          # convert only
    python scripts/build_notebooks.py --dir notebooks_ja    # Japanese recipes

Each .py under the recipe directory becomes a sibling .ipynb, executed top to
bottom with the repo root as working directory (so `import cookbook_utils`
works). Japanese recipes live in notebooks_ja/ and share the English code, so
build one tree at a time - both write to the same data/dbs/ directories.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jupytext
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def build(py_path: Path, execute: bool = True, timeout: int = 600) -> tuple[bool, float]:
    nb = jupytext.read(py_path)
    nb.metadata.setdefault("kernelspec", {}).update(
        {"name": "python3", "language": "python", "display_name": "Python 3"}
    )
    t0 = time.time()
    ok = True
    if execute:
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        )
        try:
            client.execute()
        except Exception as e:  # noqa: BLE001 - report and keep going
            ok = False
            print(f"    EXECUTION FAILED: {type(e).__name__}: {str(e)[:2000]}")
    ipynb = py_path.with_suffix(".ipynb")
    jupytext.write(nb, ipynb, fmt="ipynb")
    return ok, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="*", help="e.g. 00_fundamentals/01_quickstart")
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dir", default="notebooks", help="recipe tree: notebooks or notebooks_ja")
    args = ap.parse_args()

    notebooks = ROOT / args.dir
    if args.recipes:
        targets = [notebooks / (r if r.endswith(".py") else r + ".py") for r in args.recipes]
    else:
        targets = sorted(notebooks.glob("*/*.py"))

    failures = []
    for py in targets:
        if not py.exists():
            print(f"missing: {py}")
            failures.append(py)
            continue
        rel = py.relative_to(notebooks)
        print(f"[{rel}]")
        ok, dt = build(py, execute=not args.no_execute, timeout=args.timeout)
        print(f"    {'ok' if ok else 'FAILED'} in {dt:.1f}s")
        if not ok:
            failures.append(py)

    if failures:
        print(f"\n{len(failures)} failed: " + ", ".join(str(f.relative_to(notebooks)) for f in failures))
        return 1
    print(f"\nall {len(targets)} notebooks built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
