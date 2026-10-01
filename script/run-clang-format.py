#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==22.1.8"]
# ///
"""Apply clang-format -i to every tracked .cc/.hh in the repo.

Locates the repo from this script's path so it works no matter where
it's invoked from. Vendored third-party headers are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXTENSIONS = {".cc", ".hh"}
SKIP = {
    Path("test/doctest.h"),
}


def tracked_sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw in out.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8"))
        if rel.suffix not in EXTENSIONS:
            continue
        if rel in SKIP:
            continue
        files.append(rel)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only report unformatted files, exit 1 if any")
    args = parser.parse_args()

    clang_format = shutil.which("clang-format")
    if clang_format is None:
        print("error: clang-format not on PATH", file=sys.stderr)
        return 1

    files = tracked_sources()
    if not files:
        print("no .cc/.hh files to format")
        return 0

    # --check reports unformatted files (and fails) instead of rewriting them; used by CI.
    mode = ["--dry-run", "--Werror"] if args.check else ["-i"]
    print(f"{'checking' if args.check else 'formatting'} {len(files)} file(s) with {clang_format}")
    # Chunk to stay well under Windows' command-line length limit.
    CHUNK = 64
    ok = True
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        result = subprocess.run(
            [clang_format, *mode, "--", *(str(p) for p in chunk)],
            cwd=REPO_ROOT,
            check=not args.check,
        )
        ok = ok and result.returncode == 0
    if not ok:
        print("error: some files are not clang-formatted; run script/run-clang-format.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
