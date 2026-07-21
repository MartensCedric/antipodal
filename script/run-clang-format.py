#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["clang-format==22.1.7"]
# ///
"""Apply clang-format -i to every tracked .cc/.hh in the repo.

Locates the repo from this script's path so it works no matter where
it's invoked from. Vendored third-party headers are skipped.
"""

from __future__ import annotations

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
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        print("error: clang-format not on PATH", file=sys.stderr)
        return 1

    files = tracked_sources()
    if not files:
        print("no .cc/.hh files to format")
        return 0

    print(f"formatting {len(files)} file(s) with {clang_format}")
    # Chunk to stay well under Windows' command-line length limit.
    CHUNK = 64
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        subprocess.run(
            [clang_format, "-i", "--", *(str(p) for p in chunk)],
            cwd=REPO_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
