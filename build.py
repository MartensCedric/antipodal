#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def load_presets() -> tuple[list[str], list[str]]:
    """Return (configure_preset_names, build_preset_names) from CMakePresets.json."""
    presets_file = ROOT / "CMakePresets.json"
    data = json.loads(presets_file.read_text())
    configure = [p["name"] for p in data.get("configurePresets", []) if not p.get("hidden", False)]
    build = [p["name"] for p in data.get("buildPresets", []) if not p.get("hidden", False)]
    return configure, build


def default_preset() -> str:
    """Default to the relwithdebinfo configure preset matching the host platform."""
    if sys.platform == "win32":
        return "x64-windows-msvc-ninja-relwithdebinfo"
    return "x64-linux-clang-ninja-relwithdebinfo"


def default_vcpkg_toolchain() -> Path:
    if sys.platform == "win32":
        return Path("C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    return Path.home() / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake"


def find_vcvarsall() -> Path | None:
    """Locate vcvarsall.bat via vswhere — needed when invoking cl from a non-VS shell."""
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True, check=False,
    )
    install_path = result.stdout.strip().splitlines()
    if not install_path:
        return None
    candidate = Path(install_path[0]) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    return candidate if candidate.is_file() else None


def msvc_env(arch: str = "x64") -> dict[str, str]:
    """Return the environment after sourcing vcvarsall.bat for `arch` (Windows only)."""
    vcvarsall = find_vcvarsall()
    if vcvarsall is None:
        sys.exit("could not locate vcvarsall.bat — install Visual Studio with the C++ workload, "
                 "or run this script from a Developer Command Prompt.")
    # Use a sentinel to delimit the environment dump from any chatter vcvarsall prints.
    sentinel = "===ANTIPODAL_ENV==="
    # Feed cmd via stdin to sidestep the bash↔cmd quote-escaping mess.
    script = f'@call "{vcvarsall}" {arch} >NUL\r\n@echo {sentinel}\r\n@set\r\n@exit\r\n'
    result = subprocess.run(["cmd"], input=script, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"vcvarsall.bat failed (exit {result.returncode}):\n"
                 f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    lines = result.stdout.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == sentinel) + 1
    except StopIteration:
        sys.exit(f"vcvarsall.bat did not produce the expected environment output.\n"
                 f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    env: dict[str, str] = {}
    for line in lines[start:]:
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env


def build(preset: str, vcpkg_toolchain: Path | None) -> Path:
    build_dir = ROOT / "build" / preset

    configure = ["cmake", "--preset", preset]
    if vcpkg_toolchain is not None:
        if not vcpkg_toolchain.is_file():
            sys.exit(f"vcpkg toolchain file not found: {vcpkg_toolchain}")
        configure.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}")

    # MSVC presets need the VS dev environment for `cl` to resolve.
    env: dict[str, str] | None = None
    if sys.platform == "win32" and "msvc" in preset and "VCINSTALLDIR" not in os.environ:
        env = msvc_env("x64")

    subprocess.run(configure, cwd=ROOT, check=True, env=env)
    subprocess.run(["cmake", "--build", str(build_dir)], cwd=ROOT, check=True, env=env)
    return build_dir


def run_tests(build_dir: Path) -> None:
    exe_name = "antipodal-tests.exe" if sys.platform == "win32" else "antipodal-tests"
    candidates = sorted(build_dir.rglob(exe_name))
    if not candidates:
        sys.exit(f"test binary {exe_name} not found under {build_dir}")
    subprocess.run([str(candidates[0])], cwd=ROOT, check=True)


def main() -> None:
    configure_presets, _ = load_presets()
    parser = argparse.ArgumentParser(description="Configure and build antipodal via CMake presets.")
    parser.add_argument(
        "-p", "--preset",
        choices=configure_presets,
        default=default_preset(),
        help=f"CMake configure preset (default: {default_preset()}).",
    )
    parser.add_argument(
        "--vcpkg-toolchain",
        type=Path,
        default=default_vcpkg_toolchain(),
        help="Path to vcpkg.cmake toolchain file "
             "(default: C:/vcpkg/... on Windows, ~/vcpkg/... elsewhere).",
    )
    parser.add_argument(
        "--no-vcpkg",
        action="store_true",
        help="Configure without a vcpkg toolchain file.",
    )
    parser.add_argument(
        "-t", "--run-tests",
        action="store_true",
        help="Run the antipodal-tests binary after building.",
    )
    args = parser.parse_args()

    build_dir = build(args.preset, None if args.no_vcpkg else args.vcpkg_toolchain)
    if args.run_tests:
        run_tests(build_dir)


if __name__ == "__main__":
    main()
