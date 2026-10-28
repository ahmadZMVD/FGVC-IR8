#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launcher.py - FGVC-IR8 v9.5 Zero-Friction Bootstrap, Dual-Checkpoint & Fast-Boot Manager
========================================================================================
- First run (--setup): checks global packages; creates isolated .venv only if needed,
  installs CPU-optimized wheels, verifies/downloads BOTH model checkpoints
  (`fgvc_ir8_laptop_standalone_fp16.pt` AND `best.pt`) from GitHub Releases for the
  v9.5 Zero-Copy Dual-Head Ensemble, and writes .env_verified.json.
- Self-test mode (--selftest): runs the 17-stage verification suite in engine.py.
- Subsequent runs (--fast): skips setup checks and starts the HTTP server in < 0.8s.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STAMP_FILE = os.path.join(ROOT_DIR, ".env_verified.json")
VENV_DIR = os.path.join(ROOT_DIR, ".venv")
VENV_PY = (
    os.path.join(VENV_DIR, "Scripts", "python.exe")
    if os.name == "nt"
    else os.path.join(VENV_DIR, "bin", "python")
)

CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
STANDALONE_WEIGHTS = os.path.join(CKPT_DIR, "fgvc_ir8_laptop_standalone_fp16.pt")
BEST_HEAD_WEIGHTS = os.path.join(CKPT_DIR, "best.pt")

RELEASE_BASE_URL = "https://github.com/ahmadZMVD/FGVC-IR8/releases/download/v1.0.0"
WEIGHTS_URL = os.environ.get(
    "FGVC_WEIGHTS_URL",
    f"{RELEASE_BASE_URL}/fgvc_ir8_laptop_standalone_fp16.pt",
)
BEST_PT_URL = os.environ.get(
    "FGVC_BEST_PT_URL",
    f"{RELEASE_BASE_URL}/best.pt",
)


def has_required_packages(py_executable: str) -> bool:
    code = (
        "import torch, torchvision, numpy, PIL; "
        "print('OK' if hasattr(torch, '__version__') else 'FAIL')"
    )
    try:
        res = subprocess.run(
            [py_executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        return "OK" in res.stdout
    except Exception:
        return False


def download_file(url: str, dest_path: str) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    print(f"[DOWNLOAD] Fetching weights from: {url}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "FGVC-IR8-Launcher/9.5"}
        )
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
            length = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            t0 = time.time()
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if length > 0:
                    pct = (downloaded / length) * 100
                    speed = (downloaded / (1024 * 1024)) / max(
                        time.time() - t0, 0.001
                    )
                    sys.stdout.write(
                        f"\r  Progress: {pct:5.1f}% ({downloaded / (1024*1024):.1f}/{length / (1024*1024):.1f} MB) [{speed:.1f} MB/s]"
                    )
                    sys.stdout.flush()
            print()
        os.replace(temp_path, dest_path)
        return True
    except Exception as err:
        print(f"\n[ERROR] Failed to download {url}: {err}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def ensure_weights() -> bool:
    primary_ok = (
        os.path.exists(STANDALONE_WEIGHTS)
        and os.path.getsize(STANDALONE_WEIGHTS) > 50 * 1024 * 1024
    )
    if primary_ok:
        print(f"[WEIGHTS] Standalone FP16 weights present ({STANDALONE_WEIGHTS}).")
    else:
        print("[WEIGHTS] Downloading standalone FP16 weights (172 MB)...")
        primary_ok = download_file(WEIGHTS_URL, STANDALONE_WEIGHTS)

    best_ok = (
        os.path.exists(BEST_HEAD_WEIGHTS)
        and os.path.getsize(BEST_HEAD_WEIGHTS) > 5 * 1024 * 1024
    )
    if best_ok:
        print(f"[WEIGHTS] Auxiliary peak-validation head present ({BEST_HEAD_WEIGHTS}).")
    else:
        print("[WEIGHTS] Downloading auxiliary peak-validation head best.pt (16.1 MB)...")
        best_ok = download_file(BEST_PT_URL, BEST_HEAD_WEIGHTS)

    return primary_ok or best_ok


def setup_environment() -> int:
    t0 = time.time()
    print("=" * 68)
    print("  FGVC-IR8 v9.5 Zero-Friction Bootstrap & Fast-Boot Manager")
    print("=" * 68)

    active_py = sys.executable
    if has_required_packages(active_py):
        print(f"[ENV] Required packages verified in Python ({active_py}).")
    else:
        if not os.path.exists(VENV_PY):
            print("[ENV] Creating local virtual environment (.venv)...")
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

        active_py = VENV_PY
        if not has_required_packages(active_py):
            print("[ENV] Installing required CPU packages (torch, torchvision, numpy, pillow)...")
            subprocess.check_call([active_py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
            req_path = os.path.join(ROOT_DIR, "requirements.txt")
            if os.path.exists(req_path):
                subprocess.check_call([active_py, "-m", "pip", "install", "-r", req_path])
            else:
                subprocess.check_call([
                    active_py, "-m", "pip", "install",
                    "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu",
                ])
                subprocess.check_call([active_py, "-m", "pip", "install", "numpy", "Pillow"])

    if not ensure_weights():
        return 1

    stamp_data = {
        "version": "9.5",
        "verified": True,
        "python_executable": active_py,
        "weights_path": STANDALONE_WEIGHTS,
        "best_head_path": BEST_HEAD_WEIGHTS,
        "timestamp": time.time(),
        "setup_duration_s": round(time.time() - t0, 2),
    }
    with open(STAMP_FILE, "w", encoding="utf-8") as f:
        json.dump(stamp_data, f, indent=2)

    print(f"[OK] Environment verified and cached in {time.time() - t0:.2f}s.")
    return 0


def fast_boot() -> int:
    if not os.path.exists(STANDALONE_WEIGHTS):
        if os.path.exists(STAMP_FILE):
            os.remove(STAMP_FILE)
        print("[WARN] Model weights missing; running setup check...")
        rc = setup_environment()
        if rc != 0:
            return rc

    import app
    app.main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FGVC-IR8 v9.5 Launcher")
    parser.add_argument("--setup", action="store_true", help="Run first-time environment verification")
    parser.add_argument("--fast", action="store_true", help="Fast-boot web server directly")
    parser.add_argument("--selftest", action="store_true", help="Run 17-stage engine verification suite")
    parser.add_argument("--reset", action="store_true", help="Reset cached environment stamp")
    args = parser.parse_args()

    if args.reset and os.path.exists(STAMP_FILE):
        os.remove(STAMP_FILE)
        print("[INIT] Environment cache reset.")

    if args.selftest:
        import engine
        return engine.run_selftest(CKPT_DIR)

    if args.setup and not os.path.exists(STAMP_FILE):
        return setup_environment()

    return fast_boot()


if __name__ == "__main__":
    sys.exit(main())