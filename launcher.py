#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launcher.py — FGVC-IR8 Zero-Friction Bootstrap & Fast-Boot Manager
==================================================================
- First run (--setup): checks global packages; creates isolated .venv only if needed,
  installs CPU-optimized wheels, verifies/downloads model weights from GitHub Releases,
  and writes .env_verified.json.
- Subsequent runs (--fast): skips all checks and starts the HTTP server in < 0.8s.
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


def has_required_packages(py_exec: str) -> bool:
    check_code = "import torch, torchvision, numpy, PIL; print('OK')"
    try:
        res = subprocess.run(
            [py_exec, "-c", check_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def _download_with_progress(url: str, dest_path: str, label: str) -> bool:
    tmp_path = dest_path + ".part"
    try:
        print(f"[WEIGHTS] Downloading {label} from:\n          {url}")

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = min(total_size, block_num * block_size)
                pct = (downloaded / total_size) * 100.0
                mb_done = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r          Progress: {pct:5.1f}% ({mb_done:.1f} / {mb_tot:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        sys.stdout.write("\n")
        os.replace(tmp_path, dest_path)
        print(f"[WEIGHTS] Saved: {dest_path}")
        return True
    except Exception as e:
        sys.stdout.write("\n")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        print(f"[ERROR] Failed to download {label}: {e}")
        return False


def ensure_weights() -> bool:
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Verify or download primary standalone bundle (~172 MB)
    if not (os.path.exists(STANDALONE_WEIGHTS) and os.path.getsize(STANDALONE_WEIGHTS) > 100_000_000):
        if not _download_with_progress(WEIGHTS_URL, STANDALONE_WEIGHTS, "fgvc_ir8_laptop_standalone_fp16.pt"):
            print(f"[ERROR] Required model bundle missing at: {STANDALONE_WEIGHTS}")
            return False

    # 2. Verify or download auxiliary best.pt head (~15.4 MB) for Dual-Head Ensemble
    if not (os.path.exists(BEST_HEAD_WEIGHTS) and os.path.getsize(BEST_HEAD_WEIGHTS) > 5_000_000):
        _download_with_progress(BEST_PT_URL, BEST_HEAD_WEIGHTS, "best.pt (Step-28039 Dual-Head Ensemble)")

    return True


def setup_environment() -> int:
    t0 = time.time()
    print("------------------------------------------------------------------")
    print("FGVC-IR8 System Initialization")
    print("------------------------------------------------------------------")

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
        "verified": True,
        "python_executable": active_py,
        "weights_path": STANDALONE_WEIGHTS,
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
    parser = argparse.ArgumentParser(description="FGVC-IR8 Launcher")
    parser.add_argument("--setup", action="store_true", help="Run first-time environment verification")
    parser.add_argument("--fast", action="store_true", help="Fast-boot web server directly")
    parser.add_argument("--reset", action="store_true", help="Reset cached environment stamp")
    args = parser.parse_args()

    if args.reset and os.path.exists(STAMP_FILE):
        os.remove(STAMP_FILE)
        print("[INIT] Environment cache reset.")

    if args.setup and not os.path.exists(STAMP_FILE):
        return setup_environment()

    return fast_boot()


if __name__ == "__main__":
    sys.exit(main())