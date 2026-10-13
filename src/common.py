"""Shared helpers for all stage scripts.

Everything that more than one script needs lives here:
  * locating the project root and loading ``config.yaml``
  * writing machine-readable outputs (JSON / CSV) with consistent formatting
  * SHA-256 hashing of files (used to prove which weights / archives / engines were used)
  * downloading files with optional hash verification
  * configuring Ultralytics so it keeps its settings inside the project and never
    silently pip-installs packages
  * capturing the Ultralytics log of a step into a file
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ----------------------------------------------------------------------------------- config
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the central YAML configuration."""
    import yaml  # imported lazily so the environment report can run without PyYAML

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Resolve a path from config.yaml (relative to the project root) to an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-10-07T12:34:56Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_timestamp() -> str:
    """Timestamp safe for file names, e.g. 20261007T123456Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ----------------------------------------------------------------------------------- outputs
def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
    return path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: Iterable[dict], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Write a list of dicts as CSV. Column order = fieldnames or key order of the first row."""
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ----------------------------------------------------------------------------------- hashing
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Fast content hash, used only for duplicate detection (not for security)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------- download
def download_file(url: str, dest: str | Path, expected_sha256: str | None = None) -> dict:
    """Download ``url`` to ``dest`` (skipped if the file already exists and matches the hash).

    Returns a small provenance record: url, path, size, sha256, whether it was downloaded now.
    Raises RuntimeError if an expected hash is given and does not match.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    downloaded = False
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url}\n         -> {dest}")
        t0 = time.time()
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done, next_report = 0, 0
            while block := response.read(1 << 20):
                out.write(block)
                done += len(block)
                if total and done >= next_report:
                    print(f"  {done / 2**20:8.1f} / {total / 2**20:.1f} MiB", flush=True)
                    next_report = done + max(total // 10, 1 << 20)
        tmp.rename(dest)
        downloaded = True
        print(f"  finished in {time.time() - t0:.0f} s")
    digest = sha256_file(dest)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {dest}\n  expected {expected_sha256}\n  got      {digest}\n"
            "Delete the file and download again; if it still differs, tell your project partner."
        )
    return {
        "url": url,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": digest,
        "sha256_verified": bool(expected_sha256),
        "downloaded_now": downloaded,
    }


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` (no extra disk space); fall back to copying."""
    if dst.exists():
        return
    ensure_dir(dst.parent)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ----------------------------------------------------------------------------------- ultralytics
def configure_ultralytics(cfg: dict) -> None:
    """Must be called BEFORE ``import ultralytics``.

    * YOLO_CONFIG_DIR keeps Ultralytics' settings file and fonts inside the project, so we do not
      change anything in the user's home directory.
    * YOLO_AUTOINSTALL=false stops Ultralytics from pip-installing missing packages on its own.
      A missing package then becomes a visible error instead of a silent environment change,
      which keeps the environment reproducible (everything is installed from requirements/).
    """
    if "ultralytics" in sys.modules:
        print("WARNING: configure_ultralytics() called after ultralytics was imported.")
    os.environ["YOLO_CONFIG_DIR"] = str(ensure_dir(project_path(cfg["paths"]["ultralytics_config_dir"])))
    os.environ["YOLO_AUTOINSTALL"] = "false"


@contextmanager
def capture_ultralytics_log(log_path: str | Path):
    """Copy everything the Ultralytics logger prints during the ``with`` block into ``log_path``.

    TensorRT's own C++ messages go directly to the terminal and are NOT captured here;
    run scripts with ``2>&1 | tee <file>`` to keep a complete console log as well.
    """
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger("ultralytics")
    logger.addHandler(handler)
    try:
        yield log_path
    finally:
        logger.removeHandler(handler)
        handler.close()


def print_header(title: str) -> None:
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


# ----------------------------------------------------------------------------------- run folders
def run_dirs(cfg: dict, tag: str) -> dict[str, Path]:
    """Folders of one run. tag 'main' = the real experiment; 'dev' = quick pipeline test.

    Everything produced under tag 'dev' is a DEVELOPMENT artefact and is never reported.
    """
    if tag not in ("main", "dev"):
        raise ValueError("tag must be 'main' or 'dev'")
    models = ensure_dir(project_path(cfg["paths"]["models_dir"]) / tag)
    results = ensure_dir(project_path(cfg["paths"]["results_dir"]) / tag)
    return {"models": models, "results": results,
            "model_record": models / "model_record.json",
            "engines_record": models / "engines_record.json",
            "engines": models / "engines",
            "calibration": results / "calibration",
            "evaluation": results / "evaluation",
            "analysis": results / "analysis"}
