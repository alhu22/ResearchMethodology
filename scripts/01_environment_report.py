"""Stage 2 — Environment and hardware report.

Records everything about the machine and software that could influence the results:
GPU, driver, CUDA, Python, package versions, PyTorch/TensorRT runtime checks, CPU, RAM, disk.

The script uses ONLY the Python standard library at the top level, so it can be run
twice:
    1. before installing anything        ->  --label before_install
    2. after installing the environment  ->  --label after_install

Usage (from the project root):
    python scripts/01_environment_report.py --label before_install

Outputs (results/environment/):
    environment_<label>.json      machine-readable report (send this to your partner)
    nvidia_smi_<label>.txt        raw nvidia-smi output
    pip_freeze_<label>.txt        exact versions of every installed Python package
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT, ensure_dir, print_header, save_json, utc_timestamp  # noqa: E402

PACKAGES = [
    "torch", "torchvision", "ultralytics", "tensorrt", "tensorrt-cu12", "tensorrt-cu13",
    "tensorrt_cu12_libs", "nvidia-modelopt", "onnx", "onnxslim", "onnxruntime", "onnxruntime-gpu",
    "cupy-cuda12x", "pycocotools", "numpy", "pandas", "scipy", "matplotlib", "opencv-python",
    "pillow", "pyyaml",
]


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    """Run a command, return (exit code, combined output). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def os_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "kernel": platform.release()}
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                info["distribution"] = line.split("=", 1)[1].strip('"')
    return info


def cpu_ram_disk() -> dict:
    info: dict = {"logical_cpus": os.cpu_count()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text())
        info["cpu_model"] = m.group(1).strip() if m else None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        m = re.search(r"MemTotal:\s*(\d+) kB", meminfo.read_text())
        info["ram_total_gib"] = round(int(m.group(1)) / 2**20, 1) if m else None
    usage = shutil.disk_usage(PROJECT_ROOT)
    info["disk_free_gib_at_project"] = round(usage.free / 2**30, 1)
    info["disk_total_gib_at_project"] = round(usage.total / 2**30, 1)
    return info


def python_info() -> dict:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "in_virtualenv": sys.prefix != sys.base_prefix,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def nvidia_smi_info(label: str, out_dir: Path) -> dict:
    info: dict = {}
    code, full = run(["nvidia-smi"])
    (out_dir / f"nvidia_smi_{label}.txt").write_text(full + "\n", encoding="utf-8")
    info["available"] = code == 0
    if code != 0:
        info["error"] = full
        return info
    m = re.search(r"CUDA Version:\s*([\d.]+)", full)
    info["driver_max_cuda_version"] = m.group(1) if m else None

    fields = ["name", "memory.total", "driver_version", "compute_cap", "clocks.max.sm",
              "clocks.max.mem", "power.limit", "persistence_mode", "pstate", "temperature.gpu",
              "utilization.gpu", "memory.used"]
    code, out = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    if code != 0:  # older drivers do not know every field (e.g. compute_cap) -> retry without it
        fields = [f for f in fields if f != "compute_cap"]
        code, out = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    gpus = []
    if code == 0:
        for line in out.splitlines():
            values = [v.strip() for v in line.split(",")]
            gpus.append(dict(zip(fields, values)))
    info["gpus"] = gpus

    code, out = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader"])
    info["running_compute_processes"] = [l.strip() for l in out.splitlines() if l.strip()] if code == 0 else out
    return info


def nvcc_info() -> str | None:
    code, out = run(["nvcc", "--version"])
    if code != 0:
        return None
    m = re.search(r"release ([\d.]+)", out)
    return m.group(1) if m else out


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_runtime() -> dict:
    try:
        torch = importlib.import_module("torch")
    except Exception as e:  # noqa: BLE001
        return {"importable": False, "error": f"{type(e).__name__}: {e}"}
    info = {
        "importable": True,
        "version": torch.__version__,
        "built_with_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "default_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "default_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(p.total_memory / 2**30, 2),
                "multiprocessors": p.multi_processor_count,
            })
        try:  # tiny real computation: proves the GPU actually works from PyTorch
            x = torch.randn(256, 256, device="cuda")
            info["gpu_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
        except Exception as e:  # noqa: BLE001
            info["gpu_matmul_ok"] = False
            info["gpu_matmul_error"] = f"{type(e).__name__}: {e}"
    return info


def tensorrt_runtime() -> dict:
    try:
        trt = importlib.import_module("tensorrt")
    except Exception as e:  # noqa: BLE001
        return {"importable": False, "error": f"{type(e).__name__}: {e}"}
    info = {"importable": True, "version": trt.__version__}
    major = int(trt.__version__.split(".")[0])
    info["ultralytics_int8_path"] = (
        "explicit Q/DQ via NVIDIA ModelOpt (TensorRT >= 11)" if major >= 11
        else "implicit calibration via IInt8Calibrator (TensorRT <= 10)"
    )
    try:  # creating a Builder proves TensorRT can talk to the GPU
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        info["builder_created"] = builder is not None
        for attr in ("platform_has_fast_fp16", "platform_has_fast_int8", "platform_has_tf32"):
            if hasattr(builder, attr):
                try:
                    info[attr] = bool(getattr(builder, attr))
                except Exception:  # noqa: BLE001  (deprecated attributes may raise)
                    pass
    except Exception as e:  # noqa: BLE001
        info["builder_created"] = False
        info["builder_error"] = f"{type(e).__name__}: {e}"
    return info


def onnxruntime_runtime() -> dict:
    try:
        ort = importlib.import_module("onnxruntime")
        return {"importable": True, "version": ort.__version__, "providers": ort.get_available_providers()}
    except Exception as e:  # noqa: BLE001
        return {"importable": False, "error": f"{type(e).__name__}: {e}"}


def derived_notes(report: dict) -> list[str]:
    """Plain-language observations that matter for the experiment design."""
    notes = []
    smi = report["nvidia_smi"]
    if not smi.get("available"):
        notes.append("BLOCKER: nvidia-smi not available - no usable NVIDIA GPU/driver detected.")
        return notes
    gpus = smi.get("gpus", [])
    if len(gpus) > 1:
        notes.append(f"{len(gpus)} GPUs detected - all experiments will use GPU index 0 only.")
    cuda = smi.get("driver_max_cuda_version")
    if cuda:
        major = int(cuda.split(".")[0])
        if major < 12:
            notes.append(f"BLOCKER: driver supports CUDA {cuda}; the planned stack needs CUDA 12.x or newer.")
        else:
            notes.append(f"Driver supports up to CUDA {cuda}: CUDA 12.x wheels (cu12 profile) are compatible.")
    cap = gpus[0].get("compute_cap") if gpus else None
    if cap is None and report["torch"].get("devices"):
        cap = report["torch"]["devices"][0]["compute_capability"]
    if cap:
        c = float(cap)
        if c < 6.1:
            notes.append(f"BLOCKER: compute capability {cap} < 6.1 - TensorRT INT8 is not supported.")
        elif c < 7.5:
            notes.append(f"Compute capability {cap}: INT8 runs, but without INT8 tensor cores; speed-ups may be small.")
        if c >= 8.0:
            notes.append(f"Compute capability {cap} supports TF32 - the FP32 baseline check (Stage 4) disables "
                         "TF32 in PyTorch and compares against the TensorRT FP32 engine.")
    procs = smi.get("running_compute_processes")
    if isinstance(procs, list) and procs:
        notes.append(f"{len(procs)} other GPU process(es) running now - the latency benchmark needs an idle GPU.")
    t = report["torch"]
    if t.get("importable"):
        if not t.get("cuda_available"):
            notes.append("PROBLEM: PyTorch is installed but cannot use CUDA (CPU-only build or driver mismatch).")
        trt_pkgs = {k: v for k, v in report["packages"].items() if k.startswith("tensorrt-cu") and v}
        if t.get("built_with_cuda") and trt_pkgs:
            torch_major = t["built_with_cuda"].split(".")[0]
            for pkg in trt_pkgs:
                if pkg != f"tensorrt-cu{torch_major}":
                    notes.append(f"PROBLEM: PyTorch is built for CUDA {t['built_with_cuda']} but {pkg} is installed; "
                                 f"Ultralytics expects tensorrt-cu{torch_major}.")
    if report["system"].get("disk_free_gib_at_project", 99) < 15:
        notes.append("WARNING: less than 15 GiB free disk space at the project location (dataset ~2 GB zipped, "
                     "~4 GB extracted, plus models and results).")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="after_install",
                        help="name for this snapshot, e.g. before_install / after_install")
    args = parser.parse_args()

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "environment")
    print_header(f"Environment report ({args.label})")

    report = {
        "label": args.label,
        "timestamp_utc": utc_timestamp(),
        "project_root": str(PROJECT_ROOT),
        "os": os_info(),
        "system": cpu_ram_disk(),
        "python": python_info(),
        "nvidia_smi": nvidia_smi_info(args.label, out_dir),
        "nvcc_version": nvcc_info(),
        "packages": package_versions(),
        "torch": torch_runtime(),
        "tensorrt": tensorrt_runtime(),
        "onnxruntime": onnxruntime_runtime(),
    }
    report["notes"] = derived_notes(report)

    code, freeze = run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    (out_dir / f"pip_freeze_{args.label}.txt").write_text(freeze + "\n", encoding="utf-8")

    path = save_json(report, out_dir / f"environment_{args.label}.json")

    # ---- short human-readable summary
    smi = report["nvidia_smi"]
    for i, g in enumerate(smi.get("gpus", [])):
        print(f"GPU {i}: {g.get('name')} | {g.get('memory.total')} | driver {g.get('driver_version')} "
              f"| compute cap {g.get('compute_cap', '?')}")
    print(f"Driver max CUDA: {smi.get('driver_max_cuda_version')}   nvcc: {report['nvcc_version']}")
    print(f"Python {report['python']['version']}  ({report['python']['executable']})")
    t, r = report["torch"], report["tensorrt"]
    print(f"PyTorch: {t.get('version')} (CUDA {t.get('built_with_cuda')}, available={t.get('cuda_available')})")
    print(f"TensorRT: {r.get('version')}  -> {r.get('ultralytics_int8_path', 'n/a')}")
    print(f"Ultralytics: {report['packages']['ultralytics']}   ModelOpt: {report['packages']['nvidia-modelopt']}")
    print(f"RAM {report['system'].get('ram_total_gib')} GiB | free disk {report['system']['disk_free_gib_at_project']} GiB")
    print("\nNotes:")
    for n in report["notes"] or ["(none)"]:
        print(f"  - {n}")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
