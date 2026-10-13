"""Stage 2 — Toolchain smoke test (Gate: "INT8 export works on this machine").

PURPOSE: prove that every tool in the chain works on THIS machine before we invest days in
fine-tuning. It is a FUNCTIONAL check only. It uses the COCO-pretrained YOLOv8n and the tiny
COCO8 dataset (4 validation images); the mAP values it prints are meaningless as results and
must never be reported in the paper.

Steps:
  1. Download yolov8n.pt and COCO8 to known project paths (SHA-256 verified).
  2. PyTorch FP32 validation on COCO8 (reference).
  3. Export three TensorRT engines from the same weights: FP32, FP16, INT8 (calibrated on COCO8).
     Each export runs in its own folder so the files cannot overwrite each other.
     The Ultralytics log of each export is saved (it shows WHICH INT8 path was used).
  4. Load every engine, predict one image, validate on COCO8.
  5. Write results/smoke_test/smoke_test.json and print PASS / FAIL.

Usage (from the project root; the 'tee' keeps TensorRT's own messages too):
    python scripts/02_smoke_test.py 2>&1 | tee results/smoke_test/console.log
"""

from __future__ import annotations

import math
import re
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    capture_ultralytics_log, configure_ultralytics, download_file, ensure_dir, load_config,
    print_header, project_path, save_json, sha256_file, utc_timestamp,
)

# Log lines that reveal how Ultralytics built the INT8 engine (see utils/export/engine.py)
INT8_EVIDENCE = [
    r"quantizing ONNX to INT8 with ModelOpt.*",          # TensorRT >= 11: explicit Q/DQ
    r"keeping \d+ head layers out of INT8.*",            # TensorRT <= 10: implicit calibration
    r"building (INT8|FP16|FP32) engine.*",
    r"collecting INT8 calibration images.*",
    r".*>300 images recommended for INT8 calibration.*",
    r"starting export with TensorRT .*",
]


def prepare_coco8(cfg: dict) -> Path:
    """Download COCO8 ourselves and write a YAML with an absolute path, so Ultralytics never
    downloads datasets into a global folder outside the project."""
    import yaml
    import ultralytics

    st = cfg["smoke_test"]
    data_root = project_path(cfg["paths"]["data_root"])
    zip_path = data_root / "raw" / "coco8.zip"
    download_file(st["coco8_url"], zip_path, st["coco8_sha256"])
    target = data_root / "coco8"
    if not (target / "images" / "val").is_dir():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_root)
    packaged = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco8.yaml"
    meta = yaml.safe_load(packaged.read_text(encoding="utf-8"))
    out = {"path": str(target), "train": "images/train", "val": "images/val", "names": meta["names"]}
    yaml_path = target / "coco8_local.yaml"
    yaml_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return yaml_path


def evidence_from_log(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    found = []
    for pattern in INT8_EVIDENCE:
        found += [m.group(0).strip() for m in re.finditer(pattern, text)]
    return found


def run_step(report: dict, name: str, fn):
    """Run one step, store its result or the full traceback; never crash the whole script."""
    print(f"\n--> {name}")
    t0 = time.time()
    try:
        result = fn()
        report["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), **(result or {})}
        print(f"    OK ({report['steps'][name]['seconds']} s)")
        return result
    except Exception as e:  # noqa: BLE001
        report["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                 "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        print(f"    FAILED: {type(e).__name__}: {e}")
        return None


def main() -> int:
    cfg = load_config()
    configure_ultralytics(cfg)  # must happen before importing ultralytics
    st = cfg["smoke_test"]
    out_dir = ensure_dir(project_path(cfg["paths"]["results_dir"]) / "smoke_test")
    work_dir = ensure_dir(project_path(cfg["paths"]["models_dir"]) / "smoke_test")
    report = {"timestamp_utc": utc_timestamp(),
              "purpose": "FUNCTIONAL CHECK ONLY (COCO-pretrained YOLOv8n on 4 COCO8 images). Not a result.",
              "steps": {}}

    print_header("Stage 2 smoke test")
    import torch

    report["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        report["overall"] = "FAIL"
        save_json(report, out_dir / "smoke_test.json")
        print("FAIL: PyTorch cannot see a CUDA GPU. Fix the environment first (see README, Stage 2).")
        return 1
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    trt_major = int(trt.__version__.split(".")[0])
    report["versions"] = {"ultralytics": ultralytics.__version__, "tensorrt": trt.__version__}
    report["expected_int8_path"] = ("explicit Q/DQ via NVIDIA ModelOpt" if trt_major >= 11
                                    else "implicit calibration (IInt8Calibrator)")
    print(f"TensorRT {trt.__version__} -> Ultralytics INT8 path: {report['expected_int8_path']}")

    # 1. inputs ---------------------------------------------------------------------------
    weights = work_dir / "yolov8n.pt"
    run_step(report, "download_weights",
             lambda: download_file(st["weights_url"], weights, st["weights_sha256"]))
    coco8 = run_step(report, "prepare_coco8", lambda: {"yaml": str(prepare_coco8(cfg))})
    if not (weights.exists() and coco8):
        report["overall"] = "FAIL"
        save_json(report, out_dir / "smoke_test.json")
        return 1
    coco8_yaml = coco8["yaml"]
    val_kwargs = dict(data=coco8_yaml, imgsz=st["imgsz"], batch=1, device=st["device"], conf=0.001, iou=0.7,
                      max_det=500, rect=False, plots=False, verbose=False,
                      project=str(out_dir / "val"), exist_ok=True)

    def metrics_dict(m):
        return {"mAP50_95": float(m.box.map), "mAP50": float(m.box.map50),
                "speed_ms_per_image": {k: round(float(v), 3) for k, v in m.speed.items()}}

    # 2. PyTorch reference ------------------------------------------------------------------
    def pytorch_val():
        with capture_ultralytics_log(out_dir / "val_pytorch_fp32.log"):
            return metrics_dict(YOLO(str(weights)).val(name="pytorch_fp32", **val_kwargs))

    run_step(report, "val_pytorch_fp32", pytorch_val)

    # 3 + 4. engines --------------------------------------------------------------------------
    for cond, quantize in (("fp32", None), ("fp16", 16), ("int8", 8)):
        cond_dir = ensure_dir(work_dir / cond)
        local_weights = cond_dir / "yolov8n.pt"
        shutil.copy2(weights, local_weights)
        log_path = out_dir / f"export_{cond}.log"

        def export(cond=cond, quantize=quantize, local_weights=local_weights, log_path=log_path):
            kwargs = dict(format="engine", imgsz=st["imgsz"], batch=1, dynamic=False, simplify=True,
                          device=st["device"], quantize=quantize)
            if quantize == 8:
                kwargs["data"] = coco8_yaml  # calibration images (smoke test only)
            with capture_ultralytics_log(log_path):
                path = YOLO(str(local_weights)).export(**kwargs)
            if not path or not Path(path).exists():
                raise RuntimeError("export returned no engine file (see the export log)")
            return {"engine": str(path), "engine_size_mb": round(Path(path).stat().st_size / 2**20, 2),
                    "engine_sha256": sha256_file(path), "log": str(log_path),
                    "log_evidence": evidence_from_log(log_path)}

        exported = run_step(report, f"export_{cond}", export)
        if not exported:
            continue

        def predict_and_val(cond=cond, engine=exported["engine"]):
            model = YOLO(engine, task="detect")
            with capture_ultralytics_log(out_dir / f"val_{cond}.log"):
                pred = model.predict(source=str(ASSETS / "bus.jpg"), imgsz=st["imgsz"], device=st["device"],
                                     verbose=False)
                result = metrics_dict(model.val(name=f"engine_{cond}", **val_kwargs))
            result["boxes_on_bus_jpg"] = int(len(pred[0].boxes))
            if not math.isfinite(result["mAP50_95"]):
                raise RuntimeError("non-finite mAP")
            return result

        run_step(report, f"val_engine_{cond}", predict_and_val)

    # 5. verdict ----------------------------------------------------------------------------
    required = ["val_pytorch_fp32", "export_fp32", "val_engine_fp32", "export_int8", "val_engine_int8"]
    optional = ["export_fp16", "val_engine_fp16"]
    ok = all(report["steps"].get(s, {}).get("ok") for s in required)
    report["overall"] = "PASS" if ok else "FAIL"
    report["fp16_control_available"] = all(report["steps"].get(s, {}).get("ok") for s in optional)
    path = save_json(report, out_dir / "smoke_test.json")

    print_header("Smoke test summary (functional check only - not results)")
    for name, s in report["steps"].items():
        extra = f"mAP50-95={s['mAP50_95']:.3f}" if "mAP50_95" in s else ""
        print(f"  {'OK  ' if s['ok'] else 'FAIL'}  {name:<20s} {extra}")
    int8 = report["steps"].get("export_int8", {})
    print("\nINT8 export evidence from the log:")
    for line in int8.get("log_evidence", []) or ["(none found - send export_int8.log)"]:
        print(f"  {line}")
    print(f"\nOVERALL: {report['overall']}   ->  {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
