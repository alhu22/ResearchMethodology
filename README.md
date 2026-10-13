# INT8 post-training quantization and object size in aerial detection

Research Methodology in Computer Science (5 ECTS) project.

> **Research question.** In aerial imagery, is the relative loss in detection accuracy caused by INT8
> post-training quantization of a lightweight YOLO detector larger for small objects than for larger objects?

The frozen design is in [`PREREGISTRATION.md`](PREREGISTRATION.md). Every setting is in
[`config.yaml`](config.yaml), and the scripts read their settings only from there.

## Project structure

```
int8_size_study/
├── PREREGISTRATION.md        frozen design + decision log + deviation log
├── config.yaml               single source of truth for all settings
├── requirements/
│   ├── common.txt            pinned packages shared by both profiles
│   ├── trt11-cu12.txt        PREFERRED: TensorRT 11 (explicit INT8 via ModelOpt)
│   └── trt10-cu12.txt        FALLBACK:  TensorRT 10 (implicit INT8 calibration)
├── src/common.py             shared helpers (config, hashing, downloads, logging)
├── scripts/
│   ├── 01_environment_report.py   Stage 2
│   ├── 02_smoke_test.py           Stage 2
│   ├── 03_prepare_dataset.py      Stage 3
│   ├── 04_check_dataset.py        Stage 3
│   ├── 05_train_fp32.py           Stage 4  fine-tune YOLOv8n, freeze FP32 checkpoint
│   ├── 06_build_engines.py        Stage 5  calibration sets + FP32/FP16/INT8 engines
│   ├── 07_evaluate.py             Stage 6  AP per size bin (pycocotools)
│   └── 08_analyze.py              Stage 8  paired bootstrap, decision, tables, figures
├── src/evaluation.py         size-binned COCO evaluation + exact bootstrap AP
├── tests/test_evaluation.py  proves our AP code equals pycocotools (synthetic data)
├── data/      (created by the scripts; not part of the package)
├── models/    (created by the scripts)
└── results/   (created by the scripts; everything you send back comes from here)
```

The latency benchmark (Stage 7) will be added once the hardware is known.

## Requirements of the machine

| Item | Requirement |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA GPU, compute capability ≥ 6.1 (≥ 7.5 for INT8 tensor cores) |
| Driver | supports CUDA 12.x or newer (`nvidia-smi` shows "CUDA Version: 12.x" or higher) |
| Python | 3.12 |
| Disk | ≥ 15 GiB free in the project folder |
| Network | `pypi.org`, `pypi.nvidia.com` (TensorRT wheels), `download.pytorch.org`, `github.com` and its release-asset downloads (weights, datasets, fonts) |

The scripts never `sudo`, never change system settings, and keep Ultralytics' settings inside the
project (`.ultralytics/`). Ultralytics' automatic pip-installs are switched off, so a missing package
shows up as an error instead of a silent environment change.

## Stage 2 — Environment setup

Run all commands from the project root.

**Step 1. Report the bare machine.** This needs only Python, with nothing installed:

```bash
python3 scripts/01_environment_report.py --label before_install
```

Check the "Notes" it prints. Any line starting with **BLOCKER** means stop and send me
`results/environment/environment_before_install.json`.

**Step 2. Create a Python 3.12 environment.** Use either of these:

```bash
conda create -n int8study python=3.12 -y && conda activate int8study
# or
python3.12 -m venv .venv && source .venv/bin/activate
```

**Step 3. Install PyTorch built for CUDA 12.**

This must happen *before* the other packages. The default PyPI build of PyTorch 2.14.1 targets
CUDA 13, which conflicts with TensorRT-cu12 and ModelOpt.

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The check must print a CUDA version starting with `12.` and `True`.

If `cu128` is not offered for 2.14.1, pick the highest `cu12x` index listed on
<https://pytorch.org/get-started/previous-versions/> that your driver supports. If no CUDA 12 build
of 2.14.x exists at all, stop and tell me; the profile would then need to change.

**Step 4. Install the TensorRT 11 profile.**

```bash
pip install -r requirements/trt11-cu12.txt --extra-index-url https://pypi.nvidia.com
```

**Step 5. Report the installed environment.**

```bash
python scripts/01_environment_report.py --label after_install
```

**Step 6. Run the smoke test.** This is a functional check only: COCO-pretrained YOLOv8n on 4 COCO8
images. Its mAP values are meaningless as results.

```bash
mkdir -p results/smoke_test
python scripts/02_smoke_test.py 2>&1 | tee results/smoke_test/console.log
```

It must end with `OVERALL: PASS`. The lines under "INT8 export evidence" show which INT8 path was
used. With TensorRT 11 you should see `quantizing ONNX to INT8 with ModelOpt ...`.

**If the INT8 export fails on TensorRT 11**, switch to the fallback profile (pre-registered) and
add a row to the deviation log in `PREREGISTRATION.md`:

```bash
pip uninstall -y tensorrt-cu12 tensorrt_cu12_libs tensorrt_cu12_bindings nvidia-modelopt
pip install -r requirements/trt10-cu12.txt --extra-index-url https://pypi.nvidia.com
python scripts/01_environment_report.py --label after_install_trt10
python scripts/02_smoke_test.py 2>&1 | tee results/smoke_test/console.log
```

**Send back after Stage 2:**

- `results/environment/environment_before_install.json`
- `results/environment/environment_after_install.json`
- `results/environment/pip_freeze_after_install.txt`
- `results/smoke_test/smoke_test.json`
- `results/smoke_test/console.log`
- `results/smoke_test/export_int8.log`

## Stage 3 — Dataset preparation and sanity checks

```bash
mkdir -p results/dataset
python scripts/03_prepare_dataset.py 2>&1 | tee results/dataset/prepare_console.log
python scripts/04_check_dataset.py   2>&1 | tee results/dataset/check_console.log
```

`03_prepare_dataset.py` downloads about 1.9 GB of archives and verifies each by SHA-256. It then
converts the annotations once into two formats from the same parsed boxes:

- YOLO labels, used by Ultralytics for training, calibration and inference
- COCO JSON for val and test, used later by pycocotools to compute AP

Images are hard-linked rather than copied, so they take no extra disk space. If you later delete
`data/raw/`, the prepared images survive.

`04_check_dataset.py` is the **gate** before any modelling. It runs checks C1–C8 (see the docstring)
and exits with code 1 if any check fails; in that case do not continue to Stage 4.

**Expected output.** The archives are hash-verified, so your numbers must match these reference
values from my verification run of the scripts exactly. A mismatch means something went wrong.

| Split | Images | Objects | Small | Medium | Large |
|---|---|---|---|---|---|
| train | 6,471 | 343,204 | 60.5% | 34.0% | 5.5% |
| val | 548 | 38,759 | 68.6% | 28.7% | 2.8% |
| test | 1,610 | 75,102 | 67.7% | 29.1% | 3.2% |

All checks C1–C8 should report PASS.

**Look at these figures yourselves:**

- `results/dataset/figures/samples_*.png` — boxes must sit on the objects, coloured red (small),
  orange (medium) or blue (large)
- `results/dataset/figures/fig1_size_distribution_test.png` — this becomes Figure 1 of the paper

**Send back after Stage 3:**

- `results/dataset/dataset_checks.json`
- `results/dataset/prepare_report.json`
- `results/dataset/size_bin_counts.csv`
- the two console logs

## Stages 4–8 — Training, engines, evaluation, analysis

Work in the **dedicated Python 3.12 environment** (not `venv-ardupilot`). The prompt must show only
`(.venv)` or your own environment name. Run every command from the project root.

**First prove the evaluation code, then run the whole pipeline once as a quick `dev` test on val:**

```bash
python tests/test_evaluation.py                 # both tests must PASS
python scripts/05_train_fp32.py   --tag dev     # 3 epochs on 10 % of train (minutes)
python scripts/06_build_engines.py --tag dev    # FP32, FP16, INT8 x 3 seeds
python scripts/07_evaluate.py     --tag dev --split val
python scripts/08_analyze.py      --tag dev --split val --reps 50
```

`dev` results are labelled DEVELOPMENT and are never reported. Their only job is to show that every
stage runs on your machine before the long training starts.

**Then the real experiment:**

```bash
mkdir -p results
python scripts/05_train_fp32.py   --tag main 2>&1 | tee results/train_main_console.log    # long run
python scripts/06_build_engines.py --tag main 2>&1 | tee results/engines_main_console.log
python scripts/07_evaluate.py     --tag main --split val                                 # development check
```

If training is interrupted, run `python scripts/05_train_fp32.py --tag main --resume`.

**Stop here and send me the following, before running the final evaluation:**

- `models/main/model_record.json`
- `models/main/engines_record.json`
- `results/main/evaluation/val/*/metrics.json`

Only when the val numbers look sane, run the **final, confirmatory** evaluation **once**:

```bash
python scripts/07_evaluate.py --tag main --split test --final
python scripts/08_analyze.py  --tag main --split test --final
```

The test split is refused without `--final`, which protects the blinding rule in the
pre-registration. Stage 8 compares small, non-small, medium and large objects for every condition.
It writes `table3_accuracy.csv` (AP per size bin with 95% CIs), `table4_hypothesis.csv` (drops and
D with CIs), `summary.json` (the H1 decision) and the figures, all in `results/main/analysis/test/`.

## What each script does and why

| Script | Key design choices |
|---|---|
| `01_environment_report.py` | Uses only the standard library, so it also runs before installation. It records GPU, driver, CUDA, package versions and `pip freeze`, and whether the GPU supports TF32/INT8. It also lists other GPU processes, which matter for latency. |
| `02_smoke_test.py` | Exports FP32, FP16 and INT8 engines in **separate folders**, because Ultralytics would otherwise overwrite `model.engine`. It saves each export log as evidence of how INT8 was built, and downloads COCO8 itself so nothing lands outside the project. |
| `03_prepare_dataset.py` | Uses our own converter instead of Ultralytics' built-in one, so every skipped, clipped or dropped box is counted and reported. YOLO and COCO formats come from the same parsed boxes. COCO `category_id` = class index + 1, which matches Ultralytics' prediction JSON. |
| `05_train_fp32.py` | Refuses to run unless Stage 3 passed, and refuses to overwrite a frozen model. It records the checkpoint's SHA-256; every later stage verifies it. |
| `06_build_engines.py` | Builds each engine in its own folder, exactly once. It saves the calibration image lists and their size statistics. |
| `07_evaluate.py` | Uses identical inference settings for every condition. pycocotools does the matching, and our AP recomputation is checked against it immediately. The PyTorch FP32 reference runs with TF32 disabled. |
| `08_analyze.py` | Draws the same resampled image sets for every condition (a paired design) and computes exact COCO AP per resample. It applies the pre-registered decision, robustness and FP16 rules. |
| `04_check_dataset.py` | Fully decodes every image, validates every label line, and checks that the YOLO and COCO boxes are identical. It also checks for duplicate images (leakage), the counts per size bin, the density per image (for `max_det`) and the filename-prefix structure (for the bootstrap unit). |

## Reproducibility rules

1. Change settings only in `config.yaml`. After the design freeze, also log the change in
   `PREREGISTRATION.md` §11.
2. Never type results by hand. Every number in the paper must come from a file in `results/`.
3. INT8 is evaluated on test-dev only in the final run. All development uses val.
