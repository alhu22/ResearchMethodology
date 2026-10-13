# Pre-registration: Size-dependent accuracy loss under INT8 post-training quantization

**Status:** frozen on 2026-10-07, before any model was trained, quantized or evaluated.
**Course:** Research Methodology in Computer Science (5 ECTS). **Authors:** ALI REZA HUSSAINIMHD & MALEK KISANIEHMHD MALEK KISANIEH.

This document fixes the research question, hypotheses, measurements and analysis *before* results
exist. Any later change is recorded in the deviation log at the end, with date and reason. Settings
referred to here are implemented in `config.yaml`.

## 1. Research question

In aerial imagery, is the relative loss in detection accuracy caused by INT8 post-training
quantization of a lightweight YOLO detector larger for small objects than for larger objects?

## 2. Hypotheses and decision rule

For size group *g* ∈ {small, non-small}:

- RelDrop_g = (AP_g^FP32 − AP_g^INT8) / AP_g^FP32, with AP = COCO-style AP@[.5:.95]
- **D = RelDrop_small − RelDrop_non-small**

| | Statement |
|---|---|
| **H0** | D ≤ 0: the relative loss for small objects is not larger than for non-small objects. |
| **H1** | D > 0: the relative loss for small objects is larger than for non-small objects. |

**Decision rule.** The 95% paired-bootstrap percentile CI of D (FP32 vs. INT8 calibration seed 0)
is computed on the test set.

- Lower bound > 0 → H0 rejected, H1 supported (equivalent to a one-sided test at α = 0.025).
- CI contains 0 → inconclusive, H1 not supported.
- Upper bound < 0 → result contradicts H1.

This is the only confirmatory test. All other analyses are exploratory and reported with CIs only.

## 3. Design

- **Independent variable:** numerical precision of the TensorRT engine. The baseline is FP32 and
  the treatment is INT8 post-training quantization. FP16 is a control level used for interpretation only.
- **Moderator (observed, not manipulated):** object size group, small vs. non-small.
- **Structure:** a paired 2 (precision) × 2 (size) comparison on identical test images. H1 concerns the interaction.
- **Primary dependent variable:** AP@[.5:.95] per size group, from which RelDrop and D are derived.
- **Secondary dependent variables:** absolute AP drops; AP@0.5 per size group; small/medium/large AP;
  overall AP; median inference latency.

## 4. Materials

- **Dataset:** VisDrone2019-DET, obtained from the Ultralytics asset mirror and verified by
  SHA-256 (hashes in `config.yaml`). The split roles are fixed:

  | Split | Role |
  |---|---|
  | train (6,471 images) | Fine-tuning and calibration pool |
  | val (548 images) | Checkpoint selection and all development or debugging runs |
  | test-dev (1,610 images) | Final evaluation only |

- **Annotation policy:** rows with score 0 (VisDrone "ignored regions" and "others") are removed.
  Boxes are clipped to the image, and boxes thinner than 1 px are dropped. Detections inside the
  removed ignored regions therefore count as false positives for every condition, so absolute AP is
  not comparable with the official VisDrone toolkit.
- **Model:** YOLOv8n, starting from the COCO-pretrained `yolov8n.pt` (SHA-256 in `config.yaml`).
- **Toolchain:** Ultralytics 8.4.174 with TensorRT 11.3, whose INT8 path uses explicit Q/DQ via
  NVIDIA ModelOpt. The fallback is TensorRT 10.16, which uses implicit calibration. All versions are
  recorded by `scripts/01_environment_report.py`.

## 5. Procedure

1. **Fine-tuning:** fine-tune once on train with image size 640, 100 epochs, batch size 16, seed 0,
   deterministic mode and otherwise Ultralytics defaults. No hyper-parameter tuning is done. The best
   checkpoint according to val is selected, frozen and hashed.
2. **ONNX export:** one ONNX export of the frozen checkpoint, with a static 1×3×640×640 input.
3. **Engines:** TensorRT FP32, FP16 (`quantize=16`) and INT8 (`quantize=8`) engines, each built
   once, then saved and hashed. The toolchain's default INT8 configuration is used: max-based
   calibration, unquantized layers in FP16, and the head's output convolutions and DFL kept out of INT8.
4. **Calibration:** exactly 512 images, drawn uniformly at random from train for each calibration
   seed (0, 1, 2), with the same letterbox preprocessing as evaluation. The image lists are saved.
5. **Evaluation:** every engine is evaluated once on test-dev, with confidence 0.001, NMS IoU 0.7,
   at most 500 detections per image, `rect=False` and batch size 1. Inference with a fixed engine is
   deterministic, so repeating an evaluation adds no information.
6. **Sanity checks:**
   - A PyTorch FP32 checkpoint evaluation (TF32 disabled) is compared with the TensorRT FP32 engine.
   - Our pycocotools AP is compared with Ultralytics' own validation AP.
7. **Blinding:** INT8 results on test-dev are produced only in the final run. All development uses val.

## 6. Object-size definition

Object size is the area w × h of the ground-truth box in **original image pixels**, after clipping.

| Bin | Definition |
|---|---|
| Small | area < 32² = 1,024 px² |
| Non-small (primary contrast) | area ≥ 32² |
| Medium (secondary) | 32² ≤ area < 96² |
| Large (secondary) | area ≥ 96² = 9,216 px² |

The bins are disjoint. Detections are assigned to bins using pycocotools' standard rule. Ground-truth
boxes outside a bin are ignored, as are detections matched to them and unmatched detections whose
own area lies outside the bin.

## 7. Analysis plan

- **Unit of analysis:** the test image. Paired nonparametric bootstrap with B = 2,000 resamples of
  all 1,610 test images (with replacement, seed 0). The same resample indices are reused for every
  condition and every contrast.
- **Per-resample computation:** AP_small and AP_non-small for each condition. Matching is done once
  per condition by pycocotools; only AP is recomputed per resample. That function must reproduce
  pycocotools' full-test-set AP exactly before it is used.
- **Confirmatory:** the 95% percentile CI of D for FP32 vs. INT8 seed 0, assessed with the decision rule in §2.
- **Exploratory (CIs only):**
  - absolute drops
  - AP@0.5 by size
  - small/medium/large AP
  - INT8 seeds 1 and 2
  - the FP16 control
- **Robustness criterion:** the conclusion is called robust if D has the same sign for all three calibration seeds.
- **FP16 interpretation rule:** if the CI of FP16's D contains 0 while INT8's does not, the size
  effect is attributed to INT8 quantization. If FP16 shows a similar pattern to INT8, the paper states
  that the effect cannot be attributed to INT8 alone.

## 8. Latency (secondary, descriptive)

- **Definition:** the time of one engine forward call through Ultralytics `AutoBackend` on a fixed,
  preprocessed 1×3×640×640 tensor already in GPU memory. It is timed with `time.perf_counter()`, with
  `torch.cuda.synchronize()` before and after the call. Decoding, letterboxing, host-to-GPU copies and NMS are excluded.
- **Protocol:** FP32 and INT8 seed 0; 3 rounds in alternating order; each round consists of 200
  warm-up calls followed by 1,000 timed calls. An `nvidia-smi` snapshot is taken before the run.
- **Reported:** median, IQR, per-round medians, and speed-up as the ratio of medians. No statistical tests.

## 9. Fallback rules (decided in advance)

The **minimum viable result** is FP32 + INT8 seed 0 on test-dev, plus the confirmatory bootstrap.

- **TensorRT 11 smoke test fails:** use TensorRT 10.16 and log the switch below.
- **Fine-tuning would take too long:** reduce the number of epochs, decided before any quantization.
- **Latency is not working within half a day:** report Ultralytics' per-image inference time
  descriptively, or omit latency.
- **No time for seeds 1–2:** omit them and state this in Limitations.

## 10. Decisions taken after inspecting labels only (no model outputs)

| ID | Date | Decision | Evidence |
|---|---|---|---|
| L1 | 2026-10-07 | **Resampling unit stays "image"** (as planned). A sequence-level bootstrap is not used. | §7 allows sequence-level resampling only if filenames *unambiguously* encode sequences, and they do not. Test-dev has 61 filename prefixes for 1,610 images. The largest groups (e.g. `9999938`, 516 images) all have frame field `00000`, so they look like collections of still images rather than video sequences. Five prefixes contain 71% of test-dev, so clustering by prefix would not be a meaningful cluster structure. Possible correlation between similar scenes is listed as a threat to validity. To be re-confirmed from the Stage 3 output on the experiment machine. |
| L2 | 2026-10-07 | `max_det = 500` confirmed. | No val or test-dev image has more than 500 annotated objects (maximum 461 in test-dev). |

## 11. Deviation log

Record every change to this plan here **before** acting on it.

| Date | Change | Reason | Decided before or after seeing any test-dev results? |
|---|---|---|---|
| | | | |
