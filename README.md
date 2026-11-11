# ABC Challenge 2027 — UWB Fall Detection

Our group is developing and evaluating methods for the **9th ABC Challenge, UWB Fall Track**. The project focuses on detecting and temporally localizing falls in continuous Ultra-Wideband (UWB) radar streams, while distinguishing them from everyday activities (ADLs).

> **Project status:** The repository contains design notes, notebooks, baseline reports, and a CNN–TCN experiment implementation. The CNN–TCN experiment is documented as **not yet executed or validated**; no performance results are claimed for it. See [the experiment README](experiments/cnn_tcn/README.md) for its current status and run instructions.

## Challenge overview

The task is to identify fall events in continuous, unsegmented UWB streams and estimate each event's start and end times. The challenge is designed around privacy-preserving monitoring using radar rather than cameras or microphones.

We follow the official challenge data format and evaluation rules. The project proposal describes the released streams as synchronized three-radar CIR data (368 range bins per radar, nominally 50 Hz), with event annotations and an official evaluator. Please consult the [official challenge page](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/the-challenge#h.qs9yvo6sbnah) and [dataset/tutorial page](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/dataset-and-tutorial) for authoritative details and current rules.

## Project approach

The current experimental direction is a **frame-level CNN–TCN** that produces fall scores over time, then decodes scores into event intervals:

1. Read continuous HDF5 streams in windows while preserving a fixed radar order.
2. Apply a shared 1D CNN to each radar/frame's range-bin signal and combine the three radar feature vectors.
3. Model temporal context with a Temporal Convolutional Network (TCN).
4. Predict fall likelihood per frame (V1); V2 adds start- and end-boundary prediction heads.
5. Aggregate overlapping-window predictions, decode candidate intervals, and evaluate against the official validation protocol.

The design starts with 400-frame (8-second) windows and a 50-frame (1-second) hop. These are initial experiment settings, not tuned or validated results. The intended comparison is between V1 and V2, with validation selection based on the official event-level metrics.

The repository also includes a separate pipeline proposal and a Colab notebook. Treat proposals and notebook experiments as exploratory unless results are explicitly recorded and independently verified.

## Repository contents

| Path | Description |
|---|---|
| [`experiments/cnn_tcn/`](experiments/cnn_tcn/README.md) | CNN–TCN experiment package, configs, source, tests, and setup/train/inference instructions. |
| [`baseline-proposal.md`](baseline-proposal.md) | Vietnamese design proposal for the CNN–TCN approach; distinguishes official references from unrun team experiments. |
| [`pipeline.md`](pipeline.md) | Proposed continuous-stream inference pipeline and technical notes. |
| [`cnn_tcn_v9_colab.ipynb`](cnn_tcn_v9_colab.ipynb) | Exploratory Colab notebook. |
| [`test-baseline/`](test-baseline/) | Baseline run metadata and reports. |
| [`requirements.txt`](requirements.txt) | Top-level Python environment requirements. |

## Data and evaluation

Challenge data is **not included in this repository**. Obtain it through the official challenge channels and configure a local data root as described in [`experiments/cnn_tcn/README.md`](experiments/cnn_tcn/README.md). Keep the released data read-only and store checkpoints, logs, predictions, and reports outside the data directory.

The CNN–TCN experiment is designed to use the official train/validation streams and evaluator. Its README documents validation evaluation with `--ignore-flagged` and event-level F1 at temporal IoU thresholds 0.1, 0.3, and 0.5 as the primary model-selection measure, with additional metrics for stricter overlap, boundary error, and false alarms. Follow the current official rules for any final submission; do not use test data for model or threshold selection.

The baseline proposal records the challenge README's published reference values (mean F1 across tIoU 0.1/0.3/0.5): energy baseline **0.315**, logistic-regression baseline **0.258**. These are reference results, not results from our CNN–TCN models. The proposal marks CNN–TCN V1/V2 as not yet run; do not interpret blank result cells as scores.

## Getting started

For the CNN–TCN environment, data checks, tests, training, inference, and official evaluation, follow the detailed [experiment setup guide](experiments/cnn_tcn/README.md). Its commands assume Linux, Python 3.10+, and a locally available data package; install a PyTorch build appropriate for your machine before installing the experiment requirements.

## Reproducibility and responsible reporting

- Keep train, validation, and test roles separate; tune only on validation.
- Record configuration, random seed, environment, and evaluator output for every reported run.
- Clearly label planned, unexecuted, and unverified experiments.
- Use only data permitted by the challenge rules, and check the official site for updates before preparing a submission.

## References

- [Official 9th ABC Challenge — UWB Fall Track](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/the-challenge#h.qs9yvo6sbnah)
- [Official dataset and tutorial](https://sites.google.com/view/abc2027-uwbfalltrack-challenge/dataset-and-tutorial)
- [CNN–TCN experiment details](experiments/cnn_tcn/README.md)
