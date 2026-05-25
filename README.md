# Soft-SAD: Speech Activity Detection with Soft Event Metrics

Reference implementation accompanying *"Soft Event Metrics for Speech Activity
Detection: Adapting SoftED to a Two-Event-Class Setting"*.

The code does five things:

1. Loads audio + interval-style annotations into per-frame labels (speech /
   non-speech / silence).
2. Extracts 13-dim MFCC features.
3. Trains a small GRU classifier (5–16 hidden units) with class-weighted BCE.
4. Runs inference and turns the per-frame speech probability into a set of
   detection events (rising edges of the thresholded stream).
5. Evaluates with **both** hard event metrics (rectangular collar) and
   **Soft-SAD** event metrics (graded membership + dummy-classifier fallback +
   optional rigorous non-speech membership), and produces F1-vs-threshold
   curves, confusion matrices, ROC curves, and per-category histograms.

## Install

```bash
pip install -r requirements.txt
```

## Data format

See `DATA_FORMAT.md` for the full spec. Quick version:

```
data_root/
├── train.csv
├── val.csv
├── test.csv
├── wavs/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── annotations/
    ├── 0001.json
    ├── 0002.json
    └── ...
```

Each `*.csv` lists which utterances belong to that split. Each `*.json`
gives time intervals labelled `speech`, `nonspeech` or `silence`, plus an
optional `category` (e.g. `coughing`) used for per-category histograms.

## Run

```bash
# 1. build features from raw wavs + json annotations
python -m soft_sad.prepare --config config.yaml

# 2. train
python -m soft_sad.train --config config.yaml

# 3. evaluate (sweeps thresholds, dumps metrics & plots)
python -m soft_sad.evaluate --config config.yaml --checkpoint runs/best.pt
```

All hyper-parameters live in `config.yaml`. 

## Code layout

| file | purpose |
|---|---|
| `soft_sad/data.py`      | dataset loader, annotation → per-frame labels |
| `soft_sad/features.py`  | MFCC extraction |
| `soft_sad/model.py`     | tiny GRU classifier |
| `soft_sad/events.py`    | rising-edge detection from probability stream |
| `soft_sad/membership.py`| piecewise membership $\mu(t)$ |
| `soft_sad/metrics_hard.py` | hard event metrics|
| `soft_sad/metrics_soft.py` | Soft-SAD metrics |
| `soft_sad/prepare.py`   | feature extraction CLI |
| `soft_sad/train.py`     | training CLI |
| `soft_sad/evaluate.py`  | evaluation + plots CLI |
| `tests/`                | sanity tests including the limiting-case equivalence with hard metrics |
