# Data format

This document specifies exactly what files the pipeline expects.

## Directory layout

```
data_root/
├── train.csv          # which utterances are training
├── val.csv            # which utterances are validation
├── test.csv           # which utterances are held-out test
├── wavs/
│   ├── <utt_id>.wav   # 16 kHz mono PCM 16-bit
│   └── ...
└── annotations/
    ├── <utt_id>.json
    └── ...
```

`utt_id` is any string with no spaces. All three CSVs share the same column
layout, see below.

## Split CSVs (`train.csv`, `val.csv`, `test.csv`)

One header line plus one row per utterance. Columns:

| column     | type    | description                                          |
|------------|---------|------------------------------------------------------|
| `utt_id`   | string  | matches `wavs/<utt_id>.wav` and `annotations/<utt_id>.json` |
| `duration` | float   | total duration in seconds (cross-check with wav)     |
| `source`   | string  | free-form, e.g. `librispeech`, `nonspeech7k`. Optional. |

Example `train.csv`:

```csv
utt_id,duration,source
ls_dev_clean_001,8.42,librispeech
ns7k_cough_017,3.10,nonspeech7k
tdk_yawn_004,4.55,tdk_internal
```

## Audio (`wavs/*.wav`)

* Format: WAV, **16 kHz**, mono, PCM 16-bit signed.
* If your sources are at other rates, resample first (e.g. `sox`).
* Multi-channel files must be downmixed to mono.

The sample rate is configurable in `config.yaml`, but the defaults (16 kHz, 13
MFCC, hop 16 ms, frame 32 ms / nfft=1024 / hop=512) match the paper.

## Annotations (`annotations/<utt_id>.json`)

Time intervals labelled `speech`, `nonspeech`, or `silence`. The whole
duration of the audio must be covered exactly once — no gaps, no overlaps —
otherwise `prepare.py` will refuse to build labels and tell you which
interval is wrong.

Schema:

```json
{
  "utt_id": "ns7k_cough_017",
  "duration": 3.10,
  "sample_rate": 16000,
  "intervals": [
    {"start": 0.00, "end": 0.42, "label": "silence"},
    {"start": 0.42, "end": 1.05, "label": "nonspeech", "category": "coughing"},
    {"start": 1.05, "end": 1.30, "label": "silence"},
    {"start": 1.30, "end": 2.15, "label": "nonspeech", "category": "coughing"},
    {"start": 2.15, "end": 3.10, "label": "silence"}
  ]
}
```

Field reference:

* `utt_id`   — must match the filename.
* `duration` — seconds, must equal the wav duration to within 1 frame.
* `sample_rate` — informational; the pipeline still respects `config.yaml`.
* `intervals[*].start`, `end` — seconds, half-open `[start, end)`.
* `intervals[*].label` — one of `speech`, `nonspeech`, `silence`.
* `intervals[*].category` — optional string. Only used to break the test-set
  histogram down by category (audio_books, coughing, laughing, etc.). Ignored
  during training.

## How labels become frames

A frame at time `t` (seconds) is labelled with the interval that contains
`t`. Frame centres are `(i + 0.5) * hop_length / sample_rate` for
`i = 0, 1, ...`. The number of frames `T` for an utterance of length `L`
samples is `T = 1 + (L - n_fft) // hop_length` (the standard
`torchaudio` convention).

## How frame labels become events

An *event* of class `c` is a rising edge: the smallest frame index `i` such
that `frame_label[i] == c` and `frame_label[i-1] != c`. The first frame
counts as a rising edge if its label is `c`. Two consecutive intervals of
the same class produce only one event (the first), as expected.

## Quick check

`python -m soft_sad.prepare --config config.yaml --check-only` will:

1. Walk every utterance in all three CSVs.
2. Confirm `wavs/<utt_id>.wav` exists and has the expected sample rate.
3. Confirm `annotations/<utt_id>.json` covers `[0, duration)` exactly.
4. Print a per-class duration summary and exit without writing features.

Fix any failures here before going further.

## Minimal toy dataset for smoke-testing

`python -m soft_sad.make_toy_data --out toy_data/` synthesises a 30-utterance
toy dataset (sine bursts for "speech", noise bursts for "nonspeech",
silence for the rest). Use it to verify the pipeline end-to-end without any
real audio. See `tests/test_pipeline.py`.
