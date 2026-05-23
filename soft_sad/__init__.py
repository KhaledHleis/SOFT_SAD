"""Soft-SAD reference implementation.

Three frame labels are used throughout the pipeline:
    0 = SILENCE  (background / stationary noise)
    1 = SPEECH
    2 = NONSPEECH (human non-speech: cough, laugh, sneeze, ...)

These integer codes are exposed as module-level constants so every part of
the code uses the same convention.
"""

SILENCE = 0
SPEECH = 1
NONSPEECH = 2

LABEL_NAMES = {SILENCE: "silence", SPEECH: "speech", NONSPEECH: "nonspeech"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}
