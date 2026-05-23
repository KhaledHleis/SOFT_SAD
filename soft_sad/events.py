"""Turn frame-level signals into discrete events (rising-edge frame indices).

Two functions:
    detect_events(prob, threshold, ...)   — for the model's probability stream
    extract_ground_truth_events(labels)   — for the ground-truth label array

An *event of class c* is a rising edge into class c: the smallest frame `i`
such that `signal[i] == c` and (`i == 0` or `signal[i-1] != c`). For the
probability stream `signal[i] = (prob[i] >= threshold)`, so events are
rising edges into "speech-positive".

Optional debouncing (`min_gap_frames`) merges rising edges that occur too
close together — useful when the probability hovers near the threshold and
chatters.

`smoothing_frames > 1` applies a centred boxcar to the probability before
thresholding.
"""
from __future__ import annotations

import numpy as np

from . import NONSPEECH, SPEECH


# ---------------------------------------------------------------------
# Detector-side
# ---------------------------------------------------------------------

def _boxcar_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    # 'same' convolution with reflective padding to avoid edge artefacts
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w, dtype=x.dtype) / w
    return np.convolve(xp, k, mode="valid")[: len(x)]


def detect_events(
    prob: np.ndarray,
    threshold: float,
    *,
    min_gap_frames: int = 0,
    smoothing_frames: int = 1,
) -> np.ndarray:
    """Return the array of detection frame indices (rising edges).

    Parameters
    ----------
    prob : (T,) float in [0, 1]
        Per-frame speech probability.
    threshold : float
        Decision threshold tau in [0, 1].
    min_gap_frames : int
        If > 0, suppress a rising edge that occurs within this many frames
        of the previous one.
    smoothing_frames : int
        If > 1, smooth `prob` with a centred boxcar of this width before
        thresholding.
    """
    p = _boxcar_smooth(np.asarray(prob, dtype=np.float64), smoothing_frames)
    above = (p >= threshold).astype(np.int8)
    # rising edge: above[i] - above[i-1] == 1; the first frame counts if above.
    diff = np.diff(np.concatenate([[0], above]))
    edges = np.where(diff == 1)[0]

    if min_gap_frames > 0 and edges.size > 1:
        kept = [int(edges[0])]
        for e in edges[1:]:
            if e - kept[-1] >= min_gap_frames:
                kept.append(int(e))
        edges = np.asarray(kept, dtype=np.int64)
    return edges.astype(np.int64)


# ---------------------------------------------------------------------
# Ground-truth side
# ---------------------------------------------------------------------

def extract_ground_truth_events(labels: np.ndarray) -> dict:
    """Return rising-edge frame indices for each event-bearing class.

    Output:
        {
          "speech":    np.ndarray of int,   # rising edges into SPEECH
          "nonspeech": np.ndarray of int,   # rising edges into NONSPEECH
        }

    Also returns onset/offset interval pairs per class so the rigorous
    non-speech membership function knows where the event extents are:
        {
          "speech_intervals":    [(onset_idx, offset_idx), ...],
          "nonspeech_intervals": [(onset_idx, offset_idx), ...],
        }
    """
    labels = np.asarray(labels, dtype=np.int64)
    out: dict = {}
    intervals: dict = {}
    for cls, key in [(SPEECH, "speech"), (NONSPEECH, "nonspeech")]:
        mask = (labels == cls).astype(np.int8)
        diff = np.diff(np.concatenate([[0], mask, [0]]))  # extra 0 at both ends
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        out[key] = starts.astype(np.int64)
        intervals[key + "_intervals"] = list(zip(starts.tolist(), ends.tolist()))
    out.update(intervals)
    return out
