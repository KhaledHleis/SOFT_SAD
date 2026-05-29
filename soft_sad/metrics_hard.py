"""Hard event-based metrics (rectangular collar) for the SAD setting.

Implements the limiting case of Soft-SAD with a rectangular membership
function — i.e. the Mesaros et al. 2016 collar-based event metric, adapted
to the two-event-class SAD problem with silence/noise deferred to a
diagnostic. See `metrics_soft.py` for the full soft version; this module
is a thin convenience wrapper that calls into the same matching code with
a rectangular membership.

No changes were needed here for the data-size scaling feature: all
evaluation logic stays in metrics_soft.py / evaluate.py.
"""
from __future__ import annotations

import numpy as np

from soft_sad.membership import MembershipParams
from soft_sad.metrics_soft import compute_event_confusion


def hard_confusion(
    detections: np.ndarray,
    speech_events: np.ndarray,
    nonspeech_events: np.ndarray,
    nonspeech_intervals: list[tuple[int, int]],
    *,
    collar_frames: int,
    pred_labels: np.ndarray | None = None,
    gt_labels: np.ndarray | None = None,
) -> dict:
    """Hard event-based confusion matrix using a rectangular collar.

    Same call signature as `compute_event_confusion`; we just override the
    membership function to be rectangular.
    """
    # Build a degenerate MembershipParams that produces a rectangle.
    K = max(collar_frames, 1)
    p = MembershipParams(
        t1=-collar_frames, t2=-collar_frames,
        t3=+collar_frames, t4=+collar_frames,
        K=K, steepness=1.0,
    )
    return compute_event_confusion(
        detections=detections,
        speech_events=speech_events,
        nonspeech_events=nonspeech_events,
        nonspeech_intervals=nonspeech_intervals,
        speech_params=p,
        rigorous_nonspeech=True,
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        enable_dummy=True,         # hard metrics: no virtual-detection rescue
    )