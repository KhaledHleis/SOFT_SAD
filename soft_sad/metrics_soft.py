"""Soft-SAD event metrics.

Implements the event-symmetric soft confusion matrix from Eq. 3 of the
paper, together with:
  - bipartite matching between detections and events (Sec. IV-B);
  - the dummy-classifier fallback (Sec. IV-D) that inserts a virtual
    detection at the event time whenever the frame-level prediction
    agrees with the event's class but no detection landed inside the
    membership support;
  - the rigorous non-speech membership (Sec. IV-E) where each non-speech
    event has a rectangular membership covering its entire annotated
    extent.

Conventions
-----------
A *detection* is a frame index (rising edge of the thresholded speech
probability stream). An *event* of class c is a frame index (rising edge
of the ground-truth label stream into class c). Membership functions are
class-specific: mu_S for speech events, mu_N for non-speech events.

Symbols matching the paper:
    d_sp(d_i) = max_{e in E^S} mu_S_e(t_{d_i})
    d_ns(d_i) = max_{e in E^N} mu_N_e(t_{d_i})
    TP_s = sum_{e in E^S} mu_S_e(d_hat_e)
    FN_s = |E^S| - TP_s
    TN_s = sum_{e in E^N} (1 - d_sp(d_hat_e))
    FP_s = |E^N| - TN_s
where d_hat_e = argmax_d mu_e(t_d) under the class-appropriate mu.

In rigorous mode, mu_N_e is the indicator of the labelled extent of e, so
any detection inside the non-speech interval scores 1 against e.

Returns
-------
A `dict` with keys TP, FN, TN, FP, plus precision, recall, F1, TAR, FAR
and the lists of (event, attributed_detection) pairs for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .membership import MembershipParams, membership


# ---------------------------------------------------------------------
# Per-detection class scores (d_sp, d_ns).
# ---------------------------------------------------------------------

def _per_detection_class_scores(
    det_frames: np.ndarray,        # (n,)
    event_frames: np.ndarray,      # (m,)
    p: MembershipParams,
) -> np.ndarray:                   # (n,) = max_e mu_e(d - e)
    """For each detection, the max membership over events of one class."""
    if det_frames.size == 0:
        return np.zeros(0, dtype=np.float64)
    if event_frames.size == 0:
        return np.zeros(det_frames.size, dtype=np.float64)
    # broadcast (n, m): delta[i, j] = det[i] - event[j]
    delta = det_frames[:, None] - event_frames[None, :]
    mu = membership(delta, p)        # (n, m)
    return mu.max(axis=1)            # (n,)


def _per_detection_class_scores_rigorous(
    det_frames: np.ndarray,           # (n,)
    intervals: list[tuple[int, int]], # [(onset, offset), ...]
) -> np.ndarray:
    """Rigorous version: mu_N is the indicator of [onset, offset).

    Each detection scores 1 against a non-speech event if it falls within
    that event's labelled extent, else 0. Max over events of the class.
    """
    if det_frames.size == 0 or not intervals:
        return np.zeros(det_frames.size, dtype=np.float64)
    scores = np.zeros(det_frames.size, dtype=np.float64)
    for (on, off) in intervals:
        inside = (det_frames >= on) & (det_frames < off)
        scores = np.maximum(scores, inside.astype(np.float64))
    return scores


# ---------------------------------------------------------------------
# Per-event best detection (d_hat_e).
# ---------------------------------------------------------------------

def _best_detection_per_event(
    det_frames: np.ndarray,        # (n,)
    event_frames: np.ndarray,      # (m,)
    p: MembershipParams,
) -> tuple[np.ndarray, np.ndarray]:
    """For each event, the best score and the index of its best detection.

    Returns (best_scores[m], best_det_idx[m]).
    `best_det_idx == -1` means no detection is in this event's support.
    """
    m = event_frames.size
    if m == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    if det_frames.size == 0:
        return np.zeros(m), np.full(m, -1, dtype=np.int64)
    delta = det_frames[:, None] - event_frames[None, :]   # (n, m)
    mu = membership(delta, p)                              # (n, m)
    best = mu.max(axis=0)                                  # (m,)
    arg = mu.argmax(axis=0)                                # (m,)
    arg = np.where(best > 0, arg, -1).astype(np.int64)
    return best, arg


def _best_detection_per_event_rigorous(
    det_frames: np.ndarray,
    intervals: list[tuple[int, int]],
    speech_scores_of_detections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigorous non-speech: each event keeps the detection inside its
    interval that has the highest *speech* score (worst-case for TN).

    If no detection falls in the interval, no detection is attributed
    (best_score = 0 for the non-speech-membership, det_idx = -1).
    """
    m = len(intervals)
    best = np.zeros(m, dtype=np.float64)
    arg = np.full(m, -1, dtype=np.int64)
    for j, (on, off) in enumerate(intervals):
        inside = np.where((det_frames >= on) & (det_frames < off))[0]
        if inside.size == 0:
            continue
        # Membership score against this non-speech event is 1 for all
        # inside detections by definition; pick the worst case for the
        # model (highest d_sp).
        idx_local = inside[np.argmax(speech_scores_of_detections[inside])]
        best[j] = 1.0
        arg[j] = int(idx_local)
    return best, arg


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def compute_event_confusion(
    detections: np.ndarray,                # (n,) frame indices
    speech_events: np.ndarray,             # (m_s,) frame indices
    nonspeech_events: np.ndarray,          # (m_n,) frame indices
    nonspeech_intervals: list[tuple[int, int]],  # [(onset, offset), ...]
    *,
    speech_params: MembershipParams,
    nonspeech_params: MembershipParams | None = None,
    rigorous_nonspeech: bool = True,
    pred_labels: np.ndarray | None = None,
    gt_labels: np.ndarray | None = None,
    enable_dummy: bool = True,
) -> dict:
    """Soft event confusion matrix (Eq. 3 of the paper).

    Parameters
    ----------
    detections : array of int
        Frame indices of detected events (rising edges).
    speech_events, nonspeech_events : array of int
        Frame indices of ground-truth speech / non-speech events.
    nonspeech_intervals : list of (onset, offset)
        Frame extent of each non-speech event. Required if
        `rigorous_nonspeech=True`.
    speech_params : MembershipParams
        mu_S parameters.
    nonspeech_params : MembershipParams, optional
        mu_N parameters when `rigorous_nonspeech=False`. Defaults to
        `speech_params` (symmetric soft scoring).
    rigorous_nonspeech : bool
        If True, mu_N is the rectangular indicator of each non-speech
        event's extent (Sec. IV-E of the paper).
    pred_labels, gt_labels : array of int, optional
        Frame-level predicted / ground-truth labels. Required if
        `enable_dummy=True` to compute the virtual-detection fallback.
    enable_dummy : bool
        If True, apply the frame-level dummy-classifier fallback
        (Sec. IV-D). Disabled for hard metrics.

    Returns
    -------
    dict with keys: TP, FN, TN, FP, P, R, F1, TAR, FAR, n_speech_events,
    n_nonspeech_events, match_speech, match_nonspeech.
    """
    detections = np.asarray(detections, dtype=np.int64)
    speech_events = np.asarray(speech_events, dtype=np.int64)
    nonspeech_events = np.asarray(nonspeech_events, dtype=np.int64)
    if nonspeech_params is None:
        nonspeech_params = speech_params

    m_s = int(speech_events.size)
    m_n = int(nonspeech_events.size)

    # ---- Dummy-classifier fallback (Sec. IV-D) ----
    # Insert a virtual detection at each event whose frame-level prediction
    # already matches its class label. This is independent of, and applied
    # before, the bipartite matching.
    detections_eff = detections
    if enable_dummy:
        if pred_labels is None or gt_labels is None:
            raise ValueError(
                "enable_dummy=True requires pred_labels and gt_labels."
            )
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        T = pred_labels.size

        # Insert virtual detections at every event where the frame-level
        # prediction agrees with the event's class. We do NOT guard against
        # the existence of a nearby real detection: at low tau the "real"
        # detection is often a useless early rising edge that scores 0 in
        # the membership function. Bipartite matching takes the max over
        # all candidate detections per event, so adding a virtual at score 1
        # is harmless when a good real detection already exists (max = 1)
        # but rescues the event when the real detection is out-of-support
        # or sits in the zero-credit region.
        #
        # The asymmetry between speech and non-speech (pred == gt vs
        # pred != gt) is intentional. At a speech event the virtual fires
        # when the model is "currently calling this frame speech" (real
        # TP-ish behaviour). At a non-speech event the virtual fires when
        # the model is "currently calling this frame speech" too -- i.e.
        # currently making the operational FP error. Both reduce to
        # "pred_labels[e] == SPEECH" in the SAD binary case, which is
        # exactly what we want as tau -> 0 (FAR -> 1, TAR -> 1).
        virt = []
        for e in speech_events:
            if 0 <= e < T and pred_labels[e] == gt_labels[e]:
                virt.append(int(e))
        for e in nonspeech_events:
            if 0 <= e < T and pred_labels[e] != gt_labels[e]:
                virt.append(int(e))

        if virt:
            detections_eff = np.sort(np.concatenate([detections, np.array(virt, dtype=np.int64)]))

    # ---- Bipartite matching for speech events ----
    best_s, arg_s = _best_detection_per_event(
        detections_eff, speech_events, speech_params
    )

    # ---- d_sp for each detection (for use in TN) ----
    dsp = _per_detection_class_scores(
        detections_eff, speech_events, speech_params
    )

    # ---- Bipartite matching for non-speech events ----
    if rigorous_nonspeech:
        best_n_mu, arg_n = _best_detection_per_event_rigorous(
            detections_eff, nonspeech_intervals, dsp
        )
    else:
        best_n_mu, arg_n = _best_detection_per_event(
            detections_eff, nonspeech_events, nonspeech_params
        )

    # ---- Soft confusion matrix entries (Eq. 3) ----
    # TP_s: sum of speech-membership scores at each speech event's best detection.
    TP_s = float(best_s.sum())
    FN_s = float(m_s - TP_s)

    # TN/FP: depends on the non-speech membership flavour.
    #
    # Rigorous mode (mu_N is the indicator of the non-speech interval):
    #   "the model fired inside this non-speech window" is a binary event,
    #   exactly captured by best_n_mu[j] in {0, 1}. So
    #       TN_contrib = 1 - best_n_mu[j]
    #       FP_contrib = best_n_mu[j]
    #
    # Graded (non-rigorous) mode:
    #   We apply Eq. 3 of the paper. The attributed detection d_hat_e
    #   for each non-speech event has its FP weight given by its speech
    #   membership d_sp; the residual goes to TN. If no detection is
    #   attributed to the event (best_n_mu[j] == 0), the model correctly
    #   ignored the event and TN_contrib = 1.
    tn_contrib = np.zeros(m_n, dtype=np.float64)
    if rigorous_nonspeech:
        tn_contrib = 1.0 - best_n_mu
    else:
        for j in range(m_n):
            if arg_n[j] >= 0:
                tn_contrib[j] = 1.0 - dsp[arg_n[j]]
            else:
                tn_contrib[j] = 1.0
    TN_s = float(tn_contrib.sum())
    FP_s = float(m_n - TN_s)

    # ---- Derived rates ----
    eps = 1e-12
    P = TP_s / max(TP_s + FP_s, eps)
    R = TP_s / max(TP_s + FN_s, eps)
    F1 = 2 * P * R / max(P + R, eps)
    TAR = R
    FAR = FP_s / max(FP_s + TN_s, eps)

    return {
        "TP": TP_s, "FN": FN_s, "TN": TN_s, "FP": FP_s,
        "P": P, "R": R, "F1": F1, "TAR": TAR, "FAR": FAR,
        "n_speech_events": m_s,
        "n_nonspeech_events": m_n,
        "n_detections": int(detections.size),
        "n_virtual_detections": int(detections_eff.size - detections.size),
        # Per-event diagnostics
        "match_speech": list(zip(
            speech_events.tolist(),
            best_s.tolist(),
            arg_s.tolist(),
        )),
        "match_nonspeech": list(zip(
            nonspeech_events.tolist(),
            tn_contrib.tolist(),
            arg_n.tolist(),
        )),
    }


# ---------------------------------------------------------------------
# Convenience: aggregate per-utterance confusion matrices.
# ---------------------------------------------------------------------

def aggregate_confusions(confs: list[dict]) -> dict:
    """Sum per-utterance confusion matrices and recompute derived rates."""
    if not confs:
        return dict(TP=0, FN=0, TN=0, FP=0, P=0.0, R=0.0, F1=0.0, TAR=0.0, FAR=0.0)
    TP = sum(c["TP"] for c in confs)
    FN = sum(c["FN"] for c in confs)
    TN = sum(c["TN"] for c in confs)
    FP = sum(c["FP"] for c in confs)
    eps = 1e-12
    P = TP / max(TP + FP, eps)
    R = TP / max(TP + FN, eps)
    F1 = 2 * P * R / max(P + R, eps)
    return dict(
        TP=TP, FN=FN, TN=TN, FP=FP,
        P=P, R=R, F1=F1,
        TAR=R, FAR=FP / max(FP + TN, eps),
    )
