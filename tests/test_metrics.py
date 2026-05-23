"""Unit tests. Run with: python -m pytest tests/ -v"""

import numpy as np

from soft_sad import NONSPEECH, SILENCE, SPEECH
from soft_sad.events import detect_events, extract_ground_truth_events
from soft_sad.membership import MembershipParams, hard_collar_membership, membership
from soft_sad.metrics_hard import hard_confusion
from soft_sad.metrics_soft import compute_event_confusion


# ---------------------------------------------------------------------
# Membership function
# ---------------------------------------------------------------------

def _params(t1=-15, t2=-5, t3=5, t4=25, K=100, s=2.0):
    return MembershipParams(t1=t1, t2=t2, t3=t3, t4=t4, K=K, steepness=s)


def test_membership_at_breakpoints():
    p = _params()
    mu = membership(np.array([p.t1, p.t2, 0, p.t3, p.t4]), p)
    np.testing.assert_allclose(mu, [0.0, 1.0, 1.0, 1.0, 0.01], rtol=1e-12, atol=1e-12)


def test_membership_plateau():
    p = _params()
    # everywhere on plateau, mu == 1
    for t in range(p.t2, p.t3 + 1):
        assert membership(t, p)[0] == 1.0


def test_membership_monotone_on_ramp_and_tail():
    p = _params()
    ramp = membership(np.arange(p.t1, p.t2 + 1), p)
    assert np.all(np.diff(ramp) >= -1e-12), "ramp must be non-decreasing"
    tail = membership(np.arange(p.t3, p.K + 1), p)
    assert np.all(np.diff(tail) <= 1e-12), "tail must be non-increasing"


def test_membership_outside_support_is_zero():
    p = _params()
    assert membership(-p.K - 1, p)[0] == 0.0
    assert membership(+p.K + 1, p)[0] == 0.0
    # also between -K and t1 the mu is zero
    assert membership(-p.K, p)[0] == 0.0
    assert membership(p.t1 - 1, p)[0] == 0.0


def test_hard_collar_membership_is_rectangle():
    half = 7
    d = np.arange(-20, 21)
    mu = hard_collar_membership(d, half)
    expected = (np.abs(d) <= half).astype(np.float64)
    np.testing.assert_array_equal(mu, expected)


def test_rectangular_membership_via_degenerate_params():
    """A degenerate MembershipParams should reproduce a rectangle."""
    half = 7
    p = MembershipParams(t1=-half, t2=-half, t3=+half, t4=+half, K=20, steepness=1.0)
    d = np.arange(-20, 21)
    mu = membership(d, p)
    expected = hard_collar_membership(d, half)
    np.testing.assert_array_equal(mu, expected)


# ---------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------

def test_detect_events_rising_edges_only():
    p = np.array([0.0, 0.1, 0.6, 0.7, 0.3, 0.8, 0.9, 0.2])
    # threshold 0.5: above at frames 2,3,5,6 -> rising edges at 2 and 5
    e = detect_events(p, 0.5)
    np.testing.assert_array_equal(e, [2, 5])


def test_detect_events_first_frame_counts():
    p = np.array([0.8, 0.7, 0.2, 0.9])
    e = detect_events(p, 0.5)
    np.testing.assert_array_equal(e, [0, 3])


def test_detect_events_min_gap_debounce():
    p = np.array([0.0, 0.6, 0.3, 0.6, 0.3, 0.6])
    e = detect_events(p, 0.5, min_gap_frames=3)
    # rising edges at 1, 3, 5 → after debounce: 1, 5
    np.testing.assert_array_equal(e, [1, 5])


def test_ground_truth_events_speech_nonspeech():
    # frames: S S S _ _ N N _ S _ N
    labels = np.array([1, 1, 1, 0, 0, 2, 2, 0, 1, 0, 2])
    gt = extract_ground_truth_events(labels)
    np.testing.assert_array_equal(gt["speech"], [0, 8])
    np.testing.assert_array_equal(gt["nonspeech"], [5, 10])
    assert gt["speech_intervals"] == [(0, 3), (8, 9)]
    assert gt["nonspeech_intervals"] == [(5, 7), (10, 11)]


# ---------------------------------------------------------------------
# Soft-SAD ≡ hard in the rectangular limit
# ---------------------------------------------------------------------

def test_soft_equals_hard_when_membership_is_rectangle():
    """Soft-SAD with a degenerate rectangular mu must equal hard event metrics.

    This is the limiting-case equivalence claimed in the paper.
    """
    rng = np.random.default_rng(0)
    # Build a random label stream
    T = 500
    labels = np.zeros(T, dtype=np.int64)
    for _ in range(8):
        c = int(rng.choice([SPEECH, NONSPEECH]))
        s = int(rng.integers(0, T - 30))
        e = s + int(rng.integers(10, 30))
        labels[s:e] = c
    gt = extract_ground_truth_events(labels)

    # Random detections
    n_det = int(rng.integers(5, 20))
    detections = np.sort(rng.choice(T, size=n_det, replace=False)).astype(np.int64)
    pred_labels = np.where(rng.random(T) > 0.5, SPEECH, NONSPEECH).astype(np.int64)

    collar = 7
    rect = MembershipParams(t1=-collar, t2=-collar, t3=+collar, t4=+collar, K=collar, steepness=1.0)

    # Disable dummy & rigorous_nonspeech for clean comparison
    soft = compute_event_confusion(
        detections=detections,
        speech_events=gt["speech"],
        nonspeech_events=gt["nonspeech"],
        nonspeech_intervals=gt["nonspeech_intervals"],
        speech_params=rect,
        rigorous_nonspeech=False,
        pred_labels=pred_labels,
        gt_labels=labels,
        enable_dummy=False,
    )
    hard = hard_confusion(
        detections=detections,
        speech_events=gt["speech"],
        nonspeech_events=gt["nonspeech"],
        nonspeech_intervals=gt["nonspeech_intervals"],
        collar_frames=collar,
    )

    # In the rectangular limit the two are identical.
    np.testing.assert_allclose(soft["TP"], hard["TP"], atol=1e-12)
    np.testing.assert_allclose(soft["FN"], hard["FN"], atol=1e-12)
    np.testing.assert_allclose(soft["TN"], hard["TN"], atol=1e-12)
    np.testing.assert_allclose(soft["FP"], hard["FP"], atol=1e-12)


# ---------------------------------------------------------------------
# Confusion-matrix integrity (row sums = number of events of each class)
# ---------------------------------------------------------------------

def test_confusion_row_sums():
    rng = np.random.default_rng(1)
    T = 400
    labels = np.zeros(T, dtype=np.int64)
    for _ in range(6):
        c = int(rng.choice([SPEECH, NONSPEECH]))
        s = int(rng.integers(0, T - 30)); e = s + int(rng.integers(8, 25))
        labels[s:e] = c
    gt = extract_ground_truth_events(labels)
    m_s, m_n = gt["speech"].size, gt["nonspeech"].size

    detections = np.sort(rng.choice(T, size=10, replace=False)).astype(np.int64)
    pred_labels = np.where(rng.random(T) > 0.3, SPEECH, NONSPEECH).astype(np.int64)
    p = _params()

    for rig in (True, False):
        soft = compute_event_confusion(
            detections=detections,
            speech_events=gt["speech"],
            nonspeech_events=gt["nonspeech"],
            nonspeech_intervals=gt["nonspeech_intervals"],
            speech_params=p,
            rigorous_nonspeech=rig,
            pred_labels=pred_labels,
            gt_labels=labels,
            enable_dummy=False,
        )
        # Eq. 3 of the paper: TP + FN = m_s, TN + FP = m_n
        np.testing.assert_allclose(soft["TP"] + soft["FN"], m_s, atol=1e-12)
        np.testing.assert_allclose(soft["TN"] + soft["FP"], m_n, atol=1e-12)


# ---------------------------------------------------------------------
# Dummy-classifier endpoints: at tau->0 we should reach (FAR, TAR) ≈ (1, 1).
# ---------------------------------------------------------------------

def test_dummy_classifier_drives_endpoints():
    """With pred_labels = SPEECH everywhere (tau->0), TAR should be high and
    FAR should be high too."""
    T = 300
    labels = np.zeros(T, dtype=np.int64)
    labels[20:60] = SPEECH
    labels[100:140] = NONSPEECH
    labels[180:220] = SPEECH
    labels[260:290] = NONSPEECH
    gt = extract_ground_truth_events(labels)

    # No detections at all (e.g. tau just below 1.0 -> nothing crosses).
    # And pred_labels says SPEECH everywhere (e.g. tau just above 0).
    detections = np.array([], dtype=np.int64)
    pred_labels = np.full(T, SPEECH, dtype=np.int64)
    p = _params()

    soft = compute_event_confusion(
        detections=detections,
        speech_events=gt["speech"],
        nonspeech_events=gt["nonspeech"],
        nonspeech_intervals=gt["nonspeech_intervals"],
        speech_params=p,
        rigorous_nonspeech=True,
        pred_labels=pred_labels,
        gt_labels=labels,
        enable_dummy=True,
    )
    m_s = gt["speech"].size; m_n = gt["nonspeech"].size

    # Every speech event got a virtual detection at its frame -> TP = m_s
    np.testing.assert_allclose(soft["TP"], m_s, atol=1e-12)
    # And every non-speech event got a virtual detection (pred != gt) -> FP = m_n
    np.testing.assert_allclose(soft["FP"], m_n, atol=1e-12)
    # So (TAR, FAR) = (1, 1) — the trivial "always speech" classifier.
    assert soft["TAR"] > 0.999
    assert soft["FAR"] > 0.999
