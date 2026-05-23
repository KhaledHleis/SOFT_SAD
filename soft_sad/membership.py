"""Piecewise asymmetric membership function from Eq. 4 of the paper.

For an event at frame `t_e`, the score of a detection at frame `t_d` is
        mu(t_d - t_e)
where mu is defined on [-K, K] with breakpoints t1 <= t2 <= 0 <= t3 <= t4:

                 0,                                  t in [-K, t1)
                (10^{s (t-t1)/(t2-t1)} - 1)/(10^s-1) t in [t1, t2)
    mu(t)  =     1,                                  t in [t2, t3]
                10^{-s (t-t3)/(t4-t3)},              t in (t3, K]
                 0,                                  t > K

All breakpoints here are in *frame* units, integer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MembershipParams:
    t1: int      # start of ramp        (negative)
    t2: int      # end of ramp / start of plateau  (<= 0)
    t3: int      # end of plateau / start of decay (>= 0)
    t4: int      # end of decay tail               (positive)
    K: int       # support half-width              (>= |t1|, >= t4)
    steepness: float = 2.0

    @classmethod
    def from_ms(
        cls,
        t1_ms: float, t2_ms: float, t3_ms: float, t4_ms: float, K_ms: float,
        steepness: float,
        hop_length: int, sample_rate: int,
    ) -> "MembershipParams":
        """Build from milliseconds, given the audio frame rate."""
        f = sample_rate / hop_length / 1000.0  # frames per ms
        def to_frames(x_ms: float) -> int:
            return int(round(x_ms * f))
        params = cls(
            t1=to_frames(t1_ms),
            t2=to_frames(t2_ms),
            t3=to_frames(t3_ms),
            t4=to_frames(t4_ms),
            K=to_frames(K_ms),
            steepness=float(steepness),
        )
        params._validate()
        return params

    def _validate(self):
        assert self.t1 <= self.t2 <= 0 <= self.t3 <= self.t4
        assert self.K >= max(-self.t1, self.t4)
        assert self.steepness > 0


def membership(delta: np.ndarray | int, p: MembershipParams) -> np.ndarray:
    """Evaluate mu(delta) elementwise (delta = t_d - t_e in frames).

    Returns float64 in [0, 1].
    """
    d = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    out = np.zeros_like(d)
    s = p.steepness

    # ramp: t1 <= t < t2
    if p.t2 > p.t1:
        mask = (d >= p.t1) & (d < p.t2)
        z = s * (d[mask] - p.t1) / (p.t2 - p.t1)
        out[mask] = (np.power(10.0, z) - 1.0) / (np.power(10.0, s) - 1.0)
    else:
        # degenerate ramp: skip
        pass

    # plateau: t2 <= t <= t3
    mask = (d >= p.t2) & (d <= p.t3)
    out[mask] = 1.0

    # decay tail: t3 < t <= K
    if p.t4 > p.t3:
        mask = (d > p.t3) & (d <= p.K)
        z = -s * (d[mask] - p.t3) / (p.t4 - p.t3)
        out[mask] = np.power(10.0, z)
        # clamp to 0 past t4 if user wants exactly-zero past the tail
        # (we don't — the decay continues smoothly until K, by design)

    # outside support
    mask = (d < -p.K) | (d > p.K)
    out[mask] = 0.0

    # also: if d < t1 (still in support but before ramp), zero
    mask = (d >= -p.K) & (d < p.t1)
    out[mask] = 0.0

    return out


def hard_collar_membership(delta, half_width: int) -> np.ndarray:
    """Rectangular membership of half-width `half_width` (frames)."""
    d = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    return (np.abs(d) <= half_width).astype(np.float64)
