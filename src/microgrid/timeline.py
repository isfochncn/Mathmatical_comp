"""Absolute timeline: one contiguous series of 10-minute intervals.

Why this module exists
----------------------
The 2026-09-11 rules separate three things that used to be one array:

* the source row's 144 values use interval **start** labels, so the last column
  (``0:00+1`` / ``24:00``) actually describes [next day 00:00, next day 00:10);
* a natural day's clock has its own 144 intervals, and ``D_{d,0}`` (the first
  10 minutes of a day) is carried over from the **previous source day's last
  column**;
* a result row covers ``[day 00:10, next day 00:10)``, which is the natural-day
  clock shifted by one interval and therefore reaches one interval into the
  next day.

Flattening everything onto one absolute 10-minute grid keyed by minutes since
2025-01-01 00:00 removes the ambiguity: every quantity has exactly one owner
and every bridge/extension is an explicit, recorded provenance flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .constants import DELTA_T_HOURS, N_INTERVAL
from .data_io import Attachment1, Attachment2, Attachment4, DataBundle, DataError
from .timeaxis import (
    DATE_2025_01_01,
    DATE_2025_12_31,
    MAX_EXTENSION_HOURS,
    N_SEQ,
    SOURCE_ATTACHMENT,
    SOURCE_BRIDGE_MISSING,
    SOURCE_CARRY_OVER,
    SOURCE_EXTENDED,
    clock_index_of_abs,
    seq_index_from_label,
)

MINUTES_PER_DAY = 24 * 60
INTERVALS_PER_DAY = N_INTERVAL


@dataclass(frozen=True)
class Series:
    """One absolute 10-minute series with per-interval provenance."""

    name: str
    values: np.ndarray          # (n_abs,)
    provenance: np.ndarray      # (n_abs,) of str, aligned with values
    abs_minute0: int = 0        # absolute minute of values[0]

    def window(self, abs_from: int, abs_to: int) -> np.ndarray:
        """Half-open window [abs_from, abs_to) in absolute minutes."""
        lo = (abs_from - self.abs_minute0) // 10
        hi = (abs_to - self.abs_minute0) // 10
        if lo < 0 or hi > self.values.size or hi < lo:
            raise DataError(
                f"{self.name} 窗口越界：[{abs_from}, {abs_to}) 超出可用范围 "
                f"[{self.abs_minute0}, {self.abs_minute0 + 10 * self.values.size})"
            )
        return self.values[lo:hi]

    def provenance_window(self, abs_from: int, abs_to: int) -> np.ndarray:
        lo = (abs_from - self.abs_minute0) // 10
        hi = (abs_to - self.abs_minute0) // 10
        return self.provenance[lo:hi]

    def value_at(self, abs_minute: int) -> float:
        return float(self.values[(abs_minute - self.abs_minute0) // 10])


@dataclass(frozen=True)
class Timeline:
    """All absolute series needed by planning, execution and evaluation."""

    day_index0: date                 # date of day index 0
    n_days_clock: int                # number of natural-day clocks with data
    demand_kwh: Series
    pv_kwh: Series
    price_yuan_per_kwh: Series
    load_kw: Series
    pv_kw: Series
    notes: list[str]

    # ------------------------------------------------------------------
    # Day-level accessors
    # ------------------------------------------------------------------

    def abs_minute_of(self, day: date, t: int) -> int:
        return (day - self.day_index0).days * MINUTES_PER_DAY + t * 10

    def clock_day_slice(self, day: date) -> tuple[int, int]:
        """Absolute [from, to) covering one natural day 00:00 -> 24:00."""
        base = (day - self.day_index0).days * MINUTES_PER_DAY
        return base, base + MINUTES_PER_DAY

    def result_row_slice(self, day: date) -> tuple[int, int]:
        """Absolute [from, to) covering this day's result row.

        Result rows cover [day 00:10, next day 00:10): one interval shorter at
        the front and one interval longer at the back than the natural day.
        """
        base = (day - self.day_index0).days * MINUTES_PER_DAY
        return base + 10, base + MINUTES_PER_DAY + 10

    def demand_clock_kwh(self, day: date) -> np.ndarray:
        lo, hi = self.clock_day_slice(day)
        return self.demand_kwh.window(lo, hi).copy()

    def pv_clock_kwh(self, day: date) -> np.ndarray:
        lo, hi = self.clock_day_slice(day)
        return self.pv_kwh.window(lo, hi).copy()

    def price_clock(self, day: date) -> np.ndarray:
        lo, hi = self.clock_day_slice(day)
        return self.price_yuan_per_kwh.window(lo, hi).copy()

    def result_demand_kwh(self, day: date) -> np.ndarray:
        lo, hi = self.result_row_slice(day)
        return self.demand_kwh.window(lo, hi).copy()

    def result_pv_kwh(self, day: date) -> np.ndarray:
        lo, hi = self.result_row_slice(day)
        return self.pv_kwh.window(lo, hi).copy()

    def result_price(self, day: date) -> np.ndarray:
        lo, hi = self.result_row_slice(day)
        return self.price_yuan_per_kwh.window(lo, hi).copy()

    def result_provenance(self, day: date) -> np.ndarray:
        lo, hi = self.result_row_slice(day)
        return self.demand_kwh.provenance_window(lo, hi).copy()

    # ------------------------------------------------------------------
    # Provenance queries
    # ------------------------------------------------------------------

    def bridge_notes(self) -> list[str]:
        n_missing = int(np.sum(self.demand_kwh.provenance == SOURCE_BRIDGE_MISSING))
        n_carry = int(np.sum(self.demand_kwh.provenance == SOURCE_CARRY_OVER))
        n_ext = int(np.sum(self.demand_kwh.provenance == SOURCE_EXTENDED))
        return [
            f"首段桥接：{n_carry} 个区间由上一源日尾值提供（自然日 t=0）",
            f"首段缺失（无上一源日）：{n_missing} 个区间，用同钟点历史均值冷启动估计",
            f"源外延伸：{n_ext} 个区间，超出附件范围，以历史同钟点均值标记为 {SOURCE_EXTENDED}",
        ]


# ==========================================================================
# Construction
# ==========================================================================


def _shift_series(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply the natural-day bridge to one (365, 144) source matrix.

    ``D_{d,0} = seq(d-1, 143)`` and ``D_{d,t} = seq(d, t-1)`` for t = 1..143.
    The first natural day has no previous source row, so its first interval is
    flagged ``bridge-missing`` and estimated afterwards from history.
    """
    n_days, n_seq = values.shape
    if n_seq != N_SEQ:
        raise DataError(f"{name} 每行应为 {N_SEQ} 段，实际 {n_seq}")
    out = np.zeros((n_days, INTERVALS_PER_DAY), dtype=np.float64)
    prov = np.empty((n_days, INTERVALS_PER_DAY), dtype=object)
    # t = 0 is carried over from the previous source row's last column
    out[0, 0] = 0.0
    prov[0, 0] = SOURCE_BRIDGE_MISSING
    if n_days > 1:
        out[1:, 0] = values[:-1, N_SEQ - 1]
        prov[1:, 0] = SOURCE_CARRY_OVER
    # t = 1..143 take columns 0..142 of the same source row
    out[:, 1:] = values[:, : N_SEQ - 1]
    prov[:, 1:] = SOURCE_ATTACHMENT
    return out, prov


def _fill_first_bridge_estimate(
    values: np.ndarray, prov: np.ndarray, history: np.ndarray, name: str
) -> None:
    """Estimate the missing first interval of the very first natural day.

    Source: the mean of the same clock interval over the **later** available
    source rows (a documented cold-start estimate). We must not shift the same
    day's measured 00:10 value forward, and we must not leave a silent zero.
    """
    mask = prov == SOURCE_BRIDGE_MISSING
    if not mask.any():
        return
    same_clock = history[:, 0]
    estimate = float(np.mean(same_clock))
    values[mask] = estimate


def build_timeline(
    bundle: DataBundle,
    *,
    extension_hours: int = MAX_EXTENSION_HOURS,
) -> Timeline:
    """Flatten attachments 2 and 4 onto one absolute grid.

    Parameters
    ----------
    extension_hours : hours of extra absolute intervals appended after the last
        source day, so that a window opened on 12-31 can look one day ahead
        without inventing measured data. Extended intervals use the historical
        same-clock mean and are flagged ``extended``.
    """
    a2: Attachment2 = bundle.attachment2
    a4: Attachment4 = bundle.attachment4
    a1: Attachment1 = bundle.attachment1

    n_days = len(a2.days)
    if n_days != len(a4.days) or a2.days != a4.days:
        raise DataError("附件2 与附件4 的日期序列不一致")

    load_clock, load_prov = _shift_series(a2.load_kw, "附件2 负载")
    pv_clock, pv_prov = _shift_series(a2.pv_actual_kw, "附件2 光伏")
    price_clock, price_prov = _shift_series(a4.price_yuan_per_kwh, "附件4 电价")

    notes: list[str] = []
    _fill_first_bridge_estimate(load_clock, load_prov, a2.load_kw, "负载")
    _fill_first_bridge_estimate(pv_clock, pv_prov, a2.pv_actual_kw, "光伏")
    _fill_first_bridge_estimate(
        price_clock, price_prov, a4.price_yuan_per_kwh, "电价"
    )

    # ---- extension beyond the last source day -------------------------------
    n_ext = int(round(extension_hours * 60 / 10))
    if n_ext < 0:
        raise DataError("extension_hours 不能为负")

    def _extend(values: np.ndarray, prov: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        if n_ext == 0:
            flat_v = values.reshape(-1)
            flat_p = prov.reshape(-1)
            return flat_v, flat_p
        # Same-clock mean over the whole source year, per clock interval.
        clock_mean = values.mean(axis=0)
        ext_v = np.tile(clock_mean, n_ext // INTERVALS_PER_DAY + 1)[:n_ext]
        ext_p = np.full(n_ext, SOURCE_EXTENDED, dtype=object)
        flat_v = np.concatenate([values.reshape(-1), ext_v])
        flat_p = np.concatenate([prov.reshape(-1), ext_p])
        return flat_v, flat_p

    flat_load, flat_load_prov = _extend(load_clock, load_prov, "负载")
    flat_pv, flat_pv_prov = _extend(pv_clock, pv_prov, "光伏")
    flat_price, flat_price_prov = _extend(price_clock, price_prov, "电价")

    demand_series = Series("demand_kwh", flat_load * DELTA_T_HOURS, flat_load_prov)
    pv_series = Series("pv_kwh", flat_pv * DELTA_T_HOURS, flat_pv_prov)
    price_series = Series("price_yuan_per_kwh", flat_price, flat_price_prov)
    load_series = Series("load_kw", flat_load, flat_load_prov)
    pv_kw_series = Series("pv_kw", flat_pv, flat_pv_prov)

    notes.extend(
        [
            f"绝对时间线覆盖 {n_days} 个自然日（{a2.days[0]} .. {a2.days[-1]}）"
            f"加 {extension_hours} 小时延伸",
            "源标签按区间起点解释；自然日 t=0 由上一源日尾值桥接",
        ]
    )

    return Timeline(
        day_index0=a2.days[0],
        n_days_clock=n_days,
        demand_kwh=demand_series,
        pv_kwh=pv_series,
        price_yuan_per_kwh=price_series,
        load_kw=load_series,
        pv_kw=pv_kw_series,
        notes=notes,
    )


def source_horizon_last_abs(timeline: Timeline) -> int:
    """Absolute minute of the end of the last fully measured source interval."""
    return timeline.n_days_clock * MINUTES_PER_DAY


__all__ = [
    "Series",
    "Timeline",
    "build_timeline",
    "source_horizon_last_abs",
    "MINUTES_PER_DAY",
    "INTERVALS_PER_DAY",
]
