"""Time-axis mappings: source labels <-> clock intervals <-> result-row cells.

**This module is the ONLY place in the project allowed to do time arithmetic.**
Every other module must call these helpers; nobody may derive segment indices
from string slicing or floating-point hours (that is how 144/145 misalignment,
and the earlier "template is shifted" misreading, both happened).

Four coordinate systems
-----------------------
1. ``seq`` (source sequence interval)  v = 0..143 for a source row of
   144 values. Labels are **interval start times**:
   ``00:10`` -> [00:10, 00:20) -> v = 0;  ``23:50`` -> [23:50, 00:00) -> v = 143.
   A value labelled ``0:00+1`` or ``24:00`` is [next day 00:00, next day 00:10).

2. ``clock`` (natural-day clock)  t = 0..143 for a natural day, meaning
   [day 00:00 + t*10min, day 00:00 + (t+1)*10min).  Boundaries ``E_{d,s}``
   for s = 0..144 are instants, with ``E_{d,0}`` = day 00:00 and
   ``E_{d,144}`` = next day 00:00.

   Bridge rule: the source carries the value of the previous source day's LAST
   10 minutes into the next day's FIRST interval, i.e.
       ``D_{d,0} = seq_value(d-1, 143)``  (the ``0:00+1`` column),
       ``D_{d,t} = seq_value(d, t-1)``    for t = 1..143.

3. ``result`` (output row)  j = 0..143 with boundaries ``B_{d,j}``, j = 0..144.
   ``B_{d,j} = E_{d, j+1}`` because result rows cover
   ``[day 00:10, next day 00:10)``.
   Consequently ``B_{d,0} = E_{d,1}``, ``B_{d,143} = E_{d,144} = E_{d+1,0}``,
   and ``B_{d,144} = E_{d+1,1}`` -- which is **NOT** equal to ``B_{d+1,0}``:
   they differ by exactly the next day's first interval of energy exchange.
   Treating them as equal would execute that interval twice.

4. Template cells: the template's 144 time columns ARE the result rows, so the
   mapping ``result j -> template data slot j`` is the identity. The template
   was never "shifted"; the earlier reading came from treating labels as
   interval END times.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import numpy as np

from .constants import MINUTES_PER_INTERVAL, N_BOUNDARY, N_INTERVAL

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Sequence length of one source row (attachment 1/2/4) and of one result row.
N_SEQ = N_INTERVAL                      # 144
#: Boundaries of one natural day / one result row.
N_SEQ_BOUNDARY = N_BOUNDARY             # 145

#: Label of the last source column: [next day 00:00, next day 00:10).
SEQ_START_LABEL = "0:00+1"
#: The same instant written in the task statement's 24:00 notation.
SEQ_TAIL_LABEL = "24:00"
#: Task-statement notation for the start of a natural day (not a source label).
NATURAL_DAY_ZERO_LABEL = "0:00"

#: A source series may only extend this many hours past 2025-12-31 to serve as
#: the next-day lookahead of the 12-31 optimisation window.
MAX_EXTENSION_HOURS = 48

_TIME_LABEL_RE = re.compile(r"^(\d{1,2}):(\d{2})(\+1)?$")

#: Extension provenance markers (never present them as measured data).
SOURCE_ATTACHMENT = "attachment"
SOURCE_CARRY_OVER = "carry-over"        # first interval bridged from previous source day
SOURCE_BRIDGE_MISSING = "bridge-missing"  # first interval of the very first day
SOURCE_EXTENDED = "extended"            # beyond the source horizon (forecast only)


# --------------------------------------------------------------------------
# Source label parsing
# --------------------------------------------------------------------------


def seq_index_from_label(label: object) -> int:
    """Map one source label to its sequence interval index.

    Labels are interval START times, so ``00:10`` -> 0 and ``23:50`` -> 143.
    ``0:00+1`` / ``24:00`` denote [next day 00:00, next day 00:10) -> 143.
    """
    if isinstance(label, time):
        return _minutes_to_seq(label.hour * 60 + label.minute)
    text = str(label).strip()
    if text in (SEQ_START_LABEL, SEQ_TAIL_LABEL):
        return N_SEQ - 1
    m = _TIME_LABEL_RE.match(text)
    if not m:
        raise ValueError(f"无法解析源时间标签：{label!r}")
    # 'H:MM+1' other than 0:00+1 is out of contract.
    if m.group(3) is not None:
        raise ValueError(f"仅允许 '{SEQ_START_LABEL}' 使用 +1 后缀，收到 {label!r}")
    return _minutes_to_seq(int(m.group(1)) * 60 + int(m.group(2)))


def _minutes_to_seq(minutes: int) -> int:
    minutes %= 24 * 60
    if minutes < MINUTES_PER_INTERVAL:
        raise ValueError(
            f"{minutes} 分钟不对应任何源交易标签：上一源日尾值承担次日 00:00—00:10"
        )
    if minutes % MINUTES_PER_INTERVAL:
        raise ValueError(f"源标签不是 {MINUTES_PER_INTERVAL} 分钟整数倍：{minutes}")
    return minutes // MINUTES_PER_INTERVAL - 1


def seq_label(v: int) -> str:
    """Sequence interval index -> canonical start-time label."""
    _check_seq(v)
    if v == N_SEQ - 1:
        return SEQ_START_LABEL
    minutes = (v + 1) * MINUTES_PER_INTERVAL
    return f"{minutes // 60}:{minutes % 60:02d}"


def clock_label(t: int) -> str:
    """Natural-day clock interval index -> 'HH:MM-HH:MM' (start-end)."""
    _check_clock(t)
    lo = t * MINUTES_PER_INTERVAL
    hi = lo + MINUTES_PER_INTERVAL
    return f"{_fmt_minutes(lo)}-{_fmt_minutes(hi)}"


def _fmt_minutes(minutes: int) -> str:
    if minutes >= 24 * 60:
        return SEQ_TAIL_LABEL
    return f"{minutes // 60}:{minutes % 60:02d}"


def _check_seq(v: int) -> None:
    if not 0 <= v < N_SEQ:
        raise ValueError(f"源序列下标越界：{v}")


def _check_clock(t: int) -> None:
    if not 0 <= t < N_INTERVAL:
        raise ValueError(f"自然日区间下标越界：{t}")


# --------------------------------------------------------------------------
# Coordinate conversions
# --------------------------------------------------------------------------


def clock_to_abs_minute(day_index: int, t: int) -> int:
    """(day index, clock interval) -> minutes since 2025-01-01 00:00."""
    _check_clock(t)
    return day_index * 24 * 60 + t * MINUTES_PER_INTERVAL


def boundary_abs_minute(day_index: int, s: int) -> int:
    """(day index, boundary s in 0..144) -> minutes since 2025-01-01 00:00."""
    if not 0 <= s <= N_BOUNDARY:
        raise ValueError(f"边界下标越界：{s}")
    return day_index * 24 * 60 + s * MINUTES_PER_INTERVAL


def clock_index_of_abs(abs_minute: int) -> tuple[int, int]:
    """Absolute minute -> (day index, clock interval)."""
    if abs_minute % MINUTES_PER_INTERVAL:
        raise ValueError(f"绝对时间不是 {MINUTES_PER_INTERVAL} 分钟整数倍：{abs_minute}")
    day_index, rem = divmod(abs_minute, 24 * 60)
    return day_index, rem // MINUTES_PER_INTERVAL


#: Result row covers [day 00:10, next day 00:10): its boundaries sit at
#: clock boundary index j+1.
RESULT_BOUNDARY_OFFSET = 1


def result_boundary_clock_index(j: int) -> int:
    """Result-row boundary j -> natural-day clock boundary index."""
    if not 0 <= j <= N_SEQ_BOUNDARY - 1:
        raise ValueError(f"result 边界下标越界：{j}")
    return j + RESULT_BOUNDARY_OFFSET


def result_interval_clock_index(j: int) -> int:
    """Result-row interval j covers clock interval j+1 of the same day.

    For j = 143 this is clock interval 144, i.e. **the next day's first
    interval** ([next day 00:00, next day 00:10)).
    """
    if not 0 <= j < N_SEQ:
        raise ValueError(f"result 区间下标越界：{j}")
    return j + RESULT_BOUNDARY_OFFSET


def result_row_spans_next_day(j: int) -> bool:
    """True if result interval j belongs to the next natural day."""
    return result_interval_clock_index(j) >= N_INTERVAL


# --------------------------------------------------------------------------
# Template cells
# --------------------------------------------------------------------------
# The template's 144 time columns are exactly the 144 result intervals, so the
# mapping is the identity. Kept as an explicit function so that any future
# change has exactly one place to live.
TEMPLATE_TIME_COLUMNS = N_SEQ


def template_column_of_result(result_index: int) -> int:
    """Result interval index -> template time-column ordinal (0-based)."""
    if not 0 <= result_index < N_SEQ:
        raise ValueError(f"result 区间下标越界：{result_index}")
    return result_index


def result_of_template_column(col: int) -> int:
    """Template time-column ordinal (0-based) -> result interval index."""
    if not 0 <= col < TEMPLATE_TIME_COLUMNS:
        raise ValueError(f"模板列序号越界：{col}")
    return col


# --------------------------------------------------------------------------
# Date helpers
# --------------------------------------------------------------------------

DATE_2025_01_01 = date(2025, 1, 1)
DATE_2025_12_31 = date(2025, 12, 31)


def calendar_days(start: date = DATE_2025_01_01, end: date = DATE_2025_12_31) -> list[date]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def to_date(value: object) -> date:
    """Normalise an Excel date cell to datetime.date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip().replace("/", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"无法解析日期：{value!r}")


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------


def require_shape(arr: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} 期望形状 {shape}，得到 {out.shape}")
    return out


__all__ = [
    "N_SEQ",
    "N_SEQ_BOUNDARY",
    "SEQ_START_LABEL",
    "SEQ_TAIL_LABEL",
    "NATURAL_DAY_ZERO_LABEL",
    "MAX_EXTENSION_HOURS",
    "SOURCE_ATTACHMENT",
    "SOURCE_CARRY_OVER",
    "SOURCE_BRIDGE_MISSING",
    "SOURCE_EXTENDED",
    "RESULT_BOUNDARY_OFFSET",
    "TEMPLATE_TIME_COLUMNS",
    "seq_index_from_label",
    "seq_label",
    "clock_label",
    "clock_to_abs_minute",
    "boundary_abs_minute",
    "clock_index_of_abs",
    "result_boundary_clock_index",
    "result_interval_clock_index",
    "result_row_spans_next_day",
    "template_column_of_result",
    "result_of_template_column",
    "calendar_days",
    "to_date",
    "require_shape",
    "DATE_2025_01_01",
    "DATE_2025_12_31",
]
