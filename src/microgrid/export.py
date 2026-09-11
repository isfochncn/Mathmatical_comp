"""Export layer: fill the five result workbooks from an absolute run.

Fixed rules (2026-09-11)
------------------------
* The original templates are **never** modified; results are written to copies.
* The template's 144 time columns ARE the 144 result intervals of a date row,
  and a result row covers ``[day 00:10, next day 00:10)``. The mapping from a
  result index to a timeline interval is
  ``clock index = j + 1`` (``j = 143`` reaches into the next natural day), which
  lives in :mod:`microgrid.timeaxis` and nowhere else.
* Result-row quantity/cost totals cover ``[day 00:10, next day 00:10)`` while the
  natural-day bill and the six 4-hour blocks cover ``[day 00:00, day 24:00)``.
  The two windows are labelled explicitly and are **not** expected to be equal
  without the boundary conversion.
* The template's ``...`` placeholder rows are expanded to the full date range.
* Consecutive emergency intervals are merged only when they are truly adjacent
  and belong to the same result row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from .constants import TEMPLATE_FILES
from .data_io import data_path
from .timeaxis import result_interval_clock_index
from .timeline import MINUTES_PER_DAY
from .validation import ValidationError

BLOCK_INTERVALS = 24
BLOCK_LABELS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)
SPECIAL_START_TIMES = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00")


@dataclass
class DailyExportRow:
    """Everything needed to write one date row of a multi-day workbook."""

    day: date
    plan_initial_kwh: np.ndarray      # (144,) 00:00 plan, result-row aligned
    final_plan_kwh: np.ndarray        # (144,) final effective plan, result-row aligned
    grid_actual_kwh: np.ndarray       # (144,) executed normal purchase = O + A
    emergency_actual_kwh: np.ndarray  # (144,)
    charge_stored_kwh: np.ndarray     # (144,)
    discharge_delivered_kwh: np.ndarray
    soc_natural_start_kwh: float      # natural day 00:00 (E_{d,0})
    soc_natural_end_kwh: float        # natural day 24:00 (E_{d,144})
    execution_cost_yuan: float
    reduce_cost_yuan: float

    @property
    def result_row_cost_yuan(self) -> float:
        return self.execution_cost_yuan + self.reduce_cost_yuan


def _load_template_copy(key: str, dest: Path):
    src = data_path(TEMPLATE_FILES[key])
    dest.parent.mkdir(parents=True, exist_ok=True)
    return load_workbook(src)


def _clear_row(ws, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).value = None


def interval_index_of_start_time(hhmm: str) -> int:
    """'10:00' -> natural-day clock interval 60 (the 10:00-10:10 segment)."""
    hh, mm = hhmm.split(":")
    minutes = int(hh) * 60 + int(mm)
    t = minutes // 10
    if not 0 <= t < 144:
        raise ValidationError(f"时间 {hhmm} 不对应任何区间")
    return t


# ==========================================================================
# result1
# ==========================================================================


def export_result1(trajectory, dest_dir: Path) -> Path:
    """Problem 1: 144 plan values plus the six 4-hour blocks and day-end SOC."""
    wb = _load_template_copy("result1", dest_dir / "result1.xlsx")
    ws = wb["计划购电量"]
    if ws.max_row != 145:
        raise ValidationError(f"result1 计划购电量应为 145 行，实际 {ws.max_row}")
    grid = np.asarray(trajectory["grid_kwh"], dtype=np.float64)
    for j in range(144):
        ws.cell(row=j + 2, column=2).value = round(float(grid[j]), 6)

    ws2 = wb["充放电量"]
    charge = np.asarray(trajectory["charge_kwh"], dtype=np.float64)
    discharge = np.asarray(trajectory["discharge_kwh"], dtype=np.float64)
    soc = np.asarray(
        trajectory.get("soc_boundary_kwh", trajectory.get("soc_kwh")), dtype=np.float64
    )
    for b in range(6):
        lo, hi = b * BLOCK_INTERVALS, (b + 1) * BLOCK_INTERVALS
        row = b + 2
        ws2.cell(row=row, column=1).value = BLOCK_LABELS[b]
        ws2.cell(row=row, column=2).value = round(float(charge[lo:hi].sum()), 6)
        ws2.cell(row=row, column=3).value = round(float(discharge[lo:hi].sum()), 6)
    ws2.cell(row=2, column=5).value = round(float(soc[0]), 6)
    ws2.cell(row=3, column=5).value = round(float(soc[-1]), 6)

    out = dest_dir / "result1.xlsx"
    wb.save(out)
    wb.close()
    return out


# ==========================================================================
# result2 / result3 / result4-2 / result4-3
# ==========================================================================


def _merge_adjacent(intervals: list[int]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    out: list[tuple[int, int]] = []
    start = prev = intervals[0]
    for t in intervals[1:]:
        if t == prev + 1:
            prev = t
            continue
        out.append((start, prev + 1))
        start = prev = t
    out.append((start, prev + 1))
    return out


def _fmt_interval_range(lo: int, hi: int) -> str:
    def fmt(t: int) -> str:
        minutes = t * 10
        if minutes >= 24 * 60:
            return "24:00"
        return f"{minutes // 60}:{minutes % 60:02d}"

    return f"{fmt(lo)}-{fmt(hi)}"


def export_multiday(
    problem: str,
    rows: list[DailyExportRow],
    dest_dir: Path,
    *,
    with_adjust_sheet: bool,
) -> Path:
    key = problem.replace("problem", "result")
    if key not in TEMPLATE_FILES:
        raise ValidationError(f"未知的结果文件：{problem}")
    if not rows:
        raise ValidationError("没有可导出的日期")
    for r in rows:
        for name in (
            "plan_initial_kwh",
            "final_plan_kwh",
            "grid_actual_kwh",
            "emergency_actual_kwh",
            "charge_stored_kwh",
            "discharge_delivered_kwh",
        ):
            arr = np.asarray(getattr(r, name), dtype=np.float64)
            if arr.shape != (144,):
                raise ValidationError(f"{r.day} 的 {name} 形状错误：{arr.shape}")

    wb = _load_template_copy(key, dest_dir / f"{key}.xlsx")
    _fill_daily_matrix(wb["计划购电量"], rows, "plan_initial_kwh")
    if with_adjust_sheet:
        if "调整购电量" not in wb.sheetnames:
            raise ValidationError(f"{key} 缺少'调整购电量'工作表")
        _fill_daily_matrix(wb["调整购电量"], rows, "final_plan_kwh")
    _fill_charge_discharge(wb["充放电量"], rows)
    _fill_emergency(wb["紧急购电量"], rows)

    out = dest_dir / f"{key}.xlsx"
    wb.save(out)
    wb.close()
    return out


def _fill_daily_matrix(ws, rows: list[DailyExportRow], attr: str) -> None:
    """Write the (day x 144) matrix.

    Template layout: row 1 header, column 1 date, columns 2..145 the 144 result
    intervals, column 146 '全天购电量', column 147 '全天购电费'.
    """
    needed = len(rows) + 1
    if ws.max_row < needed:
        raise ValidationError(
            f"模板行数不足：需要 {needed} 行，模板只有 {ws.max_row} 行"
        )
    for i, r in enumerate(rows):
        excel_row = i + 2
        ws.cell(row=excel_row, column=1).value = r.day
        values = np.asarray(getattr(r, attr), dtype=np.float64)
        # result-row totals: execution + emergency over [00:10, next day 00:10)
        day_total = float(
            np.asarray(r.grid_actual_kwh, dtype=np.float64).sum()
            + np.asarray(r.emergency_actual_kwh, dtype=np.float64).sum()
        )
        for j in range(144):
            ws.cell(row=excel_row, column=j + 2).value = round(float(values[j]), 6)
        ws.cell(row=excel_row, column=146).value = round(day_total, 6)
        ws.cell(row=excel_row, column=147).value = round(r.result_row_cost_yuan, 6)
    for excel_row in range(len(rows) + 2, ws.max_row + 1):
        _clear_row(ws, excel_row, 147)


def _fill_charge_discharge(ws, rows: list[DailyExportRow]) -> None:
    """Six 4-hour blocks per natural day plus the natural-day 00:00/24:00 SOC."""
    for excel_row in range(2, ws.max_row + 1):
        _clear_row(ws, excel_row, 6)

    excel_row = 2
    for r in rows:
        charge = np.asarray(r.charge_stored_kwh, dtype=np.float64)
        discharge = np.asarray(r.discharge_delivered_kwh, dtype=np.float64)
        for b in range(6):
            ws.cell(row=excel_row, column=1).value = r.day if b == 0 else None
            ws.cell(row=excel_row, column=2).value = BLOCK_LABELS[b]
            lo, hi = b * BLOCK_INTERVALS, (b + 1) * BLOCK_INTERVALS
            ws.cell(row=excel_row, column=3).value = round(float(charge[lo:hi].sum()), 6)
            ws.cell(row=excel_row, column=4).value = round(float(discharge[lo:hi].sum()), 6)
            if b == 0:
                ws.cell(row=excel_row, column=5).value = "0:00"
                ws.cell(row=excel_row, column=6).value = round(r.soc_natural_start_kwh, 6)
            elif b == 5:
                ws.cell(row=excel_row, column=5).value = "24:00"
                ws.cell(row=excel_row, column=6).value = round(r.soc_natural_end_kwh, 6)
            excel_row += 1


def _fill_emergency(ws, rows: list[DailyExportRow]) -> None:
    for excel_row in range(2, ws.max_row + 1):
        _clear_row(ws, excel_row, 3)

    excel_row = 2
    for r in rows:
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        hit = [int(j) for j in np.flatnonzero(emg > 1e-9)]
        merged = _merge_adjacent(hit)
        if not merged:
            continue
        first = True
        for lo, hi in merged:
            ws.cell(row=excel_row, column=1).value = r.day if first else None
            ws.cell(row=excel_row, column=2).value = _fmt_interval_range(lo, hi)
            ws.cell(row=excel_row, column=3).value = round(float(emg[lo:hi].sum()), 6)
            first = False
            excel_row += 1


# ==========================================================================
# Paper tables
# ==========================================================================


def paper_table1(rows: list[DailyExportRow]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows:
        grid = np.asarray(r.grid_actual_kwh, dtype=np.float64)
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        for start in SPECIAL_START_TIMES:
            t = interval_index_of_start_time(start)
            # result index for natural-day clock interval t is t-1
            j = t - 1
            if j < 0:
                j = 0
            end_minutes = (t + 1) * 10
            out.append(
                {
                    "日期": r.day.isoformat(),
                    "时间段": f"{start}-{end_minutes // 60}:{end_minutes % 60:02d}",
                    "购电量": round(float(grid[j] + emg[j]), 6),
                }
            )
        out.append(
            {
                "日期": r.day.isoformat(),
                "时间段": "全天购电量",
                "购电量": round(float(grid.sum() + emg.sum()), 6),
            }
        )
        out.append(
            {
                "日期": r.day.isoformat(),
                "时间段": "全天购电费",
                "购电量": round(r.result_row_cost_yuan, 6),
            }
        )
    return out


def paper_table2(rows: list[DailyExportRow]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows:
        charge = np.asarray(r.charge_stored_kwh, dtype=np.float64)
        discharge = np.asarray(r.discharge_delivered_kwh, dtype=np.float64)
        for b, label in enumerate(BLOCK_LABELS):
            lo, hi = b * BLOCK_INTERVALS, (b + 1) * BLOCK_INTERVALS
            out.append(
                {
                    "日期": r.day.isoformat(),
                    "时间段": label,
                    "充电量": round(float(charge[lo:hi].sum()), 6),
                    "放电量": round(float(discharge[lo:hi].sum()), 6),
                }
            )
        out.append(
            {
                "日期": r.day.isoformat(),
                "时间段": "0:00储电量",
                "充电量": round(r.soc_natural_start_kwh, 6),
            }
        )
        out.append(
            {
                "日期": r.day.isoformat(),
                "时间段": "24:00储电量",
                "充电量": round(r.soc_natural_end_kwh, 6),
            }
        )
    return out


def paper_table3(rows: list[DailyExportRow]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows:
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        hit = [int(j) for j in np.flatnonzero(emg > 1e-9)]
        for lo, hi in _merge_adjacent(hit):
            out.append(
                {
                    "日期": r.day.isoformat(),
                    "紧急购电时间段": _fmt_interval_range(lo, hi),
                    "紧急购电量": round(float(emg[lo:hi].sum()), 6),
                }
            )
    return out


# ==========================================================================
# Building export rows from a run
# ==========================================================================


def build_daily_rows(result) -> list[DailyExportRow]:
    """Convert an AbsoluteRun into result-row records for every output day.

    ``result`` is a :class:`microgrid.absolute_run.RunResult`. January is the
    warm-up period and is not exported.
    """
    from .constants import OUTPUT_START

    step_index = {s.abs_minute: s for s in result.run.steps}
    soc_index = _soc_lookup(result.run)
    events = {e.at_abs: e for e in result.run.events}

    out: list[DailyExportRow] = []
    for day in [d for d, _ in result.row_bills]:
        if day < date(*OUTPUT_START):
            continue
        base = (day - date(2025, 1, 1)).days * MINUTES_PER_DAY
        plan = np.zeros(144)
        final = np.zeros(144)
        grid = np.zeros(144)
        emg = np.zeros(144)
        ch = np.zeros(144)
        dis = np.zeros(144)
        exec_cost = 0.0
        for j in range(144):
            abs_minute = base + result_interval_clock_index(j) * 10
            step = step_index.get(abs_minute)
            if step is not None:
                grid[j] = step.grid_kwh
                emg[j] = step.emergency_kwh
                ch[j] = step.charge_kwh
                dis[j] = step.discharge_kwh
            ev = events.get(abs_minute)
            if ev is not None:
                # "计划购电量" is the signed normal purchase O; "调整购电量" is the
                # effective purchase after the intra-day revision, i.e. O + A.
                # Setting both to O + A (as an earlier version did) made the
                # adjustment sheet an exact copy of the plan sheet.
                plan[j] = ev.o_exec_kwh
                final[j] = ev.o_exec_kwh + ev.a_exec_kwh
                exec_cost += ev.cost_yuan
        row_bill = dict(result.row_bills)[day]
        out.append(
            DailyExportRow(
                day=day,
                plan_initial_kwh=plan,
                final_plan_kwh=final,
                grid_actual_kwh=grid,
                emergency_actual_kwh=emg,
                charge_stored_kwh=ch,
                discharge_delivered_kwh=dis,
                soc_natural_start_kwh=soc_index.get(base, float("nan")),
                soc_natural_end_kwh=soc_index.get(base + MINUTES_PER_DAY, float("nan")),
                execution_cost_yuan=exec_cost,
                reduce_cost_yuan=row_bill.reduce_cost_yuan,
            )
        )
    return out


def _soc_lookup(run) -> dict[int, float]:
    """Absolute minute -> SOC boundary value for successive executed intervals."""
    out: dict[int, float] = {}
    steps = sorted(run.steps, key=lambda s: s.abs_minute)
    soc = run.soc_boundary_kwh
    for i, s in enumerate(steps):
        out[s.abs_minute] = float(soc[i]) if i < soc.size else float("nan")
        out[s.abs_minute + 10] = float(soc[i + 1]) if i + 1 < soc.size else float("nan")
    return out


__all__ = [
    "DailyExportRow",
    "build_daily_rows",
    "export_result1",
    "export_multiday",
    "paper_table1",
    "paper_table2",
    "paper_table3",
    "interval_index_of_start_time",
    "BLOCK_LABELS",
    "SPECIAL_START_TIMES",
]
