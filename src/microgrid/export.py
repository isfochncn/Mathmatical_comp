"""导出层：向模板副本定点填值。

模板处理规则（只读核验 + 用户已确认口径）
------------------------------------------
1. **原模板一字不改**。程序只读 ``data/附件5/*.xlsx``，把结果写到 ``out/`` 下的副本。
2. 时间轴：模板的 144 个时间标签比模型区间整体后移了一段
   （首列是 '0:10-0:20'，末列是 '0:00-0:10+1'），且缺 '0:00-0:10'。
   已确认口径是**保持模板标签不动**，模型第 t 段（[t*10min,(t+1)*10min)）
   写入第 t+1 个数据格，即 ``0:00-0:10`` 的值写在 '0:10-0:20' 标签下方。
   映射函数集中在 :mod:`microgrid.timeaxis`，此处只调用。
3. 模板里的 '⁝' 省略行必须展开为完整日期，不能只填示例几行。
4. "充放电量" 表每天 6 个 4 小时块；10 分钟轨迹按块求和写入。
5. "紧急购电量" 表按日、按实际发生的紧急时段逐条填写；只有**相邻**紧急段可合并。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from .constants import (
    ADJUST_NODE_INTERVALS,
    E_INIT_2025_01_01,
    N_INTERVAL,
    TEMPLATE_FILES,
)
from .data_io import data_path, project_root
from .timeaxis import boundary_labels, interval_labels
from .validation import ValidationError

# 4 小时块 = 24 段
BLOCK_INTERVALS = 24
BLOCK_LABELS = ("0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00")

#: 论文表1/表3 指定时间段（10 分钟）的起点
SPECIAL_START_TIMES = ("10:00", "12:00", "14:00", "16:00", "18:00", "20:00")


# ==========================================================================
# 工具
# ==========================================================================


def _load_template_copy(key: str, dest: Path) -> "openpyxl.workbook.workbook.Workbook":  # type: ignore[name-defined]
    src = data_path(TEMPLATE_FILES[key])
    dest.parent.mkdir(parents=True, exist_ok=True)
    wb = load_workbook(src)  # 原模板只读打开，另存为副本
    return wb


def _clear_row(ws, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).value = None


def interval_index_of_start_time(hhmm: str) -> int:
    """'10:00' -> 该 10 分钟段（10:00-10:10）的模型区间号 60。"""
    hh, mm = hhmm.split(":")
    minutes = int(hh) * 60 + int(mm)
    t = minutes // 10
    if not 0 <= t < N_INTERVAL:
        raise ValidationError(f"时间 {hhmm} 不对应任何模型区间")
    return t


# ==========================================================================
# result1
# ==========================================================================


def export_result1(record, dest_dir: Path, *, grid_kwh: np.ndarray | None = None) -> Path:
    """问题一结果：计划购电量（144 段）+ 充放电量（6 个 4 小时块 + 首末 SOC）。"""
    traj = record.trajectory
    grid = np.asarray(grid_kwh if grid_kwh is not None else traj.grid_actual_kwh, dtype=np.float64)

    wb = _load_template_copy("result1", dest_dir / "result1.xlsx")
    ws = wb["计划购电量"]
    if ws.max_row != N_INTERVAL + 1:
        raise ValidationError(f"result1 计划购电量应为 {N_INTERVAL + 1} 行，实际 {ws.max_row}")
    # 第 t 段 -> 第 t+2 行（第 1 行是表头）
    for t in range(N_INTERVAL):
        ws.cell(row=t + 2, column=2).value = round(float(grid[t]), 6)

    ws2 = wb["充放电量"]
    charge = traj.charge_stored_kwh
    discharge = traj.discharge_delivered_kwh
    for b in range(6):
        lo, hi = b * BLOCK_INTERVALS, (b + 1) * BLOCK_INTERVALS
        row = b + 2
        ws2.cell(row=row, column=1).value = BLOCK_LABELS[b]
        ws2.cell(row=row, column=2).value = round(float(charge[lo:hi].sum()), 6)
        ws2.cell(row=row, column=3).value = round(float(discharge[lo:hi].sum()), 6)
    ws2.cell(row=2, column=5).value = round(float(traj.soc_kwh[0]), 6)
    ws2.cell(row=3, column=5).value = round(float(traj.soc_kwh[-1]), 6)

    out = dest_dir / "result1.xlsx"
    wb.save(out)
    wb.close()
    return out


# ==========================================================================
# result2 / result4-2 / result3 / result4-3
# ==========================================================================


@dataclass
class DailyExportRow:
    """导出所需的单日数据（与仿真/结算解耦，便于单独测试）。"""

    day: date
    plan_initial_kwh: np.ndarray        # 0 点计划 x
    final_plan_kwh: np.ndarray          # 调整后的最终有效策略（问题二等于 x）
    grid_actual_kwh: np.ndarray         # 实际普通购电 h
    emergency_actual_kwh: np.ndarray
    charge_stored_kwh: np.ndarray
    discharge_delivered_kwh: np.ndarray
    soc_start_kwh: float
    soc_end_kwh: float
    purchase_cost_yuan: float
    penalty_yuan: float

    @property
    def full_day_cost_yuan(self) -> float:
        """模板"全天购电费"列：实际执行购电费 + 已发生违约金。"""
        return self.purchase_cost_yuan + self.penalty_yuan


def _merge_adjacent(intervals: list[int]) -> list[tuple[int, int]]:
    """把相邻段合并为 [起, 止) 区间；只有相邻才合并。"""
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
    """[起, 止) 段号 -> 'HH:MM-HH:MM' 文本（结束用 24:00 表示一天末尾）。"""
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
    """导出 result2 / result3 / result4-2 / result4-3。

    ``rows`` 必须只包含需要输出的 2025-02-01..12-31 共 334 天，且按日期升序。
    """
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
            if arr.shape != (N_INTERVAL,):
                raise ValidationError(f"{r.day} 的 {name} 形状错误：{arr.shape}")

    wb = _load_template_copy(key, dest_dir / f"{key}.xlsx")

    # ---- 计划购电量 / 调整购电量：一天一行 ----
    _fill_daily_matrix(wb["计划购电量"], rows, "plan_initial_kwh")
    if with_adjust_sheet:
        if "调整购电量" not in wb.sheetnames:
            raise ValidationError(f"{key} 缺少'调整购电量'工作表")
        # 已确认口径：写"最终有效执行策略曲线"（逐段实际生效的普通购电量）
        _fill_daily_matrix(wb["调整购电量"], rows, "final_plan_kwh")

    # ---- 充放电量：每天 6 行 4 小时块 ----
    _fill_charge_discharge(wb["充放电量"], rows)

    # ---- 紧急购电量：按日逐条展开 ----
    _fill_emergency(wb["紧急购电量"], rows)

    out = dest_dir / f"{key}.xlsx"
    wb.save(out)
    wb.close()
    return out


def _fill_daily_matrix(ws, rows: list[DailyExportRow], attr: str) -> None:
    """把 (天 × 144 段) 写进 '日期\\时间' 矩阵表。

    模板第 1 行是表头，第 1 列是日期，第 2..145 列是 144 个时间段
    （标签 '0:10-0:20' … '0:00-0:10+1'，比模型区间后移一段）。
    第 146/147 列是 '全天购电量' 与 '全天购电费'。
    """
    n_rows_needed = len(rows) + 1  # 含表头
    if ws.max_row < n_rows_needed:
        # 模板行数不足时按末行样式扩展列标题单元格
        raise ValidationError(
            f"模板行数不足：需要 {n_rows_needed} 行，模板只有 {ws.max_row} 行。"
            "请确认输出日期范围与模板一致（334 天 + 表头 = 335 行）。"
        )
    for i, r in enumerate(rows):
        excel_row = i + 2
        ws.cell(row=excel_row, column=1).value = r.day
        values = np.asarray(getattr(r, attr), dtype=np.float64)
        day_total = float(np.asarray(r.grid_actual_kwh, dtype=np.float64).sum() + np.asarray(
            r.emergency_actual_kwh, dtype=np.float64
        ).sum())
        for t in range(N_INTERVAL):
            ws.cell(row=excel_row, column=t + 2).value = round(float(values[t]), 6)
        ws.cell(row=excel_row, column=N_INTERVAL + 2).value = round(day_total, 6)
        ws.cell(row=excel_row, column=N_INTERVAL + 3).value = round(r.full_day_cost_yuan, 6)
    # 模板多余行（'⁝' 或示例日）必须清空，避免残留示例数据
    for excel_row in range(len(rows) + 2, ws.max_row + 1):
        _clear_row(ws, excel_row, N_INTERVAL + 3)


def _fill_charge_discharge(ws, rows: list[DailyExportRow]) -> None:
    """模板每天 6 行：日期 / 时间段 / 充电量 / 放电量 / 时刻 / 储电量。

    模板原有 2—3 天示例 + 一行 '⁝'。本函数按 6 行/天重建整表：
    先清空原内容，再按 334 天写入，保证不残留省略行与示例。
    """
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
                ws.cell(row=excel_row, column=6).value = round(float(r.soc_start_kwh), 6)
            elif b == 5:
                ws.cell(row=excel_row, column=5).value = "24:00"
                ws.cell(row=excel_row, column=6).value = round(float(r.soc_end_kwh), 6)
            excel_row += 1


def _fill_emergency(ws, rows: list[DailyExportRow]) -> None:
    """紧急购电表：每天按实际发生的紧急时段逐条写，相邻段合并。"""
    for excel_row in range(2, ws.max_row + 1):
        _clear_row(ws, excel_row, 3)

    excel_row = 2
    for r in rows:
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        hit = [int(t) for t in np.flatnonzero(emg > 1e-9)]
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
# 论文用表（表1 / 表2 / 表3）
# ==========================================================================


def paper_table1(rows: list[DailyExportRow]) -> list[dict[str, float | str]]:
    """表1：指定 10 分钟段的购电量 + 全天购电量与购电费。"""
    out: list[dict[str, float | str]] = []
    for r in rows:
        grid = np.asarray(r.grid_actual_kwh, dtype=np.float64)
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        for start in SPECIAL_START_TIMES:
            t = interval_index_of_start_time(start)
            end_minutes = (t + 1) * 10
            label = f"{start}-{end_minutes // 60}:{end_minutes % 60:02d}"
            out.append(
                {
                    "日期": r.day.isoformat(),
                    "时间段": label,
                    "购电量": round(float(grid[t] + emg[t]), 6),
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
                "购电量": round(float(r.full_day_cost_yuan), 6),
            }
        )
    return out


def paper_table2(rows: list[DailyExportRow]) -> list[dict[str, float | str]]:
    """表2：6 个 4 小时块的充放电量 + 0:00/24:00 储电量。"""
    out: list[dict[str, float | str]] = []
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
        out.append({"日期": r.day.isoformat(), "时间段": "0:00储电量", "充电量": round(float(r.soc_start_kwh), 6)})
        out.append({"日期": r.day.isoformat(), "时间段": "24:00储电量", "充电量": round(float(r.soc_end_kwh), 6)})
    return out


def paper_table3(rows: list[DailyExportRow]) -> list[dict[str, float | str]]:
    """表3：指定日期的紧急购电量。"""
    out: list[dict[str, float | str]] = []
    for r in rows:
        emg = np.asarray(r.emergency_actual_kwh, dtype=np.float64)
        hit = [int(t) for t in np.flatnonzero(emg > 1e-9)]
        for lo, hi in _merge_adjacent(hit):
            out.append(
                {
                    "日期": r.day.isoformat(),
                    "紧急购电时间段": _fmt_interval_range(lo, hi),
                    "紧急购电量": round(float(emg[lo:hi].sum()), 6),
                }
            )
    return out


__all__ = [
    "DailyExportRow",
    "daily_export_rows",
    "export_result1",
    "export_multiday",
    "paper_table1",
    "paper_table2",
    "paper_table3",
    "interval_index_of_start_time",
    "BLOCK_LABELS",
    "SPECIAL_START_TIMES",
    "boundary_labels",
    "interval_labels",
    "ADJUST_NODE_INTERVALS",
    "E_INIT_2025_01_01",
]


def daily_export_rows(day_records) -> list[DailyExportRow]:
    """把运行结果里的逐日记录转成导出行。

    只接受 2025-02-01 之后的日期——1 月是预热期，不进入结果文件。
    """
    first = date(2025, 2, 1)
    rows: list[DailyExportRow] = []
    for rec in day_records:
        if rec.day < first:
            continue
        traj = rec.trajectory
        plan_initial = rec.plan_initial_kwh
        final_plan = rec.final_plan_curve_kwh
        if plan_initial is None or final_plan is None:
            raise ValidationError(f"{rec.day} 缺少计划曲线，无法导出")
        rows.append(
            DailyExportRow(
                day=rec.day,
                plan_initial_kwh=np.asarray(plan_initial, dtype=np.float64),
                final_plan_kwh=np.asarray(final_plan, dtype=np.float64),
                grid_actual_kwh=np.asarray(traj.grid_actual_kwh, dtype=np.float64),
                emergency_actual_kwh=np.asarray(traj.emergency_actual_kwh, dtype=np.float64),
                charge_stored_kwh=np.asarray(traj.charge_stored_kwh, dtype=np.float64),
                discharge_delivered_kwh=np.asarray(traj.discharge_delivered_kwh, dtype=np.float64),
                soc_start_kwh=float(traj.soc_kwh[0]),
                soc_end_kwh=float(traj.soc_kwh[-1]),
                purchase_cost_yuan=float(rec.bill.purchase_cost_yuan),
                penalty_yuan=float(rec.bill.penalty_yuan),
            )
        )
    return rows
