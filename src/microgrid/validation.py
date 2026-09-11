"""Validation: physics, information, fee and export checks on absolute time.

The 2026-09-11 rules changed what has to be checked:

* simultaneous charge/discharge is a **legal operating state**, not an anomaly.
  It must be accounted for (losses enter the energy balance) but never treated as
  a feasibility criterion;
* a committed purchase that cannot be absorbed is reported as a disclosed
  surplus rather than hidden behind a rejection or a sale;
* the natural-day bill and the result-row bill cover different windows
  (``[00:00, 24:00)`` vs ``[00:10, next 00:10)``) and must be labelled as such;
* the executed fee labels are re-derived from the realised quantities, never
  from the planning objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import physics
from .constants import (
    E_INIT_2025_01_01,
    E_MAX,
    E_MIN,
    Q_DIS_MAX,
    Q_MAX,
    TOL_COST_YUAN,
    TOL_ENERGY_KWH,
)
from .timeaxis import calendar_days


class ValidationError(AssertionError):
    """强制性校验失败——不得继续导出。"""


@dataclass
class RunDiagnostics:
    """Aggregate physical diagnostics of one absolute run."""

    n_intervals: int = 0
    max_bus_residual_kwh: float = float("nan")
    max_soc_error_kwh: float = float("nan")
    min_soc_kwh: float = float("nan")
    max_soc_kwh: float = float("nan")
    max_charge_kwh: float = float("nan")
    max_discharge_kwh: float = float("nan")
    max_curtail_kwh: float = float("nan")
    total_loss_kwh: float = 0.0
    n_simultaneous: int = 0
    total_surplus_kwh: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_intervals": self.n_intervals,
            "max_bus_residual_kwh": self.max_bus_residual_kwh,
            "max_soc_error_kwh": self.max_soc_error_kwh,
            "min_soc_kwh": self.min_soc_kwh,
            "max_soc_kwh": self.max_soc_kwh,
            "max_charge_kwh": self.max_charge_kwh,
            "max_discharge_kwh": self.max_discharge_kwh,
            "max_curtail_kwh": self.max_curtail_kwh,
            "total_loss_kwh": self.total_loss_kwh,
            "n_simultaneous": self.n_simultaneous,
            "total_surplus_kwh": self.total_surplus_kwh,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def validate_absolute_run(
    *,
    abs_minutes: np.ndarray,
    grid_kwh: np.ndarray,
    emergency_kwh: np.ndarray,
    charge_kwh: np.ndarray,
    discharge_kwh: np.ndarray,
    curtail_kwh: np.ndarray,
    surplus_kwh: np.ndarray,
    soc_boundary_kwh: np.ndarray,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    soc_start_kwh: float,
    require_daily_cycle: bool = False,
    final: bool = True,
) -> RunDiagnostics:
    """Per-interval conservation, SOC recursion, device bounds and loss accounting."""
    diag = RunDiagnostics(n_intervals=int(abs_minutes.size))
    problems: list[str] = []

    n = abs_minutes.size
    for name, arr in (
        ("grid", grid_kwh),
        ("emergency", emergency_kwh),
        ("charge", charge_kwh),
        ("discharge", discharge_kwh),
        ("curtail", curtail_kwh),
        ("surplus", surplus_kwh),
        ("demand", demand_kwh),
        ("pv", pv_kwh),
    ):
        if arr.shape != (n,):
            problems.append(f"{name} 形状应为 ({n},)，得到 {arr.shape}")
        elif float(arr.min()) < -TOL_ENERGY_KWH:
            problems.append(f"{name} 出现负值 {float(arr.min()):.6f}")
    if soc_boundary_kwh.size != n + 1:
        problems.append(f"soc 边界应为 ({n + 1},)，得到 {soc_boundary_kwh.shape}")

    if problems:
        if final:
            raise ValidationError("轨迹校验失败：\n  - " + "\n  - ".join(problems))
        diag.warnings.extend(problems)
        return diag

    # Device bounds.
    if float(soc_boundary_kwh.min()) < E_MIN - 1e-6:
        problems.append(f"SOC 低于下限：{float(soc_boundary_kwh.min()):.6f} < {E_MIN}")
    if float(soc_boundary_kwh.max()) > E_MAX + 1e-6:
        problems.append(f"SOC 高于上限：{float(soc_boundary_kwh.max()):.6f} > {E_MAX}")
    if float(charge_kwh.max()) > Q_MAX + 1e-6:
        problems.append(f"充电量超过 750：{float(charge_kwh.max()):.6f}")
    if float(discharge_kwh.max()) > Q_DIS_MAX + 1e-6:
        problems.append(f"放电量超过 750：{float(discharge_kwh.max()):.6f}")
    if float((curtail_kwh - pv_kwh).max()) > 1e-6:
        problems.append("弃光超过可用光伏")

    # SOC recursion.
    rebuilt = soc_boundary_kwh[0] + np.concatenate(
        ([0.0], np.cumsum(charge_kwh - discharge_kwh / 0.9))
    )
    diag.max_soc_error_kwh = float(np.max(np.abs(rebuilt - soc_boundary_kwh)))
    if diag.max_soc_error_kwh > 1e-6:
        problems.append(f"SOC 递推不自洽：最大偏差 {diag.max_soc_error_kwh:.3e}")
    if abs(float(soc_boundary_kwh[0]) - float(soc_start_kwh)) > 1e-6:
        problems.append("SOC 起点与轨迹不一致")

    # Bus conservation. Delivered power cannot be rejected, so any positive
    # residual is the disclosed surplus and must match `surplus_kwh`.
    required = demand_kwh - pv_kwh + curtail_kwh + charge_kwh / 0.9 - discharge_kwh
    supplied = grid_kwh + emergency_kwh
    signed = supplied - required
    if float(signed.min()) < -TOL_ENERGY_KWH:
        t = int(np.argmin(signed))
        problems.append(
            f"母线守恒出现少供：区间 {t} 需要 {required[t]:.6f}，实供 {supplied[t]:.6f}"
        )
    positive = np.maximum(signed, 0.0)
    diag.max_bus_residual_kwh = float(np.max(np.abs(signed)))
    diag.total_surplus_kwh = float(positive.sum())
    if abs(diag.total_surplus_kwh - float(surplus_kwh.sum())) > 1e-3:
        problems.append(
            f"已披露富余 {float(surplus_kwh.sum()):.6f} 与实测正残差 "
            f"{diag.total_surplus_kwh:.6f} 不一致（不得静默丢弃能量）"
        )

    # Window identity:
    #   sum(h+u) = sum(D-G+r) + (19/90)*sum(q_ch) + 0.9*(E_end - E_start) + surplus
    lhs = float(np.sum(grid_kwh) + np.sum(emergency_kwh))
    rhs = (
        float(np.sum(demand_kwh - pv_kwh + curtail_kwh))
        + physics.LOSS_COEFF * float(np.sum(charge_kwh))
        + 0.9 * (float(soc_boundary_kwh[-1]) - float(soc_boundary_kwh[0]))
        + diag.total_surplus_kwh
    )
    if abs(lhs - rhs) > 1e-4:
        problems.append(f"窗口恒等式残差 {lhs - rhs:.6f} kWh")

    if require_daily_cycle:
        if abs(float(soc_boundary_kwh[-1]) - float(soc_boundary_kwh[0])) > 1e-6:
            problems.append("要求日循环但首末 SOC 不等")

    # Loss accounting; simultaneous charge/discharge is legal.
    loss = physics.loss_accounting(charge_kwh, discharge_kwh)
    diag.total_loss_kwh = float(loss["total_loss_kwh"])
    diag.n_simultaneous = int(loss["n_simultaneous_intervals"])
    diag.min_soc_kwh = float(soc_boundary_kwh.min())
    diag.max_soc_kwh = float(soc_boundary_kwh.max())
    diag.max_charge_kwh = float(charge_kwh.max())
    diag.max_discharge_kwh = float(discharge_kwh.max())
    diag.max_curtail_kwh = float(curtail_kwh.max())
    if diag.n_simultaneous:
        diag.notes.append(
            f"同时充放电 {diag.n_simultaneous} 段（合法运行状态，非异常）；"
            f"总损耗 {diag.total_loss_kwh:.6f} kWh"
        )
    if diag.total_surplus_kwh > 1e-6:
        diag.warnings.append(
            f"存在无处安放的富余 {diag.total_surplus_kwh:.3f} kWh"
            "（计划购电过量且储能已满），已如实记账，未用拒收或售电掩盖"
        )

    if final and problems:
        raise ValidationError("轨迹校验失败：\n  - " + "\n  - ".join(problems))
    diag.warnings.extend(problems)
    return diag


# ==========================================================================
# Input checks
# ==========================================================================


def validate_output_days(days: list[date]) -> None:
    """Result rows must cover 2025-02-01 .. 2025-12-31 (334 days)."""
    from .constants import OUTPUT_END, OUTPUT_START

    expected = calendar_days(date(*OUTPUT_START), date(*OUTPUT_END))
    if days != expected:
        raise ValidationError(
            f"输出日期应为 {OUTPUT_START} .. {OUTPUT_END} 共 {len(expected)} 天，"
            f"实际 {len(days)} 天"
        )


def validate_no_future_leak(*, context: str, **flags: bool) -> None:
    """Last-resort assertion; the structural guarantee lives in the forecaster."""
    bad = [k for k, v in flags.items() if v]
    if bad:
        raise ValidationError(f"[{context}] 策略信息视图出现未来量：{', '.join(bad)}")


def validate_prices_non_negative(price: np.ndarray, name: str) -> None:
    p = np.asarray(price, dtype=np.float64)
    if float(p.min()) < 0.0:
        raise ValidationError(f"{name} 出现负价：最小 {float(p.min()):.6f}（负价机制未定义）")


__all__ = [
    "ValidationError",
    "RunDiagnostics",
    "validate_absolute_run",
    "validate_output_days",
    "validate_no_future_leak",
    "validate_prices_non_negative",
    "E_INIT_2025_01_01",
]
