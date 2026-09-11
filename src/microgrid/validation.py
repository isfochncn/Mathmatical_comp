"""Validation: physics, information, fee and export checks on absolute time.

The 2026-09-11 rules changed what has to be checked:

* simultaneous charge/discharge is a **legal operating state**, not an anomaly.
  It must be accounted for (losses enter the energy balance) but never treated as
  a feasibility criterion;
* unavoidable paid overpurchase may be discarded, but remains fully billed;
  disposal is separately measured and cannot be manufactured by battery cycling;
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
    allow_spill: bool = True,
    validate_feedback: bool = True,
) -> RunDiagnostics:
    """Per-interval conservation, SOC recursion, device bounds and loss accounting."""
    n = len(abs_minutes)
    diag = RunDiagnostics(n_intervals=n)
    problems = []
    quantities = dict(grid=grid_kwh, emergency=emergency_kwh, charge=charge_kwh,
                      discharge=discharge_kwh, curtail=curtail_kwh, surplus=surplus_kwh,
                      demand=demand_kwh, pv=pv_kwh)
    if n == 0 or np.asarray(abs_minutes).shape != (n,) or np.any(np.diff(abs_minutes) != 10) or np.any(abs_minutes % 10):
        problems.append("Execution timestamps must be nonempty, contiguous and aligned")
    for name, arr in quantities.items():
        if arr.shape != (n,) or not np.isfinite(arr).all() or np.any(arr < -1e-6):
            problems.append(f"Invalid {name} quantities")
    if soc_boundary_kwh.shape != (n+1,) or not np.isfinite(soc_boundary_kwh).all():
        problems.append("Invalid SOC boundaries")
    if problems:
        raise ValidationError("; ".join(problems))
    if np.any(soc_boundary_kwh < E_MIN-1e-6) or np.any(soc_boundary_kwh > E_MAX+1e-6):
        problems.append("SOC outside device limits")
    if np.any(charge_kwh > Q_MAX+1e-6) or np.any(discharge_kwh > Q_DIS_MAX+1e-6):
        problems.append("Dispatch exceeds effective power limits")
    if np.any(curtail_kwh > pv_kwh+1e-6):
        problems.append("Curtailment exceeds PV")
    if not allow_spill and np.any(np.abs(surplus_kwh) > 1e-6):
        problems.append("External-power disposal is forbidden in this run")
    if np.any(surplus_kwh > grid_kwh + 1e-6):
        problems.append("Paid disposal exceeds ordinary purchases")
    surplus_to_charge = grid_kwh + emergency_kwh + pv_kwh - curtail_kwh + discharge_kwh - demand_kwh - surplus_kwh
    if np.any(surplus_to_charge > Q_MAX / 0.9 + 1e-6):
        problems.append("Purchase surplus exceeds charging input power")
    capacity_input = (E_MAX-soc_boundary_kwh[:-1] + discharge_kwh/0.9) / 0.9
    if np.any(surplus_to_charge > capacity_input + 1e-6):
        problems.append("Purchase surplus exceeds battery headroom")
    residual = grid_kwh + emergency_kwh + pv_kwh - curtail_kwh + discharge_kwh - demand_kwh - charge_kwh/0.9 - surplus_kwh
    diag.max_bus_residual_kwh = float(np.max(np.abs(residual)))
    if diag.max_bus_residual_kwh > 1e-6:
        problems.append("Bus energy balance failed")
    rebuilt = soc_boundary_kwh[0] + np.r_[0, np.cumsum(charge_kwh-discharge_kwh/0.9)]
    diag.max_soc_error_kwh = float(np.max(np.abs(rebuilt-soc_boundary_kwh)))
    if diag.max_soc_error_kwh > max(1e-6, n*1e-8):
        problems.append("SOC recursion failed")
    if abs(soc_boundary_kwh[0]-soc_start_kwh) > 1e-6:
        problems.append("SOC start mismatch")
    if require_daily_cycle and abs(soc_boundary_kwh[-1]-soc_boundary_kwh[0]) > 1e-6:
        problems.append("Daily cycle failed")
    diag.total_surplus_kwh = float(surplus_kwh.sum())
    if allow_spill and validate_feedback:
        # The exact least unavoidable paid disposal under battery-first feedback:
        # no discharge is used merely to burn an ordinary overpurchase.
        max_charge = np.minimum(Q_MAX, np.maximum(E_MAX-soc_boundary_kwh[:-1], 0))
        unavoidable = np.maximum(grid_kwh-demand_kwh-max_charge/0.9, 0)
        if not np.allclose(surplus_kwh, unavoidable, atol=2e-6, rtol=0):
            problems.append("Avoidable paid disposal was not eliminated")
        if np.any(discharge_kwh > np.maximum(demand_kwh-grid_kwh, 0) + 2e-6):
            problems.append("Battery discharge is not serving uncovered load")
    loss = physics.loss_accounting(charge_kwh, discharge_kwh)
    diag.total_loss_kwh = float(loss["total_loss_kwh"])
    diag.n_simultaneous = int(loss["n_simultaneous_intervals"])
    diag.min_soc_kwh, diag.max_soc_kwh = float(soc_boundary_kwh.min()), float(soc_boundary_kwh.max())
    diag.max_charge_kwh, diag.max_discharge_kwh = float(charge_kwh.max()), float(discharge_kwh.max())
    diag.max_curtail_kwh = float(curtail_kwh.max())
    if final and problems:
        raise ValidationError("; ".join(problems))
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
    if not p.size or not np.isfinite(p).all():
        raise ValidationError(f"{name} 缺少有效价格")
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
