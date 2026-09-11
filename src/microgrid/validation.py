"""校验层：物理、信息、费用、输出四类强制校验。

规范第 13 节与各问第 16 节要求：
数据不可行、未定义负价机制、未来信息泄漏必须让**正式导出失败**，
而不是仅打印警告继续输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from . import physics
from .constants import (
    E_INIT_2025_01_01,
    N_BOUNDARY,
    N_INTERVAL,
    OUTPUT_END,
    OUTPUT_START,
    Q_MAX,
    TOL_COST_YUAN,
    TOL_ENERGY_KWH,
    TOL_POWER_KW,
    FeeClass,
)
from .data_io import DataBundle
from .schemas import Bill, DayInput, Trajectory
from .settlement import SettlementError, settle_trajectory, verify_bill
from .timeaxis import calendar_days, template_column_of_interval


class ValidationError(AssertionError):
    """强制校验失败——不得继续导出。"""


# ==========================================================================
# 物理校验
# ==========================================================================


@dataclass
class TrajectoryDiagnostics:
    max_bus_residual_kwh: float = float("nan")
    max_soc_transition_error_kwh: float = float("nan")
    daily_balance_residual_kwh: float = float("nan")
    discharge_charge_relation_residual_kwh: float = float("nan")
    round_trip_ratio: float = float("nan")
    simultaneous_charge_discharge_intervals: list[int] = field(default_factory=list)
    loss_cycle_net_kwh: float = 0.0
    max_power_kw: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "max_bus_residual_kwh": self.max_bus_residual_kwh,
            "max_soc_transition_error_kwh": self.max_soc_transition_error_kwh,
            "daily_balance_residual_kwh": self.daily_balance_residual_kwh,
            "discharge_charge_relation_residual_kwh": self.discharge_charge_relation_residual_kwh,
            "round_trip_ratio": self.round_trip_ratio,
            "simultaneous_charge_discharge_intervals": self.simultaneous_charge_discharge_intervals,
            "loss_cycle_net_kwh": self.loss_cycle_net_kwh,
            "max_power_kw": self.max_power_kw,
            "warnings": self.warnings,
        }


def validate_trajectory(
    trajectory: Trajectory,
    day_input: DayInput,
    *,
    require_daily_cycle: bool = False,
    final: bool = True,
    allow_surplus_safety_valve: bool = False,
) -> TrajectoryDiagnostics:
    """逐段校验守恒、SOC 递推、容量、功率、非负性与弃光边界。

    ``final=True`` 抛出异常；``final=False`` 只收集诊断（求解过程中自查用）。

    ``allow_surplus_safety_valve``
        是否允许"计划购电过量且储能已满"造成的正残差作为安全阀。
        规范第 6 节要求如实报告该情形，而不是用拒收/售电/烧电掩盖。
        为 True 时：正残差被记录为 surplus 并进入诊断与恒等式核对；
        为 False 时：任何非零残差都直接判失败。**负残差（少供）永远不允许**。
    """
    diag = TrajectoryDiagnostics()
    problems: list[str] = []
    surplus_kwh = 0.0

    def _fail(msg: str) -> None:
        problems.append(msg)

    try:
        physics.check_soc_bounds(trajectory.soc_kwh)
    except physics.PhysicsViolation as exc:
        _fail(str(exc))
    try:
        physics.check_charge_discharge_bounds(
            trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
        )
    except physics.PhysicsViolation as exc:
        _fail(str(exc))
    try:
        physics.check_curtail_bounds(trajectory.curtail_kwh, day_input.pv_kwh)
    except physics.PhysicsViolation as exc:
        _fail(str(exc))
    if float(np.min(trajectory.soc_kwh)) < 0.0:
        _fail("储电量出现负值")

    # SOC 递推
    rebuilt = physics.soc_next_vec(
        trajectory.soc_kwh, trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
    )
    diag.max_soc_transition_error_kwh = float(np.max(np.abs(rebuilt - trajectory.soc_kwh)))
    if diag.max_soc_transition_error_kwh > TOL_ENERGY_KWH:
        _fail(f"SOC 递推不自洽，最大偏差 {diag.max_soc_transition_error_kwh:.3e}")
    if abs(float(trajectory.soc_kwh[0]) - float(rebuilt[0])) > TOL_ENERGY_KWH:
        _fail("SOC 起点与轨迹不一致")

    # 母线守恒
    required = physics.grid_required_kwh(
        day_input.demand_kwh,
        day_input.pv_kwh,
        trajectory.curtail_kwh,
        trajectory.charge_stored_kwh,
        trajectory.discharge_delivered_kwh,
    )
    supplied = trajectory.grid_actual_kwh + trajectory.emergency_actual_kwh
    signed = supplied - required  # 正 = 供大于需（无处安放），负 = 少供
    if float(signed.min()) < -TOL_ENERGY_KWH:
        t = int(np.argmin(signed))
        _fail(
            f"母线守恒出现少供：区间 {t} 需要 {required[t]:.9f}，实供 {supplied[t]:.9f}"
        )
    positive = np.maximum(signed, 0.0)
    surplus_kwh = float(positive.sum())
    declared = getattr(trajectory, "surplus_disposed_kwh", None)
    declared_kwh = 0.0 if declared is None else float(np.sum(declared))
    if surplus_kwh > TOL_ENERGY_KWH:
        if allow_surplus_safety_valve:
            if abs(declared_kwh - surplus_kwh) > 1e-3:
                _fail(
                    f"已披露的安全阀量 {declared_kwh:.3f} kWh 与实测正残差 "
                    f"{surplus_kwh:.3f} kWh 不一致（不得静默丢弃能量）"
                )
            diag.warnings.append(
                f"计划购电过量且储能已满：{int((positive > TOL_ENERGY_KWH).sum())} 个区间共 "
                f"{surplus_kwh:.3f} kWh 无处安放，已作为安全阀如实记账（非拒收、非售电），"
                "必须在论文中作为可实施性缺口披露"
            )
        else:
            t = int(np.argmax(signed))
            _fail(
                f"母线守恒出现无处安放的富余：区间 {t} 富余 {signed[t]:.9f} kWh"
                "（计划购电过量且储能已满）"
            )
    diag.max_bus_residual_kwh = float(np.max(np.abs(signed))) if signed.size else 0.0

    # 非负
    for name, arr in (
        ("普通购电", trajectory.grid_actual_kwh),
        ("紧急购电", trajectory.emergency_actual_kwh),
    ):
        if float(np.min(arr)) < -TOL_ENERGY_KWH:
            _fail(f"{name} 出现负值 {float(np.min(arr)):.9f}")

    # 全天恒等式
    diag.daily_balance_residual_kwh = physics.daily_balance_residual_kwh(
        trajectory.grid_actual_kwh,
        trajectory.emergency_actual_kwh,
        day_input.demand_kwh,
        day_input.pv_kwh,
        trajectory.curtail_kwh,
        trajectory.charge_stored_kwh,
        trajectory.discharge_delivered_kwh,
        trajectory.soc_kwh,
    ) - surplus_kwh
    if abs(diag.daily_balance_residual_kwh) > TOL_ENERGY_KWH:
        _fail(f"全天恒等式残差 {diag.daily_balance_residual_kwh:.9f} kWh")

    # 日循环（仅问题一）
    if require_daily_cycle:
        gap = abs(float(trajectory.soc_kwh[-1]) - float(trajectory.soc_kwh[0]))
        if gap > TOL_ENERGY_KWH:
            _fail(f"问题一要求 E_0 = E_144，实际相差 {gap:.9f}")
        diag.discharge_charge_relation_residual_kwh = (
            physics.discharge_charge_relation_residual_kwh(
                trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
            )
        )
        if abs(diag.discharge_charge_relation_residual_kwh) > TOL_ENERGY_KWH:
            _fail(
                f"日循环下应满足 sum(q_dis)=0.9*sum(q_ch)，残差 "
                f"{diag.discharge_charge_relation_residual_kwh:.9f}"
            )

    # 可实施性诊断（只报告，不偷偷改模型）
    diag.round_trip_ratio = physics.round_trip_ratio(
        trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
    )
    cycling = physics.diagnose_loss_cycling(
        trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
    )
    diag.simultaneous_charge_discharge_intervals = list(cycling["intervals"])  # type: ignore[arg-type]
    diag.loss_cycle_net_kwh = float(cycling["net_loss_kwh"])
    diag.max_power_kw = physics.max_bus_power_kw(
        trajectory.charge_stored_kwh, trajectory.discharge_delivered_kwh
    )
    if diag.simultaneous_charge_discharge_intervals:
        diag.warnings.append(
            "存在同时充放电段 "
            f"{diag.simultaneous_charge_discharge_intervals}，"
            f"损耗循环 {diag.loss_cycle_net_kwh:.6f} kWh；"
            "必须在论文中披露可实施性缺口，不得事后暗加互斥约束后仍称同一模型。"
        )

    if final and problems:
        raise ValidationError("轨迹校验失败：\n  - " + "\n  - ".join(problems))
    diag.warnings.extend(problems)
    return diag


# ==========================================================================
# 输入校验
# ==========================================================================


def validate_attachment1_prices(price: np.ndarray) -> None:
    """附件1 电价全为正——负价机制未定义，不得凭空引入。"""
    p = np.asarray(price, dtype=np.float64)
    if float(p.min()) <= 0.0:
        raise ValidationError(f"附件1 电价出现非正值：最小 {float(p.min()):.6f}")


def validate_attachment4_prices(price: np.ndarray) -> None:
    p = np.asarray(price, dtype=np.float64)
    if float(p.min()) <= 0.0:
        raise ValidationError(f"附件4 电价出现非正值：最小 {float(p.min()):.6f}")


def validate_day_input(day_input: DayInput) -> None:
    demand = np.asarray(day_input.demand_kwh, dtype=np.float64)
    pv = np.asarray(day_input.pv_kwh, dtype=np.float64)
    if demand.shape != (N_INTERVAL,) or pv.shape != (N_INTERVAL,):
        raise ValidationError("DayInput 形状错误")
    if float(demand.min()) < -TOL_POWER_KW:
        raise ValidationError("负载出现负值")
    if float(pv.min()) < -TOL_POWER_KW:
        raise ValidationError("光伏出现负值")


def validate_calendar(days: list[date]) -> None:
    expected = calendar_days(date(*OUTPUT_START), date(*OUTPUT_END))
    if days != expected:
        raise ValidationError(
            f"输出日期范围应为 {OUTPUT_START} .. {OUTPUT_END} 共 {len(expected)} 天，"
            f"实际 {len(days)} 天"
        )


# ==========================================================================
# 费用校验
# ==========================================================================


def validate_bill(bill: Bill, trajectory: Trajectory, initial_plan_kwh: np.ndarray | None = None) -> None:
    """用与生产路径不同的复算重算账单。"""
    recomputed = settle_trajectory(trajectory, initial_plan_kwh=initial_plan_kwh)
    if abs(recomputed.total_purchased_kwh - bill.total_purchased_kwh) > TOL_ENERGY_KWH:
        raise ValidationError(
            f"购电量复算不一致：{bill.total_purchased_kwh:.9f} vs {recomputed.total_purchased_kwh:.9f}"
        )
    if abs(recomputed.total_cost_yuan - bill.total_cost_yuan) > TOL_COST_YUAN:
        raise ValidationError(
            f"总费用复算不一致：{bill.total_cost_yuan:.6f} vs {recomputed.total_cost_yuan:.6f}"
        )
    events = settle_events(
        trajectory, initial_plan_kwh=initial_plan_kwh, adjust_events=None
    )
    try:
        verify_bill(bill, events, trajectory.updates)
    except SettlementError as exc:
        raise ValidationError(str(exc)) from exc


def settle_events(
    trajectory: Trajectory,
    initial_plan_kwh: np.ndarray | None = None,
    adjust_events: dict[int, object] | None = None,
):
    """Re-derive the executed billing events with the same classification basis."""
    from .settlement import classify_executed

    return classify_executed(
        grid_actual_kwh=trajectory.grid_actual_kwh,
        emergency_actual_kwh=trajectory.emergency_actual_kwh,
        price_actual=trajectory.price_actual,
        day=trajectory.day,
        initial_plan_kwh=initial_plan_kwh,
        adjust_events=adjust_events,
    )


def validate_fee_classes(trajectory: Trajectory) -> None:
    """费率类别必须与当时的真实执行动作一致。"""
    for e in trajectory.events:
        if e.quantity_kwh <= 0.0:
            raise ValidationError(f"{e.day} 区间 {e.interval} 出现非正购电事件")
        if e.fee_class not in (FeeClass.NORMAL, FeeClass.ADJUST_UP, FeeClass.EMERGENCY):
            raise ValidationError(f"未知费率类别：{e.fee_class}")
        if e.price_yuan_per_kwh <= 0.0:
            raise ValidationError(f"{e.day} 区间 {e.interval} 事件价格为非正：{e.price_yuan_per_kwh}")


# ==========================================================================
# 信息泄漏校验
# ==========================================================================


def validate_no_future_leak(
    *,
    known_at: date,
    known_interval: int,
    plan_uses_pv_actual: bool = False,
    plan_uses_load_actual: bool = False,
    price_source_is_actual_future: bool = False,
    context: str = "",
) -> None:
    """策略信息视图的兜底断言。

    更强的保障是结构性的：策略函数根本拿不到未来实测表（见 simulation 模块）。
    本函数只是最后一道防线。
    """
    tag = f"[{context}] " if context else ""
    if plan_uses_pv_actual:
        raise ValidationError(f"{tag}策略不应读取光伏**实际**功率（只有预报可用，且需在发布之后）")
    if plan_uses_load_actual:
        raise ValidationError(f"{tag}策略不应读取负载实测真值")
    if price_source_is_actual_future:
        raise ValidationError(f"{tag}问题4 策略不得提前读取附件4 的未来真实电价")


def validate_output_days(days: list[date]) -> None:
    """结果文件只输出 2025-02-01 .. 2025-12-31，1 月是预热期。"""
    validate_calendar(days)


def validate_fresh_start(days_all: list[date], first_output_day: date) -> None:
    """2 月 1 日不得被当作新的初值起点——必须由 1 月真实轨迹产生。"""
    if first_output_day != date(*OUTPUT_START):
        raise ValidationError(f"首个输出日应为 {OUTPUT_START}，得到 {first_output_day}")
    if date(2025, 1, 1) not in days_all:
        raise ValidationError("连续运行必须从 2025-01-01 开始（1 月预热）")


# ==========================================================================
# 汇总入口
# ==========================================================================


@dataclass
class ValidationSummary:
    n_days: int
    total_purchased_kwh: float
    total_emergency_kwh: float
    total_penalty_yuan: float
    total_cost_yuan: float
    max_bus_residual_kwh: float
    max_soc_transition_error_kwh: float
    max_daily_balance_residual_kwh: float
    loss_cycle_net_kwh: float
    n_simultaneous_intervals: int
    max_bus_charge_kw: float
    max_bus_discharge_kw: float
    soc_end_kwh: float
    infeasible_days: list[date] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_days": self.n_days,
            "total_purchased_kwh": self.total_purchased_kwh,
            "total_emergency_kwh": self.total_emergency_kwh,
            "total_penalty_yuan": self.total_penalty_yuan,
            "total_cost_yuan": self.total_cost_yuan,
            "max_bus_residual_kwh": self.max_bus_residual_kwh,
            "max_soc_transition_error_kwh": self.max_soc_transition_error_kwh,
            "max_daily_balance_residual_kwh": self.max_daily_balance_residual_kwh,
            "loss_cycle_net_kwh": self.loss_cycle_net_kwh,
            "n_simultaneous_intervals": self.n_simultaneous_intervals,
            "max_bus_charge_kw": self.max_bus_charge_kw,
            "max_bus_discharge_kw": self.max_bus_discharge_kw,
            "soc_end_kwh": self.soc_end_kwh,
            "infeasible_days": [d.isoformat() for d in self.infeasible_days],
            "warnings": self.warnings,
        }


def summarize(
    trajectories: list[Trajectory],
    bills: list[Bill],
    diagnostics: list[TrajectoryDiagnostics],
) -> ValidationSummary:
    if not (len(trajectories) == len(bills) == len(diagnostics)):
        raise ValidationError("汇总输入长度不一致")
    if not trajectories:
        raise ValidationError("没有任何轨迹可汇总")
    return ValidationSummary(
        n_days=len(trajectories),
        total_purchased_kwh=float(sum(b.total_purchased_kwh for b in bills)),
        total_emergency_kwh=float(sum(b.emergency_kwh for b in bills)),
        total_penalty_yuan=float(sum(b.penalty_yuan for b in bills)),
        total_cost_yuan=float(sum(b.total_cost_yuan for b in bills)),
        max_bus_residual_kwh=max(d.max_bus_residual_kwh for d in diagnostics),
        max_soc_transition_error_kwh=max(d.max_soc_transition_error_kwh for d in diagnostics),
        max_daily_balance_residual_kwh=max(
            abs(d.daily_balance_residual_kwh) for d in diagnostics
        ),
        loss_cycle_net_kwh=float(sum(d.loss_cycle_net_kwh for d in diagnostics)),
        n_simultaneous_intervals=int(
            sum(len(d.simultaneous_charge_discharge_intervals) for d in diagnostics)
        ),
        max_bus_charge_kw=max(d.max_power_kw.get("bus_charge_kw", 0.0) for d in diagnostics),
        max_bus_discharge_kw=max(d.max_power_kw.get("bus_discharge_kw", 0.0) for d in diagnostics),
        soc_end_kwh=float(trajectories[-1].soc_kwh[-1]),
        warnings=sorted({w for d in diagnostics for w in d.warnings}),
    )


__all__ = [
    "ValidationError",
    "TrajectoryDiagnostics",
    "ValidationSummary",
    "validate_trajectory",
    "validate_day_input",
    "validate_calendar",
    "validate_bill",
    "settle_events",
    "validate_fee_classes",
    "validate_no_future_leak",
    "validate_output_days",
    "validate_fresh_start",
    "validate_attachment1_prices",
    "validate_attachment4_prices",
    "summarize",
    "E_INIT_2025_01_01",
    "Q_MAX",
    "N_BOUNDARY",
]
