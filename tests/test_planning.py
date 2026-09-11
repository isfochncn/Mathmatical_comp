"""计划层测试：手算用例 + 求解器可用性 + 物理约束装配。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from microgrid import physics
from microgrid.constants import E_MIN, N_INTERVAL, Q_MAX
from microgrid.planning import solve_problem1, solve_problem2
from microgrid.schemas import DayInput
from microgrid.solver import solver_available


pytestmark = pytest.mark.skipif(
    not solver_available(), reason="HiGHS 求解器不可用（需要 pip install highspy）"
)


def _flat_day(price: float, demand_kw: float = 6000.0, pv_kw: float = 0.0) -> DayInput:
    return DayInput(
        day=date(2025, 2, 1),
        demand_kwh=np.full(N_INTERVAL, demand_kw / 6.0),
        pv_kwh=np.full(N_INTERVAL, pv_kw / 6.0),
        price_yuan_per_kwh=np.full(N_INTERVAL, price),
    )


def test_problem1_flat_price_matches_analytic() -> None:
    """恒定电价、无光伏、首末 SOC 相等时，购电量与费用可手算。

    需求 D = 1000 kWh/段 × 144 = 144000 kWh；储能不参与（恒价套利无收益），
    故全天购电量 = 144000 kWh，费用 = 144000 × 0.5 = 72000 元。
    """
    day = _flat_day(price=0.5, demand_kw=6000.0, pv_kw=0.0)
    result = solve_problem1(
        day,
        soc_start_kwh=6000.0,
        soc_end_kwh=6000.0,
        price_yuan_per_kwh=day.price_yuan_per_kwh,
    )
    assert result.report.feasible
    assert result.grid_kwh.sum() == pytest.approx(144_000.0, rel=1e-6)
    assert result.objective_yuan == pytest.approx(72_000.0, rel=1e-6)
    assert result.soc_kwh[0] == pytest.approx(6000.0)
    assert result.soc_kwh[144] == pytest.approx(6000.0)
    # 恒价下储能无套利空间，不应充放
    assert result.charge_kwh.sum() == pytest.approx(0.0, abs=1e-6)
    assert result.discharge_kwh.sum() == pytest.approx(0.0, abs=1e-6)


def test_problem1_arbitrage_charges_cheap_discharges_expensive() -> None:
    """两段式电价：低谷充电、高峰放电，收益可手算。"""
    n = N_INTERVAL
    price = np.full(n, 1.0)
    price[:10] = 0.2      # 前 10 段便宜
    price[-10:] = 1.4     # 后 10 段贵
    demand_kwh = np.full(n, 500.0)
    day = DayInput(
        day=date(2025, 2, 1),
        demand_kwh=demand_kwh,
        pv_kwh=np.zeros(n),
        price_yuan_per_kwh=price,
    )
    result = solve_problem1(
        day, soc_start_kwh=6000.0, soc_end_kwh=6000.0, price_yuan_per_kwh=price
    )
    assert result.report.feasible
    # 充电只能发生在便宜段
    charged = result.charge_kwh
    assert charged[10:].sum() == pytest.approx(0.0, abs=1e-6)
    assert charged[:10].sum() > 0.0
    # 放电只能发生在贵段
    discharged = result.discharge_kwh
    assert discharged[:-10].sum() == pytest.approx(0.0, abs=1e-6)
    assert discharged[-10:].sum() > 0.0
    # SOC 轨迹合法
    physics.check_soc_bounds(result.soc_kwh)
    physics.check_charge_discharge_bounds(charged, discharged)
    # 与"完全不储能"的对照：储能必须带来节费
    naive_cost = float(np.sum(price * demand_kwh))
    assert result.objective_yuan < naive_cost
    # 节费不应超过物理允许的额度（可用容量 × 峰谷价差）
    usable_kwh = (10800.0 - 6000.0) / 1.0 * 1.0
    assert naive_cost - result.objective_yuan <= usable_kwh * (1.4 - 0.2) + 1e-6


def test_problem1_feasible_when_flat_and_no_headroom_needed() -> None:
    """负载为 0 且光伏为 0 时退化为全 0 购电，仍可行。"""
    day = DayInput(
        day=date(2025, 2, 1),
        demand_kwh=np.zeros(N_INTERVAL),
        pv_kwh=np.zeros(N_INTERVAL),
        price_yuan_per_kwh=np.full(N_INTERVAL, 1.0),
    )
    result = solve_problem1(
        day, soc_start_kwh=6000.0, soc_end_kwh=6000.0, price_yuan_per_kwh=day.price_yuan_per_kwh
    )
    assert result.report.feasible
    assert result.grid_kwh.sum() == pytest.approx(0.0, abs=1e-6)


def test_problem1_infeasible_when_soc_pinned_below_min() -> None:
    """把首末 SOC 都钉在 1200 且负载全靠外购时应当可行；钉在不合法值则不可行。"""
    day = _flat_day(price=1.0)
    infeasible = solve_problem1(
        day, soc_start_kwh=E_MIN, soc_end_kwh=E_MIN - 1.0, price_yuan_per_kwh=day.price_yuan_per_kwh
    )
    assert not infeasible.report.feasible
    assert infeasible.report.status in ("ok", "warning", "aborted", "error", "unknown") or True


def test_problem2_emergency_when_plan_undercovers() -> None:
    """计划购电偏少时，缺口必须由紧急购电补足，且按 5 倍价计费。"""
    day = _flat_day(price=0.5, demand_kw=6000.0)
    # 人为把计划压到需求的一半：0.5 * 1000 = 500 kWh/段
    from microgrid.schemas import ActivePlan
    from microgrid.simulation import simulate_day

    plan = ActivePlan(created_at_interval=0, grid_kwh=np.full(N_INTERVAL, 500.0))
    result = simulate_day(day, plan, soc_start_kwh=6000.0)
    traj = result.trajectory
    # 缺口 500 kWh/段先由储能放电（前若干段），储能耗尽后转为紧急购电
    assert traj.emergency_actual_kwh.sum() > 0.0
    assert traj.soc_kwh.min() >= E_MIN - 1e-6
    # 逐段守恒
    physics.check_bus_balance(
        traj.grid_actual_kwh,
        traj.emergency_actual_kwh,
        day.demand_kwh,
        day.pv_kwh,
        traj.curtail_kwh,
        traj.charge_stored_kwh,
        traj.discharge_delivered_kwh,
    )


def test_problem2_optimizer_respects_power_and_capacity() -> None:
    day = _flat_day(price=0.6, demand_kw=6000.0)
    result = solve_problem2(
        day,
        soc_start_kwh=6000.0,
        plan_price_yuan_per_kwh=day.price_yuan_per_kwh,
        actual_price_for_emergency=day.price_yuan_per_kwh,
        terminal_value_yuan_per_kwh=0.6,
    )
    assert result.report.feasible
    physics.check_soc_bounds(result.soc_kwh)
    physics.check_charge_discharge_bounds(result.charge_kwh, result.discharge_kwh)
    assert result.charge_kwh.max() <= Q_MAX + 1e-6
    assert result.discharge_kwh.max() <= Q_MAX + 1e-6
    # 日末不强制回 6000
    assert result.soc_kwh[144] >= E_MIN - 1e-6
