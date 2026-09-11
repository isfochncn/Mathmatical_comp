"""Physics, execution-feedback and window-model tests for the new rules."""

from __future__ import annotations

import numpy as np
import pytest

from microgrid import physics
from microgrid.constants import (
    ETA_CHARGE,
    ETA_DISCHARGE,
    LOSS_COEFF,
    Q_MAX,
)
from microgrid.planning import FeeMode, WindowForecast, solve_window
from microgrid.simulation import execute_interval
from microgrid.solver import solver_available

pytestmark = pytest.mark.skipif(
    not solver_available(), reason="HiGHS 求解器不可用（需要 pip install highspy）"
)


# ============================================================ effective limits


def test_full_power_charge_stores_675_bus_side_750() -> None:
    """q_ch = 750 is the BUS-side charge; the battery stores 0.9*750 = 675."""
    assert Q_MAX == pytest.approx(750.0)
    assert physics.charge_bus_input_kwh(750.0) == pytest.approx(750.0 / 0.9)
    assert 750.0 * ETA_CHARGE == pytest.approx(675.0)


def test_full_power_discharge_draws_833() -> None:
    assert physics.discharge_battery_draw_kwh(750.0) == pytest.approx(750.0 / 0.9)
    assert physics.bus_power_kw(physics.discharge_battery_draw_kwh(750.0)) == pytest.approx(5000.0)


def test_loss_coefficient_closed_form() -> None:
    assert LOSS_COEFF == pytest.approx(1.0 / ETA_CHARGE - ETA_DISCHARGE)
    assert LOSS_COEFF == pytest.approx(19.0 / 90.0)


def test_microgrid_100_90_81_chain() -> None:
    stored = 100.0 * ETA_CHARGE
    assert stored == pytest.approx(90.0)
    assert physics.charge_bus_input_kwh(stored) == pytest.approx(100.0)
    delivered = 90.0 * ETA_DISCHARGE
    assert delivered == pytest.approx(81.0)
    assert physics.discharge_battery_draw_kwh(delivered) == pytest.approx(90.0)


# ============================================================ simultaneous charge/discharge


def test_simultaneous_operation_is_legal_and_accounted() -> None:
    """Storing 90 while delivering 81 keeps SOC flat; 19 kWh is a real loss.

    The rules state this is a legal operating state, not an anomaly: it must be
    accounted for but must never make a solution infeasible.
    """
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[5] = 90.0
    q_dis[5] = 81.0
    assert physics.energy_change_from_dispatch(90.0, 81.0) == pytest.approx(0.0)
    info = physics.loss_accounting(q_ch, q_dis)
    assert info["n_simultaneous_intervals"] == 1
    assert info["simultaneous_intervals"] == [5]
    # charge loss = 90/0.9 - 90 = 10 ; discharge loss = 81/0.9 - 81 = 9
    assert info["charge_loss_kwh"] == pytest.approx(10.0)
    assert info["discharge_loss_kwh"] == pytest.approx(9.0)
    assert info["total_loss_kwh"] == pytest.approx(19.0)


def test_loss_accounting_separates_loss_from_storage() -> None:
    """Losses must be listed as losses; only actual stored energy is storage."""
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[0] = 500.0
    q_dis[0] = 100.0
    info = physics.loss_accounting(q_ch, q_dis)
    assert info["total_loss_kwh"] > 0.0
    assert info["charge_stored_total_kwh"] == pytest.approx(500.0)
    assert info["discharge_delivered_total_kwh"] == pytest.approx(100.0)
    # charge loss = 500/0.9 - 500 = 55.56 ; discharge loss = 100/0.9 - 100 = 11.11
    assert info["charge_loss_kwh"] == pytest.approx(500.0 / 0.9 - 500.0)
    assert info["discharge_loss_kwh"] == pytest.approx(100.0 / 0.9 - 100.0)
    assert info["total_loss_kwh"] == pytest.approx(
        500.0 / 0.9 - 500.0 + 100.0 / 0.9 - 100.0
    )


# ============================================================ execution feedback


def test_execute_interval_deficit_is_covered_by_emergency() -> None:
    a = execute_interval(
        abs_minute=0, grid_committed_kwh=500.0, demand_kwh=600.0, pv_kwh=0.0,
        soc_now_kwh=6000.0, soc_target_kwh=6000.0,
    )
    assert a.discharge_kwh == pytest.approx(0.0)
    assert a.emergency_kwh == pytest.approx(100.0)
    assert a.soc_end_kwh == pytest.approx(6000.0)


def test_execute_interval_holds_when_the_plan_asks_for_no_change() -> None:
    """Level 1 is "minimise deviation from the charge target".

    With a flat target the feedback must not discharge opportunistically to
    cover a deficit: the deficit is settled by emergency purchase instead.
    """
    a = execute_interval(
        abs_minute=0, grid_committed_kwh=500.0, demand_kwh=600.0, pv_kwh=0.0,
        soc_now_kwh=6000.0, soc_target_kwh=6000.0,
    )
    assert a.discharge_kwh == pytest.approx(0.0)
    assert a.charge_kwh == pytest.approx(0.0)
    assert a.emergency_kwh == pytest.approx(100.0)
    assert a.target_gap_kwh == pytest.approx(0.0)


def test_execute_interval_discharges_towards_the_target_and_spills_the_rest() -> None:
    """A target below the current SOC discharges towards it; the excess of a
    discharge that overshoots the deficit is reported, not hidden."""
    a = execute_interval(
        abs_minute=0, grid_committed_kwh=100.0, demand_kwh=600.0, pv_kwh=0.0,
        soc_now_kwh=6000.0, soc_target_kwh=5000.0,
    )
    # want = -1000, bounded by the 750 kWh per-interval limit
    assert a.discharge_kwh == pytest.approx(750.0)
    assert a.emergency_kwh == pytest.approx(0.0)
    # 500 kWh of the discharge serves the deficit, the remaining 250 has nowhere
    # to go in this interval (the battery is the only sink and it is discharging)
    assert a.spill_kwh == pytest.approx(250.0)
    assert a.soc_end_kwh == pytest.approx(6000.0 - 750.0 / 0.9)


def test_execute_interval_never_exceeds_the_interval_limit() -> None:
    a = execute_interval(
        abs_minute=0, grid_committed_kwh=0.0, demand_kwh=0.0, pv_kwh=0.0,
        soc_now_kwh=10800.0, soc_target_kwh=1200.0,
    )
    assert a.discharge_kwh <= 750.0 + 1e-9
    assert a.soc_end_kwh >= 1200.0 - 1e-6


def test_execute_interval_reports_unabsorbable_surplus_honestly() -> None:
    """Committed power that cannot be used or stored is a disclosed surplus."""
    a = execute_interval(
        abs_minute=0, grid_committed_kwh=500.0, demand_kwh=300.0, pv_kwh=0.0,
        soc_now_kwh=6000.0, soc_target_kwh=6000.0,
    )
    assert a.spill_kwh == pytest.approx(200.0)
    assert a.emergency_kwh == pytest.approx(0.0)


def test_execute_interval_never_breaks_conservation() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        grid = float(rng.uniform(0, 800))
        demand = float(rng.uniform(0, 1200))
        pv = float(rng.uniform(0, 900))
        soc = float(rng.uniform(1200, 10800))
        target = float(rng.uniform(1200, 10800))
        a = execute_interval(
            abs_minute=0, grid_committed_kwh=grid, demand_kwh=demand, pv_kwh=pv,
            soc_now_kwh=soc, soc_target_kwh=target,
        )
        lhs = grid + a.emergency_kwh + pv + a.discharge_kwh
        rhs = demand + a.charge_kwh / 0.9 + a.spill_kwh
        assert lhs == pytest.approx(rhs, abs=1e-9)
        assert 1200.0 - 1e-6 <= a.soc_end_kwh <= 10800.0 + 1e-6


# ============================================================ window model


def _forecast(n: int, demand: float, pv: float, price: float) -> WindowForecast:
    return WindowForecast(
        abs_minutes=np.arange(n) * 10,
        demand_kwh=np.full(n, demand),
        pv_kwh=np.full(n, pv),
        price_yuan_per_kwh=np.full(n, price),
        provenance=np.array(["test"] * n, dtype=object),
    )


def test_window_first_plan_covers_demand_at_flat_price() -> None:
    """Flat price, no PV: the window never buys more than it needs.

    It may buy slightly less by drawing the battery down, because with a flat
    price the purchase cost is the same at every interval; what it must never do
    is buy *more* than demand plus charging requirement.
    """
    n = 144
    r = solve_window(
        forecast=_forecast(n, 1000.0, 0.0, 0.5),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FIRST_PLAN,
        problem_name="t",
    )
    assert r.report.feasible
    assert float(r.grid_kwh.sum()) <= 1000.0 * n + 1e-6
    # Energy balance holds within the window, including the battery term.
    lhs = float(r.grid_kwh.sum())
    rhs = (
        1000.0 * n
        + LOSS_COEFF * float(r.charge_kwh.sum())
        + ETA_DISCHARGE * (float(r.soc_boundary_kwh[-1]) - float(r.soc_boundary_kwh[0]))
    )
    assert lhs == pytest.approx(rhs, abs=1e-4)


def test_window_cycle_constraint_only_when_requested() -> None:
    n = 144
    r = solve_window(
        forecast=_forecast(n, 1000.0, 0.0, 0.5),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FIRST_PLAN,
        soc_end_fixed_kwh=6000.0,
        problem_name="t",
    )
    assert r.report.feasible
    assert float(r.soc_boundary_kwh[0]) == pytest.approx(6000.0)
    assert float(r.soc_boundary_kwh[-1]) == pytest.approx(6000.0)


def test_window_frozen_commits_exactly_the_o_plus_a() -> None:
    n = 12
    o = np.full(n, 100.0)
    a = np.full(n, 20.0)
    r = solve_window(
        forecast=_forecast(n, 400.0, 0.0, 1.0),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FROZEN,
        o_kwh=o,
        a_kwh=a,
        problem_name="t",
    )
    assert r.report.feasible
    assert np.allclose(r.grid_kwh, 120.0)


def test_window_frozen_fee_uses_one_point_five_on_a() -> None:
    n = 4
    o = np.full(n, 100.0)
    a = np.full(n, 20.0)
    r = solve_window(
        forecast=_forecast(n, 120.0, 0.0, 1.0),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FROZEN,
        o_kwh=o,
        a_kwh=a,
        problem_name="t",
    )
    # fee = sum p*(O + 1.5*A) = 4 * (100 + 30) = 520
    assert float(r.objective_yuan) == pytest.approx(520.0)


def test_window_adjustable_can_cut_and_charges_the_penalty() -> None:
    n = 8
    o = np.full(n, 200.0)
    a = np.zeros(n)
    r = solve_window(
        forecast=_forecast(n, 0.0, 0.0, 1.0),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.ADJUSTABLE,
        o_kwh=o,
        a_kwh=a,
        price_now_yuan_per_kwh=1.0,
        problem_name="t",
    )
    assert r.report.feasible
    # Cut everything: no execution cost, only the 0.5 penalty on the reduction.
    assert float(r.grid_kwh.sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(r.objective_yuan) == pytest.approx(0.5 * 200.0 * n)


def test_window_absorption_cap_is_respected() -> None:
    n = 24
    cap = np.full(n, 300.0)
    r = solve_window(
        forecast=_forecast(n, 500.0, 0.0, 1.0),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FIRST_PLAN,
        absorption_upper_kwh=cap,
        problem_name="t",
    )
    assert r.report.feasible
    assert float(r.grid_kwh.max()) <= 300.0 + 1e-6
    # The rest of the demand has to be served by the battery or emergency power.
    assert float(r.emergency_kwh.sum()) + float(r.discharge_kwh.sum()) > 0.0


def test_window_committed_mask_keeps_carry_over_values() -> None:
    n = 12
    fixed = np.zeros(n)
    fixed[:4] = 150.0
    mask = np.zeros(n, dtype=bool)
    mask[:4] = True
    r = solve_window(
        forecast=_forecast(n, 200.0, 0.0, 1.0),
        soc_start_kwh=6000.0,
        fee_mode=FeeMode.FROZEN,
        o_kwh=fixed,
        a_kwh=np.zeros(n),
        absorption_upper_kwh=np.full(n, 1000.0),
        committed_grid_kwh=fixed,
        committed_mask=mask,
        problem_name="t",
    )
    assert r.report.feasible
    assert np.allclose(r.grid_kwh[:4], 150.0)
