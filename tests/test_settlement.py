"""Settlement tests: O/A recursion and the three fee labels.

These pin down the two gold cases from the specification plus the label rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from microgrid.schemas import CommittedBalances, DispatchEvent
from microgrid import settlement as S


# ============================================================ gold cases


def test_gold_case_100_120_80() -> None:
    """Flat price 1, revisions before execution: 100 -> 120 -> 80.

    Specification: O = 80, A = 0, penalty 20, quantity 80, cost 100.
    """
    r = S.gold_case_100_120_80()
    assert r["o_kwh"] == pytest.approx(80.0)
    assert r["a_kwh"] == pytest.approx(0.0)
    assert r["penalty_yuan"] == pytest.approx(20.0)
    assert r["total_kwh"] == pytest.approx(80.0)
    assert r["total_cost_yuan"] == pytest.approx(100.0)


def test_gold_case_100_80_100() -> None:
    """Flat price 1, revisions before execution: 100 -> 80 -> 100.

    Specification: O = 80, A = 20, penalty 10, quantity 100, cost 120.
    This is the case that distinguishes "1.5x applies to A^exec" from the old
    "1.5x applies to whatever exceeds the 00:00 plan" reading.
    """
    r = S.gold_case_100_80_100()
    assert r["o_kwh"] == pytest.approx(80.0)
    assert r["a_kwh"] == pytest.approx(20.0)
    assert r["penalty_yuan"] == pytest.approx(10.0)
    assert r["total_kwh"] == pytest.approx(100.0)
    assert r["total_cost_yuan"] == pytest.approx(120.0)


def test_execution_labels() -> None:
    """10 kWh of O at price 2 -> 20; 10 kWh of A at price 2 -> 30; 10 kWh
    emergency at price 2 -> 100."""
    c = S.gold_case_execution_labels()
    assert c["plan_cost_yuan"] == pytest.approx(20.0)
    assert c["add_cost_yuan"] == pytest.approx(30.0)
    assert c["emergency_cost_yuan"] == pytest.approx(100.0)
    assert c["execution_cost_yuan"] == pytest.approx(150.0)


# ============================================================ recursion


def test_recursion_matches_the_closed_form() -> None:
    """O' = min(O, y'), A' = [y' - O]+, delta^- = [O + A - y']+."""
    state = CommittedBalances.from_initial_plan(np.arange(3) * 10, np.array([100.0, 100.0, 100.0]))
    new, delta = state.revise(np.array([120.0, 80.0, 100.0]))
    assert new.o_kwh.tolist() == [100.0, 80.0, 100.0]
    assert new.a_kwh.tolist() == [20.0, 0.0, 0.0]
    assert delta.tolist() == [0.0, 20.0, 0.0]


def test_reduce_then_restore_keeps_the_penalty() -> None:
    minutes = np.arange(1) * 10
    state = CommittedBalances.from_initial_plan(minutes, np.array([100.0]))
    state, p1 = S.revise_commitment(state, np.array([80.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    state, p2 = S.revise_commitment(state, np.array([100.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    assert p1 is not None and p1.penalty_yuan == pytest.approx(10.0)
    assert p2 is None  # restoring does not create a new penalty
    assert state.o_kwh[0] == pytest.approx(80.0)
    assert state.a_kwh[0] == pytest.approx(20.0)


def test_penalty_uses_the_price_at_the_real_moment() -> None:
    """A penalty recorded while adjusting a later interval still uses p_a."""
    minutes = np.arange(2) * 10
    state = CommittedBalances.from_initial_plan(minutes, np.array([0.0, 100.0]))
    state, event = S.revise_commitment(
        state, np.array([0.0, 60.0]), at_abs=10, price_at_yuan_per_kwh=2.0
    )
    assert event is not None
    assert event.at_abs == 10
    assert event.total_reduced_kwh == pytest.approx(40.0)
    assert event.penalty_yuan == pytest.approx(0.5 * 2.0 * 40.0)


def test_no_penalty_when_only_increasing() -> None:
    state = CommittedBalances.from_initial_plan(np.array([0]), np.array([50.0]))
    _, event = S.revise_commitment(state, np.array([70.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    assert event is None


# ============================================================ windows


def test_natural_day_and_result_row_windows_differ() -> None:
    """Execution exactly at 00:00 belongs to the natural day, not the result row."""
    events = [
        DispatchEvent(0, 10.0, 0.0, 0.0, 1.0),      # 00:00-00:10
        DispatchEvent(10, 10.0, 0.0, 0.0, 1.0),     # 00:10-00:20
        DispatchEvent(1440, 10.0, 0.0, 0.0, 1.0),   # next day 00:00
    ]
    day = S.natural_day_bill(0, 1440, events, [])
    row = S.result_row_bill(10, 1450, events, [])
    assert day.total_kwh == pytest.approx(20.0)   # 00:00 and 00:10
    assert row.total_kwh == pytest.approx(20.0)   # 00:10 and next 00:00
    # the interval at exactly 00:00 is in the natural day but not in the row
    assert S.natural_day_bill(0, 10, events, []).total_kwh == pytest.approx(10.0)
    assert S.result_row_bill(10, 20, events, []).total_kwh == pytest.approx(10.0)


def test_verify_bill_detects_tampering() -> None:
    events = [
        DispatchEvent(0, 10.0, 5.0, 1.0, 2.0),
    ]
    totals = S.LabelTotals()
    for e in events:
        pass
    totals = S.result_row_bill(0, 10, events, [])
    S.verify_bill(totals, events, [])
    bad = S.LabelTotals(
        plan_kwh=totals.plan_kwh,
        add_kwh=totals.add_kwh,
        emergency_kwh=totals.emergency_kwh,
        plan_cost_yuan=totals.plan_cost_yuan + 1.0,
        add_cost_yuan=totals.add_cost_yuan,
        emergency_cost_yuan=totals.emergency_cost_yuan,
    )
    with pytest.raises(S.SettlementError):
        S.verify_bill(bad, events, [])


def test_execution_cost_is_rederived_from_events() -> None:
    events = [
        DispatchEvent(0, 100.0, 0.0, 0.0, 0.5),
        DispatchEvent(10, 0.0, 40.0, 0.0, 0.5),
        DispatchEvent(20, 0.0, 0.0, 10.0, 0.5),
    ]
    direct = S.recompute_execution_cost_yuan(events)
    totals = S.result_row_bill(0, 30, events, [])
    assert direct == pytest.approx(totals.execution_cost_yuan)
    assert direct == pytest.approx(0.5 * 100 + 1.5 * 0.5 * 40 + 5.0 * 0.5 * 10)
