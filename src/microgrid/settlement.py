"""Settlement: three fee labels, O/A recursion and independent recomputation.

Rules fixed on 2026-09-11 (memo section 6)
------------------------------------------
Every not-yet-executed interval carries two committed remainders:

* ``O`` - the original day plan remainder,
* ``A`` - the adjustment-purchase remainder,

with effective quantity ``y = O + A``; at the start of the day ``O = x``, ``A = 0``.
A revision at an allowed node ``a`` choosing new value ``y'`` gives

    O' = min(O, y'),  A' = [y' - O]_+,  delta^- = [O + A - y']_+

and each revision charges ``0.5 * p_a^act * delta^-`` at the **real moment** it
happens. The bookkeeping convention is "reduce A first then O; restored quantity
counts as A" - it is a stated convention, not a contract-batch model.

Executed fees:

    L_plan   = p_s^act * O^exec
    L_add    = 1.5 * p_s^act * A^exec
    L_reduce = sum_a 0.5 * p_a^act * delta^-_{a,s}
    L_emg    = 5 * p_s^act * u_s

The 1.5 multiplier therefore applies to the **executed adjustment quantity
A^exec**, not to "whatever exceeds the 00:00 plan". Those two differ whenever a
plan has been revised, and the gold cases below pin the difference down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    FEE_ADJUST_UP,
    FEE_EMERGENCY,
    FEE_PENALTY_DOWN,
    TOL_COST_YUAN,
    TOL_ENERGY_KWH,
)
from .schemas import CommittedBalances, DispatchEvent, PenaltyEvent


class SettlementError(RuntimeError):
    """账务不自洽——正式导出必须失败。"""


# ==========================================================================
# 1. Fee labels
# ==========================================================================


def fee_label_costs(
    o_exec_kwh: float, a_exec_kwh: float, emergency_kwh: float, price: float
) -> dict[str, float]:
    """The three executed labels plus the emergency label, at one moment."""
    return {
        "plan": price * o_exec_kwh,
        "add": FEE_ADJUST_UP * price * a_exec_kwh,
        "emergency": FEE_EMERGENCY * price * emergency_kwh,
    }


def event_cost_yuan(event: DispatchEvent) -> float:
    return event.cost_yuan


def penalty_cost_yuan(events: list[PenaltyEvent]) -> float:
    return float(sum(e.penalty_yuan for e in events))


# ==========================================================================
# 2. Revision and penalty
# ==========================================================================


def revise_commitment(
    state: CommittedBalances,
    new_y_kwh: np.ndarray,
    *,
    at_abs: int,
    price_at_yuan_per_kwh: float,
) -> tuple[CommittedBalances, PenaltyEvent | None]:
    """Apply one allowed revision and record its penalty event.

    Returns the new O/A state and the penalty event (None when nothing was cut).
    """
    new_state, delta_minus = state.revise(new_y_kwh)
    if float(np.sum(delta_minus)) <= TOL_ENERGY_KWH:
        return new_state, None
    event = PenaltyEvent(
        at_abs=at_abs,
        price_at_yuan_per_kwh=float(price_at_yuan_per_kwh),
        reduced_kwh=delta_minus,
        reduced_abs=state.abs_minutes.copy(),
    )
    return new_state, event


def reduce_quantity_for_event(
    o_kwh: np.ndarray, a_kwh: np.ndarray, y_new_kwh: np.ndarray
) -> np.ndarray:
    """``delta^- = [O + A - y']_+`` - the quantity that incurs the 0.5 penalty."""
    return np.maximum(
        np.asarray(o_kwh, dtype=np.float64)
        + np.asarray(a_kwh, dtype=np.float64)
        - np.asarray(y_new_kwh, dtype=np.float64),
        0.0,
    )


# ==========================================================================
# 3. Bills
# ==========================================================================


@dataclass
class LabelTotals:
    """Quantity and cost split by fee label, over one reporting window."""

    plan_kwh: float = 0.0
    add_kwh: float = 0.0
    emergency_kwh: float = 0.0
    plan_cost_yuan: float = 0.0
    add_cost_yuan: float = 0.0
    emergency_cost_yuan: float = 0.0
    reduce_cost_yuan: float = 0.0

    @property
    def total_kwh(self) -> float:
        return self.plan_kwh + self.add_kwh + self.emergency_kwh

    @property
    def execution_cost_yuan(self) -> float:
        return self.plan_cost_yuan + self.add_cost_yuan + self.emergency_cost_yuan

    @property
    def total_cost_yuan(self) -> float:
        return self.execution_cost_yuan + self.reduce_cost_yuan

    def as_dict(self) -> dict[str, float]:
        return {
            "plan_kwh": self.plan_kwh,
            "add_kwh": self.add_kwh,
            "emergency_kwh": self.emergency_kwh,
            "total_kwh": self.total_kwh,
            "plan_cost_yuan": self.plan_cost_yuan,
            "add_cost_yuan": self.add_cost_yuan,
            "emergency_cost_yuan": self.emergency_cost_yuan,
            "execution_cost_yuan": self.execution_cost_yuan,
            "reduce_cost_yuan": self.reduce_cost_yuan,
            "total_cost_yuan": self.total_cost_yuan,
        }


def _accumulate(event: DispatchEvent, into: LabelTotals) -> None:
    c = fee_label_costs(event.o_exec_kwh, event.a_exec_kwh, event.emergency_kwh, event.price_actual_yuan_per_kwh)
    into.plan_kwh += event.o_exec_kwh
    into.add_kwh += event.a_exec_kwh
    into.emergency_kwh += event.emergency_kwh
    into.plan_cost_yuan += c["plan"]
    into.add_cost_yuan += c["add"]
    into.emergency_cost_yuan += c["emergency"]


def natural_day_bill(
    day_from_abs: int,
    day_to_abs: int,
    events: list[DispatchEvent],
    penalties: list[PenaltyEvent],
) -> LabelTotals:
    """Natural-day bill over [day 00:00, day 24:00).

    Events and penalties are attributed by the **real moment they happened**;
    a penalty recorded while adjusting a later delivery interval still belongs
    to the day it happened on.
    """
    out = LabelTotals()
    for e in events:
        if day_from_abs <= e.at_abs < day_to_abs:
            _accumulate(e, out)
    for p in penalties:
        if day_from_abs <= p.at_abs < day_to_abs:
            out.reduce_cost_yuan += p.penalty_yuan
    return out


def result_row_bill(
    row_from_abs: int,
    row_to_abs: int,
    events: list[DispatchEvent],
    penalties: list[PenaltyEvent],
) -> LabelTotals:
    """Result-row bill over [day 00:10, next day 00:10).

    Explicitly different from the natural-day window; the two totals are not
    expected to be equal without the boundary conversion.
    """
    out = LabelTotals()
    for e in events:
        if row_from_abs <= e.at_abs < row_to_abs:
            _accumulate(e, out)
    for p in penalties:
        if row_from_abs <= p.at_abs < row_to_abs:
            out.reduce_cost_yuan += p.penalty_yuan
    return out


# ==========================================================================
# 4. Independent recomputation
# ==========================================================================


def recompute_execution_cost_yuan(events: list[DispatchEvent]) -> float:
    """Deliberately naive re-derivation, used to cross-check the accumulators."""
    total = 0.0
    for e in events:
        total += (
            e.price_actual_yuan_per_kwh * e.o_exec_kwh
            + FEE_ADJUST_UP * e.price_actual_yuan_per_kwh * e.a_exec_kwh
            + FEE_EMERGENCY * e.price_actual_yuan_per_kwh * e.emergency_kwh
        )
    return float(total)


def verify_bill(
    totals: LabelTotals,
    events: list[DispatchEvent],
    penalties: list[PenaltyEvent],
    tol: float = TOL_COST_YUAN,
) -> None:
    """Recompute by a different path; any disagreement is a hard failure."""
    execution = recompute_execution_cost_yuan(events)
    if abs(execution - totals.execution_cost_yuan) > tol:
        raise SettlementError(
            f"购电费复算不一致：账单 {totals.execution_cost_yuan:.6f}，复算 {execution:.6f}"
        )
    reduce_sum = float(sum(p.penalty_yuan for p in penalties))
    if abs(reduce_sum - totals.reduce_cost_yuan) > tol:
        raise SettlementError(
            f"违约费复算不一致：账单 {totals.reduce_cost_yuan:.6f}，复算 {reduce_sum:.6f}"
        )
    qty = sum(e.total_kwh for e in events)
    if abs(qty - totals.total_kwh) > 1e-6:
        raise SettlementError(
            f"实际购电量不一致：账单 {totals.total_kwh:.9f}，事件合计 {qty:.9f}"
        )
    if abs(totals.total_cost_yuan - (totals.execution_cost_yuan + totals.reduce_cost_yuan)) > tol:
        raise SettlementError("费用分项与合计不符")


# ==========================================================================
# 5. Gold cases from the specification (used directly by the tests)
# ==========================================================================


def gold_case_100_120_80() -> dict[str, float]:
    """Flat price 1, all revisions before execution: 100 -> 120 -> 80.

    Expected per the specification: O = 80, A = 0, penalty 20,
    total quantity 80, total cost 100.
    """
    minutes = np.arange(2) * 10
    state = CommittedBalances.from_initial_plan(minutes, np.array([100.0, 0.0]))
    state, p1 = revise_commitment(state, np.array([120.0, 0.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    state, p2 = revise_commitment(state, np.array([80.0, 0.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    penalties = [p for p in (p1, p2) if p is not None]

    # Both intervals execute; only the first carries the committed quantity.
    events = [
        DispatchEvent(0, o_exec_kwh=80.0, a_exec_kwh=0.0, emergency_kwh=0.0, price_actual_yuan_per_kwh=1.0),
        DispatchEvent(10, o_exec_kwh=0.0, a_exec_kwh=0.0, emergency_kwh=0.0, price_actual_yuan_per_kwh=1.0),
    ]
    totals = result_row_bill(0, 20, events, penalties)
    return {
        "o_kwh": float(state.o_kwh[0]),
        "a_kwh": float(state.a_kwh[0]),
        "penalty_yuan": totals.reduce_cost_yuan,
        "total_kwh": totals.total_kwh,
        "total_cost_yuan": totals.total_cost_yuan,
    }


def gold_case_100_80_100() -> dict[str, float]:
    """Flat price 1, all revisions before execution: 100 -> 80 -> 100.

    Expected per the specification: O = 80, A = 20, penalty 10,
    total quantity 100, total cost 120.
    """
    minutes = np.arange(2) * 10
    state = CommittedBalances.from_initial_plan(minutes, np.array([100.0, 0.0]))
    state, p1 = revise_commitment(state, np.array([80.0, 0.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    state, p2 = revise_commitment(state, np.array([100.0, 0.0]), at_abs=0, price_at_yuan_per_kwh=1.0)
    penalties = [p for p in (p1, p2) if p is not None]

    events = [
        DispatchEvent(
            0,
            o_exec_kwh=float(state.o_kwh[0]),
            a_exec_kwh=float(state.a_kwh[0]),
            emergency_kwh=0.0,
            price_actual_yuan_per_kwh=1.0,
        ),
        DispatchEvent(10, o_exec_kwh=0.0, a_exec_kwh=0.0, emergency_kwh=0.0, price_actual_yuan_per_kwh=1.0),
    ]
    totals = result_row_bill(0, 20, events, penalties)
    return {
        "o_kwh": float(state.o_kwh[0]),
        "a_kwh": float(state.a_kwh[0]),
        "penalty_yuan": totals.reduce_cost_yuan,
        "total_kwh": totals.total_kwh,
        "total_cost_yuan": totals.total_cost_yuan,
    }


def gold_case_execution_labels() -> dict[str, float]:
    """Executed 10 kWh of O and 10 kWh of A at price 2 -> 20 + 30 = 50."""
    events = [
        DispatchEvent(0, o_exec_kwh=10.0, a_exec_kwh=0.0, emergency_kwh=0.0, price_actual_yuan_per_kwh=2.0),
        DispatchEvent(10, o_exec_kwh=0.0, a_exec_kwh=10.0, emergency_kwh=0.0, price_actual_yuan_per_kwh=2.0),
        DispatchEvent(20, o_exec_kwh=0.0, a_exec_kwh=0.0, emergency_kwh=10.0, price_actual_yuan_per_kwh=2.0),
    ]
    totals = result_row_bill(0, 30, events, [])
    return {
        "plan_cost_yuan": totals.plan_cost_yuan,
        "add_cost_yuan": totals.add_cost_yuan,
        "emergency_cost_yuan": totals.emergency_cost_yuan,
        "execution_cost_yuan": totals.execution_cost_yuan,
    }


__all__ = [
    "SettlementError",
    "LabelTotals",
    "fee_label_costs",
    "event_cost_yuan",
    "penalty_cost_yuan",
    "revise_commitment",
    "reduce_quantity_for_event",
    "natural_day_bill",
    "result_row_bill",
    "recompute_execution_cost_yuan",
    "verify_bill",
    "gold_case_100_120_80",
    "gold_case_100_80_100",
    "gold_case_execution_labels",
]
