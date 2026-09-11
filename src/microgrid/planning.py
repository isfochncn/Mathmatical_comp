"""Rolling-window planning models.

The 2026-09-11 rules replaced the old "one natural day + abstract terminal
value" formulation with a single deterministic rolling window:

    a = the start of the current ten-minute interval
    b = the next natural day 00:00
    T = b + 24h
    solve once over [a, T)

so the 00:00 window spans 48 h and the 06:00 / 12:00 / 18:00 windows span
42 / 36 / 30 h; at other intervals the window is the remaining part of [a, T).

The cross-day value is **not** a fitted terminal term. It is exactly the cost
of the next-day predicted dispatch, which is what a joint optimisation over
[b, T) already computes. At T only the device capacity bounds apply: no
salvage revenue, no terminal reward, no forced return to 6000.

Three fee expressions (see the memo section 6)
----------------------------------------------
* first-day plan (no adjustment possible yet):  ``f = p * x``
* frozen segment (P2 / P4-2, and P3/P4-3 outside adjust nodes):
  ``f = p * (O + 1.5*A)``
* adjustable segment (P3 / P4-3 at 06/12/18):
  ``f = p*y' + 0.5*p*[y' - O]+ + 0.5*p_a*[O + A - y']+``

All three are convex piecewise linear in y', so the whole model stays an LP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from .constants import (
    CHARGE_BUS_FACTOR,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    FEE_ADJUST_UP,
    FEE_EMERGENCY,
    FEE_PENALTY_DOWN,
    Q_DIS_MAX,
    Q_MAX,
)
from .schemas import SolveReport
from .solver import DEFAULT_SOLVER, solve_model


@dataclass
class WindowForecast:
    """One point-forecast trajectory covering the whole optimisation window."""

    abs_minutes: np.ndarray      # (n,) window interval start minutes
    demand_kwh: np.ndarray       # (n,)
    pv_kwh: np.ndarray           # (n,)
    price_yuan_per_kwh: np.ndarray  # (n,) planning price p_hat_{s|a}
    provenance: np.ndarray       # (n,) of str, same values as timeline markers

    def __post_init__(self) -> None:
        n = np.asarray(self.abs_minutes, dtype=np.int64).size
        self.abs_minutes = np.asarray(self.abs_minutes, dtype=np.int64)
        for name in ("demand_kwh", "pv_kwh", "price_yuan_per_kwh", "provenance"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(f"WindowForecast.{name} shape should be ({n},), got {arr.shape}")
            setattr(self, name, arr)

    @property
    def n(self) -> int:
        return int(self.abs_minutes.size)


@dataclass
class WindowResult:
    """A solved window: full schedule plus the solver report."""

    report: SolveReport
    abs_minutes: np.ndarray
    grid_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_boundary_kwh: np.ndarray
    objective_yuan: float

    def require_ok(self) -> "WindowResult":
        self.report.require_ok()
        return self


# ==========================================================================
# Fee expressions
# ==========================================================================


class FeeMode:
    """Which fee expression applies to the grid variable in a window."""

    FIRST_PLAN = "first_plan"     # f = p * x  (no O/A yet)
    FROZEN = "frozen"             # f = p * (O + 1.5 A)  -- O, A constant
    ADJUSTABLE = "adjustable"     # convex piecewise linear in y'


def _add_frozen_fee(
    m: pyo.ConcreteModel,
    price: np.ndarray,
    o_kwh: np.ndarray,
    a_kwh: np.ndarray,
    committed_mask: np.ndarray | None = None,
) -> None:
    """Frozen segment: O and A are constants, so the fee is a linear function.

    ``committed_mask``
        Which intervals are actually under commitment. Only those get
        ``grid == O + 1.5*A``; the rest of the window is still free and buys at
        the normal rate. Forcing the whole window to the committed quantity
        would pin the uncommitted tail to zero and make the model infeasible
        whenever that tail has to serve load.
    """
    if committed_mask is None:
        committed_mask = np.ones(np.asarray(o_kwh).size, dtype=bool)
    mask = np.asarray(committed_mask, dtype=bool)

    def _rule(m, t):
        if mask[t]:
            return price[t] * (float(o_kwh[t]) + FEE_ADJUST_UP * float(a_kwh[t]))
        return price[t] * m.grid[t]

    m.fee_fixed = pyo.Expression(m.T, rule=_rule)

    fixed = [t for t in m.T if mask[t]]
    if fixed:
        m.commit = pyo.Constraint(
            fixed, rule=lambda m, t: m.grid[t] == float(o_kwh[t] + a_kwh[t])
        )


def _add_adjustable_fee(
    m: pyo.ConcreteModel,
    price_exec: np.ndarray,
    price_now: float,
    o_kwh: np.ndarray,
    a_kwh: np.ndarray,
) -> None:
    """Adjustable segment: convex piecewise-linear fee via the epigraph of [.]_+.

    f(y') = p_s*y' + 0.5*p_s*v + 0.5*p_a*w,  v >= y'-O, v >= 0, w >= O+A-y', w >= 0

    The **actual** fee labels are always recomputed from the realised y' with the
    exact min/positive-part formulas; the auxiliary variables are only a modelling
    device (and they may be slack when a price is zero).
    """
    m.increase = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # v
    m.reduce = pyo.Var(m.T, domain=pyo.NonNegativeReals)     # w

    m.increase_lb = pyo.Constraint(
        m.T, rule=lambda m, t: m.increase[t] >= m.grid[t] - float(o_kwh[t])
    )
    m.reduce_lb = pyo.Constraint(
        m.T, rule=lambda m, t: m.reduce[t] >= float(o_kwh[t] + a_kwh[t]) - m.grid[t]
    )

    def _rule(m, t):
        return (
            price_exec[t] * m.grid[t]
            + FEE_ADJUST_UP * price_exec[t] * m.increase[t]
            + FEE_PENALTY_DOWN * float(price_now) * m.reduce[t]
        )

    m.fee_fixed = pyo.Expression(m.T, rule=_rule)


def _add_first_plan_fee(m: pyo.ConcreteModel, price: np.ndarray) -> None:
    """First plan of the day: plain normal-rate purchase, no adjustment fees."""
    m.fee_fixed = pyo.Expression(m.T, rule=lambda m, t: price[t] * m.grid[t])


# ==========================================================================
# Window model
# ==========================================================================


def solve_window(
    *,
    forecast: WindowForecast,
    soc_start_kwh: float,
    fee_mode: str,
    o_kwh: np.ndarray | None = None,
    a_kwh: np.ndarray | None = None,
    price_now_yuan_per_kwh: float | None = None,
    soc_end_fixed_kwh: float | None = None,
    absorption_upper_kwh: np.ndarray | None = None,
    commitment_lower_kwh: np.ndarray | None = None,
    committed_grid_kwh: np.ndarray | None = None,
    committed_mask: np.ndarray | None = None,
    solver_name: str = DEFAULT_SOLVER,
    problem_name: str = "window",
) -> WindowResult:
    """Solve one rolling window over the forecast's absolute intervals.

    Parameters
    ----------
    soc_start_kwh
        Measured state of charge at ``a`` (never a predicted SOC).
    fee_mode
        One of :class:`FeeMode`.
    o_kwh, a_kwh
        Committed original-plan / adjustment-purchase remainders. Needed by the
        FROZEN and ADJUSTABLE fee expressions. Intervals that are not committed
        yet are passed as 0 and are freed by ``committed_mask``.
    committed_grid_kwh, committed_mask
        Per-interval carry-over of already committed quantities. This is how the
        previous natural day's plan keeps covering the next day's first ten
        minutes while the rest of the window is being re-optimised. Intervals
        with ``committed_mask == True`` have their grid value fixed.
    soc_end_fixed_kwh
        Only problem 1 (``E_144 = E_0 = 6000``).
    absorption_upper_kwh
        Robust cap on the normal purchase quantity. Delivered power cannot be
        rejected, so a commitment has to stay absorbable.
    """
    n = forecast.n
    if n <= 0:
        raise ValueError("empty window")
    T = list(range(n))

    m = pyo.ConcreteModel(problem_name)
    m.T = pyo.Set(initialize=T, ordered=True)
    m.S = pyo.Set(initialize=list(range(n + 1)), ordered=True)

    m.demand = pyo.Param(m.T, initialize={t: float(forecast.demand_kwh[t]) for t in T})
    m.pv = pyo.Param(m.T, initialize={t: float(forecast.pv_kwh[t]) for t in T})

    m.grid = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.emergency = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_MAX))
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_DIS_MAX))
    m.curtail = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.S, domain=pyo.NonNegativeReals, bounds=(E_MIN, E_MAX))

    # Bus conservation, 2026-09-11 physics core (unchanged).
    m.balance = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.grid[t]
        + m.emergency[t]
        + m.pv[t]
        - m.curtail[t]
        + m.discharge[t]
        == m.demand[t] + m.charge[t] * CHARGE_BUS_FACTOR,
    )

    m.transition = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.soc[t + 1]
        == m.soc[t] + m.charge[t] - m.discharge[t] * DISCHARGE_BATTERY_FACTOR,
    )

    m.curtail_limit = pyo.Constraint(m.T, rule=lambda m, t: m.curtail[t] <= m.pv[t])
    m.soc_start = pyo.Constraint(expr=m.soc[0] == float(soc_start_kwh))

    if absorption_upper_kwh is not None and committed_mask is None:
        cap = np.asarray(absorption_upper_kwh, dtype=np.float64)
        if cap.size != n:
            raise ValueError(f"absorption_upper_kwh shape should be ({n},), got {cap.shape}")
        m.absorption = pyo.Constraint(m.T, rule=lambda m, t: m.grid[t] <= float(cap[t]))

    if commitment_lower_kwh is not None and committed_mask is None:
        lo = np.asarray(commitment_lower_kwh, dtype=np.float64)
        if lo.size != n:
            raise ValueError(f"commitment_lower_kwh shape should be ({n},), got {lo.shape}")
        m.commitment_floor = pyo.Constraint(
            m.T, rule=lambda m, t: m.grid[t] >= float(lo[t])
        )

    # ---- carry-over of already committed quantities -------------------------
    o_eff = np.zeros(n, dtype=np.float64)
    a_eff = np.zeros(n, dtype=np.float64)
    if o_kwh is not None and a_kwh is not None:
        o_arr = np.asarray(o_kwh, dtype=np.float64)
        a_arr = np.asarray(a_kwh, dtype=np.float64)
        if o_arr.size != n or a_arr.size != n:
            raise ValueError(f"O/A shape should be ({n},), got {o_arr.shape}/{a_arr.shape}")
        o_eff, a_eff = o_arr, a_arr

    if committed_mask is not None and committed_grid_kwh is not None:
        mask = np.asarray(committed_mask, dtype=bool)
        values = np.asarray(committed_grid_kwh, dtype=np.float64)
        if mask.size != n or values.size != n:
            raise ValueError(f"committed carry-over shape should be ({n},)")
        fixed = [t for t in T if mask[t]]
        if fixed:
            m.fix_committed = pyo.Constraint(
                fixed, rule=lambda m, t: m.grid[t] == float(values[t])
            )

    # ---- fee expression ------------------------------------------------------
    #
    # The robust purchase bounds (absorption cap / commitment floor) are guards on
    # the *decision* of how much to buy. They must NOT be applied to intervals
    # whose quantity is already committed: those are equality-fixed and re-testing
    # them against a bound computed from a different forecast is what made the
    # frozen window come out infeasible.
    if fee_mode == FeeMode.FIRST_PLAN:
        _add_first_plan_fee(m, forecast.price_yuan_per_kwh)
    elif fee_mode == FeeMode.FROZEN:
        if committed_mask is None:
            _add_frozen_fee(m, forecast.price_yuan_per_kwh, o_eff, a_eff)
        else:
            _add_frozen_fee(m, forecast.price_yuan_per_kwh, o_eff, a_eff, committed_mask)
    elif fee_mode == FeeMode.ADJUSTABLE:
        if price_now_yuan_per_kwh is None:
            raise ValueError("ADJUSTABLE mode needs the current adjustment price p_a")
        _add_adjustable_fee(
            m, forecast.price_yuan_per_kwh, float(price_now_yuan_per_kwh), o_eff, a_eff
        )
    else:
        raise ValueError(f"unknown fee mode: {fee_mode}")

    m.obj = pyo.Objective(
        rule=lambda m: sum(m.fee_fixed[t] for t in T)
        + sum(FEE_EMERGENCY * forecast.price_yuan_per_kwh[t] * m.emergency[t] for t in T),
        sense=pyo.minimize,
    )

    if soc_end_fixed_kwh is not None:
        m.soc_end = pyo.Constraint(expr=m.soc[n] == float(soc_end_fixed_kwh))

    report = solve_model(m, problem_name, solver_name=solver_name)
    if not report.feasible:
        return WindowResult(
            report=report,
            abs_minutes=forecast.abs_minutes.copy(),
            grid_kwh=np.zeros(n),
            emergency_kwh=np.zeros(n),
            charge_kwh=np.zeros(n),
            discharge_kwh=np.zeros(n),
            curtail_kwh=np.zeros(n),
            soc_boundary_kwh=np.full(n + 1, float(soc_start_kwh)),
            objective_yuan=float("nan"),
        )

    get = lambda var, i: float(pyo.value(var[i]))  # noqa: E731

    return WindowResult(
        report=report,
        abs_minutes=forecast.abs_minutes.copy(),
        grid_kwh=np.array([get(m.grid, t) for t in T]),
        emergency_kwh=np.array([get(m.emergency, t) for t in T]),
        charge_kwh=np.array([get(m.charge, t) for t in T]),
        discharge_kwh=np.array([get(m.discharge, t) for t in T]),
        curtail_kwh=np.array([get(m.curtail, t) for t in T]),
        soc_boundary_kwh=np.array([get(m.soc, s) for s in m.S]),
        objective_yuan=float(report.objective_yuan or float("nan")),
    )


__all__ = [
    "WindowForecast",
    "WindowResult",
    "FeeMode",
    "solve_window",
]
