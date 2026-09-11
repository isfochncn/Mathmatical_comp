"""Causal execution: per-interval feedback plus the rolling-window driver.

Two layers live here.

1. :func:`execute_interval` - the **reproducible execution interface** fixed by
   the memo. It uses only the current state, the current supply/demand
   observation and the committed charge/discharge target; never future data.
   Priority order:

       1. minimise the absolute deviation from this interval's charge target,
       2. among equally-deviating actions, minimise this interval's emergency
          purchase,
       3. among those, minimise throughput.

   The ordinary purchase quantity is fixed and must not be modified; curtailment
   applies to PV only; emergency purchase fills the remaining gap. Everything is
   solved analytically, which is exact for this scalar problem and much faster
   than an LP per ten minutes.

2. :func:`run_absolute` - walks the absolute timeline. At *every* interval it
   refreshes the point forecast, re-solves the window ``[a, T)`` to obtain the
   storage target, commits a plan revision only at legal nodes, executes the
   current interval, and books the realised quantities. Predicted SOC is never
   reused as measured SOC: the next step restarts from the realised state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .constants import (
    CHARGE_BUS_FACTOR,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    Q_DIS_MAX,
    Q_MAX,
    TOL_ENERGY_KWH,
)
from .forecast import Forecaster
from .planning import FeeMode, WindowForecast, WindowResult, solve_window
from .schemas import (
    AbsoluteRun,
    AbsoluteStep,
    CommittedBalances,
    DispatchEvent,
    PenaltyEvent,
)
from .settlement import revise_commitment
from .timeline import INTERVALS_PER_DAY, MINUTES_PER_DAY, Timeline


class InfeasibleInterval(RuntimeError):
    """No legal feedback action exists for this interval."""

    def __init__(self, abs_minute: int, message: str, context: dict[str, float]) -> None:
        self.abs_minute = abs_minute
        self.context = context
        super().__init__(f"[abs={abs_minute}] {message}；上下文：{context}")


# ==========================================================================
# 1. Per-interval feedback
# ==========================================================================


@dataclass(frozen=True)
class IntervalAction:
    """The executed action of one interval."""

    charge_kwh: float
    discharge_kwh: float
    emergency_kwh: float
    spill_kwh: float
    soc_end_kwh: float
    target_gap_kwh: float          # achieved |E_target - E_end|


def execute_interval(
    *,
    abs_minute: int,
    grid_committed_kwh: float,
    demand_kwh: float,
    pv_kwh: float,
    soc_now_kwh: float,
    soc_target_kwh: float,
    allow_spill: bool = True,
) -> IntervalAction:
    """One interval of causal feedback, in the memo's lexicographic order.

    ``grid_committed_kwh`` is what the frozen/effective plan delivers and it may
    not be changed here. ``spill_kwh`` is the disclosed surplus: power that was
    committed, cannot be rejected, and has nowhere to go because the battery is
    full. It is reported, never hidden.
    """
    imbalance = demand_kwh - grid_committed_kwh - pv_kwh
    # "want" is the energy the plan asks the battery to absorb this interval.
    want = soc_target_kwh - soc_now_kwh

    headroom = max(0.0, (E_MAX - soc_now_kwh) * CHARGE_BUS_FACTOR)
    charge_cap = min(Q_MAX, headroom)
    discharge_cap = min(Q_DIS_MAX, max(0.0, (soc_now_kwh - E_MIN) / DISCHARGE_BATTERY_FACTOR))

    charge = discharge = emergency = spill = 0.0

    if imbalance <= 0.0 and want >= 0.0:
        # Surplus available and the plan asks to store: 1 means "use it all".
        charge = min(want, charge_cap)
    elif imbalance > 0.0 and want <= 0.0:
        # Deficit and the plan asks to release: 1 means "stop at the target".
        discharge = min(-want, discharge_cap)
    # Mixed cases (surplus but the plan wants to discharge; deficit but the plan
    # wants to charge) are physically impossible actions, so the battery does
    # nothing and the residual is settled below.

    residual = imbalance - discharge + charge * CHARGE_BUS_FACTOR
    if residual > TOL_ENERGY_KWH:
        # Deficit: level 2 -> emergency purchase covers it (grid is committed).
        emergency = residual
    elif residual < -TOL_ENERGY_KWH:
        surplus = -residual
        if allow_spill:
            spill = surplus
        else:
            raise InfeasibleInterval(
                abs_minute,
                "计划购电过量且储能已满，没有合法消纳路径（不得拒收外购电）",
                {
                    "grid_kwh": float(grid_committed_kwh),
                    "demand_kwh": float(demand_kwh),
                    "pv_kwh": float(pv_kwh),
                    "soc_kwh": float(soc_now_kwh),
                    "surplus_kwh": float(surplus),
                },
            )

    soc_end = soc_now_kwh + charge - discharge * DISCHARGE_BATTERY_FACTOR
    return IntervalAction(
        charge_kwh=charge,
        discharge_kwh=discharge,
        emergency_kwh=emergency,
        spill_kwh=spill,
        soc_end_kwh=soc_end,
        target_gap_kwh=abs(soc_target_kwh - soc_end),
    )


# ==========================================================================
# 2. Rolling driver
# ==========================================================================


@dataclass
class Policy:
    """Static description of one branch's permissions."""

    name: str
    can_adjust_plan: bool                 # legal plan revisions inside the day
    use_published_pv: bool                # attachment-3 permission
    price_mode: str                       # "repeated" | "historical"
    adjust_nodes_abs: tuple[int, ...] = ()  # clock intervals 36/72/108

    @property
    def is_adjustment_branch(self) -> bool:
        return self.can_adjust_plan


@dataclass
class RunOptions:
    """Numeric options of a run (all technical approximations, item A1)."""

    absorption_safety_kwh: float = 0.0
    allow_spill: bool = True
    plan_every_interval: bool = True
    plan_refresh_intervals: int = 1
    min_charge_floor: bool = True
    max_infeasible_intervals: int = 0
    solver_name: str = "appsi_highs"


@dataclass
class AbsoluteRunResult:
    """Everything a run produces, on the absolute timeline."""

    policy: str
    run: AbsoluteRun
    window_solves: int = 0
    revisions: int = 0
    notes: list[str] = field(default_factory=list)
    infeasible_abs: list[int] = field(default_factory=list)


def run_absolute(
    *,
    timeline: Timeline,
    forecaster: Forecaster,
    policy: Policy,
    options: RunOptions,
    abs_from: int,
    abs_to: int,
    soc_start_kwh: float,
    current_price_lookup=None,
) -> AbsoluteRunResult:
    """Execute the absolute timeline over [abs_from, abs_to).

    ``current_price_lookup(abs_minute) -> float | None`` returns the actual
    price only when that price is *already revealed* at decision time; it is
    used for the current-adjustment and current-quote terms. The realised price
    used for bookkeeping always comes from the timeline.
    """
    notes: list[str] = []
    steps: list[AbsoluteStep] = []
    events: list[DispatchEvent] = []
    penalties: list[PenaltyEvent] = []
    soc_boundary = [float(soc_start_kwh)]
    n_windows = 0
    n_revisions = 0
    infeasible: list[int] = []

    soc_now = float(soc_start_kwh)
    commitment: CommittedBalances | None = None
    last_result: WindowResult | None = None
    last_solve_abs: int | None = None

    abs_minute = abs_from
    while abs_minute < abs_to:
        day_index = abs_minute // MINUTES_PER_DAY
        b_abs = (day_index + 1) * MINUTES_PER_DAY
        T_abs = b_abs + MINUTES_PER_DAY
        horizon_end = max(min(T_abs, abs_to), abs_minute + 10)
        is_day_start = abs_minute % MINUTES_PER_DAY == 0
        at_node = _is_adjust_node(abs_minute, policy)
        need_solve = (
            last_result is None
            or last_solve_abs is None
            or is_day_start
            or (policy.can_adjust_plan and at_node)
            or (abs_minute - last_solve_abs) >= max(1, options.plan_refresh_intervals) * 10
        )
        if not need_solve:
            # Reuse the previously solved schedule for this interval. The
            # execution feedback below still runs every ten minutes.
            shift = (abs_minute - last_solve_abs) // 10
            result = last_result
            target_soc = float(result.soc_boundary_kwh[shift + 1])
            grid_now = float(commitment.effective_kwh[0]) if commitment is not None and commitment.abs_minutes.size else 0.0
            action = execute_interval(
                abs_minute=abs_minute,
                grid_committed_kwh=grid_now,
                demand_kwh=float(timeline.demand_kwh.value_at(abs_minute)),
                pv_kwh=float(timeline.pv_kwh.value_at(abs_minute)),
                soc_now_kwh=soc_now,
                soc_target_kwh=target_soc,
                allow_spill=options.allow_spill,
            )
            o_exec, a_exec = commitment.take(abs_minute) if commitment is not None else (0.0, 0.0)
            steps.append(
                AbsoluteStep(
                    abs_minute=abs_minute,
                    grid_kwh=o_exec + a_exec,
                    emergency_kwh=action.emergency_kwh,
                    charge_kwh=action.charge_kwh,
                    discharge_kwh=action.discharge_kwh,
                    curtail_kwh=_pv_curtailment(
                        demand_kwh=float(timeline.demand_kwh.value_at(abs_minute)),
                        pv_kwh=float(timeline.pv_kwh.value_at(abs_minute)),
                        grid_kwh=o_exec + a_exec,
                        charge_kwh=action.charge_kwh,
                        discharge_kwh=action.discharge_kwh,
                    ),
                    surplus_kwh=action.spill_kwh,
                    soc_end_kwh=action.soc_end_kwh,
                    price_actual_yuan_per_kwh=float(
                        timeline.price_yuan_per_kwh.value_at(abs_minute)
                    ),
                )
            )
            events.append(
                DispatchEvent(
                    at_abs=abs_minute,
                    o_exec_kwh=o_exec,
                    a_exec_kwh=a_exec,
                    emergency_kwh=action.emergency_kwh,
                    price_actual_yuan_per_kwh=float(
                        timeline.price_yuan_per_kwh.value_at(abs_minute)
                    ),
                )
            )
            soc_now = action.soc_end_kwh
            soc_boundary.append(soc_now)
            abs_minute += 10
            continue

        current_price = None
        if current_price_lookup is not None:
            current_price = current_price_lookup(abs_minute)

        forecast: WindowForecast = forecaster.window_forecast(
            abs_minute,
            abs_minute,
            horizon_end,
            use_published_pv=policy.use_published_pv,
            current_price=current_price,
            price_mode=policy.price_mode,
        )

        n = forecast.n
        minutes = forecast.abs_minutes

        # Carry-over: the previous natural day's plan still covers every interval
        # of the new window that belongs to that day (in particular the next
        # day's first ten minutes, which a result row reaches into).
        o_arr = np.zeros(n, dtype=np.float64)
        a_arr = np.zeros(n, dtype=np.float64)
        mask = np.zeros(n, dtype=bool)
        fixed_values = np.zeros(n, dtype=np.float64)
        if commitment is not None:
            pos = np.searchsorted(minutes, commitment.abs_minutes)
            valid = (pos < n) & (minutes[np.clip(pos, 0, n - 1)] == commitment.abs_minutes)
            idx = pos[valid]
            o_arr[idx] = commitment.o_kwh[valid]
            a_arr[idx] = commitment.a_kwh[valid]
            mask[idx] = True
            fixed_values[idx] = commitment.o_kwh[valid] + commitment.a_kwh[valid]

        can_revise = policy.can_adjust_plan and at_node and not is_day_start
        if mask.any() and not is_day_start:
            # Already covered by a previous commitment: only the intervals the
            # revision may touch are freed (and only on adjustment branches).
            if can_revise:
                o_arr = np.where(mask, o_arr, 0.0)
                a_arr = np.where(mask, a_arr, 0.0)
                mask = np.zeros(n, dtype=bool)
                fee_mode = FeeMode.ADJUSTABLE
            else:
                fee_mode = FeeMode.FROZEN
        elif mask.any() and is_day_start:
            # Start of a new natural day: form today's plan afresh. The whole
            # solved window is committed below, so the next day boundary is
            # covered without another solve.
            fee_mode = FeeMode.FIRST_PLAN
            o_arr = np.zeros(n, dtype=np.float64)
            a_arr = np.zeros(n, dtype=np.float64)
            mask = np.zeros(n, dtype=bool)
            fixed_values = np.zeros(n, dtype=np.float64)
        else:
            fee_mode = FeeMode.FIRST_PLAN

        absorption = forecaster.absorption_upper_kwh(
            abs_minute,
            minutes,
            safety_kwh=options.absorption_safety_kwh,
        )

        result: WindowResult = solve_window(
            forecast=forecast,
            soc_start_kwh=soc_now,
            fee_mode=fee_mode,
            o_kwh=o_arr,
            a_kwh=a_arr,
            price_now_yuan_per_kwh=current_price,
            absorption_upper_kwh=absorption,
            committed_grid_kwh=fixed_values if mask.any() else None,
            committed_mask=mask if mask.any() else None,
            solver_name=options.solver_name,
            problem_name=f"{policy.name}@{abs_minute}",
        )
        if not result.report.feasible and mask.any():
            # The carried-over forward commitment turned out to be physically
            # impossible given the realised state of charge (a large forecast
            # deviation). Only our own natural day is binding; release the
            # next-day part and retry rather than declaring the day infeasible.
            today_only = mask & (minutes < _natural_day_end(abs_minute))
            if today_only.any() and not np.array_equal(today_only, mask):
                result = solve_window(
                    forecast=forecast,
                    soc_start_kwh=soc_now,
                    fee_mode=fee_mode,
                    o_kwh=np.where(today_only, o_arr, 0.0),
                    a_kwh=np.where(today_only, a_arr, 0.0),
                    price_now_yuan_per_kwh=current_price,
                    absorption_upper_kwh=absorption,
                    committed_grid_kwh=fixed_values,
                    committed_mask=today_only,
                    solver_name=options.solver_name,
                    problem_name=f"{policy.name}@{abs_minute}-relaxed",
                )
                if result.report.feasible:
                    mask = today_only
                    o_arr = np.where(today_only, o_arr, 0.0)
                    a_arr = np.where(today_only, a_arr, 0.0)
                    notes.append(
                        f"区间 {abs_minute}：次日前瞻承诺在实测 SOC 下不可行，"
                        "已释放次日部分并重解（仅本自然日计划保持绑定）"
                    )
        n_windows += 1
        if not result.report.feasible:
            infeasible.append(abs_minute)
            if len(infeasible) > options.max_infeasible_intervals:
                raise RuntimeError(
                    f"{policy.name}: 区间 {abs_minute} 窗口 LP 不可行"
                    f"（{result.report.status}/{result.report.termination}）"
                )
            abs_minute += 10
            continue
        last_result = result
        last_solve_abs = abs_minute

        if can_revise:
            # A revision may only touch this natural day's remaining plan; the
            # part already committed for the next day is left untouched, which
            # is why the O/A state is compared only on the freed intervals.
            new_y = np.zeros(commitment.abs_minutes.size, dtype=np.float64)
            pos = np.searchsorted(minutes, commitment.abs_minutes)
            inside = pos < n
            new_y[~inside] = commitment.effective_kwh[~inside]
            if inside.any():
                new_y[inside] = np.where(
                    minutes[pos[inside]] <= _natural_day_end(abs_minute),
                    result.grid_kwh[pos[inside]],
                    commitment.effective_kwh[inside],
                )
            new_state, penalty = revise_commitment(
                commitment,
                new_y,
                at_abs=abs_minute,
                price_at_yuan_per_kwh=float(current_price if current_price is not None else 0.0),
            )
            if penalty is not None:
                penalties.append(penalty)
                n_revisions += 1
            commitment = new_state
        elif fee_mode == FeeMode.FIRST_PLAN:
            # The 00:00 plan covers the whole solved window: today's remaining
            # intervals become O, the next-day lookahead becomes A. Committing
            # both is what lets the next day boundary carry over without another
            # re-optimisation, and the next 00:00 simply replaces this plan.
            boundary = _natural_day_end(abs_minute)
            o_new = np.where(minutes < boundary, result.grid_kwh, 0.0)
            a_new = np.where(minutes >= boundary, result.grid_kwh, 0.0)
            commitment = CommittedBalances(minutes.copy(), o_new, a_new)
            n_revisions += 1

        # ---- execute the current interval ------------------------------------
        grid_now = float(commitment.effective_kwh[0]) if commitment.abs_minutes.size else 0.0
        demand_now = float(timeline.demand_kwh.value_at(abs_minute))
        pv_now = float(timeline.pv_kwh.value_at(abs_minute))
        price_now_actual = float(timeline.price_yuan_per_kwh.value_at(abs_minute))
        target_soc = float(result.soc_boundary_kwh[1])

        action = execute_interval(
            abs_minute=abs_minute,
            grid_committed_kwh=grid_now,
            demand_kwh=demand_now,
            pv_kwh=pv_now,
            soc_now_kwh=soc_now,
            soc_target_kwh=target_soc,
            allow_spill=options.allow_spill,
        )

        o_exec, a_exec = commitment.take(abs_minute)
        curtail_now = _pv_curtailment(
            demand_kwh=demand_now,
            pv_kwh=pv_now,
            grid_kwh=o_exec + a_exec,
            charge_kwh=action.charge_kwh,
            discharge_kwh=action.discharge_kwh,
        )
        steps.append(
            AbsoluteStep(
                abs_minute=abs_minute,
                grid_kwh=o_exec + a_exec,
                emergency_kwh=action.emergency_kwh,
                charge_kwh=action.charge_kwh,
                discharge_kwh=action.discharge_kwh,
                curtail_kwh=curtail_now,
                surplus_kwh=action.spill_kwh,
                soc_end_kwh=action.soc_end_kwh,
                price_actual_yuan_per_kwh=price_now_actual,
            )
        )
        events.append(
            DispatchEvent(
                at_abs=abs_minute,
                o_exec_kwh=o_exec,
                a_exec_kwh=a_exec,
                emergency_kwh=action.emergency_kwh,
                price_actual_yuan_per_kwh=price_now_actual,
            )
        )
        soc_now = action.soc_end_kwh
        soc_boundary.append(soc_now)
        abs_minute += 10

    notes.extend(timeline.bridge_notes())
    notes.append(f"窗口求解 {n_windows} 次，计划成文/修订 {n_revisions} 次")
    if penalties:
        notes.append(
            f"违约事件 {len(penalties)} 次，合计 {sum(p.penalty_yuan for p in penalties):.4f} 元"
        )
    steps.sort(key=lambda s: s.abs_minute)
    return AbsoluteRunResult(
        policy=policy.name,
        run=AbsoluteRun(
            abs_minutes=np.array([s.abs_minute for s in steps], dtype=np.int64),
            steps=steps,
            soc_boundary_kwh=np.asarray(soc_boundary, dtype=np.float64),
            events=events,
            penalties=penalties,
        ),
        window_solves=n_windows,
        revisions=n_revisions,
        notes=notes,
        infeasible_abs=infeasible,
    )

    notes.extend(timeline.bridge_notes())
    notes.append(f"窗口求解 {n_windows} 次，计划修订/首次成文 {n_revisions} 次")
    if penalties:
        notes.append(
            f"违约事件 {len(penalties)} 次，合计 {sum(p.penalty_yuan for p in penalties):.4f} 元"
        )
    steps.sort(key=lambda s: s.abs_minute)
    return AbsoluteRunResult(
        policy=policy.name,
        run=AbsoluteRun(
            abs_minutes=np.array([s.abs_minute for s in steps], dtype=np.int64),
            steps=steps,
            soc_boundary_kwh=np.asarray(soc_boundary, dtype=np.float64),
            events=events,
            penalties=penalties,
        ),
        window_solves=n_windows,
        revisions=n_revisions,
        notes=notes,
        infeasible_abs=infeasible,
    )


def _is_adjust_node(abs_minute: int, policy: Policy) -> bool:
    """True when this interval starts an allowed adjustment node (06/12/18)."""
    clock = (abs_minute % MINUTES_PER_DAY) // 10
    return clock in policy.adjust_nodes_abs


def _natural_day_end(abs_minute: int) -> int:
    """Absolute minute of the next natural day 00:00 (exclusive day end)."""
    return (abs_minute // MINUTES_PER_DAY + 1) * MINUTES_PER_DAY


def _pv_curtailment(
    *,
    demand_kwh: float,
    pv_kwh: float,
    grid_kwh: float,
    charge_kwh: float,
    discharge_kwh: float,
) -> float:
    """Surplus PV that the load, the battery and the committed purchase cannot take.

    Curtailment applies to PV only (never to purchased power), so it is bounded
    by the available PV itself.
    """
    absorbable = (
        demand_kwh
        - grid_kwh
        - discharge_kwh
        + charge_kwh * CHARGE_BUS_FACTOR
    )
    surplus = pv_kwh - max(absorbable, 0.0)
    return float(min(max(surplus, 0.0), pv_kwh))


__all__ = [
    "InfeasibleInterval",
    "IntervalAction",
    "execute_interval",
    "Policy",
    "RunOptions",
    "AbsoluteRunResult",
    "run_absolute",
]
