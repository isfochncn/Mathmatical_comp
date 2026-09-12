"""Execute only legal commitments, with explicit charge/discharge feedback."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog

from .constants import E_MIN, E_MAX, Q_MAX, Q_DIS_MAX, ETA_CHARGE, ETA_DISCHARGE
from .constants import ROLLING_START_ABS_MINUTE
from .forecast import Forecaster
from .planning import FeeMode, solve_window
from .schemas import AbsoluteRun, AbsoluteStep, CommittedBalances, DispatchEvent
from .settlement import revise_commitment
from .timeline import Timeline, MINUTES_PER_DAY


class InfeasibleInterval(RuntimeError):
    def __init__(self, abs_minute: int, message: str, context: dict[str, float]) -> None:
        self.abs_minute, self.context = abs_minute, context
        super().__init__(f"[abs={abs_minute}] {message}; {context}")


@dataclass(frozen=True)
class IntervalAction:
    charge_kwh: float
    discharge_kwh: float
    emergency_kwh: float
    spill_kwh: float
    soc_end_kwh: float
    target_gap_kwh: float
    curtail_kwh: float = 0.0


def execute_interval(*, abs_minute: int, grid_committed_kwh: float,
                     demand_kwh: float, pv_kwh: float, soc_now_kwh: float,
                     charge_target_kwh: float | None = None,
                     discharge_target_kwh: float | None = None,
                     soc_target_kwh: float | None = None,
                     allow_spill: bool = True) -> IntervalAction:
    """Revised feedback: disposal, emergency, target deviations, then throughput.

    Variables are stored charge, delivered discharge, emergency, PV curtailment,
    paid disposal and the two absolute deviations. Disposal never cancels a bill.
    The SOC-only argument is retained for callers; the rolling driver passes both
    targets, preserving a simultaneous charge/discharge schedule.
    """
    values = [grid_committed_kwh, demand_kwh, pv_kwh, soc_now_kwh]
    if not np.isfinite(values).all() or min(values[:3]) < 0 or not E_MIN <= soc_now_kwh <= E_MAX:
        raise ValueError("Invalid actual interval inputs")
    if charge_target_kwh is None and discharge_target_kwh is None and soc_target_kwh is not None:
        delta = soc_target_kwh - soc_now_kwh
        charge_target_kwh = max(delta, 0)
        discharge_target_kwh = max(-delta, 0) * ETA_DISCHARGE
    if charge_target_kwh is None or discharge_target_kwh is None:
        raise ValueError("Both dispatch targets are required")
    if not np.isfinite([charge_target_kwh, discharge_target_kwh]).all() or min(charge_target_kwh, discharge_target_kwh) < -1e-6:
        raise ValueError("Invalid dispatch targets")
    tc, td = max(0, charge_target_kwh), max(0, discharge_target_kwh)
    # ch, dis, emergency, PV curtailment, paid disposal, charge/discharge deviations.
    eq = [[1 / ETA_CHARGE, -1, -1, 1, 1, 0, 0]]
    rhs = [grid_committed_kwh + pv_kwh - demand_kwh]
    aub = [[1, -1 / ETA_DISCHARGE, 0, 0, 0, 0, 0],
           [-1, 1 / ETA_DISCHARGE, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, -1, 0], [-1, 0, 0, 0, 0, -1, 0],
           [0, 1, 0, 0, 0, 0, -1], [0, -1, 0, 0, 0, 0, -1]]
    bub = [E_MAX - soc_now_kwh, soc_now_kwh - E_MIN, tc, -tc, td, -td]
    max_dis = min(Q_DIS_MAX, max(demand_kwh-grid_committed_kwh, 0)) if allow_spill else Q_DIS_MAX
    bounds = [(0, Q_MAX), (0, max_dis), (0, None), (0, pv_kwh),
              (0, grid_committed_kwh if allow_spill else 0), (0, None), (0, None)]
    objectives = ([[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0]] if allow_spill else
                  [[0, 0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]])
    for objective in objectives:
        result = linprog(objective, A_ub=aub, b_ub=bub, A_eq=eq, b_eq=rhs,
                         bounds=bounds, method="highs")
        if not result.success:
            raise InfeasibleInterval(abs_minute, "No legal feedback action",
                                     dict(grid_kwh=grid_committed_kwh, demand_kwh=demand_kwh,
                                          pv_kwh=pv_kwh, soc_kwh=soc_now_kwh))
        aub.append(objective)
        bub.append(float(result.fun) + 1e-8)
    ch, dis, emergency, curtail, spill = np.where(np.abs(result.x[:5]) < 1e-7, 0., result.x[:5])
    end = soc_now_kwh + ch - dis / ETA_DISCHARGE
    if not E_MIN - 1e-6 <= end <= E_MAX + 1e-6:
        raise RuntimeError("Feedback SOC validation failed")
    end = float(np.clip(end, E_MIN, E_MAX))
    return IntervalAction(float(ch), float(dis), float(emergency), float(spill), end,
                          float(abs(ch - tc) + abs(dis - td)), float(curtail))


@dataclass(frozen=True)
class Policy:
    name: str
    can_adjust_plan: bool
    use_published_pv: bool
    price_mode: str
    adjust_nodes_abs: tuple[int, ...] = ()

    @property
    def is_adjustment_branch(self) -> bool:
        return self.can_adjust_plan


@dataclass
class RunOptions:
    absorption_safety_kwh: float = 0.0
    commitment_floor_kwh: float = 0.0
    allow_spill: bool = True
    plan_refresh_intervals: int = 1
    progress_every_days: int = 0
    total_days: int = 0
    max_infeasible_intervals: int = 0
    solver_name: str = "appsi_highs"
    experiment: str = "main"
    evaluation_restart: bool = False  # paired subperiod tests only, never a main annual run


@dataclass
class AbsoluteRunResult:
    policy: str
    run: AbsoluteRun
    window_solves: int = 0
    revisions: int = 0
    notes: list[str] = field(default_factory=list)
    infeasible_abs: list[int] = field(default_factory=list)
    released_abs: list[int] = field(default_factory=list)
    forecast_audits: list[dict] = field(default_factory=list)


def _natural_day_end(abs_minute: int) -> int:
    return (abs_minute // MINUTES_PER_DAY + 1) * MINUTES_PER_DAY


def _is_adjust_node(abs_minute: int, policy: Policy) -> bool:
    return (abs_minute % MINUTES_PER_DAY) // 10 in policy.adjust_nodes_abs


def run_absolute(*, timeline: Timeline, forecaster: Forecaster, policy: Policy,
                 options: RunOptions, abs_from: int, abs_to: int,
                 soc_start_kwh: float, current_price_lookup=None) -> AbsoluteRunResult:
    valid_start = abs_from == ROLLING_START_ABS_MINUTE or (
        options.evaluation_restart and options.experiment != 'main' and abs_from > 0 and abs_from%1440 == 0)
    if abs_to <= abs_from or not valid_start or abs_to % 10:
        raise ValueError("Rolling run must start at January 1 00:10 and contain complete intervals")
    if options.evaluation_restart and options.experiment == 'main':
        raise ValueError('A main run cannot restart from an evaluation checkpoint')
    if options.max_infeasible_intervals:
        raise ValueError("Failed intervals cannot be skipped")
    if options.experiment == "main" and (
        not options.allow_spill or options.plan_refresh_intervals != 1 or options.absorption_safety_kwh or options.commitment_floor_kwh
    ):
        raise ValueError("Main model requires every-interval replanning without purchase guards")
    if options.plan_refresh_intervals < 1:
        raise ValueError("Refresh cadence must be positive")
    steps, events, penalties, audits = [], [], [], []
    initial_plans = {}
    socs = [float(soc_start_kwh)]
    commitment = None
    result = None
    solved_at = None
    n_solves = revisions = 0
    repeated_price = np.roll(forecaster.bundle.attachment1.price_yuan_per_kwh, 1)
    for now in range(abs_from, abs_to, 10):
        b = _natural_day_end(now)
        is_start = now == abs_from or now % 1440 == 0
        can_revise = policy.can_adjust_plan and _is_adjust_node(now, policy) and not is_start
        needs_solve = (result is None or is_start or can_revise
                       or now - solved_at >= options.plan_refresh_intervals * 10)
        quote = current_price_lookup(now) if current_price_lookup is not None else None
        pending = None
        if needs_solve:
            forecast = forecaster.window_forecast(now, now, b + 1440,
                use_published_pv=policy.use_published_pv, current_price=quote, price_mode=policy.price_mode)
            risk = {}
            if hasattr(forecaster, 'risk_requirements'):
                risk = forecaster.risk_requirements(now, forecast,
                    published=policy.use_published_pv, adjustable=policy.can_adjust_plan)
                forecast.reserve_energy_kwh = risk.get('reserve_energy_kwh')
                forecast.net_upper_kwh = risk.get('net_upper_kwh')
            today = forecast.abs_minutes < b
            o, a = np.zeros(forecast.n), np.zeros(forecast.n)
            if is_start:
                if commitment is not None and commitment.abs_minutes.size:
                    raise RuntimeError("Unexecuted prior-day commitments at midnight")
                mode = FeeMode.FIRST_PLAN
            else:
                if commitment is None or not np.array_equal(commitment.abs_minutes, forecast.abs_minutes[today]):
                    raise RuntimeError("Commitment coverage gap")
                o[today], a[today] = commitment.o_kwh, commitment.a_kwh
                mode = FeeMode.ADJUSTABLE if can_revise else FeeMode.FROZEN
            planning_price = float(quote) if quote is not None else float(forecast.price_yuan_per_kwh[0])
            extra = {}
            if options.experiment != "main":
                if options.absorption_safety_kwh:
                    extra["absorption_upper_kwh"] = forecaster.absorption_upper_kwh(
                        now, forecast.abs_minutes, safety_kwh=options.absorption_safety_kwh)
                if options.commitment_floor_kwh:
                    extra["commitment_lower_kwh"] = forecaster.commitment_floor_kwh(
                        now, forecast.abs_minutes, safety_kwh=options.commitment_floor_kwh)
            result = solve_window(forecast=forecast, soc_start_kwh=socs[-1], fee_mode=mode,
                o_kwh=o, a_kwh=a, price_now_yuan_per_kwh=planning_price,
                committed_mask=today if mode == FeeMode.FROZEN else None,
                committed_grid_kwh=o+a if mode == FeeMode.FROZEN else None,
                adjustable_mask=today if can_revise else None,
                allow_spill=options.allow_spill,
                solver_name=options.solver_name, problem_name=f"{policy.name}@{now}", **extra).require_ok()
            solved_at = now
            n_solves += 1
            audits.append(dict(formed_at=now, observed_end=now, window_end=b+1440,
                               source=str(forecast.provenance[0]), forecast_cost=result.objective_yuan,
                               forecast_spill_kwh=float(np.sum(getattr(result, "spill_kwh", 0))),
                               demand_forecast_kwh=float(forecast.demand_kwh[0]),
                               pv_forecast_kwh=float(forecast.pv_kwh[0]),
                               risk_sample_count=risk.get('risk_sample_count', 0),
                               reserve_required_kwh=(float(forecast.reserve_energy_kwh[0])
                                   if forecast.reserve_energy_kwh is not None else 0.),
                               reserve_shortfall_kwh=result.reserve_shortfall_kwh,
                               reserve_stock_shortfall_kwh=result.reserve_stock_shortfall_kwh,
                               reserve_power_shortfall_now_kwh=float(result.reserve_power_shortfall_kwh[0]),
                               risk_penalty_yuan=result.risk_penalty_yuan))
            if (is_start or can_revise) and risk:
                audits[-1]['reserve_path_kwh'] = forecast.reserve_energy_kwh.tolist()
                audits[-1]['net_upper_path_kwh'] = forecast.net_upper_kwh.tolist()
                audits[-1]['demand_forecast_path_kwh'] = forecast.demand_kwh.tolist()
                audits[-1]['pv_forecast_path_kwh'] = forecast.pv_kwh.tolist()
                audits[-1]['reserve_power_shortfall_path_kwh'] = result.reserve_power_shortfall_kwh.tolist()
            if is_start:
                commitment = CommittedBalances.from_initial_plan(
                    forecast.abs_minutes[today], result.grid_kwh[today])
                initial_plans.update(zip(map(int, commitment.abs_minutes), map(float, commitment.o_kwh)))
                revisions += 1
            elif can_revise:
                changed = not np.allclose(commitment.effective_kwh, result.grid_kwh[today], atol=1e-7, rtol=0)
                commitment, pending = revise_commitment(
                    commitment, result.grid_kwh[today], at_abs=now,
                    price_at_yuan_per_kwh=planning_price)
                revisions += int(changed)
        offset = (now - solved_at) // 10
        o_exec, a_exec = commitment.take(now)
        demand, pv = timeline.demand_kwh.value_at(now), timeline.pv_kwh.value_at(now)
        action = execute_interval(abs_minute=now, grid_committed_kwh=o_exec+a_exec,
            demand_kwh=demand, pv_kwh=pv, soc_now_kwh=socs[-1],
            charge_target_kwh=result.charge_kwh[offset], discharge_target_kwh=result.discharge_kwh[offset],
            allow_spill=options.allow_spill)
        # Execution/settlement may read the completed interval's actual values;
        # these are never passed back into the decision that was already made.
        actual_price = (float(repeated_price[now % 1440 // 10]) if policy.price_mode == "repeated"
                        else timeline.price_yuan_per_kwh.value_at(now))
        if not np.isfinite(actual_price) or actual_price < 0:
            raise ValueError(f"Missing/invalid execution price at {now}")
        if pending is not None:
            pending.price_at_yuan_per_kwh = actual_price
            penalties.append(pending)
        steps.append(AbsoluteStep(now, o_exec+a_exec, action.emergency_kwh,
            action.charge_kwh, action.discharge_kwh, action.curtail_kwh, action.spill_kwh,
            action.soc_end_kwh, actual_price))
        events.append(DispatchEvent(now, o_exec, a_exec, action.emergency_kwh, actual_price))
        socs.append(action.soc_end_kwh)
        if options.progress_every_days and (now+10) % (1440*options.progress_every_days) == 0:
            print(f"{policy.name}: {(now+10)//1440} days; {n_solves} solves", flush=True)
    run = AbsoluteRun(np.array([s.abs_minute for s in steps]), steps, np.array(socs),
                      events, penalties, initial_plans)
    return AbsoluteRunResult(policy.name, run, n_solves, revisions,
                             timeline.bridge_notes(), forecast_audits=audits)
