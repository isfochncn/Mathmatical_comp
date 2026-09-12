"""Orchestration: absolute timeline -> causal point forecast -> rolling LP ->
per-interval feedback -> bills -> validation -> export payloads.

This module replaces the old "natural day + 144-slot" driver. It knows which
problem is being solved (and therefore which information rights apply) and it is
the only place that decides the planning horizon, the adjust nodes and the
pricing mode. Everything numeric lives in the other modules.

NOTE: ASCII-only source on purpose, so Windows text pipelines can never mangle it.
Chinese documentation lives in the Markdown reports and in the other modules.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from . import physics
from .constants import (
    ADJUST_NODE_INTERVALS,
    DELTA_T_HOURS,
    E_INIT_2025_01_01,
    E_MAX,
    E_MIN,
    N_INTERVAL,
    OUTPUT_START,
)
from .constants import ROLLING_START_ABS_MINUTE
from .data_io import DataBundle, data_path, input_fingerprint, load_all, project_root
from .forecast import Forecaster
from .planning import FeeMode, WindowForecast, WindowResult, solve_window
from .schemas import AbsoluteRun, AbsoluteStep, DispatchEvent
from .settlement import (
    LabelTotals,
    natural_day_bill,
    result_row_bill,
    verify_bill,
)
from .simulation import Policy, RunOptions, run_absolute
from .timeaxis import (
    DATE_2025_01_01,
    DATE_2025_12_31,
    calendar_days,
    result_interval_clock_index,
)
from .timeline import INTERVALS_PER_DAY, MINUTES_PER_DAY, Timeline, build_timeline


# ==========================================================================
# Configuration
# ==========================================================================


@dataclass
class RunConfig:
    """Every tunable of one experiment (technical approximations included)."""

    problem: str = "problem2"
    out_dir: Path = field(default_factory=lambda: Path("out"))

    history_days: int = 28
    #: 2026-09-12: weekly load, calibrated PV and empirical joint-error reserve.
    load_method: str = "adaptive"
    # Auto upgrades problem three only; pooled retains the earlier PV calibration.
    pv_method: str = "auto"
    risk_quantile: float = .90
    absorption_safety_kwh: float = 0.0
    #: Lower bound added to the historical same-clock minimum net demand when
    #: forming a commitment. This is what keeps a frozen plan executable when the
    #: realised PV falls short; recorded under comparison item A1.
    commitment_floor_kwh: float = 0.0
    allow_spill: bool = True
    max_infeasible_intervals: int = 0
    solver_name: str = "appsi_highs"

    # Main model refreshes every ten minutes. Alternatives must be named experiments.
    plan_refresh_intervals: int = 1
    experiment: str = "main"

    # Reporting start and execution end; SOC always evolves from January 1.
    run_from: date | None = None
    run_to: date | None = None
    warmup_days: int = 0  # Deprecated: nonzero values are rejected.

    #: Optional progress hook: called as ``progress(days_done, days_total)``.
    #: A multi-hour run without observability is not acceptable, so the CLI
    #: passes a printer here.
    progress_every_days: int = 0

    def problem_dir(self) -> Path:
        return self.out_dir / self.problem if self.experiment == "main" else self.out_dir / self.experiment / self.problem

    def effective_pv_method(self) -> str:
        if self.pv_method == 'auto':
            return 'report_blend' if self.problem == 'problem3' and self.load_method == 'adaptive' else 'pooled'
        return self.pv_method

    def __post_init__(self) -> None:
        if self.warmup_days:
            raise ValueError("Warm-up is the actual trajectory from January 1, not a configurable reset")
        for name, value in (("run_from", self.run_from), ("run_to", self.run_to)):
            if value is not None and not date(2025, 1, 1) <= value <= date(2025, 12, 31):
                raise ValueError(f"{name} must be within 2025")
        if self.max_infeasible_intervals:
            raise ValueError("Invalid runs may not skip intervals")
        if self.experiment == "main" and not self.allow_spill and self.problem != "problem1":
            raise ValueError("Strict no-spill is a named comparison, not the revised main model")
        if self.load_method not in ('adaptive', 'same_clock_mean', 'same_weekday'):
            raise ValueError('Unknown forecast method')
        if self.pv_method not in ('auto','pooled','report_blend'):
            raise ValueError('Unknown PV forecast method')
        if self.effective_pv_method() == 'report_blend' and (self.problem != 'problem3' or self.load_method != 'adaptive'):
            raise ValueError('Report blend is currently enabled only for adaptive problem three')
        if not np.isfinite(self.risk_quantile) or not 0 <= self.risk_quantile < 1:
            raise ValueError('risk_quantile must be in [0, 1)')
        if self.load_method != 'adaptive' and self.risk_quantile:
            raise ValueError('Legacy forecast comparisons require risk_quantile=0')
        if self.experiment == "main" and (self.history_days != 28 or self.load_method != "adaptive"
                or self.risk_quantile != .90
                or self.plan_refresh_intervals != 1 or self.absorption_safety_kwh or self.commitment_floor_kwh):
            raise ValueError("Alternative assumptions need an explicit experiment name")


# ==========================================================================
# Policies
# ==========================================================================

_ADJUST_NODES = tuple(ADJUST_NODE_INTERVALS[1:])  # 36 / 72 / 108 -> 06/12/18

POLICIES: dict[str, Policy] = {
    "problem1": Policy("problem1", False, False, "repeated"),
    "problem2": Policy("problem2", False, False, "repeated"),
    "problem3": Policy("problem3", True, True, "repeated", _ADJUST_NODES),
    "problem4-2": Policy("problem4-2", False, True, "historical"),
    "problem4-3": Policy("problem4-3", True, True, "historical", _ADJUST_NODES),
}


# ==========================================================================
# Result containers
# ==========================================================================


@dataclass
class IntervalRecord:
    """Diagnostics for one executed interval."""

    abs_minute: int
    grid_kwh: float
    emergency_kwh: float
    charge_kwh: float
    discharge_kwh: float
    curtail_kwh: float
    surplus_kwh: float
    soc_end_kwh: float
    price_actual: float


@dataclass
class RunResult:
    problem: str
    config: RunConfig
    timeline: Timeline
    run: AbsoluteRun
    day_bills: list[tuple[date, LabelTotals]]
    row_bills: list[tuple[date, LabelTotals]]
    summary: dict[str, object]
    wall_seconds: float
    input_fingerprint: str
    notes: list[str] = field(default_factory=list)
    #: Series already expressed in result-row order, when the natural-day clocks
    #: cannot be used directly (problem 1). Empty for the rolling branch, where
    #: the exporter derives result rows from absolute minutes.
    export_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    forecast_audits: list[dict] = field(default_factory=list)


# ==========================================================================
# Timeline helpers
# ==========================================================================


def abs_minute_of_clock(day: date, t: int) -> int:
    """(date, clock interval) -> absolute minute since 2025-01-01 00:00."""
    return (day - DATE_2025_01_01).days * MINUTES_PER_DAY + t * 10


def abs_minute_of_result_cell(day: date, j: int) -> int:
    """Result-row cell j of ``day`` -> absolute minute of its interval start.

    ``j = 143`` maps to [next day 00:00, next day 00:10), i.e. it belongs to the
    next natural day.
    """
    return abs_minute_of_clock(day, result_interval_clock_index(j))


def natural_day_bounds(day: date) -> tuple[int, int]:
    base = (day - DATE_2025_01_01).days * MINUTES_PER_DAY
    return base, base + MINUTES_PER_DAY


def result_row_bounds(day: date) -> tuple[int, int]:
    base = (day - DATE_2025_01_01).days * MINUTES_PER_DAY
    return base + 10, base + MINUTES_PER_DAY + 10


# ==========================================================================
# Problem 1: single deterministic natural-day LP (unchanged physics, new axis)
# ==========================================================================


def run_problem1(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    """Problem 1: one typical day, E_0 = E_144 = 6000, result row via the cycle."""
    bundle = bundle or load_all()
    a1 = bundle.attachment1
    t0 = time.perf_counter()

    minutes = np.arange(0, MINUTES_PER_DAY, 10, dtype=np.int64)
    forecast = WindowForecast(
        abs_minutes=minutes,
        demand_kwh=np.roll(a1.demand_kwh, 1),
        pv_kwh=np.roll(a1.pv_kwh, 1),
        price_yuan_per_kwh=np.roll(a1.price_yuan_per_kwh, 1),
        provenance=np.array(["attachment1-typical-day"] * N_INTERVAL, dtype=object),
    )
    result: WindowResult = solve_window(
        forecast=forecast,
        soc_start_kwh=E_INIT_2025_01_01,
        fee_mode=FeeMode.FIRST_PLAN,
        soc_end_fixed_kwh=E_INIT_2025_01_01,
        solver_name=config.solver_name,
        problem_name="problem1",
        allow_emergency=False,
    ).require_ok()

    day = DATE_2025_01_01
    steps: list[AbsoluteStep] = []
    events: list[DispatchEvent] = []
    for t in range(N_INTERVAL):
        soc_end = float(result.soc_boundary_kwh[t + 1])
        steps.append(
            AbsoluteStep(
                abs_minute=int(minutes[t]),
                grid_kwh=float(result.grid_kwh[t]),
                emergency_kwh=0.0,
                charge_kwh=float(result.charge_kwh[t]),
                discharge_kwh=float(result.discharge_kwh[t]),
                curtail_kwh=float(result.curtail_kwh[t]),
                surplus_kwh=0.0,
                soc_end_kwh=soc_end,
                price_actual_yuan_per_kwh=float(forecast.price_yuan_per_kwh[t]),
            )
        )
        events.append(
            DispatchEvent(
                at_abs=int(minutes[t]),
                o_exec_kwh=float(result.grid_kwh[t]),
                a_exec_kwh=0.0,
                emergency_kwh=0.0,
                price_actual_yuan_per_kwh=float(forecast.price_yuan_per_kwh[t]),
            )
        )
    run = AbsoluteRun(
        abs_minutes=minutes,
        steps=steps,
        soc_boundary_kwh=np.asarray(result.soc_boundary_kwh, dtype=np.float64),
        events=events,
        penalties=[],
        initial_plans=dict(zip(map(int, minutes), map(float, result.grid_kwh))),
    )
    totals = natural_day_bill(0, MINUTES_PER_DAY, events, [])
    verify_bill(totals, events, [])

    # A result row covers [day 00:10, next day 00:10) while attachment 1 gives the
    # natural-day clock. For the repeated typical day the row is therefore the
    # clock series shifted by one interval, with the next day's first interval
    # (identical to this day's first interval) appended at the end.
    row_grid = np.concatenate([result.grid_kwh[1:], result.grid_kwh[:1]])
    row_charge = np.concatenate([result.charge_kwh[1:], result.charge_kwh[:1]])
    row_discharge = np.concatenate([result.discharge_kwh[1:], result.discharge_kwh[:1]])
    row_curtail = np.concatenate([result.curtail_kwh[1:], result.curtail_kwh[:1]])

    summary = _p1_summary(run, result, totals, forecast)
    return RunResult(
        problem="problem1",
        config=config,
        timeline=build_timeline(bundle),
        run=run,
        day_bills=[(day, totals)],
        row_bills=[(day, totals)],
        summary=summary,
        wall_seconds=time.perf_counter() - t0,
        input_fingerprint=input_fingerprint(_data_paths()),
        notes=["问题一为单日确定性 LP：E_0 = E_144 = 6000，结果行末段由周期延伸映射"],
        export_arrays={
            "result_grid_kwh": row_grid,
            "result_charge_kwh": row_charge,
            "result_discharge_kwh": row_discharge,
            "result_curtail_kwh": row_curtail,
            "soc_boundary_kwh": np.asarray(result.soc_boundary_kwh, dtype=np.float64),
            "demand_kwh": forecast.demand_kwh.copy(),
            "pv_kwh": forecast.pv_kwh.copy(),
        },
    )


def _p1_summary(run: AbsoluteRun, result: WindowResult, totals: LabelTotals, forecast: WindowForecast) -> dict[str, object]:
    arrays = run.arrays()
    charge = arrays["charge_kwh"]
    discharge = arrays["discharge_kwh"]
    loss = physics.loss_accounting(charge, discharge)
    soc = arrays["soc_kwh"]
    return {
        "problem": "problem1",
        "n_intervals": int(arrays["abs_minute"].size),
        "plan_kwh": totals.total_kwh,
        "add_kwh": 0.0,
        "emergency_kwh": 0.0,
        "plan_cost_yuan": totals.plan_cost_yuan,
        "add_cost_yuan": 0.0,
        "emergency_cost_yuan": 0.0,
        "execution_cost_yuan": totals.execution_cost_yuan,
        "reduce_cost_yuan": 0.0,
        "surplus_disposed_kwh": 0.0,
        "window_solves": 1,
        "plan_revisions": 1,
        "infeasible_intervals": [],
        "total_purchased_kwh": totals.total_kwh,
        "total_cost_yuan": totals.total_cost_yuan,
        "objective_yuan": result.objective_yuan,
        "soc_start_kwh": float(soc[0]),
        "soc_end_kwh": float(soc[-1]),
        "soc_min_kwh": float(soc.min()),
        "soc_max_kwh": float(soc.max()),
        "charge_total_kwh": float(np.sum(charge)),
        "discharge_total_kwh": float(np.sum(discharge)),
        "curtail_total_kwh": float(np.sum(arrays["curtail_kwh"])),
        "total_loss_kwh": loss["total_loss_kwh"],
        "n_simultaneous_intervals": loss["n_simultaneous_intervals"],
        "day_balance_residual_kwh": float(
            physics.daily_balance_residual_kwh(
                arrays["grid_kwh"],
                arrays["emergency_kwh"],
                forecast.demand_kwh,
                forecast.pv_kwh,
                arrays["curtail_kwh"],
                charge,
                discharge,
                soc,
            )
        ),
    }


# ==========================================================================
# Problems 2/3/4-2/4-3: absolute rolling run
# ==========================================================================


def make_forecaster(config: RunConfig, bundle: DataBundle, timeline: Timeline):
    if config.load_method == 'adaptive':
        if config.effective_pv_method() == 'report_blend':
            from .report_forecast import ReportAwareForecaster
            return ReportAwareForecaster(bundle,timeline,history_days=config.history_days,
                                         risk_quantile=config.risk_quantile,variant='blend',nowcast=True)
        from .adaptive_forecast import AdaptiveForecaster
        return AdaptiveForecaster(bundle,timeline,history_days=config.history_days,risk_quantile=config.risk_quantile)
    return Forecaster(bundle,timeline,history_days=config.history_days,load_method=config.load_method)


def run_rolling(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    if config.problem not in POLICIES or config.problem == "problem1":
        raise ValueError(f"run_rolling does not support {config.problem}")
    bundle = bundle or load_all()
    policy = POLICIES[config.problem]
    timeline = build_timeline(bundle)
    forecaster = make_forecaster(config,bundle,timeline)

    days = calendar_days(DATE_2025_01_01, DATE_2025_12_31)
    if config.run_to is not None:
        days = [d for d in days if d <= config.run_to]
    if not days or (config.run_from and config.run_from > days[-1]):
        raise ValueError("运行区间为空")
    # Always execute from Jan 1 so a later reporting start never resets SOC.
    abs_from = ROLLING_START_ABS_MINUTE
    abs_to = (days[-1] - DATE_2025_01_01).days * MINUTES_PER_DAY + MINUTES_PER_DAY + 10

    def price_lookup(abs_minute: int) -> float | None:
        if policy.price_mode == "repeated":
            clock = (abs_minute % MINUTES_PER_DAY) // 10
            return float(np.roll(bundle.attachment1.price_yuan_per_kwh, 1)[clock])
        # Attachment 4 contains completed-interval average prices, not a current quote.
        return None

    t0 = time.perf_counter()
    outcome = run_absolute(
        timeline=timeline,
        forecaster=forecaster,
        policy=policy,
        options=RunOptions(
            absorption_safety_kwh=config.absorption_safety_kwh,
            commitment_floor_kwh=config.commitment_floor_kwh,
            allow_spill=config.allow_spill,
            plan_refresh_intervals=config.plan_refresh_intervals,
            progress_every_days=config.progress_every_days,
            total_days=len(days),
            max_infeasible_intervals=config.max_infeasible_intervals,
            solver_name=config.solver_name,
            experiment=config.experiment,
        ),
        abs_from=abs_from,
        abs_to=abs_to,
        soc_start_kwh=E_INIT_2025_01_01,
        current_price_lookup=price_lookup,
    )
    wall = time.perf_counter() - t0

    run = outcome.run
    day_bills: list[tuple[date, LabelTotals]] = []
    row_bills: list[tuple[date, LabelTotals]] = []
    for day in days:
        df, dt = natural_day_bounds(day)
        day_bills.append((day, natural_day_bill(df, dt, run.events, run.penalties)))
        rf, rt = result_row_bounds(day)
        row_bills.append((day, result_row_bill(rf, rt, run.events, run.penalties)))

    # Always simulate from January 1; run_from changes reporting, not initialization.
    out_start = max(date(*OUTPUT_START), config.run_from or days[0])
    if days[-1] < date(*OUTPUT_START):
        out_start = config.run_from or days[0]  # explicitly short initialization smoke run
    output_day_bills = [(d, b) for d, b in day_bills if d >= out_start]
    output_row_bills = [(d, b) for d, b in row_bills if d >= out_start]

    # Independent recomputation over the whole run (warm-up included).
    all_events = run.events
    all_penalties = run.penalties
    grand = LabelTotals()
    for e in all_events:
        grand.plan_kwh += e.o_exec_kwh
        grand.add_kwh += e.a_exec_kwh
        grand.emergency_kwh += e.emergency_kwh
        grand.plan_cost_yuan += e.price_actual_yuan_per_kwh * e.o_exec_kwh
        grand.add_cost_yuan += 1.5 * e.price_actual_yuan_per_kwh * e.a_exec_kwh
        grand.emergency_cost_yuan += 5.0 * e.price_actual_yuan_per_kwh * e.emergency_kwh
    grand.reduce_cost_yuan = float(sum(p.penalty_yuan for p in all_penalties))
    verify_bill(grand, all_events, all_penalties)

    report_lo, report_hi = result_row_bounds(out_start)[0], result_row_bounds(days[-1])[1]
    out_events = [e for e in all_events if report_lo <= e.at_abs < report_hi]
    out_penalties = [p for p in all_penalties if report_lo <= p.at_abs < report_hi]
    output_totals = result_row_bill(report_lo, report_hi, out_events, out_penalties)
    verify_bill(output_totals, out_events, out_penalties)
    arrays = run.arrays()
    report_mask = (arrays["abs_minute"] >= report_lo) & (arrays["abs_minute"] < report_hi)
    start_index = int(np.searchsorted(arrays["abs_minute"], report_lo))
    out_minutes = arrays["abs_minute"][report_mask]
    out_soc = run.soc_boundary_kwh[start_index:]
    natural_start_index = int(np.searchsorted(arrays["abs_minute"], report_lo - 10))
    loss = physics.loss_accounting(arrays["charge_kwh"][report_mask], arrays["discharge_kwh"][report_mask])
    summary: dict[str, object] = {
        "problem": config.problem,
        "forecast_policy_version": ('report-hourly-blend-risk-v2' if config.effective_pv_method() == 'report_blend'
                                    else 'adaptive-risk-v1' if config.load_method == 'adaptive' else 'legacy-point-v1'),
        "forecast_method": config.load_method,
        "pv_forecast_method": config.effective_pv_method(),
        "risk_quantile": config.risk_quantile,
        "n_intervals": int(out_minutes.size),
        "warmup_intervals": int(natural_start_index),
        "execution_start_abs_minute": abs_from,
        "execution_end_abs_minute": abs_to,
        "output_soc_start_abs_minute": max(abs_from, report_lo - 10),
        "skipped_initial_intervals": 1,
        "total_purchased_kwh": output_totals.total_kwh,
        "plan_kwh": output_totals.plan_kwh,
        "add_kwh": output_totals.add_kwh,
        "emergency_kwh": output_totals.emergency_kwh,
        "plan_cost_yuan": output_totals.plan_cost_yuan,
        "add_cost_yuan": output_totals.add_cost_yuan,
        "emergency_cost_yuan": output_totals.emergency_cost_yuan,
        "execution_cost_yuan": output_totals.execution_cost_yuan,
        "reduce_cost_yuan": output_totals.reduce_cost_yuan,
        "total_cost_yuan": output_totals.total_cost_yuan,
        "warmup_total_cost_yuan": natural_day_bill(0, report_lo - 10, all_events, all_penalties).total_cost_yuan,
        "report_start_midnight_cost_yuan": natural_day_bill(report_lo - 10, report_lo, all_events, all_penalties).total_cost_yuan,
        "soc_at_output_start_kwh": float(run.soc_boundary_kwh[natural_start_index]),
        "soc_start_kwh": float(run.soc_boundary_kwh[0]),
        "soc_end_kwh": float(run.soc_boundary_kwh[-1]),
        "soc_min_kwh": float(run.soc_boundary_kwh.min()),
        "soc_max_kwh": float(run.soc_boundary_kwh.max()),
        "charge_total_kwh": float(np.sum(arrays["charge_kwh"][report_mask])),
        "discharge_total_kwh": float(np.sum(arrays["discharge_kwh"][report_mask])),
        "curtail_total_kwh": float(np.sum(arrays["curtail_kwh"][report_mask])),
        "surplus_disposed_kwh": float(np.sum(arrays["surplus_kwh"][report_mask])),
        "purchase_used_kwh": output_totals.total_kwh - float(np.sum(arrays["surplus_kwh"][report_mask])),
        "spill_intervals": int(np.sum(arrays["surplus_kwh"][report_mask] > 1e-6)),
        "spill_capacity_intervals": int(np.sum((arrays["surplus_kwh"][report_mask] > 1e-6) & (run.soc_boundary_kwh[1:][report_mask] >= 10800-1e-6))),
        "spill_power_intervals": int(np.sum((arrays["surplus_kwh"][report_mask] > 1e-6) & (arrays["charge_kwh"][report_mask] >= 750-1e-6))),
        "total_loss_kwh": loss["total_loss_kwh"],
        "n_simultaneous_intervals": loss["n_simultaneous_intervals"],
        "window_solves": outcome.window_solves,
        "plan_revisions": outcome.revisions,
        "infeasible_intervals": outcome.infeasible_abs,
        "output_days": len(output_day_bills),
        "output_row_purchase_kwh": output_totals.total_kwh,
        "output_natural_day_purchase_kwh": float(sum(b.total_kwh for _, b in output_day_bills)),
        "output_row_cost_yuan": output_totals.total_cost_yuan,
        "output_natural_day_cost_yuan": float(
            sum(b.total_cost_yuan for _, b in output_day_bills)
        ),
        "max_bus_charge_kw": float(np.max(arrays["charge_kwh"]) * 6 / 0.9),
        "max_bus_discharge_kw": float(np.max(arrays["discharge_kwh"]) * 6),
        "reserve_shortfall_windows": sum(a.get('reserve_shortfall_kwh', 0) > 1e-5 for a in outcome.forecast_audits),
        "reserve_shortfall_max_kwh": max(a.get('reserve_shortfall_kwh', 0) for a in outcome.forecast_audits),
        "uncalibrated_forecast_windows": sum(a.get('risk_sample_count', 0) < 7 for a in outcome.forecast_audits)
            if config.risk_quantile else 0,
    }
    # ``grid_kwh`` stores the delivered normal purchase O + A, which loses the
    # split the settlement needs: the plan fee applies to O at the normal rate
    # while the adjustment fee applies to A at 1.5x. Problem 3 / 4-3 must also
    # export an "adjustment" sheet, so the two components are kept separately
    # instead of being inferred from a total that cannot be decomposed.
    minutes_all = arrays["abs_minute"]
    plan_exec = np.zeros(minutes_all.size, dtype=np.float64)
    add_exec = np.zeros(minutes_all.size, dtype=np.float64)
    exec_index = {int(m): i for i, m in enumerate(minutes_all)}
    for e in all_events:
        i = exec_index.get(int(e.at_abs))
        if i is not None:
            plan_exec[i] = float(e.o_exec_kwh)
            add_exec[i] = float(e.a_exec_kwh)

    return RunResult(
        problem=config.problem,
        config=config,
        timeline=timeline,
        run=run,
        day_bills=day_bills,
        row_bills=output_row_bills,
        summary=summary,
        wall_seconds=wall,
        input_fingerprint=input_fingerprint(_data_paths()),
        notes=outcome.notes + ["Initial SOC 6000 kWh at Jan 1 00:10; preceding interval omitted."],
        export_arrays={"plan_exec_kwh": plan_exec, "add_exec_kwh": add_exec,
                       "plan_initial_kwh": np.array([run.initial_plans[int(m)] for m in minutes_all]),
                       "demand_kwh": np.array([timeline.demand_kwh.value_at(int(m)) for m in minutes_all]),
                       "pv_kwh": np.array([timeline.pv_kwh.value_at(int(m)) for m in minutes_all])},
        forecast_audits=outcome.forecast_audits,
    )


def validate_result(result: RunResult) -> None:
    from .validation import validate_absolute_run
    from .settlement import validate_ledger
    arrays = result.run.arrays()
    validate_absolute_run(abs_minutes=arrays["abs_minute"], grid_kwh=arrays["grid_kwh"],
        emergency_kwh=arrays["emergency_kwh"], charge_kwh=arrays["charge_kwh"],
        discharge_kwh=arrays["discharge_kwh"], curtail_kwh=arrays["curtail_kwh"],
        surplus_kwh=arrays["surplus_kwh"], soc_boundary_kwh=result.run.soc_boundary_kwh,
        demand_kwh=result.export_arrays["demand_kwh"], pv_kwh=result.export_arrays["pv_kwh"],
        soc_start_kwh=E_INIT_2025_01_01, require_daily_cycle=result.problem == "problem1",
        allow_spill=result.config.allow_spill and result.problem != "problem1")
    validate_ledger(abs_minutes=arrays["abs_minute"], grid_kwh=arrays["grid_kwh"],
        emergency_kwh=arrays["emergency_kwh"], price_actual=arrays["price_actual"],
        initial_plans=result.run.initial_plans, events=result.run.events,
        penalties=result.run.penalties, can_adjust=result.problem in ("problem3", "problem4-3"),
        expected_start_abs=0 if result.problem == "problem1" else ROLLING_START_ABS_MINUTE)
    if result.problem == "problem1" and np.any(arrays["emergency_kwh"] > 1e-6):
        raise ValueError("Problem 1 cannot use emergency purchases")
    expected_initial = np.array([result.run.initial_plans[int(t)] for t in arrays["abs_minute"]])
    for key, expected in (("plan_initial_kwh", expected_initial),
                          ("plan_exec_kwh", np.array([e.o_exec_kwh for e in result.run.events])),
                          ("add_exec_kwh", np.array([e.a_exec_kwh for e in result.run.events]))):
        if key in result.export_arrays and not np.allclose(result.export_arrays[key], expected, atol=1e-6, rtol=0):
            raise ValueError(f"Export array differs from ledger: {key}")
    for windows, bounds in ((result.day_bills, natural_day_bounds), (result.row_bills, result_row_bounds)):
        if result.problem == "problem1" and bounds is result_row_bounds:
            continue  # A single cyclic day is exported by a permutation of that day.
        for day, bill in windows:
            lo, hi = bounds(day)
            verify_bill(bill, [e for e in result.run.events if lo <= e.at_abs < hi],
                        [p for p in result.run.penalties if lo <= p.at_abs < hi])
    if result.problem != "problem1":
        for d, _ in result.row_bills:
            lo, hi = result_row_bounds(d)
            present = arrays["abs_minute"][(arrays["abs_minute"] >= lo) & (arrays["abs_minute"] < hi)]
            if not np.array_equal(present, np.arange(lo, hi, 10)):
                raise ValueError(f"Incomplete result row: {d}")
    if any(int(m) not in result.run.initial_plans for m in arrays["abs_minute"]):
        raise ValueError("Missing initial plan archive")


def run(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    result = run_problem1(config, bundle) if config.problem == "problem1" else run_rolling(config, bundle)
    validate_result(result)
    return result


def _json_default(obj):
    """JSON fallback for the summary payload (dates, numpy scalars, Paths)."""
    if isinstance(obj, (date,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return float(obj)


def _data_paths() -> list[Path]:
    root = project_root()
    return [
        data_path(p, root)
        for p in ("data/附件1.xlsx", "data/附件2.xlsx", "data/附件3.xlsx", "data/附件4.xlsx")
    ]


def save_run(result: RunResult) -> Path:
    validate_result(result)
    run_dir = result.config.problem_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(result.config).items()},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "validation_version": "main-model-v4-paid-spill",
                "initialization": ("attachment1 deterministic typical-day cycle" if result.problem == "problem1"
                                   else "SOC 6000 at 2025-01-01 00:10; 00:00-00:10 omitted"),
                "problem": result.problem,
                "wall_seconds": result.wall_seconds,
                "input_fingerprint": result.input_fingerprint,
                "summary": result.summary,
                "notes": result.notes,
                "policy": asdict(POLICIES[result.problem]) if result.problem in POLICIES else {},
                # Per-window bills are needed by the exporter: the natural-day bill
                # and the result-row bill cover different windows and must not be
                # conflated.
                "natural_day_bills": [
                    {"date": d.isoformat(),
                     "from_abs": max(natural_day_bounds(d)[0], int(result.run.abs_minutes[0])),
                     "to_abs": natural_day_bounds(d)[1],
                     "complete_natural_day": natural_day_bounds(d)[0] >= result.run.abs_minutes[0],
                     **b.as_dict()} for d, b in result.day_bills
                ],
                "result_row_bills": [
                    {"date": d.isoformat(), **b.as_dict()} for d, b in result.row_bills
                ],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    header = (
        "window,date,plan_kwh,add_kwh,emergency_kwh,total_kwh,"
        "plan_cost_yuan,add_cost_yuan,emergency_cost_yuan,reduce_cost_yuan,total_cost_yuan,"
        "soc_start_kwh,soc_end_kwh\n"
    )
    lines = [header]
    for label, bills in (("natural_day", result.day_bills), ("result_row", result.row_bills)):
        for day, b in bills:
            window_label = ("natural_day_partial_0010" if label == "natural_day" and result.problem != "problem1"
                            and day == DATE_2025_01_01 else label)
            lines.append(
                f"{window_label},{day.isoformat()},{b.plan_kwh:.6f},{b.add_kwh:.6f},"
                f"{b.emergency_kwh:.6f},{b.total_kwh:.6f},{b.plan_cost_yuan:.6f},"
                f"{b.add_cost_yuan:.6f},{b.emergency_cost_yuan:.6f},{b.reduce_cost_yuan:.6f},"
                f"{b.total_cost_yuan:.6f},,\n"
            )
    (run_dir / "daily_bills.csv").write_text("".join(lines), encoding="utf-8")

    (run_dir / "forecast_audit.json").write_text(json.dumps(result.forecast_audits, ensure_ascii=False), encoding="utf-8")
    (run_dir / "initial_plans.json").write_text(json.dumps(result.run.initial_plans), encoding="utf-8")
    (run_dir / "settlement_events.json").write_text(json.dumps({
        "execution": [asdict(e) for e in result.run.events],
        "penalties": [asdict(e) for e in result.run.penalties]}, default=_json_default), encoding="utf-8")
    arrays = result.run.arrays()
    # Keep physical arrays in absolute order; result_* fields carry template ordering.
    payload: dict[str, np.ndarray] = {
        "soc_boundary_kwh": result.run.soc_boundary_kwh,
        "plan_initial_kwh": np.array([result.run.initial_plans[int(m)] for m in result.run.abs_minutes]),
        **{k: v for k, v in arrays.items() if k != "soc_kwh"},
        **result.export_arrays,
    }
    # All paid purchases stay in the fee ledger, including their discarded part.
    import csv
    from datetime import datetime, timedelta
    with (run_dir / "spill_events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["abs_minute", "interval_start", "spill_kwh", "grid_paid_kwh", "demand_kwh", "soc_start_kwh", "soc_end_kwh", "charge_kwh", "capacity_limited", "power_limited"])
        for i, step in enumerate(result.run.steps):
            if step.surplus_kwh > 1e-6:
                writer.writerow([step.abs_minute, (datetime(2025, 1, 1)+timedelta(minutes=step.abs_minute)).isoformat(sep=" "), step.surplus_kwh, step.grid_kwh,
                    result.export_arrays["demand_kwh"][i], result.run.soc_boundary_kwh[i], step.soc_end_kwh,
                    step.charge_kwh, step.soc_end_kwh >= 10800-1e-6, step.charge_kwh >= 750-1e-6])
    np.savez_compressed(run_dir / "trajectory.npz", **payload)
    return run_dir


__all__ = [
    "RunConfig",
    "RunResult",
    "IntervalRecord",
    "POLICIES",
    "run",
    "run_problem1",
    "run_rolling",
    "save_run",
    "validate_result",
    "abs_minute_of_clock",
    "abs_minute_of_result_cell",
    "natural_day_bounds",
    "result_row_bounds",
]
