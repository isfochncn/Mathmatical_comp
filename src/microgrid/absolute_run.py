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
    absorption_safety_kwh: float = 250.0
    allow_spill: bool = True
    max_infeasible_intervals: int = 0
    solver_name: str = "appsi_highs"

    # How often the window LP is re-solved (in ten-minute intervals). The memo
    # asks for a refresh at every interval start; 1 is that literal cadence and
    # costs one LP per ten minutes. Larger values are a documented performance
    # approximation: the *execution feedback* still runs every interval, only
    # the re-planning frequency changes. Reported under comparison item A1.
    plan_refresh_intervals: int = 1

    # Date window to execute (smoke tests); defaults to the whole year.
    run_from: date | None = None
    run_to: date | None = None
    warmup_days: int = 0

    def problem_dir(self) -> Path:
        return self.out_dir / self.problem


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
        demand_kwh=a1.demand_kwh.copy(),
        pv_kwh=a1.pv_kwh.copy(),
        price_yuan_per_kwh=a1.price_yuan_per_kwh.copy(),
        provenance=np.array(["attachment1-typical-day"] * N_INTERVAL, dtype=object),
    )
    result: WindowResult = solve_window(
        forecast=forecast,
        soc_start_kwh=E_INIT_2025_01_01,
        fee_mode=FeeMode.FIRST_PLAN,
        soc_end_fixed_kwh=E_INIT_2025_01_01,
        solver_name=config.solver_name,
        problem_name="problem1",
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
                price_actual_yuan_per_kwh=float(a1.price_yuan_per_kwh[t]),
            )
        )
        events.append(
            DispatchEvent(
                at_abs=int(minutes[t]),
                o_exec_kwh=float(result.grid_kwh[t]),
                a_exec_kwh=0.0,
                emergency_kwh=0.0,
                price_actual_yuan_per_kwh=float(a1.price_yuan_per_kwh[t]),
            )
        )
    run = AbsoluteRun(
        abs_minutes=minutes,
        steps=steps,
        soc_boundary_kwh=np.asarray(result.soc_boundary_kwh, dtype=np.float64),
        events=events,
        penalties=[],
    )
    totals = natural_day_bill(0, MINUTES_PER_DAY, events, [])
    verify_bill(totals, events, [])

    summary = _p1_summary(run, result, totals)
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
    )


def _p1_summary(run: AbsoluteRun, result: WindowResult, totals: LabelTotals) -> dict[str, object]:
    arrays = run.arrays()
    charge = arrays["charge_kwh"]
    discharge = arrays["discharge_kwh"]
    loss = physics.loss_accounting(charge, discharge)
    soc = arrays["soc_kwh"]
    return {
        "problem": "problem1",
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
                np.array([s for s in np.full(N_INTERVAL, 0.0)]),
                np.array([s for s in np.full(N_INTERVAL, 0.0)]),
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


def run_rolling(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    if config.problem not in POLICIES or config.problem == "problem1":
        raise ValueError(f"run_rolling does not support {config.problem}")
    bundle = bundle or load_all()
    policy = POLICIES[config.problem]
    timeline = build_timeline(bundle)
    forecaster = Forecaster(bundle, timeline, history_days=config.history_days)

    days = calendar_days(DATE_2025_01_01, DATE_2025_12_31)
    if config.run_from is not None:
        days = [d for d in days if d >= config.run_from]
    if config.run_to is not None:
        days = [d for d in days if d <= config.run_to]
    if not days:
        raise ValueError("运行区间为空")
    abs_from = (days[0] - DATE_2025_01_01).days * MINUTES_PER_DAY
    abs_to = (days[-1] - DATE_2025_01_01).days * MINUTES_PER_DAY + MINUTES_PER_DAY

    price_lookup = None
    if policy.price_mode == "historical":
        # The current quote is only used once it is revealed: the plan always
        # sees realised prices strictly before the current interval.
        def price_lookup(abs_minute: int) -> float | None:  # type: ignore[misc]
            if abs_minute - 10 < 0:
                return None
            return float(timeline.price_yuan_per_kwh.value_at(abs_minute - 10))

    t0 = time.perf_counter()
    outcome = run_absolute(
        timeline=timeline,
        forecaster=forecaster,
        policy=policy,
        options=RunOptions(
            absorption_safety_kwh=config.absorption_safety_kwh,
            allow_spill=config.allow_spill,
            plan_refresh_intervals=config.plan_refresh_intervals,
            max_infeasible_intervals=config.max_infeasible_intervals,
            solver_name=config.solver_name,
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

    out_start = date(*OUTPUT_START)
    output_day_bills = [(d, b) for d, b in day_bills if d >= out_start]
    output_row_bills = [(d, b) for d, b in row_bills if d >= out_start]

    # Independent recomputation over the whole run and over the reporting window.
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

    arrays = run.arrays()
    loss = physics.loss_accounting(arrays["charge_kwh"], arrays["discharge_kwh"])
    summary: dict[str, object] = {
        "problem": config.problem,
        "n_intervals": int(arrays["abs_minute"].size),
        "total_purchased_kwh": grand.total_kwh,
        "plan_kwh": grand.plan_kwh,
        "add_kwh": grand.add_kwh,
        "emergency_kwh": grand.emergency_kwh,
        "plan_cost_yuan": grand.plan_cost_yuan,
        "add_cost_yuan": grand.add_cost_yuan,
        "emergency_cost_yuan": grand.emergency_cost_yuan,
        "execution_cost_yuan": grand.execution_cost_yuan,
        "reduce_cost_yuan": grand.reduce_cost_yuan,
        "total_cost_yuan": grand.total_cost_yuan,
        "soc_start_kwh": float(run.soc_boundary_kwh[0]),
        "soc_end_kwh": float(run.soc_boundary_kwh[-1]),
        "soc_min_kwh": float(run.soc_boundary_kwh.min()),
        "soc_max_kwh": float(run.soc_boundary_kwh.max()),
        "charge_total_kwh": float(np.sum(arrays["charge_kwh"])),
        "discharge_total_kwh": float(np.sum(arrays["discharge_kwh"])),
        "curtail_total_kwh": float(np.sum(arrays["curtail_kwh"])),
        "surplus_disposed_kwh": float(np.sum(arrays["surplus_kwh"])),
        "total_loss_kwh": loss["total_loss_kwh"],
        "n_simultaneous_intervals": loss["n_simultaneous_intervals"],
        "window_solves": outcome.window_solves,
        "plan_revisions": outcome.revisions,
        "infeasible_intervals": outcome.infeasible_abs,
        "output_days": len(output_day_bills),
        "output_row_purchase_kwh": float(sum(b.total_kwh for _, b in output_row_bills)),
        "output_natural_day_purchase_kwh": float(sum(b.total_kwh for _, b in output_day_bills)),
        "output_row_cost_yuan": float(sum(b.total_cost_yuan for _, b in output_row_bills)),
        "output_natural_day_cost_yuan": float(
            sum(b.total_cost_yuan for _, b in output_day_bills)
        ),
        "max_bus_charge_kw": float(np.max(arrays["charge_kwh"]) * 6 / 0.9),
        "max_bus_discharge_kw": float(np.max(arrays["discharge_kwh"]) * 6),
    }
    return RunResult(
        problem=config.problem,
        config=config,
        timeline=timeline,
        run=run,
        day_bills=day_bills,
        row_bills=row_bills,
        summary=summary,
        wall_seconds=wall,
        input_fingerprint=input_fingerprint(_data_paths()),
        notes=outcome.notes,
    )


def run(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    if config.problem == "problem1":
        return run_problem1(config, bundle)
    return run_rolling(config, bundle)


# ==========================================================================
# Persistence
# ==========================================================================


def _data_paths() -> list[Path]:
    root = project_root()
    return [
        data_path(p, root)
        for p in ("data/附件1.xlsx", "data/附件2.xlsx", "data/附件3.xlsx", "data/附件4.xlsx")
    ]


def save_run(result: RunResult) -> Path:
    run_dir = result.config.problem_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(result.config).items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "problem": result.problem,
                "wall_seconds": result.wall_seconds,
                "input_fingerprint": result.input_fingerprint,
                "summary": result.summary,
                "notes": result.notes,
                "policy": asdict(POLICIES[result.problem]) if result.problem in POLICIES else {},
            },
            ensure_ascii=False,
            indent=2,
            default=float,
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
            lines.append(
                f"{label},{day.isoformat()},{b.plan_kwh:.6f},{b.add_kwh:.6f},"
                f"{b.emergency_kwh:.6f},{b.total_kwh:.6f},{b.plan_cost_yuan:.6f},"
                f"{b.add_cost_yuan:.6f},{b.emergency_cost_yuan:.6f},{b.reduce_cost_yuan:.6f},"
                f"{b.total_cost_yuan:.6f},,\n"
            )
    (run_dir / "daily_bills.csv").write_text("".join(lines), encoding="utf-8")

    arrays = result.run.arrays()
    np.savez_compressed(
        run_dir / "trajectory.npz",
        soc_boundary_kwh=result.run.soc_boundary_kwh,
        **{k: v for k, v in arrays.items() if k != "soc_kwh"},
    )
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
    "abs_minute_of_clock",
    "abs_minute_of_result_cell",
    "natural_day_bounds",
    "result_row_bounds",
]
