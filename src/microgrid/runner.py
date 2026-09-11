"""运行编排：数据 -> 预测 -> 计划 -> 因果执行 -> 结算 -> 校验 -> 存档。

Orchestration layer. This is the ONLY module that knows which problem is being
solved and which information rights apply. All numeric/physical conventions live
in :mod:`microgrid.constants` and :mod:`microgrid.physics`.

NOTE: This module is intentionally written with ASCII-only source text so that
its encoding can never be mangled by Windows PowerShell text pipelines.
Chinese documentation for the project lives in the Markdown reports and in the
other modules; see ``Pr/题意解析与建模衔接备忘录.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from . import validation
from .constants import (
    ADJUST_NODE_INTERVALS,
    DELTA_T_HOURS,
    E_INIT_2025_01_01,
    E_MAX,
    E_MIN,
    FEE_RATE_BY_CLASS,
    N_BOUNDARY,
    N_INTERVAL,
    OUTPUT_START,
    TEMPLATE_FILES,
    FeeClass,
)
from .data_io import DataBundle, data_path, input_fingerprint, load_all, project_root
from .forecast import Forecaster, terminal_water_value_yuan_per_kwh
from .planning import PlanResult, solve_problem1, solve_problem2, solve_problem3_remainder
from .schemas import ActivePlan, Bill, DayInput, PlanUpdate, Trajectory
from .settlement import penalty_for_adjustment, settle_trajectory
from .simulation import simulate_day_rolling
from .timeaxis import calendar_days
from .validation import TrajectoryDiagnostics


# ==========================================================================
# Configuration
# ==========================================================================


@dataclass
class RunConfig:
    """Complete configuration of one experiment.

    Every tunable lives here; nothing is hard-coded inside the algorithms.
    Physical constants (efficiencies, power limits, initial SOC) are NOT here --
    they are frozen in ``constants.py`` per the modelling memo.
    """

    problem: str = "problem1"          # problem1 / problem2 / problem3 / problem4-2 / problem4-3
    out_dir: Path = field(default_factory=lambda: Path("out"))

    # Information / forecasting
    history_days: int = 28
    load_method: str = "same_weekday"  # same_weekday | mean
    terminal_value_mode: str = "water"  # water | zero
    lookahead_days: int = 2

    # Robust day-plan bounds (P2/P3/P4 only). See Forecaster.robust_absorption_floor
    # and Forecaster.soc_uncertainty_band_kwh.
    z_demand: float = 1.0
    z_pv: float = 1.0
    n_pv_sigma: float = 2.0
    soc_band_rho: float = 2.5
    soc_band_min_kwh: float = 0.0
    soc_band_slack_kwh: float = 4000.0
    demand_bias: float = 1.0

    # Rolling dispatch execution (storage responds causally to realised gaps)
    dispatch_window: int = 24
    dispatch_lookahead: int = 48

    # When True, a committed-but-unabsorbable surplus is reported as a safety
    # valve and the run continues; when False such a day is a hard infeasibility.
    allow_surplus_safety_valve: bool = True

    solver_name: str = "appsi_highs"
    solver_time_limit_s: float | None = 30.0

    max_infeasible_days: int = 0
    save_every_day: bool = True

    def problem_dir(self) -> Path:
        return self.out_dir / self.problem


# ==========================================================================
# Result containers
# ==========================================================================


@dataclass
class DayRecord:
    """Everything produced for a single day, plus its diagnostics."""

    day: date
    trajectory: Trajectory
    bill: Bill
    diagnostics: TrajectoryDiagnostics
    plan_initial_kwh: np.ndarray | None = None
    plan_final_kwh: np.ndarray | None = None
    final_plan_curve_kwh: np.ndarray | None = None
    objective_yuan: float = float("nan")
    solve_seconds: float = 0.0
    infeasible: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    problem: str
    config: RunConfig
    days: list[DayRecord]
    summary: validation.ValidationSummary
    wall_seconds: float
    input_fingerprint: str
    extra: dict[str, object] = field(default_factory=dict)

    def output_days(self) -> list[DayRecord]:
        """Only records on/after OUTPUT_START; January is the warm-up period."""
        first = date(*OUTPUT_START)
        return [d for d in self.days if d.day >= first]


# ==========================================================================
# Shared helpers
# ==========================================================================


def _day_input_from_arrays(
    day: date,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    price: np.ndarray,
    label: str = "",
) -> DayInput:
    return DayInput(
        day=day,
        demand_kwh=np.asarray(demand_kwh, dtype=np.float64),
        pv_kwh=np.asarray(pv_kwh, dtype=np.float64),
        price_yuan_per_kwh=np.asarray(price, dtype=np.float64),
        label=label,
    )


def _actuals(bundle: DataBundle, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realised demand / PV / price. Only the simulation layer may call this."""
    i = bundle.attachment2.index_of(day)
    j = bundle.attachment4.index_of(day)
    return (
        bundle.attachment2.demand_kwh[i].copy(),
        bundle.attachment2.pv_kwh[i].copy(),
        bundle.attachment4.price_yuan_per_kwh[j].copy(),
    )


def _data_paths() -> list[Path]:
    root = project_root()
    return [
        data_path(p, root)
        for p in ("data/附件1.xlsx", "data/附件2.xlsx", "data/附件3.xlsx", "data/附件4.xlsx")
    ]


def _robust_plan_inputs(
    forecaster: Forecaster,
    day: date,
    demand_pred_kw: np.ndarray,
    pv_pred_kw: np.ndarray,
    soc: float,
    config: RunConfig,
) -> dict[str, object]:
    """Build the robust day-plan bounds shared by P2 / P3 / P4-2 / P4-3.

    Returns keyword arguments ready to be expanded into ``solve_problem2`` /
    ``solve_problem3_remainder``, plus a ``source_tag`` string for provenance.
    """
    floor_kw, ceil_kw, rsrc = forecaster.robust_absorption_floor(
        day,
        demand_pred_kw,
        pv_pred_kw,
        z_demand=config.z_demand,
        z_pv=config.z_pv,
        n_pv_sigma=config.n_pv_sigma,
    )
    band, _, bsrc = forecaster.soc_uncertainty_band_kwh(
        day, rho=config.soc_band_rho, min_band_kwh=config.soc_band_min_kwh
    )
    # The plan's SOC window must contain the current SOC, otherwise the plan is
    # infeasible at its own starting point (execution drift can push SOC outside).
    up_val = max(min(E_MAX - band[0], soc + config.soc_band_slack_kwh), soc)
    lo_val = min(max(E_MIN + band[0], soc - config.soc_band_slack_kwh), soc)
    if up_val <= lo_val:
        raise validation.ValidationError(
            f"{day} has an empty SOC robustness window: lo={lo_val:.0f} up={up_val:.0f}"
            f" (band={band[0]:.0f})"
        )
    return {
        "absorption_floor_kwh": floor_kw * DELTA_T_HOURS,
        "absorption_pv_ceiling_kwh": ceil_kw * DELTA_T_HOURS,
        "soc_upper_kwh": np.full(N_BOUNDARY, up_val),
        "soc_lower_kwh": np.full(N_BOUNDARY, lo_val),
        "soc_end_min_kwh": soc,
        "soc_end_max_equals": True,
        "source_tag": f"{rsrc}/{bsrc}",
    }


# ==========================================================================
# Problem 1 -- deterministic single-day LP
# ==========================================================================


def run_problem1(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    """Problem 1: one typical day, min purchase cost, E_0 = E_144 = 6000."""
    bundle = bundle or load_all()
    a1 = bundle.attachment1
    day = date(2025, 1, 1)  # Attachment 1 carries no date; used as a label only
    day_input = _day_input_from_arrays(
        day, a1.demand_kwh, a1.pv_kwh, a1.price_yuan_per_kwh, "attachment1-typical-day"
    )

    t0 = time.perf_counter()
    result: PlanResult = solve_problem1(
        day_input,
        soc_start_kwh=E_INIT_2025_01_01,
        soc_end_kwh=E_INIT_2025_01_01,
        price_yuan_per_kwh=a1.price_yuan_per_kwh,
        solver_name=config.solver_name,
    )
    result.report.require_ok()

    traj = Trajectory(
        day=day,
        grid_actual_kwh=result.grid_kwh.copy(),
        emergency_actual_kwh=np.zeros(N_INTERVAL),
        charge_stored_kwh=result.charge_kwh.copy(),
        discharge_delivered_kwh=result.discharge_kwh.copy(),
        curtail_kwh=result.curtail_kwh.copy(),
        soc_kwh=result.soc_kwh.copy(),
        price_actual=a1.price_yuan_per_kwh.copy(),
    )
    diag = validation.validate_trajectory(traj, day_input, require_daily_cycle=True, final=True)
    bill = settle_trajectory(traj)
    validation.validate_bill(bill, traj)
    validation.validate_day_input(day_input)
    validation.validate_attachment1_prices(a1.price_yuan_per_kwh)

    record = DayRecord(
        day=day,
        trajectory=traj,
        bill=bill,
        diagnostics=diag,
        plan_initial_kwh=result.grid_kwh.copy(),
        plan_final_kwh=result.grid_kwh.copy(),
        final_plan_curve_kwh=result.grid_kwh.copy(),
        objective_yuan=result.objective_yuan,
        solve_seconds=result.report.wall_seconds,
        notes=list(diag.warnings),
    )
    summary = validation.summarize([traj], [bill], [diag])
    return RunResult(
        problem="problem1",
        config=config,
        days=[record],
        summary=summary,
        wall_seconds=time.perf_counter() - t0,
        input_fingerprint=input_fingerprint(_data_paths()),
        extra={
            "objective_yuan": result.objective_yuan,
            "solver_status": result.report.status,
            "solver_termination": result.report.termination,
            "total_purchased_kwh": float(result.grid_kwh.sum()),
            "total_cost_yuan": float(np.sum(a1.price_yuan_per_kwh * result.grid_kwh)),
            "curtail_total_kwh": float(result.curtail_kwh.sum()),
        },
    )


# ==========================================================================
# Problem 2 / 4-2 -- frozen daily plan + causal rolling dispatch
# ==========================================================================


def run_problem2(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    """Problem 2: freeze one plan at 00:00, execute causally, January warm-up."""
    bundle = bundle or load_all()
    if config.problem not in ("problem2", "problem4-2"):
        raise ValueError(f"run_problem2 does not support {config.problem}")
    use_real_price = config.problem == "problem4-2"

    forecaster = Forecaster(
        bundle, history_days=config.history_days, load_method=config.load_method
    )
    all_days = calendar_days(date(2025, 1, 1), date(2025, 12, 31))
    repeated_price = forecaster.repeated_price_yuan_per_kwh()

    t0 = time.perf_counter()
    soc = E_INIT_2025_01_01
    records: list[DayRecord] = []
    for day in all_days:
        demand_true, pv_true, price_true = _actuals(bundle, day)

        # ---- Information available at 00:00 (never the realised day) ----
        demand_pred, dsrc = forecaster.demand_profile_kw(day)
        pv_pred_kw = forecaster.pv_from_hourly_kw(bundle.attachment3.hourly_power_on(day))
        demand_plan_kwh = demand_pred * DELTA_T_HOURS * config.demand_bias
        pv_plan_kwh = pv_pred_kw * DELTA_T_HOURS

        robust = _robust_plan_inputs(forecaster, day, demand_pred, pv_pred_kw, soc, config)
        source_tag = str(robust.pop("source_tag"))

        if use_real_price:
            price_plan, psrc = forecaster.price_profile_yuan_per_kwh(day)
        else:
            price_plan, psrc = repeated_price, "attachment1-repeat"
        price_emergency = price_true if use_real_price else repeated_price

        plan_input = _day_input_from_arrays(
            day, demand_plan_kwh, pv_plan_kwh, price_plan, f"{dsrc}/{source_tag}/{psrc}"
        )
        water = (
            terminal_water_value_yuan_per_kwh(
                forecaster, day + timedelta(days=1), fallback=float(np.median(price_plan))
            )
            if config.terminal_value_mode == "water"
            else 0.0
        )

        plan_result = solve_problem2(
            plan_input,
            soc_start_kwh=soc,
            plan_price_yuan_per_kwh=price_plan,
            actual_price_for_emergency=price_emergency,
            terminal_value_yuan_per_kwh=water,
            solver_name=config.solver_name,
            problem_name=config.problem,
            **robust,
        )
        plan_result.report.require_ok()
        plan = ActivePlan(created_at_interval=0, grid_kwh=plan_result.grid_kwh)

        # ---- Causal execution: grid follows the frozen plan, storage is re-dispatched ----
        sim = simulate_day_rolling(
            plan_input,
            plan,
            soc_start_kwh=soc,
            actual_demand_kwh=demand_true,
            actual_pv_kwh=pv_true,
            actual_price=price_true,
            window_intervals=config.dispatch_window,
            lookahead_intervals=config.dispatch_lookahead,
            allow_emergency=config.allow_surplus_safety_valve,
            solver_name=config.solver_name,
        )
        traj = sim.trajectory
        exec_input = _day_input_from_arrays(
            day, demand_true, pv_true, price_true, plan_input.label
        )
        diag = validation.validate_trajectory(
            traj,
            exec_input,
            final=not sim.infeasible,
            allow_surplus_safety_valve=config.allow_surplus_safety_valve,
        )
        diag.warnings.extend(sim.notes)
        bill = settle_trajectory(traj, initial_plan_kwh=plan_result.grid_kwh)
        if not sim.infeasible:
            validation.validate_bill(bill, traj, initial_plan_kwh=plan_result.grid_kwh)

        records.append(
            DayRecord(
                day=day,
                trajectory=traj,
                bill=bill,
                diagnostics=diag,
                plan_initial_kwh=plan_result.grid_kwh.copy(),
                plan_final_kwh=plan_result.grid_kwh.copy(),
                final_plan_curve_kwh=plan_result.grid_kwh.copy(),
                objective_yuan=plan_result.objective_yuan,
                solve_seconds=plan_result.report.wall_seconds,
                infeasible=sim.infeasible,
                notes=list(diag.warnings),
            )
        )
        soc = float(traj.soc_kwh[-1])

    out_days = [d for d in records if d.day >= date(*OUTPUT_START)]
    validation.validate_output_days([d.day for d in out_days])
    n_bad = sum(1 for d in out_days if d.infeasible)
    if n_bad > config.max_infeasible_days:
        raise validation.ValidationError(
            f"{config.problem}: {n_bad} infeasible day(s) exceed the allowed limit "
            f"{config.max_infeasible_days}; formal export stopped as required."
        )
    summary = validation.summarize(
        [d.trajectory for d in out_days],
        [d.bill for d in out_days],
        [d.diagnostics for d in out_days],
    )
    return RunResult(
        problem=config.problem,
        config=config,
        days=records,
        summary=summary,
        wall_seconds=time.perf_counter() - t0,
        input_fingerprint=input_fingerprint(_data_paths()),
        extra={"warmup_days": len(records) - len(out_days), "soc_at_output_start": soc},
    )


# ==========================================================================
# Problem 3 / 4-3 -- node adjustments + penalty accumulation
# ==========================================================================


def run_problem3(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    """Problem 3: 00:00 plan plus re-optimisation of the remaining day at 06/12/18."""
    bundle = bundle or load_all()
    if config.problem not in ("problem3", "problem4-3"):
        raise ValueError(f"run_problem3 does not support {config.problem}")
    use_real_price = config.problem == "problem4-3"

    forecaster = Forecaster(
        bundle, history_days=config.history_days, load_method=config.load_method
    )
    all_days = calendar_days(date(2025, 1, 1), date(2025, 12, 31))
    repeated_price = forecaster.repeated_price_yuan_per_kwh()

    t0 = time.perf_counter()
    soc = E_INIT_2025_01_01
    records: list[DayRecord] = []
    for day in all_days:
        demand_true, pv_true, price_true = _actuals(bundle, day)

        demand_pred, dsrc = forecaster.demand_profile_kw(day)
        demand_plan_kwh = demand_pred * DELTA_T_HOURS * config.demand_bias
        if use_real_price:
            price_plan, psrc = forecaster.price_profile_yuan_per_kwh(day)
        else:
            price_plan, psrc = repeated_price, "attachment1-repeat"
        price_emergency = price_true if use_real_price else repeated_price

        def pv_pred_kwh_at(hour: int) -> np.ndarray:
            """Forecast visible at publication time ``hour``, spread to 144 slots."""
            try:
                hourly = forecaster.pv_forecast_kw(day, day, hour)[0]
            except Exception:
                hourly = bundle.attachment3.hourly_power_on(day)
            return forecaster.pv_from_hourly_kw(hourly) * DELTA_T_HOURS

        pv_plan_kwh = pv_pred_kwh_at(0)
        robust = _robust_plan_inputs(
            forecaster, day, demand_pred, pv_plan_kwh / DELTA_T_HOURS, soc, config
        )
        source_tag = str(robust.pop("source_tag"))

        plan_input = _day_input_from_arrays(
            day, demand_plan_kwh, pv_plan_kwh, price_plan, f"{dsrc}/{source_tag}/{psrc}"
        )
        water = (
            terminal_water_value_yuan_per_kwh(
                forecaster, day + timedelta(days=1), fallback=float(np.median(price_plan))
            )
            if config.terminal_value_mode == "water"
            else 0.0
        )

        plan_result = solve_problem2(
            plan_input,
            soc_start_kwh=soc,
            plan_price_yuan_per_kwh=price_plan,
            actual_price_for_emergency=price_emergency,
            terminal_value_yuan_per_kwh=water,
            solver_name=config.solver_name,
            problem_name=f"{config.problem}-plan",
            **robust,
        )
        plan_result.report.require_ok()
        initial_plan = plan_result.grid_kwh.copy()
        active = ActivePlan(created_at_interval=0, grid_kwh=initial_plan.copy())
        updates: list[PlanUpdate] = []
        current_soc = soc

        # ---- Adjust the not-yet-executed remainder at 06:00 / 12:00 / 18:00 ----
        for node in ADJUST_NODE_INTERVALS[1:]:
            new_pv_kwh = pv_pred_kwh_at(node // 6)
            adj_robust = _robust_plan_inputs(
                forecaster, day, demand_pred, new_pv_kwh / DELTA_T_HOURS, current_soc, config
            )
            adj_robust.pop("source_tag", None)
            adj = solve_problem3_remainder(
                day=day,
                t_start=node,
                soc_start_kwh=current_soc,
                demand_kwh=demand_plan_kwh,
                pv_kwh=new_pv_kwh,
                plan_price_yuan_per_kwh=price_plan,
                emergency_price_yuan_per_kwh=price_emergency,
                terminal_value_yuan_per_kwh=water,
                solver_name=config.solver_name,
                problem_name=f"{config.problem}-adjust-{node}",
                **adj_robust,
            )
            adj.report.require_ok()
            y_new = active.grid_kwh.copy()
            y_new[node:] = adj.grid_kwh[node:]
            new_plan = ActivePlan(
                created_at_interval=node, grid_kwh=y_new, version=len(updates) + 1
            )
            price_at_node = (
                float(price_true[node]) if use_real_price else float(repeated_price[node])
            )
            update = penalty_for_adjustment(active, new_plan, price_at_node)
            if update.penalty_yuan > 0.0:
                updates.append(update)
            active = new_plan

        # ---- Causal execution of the final effective plan ----
        sim = simulate_day_rolling(
            plan_input,
            active,
            soc_start_kwh=soc,
            actual_demand_kwh=demand_true,
            actual_pv_kwh=pv_true,
            actual_price=price_true,
            window_intervals=config.dispatch_window,
            lookahead_intervals=config.dispatch_lookahead,
            allow_emergency=config.allow_surplus_safety_valve,
            solver_name=config.solver_name,
        )
        traj = sim.trajectory
        traj.updates = updates

        exec_input = _day_input_from_arrays(
            day, demand_true, pv_true, price_true, plan_input.label
        )
        diag = validation.validate_trajectory(
            traj,
            exec_input,
            require_daily_cycle=False,
            final=not sim.infeasible,
            allow_surplus_safety_valve=config.allow_surplus_safety_valve,
        )
        diag.warnings.extend(sim.notes)
        if updates:
            diag.warnings.append(
                f"{len(updates)} downward plan revision(s) today, penalty total "
                f"{sum(x.penalty_yuan for x in updates):.4f} CNY"
            )
        bill = settle_trajectory(traj, initial_plan_kwh=initial_plan)
        if not sim.infeasible:
            validation.validate_bill(bill, traj, initial_plan_kwh=initial_plan)

        records.append(
            DayRecord(
                day=day,
                trajectory=traj,
                bill=bill,
                diagnostics=diag,
                plan_initial_kwh=initial_plan.copy(),
                plan_final_kwh=active.grid_kwh.copy(),
                final_plan_curve_kwh=traj.grid_actual_kwh.copy(),
                objective_yuan=plan_result.objective_yuan,
                solve_seconds=plan_result.report.wall_seconds,
                infeasible=sim.infeasible,
                notes=list(diag.warnings),
            )
        )
        soc = float(traj.soc_kwh[-1])

    out_days = [d for d in records if d.day >= date(*OUTPUT_START)]
    validation.validate_output_days([d.day for d in out_days])
    n_bad = sum(1 for d in out_days if d.infeasible)
    if n_bad > config.max_infeasible_days:
        raise validation.ValidationError(
            f"{config.problem}: {n_bad} infeasible day(s) exceed the allowed limit "
            f"{config.max_infeasible_days}; formal export stopped as required."
        )
    summary = validation.summarize(
        [d.trajectory for d in out_days],
        [d.bill for d in out_days],
        [d.diagnostics for d in out_days],
    )
    return RunResult(
        problem=config.problem,
        config=config,
        days=records,
        summary=summary,
        wall_seconds=time.perf_counter() - t0,
        input_fingerprint=input_fingerprint(_data_paths()),
        extra={"warmup_days": len(records) - len(out_days)},
    )


# ==========================================================================
# Dispatch and persistence
# ==========================================================================


def run(config: RunConfig, bundle: DataBundle | None = None) -> RunResult:
    if config.problem == "problem1":
        return run_problem1(config, bundle)
    if config.problem in ("problem2", "problem4-2"):
        return run_problem2(config, bundle)
    if config.problem in ("problem3", "problem4-3"):
        return run_problem3(config, bundle)
    raise ValueError(f"unknown problem id: {config.problem}")


def save_run(result: RunResult) -> Path:
    """Persist config snapshot, daily trajectories, bills and validation summary."""
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
                "extra": result.extra,
                "validation": result.summary.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    header = (
        "day,normal_kwh,adjust_up_kwh,emergency_kwh,total_kwh,"
        "purchase_cost_yuan,penalty_yuan,total_cost_yuan,"
        "soc_start_kwh,soc_end_kwh,infeasible\n"
    )
    lines = [header]
    for rec in result.days:
        b = rec.bill
        lines.append(
            f"{rec.day.isoformat()},{b.normal_kwh:.6f},{b.adjust_up_kwh:.6f},{b.emergency_kwh:.6f},"
            f"{b.total_purchased_kwh:.6f},{b.purchase_cost_yuan:.6f},{b.penalty_yuan:.6f},"
            f"{b.total_cost_yuan:.6f},{rec.trajectory.soc_start_kwh:.6f},"
            f"{rec.trajectory.soc_end_kwh:.6f},{int(rec.infeasible)}\n"
        )
    (run_dir / "daily_bills.csv").write_text("".join(lines), encoding="utf-8")

    np.savez_compressed(
        run_dir / "trajectories.npz",
        days=np.array([d.day.toordinal() for d in result.days], dtype=np.int64),
        grid=np.array([d.trajectory.grid_actual_kwh for d in result.days]),
        emergency=np.array([d.trajectory.emergency_actual_kwh for d in result.days]),
        charge=np.array([d.trajectory.charge_stored_kwh for d in result.days]),
        discharge=np.array([d.trajectory.discharge_delivered_kwh for d in result.days]),
        curtail=np.array([d.trajectory.curtail_kwh for d in result.days]),
        soc=np.array([d.trajectory.soc_kwh for d in result.days]),
        price=np.array([d.trajectory.price_actual for d in result.days]),
        plan_initial=np.array(
            [
                d.plan_initial_kwh if d.plan_initial_kwh is not None else np.zeros(N_INTERVAL)
                for d in result.days
            ]
        ),
        final_plan=np.array(
            [
                d.final_plan_curve_kwh
                if d.final_plan_curve_kwh is not None
                else np.zeros(N_INTERVAL)
                for d in result.days
            ]
        ),
    )
    return run_dir


__all__ = [
    "RunConfig",
    "DayRecord",
    "RunResult",
    "run",
    "run_problem1",
    "run_problem2",
    "run_problem3",
    "save_run",
    "FeeClass",
    "FEE_RATE_BY_CLASS",
    "TEMPLATE_FILES",
]
