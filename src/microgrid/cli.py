"""Unified command entry point.

Order preserved from the guide: audit first, then run, then validate/export.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from .absolute_run import (
    POLICIES,
    RunConfig,
    run,
    save_run,
)
from .constants import OUTPUT_END, OUTPUT_START, SPECIAL_DATES
from .data_io import load_all
from .export import (
    build_daily_rows,
    export_multiday,
    export_result1,
    paper_table1,
    paper_table2,
    paper_table3,
)
from .solver import solver_available
from .timeaxis import calendar_days


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    return date.fromisoformat(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="microgrid", description="2026 C 题微网调控求解器")
    sub = p.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="只读审计数据结构、时间映射与求解器可用性")
    audit.add_argument("--solver", default="appsi_highs")

    runp = sub.add_parser("run", help="运行某一问")
    runp.add_argument(
        "--problem",
        required=True,
        choices=("problem1", "problem2", "problem3", "problem4-2", "problem4-3"),
    )
    runp.add_argument("--out", default="out")
    runp.add_argument("--solver", default="appsi_highs")
    runp.add_argument("--history-days", type=int, default=28)
    runp.add_argument(
        "--load-method",
        choices=("same_clock_mean", "same_weekday"),
        default="same_clock_mean",
        help="负载预测主线；same_weekday 为比较事项 P1 的对照设置",
    )
    runp.add_argument(
        "--absorption-safety-kwh",
        type=float,
        default=0.0,
        help="可消纳上限的安全裕度（技术近似，比较事项 A1）",
    )
    runp.add_argument(
        "--plan-refresh-intervals",
        type=int,
        default=1,
        help="主模型每段重算；其他周期需命名为独立实验",
    )
    runp.add_argument("--experiment", default="main", help="独立实验名；主模型为 main")
    runp.add_argument("--run-from", default=None, help="报告起始日期；仍从 1 月 1 日 00:10 连续执行（YYYY-MM-DD）")
    runp.add_argument("--run-to", default=None, help="只跑到该日期（YYYY-MM-DD）")
    runp.add_argument("--max-infeasible-intervals", type=int, default=0)
    runp.add_argument(
        "--progress-every-days",
        type=int,
        default=0,
        help="每 N 个自然日打印一次进度（0 = 不打印）",
    )
    runp.add_argument("--strict-no-spill", action="store_true", help="旧版禁止弃购电对照；需指定 --experiment 名称")
    runp.add_argument("--no-save", action="store_true")

    exp = sub.add_parser("export", help="由已保存的运行结果生成结果文件与论文用表")
    exp.add_argument(
        "--problem",
        required=True,
        choices=("problem1", "problem2", "problem3", "problem4-2", "problem4-3"),
    )
    exp.add_argument("--out", default="out")
    exp.add_argument("--experiment", default="main")
    return p


# ==========================================================================
# audit
# ==========================================================================


def cmd_audit(args: argparse.Namespace) -> int:
    print("=" * 78)
    print("数据与时间轴审计（只读：不修改 data/ 与 Pr/ 下任何文件）")
    print("=" * 78)
    bundle = load_all()
    a1, a2, a3, a4 = bundle.attachment1, bundle.attachment2, bundle.attachment3, bundle.attachment4
    print(f"附件1  典型日      144 段：电价 {a1.price_yuan_per_kwh.min():.4f}~{a1.price_yuan_per_kwh.max():.4f} 元/kWh")
    print(f"附件2  {len(a2.days)} 天 × 144 段：负载 {a2.load_kw.min():.1f}~{a2.load_kw.max():.1f} kW")
    print(f"附件3  {len(a3.blocks)} 条发布块（每天 0/6/12/18），每条 24 小时预报")
    print(f"附件4  {len(a4.days)} 天 × 144 段：电价 {a4.price_yuan_per_kwh.min():.4f}~"
          f"{a4.price_yuan_per_kwh.max():.4f} 元/kWh")

    from .timeline import build_timeline

    tl = build_timeline(bundle)
    print()
    print("绝对时间线：")
    for n in tl.notes:
        print("  -", n)
    for n in tl.bridge_notes():
        print("  *", n)
    print()
    print("时间轴口径（2026-09-11 定稿）：源标签=区间起点；")
    print("  result 行覆盖 [当日 00:10, 次日 00:10)；自然日时钟 t=0 由上一源日尾值桥接")
    print("  B_{d,j} = E_{d,j+1}；B_{d,144} = E_{d+1,1} = B_{d+1,0}（共享次日00:10状态）")
    print()
    print(f"输出窗口：{OUTPUT_START} .. {OUTPUT_END}（"
          f"{len(calendar_days(date(*OUTPUT_START), date(*OUTPUT_END)))} 天）+ 1 月预热")
    print(f"重点日：{', '.join(SPECIAL_DATES)}")
    print(f"求解器 {args.solver}: {'可用' if solver_available(args.solver) else '不可用'}")
    print()
    print("已冻结口径：η_c = η_d = 0.9；每段 q_ch, q_dis ≤ 750 kWh；1200 ≤ E ≤ 10800；")
    print("            允许同时充放电（正常损耗核算）；滚动初始 6000 设于 2025-01-01 00:10，跳过年初首段。")
    return 0


# ==========================================================================
# run
# ==========================================================================


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig(
        problem=args.problem,
        out_dir=Path(args.out),
        history_days=args.history_days,
        experiment=args.experiment,
        load_method=args.load_method,
        absorption_safety_kwh=args.absorption_safety_kwh,
        plan_refresh_intervals=args.plan_refresh_intervals,
        allow_spill=not args.strict_no_spill,
        max_infeasible_intervals=args.max_infeasible_intervals,
        solver_name=args.solver,
        run_from=_parse_date(args.run_from),
        run_to=_parse_date(args.run_to),
    )
    result = run(config)
    s = result.summary
    g = lambda k, d=0.0: float(s.get(k, d))  # noqa: E731
    print(f"[{args.problem}] 用时 {result.wall_seconds:.1f}s")
    print(f"  策略              {POLICIES[args.problem].name}"
          f"（可调计划={POLICIES[args.problem].can_adjust_plan}, "
          f"用已发布光伏={POLICIES[args.problem].use_published_pv}, "
          f"价格={POLICIES[args.problem].price_mode}）")
    print(f"  执行段数          {int(g('n_intervals'))}")
    print(f"  普通购电          {g('plan_kwh'):,.3f} kWh（原计划）")
    print(f"  调整增购          {g('add_kwh'):,.3f} kWh")
    print(f"  紧急购电          {g('emergency_kwh'):,.3f} kWh")
    print(f"  执行费用          {g('execution_cost_yuan'):,.3f} 元")
    print(f"  违约金            {g('reduce_cost_yuan'):,.3f} 元")
    print(f"  总费用            {g('total_cost_yuan'):,.3f} 元")
    print(f"  SOC 起/末/最小/最大  {g('soc_start_kwh'):,.0f} / {g('soc_end_kwh'):,.0f} / "
          f"{g('soc_min_kwh'):,.0f} / {g('soc_max_kwh'):,.0f}")
    print(f"  充电/放电/弃光     {g('charge_total_kwh'):,.0f} / {g('discharge_total_kwh'):,.0f} / "
          f"{g('curtail_total_kwh'):,.0f} kWh")
    print(f"  损耗              {g('total_loss_kwh'):,.3f} kWh"
          f"（同时充放 {int(g('n_simultaneous_intervals'))} 段，属合法运行状态）")
    print(f"  富余无处安放      {g('surplus_disposed_kwh'):,.3f} kWh（已付费弃购电）")
    print(f"  窗口求解/计划成文  {int(g('window_solves'))} / {int(g('plan_revisions'))}")
    infeasible = s.get("infeasible_intervals") or []
    if infeasible:
        print(f"  不可行区间        {len(infeasible)} 个")
    for n in result.notes[:4]:
        print("  -", n)
    if not args.no_save:
        print(f"  结果目录          {save_run(result)}")
    return 0


# ==========================================================================
# export
# ==========================================================================


def cmd_export(args: argparse.Namespace) -> int:
    out_root = Path(args.out)
    run_dir = (out_root if args.experiment == "main" else out_root / args.experiment) / args.problem
    npz_path = run_dir / "trajectory.npz"
    summary_path = run_dir / "summary.json"
    if not npz_path.exists():
        print(f"找不到 {npz_path}；请先运行 microgrid run --problem {args.problem}", file=sys.stderr)
        return 2

    data = np.load(npz_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    # save_run stores the SOC boundary series under its own key.
    if "soc_kwh" not in arrays and "soc_boundary_kwh" in arrays:
        arrays["soc_kwh"] = arrays["soc_boundary_kwh"]
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    row_bills = {
        rec["date"]: rec for rec in payload.get("result_row_bills", [])
    }

    from .validation import validate_absolute_run
    if payload.get("validation_version") != "main-model-v4-paid-spill":
        raise ValueError("旧结果未经当前主模型校验，请重新运行")
    saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    validate_absolute_run(abs_minutes=arrays["abs_minute"], grid_kwh=arrays["grid_kwh"],
        emergency_kwh=arrays["emergency_kwh"], charge_kwh=arrays["charge_kwh"],
        discharge_kwh=arrays["discharge_kwh"], curtail_kwh=arrays["curtail_kwh"],
        surplus_kwh=arrays["surplus_kwh"], soc_boundary_kwh=arrays["soc_kwh"],
        demand_kwh=arrays["demand_kwh"], pv_kwh=arrays["pv_kwh"], soc_start_kwh=6000,
        require_daily_cycle=args.problem == "problem1",
        allow_spill=saved_config["allow_spill"] and args.problem != "problem1")
    if "plan_initial_kwh" not in arrays or not np.isfinite(arrays["plan_initial_kwh"]).all():
        raise ValueError("Missing midnight plan archive")
    from .schemas import DispatchEvent, PenaltyEvent
    from .settlement import validate_ledger, LabelTotals, verify_bill
    from .absolute_run import natural_day_bounds, result_row_bounds
    ledger = json.loads((run_dir / "settlement_events.json").read_text(encoding="utf-8"))
    initial = json.loads((run_dir / "initial_plans.json").read_text(encoding="utf-8"))
    events = [DispatchEvent(**e) for e in ledger["execution"]]
    penalties = [PenaltyEvent(**p) for p in ledger["penalties"]]
    validate_ledger(abs_minutes=arrays["abs_minute"], grid_kwh=arrays["grid_kwh"],
        emergency_kwh=arrays["emergency_kwh"], price_actual=arrays["price_actual"],
        initial_plans=initial, events=events, penalties=penalties,
        can_adjust=args.problem in ("problem3", "problem4-3"),
        expected_start_abs=0 if args.problem == "problem1" else 10)
    if not np.allclose(arrays["plan_initial_kwh"], [initial[str(int(t))] for t in arrays["abs_minute"]], atol=1e-6, rtol=0):
        raise ValueError("Saved initial plan differs from midnight archive")
    for key, field in (("plan_exec_kwh", "o_exec_kwh"), ("add_exec_kwh", "a_exec_kwh")):
        if args.problem != "problem1" and (key not in arrays or not np.allclose(
                arrays[key], [getattr(e, field) for e in events], atol=1e-6, rtol=0)):
            raise ValueError(f"Saved {key} differs from settlement ledger")
    for name, bounds in (("natural_day_bills", natural_day_bounds), ("result_row_bills", result_row_bounds)):
        if args.problem == "problem1" and name == "result_row_bills":
            continue
        for rec in payload.get(name, []):
            lo, hi = bounds(date.fromisoformat(rec["date"]))
            bill = LabelTotals(**{k: rec[k] for k in LabelTotals.__dataclass_fields__})
            verify_bill(bill, [e for e in events if lo <= e.at_abs < hi],
                        [p for p in penalties if lo <= p.at_abs < hi])
            if any(not np.isclose(rec[k], v, atol=1e-5, rtol=1e-9) for k, v in bill.as_dict().items()):
                raise ValueError("Saved bill totals differ from fee components")
    if args.problem == "problem1":
        if np.any(arrays["emergency_kwh"] > 1e-6) or not np.allclose(
                arrays["result_grid_kwh"], np.roll(arrays["grid_kwh"], -1), atol=1e-6, rtol=0):
            raise ValueError("Invalid problem 1 execution/export")
        dest = run_dir / "result"
        path = export_result1(arrays, dest)
        print(f"已写出 {path}")
        return 0

    from .absolute_run import abs_minute_of_result_cell, natural_day_bounds
    from .constants import OUTPUT_START as _OS
    from .export import DailyExportRow
    from .timeline import MINUTES_PER_DAY

    if not row_bills:
        print(
            "summary.json 里没有 result_row_bills，无法导出；"
            "请用当前版本的 runner 重跑该问",
            file=sys.stderr,
        )
        return 2

    minutes = arrays["abs_minute"]
    step_index = {int(minutes[i]): i for i in range(minutes.size)}
    soc_by_abs: dict[int, float] = {}
    for i, m in enumerate(minutes):
        soc_by_abs[int(m)] = float(arrays["soc_kwh"][i])
        soc_by_abs[int(m) + 10] = float(arrays["soc_kwh"][i + 1])

    # Initial midnight commitments and final executed O/A labels are distinct.
    plan_exec = arrays.get("plan_exec_kwh")
    add_exec = arrays.get("add_exec_kwh")
    has_split = plan_exec is not None and add_exec is not None
    if not has_split:
        raise ValueError("Missing executed O/A labels")

    rows: list[DailyExportRow] = []
    for day_text in sorted(row_bills):
        day = date.fromisoformat(day_text)
        rec = row_bills[day_text]
        grid = np.zeros(144)
        emg = np.zeros(144)
        ch = np.zeros(144)
        dis = np.zeros(144)
        plan = np.zeros(144)
        adjust = np.zeros(144)
        for j in range(144):
            abs_minute = abs_minute_of_result_cell(day, j)
            idx = step_index.get(abs_minute)
            if idx is None:
                raise ValueError(f"Missing executed interval {abs_minute}")
            if idx is not None:
                grid[j] = float(arrays["grid_kwh"][idx])
                emg[j] = float(arrays["emergency_kwh"][idx])
                ch[j] = float(arrays["charge_kwh"][idx])
                dis[j] = float(arrays["discharge_kwh"][idx])
                if has_split:
                    plan[j] = float(arrays["plan_initial_kwh"][idx])
                    adjust[j] = float(add_exec[idx])
        df, _dt = natural_day_bounds(day)
        # The 4-hour blocks are labelled on the natural-day clock (0:00-4:00 …),
        # and the day's own 00:00-00:10 interval sits in the PREVIOUS date's
        # result row, so it is read straight from the trajectory.
        i_start = step_index.get(df)
        start_minute = 10 if df == 0 and minutes[0] == 10 else 0
        rows.append(
            DailyExportRow(
                day=day,
                # "00:00 plan" is the signed commitment effective at midnight,
                # which for every interval of day d is the plan formed then; the
                # intra-day revisions are the deltas.
                plan_initial_kwh=plan if has_split else grid.copy(),
                # effective purchase = O + (intra-day revisions)
                final_plan_kwh=grid.copy(),
                grid_actual_kwh=grid,
                emergency_actual_kwh=emg,
                charge_stored_kwh=ch,
                discharge_delivered_kwh=dis,
                soc_natural_start_kwh=soc_by_abs.get(df + start_minute, float("nan")),
                natural_start_minute=start_minute,
                soc_natural_end_kwh=soc_by_abs.get(df + MINUTES_PER_DAY, float("nan")),
                execution_cost_yuan=float(rec.get("execution_cost_yuan", 0.0)),
                reduce_cost_yuan=float(rec.get("reduce_cost_yuan", 0.0)),
                charge_clock_start_kwh=(
                    float(arrays["charge_kwh"][i_start]) if i_start is not None else None
                ),
                discharge_clock_start_kwh=(
                    float(arrays["discharge_kwh"][i_start]) if i_start is not None else None
                ),
            )
        )

    dest = run_dir / "result"
    wants_adjust = args.problem in ("problem3", "problem4-3")
    # Recompute saved window bills from execution labels before writing a workbook.
    for row in rows:
        lo = (row.day - date(2025, 1, 1)).days * 1440 + 10
        mask = (minutes >= lo) & (minutes < lo + 1440)
        cost = float(np.sum(arrays["price_actual"][mask] *
                     (plan_exec[mask] + 1.5 * add_exec[mask] + 5 * arrays["emergency_kwh"][mask])))
        if not np.isclose(cost, row.execution_cost_yuan, atol=1e-5, rtol=1e-9):
            raise ValueError("Saved execution bill mismatch")
    path = export_multiday(
        args.problem,
        rows,
        dest,
        with_adjust_sheet=wants_adjust,
    )
    print(f"已写出 {path}（{len(rows)} 天）")

    specials = [r for r in rows if r.day.isoformat() in SPECIAL_DATES]
    tables = {
        "paper_table1.csv": paper_table1(specials),
        "paper_table2.csv": paper_table2(specials),
        "paper_table3.csv": paper_table3(specials),
    }
    for name, tbl in tables.items():
        p = run_dir / name
        keys = sorted({k for row_ in tbl for k in row_})
        lines = [",".join(keys)]
        for row_ in tbl:
            lines.append(",".join(str(row_.get(k, "")) for k in keys))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        print(f"已写出 {p}（{len(tbl)} 行）")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "export":
        return cmd_export(args)
    raise SystemExit(f"未知命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
