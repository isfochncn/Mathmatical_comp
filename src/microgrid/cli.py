"""统一命令入口。

保留"先审计、后运行、再验证/导出"的顺序（指南第 14 节）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from .constants import OUTPUT_END, OUTPUT_START, SPECIAL_DATES
from .data_io import load_all
from .export import (
    daily_export_rows,
    export_multiday,
    export_result1,
    paper_table1,
    paper_table2,
    paper_table3,
)
from .runner import RunConfig, run, save_run
from .solver import solver_available
from .timeaxis import calendar_days


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default="out", help="输出根目录（默认 out/）")
    parser.add_argument("--solver", default="appsi_highs", help="Pyomo 求解器名")
    parser.add_argument("--lookahead-days", type=int, default=2, help="滚动时域天数（含当日）")
    parser.add_argument(
        "--terminal-value",
        choices=("water", "zero"),
        default="water",
        help="跨日终端价值口径：water（次日预测价中位数）/ zero（短视对照）",
    )
    parser.add_argument("--history-days", type=int, default=28, help="预测回看天数")
    parser.add_argument(
        "--demand-bias",
        type=float,
        default=1.0,
        help="规划用负载安全系数（>1 偏保守，降低计划过量风险）",
    )
    parser.add_argument("--dispatch-window", type=int, default=24, help="滚动调度窗口（段）")
    parser.add_argument("--dispatch-lookahead", type=int, default=48, help="滚动调度 LP 视界（段）")
    parser.add_argument(
        "--strict-no-surplus",
        action="store_true",
        help="禁止把无处安放的富余作为安全阀：出现即判不可行日（规范严格要求）",
    )
    parser.add_argument(
        "--max-infeasible-days", type=int, default=0, help="允许的不可行天数上限，超过即失败"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="microgrid", description="2026 C 题微网调控求解器")
    sub = p.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="只读审计数据结构与求解器可用性")
    audit.add_argument("--out", default="out")
    audit.add_argument("--solver", default="appsi_highs", help="Pyomo 求解器名")

    runp = sub.add_parser("run", help="运行某一问")
    runp.add_argument(
        "--problem",
        required=True,
        choices=("problem1", "problem2", "problem3", "problem4-2", "problem4-3"),
    )
    _add_common(runp)
    runp.add_argument("--no-save", action="store_true", help="不写 out/ 目录")

    exp = sub.add_parser("export", help="由已保存的运行结果生成结果文件与论文用表")
    exp.add_argument(
        "--problem",
        required=True,
        choices=("problem1", "problem2", "problem3", "problem4-2", "problem4-3"),
    )
    exp.add_argument("--out", default="out")

    return p


# ==========================================================================
# audit
# ==========================================================================


def cmd_audit(args: argparse.Namespace) -> int:
    print("=" * 72)
    print("数据审计（只读：不修改 data/ 与 Pr/ 下任何文件）")
    print("=" * 72)
    bundle = load_all()
    a1, a2, a3, a4 = bundle.attachment1, bundle.attachment2, bundle.attachment3, bundle.attachment4
    print(f"附件1  典型日      144 段：电价 {a1.price_yuan_per_kwh.min():.4f}~{a1.price_yuan_per_kwh.max():.4f} 元/kWh，"
          f"负载均值 {a1.load_kw.mean():.1f} kW，光伏峰值 {a1.pv_forecast_kw.max():.1f} kW")
    print(f"附件2  {len(a2.days)} 天 × 144 段：负载 {a2.load_kw.min():.1f}~{a2.load_kw.max():.1f} kW，"
          f"光伏 {a2.pv_actual_kw.min():.1f}~{a2.pv_actual_kw.max():.1f} kW")
    print(f"附件3  {len(a3.blocks)} 条发布块（每天 0/6/12/18），每条 24 小时预报")
    print(f"附件4  {len(a4.days)} 天 × 144 段：电价 {a4.price_yuan_per_kwh.min():.4f}~"
          f"{a4.price_yuan_per_kwh.max():.4f} 元/kWh，非正价 {int((a4.price_yuan_per_kwh <= 0).sum())} 个")
    print(f"输出窗口：{OUTPUT_START} .. {OUTPUT_END}（{len(calendar_days(date(*OUTPUT_START), date(*OUTPUT_END)))} 天）+ 1 月预热")
    print(f"重点日：{', '.join(SPECIAL_DATES)}")
    print(f"求解器 {args.solver}: {'可用' if solver_available(args.solver) else '不可用'}")
    print()
    print("已确认口径：η_c = η_d = 0.9；每段 q_ch, q_dis ≤ 750 kWh；1200 ≤ E ≤ 10800；")
    print("            两向电池内部交换功率均不超过 5000 kW 额定；初始 6000 kWh（仅 2025-01-01 0:00）。")
    return 0


# ==========================================================================
# run
# ==========================================================================


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig(
        problem=args.problem,
        out_dir=Path(args.out),
        lookahead_days=args.lookahead_days,
        terminal_value_mode=args.terminal_value,
        history_days=args.history_days,
        solver_name=args.solver,
        demand_bias=args.demand_bias,
        dispatch_window=args.dispatch_window,
        dispatch_lookahead=args.dispatch_lookahead,
        allow_surplus_safety_valve=not args.strict_no_surplus,
        max_infeasible_days=args.max_infeasible_days,
    )
    result = run(config)
    print(f"[{args.problem}] 用时 {result.wall_seconds:.1f}s，共 {len(result.days)} 天（含 1 月预热）")
    s = result.summary
    print(f"  输出天数        {s.n_days}")
    print(f"  总购电量        {s.total_purchased_kwh:,.3f} kWh")
    print(f"  紧急购电量      {s.total_emergency_kwh:,.3f} kWh")
    print(f"  违约金          {s.total_penalty_yuan:,.3f} 元")
    print(f"  总费用          {s.total_cost_yuan:,.3f} 元")
    print(f"  期末 SOC        {s.soc_end_kwh:,.3f} kWh")
    print(f"  最大守恒残差    {s.max_bus_residual_kwh:.3e} kWh")
    print(f"  最大 SOC 递推误差 {s.max_soc_transition_error_kwh:.3e} kWh")
    print(f"  同时充放段数    {s.n_simultaneous_intervals}")
    print(f"  最大母线功率    充 {s.max_bus_charge_kw:,.1f} kW / 放 {s.max_bus_discharge_kw:,.1f} kW")
    if s.warnings:
        print("  警告：")
        for w in s.warnings[:10]:
            print(f"    - {w}")
    if not args.no_save:
        run_dir = save_run(result)
        print(f"  结果目录        {run_dir}")
    return 0


# ==========================================================================
# export
# ==========================================================================


def cmd_export(args: argparse.Namespace) -> int:
    from .runner import RunResult, DayRecord  # noqa: F401  (仅用于类型说明)

    out_root = Path(args.out)
    run_dir = out_root / args.problem
    npz_path = run_dir / "trajectories.npz"
    if not npz_path.exists():
        print(f"找不到 {npz_path}；请先运行 microgrid run --problem {args.problem}", file=sys.stderr)
        return 2

    data = np.load(npz_path)
    days = [date.fromordinal(int(o)) for o in data["days"]]
    first = date(*OUTPUT_START)

    from .schemas import Trajectory

    records = []
    for i, day in enumerate(days):
        # 1 月是预热期，不进入结果文件；但问题一是典型日，本身标在 1 月 1 日
        if day < first and args.problem != "problem1":
            continue
        traj = Trajectory(
            day=day,
            grid_actual_kwh=data["grid"][i],
            emergency_actual_kwh=data["emergency"][i],
            charge_stored_kwh=data["charge"][i],
            discharge_delivered_kwh=data["discharge"][i],
            curtail_kwh=data["curtail"][i],
            soc_kwh=data["soc"][i],
            price_actual=data["price"][i],
        )
        records.append((day, traj, data["plan_initial"][i], data["final_plan"][i]))

    if args.problem == "problem1":
        if not records:
            print("运行结果里没有可导出的日期", file=sys.stderr)
            return 2

        class _Rec:
            """仅承载导出所需字段的轻量容器。"""

        rec = _Rec()
        rec.trajectory = records[0][1]
        rec.day = records[0][0]
        dest = run_dir / "result"
        path = export_result1(rec, dest)
        print(f"已写出 {path}")
        return 0

    # 多日结果：重建账单后导出
    from .export import DailyExportRow
    from .settlement import settle_trajectory

    rows: list[DailyExportRow] = []
    for day, traj, plan_initial, final_plan in records:
        bill = settle_trajectory(traj, initial_plan_kwh=plan_initial)
        rows.append(
            DailyExportRow(
                day=day,
                plan_initial_kwh=plan_initial,
                final_plan_kwh=final_plan,
                grid_actual_kwh=traj.grid_actual_kwh,
                emergency_actual_kwh=traj.emergency_actual_kwh,
                charge_stored_kwh=traj.charge_stored_kwh,
                discharge_delivered_kwh=traj.discharge_delivered_kwh,
                soc_start_kwh=float(traj.soc_kwh[0]),
                soc_end_kwh=float(traj.soc_kwh[-1]),
                purchase_cost_yuan=float(bill.purchase_cost_yuan),
                penalty_yuan=float(bill.penalty_yuan),
            )
        )

    dest = run_dir / "result"
    path = export_multiday(
        args.problem,
        rows,
        dest,
        with_adjust_sheet=args.problem in ("problem3", "problem4-3"),
    )
    print(f"已写出 {path}（{len(rows)} 天）")

    # 论文用表
    specials = [r for r in rows if r.day.isoformat() in SPECIAL_DATES]
    tables = {
        "paper_table1.csv": paper_table1(specials),
        "paper_table2.csv": paper_table2(specials),
        "paper_table3.csv": paper_table3(specials),
    }
    for name, tbl in tables.items():
        p = run_dir / name
        keys = sorted({k for row in tbl for k in row})
        lines = [",".join(keys)]
        for row in tbl:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
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
