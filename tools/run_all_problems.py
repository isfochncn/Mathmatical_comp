"""Run the four rolling problems over the full year and export their workbooks.

Sequential on purpose: each run is a continuous multi-day simulation, and serial
execution keeps the log readable and the memory profile flat. Wall time is
dominated by the number of window LP solves, so the CLI reports it per run.

Usage (from the project root):

    $env:PYTHONPATH="src"; python tools/run_all_problems.py

Options are intentionally few; anything problem-specific belongs in RunConfig.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")

from microgrid.absolute_run import RunConfig, run, save_run  # noqa: E402
from microgrid.cli import build_parser, cmd_export  # noqa: E402

PROBLEMS = ("problem2", "problem3", "problem4-2", "problem4-3")
RESULTS = {
    "problem2": "result2.xlsx",
    "problem3": "result3.xlsx",
    "problem4-2": "result4-2.xlsx",
    "problem4-3": "result4-3.xlsx",
}


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(["run", "--problem", "problem2", *argv])
    out_dir = Path(getattr(args, "out", "out"))
    run_from = getattr(args, "run_from", None)
    run_to = getattr(args, "run_to", None)

    print("=" * 78)
    print("全年运行：p2 / p3 / p4-2 / p4-3（顺序执行）")
    print(f"口径：预测主线 {args.load_method}，窗口重算周期 {args.plan_refresh_intervals} 段，"
          f"可消纳裕度 {args.absorption_safety_kwh:.0f} kWh")
    if run_from or run_to:
        print(f"日期区间（冒烟测试）：{run_from or '全年起点'} .. {run_to or '全年终点'}")
    print("=" * 78)
    sys.stdout.flush()

    for problem in PROBLEMS:
        config = RunConfig(
            problem=problem,
            out_dir=out_dir,
            history_days=args.history_days,
            experiment=args.experiment,
            allow_spill=not args.strict_no_spill,
            load_method=args.load_method,
            absorption_safety_kwh=args.absorption_safety_kwh,
            plan_refresh_intervals=args.plan_refresh_intervals,
            max_infeasible_intervals=args.max_infeasible_intervals,
            solver_name=args.solver,
            run_from=date.fromisoformat(run_from) if run_from else None,
            run_to=date.fromisoformat(run_to) if run_to else None,
            progress_every_days=getattr(args, "progress_every_days", 0),
        )
        t0 = time.perf_counter()
        print(f"\n--- {problem} 开始 {time.strftime('%H:%M:%S')} ---")
        sys.stdout.flush()
        result = run(config)
        run_dir = save_run(result)
        wall = time.perf_counter() - t0
        s = result.summary
        print(
            f"--- {problem} 完成：{wall / 60:.1f} min | "
            f"购电 {s['plan_kwh']:,.0f} + 增购 {s['add_kwh']:,.0f} + 紧急 {s['emergency_kwh']:,.0f} kWh | "
            f"费用 {s['total_cost_yuan']:,.0f} 元 | "
            f"违约 {s['reduce_cost_yuan']:,.0f} 元 | "
            f"SOC 末 {s['soc_end_kwh']:,.0f} | "
            f"不可行区间 {len(s['infeasible_intervals'])} | "
            f"富余 {s['surplus_disposed_kwh']:,.0f} kWh ---"
        )
        sys.stdout.flush()
        export_args = parser.parse_args(["export", "--problem", problem, "--out", str(out_dir), "--experiment", args.experiment])
        cmd_export(export_args)
        sys.stdout.flush()

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
