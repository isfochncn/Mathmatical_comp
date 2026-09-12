"""比较事项 A1：计划重算周期敏感性。

规范写"每个区间初重算"，主模型逐段重算，其他周期仅用于比较。本脚本在同一区间上
跑不同重算周期并对比，量化该技术近似的影响。执行反馈本身始终逐 10 分钟，不随此参数变化。

用法（项目根目录）::

    $env:PYTHONIOENCODING="utf-8"
    python -u tools/compare_a1.py --problem problem3 --run-from 2025-02-01 --run-to 2025-03-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")

from microgrid.absolute_run import RunConfig, run, save_run  # noqa: E402

#: 重算周期（单位：10 分钟段）。1 = 主模型，6 = 每小时比较方案。
REFRESHES = (6, 1, 2, 12, 144)

METRICS = (
    ("total_purchased_kwh", "总购电量 kWh"),
    ("plan_kwh", "计划购电 kWh"),
    ("add_kwh", "增购 kWh"),
    ("emergency_kwh", "紧急购电 kWh"),
    ("total_cost_yuan", "总费用 元"),
    ("reduce_cost_yuan", "违约费用 元"),
    ("surplus_disposed_kwh", "富余处置 kWh"),
    ("window_solves", "窗口求解次数"),
    ("soc_end_kwh", "期末 SOC kWh"),
)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", default="problem3")
    ap.add_argument("--out", default="out")
    ap.add_argument("--run-from", default="2025-02-01")
    ap.add_argument("--run-to", default="2025-03-31")
    ap.add_argument("--refreshes", default=",".join(str(r) for r in REFRESHES))
    ap.add_argument("--progress-every-days", type=int, default=20)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    run_from = date.fromisoformat(args.run_from)
    run_to = date.fromisoformat(args.run_to)
    refreshes = [int(x) for x in args.refreshes.split(",") if x.strip()]

    results: dict[int, dict[str, object]] = {}
    walls: dict[int, float] = {}

    for refresh in refreshes:
        arm_dir = out_dir / f"_a1_r{refresh}"
        cfg = RunConfig(
            problem=args.problem,
            out_dir=arm_dir,
            run_from=run_from,
            run_to=run_to,
            plan_refresh_intervals=refresh,
            progress_every_days=args.progress_every_days,
            max_infeasible_intervals=0,
            experiment="comparison",
        )
        print(f"\n>>> 重算周期 {refresh} 段（{refresh * 10} 分钟） -> {cfg.problem_dir()}", flush=True)
        t0 = time.perf_counter()
        res = run(cfg)
        wall = time.perf_counter() - t0
        save_run(res)
        results[refresh] = res.summary
        walls[refresh] = wall
        print(f"<<< 完成 {wall / 60:.1f} min，总费用 {res.summary['total_cost_yuan']:,.0f} 元", flush=True)

    print()
    print("=" * 100)
    print(f"比较事项 A1：计划重算周期敏感性（{args.problem}，{run_from} .. {run_to}）")
    print("  主模型 = 每段（1）；其余重算周期仅为对照实验")
    print("=" * 100)
    header = f"{'指标':<20}" + "".join(f"{('每' + str(r) + '段'):>14}" for r in refreshes)
    print(header)
    print("-" * 100)
    for key, name in METRICS:
        row = f"{name:<20}"
        for r in refreshes:
            v = float(results[r].get(key, float("nan")))
            row += f"{v:>14,.1f}"
        print(row)
    row = f"{'耗时 min':<20}"
    for r in refreshes:
        row += f"{walls[r] / 60:>14.1f}"
    print(row)
    row = f"{'不可行区间':<20}"
    for r in refreshes:
        row += f"{len(results[r].get('infeasible_intervals', [])):>14}"
    print(row)
    print("-" * 100)

    base = results[1]["total_cost_yuan"] if 1 in results else None
    if base:
        print("相对主模型（每 1 段）的总费用偏差：")
        for r in refreshes:
            if r == 1:
                continue
            d = float(results[r]["total_cost_yuan"]) - float(base)
            print(f"  每 {r:>3} 段: {d:>+14,.1f} 元（{d / float(base) * 100:+.3f}%）")

    out_path = out_dir / f"a1_comparison_{args.problem}.json"
    out_path.write_text(
        json.dumps(
            {
                "problem": args.problem,
                "run_from": args.run_from,
                "run_to": args.run_to,
                "summary": {str(k): v for k, v in results.items()},
                "wall_seconds": {str(k): v for k, v in walls.items()},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
