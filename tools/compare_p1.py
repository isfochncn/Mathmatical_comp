"""Comparison item P1: main forecast line vs. the same-weekday variant.

The 2026-09-11 spec fixes the **28-day same-clock mean** as the main line for the
load forecast. The memo also asks for a controlled comparison against using the
same weekday one week earlier. This script runs both settings over exactly the
same date window and prints the differences that matter for the paper.

It never overwrites the deliverable run directories: the P1 arm is written to
``<out>/_p1_<problem>`` so that ``tools/verify_outputs.py`` keeps checking the
real artifacts.

Usage (from the project root)::

    $env:PYTHONIOENCODING="utf-8"
    python -u tools/compare_p1.py --problem problem2 --out out
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

#: The headline figures of the comparison, with the direction that means "worse".
METRICS = (
    ("total_purchased_kwh", "总购电量 kWh", "neutral"),
    ("plan_kwh", "计划购电 kWh", "neutral"),
    ("add_kwh", "增购 kWh", "lower"),
    ("emergency_kwh", "紧急购电 kWh", "lower"),
    ("total_cost_yuan", "总费用 元", "lower"),
    ("reduce_cost_yuan", "违约费用 元", "lower"),
    ("surplus_disposed_kwh", "富余处置 kWh", "lower"),
    ("window_solves", "窗口求解次数", "neutral"),
    ("soc_end_kwh", "期末 SOC kWh", "higher"),
)


def run_arm(
    problem: str,
    load_method: str,
    out_dir: Path,
    run_from: date | None,
    run_to: date | None,
    progress_every_days: int,
) -> tuple[dict[str, object], float, list[str]]:
    label = "main" if load_method == "same_clock_mean" else "P1-same_weekday"
    arm_dir = out_dir if load_method == "same_clock_mean" else out_dir / f"_p1_{problem}"
    config = RunConfig(
        problem=problem,
        out_dir=arm_dir,
        load_method=load_method,
        run_from=run_from,
        run_to=run_to,
        progress_every_days=progress_every_days,
        max_infeasible_intervals=0,
        experiment="main" if load_method == "same_clock_mean" else "comparison",
    )
    print(f"\n>>> [{label}] load_method={load_method} -> {config.problem_dir()}", flush=True)
    t0 = time.perf_counter()
    result = run(config)
    wall = time.perf_counter() - t0
    if load_method != "same_clock_mean":
        save_run(result)
    print(f"<<< [{label}] 完成 {wall / 60:.1f} min", flush=True)
    return result.summary, wall, list(result.notes)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", default="problem2")
    ap.add_argument("--out", default="out")
    ap.add_argument("--run-from", default=None, help="YYYY-MM-DD；缺省=2025-01-01")
    ap.add_argument("--run-to", default=None, help="YYYY-MM-DD；缺省=2025-12-31")
    ap.add_argument("--progress-every-days", type=int, default=10)
    ap.add_argument("--skip-main", action="store_true", help="只跑 P1 臂（主线已在 out/<problem>）")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    run_from = date.fromisoformat(args.run_from) if args.run_from else None
    run_to = date.fromisoformat(args.run_to) if args.run_to else None

    arms: dict[str, dict[str, object]] = {}
    walls: dict[str, float] = {}

    if args.skip_main:
        summary_path = out_dir / args.problem / "summary.json"
        if not summary_path.exists():
            print(f"找不到 {summary_path}；先去跑主线或去掉 --skip-main", file=sys.stderr)
            return 2
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("validation_version") != "main-model-v4-paid-spill":
            raise ValueError("Cannot compare against an obsolete model run")
        saved_config = json.loads((summary_path.parent / "config.json").read_text(encoding="utf-8"))
        if saved_config.get("run_from") != args.run_from or saved_config.get("run_to") != args.run_to:
            raise ValueError("Comparison arms must use the same reporting dates")
        arms["main"] = payload["summary"]
        walls["main"] = float(payload.get("wall_seconds", float("nan")))
        print(f">>> [main] 复用已有 {summary_path}", flush=True)
    else:
        s, w, _ = run_arm(
            args.problem, "same_clock_mean", out_dir, run_from, run_to, args.progress_every_days
        )
        arms["main"], walls["main"] = s, w

    s, w, _ = run_arm(
        args.problem, "same_weekday", out_dir, run_from, run_to, args.progress_every_days
    )
    arms["P1"], walls["P1"] = s, w

    print()
    print("=" * 88)
    print(f"比较事项 P1：负载预测方法对照（{args.problem}）")
    print("  主线 = 28 日同钟点均值（2026-09-11 定稿口径）")
    print("  P1   = 上周同一星期几")
    print("=" * 88)
    print(f"{'指标':<22}{'主线':>20}{'P1':>20}{'差异':>16}{'相对':>10}")
    print("-" * 88)
    for key, name, _direction in METRICS:
        a = float(arms["main"].get(key, float("nan")))
        b = float(arms["P1"].get(key, float("nan")))
        rel = (b - a) / a * 100.0 if a else float("nan")
        print(f"{name:<22}{a:>20,.3f}{b:>20,.3f}{b - a:>+16,.3f}{rel:>9.2f}%")
    print("-" * 88)
    def _mins(x: float) -> str:
        return "n/a" if x != x else f"{x / 60:.1f}"  # NaN-safe

    print(f"{'运行耗时 min':<22}{_mins(walls['main']):>20}{_mins(walls['P1']):>20}")
    print(
        f"{'不可行区间数':<22}{len(arms['main'].get('infeasible_intervals', [])):>20}"
        f"{len(arms['P1'].get('infeasible_intervals', [])):>20}"
    )
    a_notes = arms["main"].get("notes") or []
    b_notes = arms["P1"].get("notes") or []
    if a_notes or b_notes:
        print("\n运行备注（前 6 条）：")
        for tag, notes in (("主线", a_notes), ("P1", b_notes)):
            for n in list(notes)[:6]:
                print(f"  [{tag}] {n}")

    out_path = out_dir / f"p1_comparison_{args.problem}.json"
    out_path.write_text(
        json.dumps(
            {"problem": args.problem, "main": arms["main"], "p1": arms["P1"],
             "wall_seconds": walls},
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
