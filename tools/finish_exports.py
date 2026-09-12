"""四问跑完后：导出结果文件 → 跑全部复核。

用法（项目根目录）::

    $env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="src"
    python -u tools/finish_exports.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ("problem2", "problem3", "problem4-2", "problem4-3")
OUT = ROOT / "out"


def need_run_done(problem: str) -> bool:
    """A problem is done when trajectory.npz exists and still lists O/A arrays."""
    p = OUT / problem / "trajectory.npz"
    return p.exists() and p.stat().st_size > 10_000


def main() -> int:
    t0 = time.perf_counter()
    py = sys.executable

    print("=" * 78)
    print("① 导出四份结果文件")
    print("=" * 78)
    exported = []
    for problem in PROBLEMS:
        if not need_run_done(problem):
            print(f"  -- 跳过 {problem}：轨迹尚未就绪")
            continue
        r = subprocess.run(
            [py, "-u", "-m", "microgrid.cli", "export", "--problem", problem, "--out", str(OUT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tail = (r.stdout or "").strip().splitlines()
        print(f"  [{problem}] rc={r.returncode}")
        for line in tail[-6:]:
            print("      " + line)
        if r.returncode != 0:
            print("      stderr:", (r.stderr or "").strip()[-800:])
        else:
            exported.append(problem)

    print()
    print("=" * 78)
    print("② 复核：交付文件结构 / 时间轴 / 数值")
    print("=" * 78)
    failed: list[str] = []
    checks: list[tuple[str, list[str]]] = [("tools/verify_outputs.py", [])]
    for p in exported:
        checks.append(("tools/verify_settlement.py", [str(OUT / p)]))
        checks.append(("tools/verify_timeaxis.py", [str(OUT / p)]))
    for script, extra in checks:
        cmd = [py, "-u", script, *extra]
        r = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        label = f"{script} {Path(extra[0]).name if extra else ''}".strip()
        status = "OK" if r.returncode == 0 else f"FAIL(rc={r.returncode})"
        print(f"\n--- {label}: {status} ---")
        lines = (r.stdout or "").strip().splitlines()
        # Show any FAIL line plus the verdict tail.
        for line in [ln for ln in lines if "FAIL" in ln]:
            print("   " + line)
        for line in lines[-6:]:
            print("   " + line)
        if r.returncode != 0:
            failed.append(label)
            err = (r.stderr or "").strip()
            if err:
                print("   stderr:", err[-600:])

    print()
    print("=" * 78)
    print("③ 结果摘要")
    print("=" * 78)
    for problem in PROBLEMS:
        sp = OUT / problem / "summary.json"
        if not sp.exists():
            continue
        raw = json.loads(sp.read_text(encoding="utf-8"))
        s = raw["summary"]
        print(
            f"  {problem:<12} 购电 {s['total_purchased_kwh']:>14,.0f} kWh | "
            f"费用 {s['total_cost_yuan']:>14,.0f} 元 | "
            f"违约 {s['reduce_cost_yuan']:>10,.0f} 元 | "
            f"窗口求解 {s['window_solves']:>5} | "
            f"不可行 {len(s['infeasible_intervals']):>4} | "
            f"富余 {s['surplus_disposed_kwh']:>12,.0f} kWh | "
            f"末 SOC {s['soc_end_kwh']:,.0f}"
        )

    print()
    print("=" * 78)
    print(f"全部完成，用时 {(time.perf_counter() - t0) / 60:.1f} min")
    print("复核全部通过" if not failed else f"复核失败：{failed}")
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
