"""独立复算结算：只用 trajectory.npz 与 summary.json，不依赖被测结算代码。

对 problem2 / problem4-2（增购恒为 0，grid_kwh == O^exec）可以完整复算四类标签：

    plan_kwh    = Σ grid_kwh
    emergency   = Σ emergency_kwh
    plan_cost   = Σ price_actual * grid_kwh
    emergency_cost = Σ 5 * price_actual * emergency_kwh
    total_cost  = plan_cost + emergency_cost + reduce_cost

对 problem3 / problem4-3，grid_kwh = O^exec + A^exec 且两者费率不同（A 为 1.5 倍），
npz 未保存拆分，只能复算总量与相加上界，脚本会如实说明而不是硬凑。
"""

import json
import sys
from pathlib import Path

import numpy as np

run_dir = Path(sys.argv[1])
npz = np.load(run_dir / "trajectory.npz", allow_pickle=True)
raw = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
S = raw["summary"]
problem = S["problem"]

# Output window: the natural-day clock start of 2025-02-01, i.e. 31 days after
# 2025-01-01 00:00, expressed in ABSOLUTE MINUTES (144 intervals/day x 10 min).
# Falling back to the first interval actually present when a run starts later.
# Do NOT use the trajectory start: a smoke run may begin in January or earlier.
OUT_START_ABS = 144 * 10 * 31  # = 44640
first_abs = int(npz["abs_minute"][0])
out_start = OUT_START_ABS if first_abs <= OUT_START_ABS else first_abs
keep = npz["abs_minute"] >= out_start

g = npz["grid_kwh"][keep]
u = npz["emergency_kwh"][keep]
p = npz["price_actual"][keep]
n = int(keep.sum())

# Problem 3 / 4-3 save the O/A split. Without it the labels cannot be verified:
# grid_kwh is O + A and the two carry different rates (A is charged 1.5x), so
# summing grid_kwh would overstate plan_kwh and understate the adjustment fee.
has_split = "plan_exec_kwh" in npz and "add_exec_kwh" in npz
if has_split:
    o = npz["plan_exec_kwh"][keep]
    a = npz["add_exec_kwh"][keep]
else:
    o = g
    a = np.zeros_like(g)

print("=" * 78)
print(f"独立复算结算：{problem}（{run_dir}，输出窗口 {n} 段）")
print("=" * 78)
print(f"  轨迹 abs_minute {int(npz['abs_minute'][0])} .. {int(npz['abs_minute'][-1])}"
      f"（共 {npz['abs_minute'].size} 段）；输出窗口起点 {out_start}，保留 {n} 段")
print(f"  保留段 abs_minute {int(npz['abs_minute'][keep][0])} .. "
      f"{int(npz['abs_minute'][keep][-1])}")

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


plan_kwh = float(o.sum())
add_kwh = float(a.sum())
emg_kwh = float(u.sum())
plan_cost = float(np.sum(p * o))
add_cost = float(np.sum(1.5 * p * a))
emg_cost = float(np.sum(5.0 * p * u))
reduce_cost = float(S["reduce_cost_yuan"])

print(f"  区间段数            {n}")
print(f"  ΣO^exec（计划购电） {plan_kwh:>18,.6f} kWh")
print(f"  ΣA^exec（调整增购） {add_kwh:>18,.6f} kWh")
print(f"  Σ紧急购电           {emg_kwh:>18,.6f} kWh")
print(f"  Σ价格×O             {plan_cost:>18,.6f} 元")
print(f"  1.5×Σ价格×A         {add_cost:>18,.6f} 元")
print(f"  5×Σ价格×紧急        {emg_cost:>18,.6f} 元")
print(f"  违约费用（记录值）  {reduce_cost:>18,.6f} 元")
if not has_split:
    print("  注：轨迹未保存 O/A 拆分，本问按 A ≡ 0 处理（仅对 problem2 / 4-2 成立）。")
print()

rel = 1e-9
check(abs(plan_kwh - float(S["plan_kwh"])) <= rel * max(1.0, abs(plan_kwh)),
      f"plan_kwh 一致：{plan_kwh:,.6f} vs {float(S['plan_kwh']):,.6f}")
check(abs(add_kwh - float(S["add_kwh"])) <= rel * max(1.0, abs(add_kwh)),
      f"add_kwh 一致：{add_kwh:,.6f} vs {float(S['add_kwh']):,.6f}")
check(abs(emg_kwh - float(S["emergency_kwh"])) <= rel * max(1.0, abs(emg_kwh)),
      f"emergency_kwh 一致：{emg_kwh:,.6f} vs {float(S['emergency_kwh']):,.6f}")
check(abs(plan_cost - float(S["plan_cost_yuan"])) <= 1e-6 * max(1.0, abs(plan_cost)),
      f"plan_cost_yuan 一致：{plan_cost:,.6f} vs {float(S['plan_cost_yuan']):,.6f}")
check(abs(add_cost - float(S["add_cost_yuan"])) <= 1e-6 * max(1.0, abs(add_cost)),
      f"add_cost_yuan 一致：{add_cost:,.6f} vs {float(S['add_cost_yuan']):,.6f}")
check(abs(emg_cost - float(S["emergency_cost_yuan"])) <= 1e-6 * max(1.0, abs(emg_cost)),
      f"emergency_cost_yuan 一致：{emg_cost:,.6f} vs {float(S['emergency_cost_yuan']):,.6f}")
check(
    abs((plan_cost + add_cost + emg_cost + reduce_cost) - float(S["total_cost_yuan"]))
    <= 1e-6 * max(1.0, abs(float(S["total_cost_yuan"]))),
    f"总费用 = 计划费 + 调整费 + 紧急费 + 违约费："
    f"{plan_cost + add_cost + emg_cost + reduce_cost:,.6f} vs {float(S['total_cost_yuan']):,.6f}",
)
check(
    abs((plan_kwh + add_kwh + emg_kwh) - float(S["total_purchased_kwh"]))
    <= rel * max(1.0, plan_kwh),
    f"总购电量一致：{plan_kwh + add_kwh + emg_kwh:,.6f} vs "
    f"{float(S['total_purchased_kwh']):,.6f}",
)
check(
    abs(float(np.max(np.abs(o + a - g)))) < 1e-9,
    f"O^exec + A^exec = grid（最大差 {np.max(np.abs(o + a - g)):.3e}）",
)

print()
print("全部通过" if not fails else f"{len(fails)} 项失败")
for f in fails:
    print("  -", f)
sys.exit(0 if not fails else 1)
