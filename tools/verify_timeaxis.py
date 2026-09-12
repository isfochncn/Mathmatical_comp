"""独立复核：result 行时间轴映射是否与物理轨迹一致。

不依赖被测代码的映射函数，只用最朴素的分钟算术重新推一遍：
  result 行第 j 列 -> 区间起点 = 当日 00:00 + (j+1)*10 分钟
  该行覆盖 [当日 00:10, 次日 00:10)，共 1440 分钟 = 144 段
自然日时钟第 t 段 -> 当日 00:00 + t*10 分钟
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

RUN = sys.argv[1] if len(sys.argv) > 1 else "out/problem2"
DAY0 = date(2025, 1, 1)

arrays = dict(np.load(Path(RUN) / "trajectory.npz", allow_pickle=False))
_raw = json.loads((Path(RUN) / "summary.json").read_text(encoding="utf-8"))
# save_run nests the metric dict under "summary" and keeps the per-window bills
# at the top level; accept a flat file too so the checker works on both shapes.
summary = _raw.get("summary", _raw)
bills = _raw

minutes = arrays["abs_minute"].astype(np.int64)
by_minute = {int(m): i for i, m in enumerate(minutes)}
n = minutes.size
fails = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def idx(day: date, clock_t: int):
    return by_minute.get((day - DAY0).days * 1440 + clock_t * 10)


print("=" * 78)
print(f"复核对象：{RUN}")
print(f"轨迹区间数 {n}（{minutes[0]} .. {minutes[-1]} 分钟），"
      f"= {(minutes[-1] - minutes[0]) // 10 + 1} 段")
covered_first = DAY0 + timedelta(days=int(minutes[0]) // 1440)
covered_last = DAY0 + timedelta(days=int(minutes[-1]) // 1440)
print(f"覆盖日期：{covered_first} .. {covered_last}")
print("=" * 78)

# Pick a probe day that the trajectory actually covers, preferring a paper date
# when one is available.
_probe_pref = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]
DAY = next(
    (d for d in _probe_pref if covered_first <= d and d + timedelta(days=1) <= covered_last),
    covered_first,
)
print(f"\n1) result 行 j 与自然日时钟 t 的对应（探针日期 {DAY}）")
row = [idx(DAY, j + 1) for j in range(144)]
have = [x for x in row if x is not None]
check(len(have) == 144, f"{DAY} 的 144 个 result 单元格在轨迹中都有对应区间（实到 {len(have)}）")
check(row[0] == idx(DAY, 1), "j=0 -> 当日时钟第 1 段（00:10-00:20）")
check(row[142] == idx(DAY, 143), "j=142 -> 当日时钟第 143 段（23:50-24:00）")
check(len(have) == 144 and row[143] == idx(DAY + timedelta(days=1), 0),
      "j=143 -> 次日时钟第 0 段（00:00-00:10），即跨入次日")
# Compare the ABSOLUTE MINUTES, not the trajectory indices: consecutive indices
# differ by 1 while consecutive intervals differ by 10 minutes.
if len(have) == 144:
    mins_of_row = [int(minutes[i]) for i in have]
    check(all(mins_of_row[k + 1] - mins_of_row[k] == 10 for k in range(142)),
          "j=0..142 在当日内连续、步长 10 分钟")
    check(mins_of_row[143] - mins_of_row[142] == 10,
          "j=142 -> j=143 仍连续 10 分钟（当日 23:50-0:00 与次日 0:00-0:10 首尾相接）")
    check(mins_of_row[143] - mins_of_row[0] == 1430,
          f"首末单元格起点相隔 1430 分钟（实测 {mins_of_row[143] - mins_of_row[0]}）")
    check(mins_of_row[0] == idx(DAY, 1) * 0 + (DAY - DAY0).days * 1440 + 10,
          "行首 = 当日 clock 1（00:10）")
    check(mins_of_row[-1] + 10 == (DAY + timedelta(days=1) - DAY0).days * 1440 + 10,
          "行尾 = 次日 clock 1（次日 00:10），即行覆盖 [当日 00:10, 次日 00:10)")
    check(mins_of_row[-1] + 10 - mins_of_row[0] == 1440,
          f"result 行实际跨度 1440 分钟（实测 {mins_of_row[-1] + 10 - mins_of_row[0]}）")

print("\n2) result 行不得与自然日窗口重合（两者本来就是不同窗口）")
row_minutes = set(row)
day_minutes = {idx(DAY, t) for t in range(144)}
check(row_minutes != day_minutes, "result 行与自然日时钟是不同区间集合")
check(idx(DAY, 0) not in row_minutes, "自然日首段（00:00-00:10）不属于当日 result 行")
check(idx(DAY + timedelta(days=1), 0) in row_minutes, "次日首段属于当日 result 行")
print(f"       重合段数 {len(row_minutes & day_minutes)} / 144")

print("\n3) 功率平衡逐段复核")
E = np.asarray(arrays["soc_boundary_kwh"], dtype=np.float64)
g = np.asarray(arrays["grid_kwh"], dtype=np.float64)
u = np.asarray(arrays["emergency_kwh"], dtype=np.float64)
ch = np.asarray(arrays["charge_kwh"], dtype=np.float64)
dis = np.asarray(arrays["discharge_kwh"], dtype=np.float64)
cur = np.asarray(arrays["curtail_kwh"], dtype=np.float64)
sur = np.asarray(arrays["surplus_kwh"], dtype=np.float64)

# 需求与光伏由附件重建（实测口径，执行层用的就是这条时间轴）
from microgrid.data_io import load_all
from microgrid.timeline import build_timeline

bundle = load_all()
tl = build_timeline(bundle)
D = np.array([float(tl.demand_kwh.value_at(int(m))) for m in minutes])
P = np.array([float(tl.pv_kwh.value_at(int(m))) for m in minutes])

# 实际执行必须逐段守恒；弃光只能来自该段真实光伏。
resid = g + u + P - cur + dis - D - ch / 0.9 - sur
check(np.isfinite(resid).all() and float(np.max(np.abs(resid))) < 1e-6,
      f"真实母线逐段守恒（最大残差 {np.max(np.abs(resid)):.3e} kWh）")
check(bool(np.all((sur >= -1e-6) & (sur <= g+1e-6))), "弃购电非负且不超过已付费普通购电")
minimum_spill = np.maximum(g-D-np.minimum(750, np.maximum(10800-E[:-1], 0))/.9, 0)
check(np.allclose(sur, minimum_spill, atol=2e-6, rtol=0), "逐段弃购电已降至不可避免量")
check(bool(np.all((cur >= -1e-6) & (cur <= P + 1e-6))), "逐段弃光不超过真实光伏")

soc_resid = E[1:] - (E[:-1] + ch - dis / 0.9)
check(float(np.max(np.abs(soc_resid))) < 1e-6,
      f"E' = E + 充电 - 放电/0.9（最大残差 {np.max(np.abs(soc_resid)):.3e} kWh）")
check(float(E.min()) >= 1200 - 1e-6 and float(E.max()) <= 10800 + 1e-6,
      f"SOC 始终在 [1200, 10800]（实际 {E.min():.1f} .. {E.max():.1f}）")
check(float(ch.max()) <= 750 + 1e-6 and float(dis.max()) <= 750 + 1e-6,
      f"充放电功率不超过 750 kWh/段（实际 {ch.max():.1f} / {dis.max():.1f}）")
check(float(cur.max()) <= float(P.max()) + 1e-6, "弃光不超过可用光伏")

print("\n4) 真实物理损耗恒等式")
lhs = float(np.sum(g + u))
rhs = float(np.sum(D - P + cur) + (19.0/90.0)*np.sum(ch) + 0.9*(E[-1]-E[0]) + np.sum(sur))
check(abs(lhs-rhs) < max(1e-5, n*1e-8), f"真实能量总账残差 {lhs-rhs:+.6f} kWh")

print("\n5) result 行费用与 summary 记录一致")
row_bills = {r["date"]: r for r in bills.get("result_row_bills", [])}
day_bills = {r["date"]: r for r in bills.get("natural_day_bills", [])}
probed = 0
for d in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
    if d not in row_bills:
        print(f"       {d} 不在记录中（该次运行未覆盖）")
        continue
    dd = date.fromisoformat(d)
    cells = [idx(dd, j + 1) for j in range(144)]
    if any(i is None for i in cells):
        check(False, f"{d} 已有账单但执行轨迹缺段")
        continue
    q = sum(float(g[i] + u[i]) for i in cells)
    probed += 1
    print(f"       {d}: 144 格购电合计 {q:,.3f} kWh | 记录 {row_bills[d]['total_kwh']:,.3f} kWh"
          f" | 自然日窗口 {day_bills[d]['total_kwh']:,.3f} kWh")
    check(abs(q - row_bills[d]["total_kwh"]) < 1e-3, f"{d} result 行合计与记录一致")
if probed == 0:
    print("       该次运行未完整覆盖任何论文日期，本项不适用（全年运行会覆盖）")

print("\n6) 验证版本与有限值")
check(_raw.get("validation_version") == "main-model-v4-paid-spill", "产物经过当前主模型校验")
check(all(np.isfinite(v).all() for v in (g, u, ch, dis, cur, sur, E, D, P)), "物理量均为有限值")

print()
print("=" * 78)
print("全部通过" if not fails else f"{len(fails)} 项失败")
for f in fails:
    print("  -", f)
print("=" * 78)
sys.exit(0 if not fails else 1)
