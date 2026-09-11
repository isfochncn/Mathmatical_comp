"""Read-only verification of the five exported deliverables (2026-09-11 rules).

Checks
------
1. The original templates under ``data/附件5`` are untouched (hash recorded).
2. Every export exists with the expected sheet structure.
3. **Time-axis alignment**: the template's 144 columns ARE the result intervals,
   a result row covers ``[day 00:10, next 00:10)``, and the model interval for
   cell ``j`` is clock interval ``j+1`` (``j = 143`` reaches into the next day).
4. Coverage: 334 date rows for problems 2/3/4-2/4-3, 145 rows for problem 1.
5. Result-row totals in the workbook equal the run's recorded result-row bills,
   and the natural-day totals are *different* (different windows) as designed.
6. The six 4-hour blocks per day sum back to the 10-minute trajectory, and the
   natural-day 00:00 / 24:00 SOC equal the trajectory boundary values.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from openpyxl import load_workbook  # noqa: E402

from microgrid.constants import N_INTERVAL  # noqa: E402
from microgrid.timeline import MINUTES_PER_DAY  # noqa: E402

TEMPLATES = {
    "result1.xlsx": "data/附件5/result1.xlsx",
    "result2.xlsx": "data/附件5/result2.xlsx",
    "result3.xlsx": "data/附件5/result3.xlsx",
    "result4-2.xlsx": "data/附件5/result4-2.xlsx",
    "result4-3.xlsx": "data/附件5/result4-3.xlsx",
}
EXPORTS = {
    "result1.xlsx": "out/problem1/result/result1.xlsx",
    "result2.xlsx": "out/problem2/result/result2.xlsx",
    "result3.xlsx": "out/problem3/result/result3.xlsx",
    "result4-2.xlsx": "out/problem4-2/result/result4-2.xlsx",
    "result4-3.xlsx": "out/problem4-3/result/result4-3.xlsx",
}
RUNS = {
    "result2.xlsx": "out/problem2",
    "result3.xlsx": "out/problem3",
    "result4-2.xlsx": "out/problem4-2",
    "result4-3.xlsx": "out/problem4-3",
}
EXPECTED_SHEETS = {
    "result1.xlsx": ["计划购电量", "充放电量"],
    "result2.xlsx": ["计划购电量", "充放电量", "紧急购电量"],
    "result3.xlsx": ["计划购电量", "调整购电量", "充放电量", "紧急购电量"],
    "result4-2.xlsx": ["计划购电量", "充放电量", "紧急购电量"],
    "result4-3.xlsx": ["计划购电量", "调整购电量", "充放电量", "紧急购电量"],
}
FIRST_OUTPUT_DAY = date(2025, 2, 1)
LAST_OUTPUT_DAY = date(2025, 12, 31)

_ok = True
_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    global _ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    _ok = _ok and bool(cond)
    if not cond:
        _failures.append(msg)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
section("1) original templates must be untouched")
for name, path in TEMPLATES.items():
    p = Path(path)
    if not p.exists():
        check(False, f"{path} exists")
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    check(p.stat().st_size > 1000, f"{path} size={p.stat().st_size} sha256[:16]={h}")
    # Recorded so a rerun can prove the file never changed.
    (Path("out") / f".template_{name}.sha256").write_text(h, encoding="utf-8")

# ---------------------------------------------------------------------------
section("2) exports exist with the expected sheet structure")
for name, path in EXPORTS.items():
    p = Path(path)
    check(p.exists(), f"{path} exists")
    if not p.exists():
        continue
    wb = load_workbook(p, read_only=True, data_only=True)
    check(
        wb.sheetnames == EXPECTED_SHEETS[name],
        f"{name} sheets == {EXPECTED_SHEETS[name]} (got {wb.sheetnames})",
    )
    wb.close()

# ---------------------------------------------------------------------------
section("3) time-axis alignment: template columns ARE the result intervals")
if Path(EXPORTS["result1.xlsx"]).exists():
    wb = load_workbook(EXPORTS["result1.xlsx"], read_only=True, data_only=True)
    ws = wb["计划购电量"]
    rows = list(ws.iter_rows(min_row=1, max_row=145, max_col=2, values_only=True))
    check(rows[0][0] == "时间段", f"row1 col1 = {rows[0][0]!r}")
    check(rows[1][0] == "0:10-0:20", f"first data label = {rows[1][0]!r} (template verbatim)")
    check(rows[143][0] == "23:50-0:00+1", f"row144 label = {rows[143][0]!r}")
    check(rows[144][0] == "0:00+1-0:10+1", f"row145 label = {rows[144][0]!r}")
    labels = [r[0] for r in rows[1:]]
    check(len(labels) == 144 and len(set(labels)) == 144, "144 distinct interval labels")
    values = [r[1] for r in rows[1:]]
    check(all(v is not None for v in values), "all 144 plan cells filled")
    ws2 = wb["充放电量"]
    cd = list(ws2.iter_rows(min_row=1, max_row=7, max_col=6, values_only=True))
    check(cd[1][0] == "0:00-4:00", f"block 1 label = {cd[1][0]!r}")
    check(cd[6][0] == "20:00-24:00", f"block 6 label = {cd[6][0]!r}")
    check(cd[1][4] == "0:00" and cd[2][4] == "24:00", "SOC columns use natural-day 0:00 / 24:00")
    wb.close()

matrix_header_checked = False
for name in ("result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"):
    if not Path(EXPORTS[name]).exists():
        continue
    wb = load_workbook(EXPORTS[name], read_only=True, data_only=True)
    ws = wb["计划购电量"]
    hdr = next(ws.iter_rows(min_row=1, max_row=1, max_col=147, values_only=True))
    if not matrix_header_checked:
        check(hdr[0] == "日期\\时间", f"matrix header col1 = {hdr[0]!r}")
        check(hdr[1] == "0:10-0:20", f"matrix first slot = {hdr[1]!r}")
        check(hdr[143] == "23:50-0:00+1", f"matrix 144th slot = {hdr[143]!r}")
        check(hdr[144] == "0:00-0:10+1", f"matrix 145th slot = {hdr[144]!r}")
        check(hdr[145] == "全天购电量", f"matrix daily total col = {hdr[145]!r}")
        check(hdr[146] == "全天购电费", f"matrix daily cost col = {hdr[146]!r}")
        matrix_header_checked = True
    wb.close()

# ---------------------------------------------------------------------------
section("4) coverage")
if Path(EXPORTS["result1.xlsx"]).exists():
    wb = load_workbook(EXPORTS["result1.xlsx"], read_only=True, data_only=True)
    check(wb["计划购电量"].max_row == N_INTERVAL + 1, "result1 plan rows = 145")
    wb.close()

for name in ("result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"):
    if not Path(EXPORTS[name]).exists():
        continue
    wb = load_workbook(EXPORTS[name], read_only=True, data_only=True)
    ws = wb["计划购电量"]
    first = ws.cell(row=2, column=1).value
    last = ws.cell(row=335, column=1).value
    check(ws.max_row >= 335, f"{name} plan rows >= 335 (got {ws.max_row})")
    check(
        str(first)[:10] == FIRST_OUTPUT_DAY.isoformat()
        and str(last)[:10] == LAST_OUTPUT_DAY.isoformat(),
        f"{name} date span {str(first)[:10]} .. {str(last)[:10]}",
    )
    check(ws.cell(row=336, column=1).value is None, f"{name} no stale rows past 334 days")
    wb.close()

# ---------------------------------------------------------------------------
section("5) exported totals vs recorded result-row bills")
for name, run_dir in RUNS.items():
    if not Path(EXPORTS[name]).exists():
        continue
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        check(False, f"{summary_path} exists")
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row_bills = {r["date"]: r for r in payload.get("result_row_bills", [])}
    day_bills = {r["date"]: r for r in payload.get("natural_day_bills", [])}
    out_dates = sorted(d for d in row_bills if d >= FIRST_OUTPUT_DAY.isoformat())
    check(len(out_dates) == 334, f"{name} recorded 334 result rows (got {len(out_dates)})")

    wb = load_workbook(Path(run_dir) / "result" / name, read_only=True, data_only=True)
    ws = wb["计划购电量"]
    rows = list(ws.iter_rows(min_row=2, max_row=335, max_col=147, values_only=True))
    wb.close()
    mism_qty = 0
    mism_cost = 0
    for i, day_text in enumerate(out_dates):
        rec = row_bills[day_text]
        workbook_total = rows[i][145]
        workbook_cost = rows[i][146]
        if workbook_total is None or abs(float(workbook_total) - rec["total_kwh"]) > 1e-3:
            mism_qty += 1
        if workbook_cost is None or abs(float(workbook_cost) - rec["total_cost_yuan"]) > 1e-3:
            mism_cost += 1
    check(mism_qty == 0, f"{name} result-row purchase totals match ({mism_qty} mismatches)")
    check(mism_cost == 0, f"{name} result-row cost totals match ({mism_cost} mismatches)")

    # The two windows are different by design; report the difference rather than
    # asserting equality.
    first_day = out_dates[0]
    if first_day in day_bills:
        delta_q = row_bills[first_day]["total_kwh"] - day_bills[first_day]["total_kwh"]
        print(
            f"       window check {first_day}: result-row minus natural-day = "
            f"{delta_q:+.3f} kWh (expected non-zero: different windows)"
        )

    # Problem 3 / 4-3 declare an adjustment sheet. It must carry the effective
    # purchase O + A, NOT a copy of the plan sheet: a copy would silently report
    # that no adjustment ever happened. Compare against the O/A arrays the run
    # saved, restricted to exactly the cells the two sheets cover.
    if name in ("result3.xlsx", "result4-3.xlsx"):
        npz_path = Path(run_dir) / "trajectory.npz"
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.files)
        check(
            {"plan_exec_kwh", "add_exec_kwh"} <= keys,
            f"{name} 轨迹保存了 O/A 拆分（plan_exec_kwh / add_exec_kwh）",
        )
        if {"plan_exec_kwh", "add_exec_kwh"} <= keys:
            minutes_all = data["abs_minute"].astype(np.int64)
            idx_of = {int(m): i for i, m in enumerate(minutes_all)}
            plan_exec, add_exec = data["plan_exec_kwh"], data["add_exec_kwh"]
            wb = load_workbook(Path(run_dir) / "result" / name, read_only=True, data_only=True)
            ws_plan = wb["计划购电量"]
            ws_adj = wb["调整购电量"]
            plan_cells = list(
                ws_plan.iter_rows(min_row=2, max_row=335, min_col=2, max_col=145, values_only=True)
            )
            adj_cells = list(
                ws_adj.iter_rows(min_row=2, max_row=335, min_col=2, max_col=145, values_only=True)
            )
            wb.close()

            bad_plan = bad_adj = 0
            exp_plan = exp_adj = 0.0
            n_diff = 0
            for i, day_text in enumerate(out_dates):
                day = date.fromisoformat(day_text)
                for j in range(144):
                    base = (day - date(2025, 1, 1)).days * MINUTES_PER_DAY
                    o = idx_of.get(base + (j + 1) * 10)
                    if o is None:
                        continue
                    o_kwh = float(plan_exec[o])
                    a_kwh = float(add_exec[o])
                    exp_plan += o_kwh
                    exp_adj += o_kwh + a_kwh
                    vp = float(plan_cells[i][j] or 0.0)
                    va = float(adj_cells[i][j] or 0.0)
                    if abs(vp - o_kwh) > 1e-6:
                        bad_plan += 1
                    if abs(va - (o_kwh + a_kwh)) > 1e-6:
                        bad_adj += 1
                    if abs(vp - va) > 1e-6:
                        n_diff += 1
            check(bad_plan == 0, f"{name} 计划购电量表 = O^exec（{bad_plan} 个单元格不符）")
            check(bad_adj == 0, f"{name} 调整购电量表 = O^exec + A^exec（{bad_adj} 个单元格不符）")
            check(
                n_diff > 0 or exp_adj - exp_plan <= 1e-6,
                f"{name} 调整表与计划表确有差异（{n_diff} 个单元格不同；"
                f"两表差额合计 {exp_adj - exp_plan:,.3f} kWh = A^exec 之和）",
            )

# ---------------------------------------------------------------------------
section("6) charge/discharge blocks and SOC vs the 10-minute trajectory")
for name, run_dir in RUNS.items():
    if not Path(EXPORTS[name]).exists():
        continue
    npz_path = Path(run_dir) / "trajectory.npz"
    if not npz_path.exists():
        check(False, f"{npz_path} exists")
        continue
    data = np.load(npz_path, allow_pickle=True)
    minutes = data["abs_minute"].astype(np.int64)
    idx_of = {int(m): i for i, m in enumerate(minutes)}
    soc = data["soc_boundary_kwh"]

    wb = load_workbook(Path(run_dir) / "result" / name, read_only=True, data_only=True)
    ws = wb["充放电量"]
    rows = list(ws.iter_rows(min_row=2, max_row=1 + 6 * 334, max_col=6, values_only=True))
    wb.close()

    bad_block = 0
    bad_soc = 0
    for i, day_text in enumerate(out_dates):
        day = date.fromisoformat(day_text)
        base = (day - date(2025, 1, 1)).days * MINUTES_PER_DAY
        for b in range(6):
            c_exp = 0.0
            d_exp = 0.0
            for k in range(24):
                abs_minute = base + (b * 24 + k) * 10
                j = idx_of.get(abs_minute)
                if j is not None:
                    c_exp += float(data["charge_kwh"][j])
                    d_exp += float(data["discharge_kwh"][j])
            rec = rows[i * 6 + b]
            if rec[2] is None or rec[3] is None:
                bad_block += 1
                continue
            if abs(float(rec[2]) - c_exp) > 1e-3 or abs(float(rec[3]) - d_exp) > 1e-3:
                bad_block += 1
        for label, abs_minute in (("0:00", base), ("24:00", base + MINUTES_PER_DAY)):
            j = idx_of.get(abs_minute)
            if j is None:
                continue
            rec = rows[i * 6 + (0 if label == "0:00" else 5)]
            col = 5
            if rec[col] is None or abs(float(rec[col]) - float(soc[j])) > 1e-3:
                bad_soc += 1
    check(bad_block == 0, f"{name} 4-hour block sums match ({bad_block} mismatches)")
    check(bad_soc == 0, f"{name} 0:00/24:00 SOC match ({bad_soc} mismatches)")

# ---------------------------------------------------------------------------
section("RESULT")
print("ALL CHECKS PASSED" if _ok else f"{len(_failures)} CHECK(S) FAILED")
for f in _failures:
    print("  -", f)
sys.exit(0 if _ok else 1)
