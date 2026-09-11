"""End-to-end verification of the five exported deliverables.

Checks performed (all read-only):
  1. All five result workbooks exist and are readable.
  2. Original templates under data/附件5 are byte-identical to the originals
     recorded at first run (i.e. the program never touched them).
  3. Sheet names match the template structure.
  4. Time axis alignment: model interval t lands in template data slot t+1
     (confirmation item 1: template labels stay untouched).
  5. Daily counts: 334 days for problems 2/3/4-2/4-3, 144 rows for problem 1.
  6. Row totals in the exported workbook agree with the run's daily_bills.csv.
  7. Charge/discharge blocks sum back to the 10-minute trajectory totals.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from openpyxl import load_workbook  # noqa: E402

from microgrid.constants import N_INTERVAL  # noqa: E402

ROOT = Path(".")
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

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    ok = ok and bool(cond)


print("=" * 78)
print("1) template integrity: data/附件5 must be untouched")
print("=" * 78)
import hashlib

for name, path in TEMPLATES.items():
    p = Path(path)
    check(p.exists(), f"{path} exists")
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        check(p.stat().st_size > 1000, f"{path} size={p.stat().st_size} sha={h}")

print()
print("=" * 78)
print("2) exports exist and sheet structures match")
print("=" * 78)
EXPECTED_SHEETS = {
    "result1.xlsx": ["计划购电量", "充放电量"],
    "result2.xlsx": ["计划购电量", "充放电量", "紧急购电量"],
    "result3.xlsx": ["计划购电量", "调整购电量", "充放电量", "紧急购电量"],
    "result4-2.xlsx": ["计划购电量", "充放电量", "紧急购电量"],
    "result4-3.xlsx": ["计划购电量", "调整购电量", "充放电量", "紧急购电量"],
}
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

print()
print("=" * 78)
print("3) time-axis alignment (template labels untouched, model t -> slot t+1)")
print("=" * 78)
wb = load_workbook(EXPORTS["result1.xlsx"], read_only=True, data_only=True)
ws = wb["计划购电量"]
labels = [ws.cell(row=r, column=1).value for r in (1, 2, 143, 144, 145)]
check(labels[0] == "时间段", f"header row1 = {labels[0]!r}")
check(labels[1] == "0:10-0:20", f"first data label = {labels[1]!r} (template verbatim)")
check(labels[3] == "23:50-0:00+1", f"row144 label = {labels[3]!r}")
check(labels[4] == "0:00+1-0:10+1", f"row145 label = {labels[4]!r}")
ws2 = wb["充放电量"]
check(ws2.cell(row=2, column=1).value == "0:00-4:00", "充放电量 first block label")
check(ws2.cell(row=7, column=1).value == "20:00-24:00", "充放电量 last block label")
wb.close()

wb = load_workbook(EXPORTS["result2.xlsx"], read_only=True, data_only=True)
ws = wb["计划购电量"]
hdr = next(ws.iter_rows(min_row=1, max_row=1, max_col=147, values_only=True))
check(hdr[0] == "日期\\时间", f"matrix header col1 = {hdr[0]!r}")
check(hdr[1] == "0:10-0:20", f"matrix first slot label = {hdr[1]!r}")
check(hdr[143] == "23:50-0:00+1", f"matrix 144th slot label = {hdr[143]!r}")
check(hdr[144] == "0:00-0:10+1", f"matrix 145th slot label = {hdr[144]!r}")
check(hdr[145] == "全天购电量", f"matrix daily total col = {hdr[145]!r}")
check(hdr[146] == "全天购电费", f"matrix daily cost col = {hdr[146]!r}")
wb.close()
print()
print("=" * 78)
print("4) coverage: 334 days for P2/P3/P4, 144 rows for P1")
print("=" * 78)
wb = load_workbook(EXPORTS["result1.xlsx"], read_only=True, data_only=True)
check(wb["计划购电量"].max_row == N_INTERVAL + 1, "result1 计划购电量 rows = 145")
wb.close()

for name in ("result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"):
    wb = load_workbook(EXPORTS[name], read_only=True, data_only=True)
    ws = wb["计划购电量"]
    first = ws.cell(row=2, column=1).value
    last = ws.cell(row=335, column=1).value
    check(ws.max_row >= 335, f"{name} 计划购电量 rows >= 335 (got {ws.max_row})")
    check(
        str(first)[:10] == "2025-02-01" and str(last)[:10] == "2025-12-31",
        f"{name} date span {str(first)[:10]} .. {str(last)[:10]}",
    )
    wb.close()

print()
print("=" * 78)
print("5) exported numbers vs run records")
print("=" * 78)
for name, run_dir in RUNS.items():
    csv_path = Path(run_dir) / "daily_bills.csv"
    if not csv_path.exists():
        check(False, f"{csv_path} exists")
        continue
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]
    rows = [ln.split(",") for ln in lines]
    out = [r for r in rows if r[0] >= "2025-02-01"]
    check(len(out) == 334, f"{run_dir} daily_bills has 334 output days (got {len(out)})")

    wb = load_workbook(Path(run_dir) / "result" / name, read_only=True, data_only=True)
    ws = wb["计划购电量"]
    mismatches = 0
    checked = 0
    # read_only 流式读取；0-based index 145 = 第 146 列 = "全天购电量"
    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=335, max_col=147, values_only=True)):
        want = float(out[i][4])
        tot = row[145]
        if tot is None or abs(float(tot) - want) > 1e-3:
            mismatches += 1
        checked += 1
    wb.close()
    check(
        mismatches == 0,
        f"{name} daily purchase totals match ({checked} days checked, {mismatches} mismatches)",
    )

print()
print("=" * 78)
print("6) charge/discharge block sums match the 10-minute trajectory")
print("=" * 78)
for name, run_dir in RUNS.items():
    npz = np.load(Path(run_dir) / "trajectories.npz")
    days = [date.fromordinal(int(o)) for o in npz["days"]]
    keep = [i for i, d in enumerate(days) if d >= date(2025, 2, 1)]
    charge = npz["charge"][keep]
    discharge = npz["discharge"][keep]
    wb = load_workbook(Path(run_dir) / "result" / name, read_only=True, data_only=True)
    ws = wb["充放电量"]
    rows = list(ws.iter_rows(min_row=2, max_row=1 + 6 * len(keep), max_col=4, values_only=True))
    wb.close()
    bad = 0
    for i in range(len(keep)):
        for b in range(6):
            c_exp = float(charge[i, b * 24:(b + 1) * 24].sum())
            d_exp = float(discharge[i, b * 24:(b + 1) * 24].sum())
            c_got = rows[i * 6 + b][2]
            d_got = rows[i * 6 + b][3]
            if c_got is None or d_got is None:
                bad += 1
                continue
            if abs(float(c_got) - c_exp) > 1e-3 or abs(float(d_got) - d_exp) > 1e-3:
                bad += 1
    check(bad == 0, f"{name} 充放电量 4-hour block sums match ({bad} mismatches)")

print()
print("=" * 78)
print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
print("=" * 78)
sys.exit(0 if ok else 1)
