"""校验层与导出层测试：物理异常必须让流程失败，映射必须与模板对齐。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from microgrid import validation
from microgrid.constants import E_MAX, E_MIN, N_INTERVAL, OUTPUT_START
from microgrid.export import (
    DailyExportRow,
    _merge_adjacent,
    interval_index_of_start_time,
    paper_table1,
)
from microgrid.schemas import DayInput, Trajectory


def _day_input() -> DayInput:
    return DayInput(
        day=date(2025, 2, 1),
        demand_kwh=np.full(N_INTERVAL, 100.0),
        pv_kwh=np.full(N_INTERVAL, 20.0),
        price_yuan_per_kwh=np.full(N_INTERVAL, 0.5),
    )


def _clean_trajectory() -> Trajectory:
    """一条干净的轨迹：不充不放，全部外购，SOC 恒定 6000。"""
    n = N_INTERVAL
    grid = np.full(n, 80.0)  # D - G = 100 - 20 = 80
    return Trajectory(
        day=date(2025, 2, 1),
        grid_actual_kwh=grid,
        emergency_actual_kwh=np.zeros(n),
        charge_stored_kwh=np.zeros(n),
        discharge_delivered_kwh=np.zeros(n),
        curtail_kwh=np.zeros(n),
        soc_kwh=np.full(145, 6000.0),
        price_actual=np.full(n, 0.5),
    )


# ============================================================ 轨迹校验


def test_clean_trajectory_passes() -> None:
    diag = validation.validate_trajectory(_clean_trajectory(), _day_input())
    assert diag.max_bus_residual_kwh < 1e-9
    assert diag.max_soc_transition_error_kwh < 1e-9
    assert abs(diag.daily_balance_residual_kwh) < 1e-9
    assert diag.simultaneous_charge_discharge_intervals == []


def test_bus_imbalance_is_rejected() -> None:
    traj = _clean_trajectory()
    traj.grid_actual_kwh[7] += 5.0  # 凭空多买 5 kWh
    with pytest.raises(validation.ValidationError):
        validation.validate_trajectory(traj, _day_input())


def test_soc_out_of_range_is_rejected() -> None:
    traj = _clean_trajectory()
    traj.soc_kwh[3] = E_MAX + 1.0
    with pytest.raises(validation.ValidationError):
        validation.validate_trajectory(traj, _day_input())
    traj2 = _clean_trajectory()
    traj2.soc_kwh[3] = E_MIN - 1.0
    with pytest.raises(validation.ValidationError):
        validation.validate_trajectory(traj2, _day_input())


def test_curtail_beyond_pv_is_rejected() -> None:
    traj = _clean_trajectory()
    traj.curtail_kwh[10] = 25.0  # 光伏只有 20 kWh
    traj.grid_actual_kwh[10] -= 20.0  # 保持守恒，只让弃光越界
    traj.grid_actual_kwh[10] = max(traj.grid_actual_kwh[10], 0.0)
    with pytest.raises(validation.ValidationError):
        validation.validate_trajectory(traj, _day_input())


def test_simultaneous_charge_discharge_is_warned_not_hidden() -> None:
    """损耗循环必须被诊断并给出警告，而不是被偷偷加约束掩盖。"""
    n = N_INTERVAL
    traj = _clean_trajectory()
    traj.charge_stored_kwh[5] = 90.0
    traj.discharge_delivered_kwh[5] = 81.0  # ΔE = 0
    # 母线守恒要求该段多买 19 kWh
    traj.grid_actual_kwh[5] += 19.0
    diag = validation.validate_trajectory(traj, _day_input())
    assert diag.simultaneous_charge_discharge_intervals == [5]
    assert diag.loss_cycle_net_kwh == pytest.approx(19.0)
    assert any("同时充放电" in w for w in diag.warnings)


def test_daily_cycle_required_only_for_problem1() -> None:
    """日循环约束只对问题一强制；其它问题允许日末 SOC 不同。

    这里用真实的充放动作构造轨迹，避免"手改 SOC 数组"制造出不合物理的样本。
    """
    n = N_INTERVAL
    demand = np.full(n, 600.0)
    pv = np.zeros(n)

    def build(q_ch0: float, q_dis5: float) -> Trajectory:
        q_ch = np.zeros(n)
        q_dis = np.zeros(n)
        q_ch[0] = q_ch0
        q_dis[5] = q_dis5
        grid = demand - pv + q_ch / 0.9 - q_dis
        soc = np.concatenate(([6000.0], 6000.0 + np.cumsum(q_ch - q_dis / 0.9)))
        return Trajectory(
            day=date(2025, 2, 1),
            grid_actual_kwh=grid,
            emergency_actual_kwh=np.zeros(n),
            charge_stored_kwh=q_ch,
            discharge_delivered_kwh=q_dis,
            curtail_kwh=np.zeros(n),
            soc_kwh=soc,
            price_actual=np.full(n, 0.5),
        )

    day_input = DayInput(
        day=date(2025, 2, 1),
        demand_kwh=demand,
        pv_kwh=pv,
        price_yuan_per_kwh=np.full(n, 0.5),
    )

    # 日循环：充 500 后放 450（450/0.9 = 500），E 回到 6000
    cyclic = build(500.0, 450.0)
    assert cyclic.soc_kwh[-1] == pytest.approx(6000.0)
    validation.validate_trajectory(cyclic, day_input, require_daily_cycle=True)
    validation.validate_trajectory(cyclic, day_input, require_daily_cycle=False)

    # 非日循环：只充不放，日末 SOC 上升 500
    rising = build(500.0, 0.0)
    assert rising.soc_kwh[-1] == pytest.approx(6500.0)
    with pytest.raises(validation.ValidationError):
        validation.validate_trajectory(rising, day_input, require_daily_cycle=True)
    validation.validate_trajectory(rising, day_input, require_daily_cycle=False)


def test_output_day_window_is_enforced() -> None:
    from microgrid.timeaxis import calendar_days

    good = calendar_days(date(*OUTPUT_START), date(2025, 12, 31))
    validation.validate_output_days(good)
    assert len(good) == 334
    bad = good[:-1]
    with pytest.raises(validation.ValidationError):
        validation.validate_output_days(bad)


def test_information_leak_guard() -> None:
    validation.validate_no_future_leak(known_at=date(2025, 2, 1), known_interval=0)
    with pytest.raises(validation.ValidationError):
        validation.validate_no_future_leak(
            known_at=date(2025, 2, 1), known_interval=0, plan_uses_pv_actual=True
        )
    with pytest.raises(validation.ValidationError):
        validation.validate_no_future_leak(
            known_at=date(2025, 2, 1),
            known_interval=0,
            price_source_is_actual_future=True,
        )


def test_negative_price_is_rejected() -> None:
    validation.validate_attachment1_prices(np.array([0.5, 0.6]))
    with pytest.raises(validation.ValidationError):
        validation.validate_attachment1_prices(np.array([0.5, -0.1]))


# ============================================================ 导出映射


def test_start_time_to_interval() -> None:
    assert interval_index_of_start_time("10:00") == 60
    assert interval_index_of_start_time("12:00") == 72
    assert interval_index_of_start_time("20:00") == 120
    assert interval_index_of_start_time("0:00") == 0
    assert interval_index_of_start_time("23:50") == 143


def test_merge_adjacent_only_merges_neighbours() -> None:
    assert _merge_adjacent([]) == []
    assert _merge_adjacent([3]) == [(3, 4)]
    assert _merge_adjacent([3, 4, 5]) == [(3, 6)]
    assert _merge_adjacent([3, 5]) == [(3, 4), (5, 6)]
    assert _merge_adjacent([0, 1, 5, 6, 7, 9]) == [(0, 2), (5, 8), (9, 10)]


def _row(day: date) -> DailyExportRow:
    n = N_INTERVAL
    grid = np.full(n, 80.0)
    emg = np.zeros(n)
    emg[78] = 10.0  # 13:00-13:10
    emg[79] = 20.0  # 13:10-13:20
    emg[100] = 5.0  # 16:40-16:50
    return DailyExportRow(
        day=day,
        plan_initial_kwh=grid.copy(),
        final_plan_kwh=grid.copy(),
        grid_actual_kwh=grid,
        emergency_actual_kwh=emg,
        charge_stored_kwh=np.zeros(n),
        discharge_delivered_kwh=np.zeros(n),
        soc_start_kwh=6000.0,
        soc_end_kwh=6000.0,
        purchase_cost_yuan=100.0,
        penalty_yuan=7.5,
    )


def test_paper_table1_shape() -> None:
    rows = [_row(date(2025, 3, 20))]
    t1 = paper_table1(rows)
    # 6 个指定时间段 + 全天购电量 + 全天购电费
    assert len(t1) == 8
    labels = [r["时间段"] for r in t1]
    assert "10:00-10:10" in labels
    assert "20:00-20:10" in labels
    assert labels[-2] == "全天购电量"
    assert labels[-1] == "全天购电费"
    # 全天购电费 = 购电费 + 违约金
    assert t1[-1]["购电量"] == pytest.approx(107.5)


def test_daily_export_row_total_cost() -> None:
    r = _row(date(2025, 3, 20))
    assert r.full_day_cost_yuan == pytest.approx(107.5)


def test_export_result1_writes_template_copy(tmp_path) -> None:
    """导出必须写到副本，且 144 段按模型顺序落入第 2..145 行。"""
    from microgrid.export import export_result1

    traj = _clean_trajectory()
    traj.grid_actual_kwh = np.arange(N_INTERVAL, dtype=np.float64)

    class _Rec:
        pass

    rec = _Rec()
    rec.trajectory = traj
    rec.day = traj.day
    path = export_result1(rec, tmp_path)
    assert path.exists()

    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    ws = wb["计划购电量"]
    assert ws.max_row == N_INTERVAL + 1
    # 模型第 t 段写在模板第 t+2 行，模板标签一字不改
    assert ws.cell(row=2, column=1).value == "0:10-0:20"
    assert ws.cell(row=2, column=2).value == pytest.approx(0.0)
    assert ws.cell(row=145, column=2).value == pytest.approx(143.0)
    ws2 = wb["充放电量"]
    assert ws2.cell(row=2, column=1).value == "0:00-4:00"
    assert ws2.cell(row=7, column=1).value == "20:00-24:00"
    wb.close()


def test_original_template_is_not_modified(tmp_path) -> None:
    """导出前后原模板的字节必须完全一致。"""
    from microgrid.data_io import data_path
    from microgrid.export import export_result1

    src = data_path("data/附件5/result1.xlsx")
    before = src.read_bytes()

    traj = _clean_trajectory()

    class _Rec:
        pass

    rec = _Rec()
    rec.trajectory = traj
    rec.day = traj.day
    export_result1(rec, tmp_path)

    assert src.read_bytes() == before, "原模板被改写了！"
