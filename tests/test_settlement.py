"""结算层测试：规范第 16 节手算用例、费率分类、量费分离、独立复算。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from microgrid import settlement as st
from microgrid.constants import N_INTERVAL, FeeClass
from microgrid.schemas import ActivePlan, ExecutedEvent, PlanUpdate, Trajectory


def _plan(values: dict[int, float], version: int = 0, created_at: int = 0) -> ActivePlan:
    arr = np.zeros(N_INTERVAL)
    for t, v in values.items():
        arr[t] = v
    return ActivePlan(created_at_interval=created_at, grid_kwh=arr, version=version)


# ============================================================ 规范手算用例


def test_hand_case_flat_price_100_to_80() -> None:
    """价格均 1：原策略 100 改为 80 -> 最终量 80、电费 80、违约 10、合计 90。"""
    result = st.hand_case_price_flat_100_to_80()
    assert result["quantity_kwh"] == pytest.approx(80.0)
    assert result["purchase_cost_yuan"] == pytest.approx(80.0)
    assert result["penalty_yuan"] == pytest.approx(10.0)
    assert result["total_cost_yuan"] == pytest.approx(90.0)


def test_hand_case_100_to_120_to_80() -> None:
    """策略 100 -> 120 -> 80，都在执行前修改：最终量 80，减少 40 产生违约 20。"""
    x = _plan({0: 100.0})
    y1 = _plan({0: 120.0}, version=1)
    y2 = _plan({0: 80.0}, version=2)
    up = st.penalty_for_adjustment(x, y1, price_at_interval=1.0)
    assert up.penalty_yuan == pytest.approx(0.0)  # 上调不产生违约金
    down = st.penalty_for_adjustment(y1, y2, price_at_interval=1.0)
    assert down.total_reduced_kwh == pytest.approx(40.0)
    assert down.penalty_yuan == pytest.approx(20.0)
    events = [ExecutedEvent(date(2025, 2, 1), 0, 80.0, 1.0, FeeClass.NORMAL)]
    bill = st.make_bill(date(2025, 2, 1), events, [up, down])
    assert bill.total_purchased_kwh == pytest.approx(80.0)  # 不是 300
    assert bill.purchase_cost_yuan == pytest.approx(80.0)
    assert bill.penalty_yuan == pytest.approx(20.0)
    assert bill.total_cost_yuan == pytest.approx(100.0)


def test_hand_case_variable_price_cancellation() -> None:
    """取消 20 时当前价 2 -> 违约 20；最终执行 80 时价 1 -> 电费 80，合计 100。"""
    x = _plan({0: 100.0})
    y = _plan({0: 80.0}, version=1)
    down = st.penalty_for_adjustment(x, y, price_at_interval=2.0)
    assert down.penalty_yuan == pytest.approx(20.0)
    events = [ExecutedEvent(date(2025, 2, 1), 0, 80.0, 1.0, FeeClass.NORMAL)]
    bill = st.make_bill(date(2025, 2, 1), events, [down])
    assert bill.total_purchased_kwh == pytest.approx(80.0)
    assert bill.purchase_cost_yuan == pytest.approx(80.0)
    assert bill.penalty_yuan == pytest.approx(20.0)
    assert bill.total_cost_yuan == pytest.approx(100.0)


def test_penalty_price_is_price_at_adjustment_time() -> None:
    """违约用**调整当时**的价格，不是目标交付区间的价格。"""
    x = _plan({0: 10.0, 5: 10.0})
    y = _plan({0: 10.0, 5: 4.0}, version=1, created_at=3)
    # 调整发生在 t=3，此刻价格 2.0；目标区间 t=5 的价格与此无关
    up = st.penalty_for_adjustment(x, y, price_at_interval=2.0)
    assert up.reduced_kwh[5] == pytest.approx(6.0)
    assert up.penalty_yuan == pytest.approx(0.5 * 2.0 * 6.0)


def test_executed_intervals_cannot_be_retroactively_reduced() -> None:
    """已执行量不可追溯取消：调整段之前的历史不计入减少量。"""
    x = _plan({0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0})
    y = _plan({0: 100.0, 1: 100.0, 2: 100.0, 3: 0.0}, version=1, created_at=3)
    up = st.penalty_for_adjustment(x, y, price_at_interval=1.0)
    assert up.reduced_kwh[0] == 0.0
    assert up.reduced_kwh[1] == 0.0
    assert up.reduced_kwh[2] == 0.0
    assert up.reduced_kwh[3] == pytest.approx(100.0)


def test_abandoned_future_plan_costs_nothing() -> None:
    """被放弃的未执行方案电费净额为零——不得按每轮预测重复累计。"""
    trajectory = _trajectory(
        grid={0: 80.0},
        price=1.0,
        emergency={},
    )
    trajectory.updates = [
        st.penalty_for_adjustment(_plan({0: 100.0}), _plan({0: 80.0}, version=1), 1.0)
    ]
    bill = st.settle_trajectory(trajectory)
    # 若把每轮候选都算进去会是 100+80+... ；此处只有 80 的电费 + 10 违约
    assert bill.purchase_cost_yuan == pytest.approx(80.0)
    assert bill.total_cost_yuan == pytest.approx(90.0)


def test_penalty_survives_plan_restoration() -> None:
    """后来恢复计划不抹掉已发生的违约金。"""
    x = _plan({0: 100.0})
    y1 = _plan({0: 40.0}, version=1)
    y2 = _plan({0: 100.0}, version=2)
    k1 = st.penalty_for_adjustment(x, y1, price_at_interval=1.0)
    k2 = st.penalty_for_adjustment(y1, y2, price_at_interval=1.0)
    assert k1.penalty_yuan == pytest.approx(30.0)
    assert k2.penalty_yuan == pytest.approx(0.0)
    assert st.accumulate_penalty([k1, k2]) == pytest.approx(30.0)


# ============================================================ 费率分类


def test_emergency_uses_five_times_price() -> None:
    traj = _trajectory(grid={}, price=2.0, emergency={7: 10.0})
    bill = st.settle_trajectory(traj)
    assert bill.emergency_kwh == pytest.approx(10.0)
    assert bill.total_cost_yuan == pytest.approx(5.0 * 2.0 * 10.0)


def test_adjust_up_uses_one_point_five_times_price() -> None:
    """实际增购 10 kWh 时价 2 -> 应付 30（1.5 倍）。"""
    initial = np.zeros(N_INTERVAL)
    initial[3] = 20.0
    traj = _trajectory(grid={3: 30.0}, price=2.0, emergency={})
    bill = st.settle_trajectory(traj, initial_plan_kwh=initial)
    assert bill.normal_kwh == pytest.approx(20.0)
    assert bill.adjust_up_kwh == pytest.approx(10.0)
    assert bill.purchase_cost_yuan == pytest.approx(20 * 2 + 10 * 2 * 1.5)
    assert bill.total_cost_yuan == pytest.approx(70.0)


def test_normal_purchase_is_one_times_price() -> None:
    """实际普通购电 10 kWh 时价 2 -> 应付 20。"""
    traj = _trajectory(grid={4: 10.0}, price=2.0, emergency={})
    bill = st.settle_trajectory(traj)
    assert bill.normal_kwh == pytest.approx(10.0)
    assert bill.total_cost_yuan == pytest.approx(20.0)


def test_class_baseline_switches_to_plan_effective_at_adjustment() -> None:
    """调整生效后，1.5 倍基准换成该次调整前的有效计划，而不是 0 点原计划。"""
    initial = np.zeros(N_INTERVAL)
    initial[40] = 100.0  # 0 点原计划
    prev_effective = np.zeros(N_INTERVAL)
    prev_effective[40] = 60.0  # 调整前有效计划已降为 60
    traj = _trajectory(grid={40: 70.0}, price=1.0, emergency={})
    bill = st.settle_trajectory(traj, initial_plan_kwh=initial, adjust_events={36: prev_effective})
    assert bill.normal_kwh == pytest.approx(60.0)
    assert bill.adjust_up_kwh == pytest.approx(10.0)
    assert bill.purchase_cost_yuan == pytest.approx(60 * 1.0 + 10 * 1.5)


def test_executed_history_is_not_reclassified_by_final_net_difference() -> None:
    """事件类别由当时真实执行动作确定，不用最终相对 0 点的净差额重分类。"""
    initial = np.zeros(N_INTERVAL)
    initial[0] = 100.0
    initial[1] = 100.0
    # 最终净量与原计划相同，但 t=0 多买了 50、t=1 少买了 50
    traj = _trajectory(grid={0: 150.0, 1: 50.0}, price=1.0, emergency={})
    bill = st.settle_trajectory(traj, initial_plan_kwh=initial)
    assert bill.adjust_up_kwh == pytest.approx(50.0)   # t=0 的超出部分
    assert bill.normal_kwh == pytest.approx(150.0)     # t=1 的 50 仍是普通量
    assert bill.total_purchased_kwh == pytest.approx(200.0)


# ============================================================ 复算


def test_verify_bill_detects_tampering() -> None:
    traj = _trajectory(grid={0: 100.0}, price=0.5, emergency={1: 4.0})
    bill = st.settle_trajectory(traj)
    events = st.classify_executed(
        traj.grid_actual_kwh, traj.emergency_actual_kwh, traj.price_actual, traj.day
    )
    st.verify_bill(bill, events, traj.updates)  # 不抛异常
    bad = bill.__class__(**{**bill.__dict__, "purchase_cost_yuan": bill.purchase_cost_yuan + 1.0})
    with pytest.raises(st.SettlementError):
        st.verify_bill(bad, events, traj.updates)


def test_recompute_cost_matches_event_path() -> None:
    events = [
        ExecutedEvent(date(2025, 3, 1), 0, 10.0, 0.4, FeeClass.NORMAL),
        ExecutedEvent(date(2025, 3, 1), 1, 5.0, 0.4, FeeClass.ADJUST_UP),
        ExecutedEvent(date(2025, 3, 1), 2, 2.0, 0.4, FeeClass.EMERGENCY),
    ]
    direct = st.recompute_cost_yuan(
        np.array([e.quantity_kwh for e in events]),
        np.array([e.price_yuan_per_kwh for e in events]),
        [int(e.fee_class) for e in events],
    )
    assert direct == pytest.approx(st.events_cost_yuan(events))
    assert direct == pytest.approx(0.4 * (10 + 7.5 + 10))


def test_same_quantity_different_cost() -> None:
    """相同最终购电量不保证费用相同（违约金不同）。"""
    initial = np.zeros(N_INTERVAL)
    initial[0] = 100.0
    base = _trajectory(grid={0: 80.0}, price=1.0, emergency={})
    with_penalty = _trajectory(grid={0: 80.0}, price=1.0, emergency={})
    with_penalty.updates = [
        st.penalty_for_adjustment(_plan({0: 100.0}), _plan({0: 80.0}, version=1), 1.0)
    ]
    b1 = st.settle_trajectory(base, initial_plan_kwh=initial)
    b2 = st.settle_trajectory(with_penalty, initial_plan_kwh=initial)
    assert b1.total_purchased_kwh == pytest.approx(b2.total_purchased_kwh)
    assert b1.normal_kwh == pytest.approx(80.0)  # 无违约时全部按普通 1 倍
    assert b2.total_cost_yuan - b1.total_cost_yuan == pytest.approx(10.0)


# ============================================================ 辅助


def _trajectory(
    grid: dict[int, float],
    price: float,
    emergency: dict[int, float],
) -> Trajectory:
    g = np.zeros(N_INTERVAL)
    u = np.zeros(N_INTERVAL)
    for t, v in grid.items():
        g[t] = v
    for t, v in emergency.items():
        u[t] = v
    total = float(g.sum() + u.sum())
    # 用一条纯外购、不充不放的轨迹承载购电量（储能列留 0，SOC 恒定）
    return Trajectory(
        day=date(2025, 2, 1),
        grid_actual_kwh=g,
        emergency_actual_kwh=u,
        charge_stored_kwh=np.zeros(N_INTERVAL),
        discharge_delivered_kwh=np.zeros(N_INTERVAL),
        curtail_kwh=np.zeros(N_INTERVAL),
        soc_kwh=np.full(145, 6000.0),
        price_actual=np.full(N_INTERVAL, price),
        events=[],
        updates=[],
    )
