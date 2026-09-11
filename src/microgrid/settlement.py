"""结算层：实际执行计费、费率分类与违约累计。

本模块**必须先于优化器实现**（备忘录第 4 节）。

结算口径（备忘录第 5 节，已冻结）
--------------------------------
1. ``x`` 为 0 点策略，``y^(k)`` 为允许节点更新后的策略，最终普通实际购电记 ``h``，
   紧急实际购电记 ``u``。以每天 0 点为统计起点：:math:`Q_d^{act}=\\sum_t(h_{d,t}+u_{d,t})`。
2. 被修改放弃的**未执行**计划量不计入 Q，不计其电费，**也不要求在最终购电表中列出**。
   已执行量不可追溯取消。
3. 违约费保留：在调整时刻 a，针对允许修改的未执行区间
   :math:`K_a = 0.5\\,p_a\\sum_t [y^{old}_t-y^{new}_t]_+`，``p_a`` 为**调整当时**的实时电价。
   后来恢复计划不抹掉该违约金。
4. 最终费用
   :math:`C_d^{act}=\\sum_{e\\in\\mathcal E_d}\\alpha_e c_e q_e^{act}+\\sum_a K_a`，
   费率 α 为 普通 1 / 实际调整增购 1.5 / 紧急 5。
   事件类别由**当时真实执行动作**确定，不用最终相对 0 点的净差额重新分类。

三条不变量（规范第 16 节）
--------------------------
* 实际执行即计费；未来候选策略不是已购买量，不得把每轮预测购电费用重复累计；
* 被放弃的未执行方案电费净额为零；
* 相同最终购电量不保证费用相同——最终费用还包含已发生违约金。
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .constants import (
    FEE_ADJUST_UP,
    FEE_EMERGENCY,
    FEE_NORMAL,
    FEE_PENALTY_DOWN,
    FEE_RATE_BY_CLASS,
    N_INTERVAL,
    TOL_COST_YUAN,
    TOL_ENERGY_KWH,
    FeeClass,
)
from .schemas import ActivePlan, Bill, ExecutedEvent, PlanUpdate, Trajectory


class SettlementError(RuntimeError):
    """账务不自洽——正式导出必须失败。"""


# ==========================================================================
# 1. 违约累计
# ==========================================================================


def plan_reduction(
    old_plan: ActivePlan, new_plan: ActivePlan, new_created_at_interval: int
) -> np.ndarray:
    """计算 δ^-_{a,t} = [y^old_t - y^new_t]_+。

    只有**尚未执行**的区间可修改；已执行区间 a 之前的部分不计减少量
    （已执行量不可追溯取消）。
    """
    a = new_created_at_interval
    old = np.asarray(old_plan.grid_kwh, dtype=np.float64)
    new = np.asarray(new_plan.grid_kwh, dtype=np.float64)
    delta = np.zeros(N_INTERVAL, dtype=np.float64)
    if a < N_INTERVAL:
        delta[a:] = np.maximum(old[a:] - new[a:], 0.0)
    return delta


def penalty_for_adjustment(
    old_plan: ActivePlan, new_plan: ActivePlan, price_at_interval: float
) -> PlanUpdate:
    """一次调整产生的违约金记录。

    ``price_at_interval`` 是**调整当时**的实时电价 p_a，不是目标交付区间的价格。
    """
    a = new_plan.created_at_interval
    reduced = plan_reduction(old_plan, new_plan, a)
    penalty = FEE_PENALTY_DOWN * float(price_at_interval) * float(reduced.sum())
    return PlanUpdate(
        at_interval=a,
        price_at_interval_yuan_per_kwh=float(price_at_interval),
        reduced_kwh=reduced,
        penalty_yuan=penalty,
    )


def accumulate_penalty(updates: list[PlanUpdate]) -> float:
    """已发生违约金合计。后来恢复计划不抹掉已发生的 K_a。"""
    return float(sum(u.penalty_yuan for u in updates))


# ==========================================================================
# 2. 实际执行事件的费率分类
# ==========================================================================


def classify_executed(
    grid_actual_kwh: np.ndarray,
    emergency_actual_kwh: np.ndarray,
    price_actual: np.ndarray,
    day: date,
    initial_plan_kwh: np.ndarray | None = None,
    adjust_events: dict[int, float] | None = None,
) -> list[ExecutedEvent]:
    """把实际执行轨迹翻译成计费事件。

    Parameters
    ----------
    grid_actual_kwh : (144,) 普通实际购电 h_t
    emergency_actual_kwh : (144,) 紧急实际购电 u_t
    price_actual : (144,) 事件执行当时的价格 c_e
    initial_plan_kwh : (144,) 0 点原计划 x_t。仅用于 4-3 的"实际调整增购"分类：
        在某次调整生效之后执行的区间，超过**该次调整前有效计划**的部分按 1.5 倍计费。
        传入 None（问题一/二/4-2，或分档模式）时，改用解释 (a)：
        超过 0 点原计划的部分按 1.5 倍。
    adjust_events : {调整段号: 调整前该段的有效计划量}。键是调整发生的段号，
        值是**调整前**的有效计划值（用于计算超出量）。
    """
    h = np.asarray(grid_actual_kwh, dtype=np.float64)
    u = np.asarray(emergency_actual_kwh, dtype=np.float64)
    c = np.asarray(price_actual, dtype=np.float64)
    for name, arr in (("grid_actual_kwh", h), ("emergency_actual_kwh", u), ("price_actual", c)):
        if arr.shape != (N_INTERVAL,):
            raise SettlementError(f"{name} 应为 ({N_INTERVAL},)，得到 {arr.shape}")

    # 每个区间的 1.5 倍计费基准
    #   未提供 0 点计划（问题一/二/4-2，或分档模式）时，基准取实际普通购电本身，
    #   于是不会凭空产生"实际调整增购"，全部按普通 1 倍计费。
    if initial_plan_kwh is None:
        baseline = h.copy()
    else:
        baseline = np.asarray(initial_plan_kwh, dtype=np.float64).copy()
        if adjust_events:
            # 调整时刻 a 起，基准换成"调整前该次生效的有效计划"，而非 0 点原计划
            for at_interval in sorted(adjust_events):
                prev_plan = np.asarray(adjust_events[at_interval], dtype=np.float64)
                if prev_plan.shape == ():
                    baseline[at_interval:] = float(prev_plan)
                else:
                    if prev_plan.shape != (N_INTERVAL,):
                        raise SettlementError("adjust_events 的值必须是标量或 (144,) 数组")
                    baseline[at_interval:] = prev_plan[at_interval:]

    events: list[ExecutedEvent] = []
    for t in range(N_INTERVAL):
        if h[t] > TOL_ENERGY_KWH:
            up = min(h[t], max(h[t] - baseline[t], 0.0))
            normal = h[t] - up
            if normal > TOL_ENERGY_KWH:
                events.append(
                    ExecutedEvent(day, t, normal, float(c[t]), FeeClass.NORMAL)
                )
            if up > TOL_ENERGY_KWH:
                events.append(
                    ExecutedEvent(day, t, up, float(c[t]), FeeClass.ADJUST_UP)
                )
        if u[t] > TOL_ENERGY_KWH:
            events.append(ExecutedEvent(day, t, float(u[t]), float(c[t]), FeeClass.EMERGENCY))
    return events


# ==========================================================================
# 3. 账单
# ==========================================================================


def events_cost_yuan(events: list[ExecutedEvent]) -> float:
    """所有已执行事件的费用合计（不含违约金）。"""
    total = 0.0
    for e in events:
        total += FEE_RATE_BY_CLASS[e.fee_class] * e.price_yuan_per_kwh * e.quantity_kwh
    return float(total)


def make_bill(
    day: date,
    events: list[ExecutedEvent],
    updates: list[PlanUpdate] | None = None,
) -> Bill:
    """由执行事件与违约记录生成最终账单。

    数量只来自实际轨迹；费用 = 执行事件费用 + 已发生违约金。
    """
    qty = {FeeClass.NORMAL: 0.0, FeeClass.ADJUST_UP: 0.0, FeeClass.EMERGENCY: 0.0}
    cost = {FeeClass.NORMAL: 0.0, FeeClass.ADJUST_UP: 0.0, FeeClass.EMERGENCY: 0.0}
    for e in events:
        qty[e.fee_class] += e.quantity_kwh
        cost[e.fee_class] += e.cost_yuan
    penalty = accumulate_penalty(updates or [])
    total_q = qty[FeeClass.NORMAL] + qty[FeeClass.ADJUST_UP] + qty[FeeClass.EMERGENCY]
    purchase_cost = cost[FeeClass.NORMAL] + cost[FeeClass.ADJUST_UP] + cost[FeeClass.EMERGENCY]
    return Bill(
        day=day,
        normal_kwh=qty[FeeClass.NORMAL],
        adjust_up_kwh=qty[FeeClass.ADJUST_UP],
        emergency_kwh=qty[FeeClass.EMERGENCY],
        total_purchased_kwh=total_q,
        purchase_cost_yuan=purchase_cost,
        penalty_yuan=penalty,
        total_cost_yuan=purchase_cost + penalty,
        cost_by_class_yuan=dict(cost),
    )


def settle_trajectory(
    trajectory: Trajectory,
    initial_plan_kwh: np.ndarray | None = None,
    adjust_events: dict[int, float] | None = None,
) -> Bill:
    """从完整实际轨迹独立复算账单（不依赖优化器内部目标值）。"""
    events = classify_executed(
        grid_actual_kwh=trajectory.grid_actual_kwh,
        emergency_actual_kwh=trajectory.emergency_actual_kwh,
        price_actual=trajectory.price_actual,
        day=trajectory.day,
        initial_plan_kwh=initial_plan_kwh,
        adjust_events=adjust_events,
    )
    return make_bill(trajectory.day, events, trajectory.updates)


# ==========================================================================
# 4. 独立复算与一致性校验
# ==========================================================================


def recompute_cost_yuan(
    quantities_kwh: np.ndarray,
    prices_yuan_per_kwh: np.ndarray,
    fee_classes: list[FeeClass] | np.ndarray,
) -> float:
    """最朴素的逐事件复算，用于与 :func:`events_cost_yuan` 交叉校验。

    刻意写成与生产代码不同的路径（直接循环乘加、不做分类合并）。
    """
    q = np.asarray(quantities_kwh, dtype=np.float64)
    p = np.asarray(prices_yuan_per_kwh, dtype=np.float64)
    if q.shape != p.shape:
        raise SettlementError("复算输入形状不一致")
    classes = list(fee_classes)
    if len(classes) != q.size:
        raise SettlementError("复算输入长度不一致")
    total = 0.0
    for i in range(q.size):
        total += q[i] * p[i] * FEE_RATE_BY_CLASS[FeeClass(int(classes[i]))]
    return float(total)


def verify_bill(
    bill: Bill,
    events: list[ExecutedEvent],
    updates: list[PlanUpdate] | None = None,
    tol: float = TOL_COST_YUAN,
) -> None:
    """用与 :func:`make_bill` 不同的路径重算账单，任何不一致立即失败。"""
    if events:
        q = np.array([e.quantity_kwh for e in events], dtype=np.float64)
        p = np.array([e.price_yuan_per_kwh for e in events], dtype=np.float64)
        classes = np.array([int(e.fee_class) for e in events], dtype=np.int64)
        recomputed = recompute_cost_yuan(q, p, classes)
    else:
        recomputed = 0.0
    if abs(recomputed - bill.purchase_cost_yuan) > tol:
        raise SettlementError(
            f"购电费复算不一致：账单 {bill.purchase_cost_yuan:.6f}，复算 {recomputed:.6f}"
        )
    penalty = accumulate_penalty(updates or [])
    if abs(penalty - bill.penalty_yuan) > tol:
        raise SettlementError(f"违约费复算不一致：账单 {bill.penalty_yuan:.6f}，复算 {penalty:.6f}")
    total_q = float(sum(e.quantity_kwh for e in events))
    if abs(total_q - bill.total_purchased_kwh) > TOL_ENERGY_KWH:
        raise SettlementError(
            f"实际购电量不一致：账单 {bill.total_purchased_kwh:.9f}，事件合计 {total_q:.9f}"
        )
    if abs(bill.total_cost_yuan - (bill.purchase_cost_yuan + bill.penalty_yuan)) > tol:
        raise SettlementError("费用分项与合计不符")


# ==========================================================================
# 5. 规范第 16 节的手算用例（供测试直接调用）
# ==========================================================================


def hand_case_price_flat_100_to_80() -> dict[str, float]:
    """价格均 1：原策略 100 改为 80，最终量 80、电费 80、违约 10、合计 90。"""
    old_arr = np.zeros(N_INTERVAL, dtype=np.float64)
    old_arr[0] = 100.0
    old = ActivePlan(created_at_interval=0, grid_kwh=old_arr)
    new_arr = old_arr.copy()
    new_arr[0] = 80.0
    new = ActivePlan(created_at_interval=0, grid_kwh=new_arr, version=1)
    update = penalty_for_adjustment(old, new, price_at_interval=1.0)
    events = [ExecutedEvent(date(2025, 2, 1), 0, 80.0, 1.0, FeeClass.NORMAL)]
    bill = make_bill(date(2025, 2, 1), events, [update])
    return {
        "quantity_kwh": bill.total_purchased_kwh,
        "purchase_cost_yuan": bill.purchase_cost_yuan,
        "penalty_yuan": bill.penalty_yuan,
        "total_cost_yuan": bill.total_cost_yuan,
    }


__all__ = [
    "SettlementError",
    "plan_reduction",
    "penalty_for_adjustment",
    "accumulate_penalty",
    "classify_executed",
    "events_cost_yuan",
    "make_bill",
    "settle_trajectory",
    "recompute_cost_yuan",
    "verify_bill",
    "hand_case_price_flat_100_to_80",
    "FEE_NORMAL",
    "FEE_ADJUST_UP",
    "FEE_EMERGENCY",
    "FEE_PENALTY_DOWN",
]
