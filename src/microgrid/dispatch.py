"""滚动调度：在普通购电计划**已冻结**的前提下，逐窗口重算储能充放。

为什么需要这一层
----------------
问题二/4-2 的普通购电量 x 在 0 点一次冻结、不可拒收；但**储能怎么充放是对真实
供需残差的因果响应**（规范第 10 节："策略冻结是购电计划冻结，储能执行仍对实际
缺口作因果响应"）。若把 x 与真实负载/光伏直接相减、见到富余就灌满电池，电池会在
日中提前顶到上限，之后真实的富余就再无合法去处——这不是物理不可行，而是**执行
策略太笨**。

本模块在每个决策窗口用一次小规模 LP 重算电池动作：

    给定（已冻结）x_t、当前实际 SOC、以及"当前段用实测、未来段用预测"的边界，
    在满足储能物理约束的前提下，最小化
        紧急购电费（5 倍价）
      + 得不到消纳的富余量（这是"计划买多了却放不下"的直接度量）
    并要求计划签署的购电量全部被负载或储能消纳（不可拒收）。

只执行窗口内第一个子步的动作，然后滚动前进——标准滚动时域控制，
**不使用任何未来真实信息**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from .constants import (
    CHARGE_BUS_FACTOR,
    E_MAX,
    E_MIN,
    FEE_EMERGENCY,
    N_INTERVAL,
    Q_DIS_MAX,
    Q_MAX,
)
from .solver import DEFAULT_SOLVER, solve_model

#: 富余无法消纳时的惩罚单价（元/kWh）。取得足够大，使其在目标中优先于购电费，
#: 从而让 LP 主动避免"买进来却放不下"。
SPILL_PENALTY_YUAN_PER_KWH = 1.0e4

#: 充放电吞吐的惩罚单价（元/kWh）。
#
# 用途：在"费用相同"的多个最优解之间挑出让储能动作最小的那个。
#
# 取值理由：退化解（同一段既充又放）通常只差 0~2 元/kWh 量级的费用差，
# 而规范第 11 节明确要求**诊断**同时充放电、不能把它当成正常消纳手段。
# 若该惩罚太小（例如 1e-6），LP 会为了几分钱选择"充 750 放 750"这种
# 物理上可疑、设备上也难以执行的方案。这里取 1e-3 元/kWh：
# 它足以压掉纯退化解（750 kWh 的循环要多付 0.75 元，远超任何真实的
# 套利收益差），又小到不会改变真正的经济性结论。
THROUGHPUT_EPS = 1.0e-3


@dataclass(frozen=True)
class DispatchStep:
    """一个子步的调度结果。"""

    interval: int
    grid_kwh: float
    emergency_kwh: float
    charge_kwh: float
    discharge_kwh: float
    curtail_kwh: float
    soc_end_kwh: float
    spill_kwh: float


def solve_dispatch_window(
    *,
    grid_plan_kwh: np.ndarray,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    price_actual_yuan_per_kwh: np.ndarray,
    soc_start_kwh: float,
    t_start: int,
    t_end: int,
    horizon_end: int,
    solver_name: str = DEFAULT_SOLVER,
) -> list[DispatchStep]:
    """对 [t_start, t_end) 的每个子步求解，返回可执行的调度序列。

    ``grid_plan_kwh`` 是全 144 段的已冻结计划；本函数只使用 [t_start, t_end) 部分。
    ``horizon_end`` 是 LP 的优化视界终点（可超出 t_end，用于看到更远的未来预测）。
    """
    T = list(range(t_start, horizon_end))
    m = pyo.ConcreteModel(f"dispatch_{t_start}")
    m.T = pyo.Set(initialize=T, ordered=True)
    m.S = pyo.Set(initialize=list(range(t_start, horizon_end + 1)), ordered=True)

    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_MAX))
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_DIS_MAX))
    m.emergency = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.spill = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.S, domain=pyo.NonNegativeReals, bounds=(E_MIN, E_MAX))

    m.soc[ t_start ] = float(soc_start_kwh)
    m.fix_soc_start = pyo.Constraint(expr=m.soc[t_start] == float(soc_start_kwh))

    # 母线守恒：x + u + G - r + q_dis = D + q_ch / eta_c（r 只作用于光伏）
    # 这里把"弃光"与"无法消纳的富余"分开：弃光受光伏上限约束，spill 是
    # 计划购电过量导致的、物理上无处安放的部分，被重罚并在输出中如实报告。
    def _balance(m, t):
        supply = grid_plan_kwh[t] + m.emergency[t] + pv_kwh[t] + m.discharge[t]
        return supply == demand_kwh[t] + m.charge[t] * CHARGE_BUS_FACTOR + m.spill[t]

    m.balance = pyo.Constraint(m.T, rule=_balance)

    def _transition(m, t):
        return m.soc[t + 1] == m.soc[t] + m.charge[t] - m.discharge[t] / 0.9

    m.transition = pyo.Constraint(m.T, rule=_transition)

    def _obj(m):
        emg = sum(FEE_EMERGENCY * price_actual_yuan_per_kwh[t] * m.emergency[t] for t in m.T)
        spill = sum(SPILL_PENALTY_YUAN_PER_KWH * m.spill[t] for t in m.T)
        throughput = THROUGHPUT_EPS * sum(m.charge[t] + m.discharge[t] for t in m.T)
        return emg + spill + throughput

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    report = solve_model(m, "dispatch", solver_name=solver_name)
    if not report.feasible:
        raise RuntimeError(
            f"滚动调度 LP 不可行（t_start={t_start}, horizon_end={horizon_end}）："
            f"{report.status}/{report.termination}"
        )

    # 载入后自检：求解器给出的解必须真的满足母线守恒。
    # 这一层是防止"解没有真正写回变量"这类接口级错误被静默传播到仿真结果里。
    worst = 0.0
    worst_t = t_start
    for t in T:
        rhs = (
            float(pyo.value(m.charge[t])) * CHARGE_BUS_FACTOR
            + float(pyo.value(m.spill[t]))
            + demand_kwh[t]
        )
        lhs = (
            float(grid_plan_kwh[t])
            + float(pyo.value(m.emergency[t]))
            + pv_kwh[t]
            + float(pyo.value(m.discharge[t]))
        )
        gap = abs(lhs - rhs)
        if gap > worst:
            worst, worst_t = gap, t
    if worst > 1e-4:
        raise RuntimeError(
            f"滚动调度解未满足母线守恒：最大残差 {worst:.6f} kWh（区间 {worst_t}）。"
            "这通常意味着求解器解没有写回变量，属于接口级错误，不能继续。"
        )

    out: list[DispatchStep] = []
    for t in range(t_start, t_end):
        out.append(
            DispatchStep(
                interval=t,
                grid_kwh=float(grid_plan_kwh[t]),
                emergency_kwh=float(pyo.value(m.emergency[t])),
                charge_kwh=float(pyo.value(m.charge[t])),
                discharge_kwh=float(pyo.value(m.discharge[t])),
                curtail_kwh=0.0,
                soc_end_kwh=float(pyo.value(m.soc[t + 1])),
                spill_kwh=float(pyo.value(m.spill[t])),
            )
        )
    return out


__all__ = [
    "DispatchStep",
    "solve_dispatch_window",
    "SPILL_PENALTY_YUAN_PER_KWH",
    "THROUGHPUT_EPS",
]
