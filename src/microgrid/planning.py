"""计划层：各问的优化模型装配。

四问共用同一物理约束装配函数 :func:`attach_physics`，
只在"可用信息、策略权限、价格口径、终端条件"上切换
（规范第 22 节：后续问改变信息与交易制度，不改写基础单位和储能定义）。

模型变量（144 段）
------------------
``grid_kwh``      普通购电计划 x_t / y_t >= 0
``emergency_kwh`` 紧急购电 u_t >= 0（仅问题二/4-2/三/4-3）
``charge_kwh``    q_ch_t ∈ [0, 750]，母线侧充电量
``discharge_kwh`` q_dis_t ∈ [0, 750]，实际送达微网的放电量
``curtail_kwh``   r_t ∈ [0, G_t]，仅弃光
``soc_kwh``       E_s ∈ [1200, 10800]，s = 0..144

**不加入充放电互斥**（规范第 11 节：题面未明确的不自动加入主模型），
但求解后必须诊断同时充放电与损耗循环；若最优解依赖它，必须披露可实施性缺口。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from .constants import (
    CHARGE_BUS_FACTOR,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    ETA_CHARGE,
    FEE_EMERGENCY,
    N_BOUNDARY,
    N_INTERVAL,
    Q_DIS_MAX,
    Q_MAX,
)
from .schemas import DayInput, SolveReport
from .solver import DEFAULT_SOLVER, solve_model


@dataclass(frozen=True)
class PlanResult:
    """一次优化的完整输出——不只返回购电数组。"""

    report: SolveReport
    grid_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_kwh: np.ndarray
    objective_yuan: float


# ==========================================================================
# 共享物理约束装配
# ==========================================================================


def attach_physics(
    m: pyo.ConcreteModel,
    *,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    soc_start_kwh: float,
    soc_end_kwh: float | None,
    allow_emergency: bool,
    t_start: int = 0,
    absorption_floor_kwh: np.ndarray | None = None,
    absorption_pv_ceiling_kwh: np.ndarray | None = None,
    soc_reserve_kwh: float = 0.0,
    soc_end_min_kwh: float | None = None,
    soc_end_max_equals: bool = False,
    soc_upper_kwh: np.ndarray | None = None,
    soc_lower_kwh: np.ndarray | None = None,
) -> None:
    """把统一物理核装到模型上：守恒、状态转移、容量、功率、弃光。

    ``soc_end_kwh`` 为 None 表示不加日末约束（问题二及以后无每日首末相等）；
    仅问题一传入 6000。

    ``absorption_floor_kwh`` / ``absorption_pv_ceiling_kwh``
        稳健可消纳上限。外购电**不可拒收**，所以计划购电量必须保证
        "真实负载偏低、光伏偏高"时也放得下：

            grid_t <= absorption_floor_t - pv_ceiling_t + Q_MAX / eta_c

        两个边界都由**已成熟的历史预测误差**构造（见
        :meth:`microgrid.forecast.Forecaster.robust_absorption_floor`）。
        问题一的负载与光伏由题面直接给定、不存在预测偏差，故传 None 关闭该上限。
        这条约束只把物理可吸收性写进模型，不引入题面之外的拒收、售电或烧电机制。

    ``soc_upper_kwh`` / ``soc_lower_kwh``（**核心的可行性保障**）
        日固定计划要能真正执行，计划自身的 SOC 轨迹必须为预测偏差留出余量。
        设当日累计预测误差不超过 ``band_t``（由历史误差估计），则该段真实 SOC 与
        计划 SOC 的最大偏差不超过 ``band_t``（受功率上限截断）。于是要求

            E_s <= soc_upper_kwh[s]      # 留出吸收"买多了"的余量
            E_s >= soc_lower_kwh[s]      # 留出承受"不够用"的余量

        这两条**只约束计划**，不改变任何物理参数，也不引入拒收或售电；
        它把"计划必须可执行"这一要求写成了线性约束。问题一传入 None。
    """
    T = list(range(t_start, N_INTERVAL))
    m.T = pyo.Set(initialize=T, ordered=True)
    m.S = pyo.Set(initialize=list(range(t_start, N_BOUNDARY)), ordered=True)

    m.demand = pyo.Param(m.T, initialize={t: float(demand_kwh[t]) for t in T}, mutable=False)
    m.pv = pyo.Param(m.T, initialize={t: float(pv_kwh[t]) for t in T}, mutable=False)

    m.grid = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_MAX))
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_DIS_MAX))
    m.curtail = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.S, domain=pyo.NonNegativeReals, bounds=(E_MIN, E_MAX))

    if allow_emergency:
        m.emergency = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    else:
        m.emergency = None

    # 母线守恒：h + u + G - r + q_dis = D + q_ch / eta_c
    def _balance(m, t):
        lhs = m.grid[t] + m.pv[t] - m.curtail[t] + m.discharge[t]
        rhs = m.demand[t] + m.charge[t] * CHARGE_BUS_FACTOR
        if m.emergency is not None:
            lhs = lhs + m.emergency[t]
        return lhs == rhs

    m.balance = pyo.Constraint(m.T, rule=_balance)

    # 状态转移：E_{t+1} = E_t + q_ch - q_dis / eta_d
    def _transition(m, t):
        return m.soc[t + 1] == m.soc[t] + m.charge[t] - m.discharge[t] * DISCHARGE_BATTERY_FACTOR

    m.transition = pyo.Constraint(m.T, rule=_transition)

    # 弃光只作用于光伏
    m.curtail_limit = pyo.Constraint(m.T, rule=lambda m, t: m.curtail[t] <= m.pv[t])

    # 稳健可消纳上限：外购电不可拒收，计划量必须在"最不利的真实负载/光伏"下
    # 仍能被物理吸收，否则会出现"买多了却没地方放"的不可行轨迹
    # （规范第 6 节：紧急量只增加供给，不能解决过量购买）。
    if absorption_floor_kwh is not None and absorption_pv_ceiling_kwh is not None:
        floor = np.asarray(absorption_floor_kwh, dtype=np.float64)
        ceil_pv = np.asarray(absorption_pv_ceiling_kwh, dtype=np.float64)

        def _absorption(m, t):
            bound = floor[t] - ceil_pv[t] + Q_MAX / ETA_CHARGE
            return m.grid[t] <= max(bound, 0.0)

        m.absorption = pyo.Constraint(m.T, rule=_absorption)

    # 初值
    m.soc_start = pyo.Constraint(expr=m.soc[t_start] == float(soc_start_kwh))

    # 日内预留守备（仅滚动计划启用）：把 E_MAX 的有效上限下调，保留吸收余量
    if soc_reserve_kwh > 0.0:
        cap = E_MAX - float(soc_reserve_kwh)
        if cap < E_MIN:
            raise ValueError("soc_reserve_kwh 过大，导致有效上限低于下限")
        m.soc_reserve = pyo.Constraint(m.S, rule=lambda m, s: m.soc[s] <= cap)

    # 计划 SOC 鲁棒窗口：保证计划在预测偏差下仍可被物理执行
    if soc_upper_kwh is not None:
        upper = np.asarray(soc_upper_kwh, dtype=np.float64)
        m.soc_upper = pyo.Constraint(
            m.S, rule=lambda m, s: m.soc[s] <= min(upper[s], E_MAX)
        )
    if soc_lower_kwh is not None:
        lower = np.asarray(soc_lower_kwh, dtype=np.float64)
        m.soc_lower = pyo.Constraint(
            m.S, rule=lambda m, s: m.soc[s] >= max(lower[s], E_MIN)
        )

    # 日末锚定：把计划末端的储电量钉在给定值。
    # 为什么需要它：若只在目标里给终端储电一个"价值"，线性目标会把储能一路推到
    # 上限（多出来的电量永远"值钱"），于是电池日复一日单向积累、吸收余量被吃光，
    # 随后任何真实的富余都无处安放。把末端钉在当日初值，等价于"计划必须能在当日
    # 自我平衡"，才是可长期连续执行的口径。跨日价值用 soc_end_min_kwh + 终端项表达。
    if soc_end_kwh is not None:
        m.soc_end = pyo.Constraint(expr=m.soc[N_INTERVAL] == float(soc_end_kwh))
    elif soc_end_min_kwh is not None and soc_end_max_equals:
        m.soc_end = pyo.Constraint(expr=m.soc[N_INTERVAL] == float(soc_end_min_kwh))
    elif soc_end_min_kwh is not None:
        m.soc_end_min = pyo.Constraint(expr=m.soc[N_INTERVAL] >= float(soc_end_min_kwh))


# ==========================================================================
# 问题一：确定性单日 LP
# ==========================================================================


def solve_problem1(
    day_input: DayInput,
    *,
    soc_start_kwh: float,
    soc_end_kwh: float,
    price_yuan_per_kwh: np.ndarray,
    solver_name: str = DEFAULT_SOLVER,
) -> PlanResult:
    """问题一：0 点制定当日计划，最小化当日购电费，首末 SOC 相等。

    本问没有紧急购电、日内调整和违约费用。
    """
    m = pyo.ConcreteModel("problem1")
    attach_physics(
        m,
        demand_kwh=day_input.demand_kwh,
        pv_kwh=day_input.pv_kwh,
        soc_start_kwh=soc_start_kwh,
        soc_end_kwh=soc_end_kwh,
        allow_emergency=False,
    )

    def _obj(m):
        return sum(price_yuan_per_kwh[t] * m.grid[t] for t in m.T)

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    report = solve_model(m, "problem1", solver_name=solver_name)
    if not report.feasible:
        return PlanResult(
            report=report,
            grid_kwh=np.zeros(N_INTERVAL),
            emergency_kwh=np.zeros(N_INTERVAL),
            charge_kwh=np.zeros(N_INTERVAL),
            discharge_kwh=np.zeros(N_INTERVAL),
            curtail_kwh=np.zeros(N_INTERVAL),
            soc_kwh=np.full(N_BOUNDARY, soc_start_kwh),
            objective_yuan=float("nan"),
        )
    return _extract(m, report)


# ==========================================================================
# 问题二 / 4-2：日固定计划 + 紧急补购 + 终端价值
# ==========================================================================


def solve_problem2(
    day_input: DayInput,
    *,
    soc_start_kwh: float,
    plan_price_yuan_per_kwh: np.ndarray,
    actual_price_for_emergency: np.ndarray,
    terminal_value_yuan_per_kwh: float = 0.0,
    solver_name: str = DEFAULT_SOLVER,
    problem_name: str = "problem2",
    absorption_floor_kwh: np.ndarray | None = None,
    absorption_pv_ceiling_kwh: np.ndarray | None = None,
    soc_reserve_kwh: float = 0.0,
    soc_end_min_kwh: float | None = None,
    soc_end_max_equals: bool = False,
    soc_upper_kwh: np.ndarray | None = None,
    soc_lower_kwh: np.ndarray | None = None,
) -> PlanResult:
    """问题二：0 点冻结当日普通购电计划，缺口按 5 倍价紧急补购。

    规划目标 = 计划购电费 + 期望紧急费 - 终端储电价值（V 不进入最终账单）。
    当日**没有日末 SOC 约束**；跨日联系由终端价值承担。

    ``absorption_floor_kwh`` / ``absorption_pv_ceiling_kwh`` /
    ``soc_reserve_kwh`` / ``soc_end_min_kwh`` 见 :func:`attach_physics`。
    """
    m = pyo.ConcreteModel(problem_name)
    attach_physics(
        m,
        demand_kwh=day_input.demand_kwh,
        pv_kwh=day_input.pv_kwh,
        soc_start_kwh=soc_start_kwh,
        soc_end_kwh=None,
        allow_emergency=True,
        absorption_floor_kwh=absorption_floor_kwh,
        absorption_pv_ceiling_kwh=absorption_pv_ceiling_kwh,
        soc_reserve_kwh=soc_reserve_kwh,
        soc_end_min_kwh=soc_end_min_kwh,
        soc_end_max_equals=soc_end_max_equals,
        soc_upper_kwh=soc_upper_kwh,
        soc_lower_kwh=soc_lower_kwh,
    )

    def _obj(m):
        plan_cost = sum(plan_price_yuan_per_kwh[t] * m.grid[t] for t in m.T)
        emergency_cost = sum(
            FEE_EMERGENCY * actual_price_for_emergency[t] * m.emergency[t] for t in m.T
        )
        terminal = terminal_value_yuan_per_kwh * m.soc[N_INTERVAL]
        return plan_cost + emergency_cost - terminal

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    report = solve_model(m, problem_name, solver_name=solver_name)
    if not report.feasible:
        return PlanResult(
            report=report,
            grid_kwh=np.zeros(N_INTERVAL),
            emergency_kwh=np.zeros(N_INTERVAL),
            charge_kwh=np.zeros(N_INTERVAL),
            discharge_kwh=np.zeros(N_INTERVAL),
            curtail_kwh=np.zeros(N_INTERVAL),
            soc_kwh=np.full(N_BOUNDARY, soc_start_kwh),
            objective_yuan=float("nan"),
        )
    return _extract(m, report)


# ==========================================================================
# 问题三 / 4-3：节点调整剩余时段
# ==========================================================================


def solve_problem3_remainder(
    *,
    day: object,
    t_start: int,
    soc_start_kwh: float,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    plan_price_yuan_per_kwh: np.ndarray,
    emergency_price_yuan_per_kwh: np.ndarray,
    terminal_value_yuan_per_kwh: float = 0.0,
    solver_name: str = DEFAULT_SOLVER,
    problem_name: str = "problem3-adjust",
    absorption_floor_kwh: np.ndarray | None = None,
    absorption_pv_ceiling_kwh: np.ndarray | None = None,
    soc_reserve_kwh: float = 0.0,
    soc_end_min_kwh: float | None = None,
    soc_end_max_equals: bool = False,
    soc_upper_kwh: np.ndarray | None = None,
    soc_lower_kwh: np.ndarray | None = None,
) -> PlanResult:
    """问题三/4-3：在调整节点重优化**尚未执行**的剩余时段。

    已执行区间不进入本模型（它的物理轨迹与费用已经发生，不可重写）。
    电价口径：
      * 计划价用**当前可用**的价格信息（问题三为重复曲线，4-3 为预测价）；
      * 紧急费按发生当时价乘 5 倍。
    """
    from .schemas import DayInput as _DayInput

    sub = _DayInput(
        day=day,  # type: ignore[arg-type]
        demand_kwh=np.asarray(demand_kwh, dtype=np.float64),
        pv_kwh=np.asarray(pv_kwh, dtype=np.float64),
        price_yuan_per_kwh=np.asarray(plan_price_yuan_per_kwh, dtype=np.float64),
    )
    m = pyo.ConcreteModel(problem_name)
    attach_physics(
        m,
        demand_kwh=sub.demand_kwh,
        pv_kwh=sub.pv_kwh,
        soc_start_kwh=soc_start_kwh,
        soc_end_kwh=None,
        allow_emergency=True,
        t_start=t_start,
        absorption_floor_kwh=absorption_floor_kwh,
        absorption_pv_ceiling_kwh=absorption_pv_ceiling_kwh,
        soc_reserve_kwh=soc_reserve_kwh,
        soc_end_min_kwh=soc_end_min_kwh,
        soc_end_max_equals=soc_end_max_equals,
        soc_upper_kwh=soc_upper_kwh,
        soc_lower_kwh=soc_lower_kwh,
    )

    def _obj(m):
        plan_cost = sum(plan_price_yuan_per_kwh[t] * m.grid[t] for t in m.T)
        emergency_cost = sum(
            FEE_EMERGENCY * emergency_price_yuan_per_kwh[t] * m.emergency[t] for t in m.T
        )
        terminal = terminal_value_yuan_per_kwh * m.soc[N_INTERVAL]
        return plan_cost + emergency_cost - terminal

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    report = solve_model(m, problem_name, solver_name=solver_name)
    grid = np.zeros(N_INTERVAL)
    emg = np.zeros(N_INTERVAL)
    ch = np.zeros(N_INTERVAL)
    dis = np.zeros(N_INTERVAL)
    cur = np.zeros(N_INTERVAL)
    soc = np.full(N_BOUNDARY, float(soc_start_kwh))
    if report.feasible:
        for t in m.T:
            grid[t] = float(pyo.value(m.grid[t]))
            ch[t] = float(pyo.value(m.charge[t]))
            dis[t] = float(pyo.value(m.discharge[t]))
            cur[t] = float(pyo.value(m.curtail[t]))
            if m.emergency is not None:
                emg[t] = float(pyo.value(m.emergency[t]))
        for s in m.S:
            soc[s] = float(pyo.value(m.soc[s]))
    return PlanResult(
        report=report,
        grid_kwh=grid,
        emergency_kwh=emg,
        charge_kwh=ch,
        discharge_kwh=dis,
        curtail_kwh=cur,
        soc_kwh=soc,
        objective_yuan=float(report.objective_yuan or float("nan")),
    )


# ==========================================================================
# 提取
# ==========================================================================


def _extract(m: pyo.ConcreteModel, report: SolveReport) -> PlanResult:
    grid = np.zeros(N_INTERVAL)
    emg = np.zeros(N_INTERVAL)
    ch = np.zeros(N_INTERVAL)
    dis = np.zeros(N_INTERVAL)
    cur = np.zeros(N_INTERVAL)
    soc = np.zeros(N_BOUNDARY)
    for t in m.T:
        grid[t] = float(pyo.value(m.grid[t]))
        ch[t] = float(pyo.value(m.charge[t]))
        dis[t] = float(pyo.value(m.discharge[t]))
        cur[t] = float(pyo.value(m.curtail[t]))
        if m.emergency is not None:
            emg[t] = float(pyo.value(m.emergency[t]))
    for s in m.S:
        soc[s] = float(pyo.value(m.soc[s]))
    return PlanResult(
        report=report,
        grid_kwh=grid,
        emergency_kwh=emg,
        charge_kwh=ch,
        discharge_kwh=dis,
        curtail_kwh=cur,
        soc_kwh=soc,
        objective_yuan=float(report.objective_yuan or float("nan")),
    )


__all__ = [
    "PlanResult",
    "attach_physics",
    "solve_problem1",
    "solve_problem2",
    "solve_problem3_remainder",
    "ETA_CHARGE",
]
