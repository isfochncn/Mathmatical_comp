"""仿真层：因果事件循环与实际状态推进。

职责（规范第 5 节）：
  * 逐事件把**当时可见**的信息交给策略，策略在结构上拿不到未来实测表；
  * 只有实际能量交换改变储电量；修改未来计划不直接改变 SOC；
  * 无拒收：多买的电必须被负载或储能消纳；消纳不了就报告不可行，
    **不得**用弃置外购电、无成本拒收或损耗烧电来补洞。

与计划的耦合方式
----------------
对问题二/4-2，0 点冻结的是**普通购电量 x**。储能的实际运行是对真实供需残差的
因果响应（规范第 10 节："策略冻结是购电计划冻结，储能执行仍对实际缺口作因果响应"）。
因此本模块按下列固定优先级推进，任何一步都不读取未来信息：

    供应 = x_t + 紧急量（待定） + G_t - r_t + q_dis_t
    需求 = D_t + q_ch_t / η_c

  1. 先用普通购电 x_t 与光伏供给负载；
  2. 有余电 -> 充电（受功率与容量上限），仍有余 -> 只好报告不可行（无合法消纳路径）；
  3. 有缺口 -> 放电（受功率与下限）；
  4. 仍缺 -> 紧急购电（5 倍价）补足，补不了 -> 报告不可行（不得拒收）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import physics
from .constants import (
    CHARGE_BUS_FACTOR,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    ETA_CHARGE,
    FeeClass,
    N_INTERVAL,
    Q_DIS_MAX,
    Q_MAX,
    TOL_ENERGY_KWH,
)
from .schemas import ActivePlan, DayInput, ExecutedEvent, Trajectory


class InfeasibleDay(RuntimeError):
    """当日没有合法消纳/供给路径——必须停止把该轨迹作为合格结果导出。"""

    def __init__(self, day: date, interval: int, message: str, context: dict[str, float]) -> None:
        self.day = day
        self.interval = interval
        self.context = context
        super().__init__(f"[{day} 区间 {interval}] {message}；上下文：{context}")


@dataclass
class CausalResult:
    """因果仿真的结果与诊断。"""

    trajectory: Trajectory
    infeasible: bool = False
    infeasible_reason: str = ""
    notes: list[str] = field(default_factory=list)


def simulate_day(
    day_input: DayInput,
    plan: ActivePlan,
    *,
    actual_demand_kwh: np.ndarray | None = None,
    actual_pv_kwh: np.ndarray | None = None,
    actual_price: np.ndarray | None = None,
    soc_start_kwh: float,
    soc_target_kwh: np.ndarray | None = None,
    strict: bool = True,
) -> CausalResult:
    """按因果规则推进一天。

    ``day_input`` 提供策略看到的（预测）量；``actual_*`` 提供真实发生的量
    （问题一/二/三中负载与光伏的真实值；问题四中价格也是真实的）。
    两者分开是防止信息泄漏的结构性保证。

    ``soc_target_kwh``
        计划给出的 SOC 目标轨迹（145 个边界）。执行层**跟踪**该轨迹：
        实际充放量按"当前段需要补多少偏差"确定，而不是见到富余就灌满电池、
        见到缺口就放空电池。这个区别很关键——盲目灌满会让电池在日内提前顶到
        上限，之后真实的富余就再也无处安放；盲目放空则会丢掉后面的调节能力。
        传 None 时退回"优先充放"的贪婪策略。
    """
    demand = np.asarray(
        day_input.demand_kwh if actual_demand_kwh is None else actual_demand_kwh, dtype=np.float64
    )
    pv = np.asarray(day_input.pv_kwh if actual_pv_kwh is None else actual_pv_kwh, dtype=np.float64)
    price = np.asarray(
        day_input.price_yuan_per_kwh if actual_price is None else actual_price, dtype=np.float64
    )
    plan_grid = np.asarray(plan.grid_kwh, dtype=np.float64)
    for name, arr in (("demand", demand), ("pv", pv), ("price", price), ("plan", plan_grid)):
        if arr.shape != (N_INTERVAL,):
            raise ValueError(f"{name} 形状应为 ({N_INTERVAL},)，得到 {arr.shape}")
    target = None if soc_target_kwh is None else np.asarray(soc_target_kwh, dtype=np.float64)
    if target is not None and target.shape != (N_INTERVAL + 1,):
        raise ValueError(f"soc_target_kwh 形状应为 ({N_INTERVAL + 1},)，得到 {target.shape}")

    h = np.zeros(N_INTERVAL)          # 实际普通购电
    u = np.zeros(N_INTERVAL)          # 实际紧急购电
    q_ch = np.zeros(N_INTERVAL)       # 母线侧充电量（未扣效率）
    q_dis = np.zeros(N_INTERVAL)      # 实际送达微网的放电量
    curtail = np.zeros(N_INTERVAL)    # 仅弃光
    soc = np.zeros(N_INTERVAL + 1)
    soc[0] = float(soc_start_kwh)

    notes: list[str] = []
    for t in range(N_INTERVAL):
        e = soc[t]
        headroom_kwh = (E_MAX - e) / ETA_CHARGE  # 还能接受多少"母线侧充电量"
        charge_cap = max(0.0, min(Q_MAX, headroom_kwh))
        discharge_avail = max(0.0, (e - E_MIN) * ETA_CHARGE)  # 能送出多少电量
        discharge_cap = max(0.0, min(Q_DIS_MAX, discharge_avail))

        h[t] = plan_grid[t]
        # 不充不放时的净缺口：正数表示缺电，负数表示富余
        imbalance = demand[t] - plan_grid[t] - pv[t]

        if target is not None:
            # 跟踪计划的 SOC 轨迹，但**只做真实供需允许的动作**：
            # 没有富余就不充（避免被计划推着把电池灌满而失去调节余量），
            # 没有缺口就不放（避免把调节能力提前用掉）。
            want = float(target[t + 1]) - e
            if want > 0.0:
                q_ch[t] = min(want, charge_cap, max(0.0, -imbalance))
            elif want < 0.0:
                q_dis[t] = min(-want, discharge_cap, max(0.0, imbalance))
        else:
            # 贪婪：见到富余就充、见到缺口就放
            if imbalance < 0.0:
                q_ch[t] = min(-imbalance, charge_cap)
            else:
                q_dis[t] = min(imbalance, discharge_cap)

        # 跟踪后仍有缺口：补齐（必要时紧急购电）
        remaining = imbalance - q_dis[t] + q_ch[t] / ETA_CHARGE
        if remaining > TOL_ENERGY_KWH:
            extra_dis = min(remaining, discharge_cap - q_dis[t])
            q_dis[t] += extra_dis
            remaining -= extra_dis
            if remaining > TOL_ENERGY_KWH:
                u[t] = remaining  # 紧急购电量随实际供需残差产生

        # 跟踪后仍有富余：继续充电，实在放不下才弃光
        surplus = -(imbalance - q_dis[t] + q_ch[t] / ETA_CHARGE)
        if surplus > TOL_ENERGY_KWH:
            extra_ch = min(surplus, charge_cap - q_ch[t])
            q_ch[t] += extra_ch
            surplus -= extra_ch
            if surplus > TOL_ENERGY_KWH:
                r = min(surplus, pv[t])
                curtail[t] = r
                surplus -= r
            if surplus > TOL_ENERGY_KWH:
                ctx = {
                    "计划购电_kwh": float(plan_grid[t]),
                    "负载_kwh": float(demand[t]),
                    "光伏_kwh": float(pv[t]),
                    "SOC_kwh": float(e),
                    "剩余无法消纳_kwh": float(surplus),
                }
                msg = (
                    "计划购电过量且储能已无可充容量，没有合法消纳路径"
                    "（不得弃置外购电、不得无成本拒收）"
                )
                if strict:
                    raise InfeasibleDay(day_input.day, t, msg, ctx)
                notes.append(f"{day_input.day} 区间 {t}：{msg}，缺口 {surplus:.3f} kWh")

        soc[t + 1] = physics.soc_next(e, q_ch[t], q_dis[t])

    # 事件记账：普通计划 1 倍、实际调整增购按调用方另行分类、紧急 5 倍
    events: list[ExecutedEvent] = []
    for t in range(N_INTERVAL):
        if h[t] > TOL_ENERGY_KWH:
            events.append(ExecutedEvent(day_input.day, t, float(h[t]), float(price[t]), FeeClass.NORMAL))
        if u[t] > TOL_ENERGY_KWH:
            events.append(
                ExecutedEvent(day_input.day, t, float(u[t]), float(price[t]), FeeClass.EMERGENCY)
            )

    trajectory = Trajectory(
        day=day_input.day,
        grid_actual_kwh=h,
        emergency_actual_kwh=u,
        charge_stored_kwh=q_ch,
        discharge_delivered_kwh=q_dis,
        curtail_kwh=curtail,
        soc_kwh=soc,
        price_actual=price,
        events=events,
        updates=[],
    )
    return CausalResult(trajectory=trajectory, notes=notes)


def simulate_day_relaxed(
    day_input: DayInput,
    plan: ActivePlan,
    *,
    soc_start_kwh: float,
    actual_demand_kwh: np.ndarray | None = None,
    actual_pv_kwh: np.ndarray | None = None,
    actual_price: np.ndarray | None = None,
    soc_target_kwh: np.ndarray | None = None,
) -> CausalResult:
    """:func:`simulate_day` 的不抛异常版本——把不可行如实记入诊断后继续。

    仅用于**诊断与统计不可行天数**；正式导出仍要求逐日可行。
    """
    kwargs = dict(
        actual_demand_kwh=actual_demand_kwh,
        actual_pv_kwh=actual_pv_kwh,
        actual_price=actual_price,
        soc_start_kwh=soc_start_kwh,
        soc_target_kwh=soc_target_kwh,
    )
    try:
        return simulate_day(day_input, plan, strict=True, **kwargs)
    except InfeasibleDay as exc:
        result = simulate_day(day_input, plan, strict=False, **kwargs)
        result.infeasible = True
        result.infeasible_reason = str(exc)
        result.notes.append(str(exc))
        return result


# ==========================================================================
# 滚动调度执行（问题二/4-2/三/4-3 的正式口径）
# ==========================================================================


def simulate_day_rolling(
    day_input: DayInput,
    plan: ActivePlan,
    *,
    soc_start_kwh: float,
    actual_demand_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    actual_price: np.ndarray,
    window_intervals: int = 6,
    lookahead_intervals: int = 12,
    allow_emergency: bool = True,
    solver_name: str | None = None,
) -> CausalResult:
    """普通购电计划已冻结时的正式执行方式：逐窗口重算储能充放。

    每个决策窗口只用**已揭示**的信息：
      * 已完成区间：真实轨迹（不可改写）；
      * 当前窗口：真实负载/光伏（本段已经发生，属于当前观测）；
      * 更远的未来：策略此前给出的预测（``day_input`` 里的 demand/pv）。

    计划签署的普通购电量在窗口内固定为 ``plan.grid_kwh``，不可拒收；
    储能通过 LP 决定充放，尽量避免紧急购电、也避免出现无处安放的富余。
    """
    from .dispatch import solve_dispatch_window
    from .solver import DEFAULT_SOLVER

    solver = solver_name or DEFAULT_SOLVER
    demand_true = np.asarray(actual_demand_kwh, dtype=np.float64)
    pv_true = np.asarray(actual_pv_kwh, dtype=np.float64)
    price = np.asarray(actual_price, dtype=np.float64)
    grid_plan = np.asarray(plan.grid_kwh, dtype=np.float64)

    h = np.zeros(N_INTERVAL)
    u = np.zeros(N_INTERVAL)
    q_ch = np.zeros(N_INTERVAL)
    q_dis = np.zeros(N_INTERVAL)
    curtail = np.zeros(N_INTERVAL)
    spill = np.zeros(N_INTERVAL)
    soc = np.zeros(N_INTERVAL + 1)
    soc[0] = float(soc_start_kwh)

    notes: list[str] = []
    steps = max(1, int(window_intervals))
    look = max(steps, int(lookahead_intervals))

    t = 0
    while t < N_INTERVAL:
        t_end = min(N_INTERVAL, t + steps)
        horizon_end = min(N_INTERVAL, t + look)
        # 视界内的边界：当前窗口用实测，更远处用当时可见的预测。
        # 数组保持全 144 段索引（[0, horizon_end) 覆盖实测窗口，其余用预测），
        # 避免把"窗口内相对位移"误当成"全天绝对位移"。
        demand_h = np.array(day_input.demand_kwh, dtype=np.float64, copy=True)
        pv_h = np.array(day_input.pv_kwh, dtype=np.float64, copy=True)
        demand_h[t:horizon_end] = demand_true[t:horizon_end]
        pv_h[t:horizon_end] = pv_true[t:horizon_end]
        try:
            decisions = solve_dispatch_window(
                grid_plan_kwh=grid_plan,
                demand_kwh=demand_h,
                pv_kwh=pv_h,
                price_actual_yuan_per_kwh=price,
                soc_start_kwh=soc[t],
                t_start=t,
                t_end=t_end,
                horizon_end=horizon_end,
                solver_name=solver,
            )
        except RuntimeError as exc:  # 求解失败必须如实报告，不得伪造
            if not allow_emergency:
                raise
            notes.append(f"{day_input.day} 窗口 {t}-{t_end} 调度失败：{exc}")
            decisions = _fallback_dispatch(
                grid_plan, demand_true, pv_true, soc[t], t, t_end
            )

        for step in decisions:
            i = step.interval
            h[i] = step.grid_kwh
            u[i] = step.emergency_kwh
            q_ch[i] = step.charge_kwh
            q_dis[i] = step.discharge_kwh
            curtail[i] = step.curtail_kwh
            spill[i] = step.spill_kwh
            soc[i + 1] = step.soc_end_kwh
        t = t_end

    total_spill = float(np.sum(spill))
    infeasible = total_spill > 1e-6
    if infeasible:
        notes.append(
            f"{day_input.day} 出现无处安放的富余 {total_spill:.3f} kWh"
            "（计划购电过量且储能已无可充容量）"
        )

    events = _build_events(day_input.day, h, u, price)
    trajectory = Trajectory(
        day=day_input.day,
        grid_actual_kwh=h,
        emergency_actual_kwh=u,
        charge_stored_kwh=q_ch,
        discharge_delivered_kwh=q_dis,
        curtail_kwh=curtail,
        soc_kwh=soc,
        price_actual=price,
        surplus_disposed_kwh=spill,
        events=events,
        updates=[],
    )
    result = CausalResult(trajectory=trajectory, notes=notes)
    result.infeasible = infeasible
    result.infeasible_reason = notes[-1] if infeasible else ""
    return result


def _fallback_dispatch(
    grid_plan: np.ndarray,
    demand_true: np.ndarray,
    pv_true: np.ndarray,
    soc_start: float,
    t_start: int,
    t_end: int,
):
    """调度 LP 意外失败时的保守退化：只做当前残差的因果响应。"""
    from .dispatch import DispatchStep

    out = []
    soc = float(soc_start)
    for t in range(t_start, t_end):
        headroom = (E_MAX - soc) / ETA_CHARGE
        charge_cap = max(0.0, min(Q_MAX, headroom))
        discharge_cap = max(0.0, min(Q_DIS_MAX, max(0.0, (soc - E_MIN) * ETA_CHARGE)))
        imbalance = demand_true[t] - grid_plan[t] - pv_true[t]
        ch = dis = emg = spill = 0.0
        if imbalance < 0.0:
            ch = min(-imbalance, charge_cap)
            spill = max(0.0, -imbalance - ch)
        else:
            dis = min(imbalance, discharge_cap)
            emg = max(0.0, imbalance - dis)
        soc = physics.soc_next(soc, ch, dis)
        out.append(
            DispatchStep(
                interval=t,
                grid_kwh=float(grid_plan[t]),
                emergency_kwh=emg,
                charge_kwh=ch,
                discharge_kwh=dis,
                curtail_kwh=0.0,
                soc_end_kwh=soc,
                spill_kwh=spill,
            )
        )
    return out


def _build_events(day, h: np.ndarray, u: np.ndarray, price: np.ndarray) -> list[ExecutedEvent]:
    events: list[ExecutedEvent] = []
    for t in range(N_INTERVAL):
        if h[t] > TOL_ENERGY_KWH:
            events.append(ExecutedEvent(day, t, float(h[t]), float(price[t]), FeeClass.NORMAL))
        if u[t] > TOL_ENERGY_KWH:
            events.append(ExecutedEvent(day, t, float(u[t]), float(price[t]), FeeClass.EMERGENCY))
    return events


# ==========================================================================
# 信息视图：只把"当前已可见"的东西交给策略
# ==========================================================================


@dataclass(frozen=True)
class InformationView:
    """规范第 5 节要求的结构性隔离。

    计划模块只能拿到本对象，拿不到 ``Attachment2``/``Attachment4`` 的整表。
    """

    day: date
    interval: int
    known_demand_kwh: np.ndarray       # (144,) 预测值，不是真值
    known_pv_kwh: np.ndarray           # (144,) 预报值（取自已发布版本）
    known_price_yuan_per_kwh: np.ndarray
    soc_kwh: float
    pv_source: str = ""
    price_source: str = ""

    def assert_no_future_leak(self) -> None:
        """兜底检查：本视图不得携带"未来真实"标记。"""
        if "actual" in self.pv_source.lower():
            raise ValueError(f"信息视图泄露光伏实测：{self.pv_source}")
        if "actual" in self.price_source.lower():
            raise ValueError(f"信息视图泄露电价实测：{self.price_source}")


__all__ = [
    "InfeasibleDay",
    "CausalResult",
    "simulate_day",
    "simulate_day_relaxed",
    "InformationView",
    "Q_DIS_MAX",
    "E_MIN",
    "E_MAX",
    "CHARGE_BUS_FACTOR",
    "DISCHARGE_BATTERY_FACTOR",
]
