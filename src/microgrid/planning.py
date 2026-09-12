"""Rolling-window planning models.

The 2026-09-11 rules replaced the old "one natural day + abstract terminal
value" formulation with a single deterministic rolling window:

    a = the start of the current ten-minute interval
    b = the next natural day 00:00
    T = b + 24h
    solve once over [a, T)

so the 00:00 window spans 48 h and the 06:00 / 12:00 / 18:00 windows span
42 / 36 / 30 h; at other intervals the window is the remaining part of [a, T).

The cross-day value is **not** a fitted terminal term. It is exactly the cost
of the next-day predicted dispatch, which is what a joint optimisation over
[b, T) already computes. At T only the device capacity bounds apply: no
salvage revenue, no terminal reward, no forced return to 6000.

Three fee expressions (see the memo section 6)
----------------------------------------------
* first-day plan (no adjustment possible yet):  ``f = p * x``
* frozen segment (P2 / P4-2, and P3/P4-3 outside adjust nodes):
  ``f = p * (O + 1.5*A)``
* adjustable segment (P3 / P4-3 at 06/12/18):
  ``f = p*y' + 0.5*p*[y' - O]+ + 0.5*p_a*[O + A - y']+``

All three are convex piecewise linear in y', so the whole model stays an LP.

Structure reuse (2026 performance fix)
--------------------------------------
The rolling driver calls :func:`solve_window` 144 times per simulated day, and
consecutive windows differ **only in parameter values** (demand, PV, price, the
SOC start value, the absorption cap / commitment floor, the committed O/A).
Rebuilding the Pyomo model and re-transferring it to HiGHS every time used to
cost several hundred milliseconds per solve, dwarfing the LP itself.

:func:`solve_window` therefore keeps a small LRU cache of built models keyed by
a *structural signature* (:class:`_WindowShape`: window length, fee mode, which
optional constraint blocks exist, and the committed masks). On a hit the model
is reused and only its mutable ``Param`` values are rewritten in place, then the
same appsi solver instance is asked to solve again — appsi then takes its
``update()`` path and pushes just the changed bounds/coefficients to HiGHS.

Everything that varies between solves is a ``mutable=True`` ``Param`` (including
the SOC start / end values), so the variable set, the constraint set with its
index sets, the objective expression and every bound value are *identical* to
what the previous one-shot construction produced.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyomo.environ as pyo

from .constants import (
    CHARGE_BUS_FACTOR,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    FEE_ADJUST_UP,
    FEE_EMERGENCY,
    FEE_PENALTY_DOWN,
    Q_DIS_MAX,
    Q_MAX,
)
from .schemas import SolveReport
from .solver import DEFAULT_SOLVER, make_appsi_solver, solve_model


@dataclass
class WindowForecast:
    """One point-forecast trajectory covering the whole optimisation window."""

    abs_minutes: np.ndarray      # (n,) window interval start minutes
    demand_kwh: np.ndarray       # (n,)
    pv_kwh: np.ndarray           # (n,)
    price_yuan_per_kwh: np.ndarray  # (n,) planning price p_hat_{s|a}
    provenance: np.ndarray       # (n,) of str, same values as timeline markers
    reserve_energy_kwh: np.ndarray | None = None  # (n+1,) delivered-energy buffer
    net_upper_kwh: np.ndarray | None = None       # (n,) calibrated upper net load

    def __post_init__(self) -> None:
        n = np.asarray(self.abs_minutes, dtype=np.int64).size
        self.abs_minutes = np.asarray(self.abs_minutes, dtype=np.int64)
        if self.abs_minutes.shape != (n,) or np.any(self.abs_minutes % 10) or np.any(np.diff(self.abs_minutes) != 10):
            raise ValueError("Window timestamps must be contiguous ten-minute starts")
        for name in ("demand_kwh", "pv_kwh", "price_yuan_per_kwh", "provenance"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(f"WindowForecast.{name} shape should be ({n},), got {arr.shape}")
            setattr(self, name, arr)

    @property
    def n(self) -> int:
        return int(self.abs_minutes.size)


@dataclass
class WindowResult:
    """A solved window: full schedule plus the solver report."""

    report: SolveReport
    abs_minutes: np.ndarray
    grid_kwh: np.ndarray
    emergency_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    curtail_kwh: np.ndarray
    soc_boundary_kwh: np.ndarray
    objective_yuan: float
    spill_kwh: np.ndarray = field(default_factory=lambda: np.empty(0))
    reserve_shortfall_kwh: float = 0.0
    risk_penalty_yuan: float = 0.0  # planning preference only; never a settled bill
    reserve_stock_shortfall_kwh: float = 0.0
    reserve_power_shortfall_kwh: np.ndarray = field(default_factory=lambda: np.empty(0))

    def require_ok(self) -> "WindowResult":
        self.report.require_ok()
        return self


# ==========================================================================
# Fee expressions
# ==========================================================================


class FeeMode:
    """Which fee expression applies to the grid variable in a window."""

    FIRST_PLAN = "first_plan"     # f = p * x  (no O/A yet)
    FROZEN = "frozen"             # f = p * (O + 1.5 A)  -- O, A constant
    ADJUSTABLE = "adjustable"     # convex piecewise linear in y'


def _add_frozen_fee(m: pyo.ConcreteModel, frozen_mask: np.ndarray) -> None:
    """Frozen segment: O and A are constants, so the fee is a linear function.

    ``frozen_mask``
        Which intervals are actually under commitment. Only those get
        ``grid == O + A`` and the constant fee ``p*(O + 1.5*A)``; the rest of
        the window is still free and buys at the normal rate. Forcing the whole
        window to the committed quantity would pin the uncommitted tail to zero
        and make the model infeasible whenever that tail has to serve load.

        这里的掩码语义与结构复用前完全一致，只有一点不同：承诺量与常数费用
        存在模型的可变参数 ``m.commit_val`` / ``m.fee_const`` 里，复用模型时
        就地改写，不再重建 ``m.commit`` 的索引集合。
    """

    def _rule(m, t):
        if frozen_mask[t]:
            return m.fee_const[t]
        return m.price[t] * m.grid[t]

    m.fee_fixed = pyo.Expression(m.T, rule=_rule)

    fixed = [t for t in m.T if frozen_mask[t]]
    if fixed:
        m.commit = pyo.Constraint(
            fixed, rule=lambda m, t: m.grid[t] == m.commit_val[t]
        )


def _add_adjustable_fee(m: pyo.ConcreteModel, adjustable_mask: np.ndarray) -> None:
    """Adjustable segment: convex piecewise-linear fee via the epigraph of [.]_+.

    f(y') = p_s*y' + 0.5*p_s*v + 0.5*p_a*w,  v >= y'-O, v >= 0, w >= O+A-y', w >= 0

    The **actual** fee labels are always recomputed from the realised y' with the
    exact min/positive-part formulas; the auxiliary variables are only a modelling
    device (and they may be slack when a price is zero).
    """
    m.increase = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # v
    m.reduce = pyo.Var(m.T, domain=pyo.NonNegativeReals)     # w

    m.increase_lb = pyo.Constraint(
        m.T, rule=lambda m, t: m.increase[t] >= m.grid[t] - m.o_kwh[t]
    )
    m.reduce_lb = pyo.Constraint(
        m.T, rule=lambda m, t: m.reduce[t] >= m.oa_kwh[t] - m.grid[t]
    )

    def _rule(m, t):
        if not adjustable_mask[t]:
            return m.price[t] * m.grid[t]
        return (
            m.price[t] * m.grid[t]
            + (FEE_ADJUST_UP - 1.0) * m.price[t] * m.increase[t]
            + FEE_PENALTY_DOWN * m.price_now * m.reduce[t]
        )

    m.fee_fixed = pyo.Expression(m.T, rule=_rule)


def _add_first_plan_fee(m: pyo.ConcreteModel) -> None:
    """First plan of the day: plain normal-rate purchase, no adjustment fees."""
    m.fee_fixed = pyo.Expression(m.T, rule=lambda m, t: m.price[t] * m.grid[t])


# ==========================================================================
# Structural signature and the reuse cache
# ==========================================================================

#: Cache the shrinking ten-minute windows; eviction affects performance only.
MAX_CACHED_WINDOWS = 160

#: 结构指纹 -> 已建好的 Pyomo 模型 + 绑定其上的 appsi 求解器。
_CACHE: "OrderedDict[_WindowShape, _WindowTemplate]" = OrderedDict()
_CACHE_STATS = {"hit": 0, "miss": 0, "evict": 0}


@dataclass(frozen=True)
class _WindowShape:
    """窗口 LP 的结构指纹。

    模型的变量、约束（含索引集合与顺序）、目标表达式只由这些量决定；其余
    输入全是可变参数。指纹相同的两次求解因此可以共用同一个模型对象，
    数值上与每次重建完全等价。注意 ``problem_name`` 不进指纹：滚动驱动
    每次求解都传 ``f"{policy}@{abs_minute}"``，把它当结构特征会让缓存永不命中。
    """

    n: int
    fee_mode: str
    #: 是否加了鲁棒消纳上限（只有未承诺区间才有）
    absorption: bool
    #: 是否加了承诺下限（同上）
    commitment_floor: bool
    #: 是否固定末尾 SOC（仅问题一）
    soc_end_fixed: bool
    #: FROZEN 模式下被冻结（grid == O + A）的区间掩码；None 表示该费用式不用掩码
    frozen_mask: bytes | None
    #: 被 fix_committed 钉住的区间掩码（承诺结转）；None 表示没有结转
    committed_mask: bytes | None
    adjustable_mask: bytes | None
    allow_emergency: bool
    allow_spill: bool
    risk_reserve: bool


@dataclass
class _WindowTemplate:
    """缓存中的一个窗口 LP，以及绑定其上的求解器实例。"""

    shape: _WindowShape
    model: pyo.ConcreteModel
    solver: Any = None            # 原生 appsi Highs 实例；与 model 成对使用
    solver_name: str | None = None


def window_cache_info() -> dict[str, int]:
    """结构复用缓存的命中/未命中/淘汰统计，便于观测。"""
    info = dict(_CACHE_STATS)
    info["size"] = len(_CACHE)
    info["max_size"] = MAX_CACHED_WINDOWS
    return info


def clear_window_cache() -> None:
    """清空结构复用缓存（换问题或做对照实验时用，避免旧模型占内存）。"""
    _CACHE.clear()
    _CACHE_STATS.update(hit=0, miss=0, evict=0)


def _window_shape(
    *,
    n: int,
    fee_mode: str,
    absorption_upper_kwh: np.ndarray | None,
    commitment_lower_kwh: np.ndarray | None,
    soc_end_fixed_kwh: float | None,
    committed_grid_kwh: np.ndarray | None,
    committed_mask: np.ndarray | None,
    adjustable_mask: np.ndarray | None,
    allow_emergency: bool,
    allow_spill: bool,
    risk_reserve: bool,
) -> _WindowShape:
    """由本次调用的输入推出结构指纹（只取真正影响模型形状的部分）。"""
    mask = None
    if committed_mask is not None:
        mask = np.ascontiguousarray(np.asarray(committed_mask, dtype=bool))
    fix_mask = mask if (mask is not None and committed_grid_kwh is not None) else None
    if fee_mode == FeeMode.FROZEN:
        frozen = mask if mask is not None else np.ones(n, dtype=bool)
    else:
        frozen = None
    return _WindowShape(
        n=n,
        adjustable_mask=None if adjustable_mask is None else np.ascontiguousarray(adjustable_mask, dtype=bool).tobytes(),
        allow_emergency=allow_emergency,
        allow_spill=allow_spill,
        risk_reserve=risk_reserve,
        fee_mode=fee_mode,
        absorption=absorption_upper_kwh is not None and committed_mask is None,
        commitment_floor=commitment_lower_kwh is not None and committed_mask is None,
        soc_end_fixed=soc_end_fixed_kwh is not None,
        frozen_mask=None if frozen is None else frozen.tobytes(),
        committed_mask=None if fix_mask is None else fix_mask.tobytes(),
    )


def _evict_oldest() -> None:
    while len(_CACHE) > MAX_CACHED_WINDOWS:
        _CACHE.popitem(last=False)
        _CACHE_STATS["evict"] += 1


# ==========================================================================
# Window model construction and in-place data loading
# ==========================================================================


def _load(param: pyo.Param, values: np.ndarray) -> None:
    """把一维数组写回索引参数。

    ``m.T`` 是有序集合 0..n-1，索引参数按同样顺序构造，因此
    ``param.values()`` 与数组下标一一对应。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != len(param):
        raise ValueError(f"参数 {param.name} 需要 {len(param)} 个值，收到 {arr.size} 个")
    for data, v in zip(param.values(), arr):
        data.set_value(float(v))


def _build_window_model(
    shape: _WindowShape,
    *,
    model_name: str,
    forecast: WindowForecast,
    absorption_upper_kwh: np.ndarray | None,
    commitment_lower_kwh: np.ndarray | None,
    committed_mask: np.ndarray | None,
) -> pyo.ConcreteModel:
    """按结构指纹构造窗口 LP（组件顺序与逐次重建时逐字一致）。

    与逐次重建的唯一区别：所有每次求解都会变的输入都是 ``mutable=True`` 的
    ``Param``（包括 SOC 初值/末值），复用时就地改写取值即可，模型结构不变。
    """
    n = shape.n
    T = list(range(n))
    mask = None
    if committed_mask is not None:
        mask = np.asarray(committed_mask, dtype=bool)
    frozen = None
    if shape.frozen_mask is not None:
        frozen = np.frombuffer(shape.frozen_mask, dtype=bool)

    m = pyo.ConcreteModel(model_name)
    m.T = pyo.Set(initialize=T, ordered=True)
    m.S = pyo.Set(initialize=list(range(n + 1)), ordered=True)

    m.demand = pyo.Param(
        m.T, mutable=True, initialize={t: float(forecast.demand_kwh[t]) for t in T}
    )
    m.pv = pyo.Param(
        m.T, mutable=True, initialize={t: float(forecast.pv_kwh[t]) for t in T}
    )
    m.price = pyo.Param(
        m.T, mutable=True, initialize={t: float(forecast.price_yuan_per_kwh[t]) for t in T}
    )
    m.soc_start_kwh = pyo.Param(mutable=True, initialize=0.0)

    # Disposal is a fallback only for deliveries whose quantities are frozen.
    frozen_for_spill = np.zeros(n, dtype=bool)
    for raw_mask in (shape.frozen_mask, shape.committed_mask):
        if raw_mask is not None:
            frozen_for_spill |= np.frombuffer(raw_mask, dtype=bool)
    m.spill = pyo.Var(m.T, domain=pyo.NonNegativeReals,
                     bounds=lambda m, t: (0, None if shape.allow_spill and frozen_for_spill[t] else 0))
    m.grid = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.emergency = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, None if shape.allow_emergency else 0))
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_MAX))
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, Q_DIS_MAX))
    m.curtail = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.S, domain=pyo.NonNegativeReals, bounds=(E_MIN, E_MAX))

    # Bus conservation, 2026-09-11 physics core (unchanged).
    m.balance = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.grid[t]
        + m.emergency[t]
        + m.pv[t]
        - m.curtail[t]
        + m.discharge[t]
        == m.demand[t] + m.charge[t] * CHARGE_BUS_FACTOR + m.spill[t],
    )

    m.transition = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.soc[t + 1]
        == m.soc[t] + m.charge[t] - m.discharge[t] * DISCHARGE_BATTERY_FACTOR,
    )

    m.curtail_limit = pyo.Constraint(m.T, rule=lambda m, t: m.curtail[t] <= m.pv[t])
    m.soc_start = pyo.Constraint(expr=m.soc[0] == m.soc_start_kwh)

    # Direct grid supply serves demand without passing through the battery.
    # Only the surplus after serving demand must fit into charging. These named
    # inequalities make the implications of balance/transition/bounds explicit;
    # they use this window's forecast, never future actual demand.
    m.surplus_to_charge = pyo.Expression(
        m.T, rule=lambda m, t: m.grid[t] + m.emergency[t] + m.pv[t]
        - m.curtail[t] + m.discharge[t] - m.demand[t] - m.spill[t],
    )
    m.purchase_absorption_power = pyo.Constraint(
        m.T, rule=lambda m, t: m.surplus_to_charge[t] <= Q_MAX * CHARGE_BUS_FACTOR,
    )
    m.purchase_absorption_capacity = pyo.Constraint(
        m.T, rule=lambda m, t: m.surplus_to_charge[t]
        <= (E_MAX - m.soc[t] + m.discharge[t] * DISCHARGE_BATTERY_FACTOR) * CHARGE_BUS_FACTOR,
    )

    m.spill_from_paid_purchase = pyo.Constraint(m.T, rule=lambda m, t: m.spill[t] <= m.grid[t])
    if shape.allow_spill:
        m.discharge_serves_load = pyo.Constraint(m.T, rule=lambda m, t: m.discharge[t] <= m.demand[t])
        # Do not discharge merely to dissipate a fixed overpurchase. This cap
        # is based on a known fixed delivery, so the formulation stays linear.
        m.uncovered_load = pyo.Param(m.T, mutable=True, initialize=0.0)
        fixed_slots = [t for t in T if frozen_for_spill[t]]
        m.discharge_to_uncovered_load = pyo.Constraint(
            fixed_slots, rule=lambda m, t: m.discharge[t] <= m.uncovered_load[t])

    if shape.absorption:
        cap = np.asarray(absorption_upper_kwh, dtype=np.float64)
        m.cap = pyo.Param(m.T, mutable=True, initialize={t: float(cap[t]) for t in T})
        m.absorption = pyo.Constraint(m.T, rule=lambda m, t: m.grid[t] <= m.cap[t])

    if shape.commitment_floor:
        lo = np.asarray(commitment_lower_kwh, dtype=np.float64)
        m.floor = pyo.Param(m.T, mutable=True, initialize={t: float(lo[t]) for t in T})
        m.commitment_floor = pyo.Constraint(m.T, rule=lambda m, t: m.grid[t] >= m.floor[t])

    # ---- carry-over of already committed quantities -------------------------
    if shape.committed_mask is not None:
        m.fix_val = pyo.Param(m.T, mutable=True, initialize=0.0)
        fixed = [t for t in T if mask[t]]
        if fixed:
            m.fix_committed = pyo.Constraint(
                fixed, rule=lambda m, t: m.grid[t] == m.fix_val[t]
            )

    # ---- fee expression ------------------------------------------------------
    #
    # The robust purchase bounds (absorption cap / commitment floor) are guards on
    # the *decision* of how much to buy. They must NOT be applied to intervals
    # whose quantity is already committed: those are equality-fixed and re-testing
    # them against a bound computed from a different forecast is what made the
    # frozen window come out infeasible.
    if shape.fee_mode == FeeMode.FIRST_PLAN:
        _add_first_plan_fee(m)
    elif shape.fee_mode == FeeMode.FROZEN:
        m.commit_val = pyo.Param(m.T, mutable=True, initialize=0.0)
        m.fee_const = pyo.Param(m.T, mutable=True, initialize=0.0)
        _add_frozen_fee(m, frozen)
    elif shape.fee_mode == FeeMode.ADJUSTABLE:
        m.o_kwh = pyo.Param(m.T, mutable=True, initialize=0.0)
        m.oa_kwh = pyo.Param(m.T, mutable=True, initialize=0.0)
        m.price_now = pyo.Param(mutable=True, initialize=0.0)
        _add_adjustable_fee(m, np.ones(n, dtype=bool) if shape.adjustable_mask is None
                            else np.frombuffer(shape.adjustable_mask, dtype=bool))
    else:
        raise ValueError(f"unknown fee mode: {shape.fee_mode}")

    m.fee_objective = pyo.Expression(
        rule=lambda m: sum(m.fee_fixed[t] for t in m.T)
        + sum(FEE_EMERGENCY * m.price[t] * m.emergency[t] for t in m.T),
    )
    m.risk_cost = pyo.Expression(expr=0.)
    if shape.risk_reserve:
        m.reserve_energy = pyo.Param(m.S, mutable=True, initialize=0.)
        m.net_upper = pyo.Param(m.T, mutable=True, initialize=0.)
        m.risk_price = pyo.Param(mutable=True, initialize=0.)
        # One maximum deficit avoids charging repeatedly for the same stored
        # reserve across consecutive intervals. Slack keeps frozen plans feasible.
        m.reserve_shortfall = pyo.Var(domain=pyo.NonNegativeReals)
        m.power_shortfall = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.reserve_stock = pyo.Constraint(m.S, rule=lambda m, s:
            pyo.Constraint.Skip if s == 0 else
            (m.soc[s]-E_MIN)/DISCHARGE_BATTERY_FACTOR + m.reserve_shortfall >= m.reserve_energy[s])
        m.reserve_power = pyo.Constraint(m.T, rule=lambda m, t:
            m.grid[t]+Q_DIS_MAX+m.power_shortfall[t] >= m.net_upper[t])
        m.reserve_deliverable = pyo.Constraint(m.T, rule=lambda m, t:
            m.grid[t]+(m.soc[t]-E_MIN)/DISCHARGE_BATTERY_FACTOR+m.power_shortfall[t] >= m.net_upper[t])
        # Do not manufacture room for a *new reserve purchase* through cycling
        # losses. Paid surplus must fit into space available at the interval start.
        m.reserve_absorption = pyo.Constraint(m.T, rule=lambda m, t:
            pyo.Constraint.Skip if frozen_for_spill[t] else
            m.grid[t] <= m.demand[t]+(E_MAX-m.soc[t])*CHARGE_BUS_FACTOR)
        m.risk_cost.set_value(m.risk_price*m.reserve_shortfall
            + sum(FEE_EMERGENCY*m.price[t]*m.power_shortfall[t] for t in m.T))
    m.obj = pyo.Objective(expr=m.fee_objective+m.risk_cost, sense=pyo.minimize)

    if shape.allow_spill:
        m.spill_limit = pyo.Param(mutable=True, initialize=0.0)
        m.cost_limit = pyo.Param(mutable=True, initialize=0.0)
        m.keep_spill = pyo.Constraint(expr=sum(m.spill[t] for t in m.T) <= m.spill_limit)
        m.keep_cost = pyo.Constraint(expr=m.obj.expr <= m.cost_limit)
        m.min_spill = pyo.Objective(expr=sum(m.spill[t] for t in m.T))
        # At equal disposal and fee, prefer less early grid purchasing: stored
        # energy serves load first without sacrificing price arbitrage.
        m.prefer_storage = pyo.Objective(expr=sum((n-t)*(m.grid[t]+m.emergency[t]) for t in m.T))
        for component in (m.keep_spill, m.keep_cost, m.min_spill, m.prefer_storage):
            component.deactivate()

    if shape.soc_end_fixed:
        m.soc_end_kwh = pyo.Param(mutable=True, initialize=0.0)
        m.soc_end = pyo.Constraint(expr=m.soc[n] == m.soc_end_kwh)

    return m


def _load_window_inputs(
    m: pyo.ConcreteModel,
    shape: _WindowShape,
    *,
    forecast: WindowForecast,
    soc_start_kwh: float,
    o_eff: np.ndarray,
    a_eff: np.ndarray,
    absorption_upper_kwh: np.ndarray | None,
    commitment_lower_kwh: np.ndarray | None,
    committed_grid_kwh: np.ndarray | None,
    soc_end_fixed_kwh: float | None,
    price_now_yuan_per_kwh: float | None,
) -> None:
    """把本次求解的全部输入就地写进缓存模型的参数（不改结构）。

    每条与旧实现逐位一致：承诺量取 ``float(O + A)``，冻结段的常数费用取
    ``p * (O + 1.5*A)``，都是同一批 IEEE 双精度运算，顺序也相同。
    """
    _load(m.demand, forecast.demand_kwh)
    _load(m.pv, forecast.pv_kwh)
    _load(m.price, forecast.price_yuan_per_kwh)
    m.soc_start_kwh.set_value(float(soc_start_kwh))
    if shape.risk_reserve:
        _load(m.reserve_energy, forecast.reserve_energy_kwh)
        _load(m.net_upper, forecast.net_upper_kwh)
        # Five times the mean forecast tariff prices an uncovered delivered kWh.
        # This is a soft planning tradeoff, explicitly excluded from settlement.
        m.risk_price.set_value(FEE_EMERGENCY*float(np.mean(forecast.price_yuan_per_kwh)))

    if shape.absorption:
        _load(m.cap, absorption_upper_kwh)
    if shape.commitment_floor:
        _load(m.floor, commitment_lower_kwh)

    if shape.allow_spill:
        fixed_qty = o_eff + a_eff if committed_grid_kwh is None else committed_grid_kwh
        _load(m.uncovered_load, np.maximum(forecast.demand_kwh - fixed_qty, 0))

    if shape.fee_mode == FeeMode.FROZEN:
        _load(m.commit_val, o_eff + a_eff)
        _load(
            m.fee_const,
            np.asarray(forecast.price_yuan_per_kwh, dtype=np.float64)
            * (o_eff + FEE_ADJUST_UP * a_eff),
        )
    elif shape.fee_mode == FeeMode.ADJUSTABLE:
        _load(m.o_kwh, o_eff)
        _load(m.oa_kwh, o_eff + a_eff)
        m.price_now.set_value(float(price_now_yuan_per_kwh))

    if shape.committed_mask is not None:
        _load(m.fix_val, committed_grid_kwh)
    if shape.soc_end_fixed:
        m.soc_end_kwh.set_value(float(soc_end_fixed_kwh))


# ==========================================================================
# Window solve
# ==========================================================================


def solve_window(
    *,
    forecast: WindowForecast,
    soc_start_kwh: float,
    fee_mode: str,
    o_kwh: np.ndarray | None = None,
    a_kwh: np.ndarray | None = None,
    price_now_yuan_per_kwh: float | None = None,
    soc_end_fixed_kwh: float | None = None,
    absorption_upper_kwh: np.ndarray | None = None,
    commitment_lower_kwh: np.ndarray | None = None,
    committed_grid_kwh: np.ndarray | None = None,
    committed_mask: np.ndarray | None = None,
    adjustable_mask: np.ndarray | None = None,
    allow_emergency: bool = True,
    allow_spill: bool = False,
    solver_name: str = DEFAULT_SOLVER,
    problem_name: str = "window",
) -> WindowResult:
    """Solve one rolling window over the forecast's absolute intervals.

    Parameters
    ----------
    soc_start_kwh
        Measured state of charge at ``a`` (never a predicted SOC).
    fee_mode
        One of :class:`FeeMode`.
    o_kwh, a_kwh
        Committed original-plan / adjustment-purchase remainders. Needed by the
        FROZEN and ADJUSTABLE fee expressions. Intervals that are not committed
        yet are passed as 0 and are freed by ``committed_mask``.
    committed_grid_kwh, committed_mask
        Already committed intervals in today's remaining plan; the next-day
        lookahead is not committed. True mask entries fix ordinary quantities.
    soc_end_fixed_kwh
        Only problem 1 (``E_144 = E_0 = 6000``).
    absorption_upper_kwh
        Robust cap on the normal purchase quantity. Delivered power cannot be
        rejected, so a commitment has to stay absorbable.

    Notes
    -----
    结构相同的窗口会复用缓存中的模型对象：只把参数取值就地改写，再让绑定的
    appsi 求解器做增量更新。数值结果与逐次重建逐位一致。
    """
    n = forecast.n
    if n <= 0:
        raise ValueError("empty window")
    risk_reserve = forecast.reserve_energy_kwh is not None
    if risk_reserve != (forecast.net_upper_kwh is not None):
        raise ValueError("Reserve energy and upper net load must be supplied together")
    if risk_reserve:
        if not allow_emergency:
            raise ValueError("Risk reserve is only supported for rolling problems")
        for vector, size in ((forecast.reserve_energy_kwh, n+1), (forecast.net_upper_kwh, n)):
            if np.asarray(vector).shape != (size,) or not np.isfinite(vector).all() or np.any(np.asarray(vector) < 0):
                raise ValueError("Invalid risk reserve quantities")
    T = list(range(n))
    for name, arr in (("demand", forecast.demand_kwh), ("pv", forecast.pv_kwh), ("price", forecast.price_yuan_per_kwh)):
        if not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(soc_start_kwh) or not E_MIN <= soc_start_kwh <= E_MAX:
        raise ValueError("Invalid initial SOC")
    if adjustable_mask is not None and np.asarray(adjustable_mask).shape != (n,):
        raise ValueError("Invalid adjustable mask")
    if (committed_mask is None) != (committed_grid_kwh is None):
        raise ValueError("Committed quantities and mask must be supplied together")
    if (o_kwh is None) != (a_kwh is None):
        raise ValueError("O/A labels must be supplied together")
    for name, vector in (("O", o_kwh), ("A", a_kwh), ("committed", committed_grid_kwh),
                         ("absorption cap", absorption_upper_kwh), ("commitment floor", commitment_lower_kwh)):
        if vector is not None:
            v = np.asarray(vector)
            if v.shape != (n,) or not np.isfinite(v).all() or np.any(v < -1e-6):
                raise ValueError(f"Invalid {name} quantities")
    if price_now_yuan_per_kwh is not None and (not np.isfinite(price_now_yuan_per_kwh) or price_now_yuan_per_kwh < 0):
        raise ValueError("Invalid adjustment planning price")

    # ---- 输入校验（与结构复用前完全一致） ----------------------------------
    cap_arr: np.ndarray | None = None
    if absorption_upper_kwh is not None and committed_mask is None:
        cap_arr = np.asarray(absorption_upper_kwh, dtype=np.float64)
        if cap_arr.size != n:
            raise ValueError(f"absorption_upper_kwh shape should be ({n},), got {cap_arr.shape}")

    floor_arr: np.ndarray | None = None
    if commitment_lower_kwh is not None and committed_mask is None:
        floor_arr = np.asarray(commitment_lower_kwh, dtype=np.float64)
        if floor_arr.size != n:
            raise ValueError(f"commitment_lower_kwh shape should be ({n},), got {floor_arr.shape}")

    o_eff = np.zeros(n, dtype=np.float64)
    a_eff = np.zeros(n, dtype=np.float64)
    if o_kwh is not None and a_kwh is not None:
        o_arr = np.asarray(o_kwh, dtype=np.float64)
        a_arr = np.asarray(a_kwh, dtype=np.float64)
        if o_arr.size != n or a_arr.size != n:
            raise ValueError(f"O/A shape should be ({n},), got {o_arr.shape}/{a_arr.shape}")
        o_eff, a_eff = o_arr, a_arr

    mask_arr: np.ndarray | None = None
    fixed_arr: np.ndarray | None = None
    if committed_mask is not None and committed_grid_kwh is not None:
        mask_arr = np.asarray(committed_mask, dtype=bool)
        fixed_arr = np.asarray(committed_grid_kwh, dtype=np.float64)
        if mask_arr.size != n or fixed_arr.size != n:
            raise ValueError(f"committed carry-over shape should be ({n},)")

    if fee_mode == FeeMode.ADJUSTABLE and price_now_yuan_per_kwh is None:
        raise ValueError("ADJUSTABLE mode needs the current adjustment price p_a")

    # ---- 取（或建）结构模板 -------------------------------------------------
    shape = _window_shape(
        n=n,
        fee_mode=fee_mode,
        absorption_upper_kwh=absorption_upper_kwh,
        commitment_lower_kwh=commitment_lower_kwh,
        soc_end_fixed_kwh=soc_end_fixed_kwh,
        committed_grid_kwh=committed_grid_kwh,
        committed_mask=committed_mask,
        adjustable_mask=adjustable_mask,
        allow_emergency=allow_emergency,
        allow_spill=allow_spill,
        risk_reserve=risk_reserve,
    )
    template = _CACHE.get(shape)
    if template is None:
        m = _build_window_model(
            shape,
            model_name=problem_name,
            forecast=forecast,
            absorption_upper_kwh=cap_arr,
            commitment_lower_kwh=floor_arr,
            committed_mask=mask_arr,
        )
        template = _WindowTemplate(shape=shape, model=m)
        _CACHE[shape] = template
        _CACHE_STATS["miss"] += 1
        _evict_oldest()
    else:
        _CACHE.move_to_end(shape)
        _CACHE_STATS["hit"] += 1
        m = template.model

    _load_window_inputs(
        m,
        shape,
        forecast=forecast,
        soc_start_kwh=soc_start_kwh,
        o_eff=o_eff,
        a_eff=a_eff,
        absorption_upper_kwh=cap_arr,
        commitment_lower_kwh=floor_arr,
        committed_grid_kwh=fixed_arr,
        soc_end_fixed_kwh=soc_end_fixed_kwh,
        price_now_yuan_per_kwh=price_now_yuan_per_kwh,
    )

    # 求解器与模型成对复用：appsi 只有拿到同一个实例才会走 update() 增量路径；
    # 非 appsi 后端不缓存实例，行为与改动前一致。
    if template.solver_name is not None and template.solver_name != solver_name:
        template.solver = None
        template.solver_name = None
    if template.solver is None and solver_name.startswith("appsi"):
        template.solver = make_appsi_solver(solver_name)
        template.solver_name = solver_name

    def stage_solve():
        report = solve_model(m, problem_name, solver_name=solver_name, solver=template.solver)
        if not report.feasible:
            active = next(m.component_data_objects(pyo.Objective, active=True)).name
            report.message += f" Lexicographic stage: {active}"
        return report

    if allow_spill:
        for component in (m.obj, m.keep_spill, m.keep_cost, m.prefer_storage):
            component.deactivate()
        m.min_spill.activate()
        report = stage_solve()
        if report.feasible:
            optimum_spill = max(0., float(pyo.value(m.min_spill.expr)))
            m.spill_limit.set_value(optimum_spill + max(1e-6, 1e-9*optimum_spill))
            m.min_spill.deactivate()
            m.keep_spill.activate()
            m.obj.activate()
            report = stage_solve()
        if report.feasible:
            optimum_cost = float(pyo.value(m.obj.expr))
            m.cost_limit.set_value(optimum_cost + max(1e-6, 1e-9*abs(optimum_cost)))
            m.obj.deactivate()
            m.keep_cost.activate()
            m.prefer_storage.activate()
            report = stage_solve()
        if report.feasible:
            report.objective_yuan = float(pyo.value(m.fee_objective))
    else:
        report = stage_solve()
        if report.feasible:
            report.objective_yuan = float(pyo.value(m.fee_objective))
    if not report.feasible:
        frozen = np.zeros(n, dtype=bool)
        quantities = np.zeros(n)
        if fee_mode == FeeMode.FROZEN:
            frozen[:] = True if mask_arr is None else mask_arr
            quantities = o_eff + a_eff if fixed_arr is None else fixed_arr
        elif mask_arr is not None:
            frozen, quantities = mask_arr, fixed_arr
        excess = np.where(frozen, quantities - forecast.demand_kwh - Q_MAX * CHARGE_BUS_FACTOR, -np.inf)
        if np.max(excess) > 1e-6:
            i = int(np.argmax(excess))
            report.message += (f" Frozen purchase at abs={int(forecast.abs_minutes[i])} exceeds forecast demand"
                               f" plus maximum bus charging input by {excess[i]:.6f} kWh; even full PV curtailment cannot absorb it.")
        return WindowResult(
            report=report,
            abs_minutes=forecast.abs_minutes.copy(),
            grid_kwh=np.full(n, np.nan),
            emergency_kwh=np.full(n, np.nan),
            charge_kwh=np.full(n, np.nan),
            discharge_kwh=np.full(n, np.nan),
            curtail_kwh=np.full(n, np.nan),
            soc_boundary_kwh=np.full(n + 1, np.nan),
            objective_yuan=float("nan"),
            spill_kwh=np.full(n, np.nan),
        )

    get = lambda var, i: float(pyo.value(var[i]))  # noqa: E731

    answer = WindowResult(
        report=report,
        abs_minutes=forecast.abs_minutes.copy(),
        grid_kwh=np.array([get(m.grid, t) for t in T]),
        emergency_kwh=np.array([get(m.emergency, t) for t in T]),
        charge_kwh=np.array([get(m.charge, t) for t in T]),
        discharge_kwh=np.array([get(m.discharge, t) for t in T]),
        curtail_kwh=np.array([get(m.curtail, t) for t in T]),
        soc_boundary_kwh=np.array([get(m.soc, s) for s in m.S]),
        objective_yuan=float(report.objective_yuan),
        spill_kwh=np.array([get(m.spill, t) for t in T]),
        reserve_shortfall_kwh=max(float(pyo.value(m.reserve_shortfall)),
            max(get(m.power_shortfall, t) for t in T)) if risk_reserve else 0.,
        risk_penalty_yuan=float(pyo.value(m.risk_cost)),
        reserve_stock_shortfall_kwh=float(pyo.value(m.reserve_shortfall)) if risk_reserve else 0.,
        reserve_power_shortfall_kwh=np.array([get(m.power_shortfall, t) for t in T]) if risk_reserve else np.zeros(n),
    )


    from .validation import validate_absolute_run
    diag = validate_absolute_run(
        abs_minutes=answer.abs_minutes, grid_kwh=answer.grid_kwh,
        emergency_kwh=answer.emergency_kwh, charge_kwh=answer.charge_kwh,
        discharge_kwh=answer.discharge_kwh, curtail_kwh=answer.curtail_kwh,
        surplus_kwh=answer.spill_kwh, soc_boundary_kwh=answer.soc_boundary_kwh,
        demand_kwh=forecast.demand_kwh, pv_kwh=forecast.pv_kwh,
        soc_start_kwh=soc_start_kwh, allow_spill=allow_spill, validate_feedback=False)
    # Recompute the exact labels, including zero-price epigraph degeneracy.
    price = forecast.price_yuan_per_kwh
    exact = price * answer.grid_kwh
    if fee_mode == FeeMode.FROZEN:
        frozen = np.ones(n, dtype=bool) if mask_arr is None else mask_arr
        exact[frozen] = price[frozen] * (o_eff[frozen] + FEE_ADJUST_UP * a_eff[frozen])
    elif fee_mode == FeeMode.ADJUSTABLE:
        adj = np.ones(n, dtype=bool) if adjustable_mask is None else np.asarray(adjustable_mask, dtype=bool)
        exact[adj] += ((FEE_ADJUST_UP - 1) * price[adj] * np.maximum(answer.grid_kwh[adj] - o_eff[adj], 0)
                       + FEE_PENALTY_DOWN * price_now_yuan_per_kwh * np.maximum(o_eff[adj] + a_eff[adj] - answer.grid_kwh[adj], 0))
    exact_total = float(exact.sum() + FEE_EMERGENCY * np.dot(price, answer.emergency_kwh))
    if not np.isclose(exact_total, answer.objective_yuan, atol=1e-5, rtol=1e-9):
        raise RuntimeError("Window fee recomputation failed")
    report.max_residual = max(diag.max_bus_residual_kwh, diag.max_soc_error_kwh)
    return answer


__all__ = [
    "WindowForecast",
    "WindowResult",
    "FeeMode",
    "solve_window",
    "MAX_CACHED_WINDOWS",
    "window_cache_info",
    "clear_window_cache",
]
