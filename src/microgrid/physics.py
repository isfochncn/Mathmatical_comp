"""统一物理核：储能状态转移、母线守恒与损耗代数。

四问共用本模块，任何一问都不得改写这里的单位与效率定义。

变量定义（备忘录第 3 节，已冻结）
--------------------------------
``q_ch``  效率损耗后**实际存入电池**的电量（kWh）。母线必须付出 ``q_ch / 0.9``。
``q_dis`` 效率损耗后**实际送达微网**的电量（kWh）。电池必须消耗 ``q_dis / 0.9``。

于是（η_c = η_d = 0.9，往返 0.81）：

* 状态转移   ``E_{t+1} = E_t + q_ch[t] - q_dis[t] / 0.9``
* 母线守恒   ``h[t] + u[t] + G[t] - r[t] + q_dis[t] = D[t] + q_ch[t] / 0.9``
* 每段上限   ``0 <= q_ch[t], q_dis[t] <= 4500/6 = 750``
  （q_ch = 750 -> 电池存入 750、母线侧 5000 kW；
    q_dis = 750 -> 电池放出 833.333、母线侧 4500 kW。
    两向电池内部交换功率都不超过 5000 kW 额定值。
    绝不可把 750 再乘 0.9 得到 675。）
* 电量边界   ``1200 <= E_s <= 10800``（额定 12000 不是运行上限）

以上母线式及下方损耗恒等式为 w=0 的基础形式，仍用于问题一。
滚动模型按备忘录第4.0节允许已付费弃购电 w，母线右侧和全天购电总账
右侧分别增加 w 和 Σw；planning/simulation/validation 负责该扩展。

命名提醒
--------
``q_ch`` 是电池实际存入量；母线侧输入为 ``q_ch / 0.9``。

损耗代数恒等式（全天收尾恒等式与"损耗循环"诊断共用）
-----------------------------------------------------
由状态转移得 ``q_dis[t] = η (q_ch[t] - δE_t)``，代入母线守恒并整理：

    h + u - (D - G + r) = q_ch * K + η * δE,     K = 1/η - η = 19/90

全天求和（Σ δE = E_144 - E_0）即备忘录第 3 节给出、并已用随机轨迹数值验证
（残差 ~1e-12）的形式

    Σ(h + u) = Σ(D - G + r) + (19/90)·Σq_ch + 0.9·(E_144 - E_0)

若某段同时充放（δE = 0），则 ``q_dis = 0.9 q_ch``，母线净付出恰为 ``K·q_ch``：
**买进来却没进电池的那部分，恰好等于电池里少掉的那部分**。
它只增加买电量与损耗，对目标无益；但若最优解依赖它，仍属可实施性缺口，
必须报告，不得事后暗加互斥约束。
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CHARGE_BUS_FACTOR,
    DELTA_T_HOURS,
    DISCHARGE_BATTERY_FACTOR,
    E_MAX,
    E_MIN,
    ETA_CHARGE,
    ETA_DISCHARGE,
    LOSS_COEFF,
    N_BOUNDARY,
    N_INTERVAL,
    Q_MAX,
    TOL_ENERGY_KWH,
    TOL_RELATIVE,
)

# ==========================================================================
# 单步推进
# ==========================================================================


def soc_next(soc_kwh: float, charge_stored_kwh: float, discharge_delivered_kwh: float) -> float:
    """由本段实际充放量推进到下一段边界的储电量。"""
    return soc_kwh + charge_stored_kwh - discharge_delivered_kwh / ETA_DISCHARGE


def soc_next_vec(
    soc_kwh: np.ndarray, charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray
) -> np.ndarray:
    """向量版状态转移。输入 (144,)+(145,) -> 输出 (145,)，E[0] = soc_kwh[0]。"""
    q_ch = np.asarray(charge_stored_kwh, dtype=np.float64)
    q_dis = np.asarray(discharge_delivered_kwh, dtype=np.float64)
    return soc_kwh[0] + np.concatenate(
        ([0.0], np.cumsum(q_ch - q_dis / ETA_DISCHARGE))
    )


def energy_change_from_dispatch(
    charge_stored_kwh: float, discharge_delivered_kwh: float
) -> float:
    """本段储电量净变化 ΔE = q_ch - q_dis / 0.9。"""
    return charge_stored_kwh - discharge_delivered_kwh / ETA_DISCHARGE


def charge_for_energy_change(delta_e_kwh: float, discharge_delivered_kwh: float = 0.0) -> float:
    """反解：要在扣除放电影响后让 E 变化 ΔE，需要实际存入多少。

    ΔE = q_ch - q_dis / 0.9  =>  q_ch = ΔE + q_dis / 0.9
    """
    return delta_e_kwh + discharge_delivered_kwh / ETA_DISCHARGE


# ==========================================================================
# 损耗与母线平衡
# ==========================================================================


def loss_cycle_coefficient() -> float:
    """K = 1/η_c - η_d（η 均为 0.9 时为 19/90）。"""
    return LOSS_COEFF


def net_load_kwh(demand_kwh: np.ndarray, pv_kwh: np.ndarray, curtail_kwh: np.ndarray) -> np.ndarray:
    """母线需要净供给的电量 D - G + r（弃光只作用于光伏，不能用 r 丢已购电）。"""
    return np.asarray(demand_kwh, dtype=np.float64) - np.asarray(
        pv_kwh, dtype=np.float64
    ) + np.asarray(curtail_kwh, dtype=np.float64)


def grid_required_kwh(
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    curtail_kwh: np.ndarray,
    charge_stored_kwh: np.ndarray,
    discharge_delivered_kwh: np.ndarray,
) -> np.ndarray:
    """母线守恒左端需要的外购总量 h + u（可为负，表示计划过量）。"""
    return (
        net_load_kwh(demand_kwh, pv_kwh, curtail_kwh)
        + np.asarray(charge_stored_kwh, dtype=np.float64) * CHARGE_BUS_FACTOR
        - np.asarray(discharge_delivered_kwh, dtype=np.float64)
    )


def loss_cycle_kwh(charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray) -> np.ndarray:
    """本段的损耗循环量 K * q_ch（同时充放时为正）。"""
    return LOSS_COEFF * np.asarray(charge_stored_kwh, dtype=np.float64)


def bus_power_kw(quantity_kwh: float) -> float:
    """区间电量 -> 母线侧平均功率。"""
    return quantity_kwh / DELTA_T_HOURS


def charge_bus_input_kwh(charge_stored_kwh: float) -> float:
    """充电时母线必须付出的电量（含充电损耗）。"""
    return charge_stored_kwh * CHARGE_BUS_FACTOR


def discharge_battery_draw_kwh(discharge_delivered_kwh: float) -> float:
    """放电时电池内部必须消耗的电量（含放电损耗）。"""
    return discharge_delivered_kwh * DISCHARGE_BATTERY_FACTOR


# ==========================================================================
# 全天恒等式（规范第 16 节强制校验）
# ==========================================================================


def daily_balance_residual_kwh(
    grid_actual_kwh: np.ndarray,
    emergency_actual_kwh: np.ndarray,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    curtail_kwh: np.ndarray,
    charge_stored_kwh: np.ndarray,
    discharge_delivered_kwh: np.ndarray,
    soc_kwh: np.ndarray,
) -> float:
    """全天恒等式残差。

    sum(h + u) = sum(D - G + r) + (19/90) * sum(q_ch) + 0.9 * (E_end - E_start)
    """
    lhs = float(np.sum(grid_actual_kwh) + np.sum(emergency_actual_kwh))
    rhs = (
        float(np.sum(net_load_kwh(demand_kwh, pv_kwh, curtail_kwh)))
        + LOSS_COEFF * float(np.sum(charge_stored_kwh))
        + ETA_DISCHARGE * (float(soc_kwh[-1]) - float(soc_kwh[0]))
    )
    return lhs - rhs


def discharge_charge_relation_residual_kwh(
    charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray
) -> float:
    """日循环（E_144 = E_0）条件下应满足 sum(q_dis) = η_d · Σq_ch = 0.9 Σq_ch。

    推导：Σ ΔE = 0，而 ΔE = q_ch - q_dis/η_d  =>  Σq_dis = η_d · Σq_ch。
    注意这里**不是** 0.81——0.81 才是一次完整充放循环（充进去再放出来）的往返效率。
    """
    return (
        float(np.sum(discharge_delivered_kwh))
        - ETA_DISCHARGE * float(np.sum(charge_stored_kwh))
    )


def loss_cycle_relation_residual_kwh(
    charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray
) -> float:
    """ΔE = 0 的损耗循环特例与日循环满足同一个关系式 sum(q_dis) = η_d·Σq_ch。

    保留本函数是为了让"损耗循环"这一异常在代码里有独立可检索的名字。
    """
    return discharge_charge_relation_residual_kwh(
        charge_stored_kwh, discharge_delivered_kwh
    )


def round_trip_ratio(charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray) -> float:
    """实际往返比 = sum(q_dis) / sum(q_ch)。

    * 一次完整充放循环（ΔE = 0 时 q_dis = 0.9 q_ch）下为 0.9；
    * 正常日循环（E_144 = E_0，q_dis = 0.81 q_ch）下为 0.81 = η_c·η_d。

    两件事不矛盾：0.9 来自状态转移里的 1/η_d，0.81 才是"充进去再放出来"的真实往返效率。
    """
    total_ch = float(np.sum(charge_stored_kwh))
    if total_ch <= 0.0:
        return float("nan")
    return float(np.sum(discharge_delivered_kwh)) / total_ch


# ==========================================================================
# 可行性检查
# ==========================================================================


class PhysicsViolation(AssertionError):
    """物理约束被违反——正式导出必须失败，而不是仅打印警告。"""


def check_soc_bounds(soc_kwh: np.ndarray, tol: float = TOL_ENERGY_KWH) -> None:
    arr = np.asarray(soc_kwh, dtype=np.float64)
    if arr.shape != (N_BOUNDARY,):
        raise PhysicsViolation(f"SOC 数组应为 ({N_BOUNDARY},)，得到 {arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if lo < E_MIN - tol:
        raise PhysicsViolation(f"储电量低于下限：最小 {lo:.6f} < {E_MIN}")
    if hi > E_MAX + tol:
        raise PhysicsViolation(f"储电量高于上限：最大 {hi:.6f} > {E_MAX}")


def check_charge_discharge_bounds(
    charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray, tol: float = TOL_ENERGY_KWH
) -> None:
    for name, arr in (("q_ch", charge_stored_kwh), ("q_dis", discharge_delivered_kwh)):
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (N_INTERVAL,):
            raise PhysicsViolation(f"{name} 应为 ({N_INTERVAL},)，得到 {a.shape}")
        if float(a.min()) < -tol:
            raise PhysicsViolation(f"{name} 出现负值：{float(a.min()):.6f}")
        if float(a.max()) > Q_MAX + tol:
            raise PhysicsViolation(
                f"{name} 超过有效电量上限：最大 {float(a.max()):.6f} > {Q_MAX}（不得再乘 0.9）"
            )


def check_soc_transition(
    soc_kwh: np.ndarray,
    charge_stored_kwh: np.ndarray,
    discharge_delivered_kwh: np.ndarray,
    tol: float = TOL_ENERGY_KWH,
) -> None:
    rebuilt = soc_next_vec(soc_kwh, charge_stored_kwh, discharge_delivered_kwh)
    err = float(np.max(np.abs(rebuilt - np.asarray(soc_kwh, dtype=np.float64))))
    if err > tol:
        raise PhysicsViolation(f"SOC 递推不自洽：最大偏差 {err:.3e} > {tol}")


def check_bus_balance(
    grid_actual_kwh: np.ndarray,
    emergency_actual_kwh: np.ndarray,
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    curtail_kwh: np.ndarray,
    charge_stored_kwh: np.ndarray,
    discharge_delivered_kwh: np.ndarray,
    tol: float = TOL_ENERGY_KWH,
) -> float:
    """逐段检查母线守恒，返回最大残差。"""
    required = grid_required_kwh(
        demand_kwh, pv_kwh, curtail_kwh, charge_stored_kwh, discharge_delivered_kwh
    )
    supplied = np.asarray(grid_actual_kwh, dtype=np.float64) + np.asarray(
        emergency_actual_kwh, dtype=np.float64
    )
    resid = np.abs(supplied - required)
    worst = float(resid.max())
    if worst > tol:
        t = int(resid.argmax())
        raise PhysicsViolation(
            f"母线守恒残差过大：区间 {t} 需要 {required[t]:.6f}，实供 {supplied[t]:.6f}，"
            f"差 {resid[t]:.6f} > {tol}"
        )
    return worst


def check_curtail_bounds(curtail_kwh: np.ndarray, pv_kwh: np.ndarray, tol: float = TOL_ENERGY_KWH) -> None:
    r = np.asarray(curtail_kwh, dtype=np.float64)
    g = np.asarray(pv_kwh, dtype=np.float64)
    if float(r.min()) < -tol:
        raise PhysicsViolation(f"弃光出现负值：{float(r.min()):.6f}")
    if float((r - g).max()) > tol:
        t = int(np.argmax(r - g))
        raise PhysicsViolation(f"弃光超过光伏：区间 {t} 弃 {r[t]:.6f} > 光伏 {g[t]:.6f}")


def simultaneous_charge_discharge_mask(
    charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray, tol: float = 1e-6
) -> np.ndarray:
    """Mask of intervals where both directions are active.

    2026-09-11 定稿口径：储能设备被视为**支持同时充放电**的系统，
    因此这**不是异常**，不能作为不可行判据。本掩码只用于统计与损耗核算。
    """
    a = np.asarray(charge_stored_kwh, dtype=np.float64)
    b = np.asarray(discharge_delivered_kwh, dtype=np.float64)
    return (a > tol) & (b > tol)


def loss_accounting(
    charge_stored_kwh: np.ndarray,
    discharge_delivered_kwh: np.ndarray,
    tol: float = 1e-6,
) -> dict[str, float | int | list[int]]:
    """能量损耗核算（同时充放电是合法运行状态，此处只如实记账）。

    充电损耗 = Σ q_ch/0.9 − Σ q_ch（母线付出多于实际存入的部分）
    放电损耗 = Σ q_dis/0.9 − Σ q_dis（电池消耗多于实际送达的部分）
    总损耗   = (1/0.9 - 1)·Σ(q_ch + q_dis)

    规范原话：同段存入 90、送达 81 使 SOC 净变化为 0，母线进 100 出 81，
    **19 kWh 是正常效率损耗**，必须如实列出，但不能称为"净储存 19 kWh"。
    """
    q_ch = np.asarray(charge_stored_kwh, dtype=np.float64)
    q_dis = np.asarray(discharge_delivered_kwh, dtype=np.float64)
    mask = simultaneous_charge_discharge_mask(q_ch, q_dis, tol)
    charge_loss = float(np.sum(charge_bus_input_kwh(q_ch)) - np.sum(q_ch))
    discharge_loss = float(np.sum(discharge_battery_draw_kwh(q_dis)) - np.sum(q_dis))
    return {
        "n_simultaneous_intervals": int(mask.sum()),
        "simultaneous_intervals": [int(i) for i in np.flatnonzero(mask)],
        "charge_loss_kwh": charge_loss,
        "discharge_loss_kwh": discharge_loss,
        "total_loss_kwh": charge_loss + discharge_loss,
        "charge_stored_total_kwh": float(np.sum(q_ch)),
        "discharge_delivered_total_kwh": float(np.sum(q_dis)),
    }


def max_bus_power_kw(
    charge_stored_kwh: np.ndarray, discharge_delivered_kwh: np.ndarray
) -> dict[str, float]:
    """本轨迹上各侧的最大瞬时功率（kW），用于核对 5000 kW 额定。"""
    q_ch = np.asarray(charge_stored_kwh, dtype=np.float64)
    q_dis = np.asarray(discharge_delivered_kwh, dtype=np.float64)
    return {
        "bus_charge_kw": float((q_ch * CHARGE_BUS_FACTOR).max() / DELTA_T_HOURS) if q_ch.size else 0.0,
        "bus_discharge_kw": float(q_dis.max() / DELTA_T_HOURS) if q_dis.size else 0.0,
        "battery_charge_kw": float(q_ch.max() / DELTA_T_HOURS) if q_ch.size else 0.0,
        "battery_discharge_kw": float((q_dis * DISCHARGE_BATTERY_FACTOR).max() / DELTA_T_HOURS)
        if q_dis.size
        else 0.0,
    }


__all__ = [
    "PhysicsViolation",
    "soc_next",
    "soc_next_vec",
    "energy_change_from_dispatch",
    "charge_for_energy_change",
    "loss_cycle_coefficient",
    "net_load_kwh",
    "grid_required_kwh",
    "loss_cycle_kwh",
    "bus_power_kw",
    "charge_bus_input_kwh",
    "discharge_battery_draw_kwh",
    "daily_balance_residual_kwh",
    "discharge_charge_relation_residual_kwh",
    "loss_cycle_relation_residual_kwh",
    "round_trip_ratio",
    "check_soc_bounds",
    "check_charge_discharge_bounds",
    "check_soc_transition",
    "check_bus_balance",
    "check_curtail_bounds",
    "simultaneous_charge_discharge_mask",
    "loss_accounting",
    "max_bus_power_kw",
    "ETA_CHARGE",
    "ETA_DISCHARGE",
    "Q_MAX",
    "E_MIN",
    "E_MAX",
    "TOL_RELATIVE",
]
