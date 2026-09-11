"""物理核测试：规范第 16 节的手算用例 + 代数恒等式。"""

from __future__ import annotations

import numpy as np
import pytest

from microgrid import physics
from microgrid.constants import ETA_CHARGE, ETA_DISCHARGE, LOSS_COEFF, Q_MAX


# ------------------------------------------------------------ 有效量定义


def test_full_power_charge_stores_750() -> None:
    """满功率 10 分钟充电：母线侧充电量 750 kWh，母线付出 750/0.9，电池实际存入 675。"""
    bus_charge = Q_MAX
    assert bus_charge == pytest.approx(750.0)
    assert physics.charge_bus_input_kwh(bus_charge) == pytest.approx(750.0 / 0.9)
    # 电池侧功率 4050 kW，未超 5000 kW 额定
    assert physics.bus_power_kw(bus_charge * ETA_CHARGE) == pytest.approx(4050.0)
    # 电池真正存进去的只有 90%
    assert bus_charge * ETA_CHARGE == pytest.approx(675.0)


def test_full_power_discharge_delivers_750() -> None:
    """满功率 10 分钟放电：实际送达 750 kWh，电池消耗 750/0.9。"""
    delivered = Q_MAX
    assert physics.discharge_battery_draw_kwh(delivered) == pytest.approx(750.0 / 0.9)
    assert physics.bus_power_kw(delivered) == pytest.approx(4500.0)
    # 电池侧功率 5000 kW，正好是额定值
    assert physics.bus_power_kw(physics.discharge_battery_draw_kwh(delivered)) == pytest.approx(5000.0)


def test_effective_limit_is_not_multiplied_again() -> None:
    """不得把 750 再乘 0.9 作为实际充电上限。"""
    assert Q_MAX == pytest.approx(750.0)
    assert Q_MAX != pytest.approx(675.0)
    assert Q_MAX * ETA_CHARGE == pytest.approx(675.0)  # 这是"电池存入"，不是上限


def test_battery_side_exchange_respects_rated_power() -> None:
    """两向的电池内部交换功率都不得超过 5000 kW 额定值。"""
    charge_side_kw = physics.bus_power_kw(Q_MAX * ETA_CHARGE)
    discharge_side_kw = physics.bus_power_kw(physics.discharge_battery_draw_kwh(Q_MAX))
    assert charge_side_kw <= 5000.0 + 1e-9
    assert discharge_side_kw <= 5000.0 + 1e-9


def test_microgrid_100_stores_90_then_81_delivered() -> None:
    """微网输入 100 -> 电池实际存入 90；电池消耗 90 -> 送达微网 81。"""
    bus_in = 100.0
    stored = bus_in * ETA_CHARGE
    assert stored == pytest.approx(90.0)
    assert physics.charge_bus_input_kwh(stored) == pytest.approx(100.0)
    battery_draw = 90.0
    delivered = battery_draw * ETA_DISCHARGE
    assert delivered == pytest.approx(81.0)
    assert physics.discharge_battery_draw_kwh(delivered) == pytest.approx(90.0)


def test_loss_cycle_keeps_soc_but_costs_energy() -> None:
    """90 kWh 实际存入 + 81 kWh 实际放出可保持 SOC 不变，但母线净损失 19 kWh。

    由 ΔE = 0 即 q_ch - q_dis/0.9 = 0（q_dis = 0.9 q_ch = 81）唯一确定。
    注意此处的 0.9 是 η_d，与往返效率 0.81 不是同一件事。
    """
    q_ch, q_dis = 90.0, 81.0
    delta_e = physics.energy_change_from_dispatch(q_ch, q_dis)
    assert delta_e == pytest.approx(0.0, abs=1e-12)
    # 母线净付出 = 充电输入 - 放电送达 = 100 - 81 = 19
    net_bus = physics.charge_bus_input_kwh(q_ch) - q_dis
    assert net_bus == pytest.approx(19.0)
    # 恰等于 K * q_ch，K = 19/90
    assert net_bus == pytest.approx(LOSS_COEFF * q_ch)
    assert LOSS_COEFF == pytest.approx(19.0 / 90.0)


def test_loss_coefficient_closed_form() -> None:
    """K = 1/η_c - η_d = 19/90；不是 1/η_c + 1/η_d - 1 那个常见错误写法。"""
    assert LOSS_COEFF == pytest.approx(1.0 / ETA_CHARGE - ETA_DISCHARGE)
    assert LOSS_COEFF == pytest.approx(19.0 / 90.0)
    assert LOSS_COEFF != pytest.approx(1.0 / ETA_CHARGE + 1.0 / ETA_DISCHARGE - 1.0)


def test_round_trip_efficiency_is_081() -> None:
    """往返效率 = η_c · η_d = 0.81：母线充 100，最终只回 81。

    走一遍有效量变量的换算链：
      母线充 100 --×η_c--> q_ch = 90（实际存入）
      q_dis = 90（实际送达）-- 电池消耗 q_dis/η_d = 100
      电池里净减少 100 - 90 = 10，即放电效率损失 10
    故 q_ch -> q_dis 这一段效率是 0.9，再乘充电段 0.9 得到整体 0.81。
    """
    assert ETA_CHARGE * ETA_DISCHARGE == pytest.approx(0.81)
    bus_in = 100.0
    q_ch = bus_in * ETA_CHARGE  # 90 实际存入
    q_dis = q_ch  # 把刚存进去的全放出来 = 实际送达 90
    battery_draw = physics.discharge_battery_draw_kwh(q_dis)  # 90 / 0.9 = 100
    assert q_ch == pytest.approx(90.0)
    assert battery_draw == pytest.approx(100.0)
    assert q_dis / bus_in == pytest.approx(0.9)  # 单次循环口径
    # 而"从母线进、从母线出"的完整往返：母线得 81 才对应 0.81
    assert bus_in * ETA_CHARGE * ETA_CHARGE == pytest.approx(81.0)
    # 日循环轨迹（E_144 = E_0）的 q_dis/q_ch 是 0.9，与 0.81 不矛盾
    q_ch_arr = np.zeros(144)
    q_dis_arr = np.zeros(144)
    q_ch_arr[0] = 500.0
    q_dis_arr[5] = 0.9 * 500.0
    assert physics.round_trip_ratio(q_ch_arr, q_dis_arr) == pytest.approx(0.9)


def test_round_trip_ratio_distinguishes_two_relations() -> None:
    """0.81（母线进母线出的完整往返）与 0.9（日循环的 q_dis/q_ch）是两件不同的事。"""
    # 完整往返：母线充 100 -> 存 90 -> 取 90 -> 母线得 81
    bus_in = 100.0
    stored = bus_in * ETA_CHARGE
    assert stored == pytest.approx(90.0)
    assert stored * ETA_CHARGE == pytest.approx(81.0)
    assert stored * ETA_CHARGE / bus_in == pytest.approx(0.81)
    # 日循环（E 不变）里 q_dis = 0.9 q_ch
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[0] = 100.0
    q_dis[1] = 90.0
    assert physics.energy_change_from_dispatch(100.0, 90.0) == pytest.approx(0.0)
    assert physics.round_trip_ratio(q_ch, q_dis) == pytest.approx(0.9)


# ------------------------------------------------------------ 状态推进


def test_soc_next_scalar() -> None:
    assert physics.soc_next(6000.0, 90.0, 0.0) == pytest.approx(6090.0)
    assert physics.soc_next(6000.0, 0.0, 90.0) == pytest.approx(6000.0 - 100.0)


def test_soc_next_vec_matches_scalar_loop() -> None:
    rng = np.random.default_rng(7)
    q_ch = rng.uniform(0, Q_MAX, size=144)
    q_dis = rng.uniform(0, Q_MAX, size=144)
    arr = physics.soc_next_vec(np.zeros(145) + 6000.0, q_ch, q_dis)
    e = 6000.0
    manual = [e]
    for t in range(144):
        e = physics.soc_next(e, q_ch[t], q_dis[t])
        manual.append(e)
    assert np.allclose(arr, np.array(manual))


def test_charge_for_energy_change_inverts_transition() -> None:
    e0, q_dis, target_delta = 6000.0, 300.0, 150.0
    q_ch = physics.charge_for_energy_change(target_delta, q_dis)
    assert physics.energy_change_from_dispatch(q_ch, q_dis) == pytest.approx(target_delta)


# ------------------------------------------------------------ 校验函数


def test_soc_bounds_rejected() -> None:
    soc = np.full(145, 6000.0)
    physics.check_soc_bounds(soc)
    soc[7] = 1199.0
    with pytest.raises(physics.PhysicsViolation):
        physics.check_soc_bounds(soc)
    soc[7] = 10801.0
    with pytest.raises(physics.PhysicsViolation):
        physics.check_soc_bounds(soc)


def test_charge_discharge_bounds_rejected() -> None:
    ok = np.full(144, 700.0)
    physics.check_charge_discharge_bounds(ok, ok)
    bad = ok.copy()
    bad[0] = 750.0001
    with pytest.raises(physics.PhysicsViolation):
        physics.check_charge_discharge_bounds(bad, ok)
    with pytest.raises(physics.PhysicsViolation):
        physics.check_charge_discharge_bounds(ok, -1.0 * ok)


def test_bus_balance_detects_error() -> None:
    demand = np.full(144, 100.0)
    pv = np.zeros(144)
    curtail = np.zeros(144)
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    grid = np.full(144, 100.0)
    assert physics.check_bus_balance(grid, np.zeros(144), demand, pv, curtail, q_ch, q_dis) < 1e-9
    grid_bad = grid.copy()
    grid_bad[3] += 1.0
    with pytest.raises(physics.PhysicsViolation):
        physics.check_bus_balance(grid_bad, np.zeros(144), demand, pv, curtail, q_ch, q_dis)


def test_curtail_cannot_exceed_pv() -> None:
    pv = np.zeros(144)
    pv[10] = 50.0
    physics.check_curtail_bounds(np.zeros(144), pv)
    r = np.zeros(144)
    r[10] = 50.0
    physics.check_curtail_bounds(r, pv)
    r[10] = 50.001
    with pytest.raises(physics.PhysicsViolation):
        physics.check_curtail_bounds(r, pv)


def test_daily_balance_identity() -> None:
    """构造一条满足守恒且日循环的轨迹，验证全天恒等式与充放关系残差为 0。"""
    n = 144
    demand = np.full(n, 120.0)
    pv = np.full(n, 20.0)
    curtail = np.zeros(n)
    q_ch = np.zeros(n)
    q_dis = np.zeros(n)
    q_ch[0] = 200.0
    q_dis[10:14] = 45.0  # 分四段放，保证每段外购非负；合计 180 = 0.9 * 200
    needed = physics.grid_required_kwh(demand, pv, curtail, q_ch, q_dis)
    assert (needed >= 0).all(), "本构造下不应出现计划过量"
    grid = needed.copy()
    u = np.zeros(n)
    soc = np.concatenate(([6000.0], 6000.0 + np.cumsum(q_ch - q_dis / ETA_DISCHARGE)))
    assert soc[-1] == pytest.approx(soc[0]), "日循环应使 E_144 = E_0"
    resid = physics.daily_balance_residual_kwh(grid, u, demand, pv, curtail, q_ch, q_dis, soc)
    assert resid == pytest.approx(0.0, abs=1e-9)
    assert physics.discharge_charge_relation_residual_kwh(q_ch, q_dis) == pytest.approx(0.0)


def test_daily_balance_identity_allows_surplus() -> None:
    """母线需求为负（光伏富余）时分段守恒仍成立，只是需要弃光而非购电。"""
    n = 144
    demand = np.full(n, 120.0)
    pv = np.full(n, 20.0)
    curtail = np.zeros(n)
    q_ch = np.zeros(n)
    q_dis = np.zeros(n)
    q_dis[10] = 180.0  # 单段放 180 kWh，超过该段净需求 100 -> 出现富余
    needed = physics.grid_required_kwh(demand, pv, curtail, q_ch, q_dis)
    assert needed[10] < 0.0, "该构造应当出现富余"
    # 现实口径是不得拒收：富余不能靠"少买"消化，故实际外购按 0 计并需另行处理
    grid = np.maximum(needed, 0.0)
    soc = np.concatenate(([6000.0], 6000.0 + np.cumsum(q_ch - q_dis / ETA_DISCHARGE)))
    resid = physics.daily_balance_residual_kwh(
        grid, np.zeros(n), demand, pv, curtail, q_ch, q_dis, soc
    )
    assert resid == pytest.approx(-needed[10])  # 残差恰是那段无法消纳的富余


def test_daily_balance_identity_matches_memo_form() -> None:
    """全天恒等式与备忘录第 3 节的 19/90 写法给出同一个零残差。"""
    n = 144
    demand = np.full(n, 130.0)
    pv = np.full(n, 30.0)
    curtail = np.zeros(n)
    q_ch = np.zeros(n)
    q_dis = np.zeros(n)
    q_ch[0] = 300.0
    q_ch[20] = 150.0
    q_dis[40] = 400.0
    # 该构造下区间 40 的放电超过净需求，故意保留以检验恒等式的普适性：
    # 恒等式是代数结果，不依赖购电量非负。
    needed = physics.grid_required_kwh(demand, pv, curtail, q_ch, q_dis)
    soc = np.concatenate(([6000.0], 6000.0 + np.cumsum(q_ch - q_dis / ETA_DISCHARGE)))
    ours = physics.daily_balance_residual_kwh(
        needed, np.zeros(n), demand, pv, curtail, q_ch, q_dis, soc
    )
    memo = (
        float(np.sum(needed))
        - float(np.sum(demand - pv + curtail))
        - (19.0 / 90.0) * float(np.sum(q_ch))
        - ETA_DISCHARGE * (float(soc[-1]) - float(soc[0]))
    )
    assert ours == pytest.approx(0.0, abs=1e-9)
    assert memo == pytest.approx(0.0, abs=1e-9)


def test_discharge_charge_relation_holds_for_daily_cycle() -> None:
    """日循环下 sum(q_dis) = 0.9 * sum(q_ch)，残差校验为 0。"""
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[0] = 500.0
    q_dis[5] = 450.0
    assert physics.discharge_charge_relation_residual_kwh(q_ch, q_dis) == pytest.approx(0.0)
    assert physics.loss_cycle_relation_residual_kwh(q_ch, q_dis) == pytest.approx(0.0)


def test_loss_cycling_diagnosis() -> None:
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[5] = 90.0
    q_dis[5] = 90.0 * ETA_DISCHARGE
    diag = physics.diagnose_loss_cycling(q_ch, q_dis)
    assert diag["n_intervals"] == 1
    assert diag["intervals"] == [5]
    assert diag["cycled_stored_kwh"] == pytest.approx(90.0)


def test_max_bus_power_reports_both_sides() -> None:
    q_ch = np.zeros(144)
    q_dis = np.zeros(144)
    q_ch[0] = Q_MAX
    q_dis[1] = Q_MAX
    info = physics.max_bus_power_kw(q_ch, q_dis)
    assert info["bus_charge_kw"] == pytest.approx(5000.0)
    assert info["bus_discharge_kw"] == pytest.approx(4500.0)
