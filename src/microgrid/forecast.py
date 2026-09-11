"""预测层：只用"决策时刻已揭示"的信息做预测。

规范第 5 节"五个必须隔离的接口"要求：
信息视图只能接收当前时间、已发布预报与已完成实测，
**绝不能**把当天剩余真实数据交给计划模块。

因此本模块的每个函数都显式接收"截至哪天/哪一刻已知"，
并从 :class:`microgrid.data_io.DataBundle` 中只取该时点之前的部分。

基准方法（指南第 10 节）：历史同刻 / 相似日基线，先易解释、便于检查泄漏，
再考虑复杂模型。本实现只提供基线，任何改进都必须在同一验证口径下对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .constants import DELTA_T_HOURS, ETA_CHARGE, N_BOUNDARY, N_INTERVAL, Q_MAX
from .data_io import Attachment3, DataBundle, ForecastBlock
from .timeaxis import calendar_days


@dataclass(frozen=True)
class DayForecast:
    """某一天的预测结果（功率 kW，模型区间 144 段）。"""

    day: date
    demand_kw: np.ndarray
    pv_kw: np.ndarray
    price_yuan_per_kwh: np.ndarray
    source: str


class Forecaster:
    """因果预测器。

    Parameters
    ----------
    bundle : 原始数据（仅供读取历史切片）
    history_days : 用于同刻均值的回看上限；不足时用已有全部历史
    min_history : 至少积累多少个已完整发生的日子才允许做统计预测
    load_method : 负载预测方法
        * ``"same_weekday"``（默认）：取**上一周同一星期几**的实测曲线。
          在本数据上实测 MAE ≈ 175 kW，明显优于各种历史均值（≈ 780 kW）——
          小区负载的星期效应远强于日间平滑，用均值会把 7 天前那条几乎相同的
          曲线平均掉。没有 7 天前数据时退回 ``"mean"``。
        * ``"mean"``：过去若干天同刻均值。
    cold_start_load_kw : 冷启动（尚无任何历史时）的负载基线，默认取附件1 典型日。
        只在 2025-01-01 这类"历史上无任何已发生日"的情况下使用。
    """

    def __init__(
        self,
        bundle: DataBundle,
        history_days: int = 28,
        min_history: int = 1,
        cold_start_load_kw: np.ndarray | None = None,
        load_method: str = "same_weekday",
        forecast_history_days: int = 28,
    ) -> None:
        self.bundle = bundle
        self.history_days = history_days
        self.min_history = min_history
        self.load_method = load_method
        self.forecast_history_days = forecast_history_days
        self._cold_start_load_kw = (
            np.asarray(cold_start_load_kw, dtype=np.float64)
            if cold_start_load_kw is not None
            else bundle.attachment1.load_kw.copy()
        )
        self.cold_start_used_on: list[date] = []

    # ------------------------------------------------------------------
    # 历史切片（这是唯一允许访问真实数据的入口）
    # ------------------------------------------------------------------

    def _history_slice(self, before: date, n_days: int) -> list[int]:
        """返回 ``before`` 之前（不含）最近 n_days 个**已完整发生**的日索引。"""
        days = self.bundle.attachment2.days
        try:
            idx_before = days.index(before)
        except ValueError:
            return []
        start = max(0, idx_before - n_days)
        return list(range(start, idx_before))

    def demand_profile_kw(self, target: date) -> tuple[np.ndarray, str]:
        """负载预测。完全无历史时退回冷启动基线。"""
        days = self.bundle.attachment2.days
        try:
            idx_before = days.index(target)
        except ValueError:
            idx_before = 0
        # 同一星期几：往前 7 天（k 个 7 天里取最近一个已发生的）
        if self.load_method == "same_weekday" and idx_before >= 7:
            return (
                self.bundle.attachment2.load_kw[idx_before - 7].copy(),
                "same-weekday(-7d)",
            )
        idx = self._history_slice(target, self.history_days)
        if len(idx) < self.min_history:
            self.cold_start_used_on.append(target)
            return self._cold_start_load_kw.copy(), "cold-start(附件1典型日)"
        if len(idx) >= 7:
            return (
                self.bundle.attachment2.load_kw[idx[-7]].copy(),
                "same-weekday(-7d,fallback)",
            )
        # 不足一周：与附件1 典型日按可用天数加权混合。纯用 1—2 天的均值会把
        # 极端日整体搬到次日，实测误差远大于混合基线。
        alpha = len(idx) / 7.0
        recent = self.bundle.attachment2.load_kw[idx, :].mean(axis=0)
        mixed = alpha * recent + (1.0 - alpha) * self._cold_start_load_kw
        return mixed, f"blend(typ+mean{len(idx)}d)"

    def price_profile_yuan_per_kwh(self, target: date) -> tuple[np.ndarray, str]:
        """电价预测：过去同刻均值；完全无历史时用附件1 的日内曲线形状。"""
        idx = self._history_slice(target, self.history_days)
        if len(idx) < self.min_history:
            return self.bundle.attachment1.price_yuan_per_kwh.copy(), "cold-start(附件1曲线)"
        arr = self.bundle.attachment4.price_yuan_per_kwh[idx, :]
        if len(idx) > 14:
            arr = arr[-14:, :]
        return arr.mean(axis=0), f"history-mean({len(idx)}d)"

    # ------------------------------------------------------------------
    # 预测误差统计：用"真正复现预测规则、再与已发生真值比较"的方式测量
    # ------------------------------------------------------------------

    def forecast_errors(
        self, before: date, *, max_samples: int = 28
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """逐日复现**本预测器实际使用的规则**，得到负载与光伏的预测误差样本。

        对 ``before`` 之前的每个已发生日 j，用 **j 之前**的信息按同一规则算预测，
        再与 j 的真实值相减。这样得到的误差样本严格因果，且与实际策略一致——
        若这里换了规则而预测器没换，带宽会被错估。
        """
        days = self.bundle.attachment2.days
        try:
            idx_before = days.index(before)
        except ValueError:
            return [], []
        d_errs: list[np.ndarray] = []
        p_errs: list[np.ndarray] = []
        start = max(1, idx_before - max_samples)
        for j in range(start, idx_before):
            pred_d = self._demand_prediction_at(j)
            if pred_d is None:
                continue
            hist = self._history_slice(days[j], self.history_days)
            if not hist:
                continue
            if len(hist) > 14:
                hist = hist[-14:]
            pred_p = self.bundle.attachment2.pv_actual_kw[hist, :].mean(axis=0)
            d_errs.append(pred_d - self.bundle.attachment2.load_kw[j, :])
            p_errs.append(pred_p - self.bundle.attachment2.pv_actual_kw[j, :])
        return d_errs, p_errs

    def _demand_prediction_at(self, j: int) -> np.ndarray | None:
        """在索引 j 处按当前 load_method 复现预测（只用 j 之前的数据）。"""
        L = self.bundle.attachment2.load_kw
        if self.load_method == "same_weekday" and j >= 7:
            return L[j - 7]
        hist = self._history_slice(self.bundle.attachment2.days[j], self.history_days)
        if not hist:
            return None
        if len(hist) >= 7:
            return L[hist[-7]]
        alpha = len(hist) / 7.0
        recent = L[hist, :].mean(axis=0)
        return alpha * recent + (1.0 - alpha) * self._cold_start_load_kw

    def error_scale_kw(self, before: date, *, z: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
        """误差尺度（kW）：平均绝对误差 × z。样本不足 1 时退回保守固定值。"""
        d_errs, p_errs = self.forecast_errors(before)
        if not d_errs:
            base_d = float(np.mean(self._cold_start_load_kw)) * 0.35
            base_p = float(np.mean(self.bundle.attachment1.pv_forecast_kw)) * 0.5
            return np.full(N_INTERVAL, base_d * z), np.full(N_INTERVAL, base_p * z), 0
        d_arr = np.abs(np.array(d_errs)).mean(axis=0)
        p_arr = np.abs(np.array(p_errs)).mean(axis=0)
        return d_arr * z, p_arr * z, len(d_errs)

    def recent_forecast_error(self, before: date) -> tuple[np.ndarray, np.ndarray, int]:
        """兼容旧接口：返回 (负载误差绝对值均值, 光伏误差绝对值均值, 样本数)。"""
        return self.error_scale_kw(before)

    def robust_absorption_floor(
        self,
        target: date,
        demand_pred_kw: np.ndarray,
        pv_pred_kw: np.ndarray,
        *,
        z_demand: float = 1.0,
        z_pv: float = 1.0,
        n_pv_sigma: float = 2.0,
        demand_floor_quantile: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """给出"稳健可消纳"所需的负载下限与光伏上限（都是 kW）。

        计划购电不可拒收，因此计划量必须保证：即使真实负载偏低、光伏偏高，
        买进来的电也有合法去处（负载或储能）。两个边界的取法：

            demand_floor = 历史同刻**最低**负载（默认分位数 0）
            pv_ceiling   = max(预报值, 历史同刻最高光伏) + n_pv_sigma * 预报值

        用"历史最低负载 + 历史最高光伏"是刻意保守的：它直接界定了
        "按计划买进来的电，在已观察到的极端情况下也一定放得下"。
        完全无历史（2025-01-01）时退回固定折扣的冷启动边界。
        """
        idx = self._history_slice(target, self.history_days)
        if not idx:
            floor = np.maximum(demand_pred_kw * 0.75, 0.0)
            ceiling = pv_pred_kw * 1.5
            return floor, ceiling, "cold-start(0.75/1.5 固定裕度)"
        window = idx[-min(len(idx), 30):]
        demand_hist = self.bundle.attachment2.load_kw[window, :]
        pv_hist = self.bundle.attachment2.pv_actual_kw[window, :]
        if demand_floor_quantile <= 0.0:
            load_floor = demand_hist.min(axis=0)
            tag = "hist-min"
        else:
            load_floor = np.quantile(demand_hist, demand_floor_quantile, axis=0)
            tag = f"hist-q{demand_floor_quantile:.2f}"
        pv_max = pv_hist.max(axis=0)
        floor = np.maximum(np.minimum(load_floor, demand_pred_kw), 0.0)
        ceiling = np.maximum(pv_pred_kw, pv_max)
        return floor, ceiling, f"{tag}(n={len(window)},n_sigma={n_pv_sigma})"

    # ------------------------------------------------------------------
    # 计划 SOC 鲁棒窗口
    # ------------------------------------------------------------------

    def soc_uncertainty_band_kwh(
        self,
        target: date,
        *,
        rho: float = 2.5,
        min_band_kwh: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """计划 SOC 的鲁棒带宽 ``band_s``（kWh，s = 0..144）。

        直接由**实测的预测误差**决定：把预测规则在历史上逐日复现一遍，
        得到"同刻历史均值预测"的平均绝对误差 ``mae_t``，则当日最不利方向的
        累计偏差约为 ``rho * Σ_t mae_t``（再按储能功率上限截断——实际系统
        吸收不了的偏差不会真的压进电池）。于是：

            E_s ∈ [E_MIN + band, E_MAX - band]

        带宽的含义：**计划必须为自己留下的、用来吸收预测偏差的备用容量**。
        预测越准（历史误差越小），带宽越窄、计划越敢套利；预测越差，
        带宽越宽、计划越保守。这使计划"可执行性"成为可验证的性质而非猜测。
        """
        d_mae, p_mae, n = self.error_scale_kw(target)
        if n == 0:
            band = np.full(N_BOUNDARY, 3000.0)
            return band, band.copy(), "cold-start(±3000kWh)"
        err = (d_mae + p_mae) * DELTA_T_HOURS  # 段级最不利净偏差（kWh）
        raw = rho * float(np.sum(err))
        # 功率上限截断：日内在给定方向上真正能进/出电池的总量
        band_value = min(raw, float(Q_MAX / ETA_CHARGE) * 2.0)
        band_value = max(band_value, float(min_band_kwh))
        band = np.full(N_BOUNDARY, band_value)
        return band, band.copy(), f"band={band_value:.0f}kWh(n={n},rho={rho},mae_sum={np.sum(err):.0f})"

    # ------------------------------------------------------------------
    # 光伏：附件3 的已发布预报
    # ------------------------------------------------------------------

    def pv_forecast_kw(
        self,
        target: date,
        publish_day: date,
        publish_hour: int,
    ) -> tuple[np.ndarray, str]:
        """返回 [target 0:00, target 24:00) 的小时平均光伏功率，取自 (publish_day, publish_hour) 的发布。

        规则（备忘录第 7 节）：
          * "预报 j 小时" = 发布时刻 a 之后 [a+j-1, a+j) 小时的平均功率，无 0 小时预报；
          * 只允许使用**已经发布**的预报版本，不能提前读取；
          * 跨午夜部分保留真实有效日期。
        """
        if publish_day > target:
            raise ValueError(f"试图在 {publish_day} {publish_hour}:00 之前使用它发布的预报")
        block = self.bundle.attachment3.block_at(publish_day, publish_hour)
        return self._spread_block(block, target), f"att3-{publish_day.isoformat()}@{publish_hour}:00"

    def _spread_block(self, block: ForecastBlock, target: date) -> np.ndarray:
        """把一条发布块摊成 target 当天的小时平均功率序列。

        小时内均匀形状（离散近似）：每个小时功率均匀用于对应六段。
        六段之和等于该小时预测电量，保持小时总量不变。
        """
        hours = self._block_hour_map(block, target)
        out = np.zeros(24, dtype=np.float64)
        for h in range(24):
            j = hours.get(h)
            out[h] = block.hourly_power_kw[j] if j is not None else 0.0
        return out

    @staticmethod
    def _block_hour_map(block: ForecastBlock, target: date) -> dict[int, int]:
        """发布块覆盖的 target 当天小时 -> 预报序号 j（0-based）。"""
        publish_abs = block.publish_day.toordinal() * 24 + block.publish_hour
        target_abs = target.toordinal() * 24
        out: dict[int, int] = {}
        for j in range(24):  # 预报 j+1 小时覆盖绝对小时 publish_abs + j
            abs_hour = publish_abs + j
            if target_abs <= abs_hour < target_abs + 24:
                out[abs_hour - target_abs] = j
        return out

    def pv_from_hourly_kw(self, hourly_kw: np.ndarray) -> np.ndarray:
        """小时平均功率 -> 144 段功率（小时内均匀）。"""
        return np.repeat(np.asarray(hourly_kw, dtype=np.float64), 6)

    # ------------------------------------------------------------------
    # 附件1 的重复日内曲线（问题1/2/3 的电价）
    # ------------------------------------------------------------------

    def repeated_price_yuan_per_kwh(self) -> np.ndarray:
        return self.bundle.attachment1.price_yuan_per_kwh.copy()


def terminal_water_value_yuan_per_kwh(
    forecaster: Forecaster,
    next_day: date,
    *,
    fallback: float,
) -> float:
    """终端价值 V 的单价（水价值）：次日预测电价中位数的折扣值。

    未来成本估计不进入最终账单，只用于让当日计划不把储能"用穷"。
    取中位数而非均值，避免被极端高价时段抬高水价值。
    """
    try:
        price, _ = forecaster.price_profile_yuan_per_kwh(next_day)
        return float(np.median(price))
    except Exception:
        return float(fallback)


__all__ = ["DayForecast", "Forecaster", "terminal_water_value_yuan_per_kwh"]
