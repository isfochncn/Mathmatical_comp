"""Causal forecasts in kW, converted to interval kWh exactly once."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import DELTA_T_HOURS
from .data_io import DataBundle
from .planning import WindowForecast
from .timeaxis import SOURCE_BRIDGE_MISSING, SOURCE_EXTENDED
from .timeline import Timeline, MINUTES_PER_DAY, INTERVALS_PER_DAY

HISTORY_DAYS = 28
FALLBACK_HOURS = 24


@dataclass(frozen=True)
class ForecastRecord:
    values: np.ndarray
    source: str
    fallback_mask: np.ndarray


class Forecaster:
    def __init__(self, bundle: DataBundle, timeline: Timeline,
                 history_days: int = HISTORY_DAYS, fallback_hours: int = FALLBACK_HOURS,
                 load_method: str = "same_clock_mean") -> None:
        if history_days < 1 or fallback_hours < 1:
            raise ValueError("History windows must be positive")
        if load_method not in ("same_clock_mean", "same_weekday"):
            raise ValueError("Unknown load prediction method")
        self.bundle, self.timeline = bundle, timeline
        self.history_days, self.fallback_hours = history_days, fallback_hours
        self.load_method = load_method

    def _observed_end_abs(self, now_abs: int) -> int:
        return now_abs - 10

    def _valid_history(self, indices: np.ndarray, values: np.ndarray, now_abs: int) -> np.ndarray:
        valid = (indices >= 0) & (indices < len(values)) & ((indices + 1) * 10 <= now_abs)
        safe = np.clip(indices, 0, len(values) - 1)
        provenance = self.timeline.demand_kwh.provenance[safe]
        return valid & np.isfinite(values[safe]) & ~np.isin(
            provenance, [SOURCE_BRIDGE_MISSING, SOURCE_EXTENDED])

    def _same_clock_samples(self, now_abs: int, series_values: np.ndarray) -> tuple[np.ndarray, int]:
        day = now_abs // MINUTES_PER_DAY
        days = np.arange(max(0, day - self.history_days), day + 1)
        indices = days[:, None] * INTERVALS_PER_DAY + np.arange(INTERVALS_PER_DAY)
        safe = np.clip(indices, 0, len(series_values) - 1)
        samples = np.where(self._valid_history(indices, series_values, now_abs),
                           series_values[safe], np.nan)
        # At most the latest 28 completed observations for each clock slot.
        for t in range(INTERVALS_PER_DAY):
            rows = np.flatnonzero(np.isfinite(samples[:, t]))
            samples[rows[:-self.history_days], t] = np.nan
        return samples, int(np.any(np.isfinite(samples), axis=1).sum())

    def cold_start_prior(self, now_abs: int, clock_intervals: np.ndarray, kind: str) -> np.ndarray:
        """Explicit initialization approximation: supplied attachment-1 typical day.

        This prior never reads actual future observations or attachment 3.
        Source columns start at 00:10; roll once to obtain natural-day slots.
        """
        a1 = self.bundle.attachment1
        source = {"load": a1.load_kw, "pv": a1.pv_forecast_kw,
                  "price": a1.price_yuan_per_kwh}[kind]
        return np.roll(np.asarray(source, dtype=float), 1)[clock_intervals].copy()

    def _recent_hours_mean(self, now_abs: int, series_values: np.ndarray,
                           clock_t: int = 0) -> float | None:
        end = now_abs // 10
        indices = np.arange(max(0, end - self.fallback_hours * 6), end)
        indices = indices[self._valid_history(indices, series_values, now_abs)]
        return float(np.mean(series_values[indices])) if indices.size else None

    def point_forecast(self, now_abs: int, series_values: np.ndarray,
                       clock_intervals: np.ndarray, name: str, *,
                       clip_non_negative: bool = True,
                       cold_start_kw: np.ndarray | None = None) -> ForecastRecord:
        samples, n_days = self._same_clock_samples(now_abs, series_values)
        recent = self._recent_hours_mean(now_abs, series_values)
        out = np.empty(len(clock_intervals))
        fallback = np.zeros(len(out), dtype=bool)
        for i, t in enumerate(clock_intervals):
            values = samples[:, t]
            values = values[np.isfinite(values)]
            if name == "demand" and self.load_method == "same_weekday":
                idx = (now_abs // MINUTES_PER_DAY - 7) * 144 + int(t)
                if self._valid_history(np.array([idx]), series_values, now_abs)[0]:
                    out[i] = series_values[idx]
                    continue
            if values.size:
                out[i] = np.mean(values)
            elif recent is not None:
                out[i], fallback[i] = recent, True
            elif cold_start_kw is not None:
                out[i], fallback[i] = cold_start_kw[int(t)], True
            else:
                raise ValueError(f"{name}: no mature history or sourced initialization prior")
        if clip_non_negative:
            out = np.maximum(out, 0)
        return ForecastRecord(out, f"{self.load_method}({n_days}d); fallback=recent24h/attachment1",
                              fallback)

    def price_forecast(self, now_abs: int, clock_intervals: np.ndarray, *,
                       current_price: float | None) -> ForecastRecord:
        base = self.point_forecast(now_abs, self.timeline.price_yuan_per_kwh.values,
                                   clock_intervals, "price",
                                   cold_start_kw=self.cold_start_prior(now_abs, np.arange(144), "price"))
        if current_price is None:
            return base
        samples, _ = self._same_clock_samples(now_abs, self.timeline.price_yuan_per_kwh.values)
        values = samples[:, (now_abs % 1440) // 10]
        values = values[np.isfinite(values)]
        if not values.size:
            return base
        return ForecastRecord(np.maximum(base.values + current_price - values.mean(), 0),
                              base.source + "|current-quote-bias", base.fallback_mask)

    def pv_forecast_kwh(self, now_abs: int, abs_minutes: np.ndarray, *,
                        use_published: bool) -> ForecastRecord:
        clocks = (abs_minutes % 1440) // 10
        hist = self.point_forecast(now_abs, self.timeline.pv_kw.values, clocks, "pv",
                                  cold_start_kw=self.cold_start_prior(now_abs, np.arange(144), "pv"))
        out = hist.values * DELTA_T_HOURS
        fallback = hist.fallback_mask.copy()
        if not use_published:
            return ForecastRecord(out, "historical;no-att3;" + hist.source, fallback)
        from datetime import timedelta
        day = self.timeline.day_index0 + timedelta(days=now_abs // 1440)
        origin_hour = self.timeline.day_index0.toordinal() * 24
        covered = 0
        for i, minute in enumerate(abs_minutes):
            found = self.bundle.attachment3.latest_published_covering(
                day, (now_abs % 1440) // 60, origin_hour + int(minute) // 60)
            if found is not None:
                block, offset = found
                out[i] = block.hourly_power_kw[offset] * DELTA_T_HOURS
                fallback[i] = False
                covered += 1
        return ForecastRecord(np.maximum(out, 0), f"att3({covered});" + hist.source, fallback)

    def window_forecast(self, now_abs: int, abs_from: int, abs_to: int, *,
                        use_published_pv: bool, current_price: float | None,
                        price_mode: str = "historical") -> WindowForecast:
        if abs_from < now_abs or abs_to <= abs_from or any(x % 10 for x in (now_abs, abs_from, abs_to)):
            raise ValueError("Invalid forecast interval")
        minutes = np.arange(abs_from, abs_to, 10, dtype=np.int64)
        clocks = (minutes % 1440) // 10
        demand = self.point_forecast(now_abs, self.timeline.load_kw.values, clocks, "demand",
                                    cold_start_kw=self.cold_start_prior(now_abs, np.arange(144), "load"))
        pv = self.pv_forecast_kwh(now_abs, minutes, use_published=use_published_pv)
        if price_mode == "repeated":
            price = ForecastRecord(self.cold_start_prior(now_abs, clocks, "price"),
                                   "attachment1-repeated", np.zeros(len(minutes), dtype=bool))
        elif price_mode == "historical":
            price = self.price_forecast(now_abs, clocks, current_price=current_price)
        else:
            raise ValueError("Unknown price mode")
        return WindowForecast(minutes, demand.values * DELTA_T_HOURS, pv.values,
                              np.maximum(price.values, 0),
                              np.full(len(minutes), f"formed@{now_abs};{demand.source};{pv.source};{price.source}", dtype=object))

    def absorption_upper_kwh(self, now_abs: int, abs_minutes: np.ndarray, *, safety_kwh: float) -> np.ndarray:
        """Optional experiment only; uses mature history and the sourced prior."""
        from .constants import Q_MAX, ETA_CHARGE
        clocks = abs_minutes % 1440 // 10
        samples, _ = self._same_clock_samples(now_abs, self.timeline.load_kw.values)
        prior = self.cold_start_prior(now_abs, np.arange(144), "load")
        floors = [np.nanmin(samples[:, t]) if np.isfinite(samples[:, t]).any() else prior[t] for t in clocks]
        return np.maximum(np.asarray(floors) / 6 + Q_MAX / ETA_CHARGE - safety_kwh, 0)

    def commitment_floor_kwh(self, now_abs: int, abs_minutes: np.ndarray, *, safety_kwh: float) -> np.ndarray:
        clocks = abs_minutes % 1440 // 10
        loads, _ = self._same_clock_samples(now_abs, self.timeline.load_kw.values)
        pvs, _ = self._same_clock_samples(now_abs, self.timeline.pv_kw.values)
        net = loads - pvs
        return np.array([max(0, (np.nanmin(net[:, t]) / 6 if np.isfinite(net[:, t]).any() else 0)
                             + safety_kwh) for t in clocks])
