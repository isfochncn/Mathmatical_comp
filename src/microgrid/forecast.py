"""Point forecasting on the absolute timeline.

Main-model baseline (memo section 5, fixed 2026-09-11)
-----------------------------------------------------
For load, historical PV and problem-4 price, take the mean of samples at the
**same clock interval** over the last 28 natural days that have already been
observed. Fewer than 28 days -> use whatever exists. When a clock interval has
no sample at all but history exists, fall back to the mean of the last 24
completed hours **of that same variable** and record the fallback flag.
Forecast power is clipped to be non-negative.

Problem 4 adds a causal bias correction: when the current quote is known and a
same-clock historical mean exists, shift the remaining price forecast by their
difference, then clip to non-negative. When it is not known, use the historical
baseline and do not invent a quote.

PV from attachment 3
--------------------
Problems 3, 4-2 and 4-3 prefer the **latest version published at or before the
current moment** that covers the target interval. Versions not yet published
must never be read. Whatever the single 24-hour release cannot cover (the
window routinely reaches past it) is filled by the historical PV point forecast;
that splicing step is a necessary part of the main model and is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DELTA_T_HOURS, N_INTERVAL
from .data_io import DataBundle
from .planning import WindowForecast
from .timeaxis import SOURCE_ATTACHMENT, SOURCE_EXTENDED
from .timeline import INTERVALS_PER_DAY, MINUTES_PER_DAY, Timeline

HISTORY_DAYS = 28
FALLBACK_HOURS = 24


@dataclass(frozen=True)
class ForecastRecord:
    """One point forecast plus where it came from."""

    values: np.ndarray
    source: str
    fallback_mask: np.ndarray


class Forecaster:
    """Causal point forecaster over the absolute timeline.

    Parameters
    ----------
    bundle : read-only attachments.
    timeline : absolute series with provenance.
    history_days : same-clock lookback window (technical baseline, not a task
        parameter; its sensitivity is comparison item A1).
    """

    def __init__(
        self,
        bundle: DataBundle,
        timeline: Timeline,
        history_days: int = HISTORY_DAYS,
        fallback_hours: int = FALLBACK_HOURS,
    ) -> None:
        self.bundle = bundle
        self.timeline = timeline
        self.history_days = history_days
        self.fallback_hours = fallback_hours

    # ------------------------------------------------------------------
    # History access: only samples that have already been observed
    # ------------------------------------------------------------------

    def _observed_end_abs(self, now_abs: int) -> int:
        """Last absolute interval start strictly before ``now_abs``."""
        return now_abs - 10

    def _same_clock_samples(self, now_abs: int, series_values: np.ndarray) -> tuple[np.ndarray, int]:
        """(n_days, clock_interval) samples at the same clock slot.

        Rows are the most recent ``history_days`` natural days whose intervals
        are all before ``now_abs``.
        """
        last_start = self._observed_end_abs(now_abs)
        if last_start < 0:
            return np.zeros((0, INTERVALS_PER_DAY)), 0
        last_day = last_start // MINUTES_PER_DAY
        day_index = last_day - 1  # only fully completed days
        if day_index < 0:
            return np.zeros((0, INTERVALS_PER_DAY)), 0
        first_day = max(0, day_index - self.history_days + 1)
        rows = series_values[first_day * INTERVALS_PER_DAY : (day_index + 1) * INTERVALS_PER_DAY]
        return rows.reshape(-1, INTERVALS_PER_DAY), day_index - first_day + 1

    def cold_start_prior(
        self, now_abs: int, clock_intervals: np.ndarray, kind: str
    ) -> np.ndarray:
        """Sourced cold-start prior for the very first hours, when no history exists.

        The memo requires an initialisation prior **with a stated source** and
        forbids calling future January data to initialise. Two different sources
        are used, because a single one would be badly wrong:

        * ``"pv"``   -- attachment 3's own 00:00 release for the current day.
          It is genuinely published at the decision moment and, unlike
          attachment 1, is seasonally correct: attachment 1's typical-day PV is
          about six times larger than the real 1 January output, so using it
          would make the model plan as if it were summer.
        * ``"load"`` -- attachment 1's typical-day profile **scaled** by the
          ratio of the first already-observed load sample to the same slot of
          that profile, so the level is anchored to reality rather than trusting
          a possibly unrepresentative level.
        """
        a1 = self.bundle.attachment1
        if kind == "pv":
            day_index = now_abs // MINUTES_PER_DAY
            day = self.timeline.day_index0.fromordinal(
                self.timeline.day_index0.toordinal() + day_index
            )
            try:
                hourly = self.bundle.attachment3.block_at(day, 0).hourly_power_kw
            except Exception:
                return a1.pv_forecast_kw[clock_intervals].astype(np.float64)
            return (hourly[clock_intervals // 6] * DELTA_T_HOURS).astype(np.float64)

        base = a1.load_kw.astype(np.float64).copy()
        observed = self.timeline.load_kw.values
        first_observed = float(observed[1]) if observed.size > 1 else float(base[1])
        scale = first_observed / float(base[1]) if base[1] > 0 else 1.0
        return base[clock_intervals] * scale

    def _recent_hours_mean(self, now_abs: int, series_values: np.ndarray, clock_t: int) -> float | None:
        """Fallback: mean of the last 24 completed hours at the *same clock slot*.

        The sample is taken by absolute time modulo one day, so it really is the
        same time of day; a naive stride over the recent window would pick up
        unrelated clock slots.
        """
        last_start = self._observed_end_abs(now_abs)
        if last_start < MINUTES_PER_DAY:
            return None
        first_start = last_start - (MINUTES_PER_DAY - 10)
        target = clock_t * 10
        abs_minute = first_start + ((target - first_start) % MINUTES_PER_DAY)
        samples = []
        while abs_minute <= last_start:
            samples.append(float(series_values[abs_minute // 10]))
            abs_minute += MINUTES_PER_DAY
        if not samples:
            return None
        return float(np.mean(samples))

    def point_forecast(
        self,
        now_abs: int,
        series_values: np.ndarray,
        clock_intervals: np.ndarray,
        name: str,
        *,
        clip_non_negative: bool = True,
        cold_start_kw: np.ndarray | None = None,
    ) -> ForecastRecord:
        """Same-clock 28-day mean forecast for the requested clock intervals.

        Cold start: on 2025-01-01 no same-clock sample exists yet. The memo
        requires an initialisation prior **with a stated source** and forbids
        calling future January data to initialise. We therefore fall back to
        attachment 1's published typical-day profile, which is given by the task
        statement itself, and tag every such interval as ``cold_start``.
        """
        samples, n_days = self._same_clock_samples(now_abs, series_values)
        out = np.zeros(clock_intervals.size, dtype=np.float64)
        fallback = np.zeros(clock_intervals.size, dtype=bool)
        cold = np.zeros(clock_intervals.size, dtype=bool)
        n_24h = 0
        n_cold = 0
        for i, t in enumerate(clock_intervals):
            est: float | None = None
            if n_days > 0:
                est = float(np.mean(samples[:, t]))
            if est is None:
                est = self._recent_hours_mean(now_abs, series_values, int(t))
                if est is not None:
                    fallback[i] = True
                    n_24h += 1
            if est is None:
                if cold_start_kw is None:
                    raise ValueError(
                        f"{name}: 在 {now_abs} 之前没有任何已观测样本，且未给冷启动先验"
                    )
                est = float(cold_start_kw[int(t)])
                cold[i] = True
                fallback[i] = True
                n_cold += 1
            out[i] = est
        if clip_non_negative:
            out = np.maximum(out, 0.0)
        parts = [f"same-clock-mean({n_days}d)"]
        if n_24h:
            parts.append(f"24h-fallback({n_24h})")
        if n_cold:
            parts.append(f"cold-start-attachment1({n_cold})")
        return ForecastRecord(values=out, source="+".join(parts), fallback_mask=fallback)

    # ------------------------------------------------------------------
    # Price with causal bias correction (problem 4)
    # ------------------------------------------------------------------

    def price_forecast(
        self,
        now_abs: int,
        clock_intervals: np.ndarray,
        *,
        current_price: float | None,
    ) -> ForecastRecord:
        """Historical same-clock mean, optionally shifted by the current quote.

        The shift is causal: it uses only the currently published price and the
        same-clock historical mean. The shift value is one scalar (the current
        deviation), never the future average.
        """
        base = self.point_forecast(
            now_abs,
            self.timeline.price_yuan_per_kwh.values,
            clock_intervals,
            "price",
            cold_start_kw=self.bundle.attachment1.price_yuan_per_kwh,
        )
        if current_price is None:
            return ForecastRecord(base.values, base.source + "|no-quote", base.fallback_mask)

        # Same-clock historical mean at the *current* clock slot.
        current_clock = (now_abs % MINUTES_PER_DAY) // 10
        samples, n_days = self._same_clock_samples(now_abs, self.timeline.price_yuan_per_kwh.values)
        if n_days == 0:
            return ForecastRecord(base.values, base.source + "|quote-no-history", base.fallback_mask)
        hist_now = float(np.mean(samples[:, current_clock]))
        shift = float(current_price) - hist_now
        shifted = np.maximum(base.values + shift, 0.0)
        return ForecastRecord(
            shifted,
            base.source + f"|bias-shift({shift:+.4f})",
            base.fallback_mask,
        )

    # ------------------------------------------------------------------
    # PV: published attachment-3 releases, spliced with history
    # ------------------------------------------------------------------

    def pv_forecast_kwh(
        self,
        now_abs: int,
        abs_minutes: np.ndarray,
        *,
        use_published: bool,
    ) -> ForecastRecord:
        """PV energy forecast for the window's absolute intervals.

        ``use_published=False`` (problem 2) uses the historical point forecast
        only. Otherwise every interval first tries the latest release published
        at or before ``now_abs``; uncovered intervals fall back to history.
        """
        clock_intervals = (abs_minutes % MINUTES_PER_DAY) // 10
        hist = self.point_forecast(
            now_abs,
            self.timeline.pv_kw.values,
            clock_intervals,
            "pv",
            cold_start_kw=self.cold_start_prior(now_abs, clock_intervals, "pv"),
        )
        out = hist.values.copy()
        fallback = hist.fallback_mask.copy()
        source = hist.source

        if not use_published:
            return ForecastRecord(out, "historical|no-att3", fallback)

        a3 = self.bundle.attachment3
        now_day_index = now_abs // MINUTES_PER_DAY
        now_clock = (now_abs % MINUTES_PER_DAY) // 10
        request_date = self.timeline.day_index0.fromordinal(
            self.timeline.day_index0.toordinal() + now_day_index
        )
        request_hour = now_clock // 6  # 00 / 06 / 12 / 18 publication clock
        covered = np.zeros(abs_minutes.size, dtype=bool)

        for i, abs_minute in enumerate(abs_minutes):
            target_day_index = abs_minute // MINUTES_PER_DAY
            target_clock = (abs_minute % MINUTES_PER_DAY) // 10
            target_hour = target_clock // 6
            target_date = self.timeline.day_index0.fromordinal(
                self.timeline.day_index0.toordinal() + target_day_index
            )
            target_abs_hour = target_date.toordinal() * 24 + target_hour
            found = a3.latest_published_covering(request_date, request_hour, target_abs_hour)
            if found is None:
                continue
            block, offset = found
            out[i] = float(block.hourly_power_kw[offset]) / 6.0 * 1.0  # kW -> kWh per interval
            out[i] = float(block.hourly_power_kw[offset]) * DELTA_T_HOURS
            covered[i] = True
            fallback[i] = False

        if covered.any():
            n_cov = int(covered.sum())
            source = f"att3-latest-published({n_cov}/{abs_minutes.size})+historical"
        return ForecastRecord(np.maximum(out, 0.0), source, fallback)

    # ------------------------------------------------------------------
    # One whole window
    # ------------------------------------------------------------------

    def window_forecast(
        self,
        now_abs: int,
        abs_from: int,
        abs_to: int,
        *,
        use_published_pv: bool,
        current_price: float | None,
        price_mode: str = "historical",
    ) -> WindowForecast:
        """Build the point-forecast trajectory for [abs_from, abs_to).

        Parameters
        ----------
        price_mode : ``"repeated"`` (problems 1-3 use the given daily curve) or
            ``"historical"`` (problem 4 predicts the price).
        """
        if abs_to <= abs_from:
            raise ValueError("窗口为空")
        # Only intervals the model will actually trade: those not yet executed.
        minutes = np.arange(abs_from, abs_to, 10, dtype=np.int64)
        clock_intervals = (minutes % MINUTES_PER_DAY) // 10

        demand = self.point_forecast(
            now_abs,
            self.timeline.load_kw.values,
            clock_intervals,
            "demand",
            cold_start_kw=self.cold_start_prior(now_abs, clock_intervals, "load"),
        )
        demand_kwh = demand.values * DELTA_T_HOURS

        pv = self.pv_forecast_kwh(now_abs, minutes, use_published=use_published_pv)

        if price_mode == "repeated":
            price = ForecastRecord(
                values=self.bundle.attachment1.price_yuan_per_kwh[clock_intervals].astype(np.float64),
                source="attachment1-daily-curve",
                fallback_mask=np.zeros(minutes.size, dtype=bool),
            )
        elif price_mode == "historical":
            price = self.price_forecast(now_abs, clock_intervals, current_price=current_price)
        else:
            raise ValueError(f"未知 price_mode：{price_mode}")

        provenance = np.array(
            [SOURCE_ATTACHMENT] * minutes.size, dtype=object
        )
        # Mark intervals that rely on a non-measured extension.
        ext = self.timeline.demand_kwh.provenance_window(abs_from, abs_to)
        provenance[:] = ext

        return WindowForecast(
            abs_minutes=minutes,
            demand_kwh=demand_kwh,
            pv_kwh=pv.values,
            price_yuan_per_kwh=np.maximum(price.values, 0.0),
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Robustness bounds (technical approximation, comparison item A1)
    # ------------------------------------------------------------------

    def absorption_upper_kwh(
        self,
        now_abs: int,
        abs_minutes: np.ndarray,
        *,
        safety_kwh: float,
    ) -> np.ndarray:
        """Upper bound on the normal purchase quantity per interval.

        Delivered power cannot be rejected, so a commitment must stay
        absorbable even when the realised load is low. The bound uses the
        historical same-clock **minimum** load plus the maximum charging rate::

            grid <= load_min_same_clock + Q_MAX/eta_c - safety

        This is a technical guard, not a task condition; it is recorded so the
        sensitivity of results to it can be reported under A1.
        """
        from .constants import ETA_CHARGE, Q_MAX

        clock_intervals = (abs_minutes % MINUTES_PER_DAY) // 10
        samples, n_days = self._same_clock_samples(now_abs, self.timeline.load_kw.values)
        cap = np.zeros(abs_minutes.size, dtype=np.float64)
        for i, t in enumerate(clock_intervals):
            if n_days > 0:
                floor_kw = float(np.min(samples[:, t]))
            else:
                floor_kw = float(self.timeline.load_kw.values[:INTERVALS_PER_DAY][t]) * 0.75
            cap[i] = max(
                floor_kw * DELTA_T_HOURS + Q_MAX / ETA_CHARGE - float(safety_kwh), 0.0
            )
        return cap


__all__ = ["Forecaster", "ForecastRecord", "HISTORY_DAYS", "FALLBACK_HOURS"]
