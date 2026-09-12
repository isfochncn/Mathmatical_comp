"""Causal weekly load forecasts, calibrated PV and joint-error reserves.

All training rows must have completed before the decision. Calibration uses
historical *forecasts*, reconstructed with their original information cutoff,
not deviations from a model fitted retrospectively to the validation year.
"""
from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
import numpy as np
from scipy.optimize import nnls

from .forecast import Forecaster
from .planning import WindowForecast


class AdaptiveForecaster(Forecaster):
    def __init__(self, bundle, timeline, *, history_days=28, risk_quantile=.90):
        super().__init__(bundle, timeline, history_days=history_days)
        if not np.isfinite(risk_quantile) or not 0 <= risk_quantile < 1:
            raise ValueError("risk_quantile must be in [0, 1)")
        self.risk_quantile = risk_quantile

    def _profile(self, now, minutes, kind, days=7):
        series = self.timeline.load_kw.values if kind == 'load' else self.timeline.pv_kw.values
        clocks = minutes % 1440 // 10
        # Latest completed value of each clock slot, including today's mature slots.
        last = (now // 10 - 1 - clocks) // 144
        indices = (last[None, :] - np.arange(days)[:, None]) * 144 + clocks
        safe = np.clip(indices, 0, len(series)-1)
        valid = self._valid_history(indices, series, now)
        counts = valid.sum(axis=0)
        prior = self.cold_start_prior(now, clocks, 'load' if kind == 'load' else 'pv')
        sums = np.where(valid, series[safe], 0).sum(axis=0)
        return np.divide(sums, counts, out=prior.copy(), where=counts > 0) / 6

    def _load_prediction(self, now, minutes):
        series = self.timeline.load_kw.values
        # Target weekday matters even when the horizon crosses into tomorrow.
        idx = minutes[None, :] // 10 - 7*144*np.arange(1, 5)[:, None]
        safe = np.clip(idx, 0, len(series)-1)
        valid = self._valid_history(idx, series, now)
        weights = np.array([.7, .15, .1, .05])[:, None] * valid
        den = weights.sum(axis=0)
        pred = np.divide((np.where(valid, series[safe], 0)*weights).sum(axis=0), den,
                         out=self._profile(now, minutes, 'load')*6, where=den > 0) / 6
        # Short-term innovation relative to last week's observation, with lead decay.
        recent = np.arange(max(0, now//10-18), now//10)
        past = recent-7*144
        ok = self._valid_history(recent, series, now) & self._valid_history(past, series, now)
        if ok.any():
            bias = np.median((series[recent[ok]]-series[past[ok]])/6)
            pred += bias*np.exp(-(minutes-now)/360)
        return np.maximum(pred, 0)

    @lru_cache(maxsize=2048)
    def _published_day(self, origin):
        """Only legal publications as of origin; no actuals are accessed here."""
        minutes = np.arange(origin, origin+1440, 10)
        day = self.timeline.day_index0 + timedelta(days=origin//1440)
        base = self.timeline.day_index0.toordinal()*24
        out = np.full(144, np.nan)
        known_hour = day.toordinal()*24+origin%1440//60
        eligible = [(b.publish_day.toordinal()*24+b.publish_hour, b)
                    for b in self.bundle.attachment3.blocks
                    if known_hour-24 < b.publish_day.toordinal()*24+b.publish_hour <= known_hour]
        for published_hour, block in sorted(eligible, key=lambda pair: pair[0]):
            offsets = base+minutes//60-published_hour
            covered = (offsets >= 0) & (offsets < 24)
            out[covered] = block.hourly_power_kw[offsets[covered]]/6
        return out

    @lru_cache(maxsize=1536)
    def _pv_coefficients(self, anchor):
        xs, ys = [], []
        for lag in range(1, self.history_days+1):
            old = anchor-lag*1440
            if old < 10:
                continue
            minutes = np.arange(old, old+1440, 10)
            raw = self._published_day(old)
            hist = self._profile(old, minutes, 'pv')
            idx = minutes//10
            valid = self._valid_history(idx, self.timeline.pv_kw.values, anchor) & np.isfinite(raw)
            if valid.any():
                xs.append(np.column_stack((raw[valid], hist[valid])))
                ys.append(self.timeline.pv_kw.values[idx[valid]]/6)
        if len(xs) < 7:
            return np.array([1., 0.])
        x, y = np.concatenate(xs), np.concatenate(ys)
        # Shrink toward the supplied forecast; nonnegative coefficients avoid
        # unphysical predictions. Coefficients are learned exclusively on history.
        strength = max(1., .02*np.sum(x*x))
        return nnls(np.vstack((x, np.eye(2)*np.sqrt(strength))),
                    np.r_[y, np.sqrt(strength), 0])[0]

    @lru_cache(maxsize=3072)
    def _point(self, now, end, published):
        minutes = np.arange(now, end, 10)
        load = self._load_prediction(now, minutes)
        pv = self._profile(now, minutes, 'pv')
        if published:
            anchor = now//360*360
            raw = self._published_day(anchor)
            offsets = (minutes-anchor)//10
            covered = (offsets >= 0) & (offsets < 144)
            valid = covered.copy()
            valid[covered] &= np.isfinite(raw[offsets[covered]])
            coeff = self._pv_coefficients(anchor)
            pv[valid] = coeff[0]*raw[offsets[valid]] + coeff[1]*pv[valid]
        return load, np.maximum(pv, 0)

    def window_forecast(self, now_abs, abs_from, abs_to, *, use_published_pv,
                        current_price, price_mode='historical'):
        # Reuse the original price logic and timestamp validation unchanged.
        base = super().window_forecast(now_abs, abs_from, abs_to,
            use_published_pv=False, current_price=current_price, price_mode=price_mode)
        load, pv = self._point(now_abs, abs_to, use_published_pv)
        offset = (abs_from-now_abs)//10
        return WindowForecast(base.abs_minutes, load[offset:].copy(), pv[offset:].copy(),
            base.price_yuan_per_kwh,
            np.full(base.n, f'formed@{now_abs};weekly-load+recent-innovation;'
                    f'pv7d+causal-nnls={use_published_pv};price={price_mode}', dtype=object))

    @lru_cache(maxsize=1024)
    def _errors(self, anchor, end, published):
        samples = []
        for lag in range(1, self.history_days+1):
            old, stop = anchor-lag*1440, end-lag*1440
            if old < 10 or stop > anchor:
                continue
            idx = np.arange(old//10, stop//10)
            # A whole path must have matured. Preserve within-day correlation.
            if not (self._valid_history(idx, self.timeline.load_kw.values, anchor).all()
                    and self._valid_history(idx, self.timeline.pv_kw.values, anchor).all()):
                continue
            load, pv = self._point(old, stop, published)
            actual = (self.timeline.load_kw.values[idx]-self.timeline.pv_kw.values[idx])/6
            samples.append(actual-(load-pv))
        return np.array(samples).reshape(-1, (end-anchor)//10)

    @lru_cache(maxsize=1024)
    def _reserve_at(self, anchor, end, published, adjustable):
        errors = self._errors(anchor, end, published)
        n = (end-anchor)//10
        power, energy = np.zeros(n), np.zeros(n+1)
        if len(errors) and self.risk_quantile:
            # Upper empirical order statistic; no normality/independence assumption.
            rank = min(len(errors), int(np.ceil((len(errors)+1)*self.risk_quantile)))-1
            power = np.maximum(0, np.sort(errors, axis=0)[rank])
            # Largest cumulative shortage before the next legal purchase change.
            for t in range(n):
                minute = anchor+10*t
                block = 360 if adjustable else 1440
                stop = min(n, (((minute//block+1)*block)-anchor)//10)
                paths = np.cumsum(errors[:, t:stop], axis=1)
                maxima = np.maximum(0, paths.max(axis=1))
                energy[t] = np.sort(maxima)[rank]
        return power, energy, len(errors)

    def risk_requirements(self, now, forecast, *, published, adjustable):
        if not self.risk_quantile:
            return {}
        block = 360 if adjustable else 1440
        anchor = now//block*block
        if anchor == 0:
            anchor = 10
        end = int(forecast.abs_minutes[-1])+10
        power, energy, count = self._reserve_at(anchor, end, published, adjustable)
        offset = (now-anchor)//10
        return {'reserve_energy_kwh': energy[offset:].copy(),
                'net_upper_kwh': np.maximum(0, forecast.demand_kwh-forecast.pv_kwh+power[offset:]),
                'risk_sample_count': count}
