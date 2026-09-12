"""Causal, hour-specific correction of published PV for problem three.

The dispatch runner selects the tested blend with causal nowcasting. Coefficients and
blend weights use only completed observations; archived forecasts are reconstructed
with their original cutoff. All PV values are per-ten-minute kWh.
"""
from functools import lru_cache
import numpy as np
from .adaptive_forecast import AdaptiveForecaster


class ReportAwareForecaster(AdaptiveForecaster):
    def __init__(self, bundle, timeline, *, variant='blend', nowcast=True, **kwargs):
        super().__init__(bundle, timeline, **kwargs)
        if variant not in ('history', 'shape', 'calibrated', 'blend'):
            raise ValueError('Unknown published-PV candidate')
        self.variant, self.nowcast = variant, nowcast

    @staticmethod
    def distribute_hourly(raw, history):
        """Disaggregate each hourly mean, preserving the supplied hourly energy."""
        r, h = np.asarray(raw).reshape(-1, 6), np.asarray(history).reshape(-1, 6)
        means = h.mean(axis=1, keepdims=True)
        shape = np.divide(h, means, out=np.ones_like(h), where=means > 1e-8)
        return (r*shape).reshape(-1)

    @lru_cache(maxsize=1024)
    def _report_components(self, anchor):
        minutes = np.arange(anchor, anchor+1440, 10)
        history = self._profile(anchor, minutes, 'pv')
        raw = self._published_day(anchor)
        shaped = self.distribute_hourly(raw, history)
        numerator, denominator, mass = np.zeros(24), np.zeros(24), np.zeros(24)
        counts = np.zeros(24, dtype=int)
        for lag in range(1, self.history_days+1):
            old = anchor-lag*1440
            if old < 0:
                continue
            idx = np.arange(old//10, old//10+144)
            old_raw = self._published_day(old).reshape(24, 6)
            valid = (self._valid_history(idx, self.timeline.pv_kw.values, anchor).reshape(24, 6).all(axis=1)
                     & np.isfinite(old_raw).all(axis=1))
            actual = self.timeline.pv_kw.values[np.clip(idx, 0, len(self.timeline.pv_kw.values)-1)].reshape(24, 6).mean(axis=1)/6
            x = np.nan_to_num(old_raw.mean(axis=1))
            weight = 2**(-lag/7)
            numerator += np.where(valid, weight*x*actual, 0.)
            denominator += np.where(valid, weight*x*x, 0.)
            mass += weight*valid
            counts += valid
        # Two effective prior days shrink the per-hour multiplicative calibration
        # toward the supplied forecast. No cross-hour pooling hides morning bias.
        prior = 2*np.divide(denominator, mass, out=np.zeros(24), where=mass > 0)
        ratio = np.divide(numerator+prior, denominator+prior, out=np.ones(24), where=denominator+prior > 1e-8)
        ratio = np.where(counts >= 7, np.clip(ratio, 0., 2.), 1.)
        calibrated = shaped*np.repeat(ratio, 6)
        covered = np.isfinite(raw)
        return {'history':history,'shape':np.where(covered,shaped,history),
                'calibrated':np.where(covered,calibrated,history),
                'covered':covered,'counts':counts,'ratio':ratio}

    @lru_cache(maxsize=1024)
    def _report_weights(self, anchor):
        numerator, denominator = np.zeros(24), np.zeros(24)
        mass, counts = np.zeros(24), np.zeros(24,dtype=int)
        for lag in range(1, 15):
            old = anchor-lag*1440
            if old < 10:
                continue
            old_pred = self._report_components(old)
            idx = np.arange(old//10,old//10+144)
            valid = (self._valid_history(idx,self.timeline.pv_kw.values,anchor).reshape(24,6).all(axis=1)
                     & old_pred['covered'].reshape(24,6).all(axis=1))
            actual = self.timeline.pv_kw.values[np.clip(idx,0,len(self.timeline.pv_kw.values)-1)]/6
            delta = old_pred['calibrated']-old_pred['history']
            error = actual-old_pred['history']
            weight = 2**(-lag/7)
            numerator += np.where(valid,weight*(delta*error).reshape(24,6).sum(axis=1),0.)
            denominator += np.where(valid,weight*(delta*delta).reshape(24,6).sum(axis=1),0.)
            mass += weight*valid; counts += valid
        prior = 4*np.divide(denominator,mass,out=np.zeros(24),where=mass>0)
        weights = np.divide(numerator,denominator+prior,out=np.zeros(24),where=denominator+prior>1e-8)
        return np.where(counts>=7,np.clip(weights,0.,1.),0.)

    @lru_cache(maxsize=1024)
    def _node_pv(self, anchor):
        c = self._report_components(anchor)
        if self.variant != 'blend':
            return c[self.variant].copy()
        w = np.repeat(self._report_weights(anchor),6)
        return c['history']+w*(c['calibrated']-c['history'])

    @lru_cache(maxsize=3072)
    def _point(self, now, end, published):
        if not published:
            return super()._point(now,end,False)
        minutes = np.arange(now,end,10)
        load = self._load_prediction(now,minutes)
        pv = self._profile(now,minutes,'pv')
        anchor = now//360*360
        node = self._node_pv(anchor)
        offsets = (minutes-anchor)//10
        covered = (offsets>=0)&(offsets<144)
        pv[covered] = node[offsets[covered]]
        # Once an interval has finished, its innovation can correct the remaining
        # report. It never revises a forecast already issued at 00/06/12/18.
        if self.nowcast:
            recent = np.arange(max(anchor//10,now//10-6),now//10)
            valid = self._valid_history(recent,self.timeline.pv_kw.values,now)
            if valid.any():
                idx = recent[valid]
                residual = self.timeline.pv_kw.values[idx]/6-node[idx-anchor//10]
                bias = float(np.median(residual))
                # Do not carry a daylight innovation into zero-generation hours.
                pv += bias*np.exp(-(minutes-now)/120)*(pv>1e-8)
        return load,np.maximum(pv,0.)

    def window_forecast(self,*args,**kwargs):
        result = super().window_forecast(*args,**kwargs)
        if kwargs.get('use_published_pv',False):
            result.provenance = np.array([str(s)+f';report-v2={self.variant};nowcast={self.nowcast}'
                                          for s in result.provenance],dtype=object)
        return result
