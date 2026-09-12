"""Deterministic cumulative-error stress path and legal-node reduction check.

No future actuals or random scenario tree enter this model. The stress battery
is a feasibility copy sharing ordinary purchases, not a second physical device.
"""
import numpy as np
from functools import lru_cache
from .report_forecast import ReportAwareForecaster
from .planning import WindowForecast, FeeMode, solve_window


def cumulative_envelope(errors, minutes, demand, quantile, adjustable):
    """Upper envelope of prefix shortages, restarting only at legal nodes."""
    n=len(minutes);count=len(errors);energy=np.zeros(n+1);extra=np.zeros(n)
    rank=min(count,int(np.ceil((count+1)*quantile)))-1 if count else 0
    power=np.maximum(0,np.sort(errors,axis=0)[rank]) if count else np.zeros(n)
    block=360 if adjustable else 1440
    start=0
    while start<n:
        end=min(n,start+(block-int(minutes[start])%block)//10)
        if count:
            # A nondecreasing prefix envelope cannot release already used margin.
            paths=np.maximum.accumulate(np.maximum(0,np.cumsum(errors[:,start:end],axis=1)),axis=1)
            prefix=np.sort(paths,axis=0)[rank]
        else:
            prefix=np.zeros(end-start)
        if count<7:
            # Explicit small-sample technical prior, not inferred future data.
            prefix=np.maximum(prefix,.03*np.cumsum(demand[start:end]))
            power[start:end]=np.maximum(power[start:end],.03*demand[start:end])
        energy[start+1:end+1]=prefix
        extra[start:end]=np.diff(np.r_[0.,prefix])
        start=end
    return energy,extra,power


class CumulativeReportForecaster(ReportAwareForecaster):
    @lru_cache(maxsize=1024)
    def _errors(self,anchor,end,published):
        samples=[]
        for lag in range(1,self.history_days+1):
            old,stop=anchor-lag*1440,end-lag*1440
            # The weekly predictor changes regime after its first week. Do not
            # use untrained initialization forecasts as its mature error model.
            if old<7*1440+10 or stop>anchor:
                continue
            idx=np.arange(old//10,stop//10)
            if not (self._valid_history(idx,self.timeline.load_kw.values,anchor).all()
                    and self._valid_history(idx,self.timeline.pv_kw.values,anchor).all()):
                continue
            load,pv=self._point(old,stop,published)
            actual=(self.timeline.load_kw.values[idx]-self.timeline.pv_kw.values[idx])/6
            samples.append(actual-(load-pv))
        return np.array(samples).reshape(-1,(end-anchor)//10)

    def risk_requirements(self,now,forecast,*,published,adjustable):
        if not self.risk_quantile:
            return {}
        end=int(forecast.abs_minutes[-1])+10
        # Match the formation time and lead of the current point forecast.
        initialization=not adjustable and now<7*1440+10
        # A frozen first-week plan has no intra-day recovery opportunity.
        # Its own causal initialization errors are relevant until the weekly
        # predictor is established, then they are excluded from mature risk.
        errors=super()._errors(now,end,published) if initialization else self._errors(now,end,published)
        energy,extra,power=cumulative_envelope(errors,forecast.abs_minutes,
            forecast.demand_kwh,self.risk_quantile,adjustable)
        return dict(reserve_energy_kwh=energy,
                    net_upper_kwh=np.maximum(0,forecast.demand_kwh-forecast.pv_kwh+power),
                    stress_demand_kwh=forecast.demand_kwh+extra,
                    risk_sample_count=len(errors),reserve_policy='economic-cumulative-v3',
                    stress_weight=.5 if initialization else .2,
                    cold_start_margin=len(errors)<7)

    def window_forecast(self,*args,**kwargs):
        result=super().window_forecast(*args,**kwargs)
        result.provenance=np.array([str(x)+';reserve=economic-cumulative-v3' for x in result.provenance],dtype=object)
        return result


def check_reduction(forecast,soc,original,candidate):
    """Compare fixed-plan supply until the next legal node using current stress.

    Returns a restored candidate and evidence. Restoring only reduced deliveries
    preserves proposed additions. The caller re-solves physical/fee constraints.
    """
    now=int(forecast.abs_minutes[0]);stop=(now//360+1)*360
    n=min(forecast.n,(stop-now)//10)
    evidence=dict(checked=False,blocked=False,restored_kwh=0.,until_abs=stop)
    reduced=np.maximum(original[:n]-candidate[:n],0)
    if not np.any(reduced>1e-6):
        return candidate,evidence
    def minimum(grid):
        f=WindowForecast(forecast.abs_minutes[:n],forecast.stress_demand_kwh[:n],
                         forecast.pv_kwh[:n],np.ones(n),np.full(n,'legal-node stress check',dtype=object))
        r=solve_window(forecast=f,soc_start_kwh=soc,fee_mode=FeeMode.FROZEN,
            o_kwh=grid[:n],a_kwh=np.zeros(n),allow_spill=True).require_ok()
        return float(r.emergency_kwh.sum())
    before,after=minimum(original),minimum(candidate)
    evidence.update(checked=True,original_stress_emergency_kwh=before,candidate_stress_emergency_kwh=after)
    restored=candidate.copy()
    if after>before+1e-5:
        restored[:n]=np.maximum(original[:n],candidate[:n])
        repaired=minimum(restored)
        if repaired>before+1e-4:
            raise RuntimeError('Reduction guard did not restore the prior supply capability')
        evidence.update(blocked=True,restored_kwh=float(reduced.sum()),restored_stress_emergency_kwh=repaired)
    return restored,evidence
