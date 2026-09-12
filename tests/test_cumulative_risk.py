import numpy as np
from microgrid.cumulative_risk import cumulative_envelope, check_reduction
from microgrid.planning import WindowForecast, FeeMode, solve_window


def forecast(demand, stress, start=360):
    d=np.asarray(demand,dtype=float);s=np.asarray(stress,dtype=float);n=len(d)
    return WindowForecast(np.arange(start,start+10*n,10),d,np.zeros(n),np.ones(n),
        np.full(n,'test',dtype=object),reserve_energy_kwh=np.r_[0,np.cumsum(s-d)],
        net_upper_kwh=s,stress_demand_kwh=s)


def test_prefix_errors_cannot_release_already_used_margin():
    errors=np.tile([100.,-80.,100.,-120.],(10,1))
    energy,extra,power=cumulative_envelope(errors,np.arange(360,400,10),np.ones(4)*100,.9,True)
    np.testing.assert_allclose(energy,[0,100,100,120,120])
    np.testing.assert_allclose(extra,[100,0,20,0])


def test_margin_restarts_at_legal_node_and_cold_start_is_explicit():
    energy,extra,_=cumulative_envelope(np.tile([100.,100.],(10,1)),np.array([350,360]),np.ones(2)*100,.9,True)
    np.testing.assert_allclose(extra,[100,100])
    _,extra,power=cumulative_envelope(np.empty((0,2)),np.array([350,360]),np.array([200.,300.]),.9,True)
    np.testing.assert_allclose(extra,[6,9]);np.testing.assert_allclose(power,[6,9])


def test_stress_trajectory_buys_energy_and_respects_capacity():
    f=forecast([100.]*12,[200.]*12)
    f.stress_weight=.4
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FIRST_PLAN,allow_spill=True).require_ok()
    assert r.grid_kwh.sum()>=2400-1e-3
    assert r.stress_emergency_kwh.sum()<1e-3
    assert min(r.stress_soc_boundary_kwh)>=1200-1e-5
    assert max(r.stress_soc_boundary_kwh)<=10800+1e-5
    # With no contracted supply, the stress copy cannot invent stored energy.
    z=np.zeros(f.n)
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FROZEN,o_kwh=z,a_kwh=z,allow_spill=True).require_ok()
    assert abs(r.stress_emergency_kwh.sum()-2400)<1e-3


def test_weighted_risk_reduces_cost_without_phantom_nominal_emergency():
    f=forecast([100.]*12,[200.]*12)
    f.stress_weight=.1
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FIRST_PLAN,allow_spill=True).require_ok()
    assert abs(r.grid_kwh.sum()-1200)<1e-3
    assert r.emergency_kwh.sum()<1e-4
    assert abs(r.objective_yuan-1200)<1e-3  # Actual forecast bill excludes risk adjustment.
    # Same cached shape must reload weights and retain a positive nominal cost.
    f.stress_weight=.4
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FIRST_PLAN,allow_spill=True).require_ok()
    assert r.grid_kwh.sum()>2399.99
    z=np.zeros(f.n)
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FROZEN,o_kwh=z,a_kwh=z,allow_spill=True).require_ok()
    assert abs(r.emergency_kwh.sum()-1200)<1e-3
    assert abs(r.stress_emergency_kwh.sum()-2400)<1e-3


def test_cumulative_path_does_not_double_penalize_pointwise_margin():
    f=forecast([100.]*6,[100.]*6);f.net_upper_kwh[:]=5000
    r=solve_window(forecast=f,soc_start_kwh=1200,fee_mode=FeeMode.FIRST_PLAN,allow_spill=True).require_ok()
    assert abs(r.grid_kwh.sum()-600)<1e-3
    assert r.reserve_power_shortfall_kwh.sum()==0


def test_reduction_guard_restores_harmful_reductions_and_preserves_additions():
    f=forecast([100.]*6,[200.]*6)
    original=np.full(6,200.);candidate=np.array([250.,0,0,0,0,0])
    checked,e=check_reduction(f,1200,original,candidate)
    assert e['checked'] and e['blocked']
    np.testing.assert_allclose(checked,[250.,200,200,200,200,200])
    assert e['restored_stress_emergency_kwh']<1e-4


def test_reduction_guard_permits_safe_reduction_and_stops_at_next_node():
    f=forecast([100.]*12,[100.]*12,start=660)
    original=np.full(12,100.);candidate=np.zeros(12)
    checked,e=check_reduction(f,6000,original,candidate)
    assert e['checked'] and not e['blocked'] and e['until_abs']==720
    np.testing.assert_array_equal(checked,candidate)
    checked,e=check_reduction(f,1200,original,candidate)
    assert e['blocked']
    np.testing.assert_allclose(checked[:6],100)
    np.testing.assert_allclose(checked[6:],0)
