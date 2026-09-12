from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from test_main_model import synthetic, forecast
from microgrid.adaptive_forecast import AdaptiveForecaster
from microgrid.timeline import build_timeline
from microgrid.planning import solve_window, FeeMode, clear_window_cache, window_cache_info
from microgrid.simulation import RunOptions, run_absolute
from microgrid.absolute_run import POLICIES


def test_adaptive_and_reserve_cannot_read_future_actuals_or_publications(synthetic):
    now, end = 40*1440+360, 42*1440
    tl = build_timeline(synthetic)
    f = AdaptiveForecaster(synthetic, tl)
    first = f.window_forecast(now, now, end, use_published_pv=True, current_price=None)
    risk = f.risk_requirements(now, first, published=True, adjustable=True)
    changed = deepcopy(synthetic)
    changed = replace(changed, attachment3=replace(changed.attachment3, blocks=tuple(
        replace(b, hourly_power_kw=np.full(24, 999999.))
        if (b.publish_day-changed.attachment2.days[0]).days*1440+b.publish_hour*60 > now else b
        for b in changed.attachment3.blocks)))
    tl2 = build_timeline(changed)
    for field in (tl2.load_kw, tl2.pv_kw, tl2.demand_kwh, tl2.price_yuan_per_kwh):
        field.values[now//10:] = 999999.
    g = AdaptiveForecaster(changed, tl2)
    second = g.window_forecast(now, now, end, use_published_pv=True, current_price=None)
    second_risk = g.risk_requirements(now, second, published=True, adjustable=True)
    for field in ('demand_kwh', 'pv_kwh', 'price_yuan_per_kwh'):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    for key in risk:
        np.testing.assert_array_equal(risk[key], second_risk[key])


def test_weekly_model_tracks_target_weekday_across_midnight(synthetic):
    synthetic.attachment2.load_kw[:] = (600+60*(np.arange(365)%7))[:, None]
    tl = build_timeline(synthetic)
    f = AdaptiveForecaster(synthetic, tl)
    now, end = 40*1440, 42*1440
    r = f.window_forecast(now, now, end, use_published_pv=False, current_price=None)
    expected = tl.load_kw.values[np.arange(now//10, end//10)-7*144]/6
    np.testing.assert_allclose(r.demand_kwh, expected)


def test_problem_two_adaptive_and_reserve_never_access_attachment_three(synthetic):
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError('attachment3 accessed')
    b = replace(synthetic, attachment3=Forbidden())
    f = AdaptiveForecaster(b, build_timeline(b))
    r = f.window_forecast(40*1440, 40*1440, 42*1440, use_published_pv=False, current_price=None)
    risk = f.risk_requirements(40*1440, r, published=False, adjustable=False)
    assert risk['risk_sample_count'] > 20


def test_cold_start_is_finite_and_reports_no_calibration(synthetic):
    f = AdaptiveForecaster(synthetic, build_timeline(synthetic))
    r = f.window_forecast(10, 10, 2880, use_published_pv=True, current_price=None)
    assert np.isfinite(r.demand_kwh).all() and np.isfinite(r.pv_kwh).all()
    risk = f.risk_requirements(10, r, published=True, adjustable=True)
    assert risk['risk_sample_count'] == 0
    assert not risk['reserve_energy_kwh'].any()


def test_reserve_causes_real_purchase_and_storage_not_fictitious_demand():
    f = forecast([1000])
    f.reserve_energy_kwh = np.array([0., 0.])
    f.net_upper_kwh = np.array([1200.])
    r = solve_window(forecast=f, soc_start_kwh=1200, fee_mode=FeeMode.FIRST_PLAN, allow_spill=True).require_ok()
    assert r.grid_kwh == pytest.approx([1200], abs=1e-4)
    assert r.charge_kwh == pytest.approx([180], abs=1e-4)
    assert r.discharge_kwh == pytest.approx([0], abs=1e-4)
    assert r.objective_yuan == pytest.approx(1200, abs=1e-4)
    assert r.reserve_shortfall_kwh == pytest.approx(0, abs=1e-4)


def test_impossible_reserve_is_reported_and_penalty_not_billed():
    f = forecast([0])
    f.reserve_energy_kwh = np.array([20000., 20000.])
    f.net_upper_kwh = np.zeros(1)
    r = solve_window(forecast=f, soc_start_kwh=10800, fee_mode=FeeMode.FIRST_PLAN, allow_spill=True).require_ok()
    assert r.reserve_shortfall_kwh >= 20000-8640-1e-5
    assert r.risk_penalty_yuan > 0
    assert r.objective_yuan == pytest.approx(0, abs=1e-5)


def test_full_battery_cannot_burn_extra_reserve_purchases_by_cycling():
    f = forecast([1000])
    f.reserve_energy_kwh, f.net_upper_kwh = np.zeros(2), np.array([2000.])
    r = solve_window(forecast=f, soc_start_kwh=10800, fee_mode=FeeMode.FIRST_PLAN, allow_spill=True).require_ok()
    assert r.grid_kwh[0] <= 1000+1e-5
    assert r.charge_kwh[0] == pytest.approx(0, abs=1e-4)
    assert r.reserve_shortfall_kwh >= 250-1e-4


def test_risk_does_not_rewrite_frozen_purchases():
    f = forecast([1000])
    f.reserve_energy_kwh = np.zeros(2)
    f.net_upper_kwh = np.array([1400.])
    r = solve_window(forecast=f, soc_start_kwh=1200, fee_mode=FeeMode.FROZEN,
        o_kwh=np.array([900.]), a_kwh=np.zeros(1), allow_spill=True).require_ok()
    assert r.grid_kwh == pytest.approx([900])
    assert r.emergency_kwh == pytest.approx([100], abs=1e-4)
    assert r.reserve_shortfall_kwh >= 500-1e-4
    assert r.objective_yuan == pytest.approx(1400, abs=1e-4)


def test_reserve_cache_reloads_values():
    clear_window_cache()
    f = forecast([1000])
    f.reserve_energy_kwh, f.net_upper_kwh = np.zeros(2), np.array([1100.])
    a = solve_window(forecast=f, soc_start_kwh=1200, fee_mode=FeeMode.FIRST_PLAN, allow_spill=True).require_ok()
    f.net_upper_kwh[:] = 1200
    b = solve_window(forecast=f, soc_start_kwh=1200, fee_mode=FeeMode.FIRST_PLAN, allow_spill=True).require_ok()
    assert b.grid_kwh[0]-a.grid_kwh[0] == pytest.approx(100, abs=1e-4)
    assert window_cache_info()['hit'] == 1


def test_main_cannot_restart_at_paired_evaluation_checkpoint(synthetic):
    tl = build_timeline(synthetic)
    with pytest.raises(ValueError):
        run_absolute(timeline=tl, forecaster=AdaptiveForecaster(synthetic, tl), policy=POLICIES['problem2'],
            options=RunOptions(evaluation_restart=True), abs_from=1440, abs_to=2880, soc_start_kwh=6000)
