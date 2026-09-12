from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from test_main_model import synthetic
from microgrid.timeline import build_timeline
from microgrid.adaptive_forecast import AdaptiveForecaster
from microgrid.report_forecast import ReportAwareForecaster
from microgrid.absolute_run import RunConfig, make_forecaster
from microgrid.cumulative_risk import CumulativeReportForecaster


def test_hourly_disaggregation_preserves_every_report_hour():
    raw=np.repeat([100.,20.,0.],6)
    history=np.r_[np.arange(1.,7.),np.zeros(6),np.arange(6.)]
    shaped=ReportAwareForecaster.distribute_hourly(raw,history)
    np.testing.assert_allclose(shaped.reshape(-1,6).sum(axis=1),raw.reshape(-1,6).sum(axis=1))
    assert np.all(shaped>=0)


@pytest.mark.parametrize('forecaster_class',[ReportAwareForecaster,CumulativeReportForecaster])
def test_future_actuals_and_unreleased_reports_cannot_change_forecast(synthetic,forecaster_class):
    now,end=40*1440+550,42*1440
    tl=build_timeline(synthetic)
    a=forecaster_class(synthetic,tl)
    fa=a.window_forecast(now,now,end,use_published_pv=True,current_price=None,price_mode='repeated')
    ra=a.risk_requirements(now,fa,published=True,adjustable=True)
    changed=deepcopy(synthetic)
    changed=replace(changed,attachment3=replace(changed.attachment3,blocks=tuple(
        replace(b,hourly_power_kw=np.full(24,999999.))
        if (b.publish_day-changed.attachment2.days[0]).days*1440+b.publish_hour*60>now else b
        for b in changed.attachment3.blocks)))
    tl2=build_timeline(changed)
    for field in (tl2.load_kw,tl2.pv_kw,tl2.demand_kwh,tl2.price_yuan_per_kwh):
        field.values[now//10:]=999999.
    b=forecaster_class(changed,tl2)
    fb=b.window_forecast(now,now,end,use_published_pv=True,current_price=None,price_mode='repeated')
    rb=b.risk_requirements(now,fb,published=True,adjustable=True)
    for field in ('demand_kwh','pv_kwh','price_yuan_per_kwh'):
        np.testing.assert_array_equal(getattr(fa,field),getattr(fb,field))
    for key in ra:
        np.testing.assert_array_equal(ra[key],rb[key])
    assert 'report-v2' in fa.provenance[0]


def test_recent_observations_update_rolling_but_not_already_issued_node(synthetic):
    anchor=40*1440+360;now=anchor+180
    tl=build_timeline(synthetic);tl.pv_kw.values[:]=600.
    a=ReportAwareForecaster(synthetic,tl)
    node=a._point(anchor,anchor+360,True)[1]
    roll=a._point(now,now+60,True)[1]
    tl2=deepcopy(tl);tl2.pv_kw.values[now//10-6:now//10]+=600
    b=ReportAwareForecaster(synthetic,tl2)
    np.testing.assert_array_equal(node,b._point(anchor,anchor+360,True)[1])
    assert np.max(abs(roll-b._point(now,now+60,True)[1]))>1


def test_no_attachment_three_when_permission_is_false(synthetic):
    class Forbidden:
        def __getattr__(self,key):raise AssertionError('attachment3 accessed')
    b=replace(synthetic,attachment3=Forbidden())
    tl=build_timeline(b);new=ReportAwareForecaster(b,tl);old=AdaptiveForecaster(b,tl)
    for x,y in zip(new._point(40*1440,42*1440,False),old._point(40*1440,42*1440,False)):
        np.testing.assert_array_equal(x,y)


def test_cold_start_midnight_and_long_horizon_are_finite(synthetic):
    f=ReportAwareForecaster(synthetic,build_timeline(synthetic))
    for now in (10,360,40*1440+1430):
        end=(now//1440+2)*1440
        r=f.window_forecast(now,now,end,use_published_pv=True,current_price=None,price_mode='repeated')
        assert len(r.pv_kwh)==(end-now)//10
        assert np.isfinite(r.pv_kwh).all() and (r.pv_kwh>=0).all()


def test_positive_daylight_innovation_does_not_create_night_pv(synthetic):
    tl=build_timeline(synthetic);f=ReportAwareForecaster(synthetic,tl)
    anchor=40*1440+720;now=anchor+60
    node=np.r_[np.full(36,100.),np.zeros(108)]
    f._node_pv=lambda at:node.copy()
    tl.pv_kw.values[anchor//10:now//10]=1200.
    _,pv=f._point(now,anchor+1440,True)
    np.testing.assert_array_equal(pv[30:],0.)


def test_dispatch_routes_problem_three_and_four_to_optimized_pv(synthetic):
    tl=build_timeline(synthetic)
    assert isinstance(make_forecaster(RunConfig(problem='problem3'),synthetic,tl),ReportAwareForecaster)
    assert type(make_forecaster(RunConfig(problem='problem2'),synthetic,tl)) is AdaptiveForecaster
    for problem in ('problem4-2','problem4-3'):
        assert isinstance(make_forecaster(RunConfig(problem=problem),synthetic,tl),CumulativeReportForecaster)
        assert type(make_forecaster(RunConfig(problem=problem,pv_method='pooled'),synthetic,tl)) is AdaptiveForecaster
    assert type(make_forecaster(RunConfig(problem='problem3',pv_method='pooled'),synthetic,tl)) is AdaptiveForecaster


def test_cli_preserves_explicit_forecast_choice():
    from microgrid.cli import build_parser
    from microgrid.absolute_run import describe_model
    args=build_parser().parse_args(['run','--problem','problem4-3','--pv-method','pooled'])
    contract=describe_model(RunConfig(problem=args.problem,pv_method=args.pv_method))
    assert contract['pv_method']=='pooled' and contract['price_mode']=='historical'
    assert 'Historical reproduction' in contract['notice']
    assert describe_model(RunConfig(problem='problem3'))['forecast_class']=='ReportAwareForecaster'
    assert describe_model(RunConfig(problem='problem1'))['forecast_class']=='GivenDay'
    assert describe_model(RunConfig(problem='problem1'))['risk_quantile']==0


def test_economic_risk_excludes_untrained_week_but_keeps_mature_paths(synthetic):
    f=CumulativeReportForecaster(synthetic,build_timeline(synthetic))
    assert len(f._errors(8*1440+10,10*1440+10,True))==0
    assert len(f._errors(9*1440+10,11*1440+10,True))==1


def test_only_frozen_initialization_uses_cold_error_paths(synthetic):
    f=CumulativeReportForecaster(synthetic,build_timeline(synthetic))
    for now,adjustable,weight in [(5*1440,False,.5),(5*1440,True,.2),(8*1440,False,.2)]:
        window=f.window_forecast(now,now,(now//1440+2)*1440,use_published_pv=True,current_price=None,price_mode='historical')
        risk=f.risk_requirements(now,window,published=True,adjustable=adjustable)
        assert risk['stress_weight']==weight
        if now==5*1440:assert (risk['risk_sample_count']>0)==(not adjustable)
