"""数据层测试：结构、时间映射、单位换算、日期覆盖。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from microgrid import data_io, timeaxis
from microgrid.constants import (
    DELTA_T_HOURS,
    N_BOUNDARY,
    N_INTERVAL,
    OUTPUT_START,
)


@pytest.fixture(scope="module")
def bundle() -> data_io.DataBundle:
    return data_io.load_all()


# ---------------------------------------------------------------- 时间映射


def test_label_is_interval_end() -> None:
    assert timeaxis.interval_index_from_label("0:10") == 0
    assert timeaxis.interval_index_from_label("0:20") == 1
    assert timeaxis.interval_index_from_label("1:00") == 5
    assert timeaxis.interval_index_from_label("23:50") == 142
    assert timeaxis.interval_index_from_label("0:00+1") == 143


def test_labels_round_trip() -> None:
    labels = timeaxis.interval_labels()
    assert len(labels) == N_INTERVAL
    assert labels[0] == "0:10"
    assert labels[143] == "0:00+1"
    assert [timeaxis.interval_index_from_label(x) for x in labels] == list(range(N_INTERVAL))


def test_boundary_labels() -> None:
    labels = timeaxis.boundary_labels()
    assert len(labels) == N_BOUNDARY
    assert labels[0] == "0:00"
    assert labels[1] == "0:10"
    assert labels[-1] == "0:00+1"


def test_boundary_index() -> None:
    assert timeaxis.boundary_index(0, "start") == 0
    assert timeaxis.boundary_index(0, "end") == 1
    assert timeaxis.boundary_index(143, "end") == 144


def test_invalid_labels_rejected() -> None:
    with pytest.raises(ValueError):
        timeaxis.interval_index_from_label("0:05")
    with pytest.raises(ValueError):
        timeaxis.interval_index_from_label("24:00")
    with pytest.raises(ValueError):
        timeaxis.interval_index_from_label("垃圾")


def test_template_column_mapping_is_identity() -> None:
    """已确认口径：模板标签一字不动，模型第 t 段写第 t+1 个数据格。"""
    for t in range(N_INTERVAL):
        assert timeaxis.template_column_of_interval(t) == t
        assert timeaxis.interval_of_template_column(t) == t


# ---------------------------------------------------------------- 附件1


def test_attachment1_shape_and_units(bundle: data_io.DataBundle) -> None:
    a1 = bundle.attachment1
    for arr in (a1.price_yuan_per_kwh, a1.load_kw, a1.pv_forecast_kw):
        assert arr.shape == (N_INTERVAL,)
        assert arr.dtype == np.float64
    assert a1.price_yuan_per_kwh.min() > 0.0
    assert (a1.load_kw > 0).all()
    assert (a1.pv_forecast_kw >= 0).all()
    # 电量换算 = 功率 / 6
    assert np.allclose(a1.demand_kwh, a1.load_kw * DELTA_T_HOURS)
    assert np.allclose(a1.pv_kwh, a1.pv_forecast_kw * DELTA_T_HOURS)


def test_attachment1_matches_documented_stats(bundle: data_io.DataBundle) -> None:
    """只读核验得到的统计量，防止读取错位。"""
    a1 = bundle.attachment1
    assert a1.price_yuan_per_kwh.min() == pytest.approx(0.3713, abs=5e-5)
    assert a1.price_yuan_per_kwh.max() == pytest.approx(1.3952, abs=5e-5)
    assert a1.load_kw.mean() == pytest.approx(4626.0, abs=0.05)
    assert a1.pv_forecast_kw.max() == pytest.approx(7612.3, abs=0.05)


# ---------------------------------------------------------------- 附件2


def test_attachment2_calendar_and_shape(bundle: data_io.DataBundle) -> None:
    a2 = bundle.attachment2
    assert len(a2.days) == 365
    assert a2.days[0] == date(2025, 1, 1)
    assert a2.days[-1] == date(2025, 12, 31)
    assert a2.load_kw.shape == (365, N_INTERVAL)
    assert a2.pv_actual_kw.shape == (365, N_INTERVAL)
    assert not np.isnan(a2.load_kw).any()
    assert not np.isnan(a2.pv_actual_kw).any()


def test_attachment2_mean_matches_attachment1(bundle: data_io.DataBundle) -> None:
    """附件1 是典型日：其负载均值应等于附件2 全年均值。"""
    a1_mean = bundle.attachment1.load_kw.mean()
    a2_mean = bundle.attachment2.load_kw.mean()
    assert a1_mean == pytest.approx(a2_mean, abs=0.05)


# ---------------------------------------------------------------- 附件3


def test_attachment3_blocks(bundle: data_io.DataBundle) -> None:
    a3 = bundle.attachment3
    assert len(a3.blocks) == 1460
    days = {b.publish_day for b in a3.blocks}
    assert len(days) == 365
    for b in a3.blocks:
        assert b.publish_hour in (0, 6, 12, 18)
        assert b.hourly_power_kw.shape == (24,)
        assert b.hourly_power_kw.min() >= 0.0


def test_attachment3_hourly_profile_crosses_midnight(bundle: data_io.DataBundle) -> None:
    """18:00 的发布覆盖次日 0:00—18:00，跨午夜有效日期必须保留。

    选夏季日期并用白天小时，避免冬季夜间全零掩盖错位。
    注意 18:00 那条与次日 0:00 那条是**两次不同的发布**，
    前者覆盖 [次日 0:00, 次日 18:00)，后者覆盖 [次日 0:00, 次日 24:00)，
    两者数值不同正是"预报随时点更新"的体现，不能要求相等。
    """
    a3 = bundle.attachment3
    prev = a3.block_at(date(2025, 6, 21), 18).hourly_power_kw
    # 第 1..6 小时落在 6/21 晚上，基本无出力；第 7..24 小时是 6/22 的 0:00—18:00
    assert prev[:6].max() < 10.0
    assert prev[6:24].max() > 100.0
    # 次日 0:00 的发布完整覆盖当天 24 小时，其前 18 小时应与前一日 18:00 的发布同段对齐
    nxt = a3.hourly_power_on(date(2025, 6, 22))
    assert nxt.shape == (24,)
    assert np.allclose(nxt[6:18], a3.block_at(date(2025, 6, 22), 0).hourly_power_kw[6:18])
    assert nxt[12] > 100.0


def test_attachment3_profiles_are_close_to_actual(bundle: data_io.DataBundle) -> None:
    """小时平均预报应与同日实际光伏在同一量级（仅作结构合理性检查）。"""
    a3, a2 = bundle.attachment3, bundle.attachment2
    day = date(2025, 6, 21)
    idx = a2.index_of(day)
    actual_hourly = a2.pv_actual_kw[idx].reshape(24, 6).mean(axis=1)
    forecast = a3.hourly_power_on(day)
    day_total_f = forecast.sum()
    day_total_a = actual_hourly.sum()
    assert 0.5 < day_total_f / day_total_a < 2.0


def test_attachment3_noon_block_is_consistent_with_actual(bundle: data_io.DataBundle) -> None:
    """12:00 发布的"预报1小时"应贴近当天 12:00—13:00 的实际光伏。"""
    a3 = bundle.attachment3
    a2 = bundle.attachment2
    day = date(2025, 6, 21)
    block = a3.block_at(day, 12)
    idx = a2.index_of(day)
    actual_hour = a2.pv_actual_kw[idx, 72:78].mean()  # 12:00—13:00 的 6 段
    assert block.hourly_power_kw[0] == pytest.approx(actual_hour, rel=0.35)


# ---------------------------------------------------------------- 附件4


def test_attachment4_calendar_and_prices(bundle: data_io.DataBundle) -> None:
    a4 = bundle.attachment4
    assert a4.days == bundle.attachment2.days
    assert a4.price_yuan_per_kwh.shape == (365, N_INTERVAL)
    assert a4.price_yuan_per_kwh.min() > 0.0
    assert a4.price_yuan_per_kwh.max() == pytest.approx(1.7936, abs=5e-5)


def test_attachment4_mean_matches_attachment1(bundle: data_io.DataBundle) -> None:
    assert bundle.attachment1.price_yuan_per_kwh.mean() == pytest.approx(
        bundle.attachment4.price_yuan_per_kwh.mean(), abs=5e-5
    )


# ---------------------------------------------------------------- 其它


def test_lock_files_are_rejected() -> None:
    with pytest.raises(data_io.DataError):
        data_io.data_path("data/~$附件1.xlsx")


def test_input_fingerprint_is_stable() -> None:
    root = data_io.project_root()
    paths = [data_io.data_path(p, root) for p in ("data/附件1.xlsx", "data/附件2.xlsx")]
    assert data_io.input_fingerprint(paths) == data_io.input_fingerprint(paths)


def test_output_window_is_334_days() -> None:
    days = timeaxis.calendar_days(date(*OUTPUT_START), date(2025, 12, 31))
    assert len(days) == 334
    assert days[0] == date(2025, 2, 1)
    assert days[-1] == date(2025, 12, 31)
