"""Time-axis, absolute timeline and data-loading tests (2026-09-11 rules)."""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pytest

from microgrid import data_io, timeaxis as T
from microgrid.constants import DELTA_T_HOURS, N_INTERVAL


@pytest.fixture(scope="module")
def bundle() -> data_io.DataBundle:
    return data_io.load_all()


@pytest.fixture(scope="module")
def timeline(bundle):
    from microgrid.timeline import build_timeline

    return build_timeline(bundle)


# ---------------------------------------------------------------- source labels


def test_source_label_is_interval_start() -> None:
    """00:10 means [00:10, 00:20); the tail label means the next day's first 10 min."""
    assert T.seq_index_from_label("0:10") == 0
    assert T.seq_index_from_label("0:20") == 1
    assert T.seq_index_from_label("1:00") == 5
    assert T.seq_index_from_label("23:50") == 142
    assert T.seq_index_from_label("0:00+1") == 143
    assert T.seq_index_from_label("24:00") == 143
    assert T.seq_index_from_label(time(0, 10)) == 0


def test_labels_cover_all_144_intervals() -> None:
    labels = [T.seq_label(v) for v in range(T.N_SEQ)]
    assert labels[0] == "0:10"
    assert labels[142] == "23:50"
    assert labels[143] == "0:00+1"
    assert [T.seq_index_from_label(lb) for lb in labels] == list(range(144))


def test_midnight_label_is_rejected_as_source_label() -> None:
    """00:00 is a natural-day boundary, not a source label."""
    with pytest.raises(ValueError):
        T.seq_index_from_label("0:00")


def test_clock_labels() -> None:
    assert T.clock_label(0) == "0:00-0:10"
    assert T.clock_label(143) == "23:50-24:00"


# ---------------------------------------------------------------- result-row axis


def test_result_row_maps_to_clock_interval_plus_one() -> None:
    for j in range(144):
        assert T.result_interval_clock_index(j) == j + 1
        assert T.result_boundary_clock_index(j) == j + 1


def test_result_row_reaches_into_next_day() -> None:
    assert not T.result_row_spans_next_day(142)
    assert T.result_row_spans_next_day(143)


def test_template_column_is_identity() -> None:
    for j in range(144):
        assert T.template_column_of_result(j) == j
        assert T.result_of_template_column(j) == j


# ---------------------------------------------------------------- timeline bridge


def test_natural_day_first_interval_comes_from_previous_source_tail(
    bundle: data_io.DataBundle, timeline
) -> None:
    """D_{d,0} = source(d-1, last column); D_{d,t} = source(d, t-1)."""
    d1 = bundle.attachment2.days[1]
    clock = timeline.demand_clock_kwh(d1) * 6  # back to kW
    assert clock[0] == pytest.approx(bundle.attachment2.load_kw[0, 143])
    assert clock[1] == pytest.approx(bundle.attachment2.load_kw[1, 0])
    assert clock[143] == pytest.approx(bundle.attachment2.load_kw[1, 142])


def test_result_row_spans_next_day_first_interval(
    bundle: data_io.DataBundle, timeline
) -> None:
    """A result row is [day 00:10, next day 00:10): first cell = clock t=1,
    last cell = next day's clock t=0."""
    d1 = bundle.attachment2.days[1]
    d2 = bundle.attachment2.days[2]
    row = timeline.result_demand_kwh(d1) * 6
    assert row[0] == pytest.approx(timeline.demand_clock_kwh(d1)[1] * 6)
    assert row[143] == pytest.approx(timeline.demand_clock_kwh(d2)[0] * 6)


def test_first_bridge_is_missing_and_estimated(timeline) -> None:
    """2025-01-01 00:00-00:10 has no previous source row; it must be flagged."""
    from microgrid.timeaxis import SOURCE_BRIDGE_MISSING

    prov = timeline.demand_kwh.provenance
    assert prov[0] == SOURCE_BRIDGE_MISSING
    assert int(np.sum(prov == SOURCE_BRIDGE_MISSING)) == 1


def test_carry_over_provenance_count(timeline) -> None:
    from microgrid.timeaxis import SOURCE_CARRY_OVER

    n_carry = int(np.sum(timeline.demand_kwh.provenance == SOURCE_CARRY_OVER))
    assert n_carry == 364  # one per natural day except the very first


def test_year_end_extension_is_marked(timeline) -> None:
    from microgrid.timeaxis import SOURCE_EXTENDED

    n_ext = int(np.sum(timeline.demand_kwh.provenance == SOURCE_EXTENDED))
    assert n_ext == 288  # 48 hours of ten-minute intervals
    assert n_ext == timeline.demand_kwh.values.size - 365 * 144


def test_clock_and_result_windows_differ_by_one_interval(timeline) -> None:
    d = date(2025, 6, 15)
    c_lo, c_hi = timeline.clock_day_slice(d)
    r_lo, r_hi = timeline.result_row_slice(d)
    assert r_lo == c_lo + 10
    assert r_hi == c_hi + 10


# ---------------------------------------------------------------- data loading


def test_attachment_shapes(bundle: data_io.DataBundle) -> None:
    assert bundle.attachment1.price_yuan_per_kwh.shape == (N_INTERVAL,)
    assert bundle.attachment2.load_kw.shape == (365, N_INTERVAL)
    assert bundle.attachment3.block_at(date(2025, 1, 1), 0).hourly_power_kw.shape == (24,)
    assert bundle.attachment4.price_yuan_per_kwh.shape == (365, N_INTERVAL)


def test_attachment3_24h_blocks_are_self_contained(bundle: data_io.DataBundle) -> None:
    """The 00:00 release covers [day 00:00, day 24:00) by itself, no splicing."""
    hourly = bundle.attachment3.hourly_power_on(date(2025, 6, 21))
    assert hourly.shape == (24,)
    assert hourly.max() > 100.0


def test_attachment3_latest_published_never_reads_the_future(
    bundle: data_io.DataBundle,
) -> None:
    """A version published after the request moment must never be returned."""
    a3 = bundle.attachment3
    target = date(2025, 1, 1).toordinal() * 24 + 7  # 07:00 on 1 Jan
    found = a3.latest_published_covering(date(2025, 1, 1), 0, target)
    assert found is not None
    block, offset = found
    assert (block.publish_day, block.publish_hour) == (date(2025, 1, 1), 0)
    assert offset == 7
    # Asking at 00:00 cannot see the 06:00 release for an hour it did not cover
    later = a3.latest_published_covering(date(2025, 1, 1), 0, date(2025, 1, 2).toordinal() * 24 + 1)
    assert later is None


def test_unit_conversion(bundle: data_io.DataBundle) -> None:
    a1 = bundle.attachment1
    assert np.allclose(a1.demand_kwh, a1.load_kw * DELTA_T_HOURS)
    assert np.allclose(a1.pv_kwh, a1.pv_forecast_kw * DELTA_T_HOURS)


def test_statistics_still_match_the_documented_values(bundle: data_io.DataBundle) -> None:
    a1 = bundle.attachment1
    assert a1.price_yuan_per_kwh.min() == pytest.approx(0.3713, abs=5e-5)
    assert a1.load_kw.mean() == pytest.approx(4626.0, abs=0.05)
    a4 = bundle.attachment4
    assert a4.price_yuan_per_kwh.min() > 0.0


def test_output_window_is_334_days() -> None:
    days = T.calendar_days(date(2025, 2, 1), date(2025, 12, 31))
    assert len(days) == 334
