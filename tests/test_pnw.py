"""PNW metadata parsing, event identification, selection and schema conversion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismic_ml.data import pnw
from seismic_ml.data.schemas import SeismicEvent
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore, parse_trace_name


@pytest.fixture
def comcat(toy_raw_dir: Path) -> pd.DataFrame:
    return pnw.load_metadata(toy_raw_dir / "comcat_metadata.csv")


@pytest.fixture
def store(toy_raw_dir: Path):
    with SeisBenchWaveformStore(toy_raw_dir / "comcat_waveforms.hdf5") as s:
        yield s


# --- trace addressing ----------------------------------------------------------------

def test_parse_trace_name() -> None:
    loc = parse_trace_name("bucket4$17,:3,:15001")
    assert (loc.bucket, loc.row, loc.n_channels, loc.n_samples) == ("bucket4", 17, 3, 15001)


@pytest.mark.parametrize("bad", ["bucket4-17", "bucket4$x,:3,:10", "$1,:3,:10", "bucket1$1,3,10"])
def test_parse_trace_name_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_trace_name(bad)


def test_store_reads_by_trace_name(store: SeisBenchWaveformStore) -> None:
    assert store.component_order == ("E", "N", "Z") and not store.is_subset
    x = store.get("bucket1$2,:3,:200")
    assert x.shape == (3, 200) and x.dtype == np.float64
    assert (x[:2] == 0).all() and not (x[2] == 0).all()      # vertical-only station
    assert "bucket1$99,:3,:200" not in store


# --- metadata parsing -------------------------------------------------------------------

def test_codes_stay_strings(comcat: pd.DataFrame) -> None:
    assert set(comcat["station_location_code"]) == {"--", "00"}   # "00" not parsed as 0
    assert comcat["event_id"].dtype == "string"


def test_station_identity_and_timestamps() -> None:
    assert pnw.station_id("UW", "GNW") == "UW.GNW"
    t = pnw.parse_utc("2002-10-03T01:55:59.530000Z")
    assert t == datetime(2002, 10, 3, 1, 55, 59, 530000, tzinfo=timezone.utc)


def test_parse_snr() -> None:
    assert pnw.parse_snr_db("6.1|3.0|11.7") == (6.1, 3.0, 11.7)
    snr = pnw.parse_snr_db("nan|nan|5.0")
    assert np.isnan(snr[0]) and snr[2] == 5.0
    assert pnw.parse_snr_db(float("nan")) is None and pnw.parse_snr_db(pd.NA) is None


def test_event_table(comcat: pd.DataFrame) -> None:
    table = pnw.event_table(comcat, "source_type_pnsn_label")
    assert table.loc["uw1", "n_traces"] == 3 and table.loc["uw4", "label"] == "lf"


def test_event_table_rejects_mixed_labels(comcat: pd.DataFrame) -> None:
    broken = comcat.copy()
    broken.loc[broken.index[0], "source_type_pnsn_label"] = "px"
    with pytest.raises(ValueError, match="several"):
        pnw.event_table(broken, "source_type_pnsn_label")


# --- selection -----------------------------------------------------------------------------

def test_select_events_is_seeded_and_event_complete(comcat: pd.DataFrame) -> None:
    kwargs = dict(label_column="source_type_pnsn_label", events_per_label={"eq": 1, "px": -1})
    first = pnw.select_events(comcat, seed=1, **kwargs)
    assert first == pnw.select_events(comcat, seed=1, **kwargs)      # deterministic
    assert "uw2" in first and len(first) == 2                        # px: all; eq: one of two
    assert {s for seed in range(20) for s in pnw.select_events(comcat, seed=seed, **kwargs)} \
        == {"uw1", "uw2", "uw3"}                                     # both eq events reachable


def test_select_events_label_streams_are_independent(comcat: pd.DataFrame) -> None:
    base = pnw.select_events(comcat, label_column="source_type_pnsn_label",
                             events_per_label={"eq": 1}, seed=3)
    more = pnw.select_events(comcat, label_column="source_type_pnsn_label",
                             events_per_label={"eq": 1, "lf": -1}, seed=3)
    assert set(base) <= set(more)


def test_select_events_unknown_label(comcat: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not found"):
        pnw.select_events(comcat, label_column="source_type_pnsn_label",
                          events_per_label={"volcano": 1}, seed=0)


# --- schema conversion ----------------------------------------------------------------------

def test_rows_to_event(comcat: pd.DataFrame, store: SeisBenchWaveformStore) -> None:
    rows = comcat[comcat["event_id"] == "uw1"]
    event = pnw.rows_to_event(rows, store, label_column="source_type_pnsn_label")
    assert isinstance(event, SeismicEvent)
    assert event.event_id == "uw1" and event.label == "eq"
    assert event.station_ids == ("CC.CCC", "UW.AAA", "UW.BBB")      # sorted, NET.STA
    assert event.origin_time == datetime(2020, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
    assert (event.latitude, event.depth_km, event.magnitude) == (47.0, 5.0, 1.5)
    assert event.metadata["source_type"] == "earthquake"

    obs = event.observations[1]                                      # UW.AAA
    assert obs.station.latitude == 47.1 and obs.station.network == "UW"
    wf = obs.waveform
    assert wf.channels == ("E", "N", "Z") and wf.sampling_rate == 100.0 and wf.num_samples == 200
    assert obs.p_arrival_time - wf.start_time == timedelta(seconds=0.6)
    assert obs.s_arrival_time - wf.start_time == timedelta(seconds=0.9)
    assert obs.metadata["station_channel_code"] == "HH"
    assert obs.metadata["trace_snr_db"] == (4.0, 3.0, 5.0)
    missing = event.observations[0]                                  # CC.CCC vertical only
    assert missing.metadata["trace_missing_channel"] == 2
    assert (missing.waveform.data[:2] == 0).all()


def test_rows_to_event_requires_single_event(comcat: pd.DataFrame,
                                             store: SeisBenchWaveformStore) -> None:
    with pytest.raises(ValueError):
        pnw.rows_to_event(comcat, store, label_column="source_type_pnsn_label")


def test_exotic_event_without_origin_or_s_pick(toy_raw_dir: Path) -> None:
    meta = pnw.load_metadata(toy_raw_dir / "exotic_metadata.csv")
    with SeisBenchWaveformStore(toy_raw_dir / "exotic_waveforms.hdf5") as store:
        event = pnw.rows_to_event(meta[meta["event_id"] == "pnsn2"], store,
                                  label_column="source_type")
    assert event.label == "sonic boom" and event.origin_time is None and event.magnitude is None
    assert all(o.s_arrival_time is None and o.p_arrival_time is not None
               for o in event.observations)


def test_iter_events_and_completeness(comcat: pd.DataFrame, store: SeisBenchWaveformStore) -> None:
    assert pnw.complete_event_ids(comcat, store) == ["uw1", "uw2", "uw3", "uw4"]
    events = list(pnw.iter_events(comcat, store, label_column="source_type_pnsn_label",
                                  event_ids=["uw4", "uw2"]))
    assert [e.event_id for e in events] == ["uw4", "uw2"]
    assert [e.num_stations for e in events] == [2, 1]
