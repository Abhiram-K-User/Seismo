"""S1 temporal split logic on small hand-made metadata tables (no real data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seismic_ml.data.splits import (
    assign_roles,
    event_table,
    load_split_table,
    split_summary,
    temporal_split,
)

CLASS_MAP = {"eq": "earthquake", "px": "anthropogenic"}


def _rows(event_id: str, label: str, source_type: str, time: str, lat: float = 47.0,
          lon: float = -122.0, n: int = 1) -> list[dict]:
    return [{"event_id": event_id, "source_type_pnsn_label": label, "source_type": source_type,
             "source_origin_time": time, "source_latitude_deg": lat, "source_longitude_deg": lon,
             "trace_name": f"b${event_id}{i},:3,:10", "trace_start_time": time}
            for i in range(n)]


@pytest.fixture
def comcat() -> pd.DataFrame:
    rows = []
    rows += _rows("tr_eq", "eq", "earthquake", "2015-06-01T00:00:00Z", n=3)
    rows += _rows("tr_px", "px", "explosion", "2016-06-01T00:00:00Z", lat=46.0, lon=-121.0)
    rows += _rows("buf", "eq", "earthquake", "2018-12-28T00:00:00Z")          # < 7 d before train_end
    rows += _rows("va_eq", "eq", "earthquake", "2019-06-01T00:00:00Z")
    rows += _rows("te_px_seen", "px", "explosion", "2021-06-01T00:00:00Z", lat=46.0, lon=-121.0)
    rows += _rows("te_px_new", "px", "explosion", "2022-06-01T00:00:00Z", lat=44.0, lon=-118.0)
    rows += _rows("conf_tr", "px", "earthquake", "2014-01-01T00:00:00Z")      # conflict, train
    rows += _rows("conf_te", "eq", "explosion", "2022-01-05T00:00:00Z")       # conflict, test
    rows += _rows("ood_lf", "lf", "earthquake", "2017-01-01T00:00:00Z")
    rows += _rows("unk", "uk", "earthquake", "2017-02-01T00:00:00Z")
    rows += _rows("dup_a", "eq", "earthquake", "2020-12-20T00:00:00Z", lat=45.0)   # val
    rows += _rows("dup_b", "eq", "earthquake", "2020-12-20T00:00:01Z", lat=45.0)   # near-identical
    return pd.DataFrame(rows)


@pytest.fixture
def exotic() -> pd.DataFrame:
    return pd.DataFrame([{"event_id": "pnsn1", "source_type": "surface event",
                          "trace_name": "b$x,:3,:10", "trace_start_time": "2012-03-01T00:00:00Z"}])


def _split(comcat: pd.DataFrame, exotic: pd.DataFrame, **overrides) -> pd.DataFrame:
    kwargs = dict(label_columns={"comcat": "source_type_pnsn_label", "exotic": "source_type"},
                  class_map=CLASS_MAP, ood_labels={"comcat": ["lf"], "exotic": ["surface event"]},
                  excluded_labels={"comcat": ["uk"]}, train_end="2019-01-01", val_end="2021-01-01",
                  buffer_days=7, site_cell_deg=0.1)
    kwargs.update(overrides)
    return temporal_split({"comcat": comcat, "exotic": exotic}, **kwargs).set_index("event_id")


def test_event_table_one_row_per_event(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    events = event_table(comcat, collection="comcat", label_column="source_type_pnsn_label")
    assert len(events) == comcat["event_id"].nunique()
    assert events.set_index("event_id").loc["tr_eq", "n_stations"] == 3
    ex = event_table(exotic, collection="exotic", label_column="source_type").iloc[0]
    assert pd.isna(ex["origin_time"]) and ex["year"] == 2012        # falls back to trace start


def test_unmapped_labels_raise(comcat: pd.DataFrame) -> None:
    events = event_table(comcat, collection="comcat", label_column="source_type_pnsn_label")
    with pytest.raises(ValueError, match="without a configured role"):
        assign_roles(events, class_map=CLASS_MAP, ood_labels=[], excluded_labels=[])


def test_temporal_assignment(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    t = _split(comcat, exotic)
    assert t.loc["tr_eq", "split"] == "train" and t.loc["tr_eq", "class"] == "earthquake"
    assert t.loc["tr_px", "split"] == "train" and t.loc["tr_px", "class"] == "anthropogenic"
    assert t.loc["va_eq", "split"] == "val"
    assert t.loc["te_px_seen", "split"] == "test" and t.loc["te_px_new", "split"] == "test"


def test_exclusions_have_reasons(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    t = _split(comcat, exotic)
    assert tuple(t.loc["buf", ["split", "reason"]]) == ("excluded", "boundary_buffer")
    assert tuple(t.loc["unk", ["split", "reason"]]) == ("excluded", "excluded_label")
    assert tuple(t.loc["conf_te", ["split", "reason"]]) == ("excluded", "label_conflict")
    assert t.loc["conf_tr", "split"] == "train" and bool(t.loc["conf_tr", "label_conflict"])
    assert t.loc["conf_tr", "class"] == "anthropogenic"             # PNSN label is authoritative
    assert (t.loc[t["split"] == "excluded", "reason"] != "").all()


def test_ood_events_never_supervised(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    t = _split(comcat, exotic)
    assert t.loc["ood_lf", "split"] == "ood" and pd.isna(t.loc["ood_lf", "class"])
    assert t.loc["pnsn1", "split"] == "ood" and t.loc["pnsn1", "collection"] == "exotic"


def test_duplicates_are_kept_together(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    assert _split(comcat, exotic).loc[["dup_a", "dup_b"], "split"].tolist() == ["val", "val"]
    moved = comcat.copy()
    moved.loc[moved["event_id"] == "dup_b", "source_origin_time"] = "2021-01-01T00:00:00.5Z"
    moved.loc[moved["event_id"] == "dup_a", "source_origin_time"] = "2020-12-31T23:59:59Z"
    t = _split(moved, exotic, buffer_days=0)                        # pair straddles val/test
    assert set(t.loc[["dup_a", "dup_b"], "reason"]) == {"duplicate_straddles_split"}


def test_site_seen_in_train(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    t = _split(comcat, exotic)
    assert t.loc["te_px_seen", "site_seen_in_train"] is True or t.loc["te_px_seen", "site_seen_in_train"] == True  # noqa: E712
    assert t.loc["te_px_new", "site_seen_in_train"] == False  # noqa: E712
    assert pd.isna(t.loc["tr_eq", "site_seen_in_train"])            # only defined for val/test


def test_invalid_boundaries(comcat: pd.DataFrame, exotic: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        _split(comcat, exotic, train_end="2021-01-01", val_end="2019-01-01")


def test_summary_and_roundtrip(comcat: pd.DataFrame, exotic: pd.DataFrame, tmp_path: Path) -> None:
    table = _split(comcat, exotic).reset_index()
    summary = split_summary(table)
    assert summary["events_total"] == len(table)
    assert summary["supervised"]["train"] == {"anthropogenic": 2, "earthquake": 1}
    assert summary["excluded"] == {"boundary_buffer": 1, "excluded_label": 1, "label_conflict": 1}
    assert summary["test_sites_seen_in_train"]["anthropogenic"] == {"seen": 1, "new": 1}
    path = tmp_path / "events.csv"
    table.to_csv(path, index=False)
    back = load_split_table(path).set_index("event_id")
    assert back.loc["te_px_new", "site_seen_in_train"] is False
    assert back.loc["conf_tr", "label_conflict"] and back.loc["tr_eq", "reason"] == ""
    assert str(back["reference_time"].dt.tz) == "UTC"
