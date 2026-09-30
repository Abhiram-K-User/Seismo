"""Shared fixtures.

Fixtures build tiny schema objects and a toy PNW-layout dataset purely to
exercise code paths. They are NOT data and must never be used to produce results.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

import pnw_fixture
from seismic_ml.config import load_config
from seismic_ml.data.schemas import SeismicEvent, Station, StationObservation, WaveformSample

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config_path() -> Path:
    return REPO_ROOT / "configs" / "config.yaml"


@pytest.fixture
def config(config_path: Path) -> dict:
    return load_config(config_path)


@pytest.fixture
def data_config(config: dict) -> dict:
    return config["data"]


@pytest.fixture
def station() -> Station:
    return Station(station_id="XX.STA1", latitude=10.0, longitude=20.0)


@pytest.fixture
def waveform() -> WaveformSample:
    return WaveformSample(
        data=np.zeros((3, 200), dtype=np.float32),
        sampling_rate=100.0,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def event(station: Station, waveform: WaveformSample) -> SeismicEvent:
    other = Station(station_id="XX.STA2", latitude=10.5, longitude=20.5)
    return SeismicEvent(
        event_id="evt-0001",
        observations=(
            StationObservation(station=station, waveform=waveform),
            StationObservation(station=other, waveform=waveform),
        ),
        label="earthquake",
    )


@pytest.fixture
def toy_data_root(tmp_path: Path) -> Path:
    """A data root containing the toy PNW-layout dataset under raw/pnw/."""
    root = tmp_path / "data_root"
    pnw_fixture.build(root)
    return root


@pytest.fixture
def toy_raw_dir(toy_data_root: Path) -> Path:
    return toy_data_root / "raw" / "pnw"


def _range_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # keep test output quiet
            pass

        def _target(self) -> Path | None:
            path = root / self.path.lstrip("/")
            return path if path.is_file() else None

        def do_HEAD(self) -> None:
            path = self._target()
            if path is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"toy-etag"')
            self.end_headers()

        def do_GET(self) -> None:
            path = self._target()
            if path is None:
                self.send_error(404)
                return
            data = path.read_bytes()
            header = self.headers.get("Range")
            if header:
                start_s, end_s = header.removeprefix("bytes=").split("-")
                start, end = int(start_s), int(end_s) if end_s else len(data) - 1
                body = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture
def range_server(toy_raw_dir: Path) -> Iterator[str]:
    """Local HTTP server with byte-range support serving the toy raw dir; yields base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _range_handler(toy_raw_dir))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
