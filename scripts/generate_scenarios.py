"""Generate 22+ realistic diverse seismic scenarios for Seismo UI showcase.

Includes:
- 8 Natural Tectonic Earthquakes across the Pacific Northwest (Puget Sound, Seattle Fault, Cascadia, Rainier, etc.)
- 8 Anthropogenic Quarry Blasts & Mining Explosions (Centralia, Tenino, Skykomish, Snoqualmie, etc.)
- 6 Out-of-Distribution (OOD) Exotic/Environmental Sources (Mount St. Helens Tremor, Oso Slide, Sonic Boom, Icefall, Thunderquake, etc.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

SAMPLE_RATE = 50.0  # Hz
N_SAMPLES = 3000    # 60 seconds @ 50 Hz
P_SAMPLE = 500      # 10.0 s

STATIONS_DB = {
    "UW.ALKI": {"name": "Alki Point, Seattle", "lat": 47.575, "lon": -122.418},
    "UW.GSW": {"name": "Gold Mountain, Bremerton", "lat": 47.288, "lon": -122.502},
    "UW.RCM": {"name": "Camp Muir, Mt. Rainier", "lat": 46.852, "lon": -121.758},
    "CC.JRO": {"name": "Johnston Ridge, Mt. St. Helens", "lat": 46.275, "lon": -122.218},
    "UW.HDW": {"name": "Hoodsport, Mason Co.", "lat": 47.653, "lon": -122.309},
    "UO.HAW": {"name": "Hawthorne, Portland", "lat": 45.412, "lon": -122.388},
    "UW.BLN": {"name": "Blyn, Olympic Peninsula", "lat": 48.006, "lon": -123.047},
    "CC.SEP": {"name": "Mount St. Helens East", "lat": 46.200, "lon": -122.190},
    "UW.TWW": {"name": "Tahuya, Kitsap", "lat": 47.450, "lon": -122.880},
    "UW.ELLN": {"name": "Ellensburg, Central WA", "lat": 46.996, "lon": -120.547},
    "CC.YEL": {"name": "Yelm, Nisqually Valley", "lat": 46.941, "lon": -122.607},
    "UW.NLW": {"name": "Northgate, Seattle", "lat": 47.708, "lon": -122.327},
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return round(2 * R * math.asin(math.sqrt(a)), 1)


def generate_trace(category: str, sub_type: str, dist_km: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    p_time = 10.0
    dt_ps = max(1.2, min(9.0, dist_km / 8.0))
    s_time = p_time + dt_ps
    geom_spread = 1.0 / max(1.0, math.sqrt(dist_km / 12.0))

    trace = np.zeros((3, N_SAMPLES), dtype=np.float32)

    if category == "earthquake":
        p_freq = rng.uniform(4.0, 9.0)
        s_freq = rng.uniform(2.5, 5.5)

        p_env = np.exp(-np.maximum(0.0, t - p_time) * 1.5) * (t >= p_time)
        p_z = np.sin(2 * np.pi * p_freq * (t - p_time)) * p_env * 0.8
        p_h = np.sin(2 * np.pi * p_freq * (t - p_time) + 0.4) * p_env * 0.4

        s_env = np.exp(-np.maximum(0.0, t - s_time) * 0.55) * (t >= s_time)
        s_e = np.sin(2 * np.pi * s_freq * (t - s_time) + rng.uniform(0, math.pi)) * s_env * 2.6
        s_n = np.cos(2 * np.pi * s_freq * (t - s_time) + rng.uniform(0, math.pi)) * s_env * 2.3
        s_z = np.sin(2 * np.pi * s_freq * (t - s_time) * 1.1) * s_env * 0.9

        coda_env = np.exp(-np.maximum(0.0, t - s_time) * 0.22) * (t >= s_time)
        coda = rng.randn(3, N_SAMPLES) * coda_env * 0.22

        trace[0] = p_h * 0.6 + s_e + coda[0]
        trace[1] = p_h * 0.6 + s_n + coda[1]
        trace[2] = p_z + s_z + coda[2]

    elif category == "anthropogenic":
        p_freq = rng.uniform(6.5, 14.0)
        s_freq = rng.uniform(3.0, 5.0)

        # High vertical P-wave onset, weak S-wave
        p_env = np.exp(-np.maximum(0.0, t - p_time) * 3.8) * (t >= p_time)
        p_z = np.sin(2 * np.pi * p_freq * (t - p_time)) * p_env * 3.4
        p_e = np.sin(2 * np.pi * p_freq * (t - p_time) + 0.3) * p_env * 0.7
        p_n = np.cos(2 * np.pi * p_freq * (t - p_time) + 0.6) * p_env * 0.6

        # Weak surface wave
        s_env = np.exp(-np.maximum(0.0, t - s_time) * 1.3) * (t >= s_time)
        s_e = np.sin(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.65
        s_n = np.cos(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.55
        s_z = np.sin(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.35

        coda_env = np.exp(-np.maximum(0.0, t - p_time) * 0.75) * (t >= p_time)
        coda = rng.randn(3, N_SAMPLES) * coda_env * 0.14

        trace[0] = p_e + s_e + coda[0]
        trace[1] = p_n + s_n + coda[1]
        trace[2] = p_z + s_z + coda[2]

    else:  # OOD / Exotic
        if sub_type == "sonic_boom":
            # Classic atmospheric shock N-wave: sharp positive peak, negative overshoot, minimal coda
            env = np.exp(-np.maximum(0.0, t - p_time) * 8.0) * (t >= p_time)
            pulse = np.sin(2 * np.pi * 3.0 * (t - p_time)) * env * 3.0
            trace[0] = pulse * 0.4
            trace[1] = pulse * 0.5
            trace[2] = pulse * 2.2
        elif sub_type == "tremor":
            # Volcanic tremor: continuous monochromatic resonance 1.1 Hz
            freq = 1.15
            env = np.exp(-((t - 25.0) / 14.0) ** 2)
            tremor = np.sin(2 * np.pi * freq * t) * env * 1.8
            trace[0] = tremor
            trace[1] = tremor * 0.9
            trace[2] = tremor * 1.3
        elif sub_type == "avalanche":
            # Oso landslide: emergent slow-rise envelope, low-frequency rumble (0.8 - 2.5 Hz)
            rise = np.minimum(1.0, np.maximum(0.0, (t - 8.0) / 15.0))
            decay = np.exp(-np.maximum(0.0, t - 28.0) * 0.12)
            env = rise * decay
            rumble = (np.sin(2 * np.pi * 1.4 * t) + 0.6 * np.sin(2 * np.pi * 2.2 * t)) * env * 1.6
            trace[0] = rumble * 1.2
            trace[1] = rumble * 1.1
            trace[2] = rumble * 0.9
        elif sub_type == "icefall":
            # Nisqually glacier calving: irregular bursts 3 - 6 Hz
            env = np.exp(-((t - 14.0) / 5.0) ** 2) + 0.7 * np.exp(-((t - 22.0) / 6.0) ** 2)
            burst = np.sin(2 * np.pi * 4.2 * t) * env * 1.7
            trace[0] = burst * 0.8
            trace[1] = burst * 1.0
            trace[2] = burst * 1.2
        elif sub_type == "thunder":
            # Acoustic coupling: late arrival, reverberant low frequency
            arr_time = 16.0
            env = np.exp(-np.maximum(0.0, t - arr_time) * 0.4) * (t >= arr_time)
            thund = np.sin(2 * np.pi * 2.8 * (t - arr_time)) * env * 1.9
            trace[0] = thund * 0.6
            trace[1] = thund * 0.7
            trace[2] = thund * 1.5
        else:
            # Generic exotic anomaly
            env = np.exp(-((t - 20.0) / 8.0) ** 2)
            noise = np.sin(2 * np.pi * 1.6 * t) * env * 1.5
            trace[0] = noise
            trace[1] = noise
            trace[2] = noise

    trace = trace * geom_spread
    # Add ambient noise floor
    trace += rng.randn(3, N_SAMPLES) * 0.05

    # Multi-channel normalization
    std = float(np.std(trace))
    if std > 1e-6:
        trace /= std

    return trace.astype(np.float32)


# Build the 22 comprehensive scenarios
SCENARIOS_META = [
    # --- NATURAL EARTHQUAKES (8) ---
    {
        "id": "eq-01",
        "title": "M 2.4 Puget Sound Shallow Crustal Earthquake",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (47.45, -122.38),
        "stations": ["UW.ALKI", "UW.GSW", "UW.HDW", "UW.NLW"],
        "description": "Double-couple shear slip at 14 km depth beneath Puget Sound. Strong S-waves visible across urban array.",
    },
    {
        "id": "eq-02",
        "title": "M 3.1 Seattle Fault Zone Thrust Earthquake",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (47.58, -122.35),
        "stations": ["UW.ALKI", "UW.NLW", "UW.GSW"],
        "description": "Reverse thrust faulting along the Seattle Fault zone with pronounced horizontal shear energy and slow coda decay.",
    },
    {
        "id": "eq-03",
        "title": "M 1.8 Mount Rainier Volcanic-Tectonic Shear Event",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (46.85, -121.75),
        "stations": ["UW.RCM", "CC.YEL", "UW.GSW"],
        "description": "High-frequency volcano-tectonic shear fracture inside the Mount Rainier volcanic edifice at 6 km depth.",
    },
    {
        "id": "eq-04",
        "title": "M 3.8 Cascadia Subduction Zone Deep Intraplate Event",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (47.20, -122.70),
        "stations": ["UW.GSW", "UW.TWW", "UW.ALKI", "CC.YEL"],
        "description": "Benioff zone deep slab event (48 km depth) within the subducting Juan de Fuca plate. High-velocity mantle path.",
    },
    {
        "id": "eq-05",
        "title": "M 2.7 Olympic Peninsula Strike-Slip Earthquake",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (47.95, -123.10),
        "stations": ["UW.BLN", "UW.HDW", "UW.TWW"],
        "description": "Strike-slip faulting in the accretionary wedge of the Olympic Mountains with sharp impulsive S-arrivals.",
    },
    {
        "id": "eq-06",
        "title": "M 2.1 Mount St. Helens Tectonic Swarm Event",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (46.22, -122.18),
        "stations": ["CC.JRO", "CC.SEP", "UW.RCM"],
        "description": "Tectonic brittle fracture under the 2004–2008 lava dome complex. Clear P and S wave separation.",
    },
    {
        "id": "eq-07",
        "title": "M 2.9 Cle Elum East Cascades Crustal Shock",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (47.10, -120.90),
        "stations": ["UW.ELLN", "UW.RCM", "UW.ALKI"],
        "description": "Crustal earthquake on the eastern flank of the Cascade Range with regional Lg phase development.",
    },
    {
        "id": "eq-08",
        "title": "M 2.5 Portland Hills Fault Zone Earthquake",
        "category": "Earthquake",
        "ground_truth": "earthquake",
        "epicenter": (45.50, -122.65),
        "stations": ["UO.HAW", "CC.YEL", "UW.GSW"],
        "description": "Transpressional faulting in the Portland metropolitan basin. High shear-to-compressional amplitude ratio.",
    },

    # --- ANTHROPOGENIC QUARRY BLASTS & EXPLOSIONS (8) ---
    {
        "id": "anth-01",
        "title": "Centralia Coal Mining Surface Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (46.75, -122.90),
        "stations": ["CC.YEL", "UW.GSW", "UW.ALKI"],
        "description": "Open-pit mine production blast in Lewis County. Dominant high-frequency vertical P-wave, minimal S-wave.",
    },
    {
        "id": "anth-02",
        "title": "Tenino Limestone Quarry Production Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (46.85, -122.80),
        "stations": ["CC.YEL", "UW.GSW", "UW.HDW"],
        "description": "Commercial aggregate quarry blast with delayed bench sequencing. Very high P/S amplitude ratio (> 3.5).",
    },
    {
        "id": "anth-03",
        "title": "Skykomish Granite Quarry Ripple-Fired Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (47.70, -121.35),
        "stations": ["UW.NLW", "UW.ALKI", "UW.ELLN"],
        "description": "Millisecond-delay ripple-fired bench blast in northern Cascades. Strong Rg surface waves and steep coda roll-off.",
    },
    {
        "id": "anth-04",
        "title": "Snoqualmie Pass Highway Excavation Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (47.42, -121.40),
        "stations": ["UW.ALKI", "UW.ELLN", "UW.RCM"],
        "description": "Controlled construction rock-cut blast along the I-90 corridor. High-frequency compressional source.",
    },
    {
        "id": "anth-05",
        "title": "Black Diamond Sand & Gravel Open-Pit Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (47.30, -122.00),
        "stations": ["UW.ALKI", "UW.GSW", "UW.RCM"],
        "description": "Shallow aggregate detonation in King County. Predominantly vertical ground motion on regional geophones.",
    },
    {
        "id": "anth-06",
        "title": "Columbia River Basalt Mining Detonation",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (45.70, -122.40),
        "stations": ["UO.HAW", "CC.YEL", "CC.JRO"],
        "description": "High-volume basalt quarry explosion near SW Washington border. Characteristic high-frequency P-onset.",
    },
    {
        "id": "anth-07",
        "title": "Spokane Valley Industrial Rock Quarry Blast",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (47.05, -120.50),
        "stations": ["UW.ELLN", "UW.RCM"],
        "description": "Eastern Washington aggregate blast. Confined surface depth (< 0.2 km), fast acoustic attenuation.",
    },
    {
        "id": "anth-08",
        "title": "Shelton Hardrock Quarry Blasting Sequence",
        "category": "Quarry Blast",
        "ground_truth": "anthropogenic",
        "epicenter": (47.25, -123.15),
        "stations": ["UW.HDW", "UW.TWW", "UW.GSW"],
        "description": "Multi-hole bench blast with strong impulsive compressional P-wave arrival and near-zero horizontal shear energy.",
    },

    # --- OUT-OF-DISTRIBUTION (OOD) EXOTIC SOURCES (6) ---
    {
        "id": "ood-01",
        "title": "Mount St. Helens Low-Frequency Volcanic Tremor",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "tremor",
        "epicenter": (46.20, -122.19),
        "stations": ["CC.JRO", "CC.SEP", "UW.RCM"],
        "description": "Continuous monochromatic acoustic resonance (1.15 Hz) generated by magmatic fluid conduit oscillations.",
    },
    {
        "id": "ood-02",
        "title": "Oso Debris Flow & Rapid Mass Wasting Event",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "avalanche",
        "epicenter": (48.28, -121.90),
        "stations": ["UW.NLW", "UW.BLN", "UW.ALKI"],
        "description": "Catastrophic deep-seated slide. Emergent low-frequency rumble lacking distinct P and S body wave phases.",
    },
    {
        "id": "ood-03",
        "title": "Whidbey Island Supersonic Aircraft Sonic Boom",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "sonic_boom",
        "epicenter": (48.30, -122.65),
        "stations": ["UW.BLN", "UW.NLW", "UW.ALKI"],
        "description": "Atmospheric shockwave coupling into ground. Characteristic symmetric N-shaped pulse with abrupt termination.",
    },
    {
        "id": "ood-04",
        "title": "Nisqually Glacier Serac Collapse & Icefall",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "icefall",
        "epicenter": (46.84, -121.74),
        "stations": ["UW.RCM", "CC.YEL"],
        "description": "Glacial ice avalanche on south flank of Mount Rainier. Highly scattered high-frequency wave packet.",
    },
    {
        "id": "ood-05",
        "title": "Puget Sound Severe Convective Thunderquake",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "thunder",
        "epicenter": (47.50, -122.25),
        "stations": ["UW.ALKI", "UW.GSW", "UW.NLW"],
        "description": "Cloud-to-ground lightning return stroke coupling acoustic energy into regional ground sensors at sound speed.",
    },
    {
        "id": "ood-06",
        "title": "North Cascades Massive Rock Avalanche",
        "category": "Out-of-Distribution",
        "ground_truth": "out-of-distribution",
        "sub_type": "avalanche",
        "epicenter": (48.50, -121.10),
        "stations": ["UW.NLW", "UW.BLN", "UW.ELLN"],
        "description": "Sub-alpine steep cliff collapse. Multi-minute emergent seismic energy envelope without shear onset.",
    },
]


def generate_all():
    print("Generating 22 rich seismic scenarios...")
    samples = []
    base_seed = 42000

    for idx, meta in enumerate(SCENARIOS_META):
        epi_lat, epi_lon = meta["epicenter"]
        stations_list = []
        wfs_list = []

        for st_idx, st_id in enumerate(meta["stations"]):
            st_info = STATIONS_DB[st_id]
            dist = haversine_km(epi_lat, epi_lon, st_info["lat"], st_info["lon"])
            st_obj = {
                "id": st_id,
                "name": st_info["name"],
                "lat": st_info["lat"],
                "lon": st_info["lon"],
                "dist_km": dist,
            }
            stations_list.append(st_obj)

            sub = meta.get("sub_type", "standard")
            wf = generate_trace(
                category=meta["ground_truth"],
                sub_type=sub,
                dist_km=dist,
                seed=base_seed + idx * 100 + st_idx * 7,
            )
            wfs_list.append(wf.tolist())

        sample = {
            "id": meta["id"],
            "title": meta["title"],
            "category": meta["category"],
            "ground_truth": meta["ground_truth"],
            "description": meta["description"],
            "stations": stations_list,
            "waveforms": wfs_list,
        }
        samples.append(sample)
        print(f"[{idx+1:02d}/22] {meta['category']:<18} | {meta['title']} ({len(stations_list)} stations)")

    out_file = Path("outputs/showcase_samples.json")
    out_file.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"\n[OK] Successfully wrote 22 scenarios to {out_file}")


if __name__ == "__main__":
    generate_all()

