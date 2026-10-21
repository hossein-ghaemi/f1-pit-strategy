"""Download race laps from OpenF1 and clean them for modelling.

F1's live-timing archive, which FastF1 reads laps from, rejects requests, so
laps, tyre stints, pit stops and weather come from OpenF1 instead. Raw
responses are cached on disk so each race downloads only once.
"""

import json
import time

import pandas as pd
import requests

from pitstrat import config

API = "https://api.openf1.org/v1/"
FEEDS = ("drivers", "laps", "stints", "pit", "weather")

COLUMNS = [
    "Year", "Circuit", "Driver", "Team", "LapNumber", "TotalLaps", "LapTime",
    "Compound", "TyreLife", "Stint", "PitIn", "PitOut", "AirTemp", "TrackTemp",
    "Rainfall",
]


def fetch(feed, **params):
    """GET one OpenF1 feed, waiting and retrying when rate limited."""
    for attempt in range(7):
        time.sleep(0.5)  # stay under OpenF1's free request rate
        response = requests.get(API + feed, params=params, timeout=60)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 404:  # OpenF1 answers 404 when a feed has no rows
            return []
        response.raise_for_status()
        return response.json()
    response.raise_for_status()


def fetch_cached(feed, session_key):
    path = config.CACHE_DIR / str(session_key) / f"{feed}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = fetch(feed, session_key=session_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def race_sessions(year):
    """Finished races of one season."""
    sessions = fetch("sessions", year=year, session_name="Race")
    now = pd.Timestamp.now(tz="UTC")
    return [s for s in sessions
            if not s.get("is_cancelled") and pd.Timestamp(s["date_end"]) < now]


def race_laps(session):
    """Every lap of one race as a flat table with tyre and weather attached."""
    feeds = {feed: fetch_cached(feed, session["session_key"]) for feed in FEEDS}

    laps = pd.DataFrame(feeds["laps"])
    laps = laps[["driver_number", "lap_number", "lap_duration", "is_pit_out_lap", "date_start"]]

    drivers = pd.DataFrame(feeds["drivers"])[["driver_number", "name_acronym", "team_name"]]
    laps = laps.merge(drivers.drop_duplicates("driver_number"), on="driver_number", how="left")

    # Each lap belongs to the stint whose lap range contains it.
    stints = pd.DataFrame(feeds["stints"]).dropna(subset=["lap_start", "lap_end"])
    laps = laps.merge(stints[["driver_number", "stint_number", "lap_start", "lap_end",
                              "compound", "tyre_age_at_start"]], on="driver_number", how="left")
    laps = laps[(laps["lap_number"] >= laps["lap_start"]) & (laps["lap_number"] <= laps["lap_end"])]

    pits = pd.DataFrame(feeds["pit"], columns=["driver_number", "lap_number"])
    pits["pit_in"] = True
    laps = laps.merge(pits.drop_duplicates(), on=["driver_number", "lap_number"], how="left")

    # Attach the latest weather reading taken before each lap started.
    weather = pd.DataFrame(feeds["weather"])
    weather["date"] = pd.to_datetime(weather["date"], format="ISO8601")
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601")
    laps = pd.merge_asof(laps.dropna(subset=["date_start"]).sort_values("date_start"),
                         weather.sort_values("date"), left_on="date_start", right_on="date",
                         direction="backward")

    return pd.DataFrame({
        "Year": session["year"],
        "Circuit": session["circuit_short_name"],
        "Driver": laps["name_acronym"],
        "Team": laps["team_name"],
        "LapNumber": laps["lap_number"],
        "TotalLaps": int(laps["lap_number"].max()),
        "LapTime": laps["lap_duration"],
        "Compound": laps["compound"],
        # The first lap on a tyre counts as tyre life 1.
        "TyreLife": laps["tyre_age_at_start"].fillna(0) + laps["lap_number"] - laps["lap_start"] + 1,
        "Stint": laps["stint_number"],
        "PitIn": laps["pit_in"].notna(),
        "PitOut": laps["is_pit_out_lap"].fillna(False).astype(bool),
        "AirTemp": laps["air_temperature"],
        "TrackTemp": laps["track_temperature"],
        "Rainfall": laps["rainfall"].fillna(0).astype(bool),
    })[COLUMNS].sort_values(["Driver", "LapNumber"])


def build_dataset(seasons=config.SEASONS):
    """Download all races of the given seasons and save them to LAPS_FILE."""
    frames = []
    for year in seasons:
        for session in race_sessions(year):
            name = f"{year} {session['circuit_short_name']}"
            try:
                frames.append(race_laps(session))
                print(f"loaded {name}")
            except Exception as error:  # one broken race should not stop the build
                print(f"skipped {name}: {error}")

    laps = pd.concat(frames, ignore_index=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    laps.to_csv(config.LAPS_FILE, index=False)
    return laps


def load_dataset():
    return pd.read_csv(config.LAPS_FILE)


def clean_laps(laps):
    """Keep only representative laps on dry tyres from dry races.

    Safety car, VSC and traffic-heavy laps are much slower than normal, so
    the outlier limit removes them along with the opening lap and pit laps.
    """
    race = ["Year", "Circuit"]
    dry = laps["Compound"].isin(config.DRY_COMPOUNDS) & ~laps["Rainfall"]
    laps = laps[dry.groupby([laps[c] for c in race]).transform("mean") >= config.MIN_DRY_SHARE]

    laps = laps[
        (laps["LapNumber"] > 1)
        & ~laps["PitIn"]
        & ~laps["PitOut"]
        & ~laps["Rainfall"]
        & laps["Compound"].isin(config.DRY_COMPOUNDS)
        & laps["LapTime"].notna()
    ].copy()

    race_driver = race + ["Driver"]
    median = laps.groupby(race_driver)["LapTime"].transform("median")
    laps = laps[laps["LapTime"] <= median * config.OUTLIER_LIMIT].copy()

    # The model learns how much slower or faster a lap is than the driver's
    # typical lap in that race, which removes car and driver pace.
    laps["DriverMedian"] = laps.groupby(race_driver)["LapTime"].transform("median")
    laps["LapDelta"] = laps["LapTime"] - laps["DriverMedian"]
    return laps.reset_index(drop=True)
