"""Measure how much time a pit stop costs at each circuit."""

from pitstrat import config
from pitstrat.data import clean_laps

# Stops outside this range happened under a safety car or went wrong.
VALID_LOSS = (10.0, 45.0)


def stop_losses(raw_laps):
    """Time lost by each green-flag stop: in-lap plus out-lap minus two normal laps."""
    race_driver = ["Year", "Circuit", "Driver"]
    medians = clean_laps(raw_laps).groupby(race_driver)["DriverMedian"].first().reset_index()

    in_laps = raw_laps[raw_laps["PitIn"]][race_driver + ["LapNumber", "LapTime"]]
    out_laps = raw_laps[raw_laps["PitOut"]][race_driver + ["LapNumber", "LapTime"]].copy()
    out_laps["LapNumber"] -= 1  # line each out-lap up with the in-lap before it

    stops = in_laps.merge(out_laps, on=race_driver + ["LapNumber"], suffixes=("In", "Out"))
    stops = stops.merge(medians, on=race_driver)
    stops["Loss"] = stops["LapTimeIn"] + stops["LapTimeOut"] - 2 * stops["DriverMedian"]
    return stops[stops["Loss"].between(*VALID_LOSS)]


def pit_loss(raw_laps, circuit):
    losses = stop_losses(raw_laps)
    losses = losses[losses["Circuit"] == circuit]["Loss"]
    return float(losses.median()) if len(losses) else config.DEFAULT_PIT_LOSS
