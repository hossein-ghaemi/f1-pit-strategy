"""Simulate pit-stop strategies and rank them by predicted race time."""

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import pandas as pd

from pitstrat import config
from pitstrat.data import clean_laps, dry_races
from pitstrat.pit_loss import pit_loss
from pitstrat.train import predict


@dataclass
class Race:
    circuit: str
    year: int
    total_laps: int
    base_lap: float  # a typical green-flag lap of the field, in seconds
    pit_loss: float
    max_stint: dict = field(default_factory=lambda: dict(config.MAX_STINT))


def stint_limits(raw_laps, circuit):
    """Longest stint per compound worth simulating, from the stints teams ran at a circuit."""
    laps = dry_races(raw_laps[raw_laps["Circuit"] == circuit])
    laps = laps[laps["Compound"].isin(config.DRY_COMPOUNDS)]
    stints = laps.groupby(["Year", "Driver", "Stint"]).agg(
        Compound=("Compound", "first"), Length=("LapNumber", "size"))
    limits = {}
    for compound, cap in config.MAX_STINT.items():
        lengths = stints[stints["Compound"] == compound]["Length"]
        if len(lengths) < config.MIN_STINTS:
            limits[compound] = cap
        else:
            limits[compound] = min(cap, int(round(lengths.quantile(config.STINT_QUANTILE))))
    return limits


def race_conditions(raw_laps, circuit, year=None):
    """Race length, pace and pit loss of a circuit, from its latest dry race by default."""
    at_circuit = raw_laps[raw_laps["Circuit"] == circuit]
    if at_circuit.empty:
        raise ValueError(f"No races at {circuit!r}. Known circuits: "
                         + ", ".join(sorted(raw_laps["Circuit"].unique())))
    clean = clean_laps(at_circuit)
    year = year or clean["Year"].max()
    clean = clean[clean["Year"] == year]
    if clean.empty:
        raise ValueError(f"No dry race at {circuit!r} in {year}.")
    race = at_circuit[at_circuit["Year"] == year]
    return Race(
        circuit=circuit,
        year=int(year),
        total_laps=int(race["TotalLaps"].max()),
        base_lap=float(clean.groupby("Driver")["DriverMedian"].first().median()),
        pit_loss=pit_loss(raw_laps, circuit),
        max_stint=stint_limits(raw_laps, circuit),
    )


def stint_costs(bundle, race):
    """Predicted time of every possible stint: cost[compound][start lap][length].

    Predicting each (compound, lap, tyre age) once and summing along stints is
    much faster than predicting every lap of every strategy.
    """
    longest = max(race.max_stint.values())
    grid = pd.DataFrame(
        product(config.DRY_COMPOUNDS, range(1, race.total_laps + 1), range(1, longest + 1)),
        columns=["Compound", "LapNumber", "TyreLife"])
    grid = grid.assign(Circuit=race.circuit, TotalLaps=race.total_laps)
    lap_times = race.base_lap + predict(bundle, grid)
    lap_times = lap_times.reshape(len(config.DRY_COMPOUNDS), race.total_laps, longest)

    costs = {}
    for c, compound in enumerate(config.DRY_COMPOUNDS):
        costs[compound] = np.zeros((race.total_laps + 2, longest + 1))
        for start in range(1, race.total_laps + 1):
            for length in range(1, min(longest, race.total_laps - start + 1) + 1):
                life = np.arange(length)
                costs[compound][start, length] = lap_times[c, start - 1 + life, life].sum()
    return costs


def strategies(total_laps, max_stops=2, max_stint=config.MAX_STINT):
    """Yield (compounds, pit laps) for every legal strategy.

    A pit lap is the last lap of a stint. Every stint must fit between MIN_STINT
    and the compound's ``max_stint``, and a dry race must use two different compounds.
    """
    for stops in range(1, max_stops + 1):
        for compounds in product(config.DRY_COMPOUNDS, repeat=stops + 1):
            if len(set(compounds)) < 2:
                continue
            for pit_laps in combinations(range(1, total_laps), stops):
                edges = (0, *pit_laps, total_laps)
                lengths = np.diff(edges)
                if all(config.MIN_STINT <= length <= max_stint[compound]
                       for compound, length in zip(compounds, lengths)):
                    yield compounds, pit_laps


def simulate(bundle, race, max_stops=2):
    """Every legal strategy with its predicted race time, fastest first."""
    costs = stint_costs(bundle, race)
    rows = []
    for compounds, pit_laps in strategies(race.total_laps, max_stops, race.max_stint):
        edges = (0, *pit_laps, race.total_laps)
        time = len(pit_laps) * race.pit_loss
        for compound, start, end in zip(compounds, edges, edges[1:]):
            time += costs[compound][start + 1, end - start]
        rows.append({
            "Strategy": "-".join(compound[0] for compound in compounds),
            "Stops": len(pit_laps),
            "PitLaps": ", ".join(map(str, pit_laps)),
            "RaceTime": time,
        })
    result = pd.DataFrame(rows).sort_values("RaceTime", ignore_index=True)
    result["Gap"] = result["RaceTime"] - result["RaceTime"].iloc[0]
    return result


def format_time(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{rest:06.3f}"
