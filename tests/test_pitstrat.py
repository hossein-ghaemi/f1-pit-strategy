"""Offline tests on synthetic laps; nothing here touches the network."""

import pandas as pd
import pytest

from pitstrat import config, data
from pitstrat.data import clean_laps, strategies_used
from pitstrat.features import make_features
from pitstrat.pit_loss import stop_losses
from pitstrat.simulate import Race, simulate, stint_limits, strategies


def lap(number, time, **extra):
    row = {"Year": 2024, "Circuit": "Testville", "Driver": "AAA", "Team": "T",
           "LapNumber": number, "TotalLaps": 10, "LapTime": time, "Compound": "MEDIUM",
           "TyreLife": number, "Stint": 1, "PitIn": False, "PitOut": False,
           "AirTemp": 25.0, "TrackTemp": 40.0, "Rainfall": False}
    return row | extra


def test_clean_laps_drops_unrepresentative_laps():
    laps = pd.DataFrame([
        lap(1, 95.0),                          # opening lap
        lap(2, 90.0), lap(3, 90.2), lap(4, 90.4),
        lap(5, 91.0, PitIn=True),
        lap(6, 110.0, PitOut=True, Compound="HARD", TyreLife=1),
        lap(7, 120.0, Compound="HARD", TyreLife=2),   # safety car pace
        lap(8, 89.8, Compound="INTERMEDIATE"),
        lap(9, 89.9, Rainfall=True),
        lap(10, 90.1, Compound="HARD", TyreLife=5),
    ])
    # A second driver keeps the race above the dry-race threshold.
    other = pd.DataFrame([lap(n, 91.0, Driver="BBB") for n in range(1, 11)])
    clean = clean_laps(pd.concat([laps, other]))
    clean = clean[clean["Driver"] == "AAA"]
    assert clean["LapNumber"].tolist() == [2, 3, 4, 10]
    assert clean["DriverMedian"].iloc[0] == pytest.approx(90.15)
    assert clean["LapDelta"].sum() == pytest.approx(0.1)


def test_clean_laps_drops_mixed_weather_races():
    laps = pd.DataFrame([lap(n, 90.0, Compound="INTERMEDIATE" if n > 5 else "MEDIUM")
                         for n in range(1, 11)])
    assert clean_laps(laps).empty


def test_stop_loss_is_in_and_out_lap_minus_two_normal_laps():
    laps = pd.DataFrame([
        lap(2, 90.0), lap(3, 90.0), lap(4, 90.0),
        lap(5, 92.0, PitIn=True),
        lap(6, 110.0, PitOut=True),
        lap(7, 90.0), lap(8, 90.0),
    ])
    assert stop_losses(laps)["Loss"].tolist() == [pytest.approx(22.0)]


def test_features_align_to_training_columns():
    train = make_features(pd.DataFrame([lap(2, 90.0), lap(3, 90.0, Compound="HARD")]))
    other = make_features(pd.DataFrame([lap(4, 90.0, Compound="SOFT", Circuit="Elsewhere")]),
                          train.columns)
    assert list(other.columns) == list(train.columns)
    assert other[["Compound_HARD", "Compound_MEDIUM"]].sum().sum() == 0
    assert other["RaceProgress"].iloc[0] == pytest.approx(0.4)


def test_strategies_follow_the_rules():
    found = list(strategies(total_laps=50, max_stops=2))
    assert found
    for compounds, pit_laps in found:
        assert len(set(compounds)) >= 2
        lengths = pd.Series((0, *pit_laps, 50)).diff().dropna()
        for compound, length in zip(compounds, lengths):
            assert config.MIN_STINT <= length <= config.MAX_STINT[compound]


class TyreWear:
    """Fake model: soft is fast but wears quickly, hard is slow but lasts."""
    pace = {"SOFT": -1.0, "MEDIUM": -0.5, "HARD": 0.0}
    wear = {"SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.02}

    def predict(self, features):
        total = 0.0
        for compound in self.pace:
            on = features[f"Compound_{compound}"]
            total = total + on * (self.pace[compound] + self.wear[compound] * features["TyreLife"])
        return total.to_numpy()


def test_simulate_ranks_strategies_by_total_time():
    bundle = {"model": TyreWear(),
              "columns": ["TyreLife", "Compound_SOFT", "Compound_MEDIUM", "Compound_HARD"]}
    race = Race("Testville", 2024, total_laps=40, base_lap=90.0, pit_loss=20.0)
    result = simulate(bundle, race, max_stops=2)

    assert result["RaceTime"].is_monotonic_increasing
    assert result["Gap"].iloc[0] == 0
    # With a 20 s pit loss and gentle wear, one stop beats two.
    assert result["Stops"].iloc[0] == 1

    # Hand-check one strategy: soft for laps 1-15, hard for laps 16-40.
    soft = sum(90.0 - 1.0 + 0.10 * life for life in range(1, 16))
    hard = sum(90.0 + 0.02 * life for life in range(1, 26))
    row = result[(result["Strategy"] == "S-H") & (result["PitLaps"] == "15")]
    assert row["RaceTime"].iloc[0] == pytest.approx(soft + hard + 20.0)


def test_strategies_used_reads_compounds_and_pit_laps_from_stints():
    laps = pd.DataFrame(
        [lap(n, 90.0, Stint=1, Compound="SOFT") for n in range(1, 16)]
        + [lap(n, 90.0, Stint=2, Compound="HARD") for n in range(16, 41)]
        + [lap(n, 90.0, Stint=3, Compound="MEDIUM") for n in range(41, 51)])
    used = strategies_used(laps).loc["AAA"]
    assert used["Strategy"] == "S-H-M"
    assert used["PitLaps"] == "15, 40"


def test_stint_limits_come_from_real_stints():
    rows = []
    for number, driver in enumerate("ABCDEFGHIJ"):
        hard = 30 + number  # hard stints of 30-39 laps
        rows += [lap(n, 90.0, Driver=driver, Stint=1, Compound="MEDIUM", TotalLaps=hard + 15)
                 for n in range(1, 16)]
        rows += [lap(n, 90.0, Driver=driver, Stint=2, Compound="HARD", TotalLaps=hard + 15)
                 for n in range(16, hard + 16)]
    limits = stint_limits(pd.DataFrame(rows), "Testville")
    assert limits["MEDIUM"] == 15
    assert limits["HARD"] == round(pd.Series(range(30, 40)).quantile(config.STINT_QUANTILE))
    assert limits["SOFT"] == config.MAX_STINT["SOFT"]  # never used, so the fallback


def fake_race_feeds(pit):
    stints = [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3,
               "compound": "MEDIUM", "tyre_age_at_start": 0},
              {"driver_number": 1, "stint_number": 2, "lap_start": 4, "lap_end": 6,
               "compound": "HARD", "tyre_age_at_start": 0}]
    laps = [{"driver_number": 1, "lap_number": n, "lap_duration": 90.0, "is_pit_out_lap": n == 4,
             "date_start": f"2024-01-01T12:0{n}:00+00:00"} for n in range(1, 7)]
    return {"drivers": [{"driver_number": 1, "name_acronym": "AAA", "team_name": "T"}],
            "laps": laps, "stints": stints, "pit": pit,
            "weather": [{"date": "2024-01-01T11:59:00+00:00", "air_temperature": 25.0,
                         "track_temperature": 40.0, "rainfall": 0}]}


@pytest.mark.parametrize("pit", [[{"driver_number": 1, "lap_number": 3}], []])
def test_race_laps_finds_stops_with_or_without_pit_feed(monkeypatch, pit):
    feeds = fake_race_feeds(pit)
    monkeypatch.setattr(data, "fetch_cached", lambda feed, key: feeds[feed])
    laps = data.race_laps({"session_key": 1, "year": 2024, "circuit_short_name": "Testville"})
    assert laps[laps["PitIn"]]["LapNumber"].tolist() == [3]
    assert laps[laps["PitOut"]]["LapNumber"].tolist() == [4]
    assert laps["TyreLife"].tolist() == [1, 2, 3, 1, 2, 3]
