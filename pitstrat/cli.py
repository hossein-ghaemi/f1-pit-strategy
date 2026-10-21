"""Command line: download data, train the model, simulate strategies.

    python -m pitstrat.cli data
    python -m pitstrat.cli train
    python -m pitstrat.cli simulate --circuit Monza
"""

import argparse

from pitstrat import config
from pitstrat.data import build_dataset, load_dataset
from pitstrat.plot import plot_one_stop
from pitstrat.simulate import format_time, race_conditions, simulate
from pitstrat.train import load_model, train


def run_data(args):
    laps = build_dataset(args.seasons)
    print(f"saved {len(laps)} laps to {config.LAPS_FILE}")


def run_train(args):
    bundle = train(load_dataset())
    print("Lap delta error on held-out races, 5-fold by race (seconds):")
    for name, score in bundle["scores"].items():
        print(f"  {name:<14} MAE {score['mae']:.3f}  RMSE {score['rmse']:.3f}")
    print(f"saved {bundle['name']} to {config.MODEL_FILE}")


def run_simulate(args):
    laps = load_dataset()
    race = race_conditions(laps, args.circuit, args.year)
    result = simulate(load_model(), race, args.max_stops)

    print(f"{race.circuit}: {race.total_laps} laps, typical lap {race.base_lap:.3f}s, "
          f"pit loss {race.pit_loss:.1f}s")
    print(f"{len(result)} strategies simulated\n")
    top = result.head(args.top).assign(
        RaceTime=lambda frame: frame["RaceTime"].map(format_time),
        Gap=lambda frame: frame["Gap"].map(lambda gap: f"+{gap:.3f}s"))
    print(top.to_string(index=False))

    slug = race.circuit.lower().replace(" ", "-")
    csv_path = config.OUTPUTS_DIR / f"{slug}-strategies.csv"
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    plot_path = config.OUTPUTS_DIR / f"{slug}-one-stop.png"
    plot_one_stop(result, race, plot_path)
    print(f"\nsaved {csv_path}\nsaved {plot_path}")


def main():
    parser = argparse.ArgumentParser(prog="pitstrat")
    commands = parser.add_subparsers(required=True)

    data = commands.add_parser("data", help="download race laps from OpenF1")
    data.add_argument("--seasons", type=int, nargs="+", default=config.SEASONS)
    data.set_defaults(run=run_data)

    commands.add_parser("train", help="train and save the lap-time model").set_defaults(run=run_train)

    sim = commands.add_parser("simulate", help="rank pit strategies for a circuit")
    sim.add_argument("--circuit", required=True, help="OpenF1 circuit name, e.g. Monza")
    sim.add_argument("--year", type=int, help="race whose conditions to use (default: latest)")
    sim.add_argument("--max-stops", type=int, default=2)
    sim.add_argument("--top", type=int, default=10)
    sim.set_defaults(run=run_simulate)

    args = parser.parse_args()
    try:
        args.run(args)
    except ValueError as error:  # e.g. an unknown circuit name
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
