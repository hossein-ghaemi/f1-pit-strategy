# F1 Pit Strategy

Predicts Formula 1 lap times with a tree model and simulates pit-stop strategies to find the one with the lowest predicted race time.

## How it works

1. **Data.** Laps, tyre stints, pit stops and weather for every 2023–2025 race come from [OpenF1](https://openf1.org). F1's live-timing archive, which FastF1 uses, rejects requests. Raw responses are cached under `data/openf1/`.
2. **Cleaning.** Mixed-weather races, where under 90% of laps ran on dry tyres in the dry, are dropped. From the rest, these laps are dropped: the opening lap, in-laps and out-laps, wet laps, non-dry compounds, and any lap more than 7% slower than the driver's median for that race. The last rule removes safety car, VSC and heavy-traffic laps.
3. **Target.** Each lap is turned into a delta from the driver's median lap in that race. Car and driver pace cancel out, and the model learns how tyre wear and fuel burn change lap time.
4. **Model.** Random Forest and XGBoost are trained on these features: tyre age, compound, lap number, race progress (fuel load), and circuit. Both are scored with 5-fold cross-validation by race, so each test fold holds whole races the model never saw. The one with the lower error is refit on all laps and saved. Air and track temperature were dropped: they mostly identify single races, and including them made held-out error worse.
5. **Pit loss.** Each green-flag stop costs its in-lap plus out-lap, minus two normal laps. A circuit's pit loss is the median over its stops.
6. **Simulation.** The circuit's latest dry race supplies the race length, typical lap and pit loss. Every legal one- and two-stop strategy is built: at least two dry compounds, with stint lengths between `MIN_STINT` and each compound's `MAX_STINT`. Each lap is predicted as the circuit's typical lap plus the model's delta, pit losses are added, and the strategies are ranked by total time.

## Repository

- `pitstrat/data.py`: OpenF1 download and lap cleaning.
- `pitstrat/features.py`: model inputs.
- `pitstrat/train.py`: model comparison, training and saving.
- `pitstrat/pit_loss.py`: pit-stop time loss per circuit.
- `pitstrat/simulate.py`: strategy generation and race-time simulation.
- `pitstrat/plot.py`: one-stop strategy chart.
- `pitstrat/cli.py`: command line.
- `pitstrat/config.py`: seasons, outlier limit, stint limits and paths.
- `tests/`: offline tests on synthetic laps.

## Setup

Use Python 3.12 and run commands from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Usage

Activate the environment first in every new terminal:

```sh
source .venv/bin/activate
```

### First run

Download the races, then train the model. Both only need to run once; run them again only after deleting `data/` or `models/`, or to pick up new races.

```sh
python -m pitstrat.cli data    # download 2023-2025 races (about 10 minutes the first time)
python -m pitstrat.cli train   # compare both models, save the best to models/ (1-2 minutes)
```

`train` prints each model's error on held-out races next to a baseline that predicts every lap as the driver's median lap.

### Simulate strategies

```sh
python -m pitstrat.cli simulate --circuit Monza
```

This prints the circuit's race length, typical lap and pit loss, then the 10 fastest strategies. Each row shows:

- **Strategy:** the compounds in order. `H-S` means Hard, then Soft.
- **Stops:** how many pit stops the strategy makes.
- **PitLaps:** the last lap of each stint.
- **RaceTime:** the predicted race time.
- **Gap:** how far behind the fastest strategy it is.

Below that it prints the real result of the race the conditions came from, so you can compare. It shows each finisher's position, driver, team, race time, laps completed, and the compounds and pit laps they actually used. Real race times include safety cars and traffic, which the prediction leaves out.

It also writes the full ranking to `outputs/<circuit>-strategies.csv`. A chart of one-stop race time by pit lap goes to `outputs/<circuit>-one-stop.png`.

The `simulate` command also takes these options:

- `--year`: use the conditions of a specific season's race. The default is the latest dry race at that circuit.
- `--max-stops`: the most stops a strategy can have. The default is 2.
- `--top`: how many strategies to print. The default is 10.
- `--results`: how many finishers of the real race to show. The default is 5.

```sh
python -m pitstrat.cli simulate --circuit Sakhir --top 20         # show 20 strategies
python -m pitstrat.cli simulate --circuit Sakhir --year 2024      # use the 2024 race
python -m pitstrat.cli simulate --circuit Sakhir --max-stops 1    # one-stop strategies only
python -m pitstrat.cli simulate --circuit Sakhir --results 20      # show the whole real result
```

Circuit names follow OpenF1, for example `Monza`, `Sakhir`, `Silverstone`, `Suzuka` and `Spa-Francorchamps`. An unknown name lists the valid ones. Quote names that contain spaces:

```sh
python -m pitstrat.cli simulate --circuit "Yas Marina Circuit"
```

### PyCharm

1. Open the `f1-pit-strategy` folder.
2. Set the project interpreter to `.venv/bin/python`.
3. Add a **Python** run configuration:
   - Switch **script** to **module** and enter `pitstrat.cli`.
   - Set **Parameters** to `simulate --circuit Monza`.
   - Set **Working directory** to the repository root.

## Checks

```sh
python -m pytest -q
```

## Limits

- The model predicts a typical car on a clear track. It ignores traffic, undercut and overcut effects, and safety cars.
- Every stint starts on new tyres, and the opening lap is predicted like any other lap. That makes the total race time a little optimistic, but it shifts every strategy by the same amount.
- Tree models cannot extrapolate past the tyre ages they were trained on, so `MAX_STINT` caps stint length.
