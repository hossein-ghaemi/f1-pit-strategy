"""Project settings. Paths can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PITSTRAT_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "openf1"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

LAPS_FILE = DATA_DIR / "laps.csv"
MODEL_FILE = MODELS_DIR / "lap_model.pkl"

# OpenF1 starts in 2023; 2023-2025 share one set of car rules.
SEASONS = [2023, 2024, 2025]

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Races where fewer laps than this share ran on dry tyres in the dry are
# mixed-weather races, whose lap times say little about tyre wear.
MIN_DRY_SHARE = 0.9

# Laps slower than this multiple of the driver's median lap are treated as outliers.
OUTLIER_LIMIT = 1.07

# Longest stint allowed per compound in the simulator. Tree models cannot
# extrapolate past the tyre ages they saw, so stints are capped.
MAX_STINT = {"SOFT": 25, "MEDIUM": 35, "HARD": 45}
MIN_STINT = 5

# Used when a circuit has no clean pit stops in the data.
DEFAULT_PIT_LOSS = 22.0
