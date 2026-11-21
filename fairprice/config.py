"""Central configuration: paths, column names, validation rules and model settings.

Keeping every "magic number" here means the rest of the code reads as logic only,
and any setting can be changed (and defended) in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "california_housing.csv"
CLEAN_DATA_PATH = ROOT_DIR / "data" / "processed" / "housing_clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
MODEL_META_PATH = ROOT_DIR / "models" / "model_meta.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
LEADERBOARD_PATH = ROOT_DIR / "reports" / "leaderboard.csv"

# ---------------------------------------------------------------- columns
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"          # median house value, in $100,000s (1990 dollars)
PRICE_UNIT = 100_000            # multiply the target by this to get dollars

# ---------------------------------------------------------------- validation rules
# (min, max) a value must fall in to be physically/geographically plausible.
# Rows outside these ranges are treated as data errors.
VALID_RANGES = {
    "MedInc": (0.0, 20.0),        # in $10,000s
    "HouseAge": (0.0, 100.0),
    "AveRooms": (0.5, 200.0),
    "AveBedrms": (0.1, 50.0),
    "Population": (1.0, 50_000.0),
    "AveOccup": (0.5, 1_500.0),
    "Latitude": (32.0, 42.5),      # California bounding box
    "Longitude": (-124.6, -114.0),
    TARGET: (0.1, 10.0),
}
TARGET_CAP = 5.0                # the source data caps house values at 5.00001 ($500k)

# ---------------------------------------------------------------- feature engineering
# Major Californian cities: distance to the nearest one is a strong price signal.
CITY_COORDS = {
    "Los Angeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89),
    "Sacramento": (38.58, -121.49),
}
# AveBedrms is replaced by BedroomRatio (it is 0.85 correlated with AveRooms).
DROP_AFTER_ENGINEERING = ["AveBedrms"]
CORRELATION_THRESHOLD = 0.8     # pairs above this are flagged as redundant

# ---------------------------------------------------------------- outliers
CLIP_QUANTILES = (0.01, 0.99)   # winsorize features at these train-set percentiles

# ---------------------------------------------------------------- training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 20              # RandomizedSearchCV iterations (reduced in --fast mode)
N_ITER_SEARCH_FAST = 4
# One ANN fit takes ~30-45 s on CPU (vs < 1 s for trees), so it gets a smaller search budget.
ANN_N_ITER_SEARCH = 6
ANN_N_ITER_SEARCH_FAST = 2
ANN_MODEL_PATH = ROOT_DIR / "models" / "ann_model.pkl"
