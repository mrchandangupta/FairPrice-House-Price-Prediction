import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fairprice import config  # noqa: E402


@pytest.fixture
def valid_row() -> dict:
    """A realistic Los Angeles district."""
    return {"MedInc": 4.5, "HouseAge": 25, "AveRooms": 5.0, "AveBedrms": 1.0,
            "Population": 1200, "AveOccup": 3.0, "Latitude": 34.05, "Longitude": -118.24}


@pytest.fixture
def small_df(valid_row) -> pd.DataFrame:
    """Tiny dataset with one clean row plus one example of each problem we clean."""
    rows = [
        {**valid_row, config.TARGET: 2.0},
        {**valid_row, config.TARGET: 2.0},                       # duplicate
        {**valid_row, "AveOccup": 5000, config.TARGET: 1.5},     # impossible occupancy
        {**valid_row, "MedInc": 6.0, config.TARGET: 5.00001},    # capped target
        {**valid_row, "MedInc": 3.0, config.TARGET: None},       # missing target
        {**valid_row, "HouseAge": None, config.TARGET: 1.8},     # missing feature (kept, imputed later)
    ]
    return pd.DataFrame(rows)
