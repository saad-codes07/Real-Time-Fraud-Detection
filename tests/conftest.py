"""
conftest.py - shared test setup.

Tests must not depend on your real trained model or the 150 MB dataset
(neither exists on GitHub). So we train a tiny model on made-up data and
save it to a temporary folder before the API starts.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

# Same feature names and order as the real model.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
THRESHOLD = 0.5


def make_tiny_model(model_dir):
    """Train a small model where 'fraud' simply means a large V1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, len(FEATURES))), columns=FEATURES)
    X["Amount"] = rng.uniform(0, 300, size=600)
    X["Hour"] = rng.integers(0, 24, size=600)
    y = (X["V1"] > 1.0).astype(int)

    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(X, y)
    model.save_model(model_dir / "model.json")
    meta = {"threshold": THRESHOLD, "features": FEATURES}
    (model_dir / "metadata.json").write_text(json.dumps(meta))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """A test client for the API, started with the tiny model."""
    model_dir = tmp_path_factory.mktemp("model")
    make_tiny_model(model_dir)

    # api/main.py reads MODEL_DIR when it is imported, so set it first.
    os.environ["MODEL_DIR"] = str(model_dir)
    from api.main import app
    from fastapi.testclient import TestClient

    # "with" runs the API's startup code, which loads the model.
    with TestClient(app) as test_client:
        yield test_client