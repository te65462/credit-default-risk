"""Shared test setup: import the deployed code from space/ and load the 6,000 test clients."""
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPACE = REPO / "space"
sys.path.insert(0, str(SPACE))  # import predictor.py exactly as it is deployed

import predictor  # noqa: E402


def inputs_of(row):
    """The 23 model inputs of one test client (drops ID, DEFAULT and MODULE8_RISK)."""
    return {name: row[name] for name in predictor.NAMES}


@pytest.fixture(scope="session")
def test_clients():
    """The 6,000 test clients of the fixed Module 3 split: raw values, outcome and Module 8 risk."""
    with open(REPO / "tests" / "data" / "test_clients.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="session")
def deployed_risk(test_clients):
    """Risk scores from the deployed code (predictor.py and the exported model) for every test client."""
    return [predictor.predict(inputs_of(row))["risk"] for row in test_clients]


@pytest.fixture
def example():
    """A valid client (the first worked example on the web page), as a fresh dict."""
    return dict(predictor.INFO["examples"][0]["values"])
