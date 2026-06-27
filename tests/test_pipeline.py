"""
test_pipeline.py
------------------
Unit tests for src/pipeline.py (ChurnPipeline).

These mock out ChurnPredictor and analyze_batch so we're testing the
orchestration logic (row alignment, customerID handling, dropped-row
bookkeeping) in isolation, without depending on the real trained model
artifact being present in the test environment.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline import ChurnPipeline


def make_raw_df(n=3, missing_total_charges_row=None):
    """
    Builds a small valid raw-schema dataframe with n rows.
    If missing_total_charges_row is an int index, that row's TotalCharges
    is set to a blank string to simulate the real dataset's known dirty
    rows -- used to test that dropped rows are handled correctly downstream.
    """
    base = {
        "customerID": [f"ID-{i}" for i in range(n)],
        "gender": ["Female"] * n,
        "SeniorCitizen": [0] * n,
        "Partner": ["Yes"] * n,
        "Dependents": ["No"] * n,
        "tenure": [1 + i for i in range(n)],
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No"] * n,
        "InternetService": ["DSL"] * n,
        "OnlineSecurity": ["No"] * n,
        "OnlineBackup": ["Yes"] * n,
        "DeviceProtection": ["No"] * n,
        "TechSupport": ["No"] * n,
        "StreamingTV": ["No"] * n,
        "StreamingMovies": ["No"] * n,
        "Contract": ["Month-to-month"] * n,
        "PaperlessBilling": ["Yes"] * n,
        "PaymentMethod": ["Electronic check"] * n,
        "MonthlyCharges": [29.85 + i for i in range(n)],
        "TotalCharges": [str(29.85 + i) for i in range(n)],
    }
    df = pd.DataFrame(base)
    if missing_total_charges_row is not None:
        df.loc[missing_total_charges_row, "TotalCharges"] = " "
    return df


@pytest.fixture
def mock_predictor():
    """Patches ChurnPredictor so no real model artifact needs to load."""
    with patch("src.pipeline.ChurnPredictor") as MockPredictor:
        instance = MockPredictor.return_value

        def fake_predict(model_input):
            return pd.DataFrame({
                "churn_probability": [0.5] * len(model_input),
                "churn_prediction": [0] * len(model_input),
            })

        instance.predict.side_effect = fake_predict
        yield MockPredictor


@pytest.fixture
def mock_analyze_batch():
    with patch("src.pipeline.analyze_batch") as mock_fn:
        def fake_analyze(readable_df, probs):
            return [
                {"risk_tier": "Low", "risk_drivers": [], "recommendations": []}
                for _ in range(len(readable_df))
            ]
        mock_fn.side_effect = fake_analyze
        yield mock_fn


class TestChurnPipeline:
    def test_run_returns_one_row_per_input_row(self, mock_predictor, mock_analyze_batch):
        raw = make_raw_df(n=3)
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        result = pipeline.run(raw)
        assert len(result) == 3

    def test_run_preserves_customer_ids(self, mock_predictor, mock_analyze_batch):
        raw = make_raw_df(n=3)
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        result = pipeline.run(raw)
        assert list(result["customerID"]) == ["ID-0", "ID-1", "ID-2"]

    def test_run_generates_placeholder_ids_when_missing(self, mock_predictor, mock_analyze_batch):
        raw = make_raw_df(n=2).drop(columns=["customerID"])
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        result = pipeline.run(raw)
        assert list(result["customerID"]) == ["ROW-0", "ROW-1"]

    def test_run_drops_dirty_row_and_aligns_ids_correctly(self, mock_predictor, mock_analyze_batch):
        """
        Row 1 has a blank TotalCharges and gets dropped inside clean().
        The surviving customerIDs (ID-0, ID-2) must still line up with
        the surviving rows -- this is the exact bug class that's easy to
        introduce when re-aligning dataframes after a dropna().
        """
        raw = make_raw_df(n=3, missing_total_charges_row=1)
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        result = pipeline.run(raw)
        assert len(result) == 2
        assert list(result["customerID"]) == ["ID-0", "ID-2"]

    def test_run_attaches_risk_tier_and_recommendations(self, mock_predictor, mock_analyze_batch):
        raw = make_raw_df(n=2)
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        result = pipeline.run(raw)
        assert "risk_tier" in result.columns
        assert "recommendations" in result.columns
        assert "churn_probability" in result.columns

    def test_run_drops_churn_column_before_scoring_if_present(self, mock_predictor, mock_analyze_batch):
        """
        If raw_df happens to include historical Churn labels (e.g. someone
        passes the original training file straight in), the model input
        must not include Churn as a feature.
        """
        raw = make_raw_df(n=2)
        raw["Churn"] = ["No", "Yes"]
        pipeline = ChurnPipeline(artifact_path="fake/path.pkl")
        # Should not raise, and should still return scored rows
        result = pipeline.run(raw)
        assert len(result) == 2