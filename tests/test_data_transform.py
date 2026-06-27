"""
test_data_transform.py
------------------------
Unit tests for src/data_transform.py (Datatransform).

These tests build small synthetic dataframes that match the real Telco
schema (19 raw columns) so we don't depend on the actual data/raw/churn.csv
file being present in the test environment.
"""

import pandas as pd
import pytest

from src.data_transform import Datatransform


def make_row(**overrides):
    """
    Returns a single valid raw-schema customer row as a dict.
    Override any field via kwargs, e.g. make_row(TotalCharges=" ")
    """
    row = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def make_df(rows):
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# clean()
# ----------------------------------------------------------------------

class TestClean:
    def test_drops_rows_with_blank_total_charges(self):
        """
        The real dataset has 11 rows where TotalCharges is a blank string
        ' ' instead of a number (new customers with 0 tenure). clean()
        must coerce to numeric and drop the resulting NaNs.
        """
        df = make_df([
            make_row(customerID="1", TotalCharges="29.85"),
            make_row(customerID="2", TotalCharges=" "),  # blank -> NaN -> dropped
        ])
        result = Datatransform(df).clean()
        assert len(result) == 1
        assert result["TotalCharges"].iloc[0] == 29.85

    def test_total_charges_is_numeric_after_clean(self):
        df = make_df([make_row(TotalCharges="1889.5")])
        result = Datatransform(df).clean()
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_gender_is_mapped_to_binary(self):
        df = make_df([
            make_row(customerID="1", gender="Female"),
            make_row(customerID="2", gender="Male"),
        ])
        result = Datatransform(df).clean()
        assert set(result["gender"].unique()) <= {0, 1}
        assert result.loc[result.index[0], "gender"] == 0
        assert result.loc[result.index[1], "gender"] == 1

    def test_customer_id_is_dropped(self):
        df = make_df([make_row()])
        result = Datatransform(df).clean()
        assert "customerID" not in result.columns

    def test_churn_is_mapped_when_present(self):
        df = make_df([
            make_row(customerID="1", Churn="Yes"),
            make_row(customerID="2", Churn="No"),
        ])
        result = Datatransform(df).clean()
        assert result.loc[result.index[0], "Churn"] == 1
        assert result.loc[result.index[1], "Churn"] == 0

    def test_clean_works_without_churn_column(self):
        """At inference time the raw input has no Churn column at all."""
        df = make_df([make_row()])
        df = df.drop(columns=["Churn"])
        result = Datatransform(df).clean()
        # should not raise, and Churn should simply not appear
        assert "Churn" not in result.columns


# ----------------------------------------------------------------------
# features()
# ----------------------------------------------------------------------

class TestFeatures:
    def test_new_high_risk_flag_for_new_month_to_month_customer(self):
        df = make_df([make_row(tenure=2, Contract="Month-to-month")])
        result = Datatransform(df).features()
        assert result["NewHighRisk"].iloc[0] == 1

    def test_new_high_risk_flag_false_for_long_tenure_customer(self):
        df = make_df([make_row(tenure=48, Contract="Two year")])
        result = Datatransform(df).features()
        assert result["NewHighRisk"].iloc[0] == 0

    def test_num_services_counts_subscribed_addons(self):
        df = make_df([make_row(
            OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="No",
            TechSupport="No", StreamingTV="No", StreamingMovies="No",
        )])
        result = Datatransform(df).features()
        assert result["NumServices"].iloc[0] == 2

    def test_no_protection_flag(self):
        df = make_df([make_row(
            OnlineSecurity="No", DeviceProtection="No", TechSupport="No",
        )])
        result = Datatransform(df).features()
        assert result["NoProtection"].iloc[0] == 1

    def test_clv_proxy_is_monthly_charges_times_tenure(self):
        df = make_df([make_row(MonthlyCharges=50.0, tenure=10)])
        result = Datatransform(df).features()
        assert result["CLV_proxy"].iloc[0] == pytest.approx(500.0)


# ----------------------------------------------------------------------
# encode()
# ----------------------------------------------------------------------

class TestEncode:
    def test_encode_produces_no_nulls(self):
        df = make_df([
            make_row(customerID="1", InternetService="DSL", PaymentMethod="Electronic check"),
            make_row(customerID="2", InternetService="Fiber optic", PaymentMethod="Mailed check"),
            make_row(customerID="3", InternetService="No", PaymentMethod="Bank transfer (automatic)"),
        ])
        result = Datatransform(df).encode()
        assert result.isnull().sum().sum() == 0

    def test_encode_output_is_fully_numeric(self):
        df = make_df([make_row()])
        result = Datatransform(df).encode()
        non_numeric = result.select_dtypes(exclude=["number"]).columns.tolist()
        assert non_numeric == [], f"Found non-numeric columns after encode: {non_numeric}"

    def test_encode_handles_unseen_internet_service_category(self):
        """
        Train-time the encoder only ever sees DSL / Fiber optic / No.
        At inference time a malformed or future row could carry a category
        the encoder was never fit on. handle_unknown='ignore' should mean
        this does not raise -- it should just produce all-zero one-hot
        columns for that row instead of crashing the pipeline.
        """
        df = make_df([make_row(InternetService="SatelliteBeam9000")])
        # should not raise
        result = Datatransform(df).encode()
        assert len(result) == 1

    def test_encode_drops_contract_after_mapping(self):
        df = make_df([make_row()])
        result = Datatransform(df).encode()
        assert "Contract" not in result.columns