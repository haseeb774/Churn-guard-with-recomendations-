"""
simulate.py
------------
Simulates "new" or "changed" customers for the demo SaaS dashboard.

Since this is a portfolio project (no live CRM feed), we generate
realistic incoming customer activity by sampling real rows from the
original Telco dataset and lightly perturbing them. This preserves the
real feature correlations the model was trained on (so predictions stay
meaningful) while making the dashboard feel "live" across refreshes.

Two kinds of events are simulated:
    1. New sign-ups       -> sampled row, tenure reset low, new ID
    2. Existing customer  -> sampled row, small random changes to
       changes               MonthlyCharges / tenure / services (mimics
                              a plan change or usage shift)

Both return raw-schema rows (same columns as churn.csv, no Churn label)
ready to be passed into the same transform + predict pipeline used for
the real dataset.
"""

import sys
import uuid
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logging import logging


def _new_customer_id() -> str:
    """Generate a fake but realistically-formatted customer ID."""
    return f"SIM-{uuid.uuid4().hex[:8].upper()}"


def simulate_new_signups(raw_df: pd.DataFrame, n: int = 5, random_state=None) -> pd.DataFrame:
    """
    Simulate brand-new customers signing up.

    Samples n rows from raw_df (real customers) as a profile template,
    then resets tenure-dependent fields to reflect a fresh signup:
        - tenure -> small random value (0-3 months)
        - TotalCharges -> recalculated from tenure * MonthlyCharges
        - new customerID
        - Churn column dropped (unknown — that's what we predict)
    """
    try:
        rng = np.random.default_rng(random_state)
        sample = raw_df.sample(n=n, replace=True, random_state=random_state).copy()

        sample["customerID"] = [_new_customer_id() for _ in range(n)]
        sample["tenure"] = rng.integers(0, 4, size=n)  # brand new: 0-3 months

        # MonthlyCharges: keep close to the sampled profile but add small noise
        noise = rng.normal(loc=0, scale=5, size=n)
        sample["MonthlyCharges"] = (sample["MonthlyCharges"].astype(float) + noise).clip(lower=18.0)

        # TotalCharges for a brand new customer should roughly equal tenure * monthly
        sample["TotalCharges"] = (sample["tenure"] * sample["MonthlyCharges"]).round(2)

        if "Churn" in sample.columns:
            sample = sample.drop(columns=["Churn"])

        sample["_event_type"] = "new_signup"
        sample.reset_index(drop=True, inplace=True)

        logging.info(f"Simulated {n} new signups")
        return sample
    except Exception as e:
        logging.info("error occurred in simulate.simulate_new_signups")
        raise CustomException(e, sys)


def simulate_existing_customer_changes(
    raw_df: pd.DataFrame, n: int = 5, random_state=None
) -> pd.DataFrame:
    """
    Simulate existing customers whose usage/billing changed since last
    snapshot (e.g. added/dropped a service, bill went up, tenure ticked
    up by a month). This is what would normally come from a CRM/billing
    system update.
    """
    try:
        rng = np.random.default_rng(random_state)
        sample = raw_df.sample(n=n, replace=True, random_state=random_state).copy()

        # tenure ticks forward by 1 (one more billing cycle passed)
        sample["tenure"] = sample["tenure"].astype(int) + 1

        # small random bill change (+/- up to 15%), simulates a plan/usage change
        pct_change = rng.uniform(-0.15, 0.15, size=n)
        sample["MonthlyCharges"] = (
            sample["MonthlyCharges"].astype(float) * (1 + pct_change)
        ).clip(lower=18.0).round(2)

        # TotalCharges grows by one more month at the new rate
        sample["TotalCharges"] = (
            pd.to_numeric(sample["TotalCharges"], errors="coerce").fillna(0)
            + sample["MonthlyCharges"]
        ).round(2)

        if "Churn" in sample.columns:
            sample = sample.drop(columns=["Churn"])

        sample["_event_type"] = "usage_update"
        sample.reset_index(drop=True, inplace=True)

        logging.info(f"Simulated {n} existing-customer updates")
        return sample
    except Exception as e:
        logging.info("error occurred in simulate.simulate_existing_customer_changes")
        raise CustomException(e, sys)


def simulate_batch(raw_df: pd.DataFrame, n_new: int = 3, n_updates: int = 3, random_state=None) -> pd.DataFrame:
    """
    Convenience wrapper: generate a mixed batch of new signups + existing
    customer changes in one call, ready to feed into the predict pipeline.
    """
    try:
        new_signups = simulate_new_signups(raw_df, n=n_new, random_state=random_state)
        updates = simulate_existing_customer_changes(raw_df, n=n_updates, random_state=random_state)
        batch = pd.concat([new_signups, updates], ignore_index=True)
        logging.info(f"Simulated batch of {len(batch)} customer events")
        return batch
    except Exception as e:
        logging.info("error occurred in simulate.simulate_batch")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick smoke test against the raw dataset
    raw = pd.read_csv("data/raw/churn.csv")
    batch = simulate_batch(raw, n_new=3, n_updates=3, random_state=42)
    print(batch[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "_event_type"]])