"""
recommend.py
-------------
Rule-based recommendation engine for the churn prediction SaaS.

Takes a single customer row (post Datatransform.features(), i.e. BEFORE
encoding — so it still has human-readable categorical values like
Contract == "Month-to-month") plus the model's churn probability, and
returns:
    - a risk tier (Low / Medium / High / Critical)
    - a list of identified risk drivers (human-readable reasons)
    - a list of recommended retention actions

This is intentionally rule-based (no extra model) so it's fully
explainable to a business owner looking at the dashboard — every
recommendation can be traced back to a specific customer attribute.
"""

import sys
from src.exception import CustomException
from src.logging import logging


# ----------------------------------------------------------------------
# Risk tier thresholds (based on churn probability from the model)
# ----------------------------------------------------------------------
RISK_TIERS = [
    (0.75, "Critical"),
    (0.50, "High"),
    (0.25, "Medium"),
    (0.0, "Low"),
]


def get_risk_tier(probability: float) -> str:
    """Map a churn probability to a risk tier label."""
    for threshold, label in RISK_TIERS:
        if probability >= threshold:
            return label
    return "Low"


# ----------------------------------------------------------------------
# Risk driver detection + matching recommendation
# Each rule is (condition_fn, driver_text, recommendation_text)
# ----------------------------------------------------------------------
def _build_rules():
    return [
        (
            lambda r: r.get("Contract") == "Month-to-month",
            "On a month-to-month contract (no lock-in)",
            "Offer a discounted 1-year contract upgrade to increase switching cost",
        ),
        (
            lambda r: r.get("tenure", 999) < 12,
            "New customer (tenure under 12 months)",
            "Enroll in a new-customer onboarding/check-in call within first 90 days",
        ),
        (
            lambda r: r.get("PaymentMethod") == "Electronic check",
            "Pays via electronic check (historically highest-risk payment method)",
            "Incentivize switch to autopay/credit card with a small bill credit",
        ),
        (
            lambda r: r.get("OnlineSecurity") == "No" and r.get("TechSupport") == "No",
            "No online security or tech support add-ons",
            "Offer a free 1-month trial of the Security + Support bundle",
        ),
        (
            lambda r: r.get("NumServices", 0) <= 1,
            "Very low service adoption (0-1 add-on services)",
            "Cross-sell a starter bundle (e.g. backup + streaming) at a discount",
        ),
        (
            lambda r: r.get("ChargeGap", 0) > 10,
            "Recent monthly charge spike vs. historical average spend",
            "Proactively explain the bill increase or offer a loyalty discount",
        ),
        (
            lambda r: r.get("MonthlyCharges", 0) > 80 and r.get("Contract") == "Month-to-month",
            "High monthly spend with no contract commitment",
            "Assign to a retention specialist for a personalized save offer",
        ),
        (
            lambda r: r.get("SeniorCitizen") == 1,
            "Senior citizen segment (often values simplicity/support)",
            "Offer a simplified plan or dedicated support line",
        ),
        (
            lambda r: r.get("StableHousehold", 0) == 0 and r.get("FamilyScore", 0) == 0,
            "Single-person household (no partner/dependents) — lower switching friction",
            "Highlight individual-focused value props (flexibility, mobile add-ons)",
        ),
        (
            lambda r: r.get("InternetService") == "Fiber optic" and r.get("TechSupport") == "No",
            "Fiber customer without tech support — higher expectations, no safety net",
            "Bundle in free tech support for fiber customers in their first 6 months",
        ),
    ]


_RULES = _build_rules()
    

def analyze_customer(row: dict, churn_probability: float) -> dict:
    """
    Analyze a single customer (as a dict of feature_name -> value, taken
    from the pre-encoding feature dataframe) plus the model's predicted
    churn probability.

    Returns:
        {
            "risk_tier": str,
            "churn_probability": float,
            "risk_drivers": [str, ...],
            "recommendations": [str, ...],
        }
    """
    try:
        risk_tier = get_risk_tier(churn_probability)

        drivers = []
        recommendations = []

        for condition_fn, driver_text, rec_text in _RULES:
            try:
                if condition_fn(row):
                    drivers.append(driver_text)
                    recommendations.append(rec_text)
            except Exception:
                # Be defensive: missing column for this row shouldn't crash analysis
                continue

        if not drivers:
            drivers.append("No major individual risk drivers detected")
            recommendations.append("Continue standard engagement; monitor at next billing cycle")

        return {
            "risk_tier": risk_tier,
            "churn_probability": round(float(churn_probability), 4),
            "risk_drivers": drivers,
            "recommendations": recommendations,
        }
    except Exception as e:
        logging.info("error occurred in recommend.analyze_customer")
        raise CustomException(e, sys)


def analyze_batch(rows_df, probabilities) -> list:
    """
    Vectorized-friendly wrapper: run analyze_customer for every row in a
    dataframe alongside a matching array/series of churn probabilities.

    rows_df: pandas DataFrame (pre-encoding feature frame)
    probabilities: array-like of same length, churn probability per row

    Returns a list of dicts (same order as rows_df), each containing the
    analyze_customer() output plus the original row index.
    """
    try:
        results = []
        for (idx, row), prob in zip(rows_df.iterrows(), probabilities):
            analysis = analyze_customer(row.to_dict(), prob)
            analysis["index"] = idx
            results.append(analysis)
        return results
    except Exception as e:
        logging.info("error occurred in recommend.analyze_batch")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick manual smoke test with a synthetic high-risk customer
    sample_customer = {
        "Contract": "Month-to-month",
        "tenure": 3,
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "NumServices": 1,
        "ChargeGap": 15.0,
        "MonthlyCharges": 95.0,
        "SeniorCitizen": 0,
        "StableHousehold": 0,
        "FamilyScore": 0,
        "InternetService": "Fiber optic",
    }
    result = analyze_customer(sample_customer, churn_probability=0.82)
    import json
    print(json.dumps(result, indent=2))