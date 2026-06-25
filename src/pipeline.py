"""
pipeline.py
------------
The single orchestration layer the dashboard (and anything else) calls.

Takes raw-schema customer rows (same columns as data/raw/churn.csv, no
Churn column required) and returns a fully scored + explained dataframe:

    customerID | ...original columns... | churn_probability |
    churn_prediction | risk_tier | risk_drivers | recommendations

Internally it:
    1. Runs Datatransform.features()  -> human-readable engineered df
       (used for recommendation reasoning)
    2. Runs Datatransform.encode()    -> fully encoded df
       (used for the model, matches training-time feature schema)
    3. Runs ChurnPredictor.predict()  -> churn_probability / churn_prediction
    4. Runs recommend.analyze_batch() -> risk_tier / drivers / recommendations
    5. Stitches everything back together, re-attaching customerID
"""

import sys
import pandas as pd

from src.exception import CustomException
from src.logging import logging
from src.data_transform import Datatransform
from src.predict import ChurnPredictor
from src.recommend import analyze_batch


class ChurnPipeline:
    def __init__(self, artifact_path: str = "artifacts/best_model.pkl"):
        try:
            self.predictor = ChurnPredictor(artifact_path=artifact_path)
        except Exception as e:
            raise CustomException(e, sys)

    def run(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        raw_df: raw-schema customer rows (e.g. loaded from churn.csv, or
                produced by simulate.simulate_batch()). Must NOT require
                a Churn column.
        """
        try:
            raw_df = raw_df.reset_index(drop=True)

            # Preserve customer identifiers before the transform pipeline
            # drops them (clean() removes customerID as a training feature).
            if "customerID" in raw_df.columns:
                customer_ids = raw_df["customerID"].copy()
            else:
                customer_ids = pd.Series(
                    [f"ROW-{i}" for i in range(len(raw_df))]
                )

            transformer = Datatransform(data=raw_df)

            # Human-readable engineered features (pre one-hot encoding)
            # used for explainable recommendations.
            readable_df = transformer.features()

            # Fully encoded features used for the model itself.
            encoded_df = transformer.encode()

            # Some rows may get dropped inside clean() (e.g. bad TotalCharges
            # values, dropna). encoded_df/readable_df indices reflect the
            # surviving rows -- align customer_ids the same way.
            surviving_idx = encoded_df.index
            customer_ids = customer_ids.loc[surviving_idx].reset_index(drop=True)
            readable_df = readable_df.loc[surviving_idx].reset_index(drop=True)
            encoded_df = encoded_df.reset_index(drop=True)

            dropped = len(raw_df) - len(encoded_df)
            if dropped > 0:
                logging.info(
                    f"Pipeline: {dropped} row(s) dropped during cleaning "
                    f"(missing/invalid values)"
                )

            # Drop Churn column if present in encoded df (only happens if
            # raw_df included historical labels, e.g. existing churn.csv)
            model_input = encoded_df.drop(columns=["Churn"], errors="ignore")

            scored = self.predictor.predict(model_input)

            analyses = analyze_batch(
                readable_df, scored["churn_probability"].values
            )

            final = scored.copy()
            final.insert(0, "customerID", customer_ids.values)
            final["risk_tier"] = [a["risk_tier"] for a in analyses]
            final["risk_drivers"] = [a["risk_drivers"] for a in analyses]
            final["recommendations"] = [a["recommendations"] for a in analyses]

            logging.info(f"Pipeline scored {len(final)} customers")
            # final.to_csv("artifacts/prediction.csv")
            return final
        except Exception as e:
            logging.info("error occurred in pipeline.run")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Smoke test against the raw dataset
    raw = pd.read_csv("data/raw/churn.csv")
    pipeline = ChurnPipeline(artifact_path="artifacts/best_model.pkl")
    result = pipeline.run(raw.drop(columns=["Churn"], errors="ignore"))
    print(result.head())