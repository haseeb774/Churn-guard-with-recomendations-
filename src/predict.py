"""
predict.py
----------
Loads the saved production artifact (model + threshold + feature list)
and runs inference on new customer data.

Usage:
    python -m src.predict --input data/new_customers.csv --output predictions.csv
"""

import sys
import argparse

import pandas as pd

from src.exception import CustomException
from src.utils import  load_object
from src.logging import logging


class ChurnPredictor:
    def __init__(self, artifact_path: str = "artifacts/best_model.pkl"):
        try:
            artifact = load_object(artifact_path)
            self.model = artifact["model"]
            self.threshold = artifact["threshold"]
            self.features = artifact["features"]
            logging.info(
                f"Loaded model artifact. Threshold={self.threshold}, "
                f"#features={len(self.features)}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the incoming dataframe has exactly the expected columns,
        in the same order the model was trained on."""
        try:
            missing = set(self.features) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            return df[self.features]
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            X = self._validate_columns(df)
            churn_proba = self.model.predict_proba(X)[:, 1]
            churn_pred = (churn_proba >= self.threshold).astype(int)

            result = df.copy()
            result["churn_probability"] = churn_proba
            result["churn_prediction"] = churn_pred
            return result
        except Exception as e:
            raise CustomException(e, sys)


def main():
    parser = argparse.ArgumentParser(description="Run churn predictions on new data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output", default="predictions.csv", help="Path to write predictions CSV."
    )
    parser.add_argument(
        "--artifact", default="artifacts/best_model.pkl", help="Path to model artifact."
    )
    args = parser.parse_args()

    try:
        predictor = ChurnPredictor(artifact_path=args.artifact)
        df = pd.read_csv(args.input)
        result = predictor.predict(df)
        result.to_csv(args.output, index=False)
        logging.info(f"Predictions written to: {args.output}")
        print(f"Predictions saved to {args.output}")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()