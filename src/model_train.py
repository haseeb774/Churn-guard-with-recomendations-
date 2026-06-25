"""
model_train.py
---------------
Production training pipeline for the Telco churn prediction system.

Pipeline stages:
    1. Load processed data
    2. Stratified Train(64%) / Val(16%) / Test(20%) split
    3. Train 4 models: LogisticRegression, RandomForest, XGBoost, CatBoost
    4. Tune RandomForest / XGBoost / CatBoost with Optuna (LogReg = baseline, no tuning)
    5. Select winner by ROC-AUC
    6. Search threshold range 0.10 -> 0.90 to maximize F1
    7. Log everything (metrics, params, threshold) to MLflow
    8. Save final artifact (model + threshold + feature list) via joblib

Run:
    python -m src.model_train
"""
import xgboost
import os
from src.utils import save_object
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import mlflow.sklearn
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.utils import create_directories
from src.data_transform import Datatransform
from src.logging import logging

# Reduce Optuna's default verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    def __init__(
        self,
        data_path: str = "data/processed/processed_churn.csv",
        target_column: str = "Churn",
        artifact_path: str = "artifacts/best_model.pkl",
        mlflow_experiment_name: str = "telco_churn_prediction",
        n_trials: int = 30,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.artifact_path = artifact_path
        self.mlflow_experiment_name = mlflow_experiment_name
        self.n_trials = n_trials
        self.random_state = random_state

        # self.transformer = Datatransform(data=self.data_path)    

        # Populated during run()
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.feature_names = None

        mlflow.set_experiment(self.mlflow_experiment_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            self.feature_names = [
                c for c in df.columns
                if c != self.target_column
            ]
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(
        self,
        df,
        random_state=42
    ):
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            stratify=y_train_full,
            random_state=random_state
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

    # ------------------------------------------------------------------
    # Threshold optimization
    # ------------------------------------------------------------------
    def find_best_threshold(self, model, X_val, y_val):
        """
        Search thresholds from 0.10 to 0.90 (step 0.01) and return the
        threshold that maximizes F1 score on the validation set.
        """
        try:
            y_proba = model.predict_proba(X_val)[:, 1]

            best_threshold = 0.5
            best_f1 = -1.0

            for threshold in np.arange(0.10, 0.91, 0.01):
                y_pred = (y_proba >= threshold).astype(int)
                score = f1_score(y_val, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_threshold = round(float(threshold), 2)

            logging.info(
                f"Best threshold: {best_threshold} (F1 on val: {best_f1:.4f})"
            )
            return best_threshold, best_f1
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, model, X, y, threshold: float = 0.5):
        """Compute accuracy, precision, recall, f1, roc_auc at a given threshold."""
        try:
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_proba),
            }
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Logistic Regression (baseline, no Optuna)
    # ------------------------------------------------------------------
    def train_logistic(self):
        try:
            with mlflow.start_run(run_name="LogisticRegression", nested=True):
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.random_state
                    ))
                ])
                print(type(self.X_train))
                print(type(self.y_train))

                print(self.X_train.shape if hasattr(self.X_train, "shape") else self.X_train)
                print(self.y_train.shape if hasattr(self.y_train, "shape") else self.y_train)
                model.fit(self.X_train, self.y_train)

                best_threshold, _ = self.find_best_threshold(
                    model, self.X_val, self.y_val
                )
                val_metrics = self.evaluate(model, self.X_val, self.y_val, best_threshold)

                mlflow.log_params(model.get_params())
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_metrics(val_metrics)
                mlflow.sklearn.log_model(model, "model")

                logging.info(f"LogisticRegression val metrics: {val_metrics}")

                return {
                    "model": model,
                    "threshold": best_threshold,
                    "metrics": val_metrics,
                    "params": model.get_params(),
                }
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Random Forest (Optuna tuned)
    # ------------------------------------------------------------------
    def train_rf(self):
        try:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "class_weight": "balanced",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "max_features": trial.suggest_categorical("max_features",["sqrt", "log2"])
                }
                model = RandomForestClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_proba = model.predict_proba(self.X_val)[:, 1]
                return roc_auc_score(self.y_val, y_proba)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params.update(
                {"class_weight": "balanced", "random_state": self.random_state, "n_jobs": -1}
            )

            with mlflow.start_run(run_name="RandomForest", nested=True):
                model = RandomForestClassifier(**best_params)
                model.fit(self.X_train, self.y_train)

                best_threshold, _ = self.find_best_threshold(
                    model, self.X_val, self.y_val
                )
                val_metrics = self.evaluate(model, self.X_val, self.y_val, best_threshold)

                mlflow.log_params(best_params)
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_metrics(val_metrics)
                mlflow.sklearn.log_model(model, "model")

                logging.info(f"RandomForest val metrics: {val_metrics}")

                return {
                    "model": model,
                    "threshold": best_threshold,
                    "metrics": val_metrics,
                    "params": best_params,
                }
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # XGBoost (Optuna tuned)
    # ------------------------------------------------------------------
    def train_xgb(self):
        try:
            scale_pos_weight = (
                self.y_train.value_counts()[0] / self.y_train.value_counts()[1]
            )

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha",0.0,5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda",0.0,10.0),
                    "scale_pos_weight": scale_pos_weight,
                    "random_state": self.random_state,
                    "eval_metric": "auc",
                    "n_jobs": -1,
                }
                model = XGBClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_proba = model.predict_proba(self.X_val)[:, 1]
                return roc_auc_score(self.y_val, y_proba)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params.update(
                {
                    "scale_pos_weight": scale_pos_weight,
                    "random_state": self.random_state,
                    "eval_metric": "auc",
                    "n_jobs": -1,
                }
            )

            with mlflow.start_run(run_name="XGBoost", nested=True):
                model = XGBClassifier(**best_params)
                model.fit(self.X_train, self.y_train)

                best_threshold, _ = self.find_best_threshold(
                    model, self.X_val, self.y_val
                )
                val_metrics = self.evaluate(model, self.X_val, self.y_val, best_threshold)

                mlflow.log_params(best_params)
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_metrics(val_metrics)
                mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

                logging.info(f"XGBoost val metrics: {val_metrics}")

                return {
                    "model": model,
                    "threshold": best_threshold,
                    "metrics": val_metrics,
                    "params": best_params,
                }
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # CatBoost (Optuna tuned)
    # ------------------------------------------------------------------
    def train_cat(self):
        try:
            def objective(trial):
                params = {
                    "depth": trial.suggest_int("depth", 3, 10),
                    "iterations": trial.suggest_int("iterations", 100, 600, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "auto_class_weights": "Balanced",
                    "random_state": self.random_state,
                    "verbose": False,
                }
                model = CatBoostClassifier(**params)
                model.fit(self.X_train,self.y_train,eval_set=(self.X_val,self.y_val),verbose=False)
                y_proba = model.predict_proba(self.X_val)[:, 1]
                return roc_auc_score(self.y_val, y_proba)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params.update(
                {
                    "auto_class_weights": "Balanced",
                    "random_state": self.random_state,
                    "verbose": False,
                }
            )

            with mlflow.start_run(run_name="CatBoost", nested=True):
                model = CatBoostClassifier(**best_params)
                model.fit(self.X_train,self.y_train,eval_set=(self.X_val, self.y_val),verbose=False)

                best_threshold, _ = self.find_best_threshold(
                    model, self.X_val, self.y_val
                )
                val_metrics = self.evaluate(model, self.X_val, self.y_val, best_threshold)

                mlflow.log_params(best_params)
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_metrics(val_metrics)
                mlflow.catboost.log_model(model, "model")

                logging.info(f"CatBoost val metrics: {val_metrics}")

                return {
                    "model": model,
                    "threshold": best_threshold,
                    "metrics": val_metrics,
                    "params": best_params,
                }
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Compare models & pick winner
    # ------------------------------------------------------------------
    def compare_models(self, results: dict):
        try:
            winner_name = max(
                results, key=lambda name: results[name]["metrics"]["roc_auc"]
            )
            logging.info(
                f"Winner: {winner_name} "
                f"(ROC-AUC: {results[winner_name]['metrics']['roc_auc']:.4f})"
            )
            winner = results[winner_name]
            if winner_name == "XGBoost":
                importance = pd.DataFrame({
                    "feature": self.X_train.columns,
                    "importance": winner["model"].feature_importances_
                })

                importance.sort_values(
                    "importance",
                    ascending=False
                ).to_csv(
                    "artifacts/feature_importance.csv",
                    index=False
                )
            return winner_name, results[winner_name]
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Save best model artifact
    # ------------------------------------------------------------------
    def save_best_model(self, best_model, best_threshold, features: list):
        try:
            create_directories(os.path.dirname(self.artifact_path))
            artifact = {
                "model": best_model,
                "threshold": best_threshold,
                "features": features,
            }

            save_object(
                self.artifact_path,
                artifact
            )
            logging.info(f"Best model artifact saved to: {self.artifact_path}")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Full pipeline runner
    # ------------------------------------------------------------------
    def run(self):
        try:
            logging.info("=== Stage 1: Load processed data ===")
            df = self.load_data()

            logging.info("=== Stage 2: Train/Val/Test split (64/16/20) ===")
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test
            ) = self.split_data(df)
            print("X_train:", self.X_train.shape)
            print("X_val:", self.X_val.shape)
            print("X_test:", self.X_test.shape)

            print("y_train:", self.y_train.shape)
            print("y_val:", self.y_val.shape)
            print("y_test:", self.y_test.shape)
            logging.info("=== Stage 3 & 4: Train + tune models ===")
            with mlflow.start_run(run_name="churn_training_pipeline"):
                results = {
                    "LogisticRegression": self.train_logistic(),
                    "RandomForest": self.train_rf(),
                    "XGBoost": self.train_xgb(),
                    "CatBoost": self.train_cat(),
                }

                logging.info("=== Stage 5: Select winner by ROC-AUC (validation) ===")
                winner_name, winner = self.compare_models(results)

                logging.info("=== Final evaluation on held-out TEST set ===")
                test_metrics = self.evaluate(
                    winner["model"], self.X_test, self.y_test, winner["threshold"]
                )
                logging.info(f"{winner_name} TEST metrics: {test_metrics}")

                mlflow.log_param("winning_model", winner_name)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                logging.info("=== Stage 8: Save best model artifact ===")
                self.save_best_model(
                    best_model=winner["model"],
                    best_threshold=winner["threshold"],
                    features=self.feature_names,
                )

            summary = {
                "winner": winner_name,
                "val_metrics": winner["metrics"],
                "test_metrics": test_metrics,
                "threshold": winner["threshold"],
                "all_results": {
                    name: res["metrics"] for name, res in results.items()
                },
                "threshold": winner["threshold"],
            }
            return summary
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer(
        data_path="data/processed/processed_churn.csv",
        target_column="Churn",
        artifact_path="artifacts/best_model.pkl",
        n_trials=30,
    )
    summary = trainer.run()

    with open(
        "artifacts/training_summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
            default=str
        )
    print("\n================ TRAINING SUMMARY ================")
    for k, v in summary.items():
        print(f"{k}: {v}")