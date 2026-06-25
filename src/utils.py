"""
utils.py
--------
Shared utilities: logger setup, generic save/load helpers, and
small reusable functions used across the training pipeline.
"""

import os
import sys
import logging
import joblib

from src.exception import CustomException

from src.logging import logging


def save_object(file_path: str, obj) -> None:
    """Serialize and save any Python object using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a previously saved joblib object."""
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(*paths) -> None:
    """Create directories if they do not already exist."""
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)