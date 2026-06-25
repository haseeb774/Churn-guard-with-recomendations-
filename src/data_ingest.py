import pandas as pd
import sys
from src.logging import logging
import os
from src.exception import CustomException
class DataIngest:
    def __init__(self, file_path):
        self.file_path = file_path

    def data_import(self):
        try:
            df = pd.read_csv(self.file_path)
            logging.info("file is load and returned from data_ingest")


            file_path = "data/raw/churn.csv"

            if os.path.exists(file_path):
                pass  # File exists, do nothing
            else:
                # Ensure the directory "data/raw" exists before saving
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
            print(df.head())
            return df    

            
            
        except Exception as e:
            logging.info("error occured in data_ingest")
            raise CustomException(e,sys)
if __name__ == "__main__":
    ingest = DataIngest("data/raw/churn.csv")
    ingest.data_import()