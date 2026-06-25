import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder    
import sys
from src.logging import logging
from src.exception import CustomException
class Datatransform:
    def __init__(self, data):
        self.data = data

    def clean(self):

        if isinstance(self.data, pd.DataFrame):
            df = self.data.copy()
        else:
            df = pd.read_csv(self.data)

        # target (only present during training; absent at inference time)
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({
                "No":0,
                "Yes":1
            })

        # gender
        df["gender"] = df["gender"].map({
            "Female":0,
            "Male":1
        })

        # total charges
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        df.dropna(inplace=True)

        # ids
        df.drop(
            columns=["customerID"],
            inplace=True,
            errors="ignore"
        )

        return df
    

    def features(self):
        try:
            df = self.clean()
            """ Customer Profile"""
            df['FamilyScore'] = (
                (df['Partner']=='Yes').astype(int)
                +
                (df['Dependents']=='Yes').astype(int)
            )
            df['StableHousehold'] = (
                (df['Partner']=='Yes')
                &
                (df['Dependents']=='Yes')   
            ).astype(int)

            """Subscription Profile"""

            df['TenureGroup'] = pd.cut(
                df['tenure'],
                bins=[0,12,24,48,72],
                labels=[0,1,2,3]
            ).astype(int)
            df['CLV_proxy'] = (
                df['MonthlyCharges']
                *
                df['tenure']
            )        
            contract_map = {
                'Month-to-month':0,
                'One year':1,
                'Two year':2
            }

            df['ContractLength'] = df['Contract'].map(contract_map)

            """Usage/Engagement"""
            service_cols = [
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies'
            ]

            service_map = {
                "Yes": 1,
                "No": 0,
                "No internet service": 0
            }

            temp_services = df[service_cols].apply(
                lambda col: col.map(service_map)
            )

            df["NumServices"] = temp_services.sum(axis=1)
            df['ServiceAdoptionRate'] = (
                df['NumServices'] / 6
            )
            df['SecurityBundle'] = (
                (df['OnlineSecurity']=='Yes') &
                (df['TechSupport']=='Yes')
            ).astype(int)
            df['EntertainmentBundle'] = (
                (df['StreamingTV']=='Yes') &
                (df['StreamingMovies']=='Yes')
            ).astype(int)

            
            df['ServiceDensity'] = (
                df['NumServices']
                /
                (df['tenure'] + 1)
            )
            df['NoProtection'] = (
                (df['OnlineSecurity'] == 'No')
                &
                (df['DeviceProtection'] == 'No')
                &
                (df['TechSupport'] == 'No')
            ).astype(int)   
            df['NewHighRisk'] = (
                (df['tenure'] < 12)
                &
                (df['Contract'] == 'Month-to-month')
            ).astype(int)

            """Billing & Payment"""
            df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

            df['ChargeGap'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
            df['AutoPay'] = (
                df['PaymentMethod']
                .str.contains('automatic')
            ).astype(int)
            df['ContractRisk'] = (
                df['Contract'] == 'Month-to-month'
            ).astype(int)

            df['SpendPerService'] = (
                df['MonthlyCharges']
                /
                (df['NumServices'] + 1)
            )
            df['RiskScore'] = 0

            df['RiskScore'] += (df['Contract']=='Month-to-month')*3
            df['RiskScore'] += (df['tenure']<12)*2
            df['RiskScore'] += (df['PaymentMethod']=='Electronic check')*2
            df['RiskScore'] += (df['TechSupport']=='No')*1
            df['RiskScore'] += (df['OnlineSecurity']=='No')*1
            return df
            
        except Exception as e:
            logging.info("error occured in datatransform")
            raise CustomException(e,sys)
    def encode(self):
        try:
            df = self.features()

            # ==================
            # Binary Columns
            # ==================

            binary_cols = [
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling"
            ]

            for col in binary_cols:
                df[col] = df[col].map({
                    "Yes": 1,
                    "No": 0
                })

            # ==================
            # Service Columns
            # ==================

            service_cols = [
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies"
            ]

            service_mapping = {
                "Yes": 1,
                "No": 0,
                "No internet service": 0,
                "No phone service": 0
            }

            for col in service_cols:
                df[col] = df[col].map(service_mapping)

            # ==================
            # Contract Encoding
            # ==================

            contract_map = {
                "Month-to-month": 0,
                "One year": 1,
                "Two year": 2
            }

            if "Contract" in df.columns:
                df["Contract"] = df["Contract"].map(contract_map)

            # ==================
            # One-Hot Encoding
            # ==================

            categorical_cols = [
                "InternetService",
                "PaymentMethod"
            ]

            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop="first"
            )

            encoded = encoder.fit_transform(
                df[categorical_cols]
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(
                    categorical_cols
                ),
                index=df.index
            )

            df.drop(
                columns=categorical_cols,
                inplace=True
            )

            df = pd.concat(
                [df, encoded_df],
                axis=1
            )

            # ==================
            # Drop Redundant Columns
            # ==================

            redundant_cols = [
                "Contract"
            ]

            df.drop(
                columns=redundant_cols,
                errors="ignore",
                inplace=True
            )

            # ==================
            # Final Cleanup
            # ==================

            df = df.dropna()

            # convert bool columns to int
            bool_cols = df.select_dtypes(
                include=["bool"]
            ).columns

            df[bool_cols] = df[bool_cols].astype(int)

            # convert one-hot columns to int
            onehot_cols = encoded_df.columns

            for col in onehot_cols:
                df[col] = df[col].astype(int)

            logging.info(
                f"Encoding completed successfully. Shape: {df.shape}"
            )

            return df

        except Exception as e:
            logging.info("Encoding failed")
            raise CustomException(e, sys)


    def save(self):
        try:
            df = self.encode()
            file_path = "data/processed/processed_churn.csv"

            if os.path.exists(file_path):
                pass  # File exists, do nothing
            else:
                # Ensure the directory "data/raw" exists before saving
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
        except Exception as e:
            raise CustomException(e,sys)

    def importance(self):
        df = self.encode()
        print(df.head())
        corr = df.corr(numeric_only=True)

        target_corr = corr['Churn'].sort_values(
            ascending=False
        )
        print(df.isna().sum().sum())
        print(df.shape)
        print(df.dtypes)
        print(target_corr)
        print(df['Churn'].value_counts(normalize=True))
if __name__ == "__main__":
    pipeline = Datatransform("data/raw/churn.csv")
    pipeline.save()
    pipeline.importance()