import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        agg = (
            X.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount","sum"),
                avg_transaction_amount=("Amount","mean"),
                transaction_count=("TransactionId","count"),
                std_transaction_amount=("Amount","std")
            )
            .reset_index()
        )

        X = X.merge(
            agg,
            on="CustomerId",
            how="left"
        )

        return X
class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["transaction_hour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["transaction_day"] = (
            X["TransactionStartTime"].dt.day
        )

        X["transaction_month"] = (
            X["TransactionStartTime"].dt.month
        )

        X["transaction_year"] = (
            X["TransactionStartTime"].dt.year
        )

        return X
class RFMFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        snapshot_date = (
            pd.to_datetime(
                X["TransactionStartTime"]
            ).max()
            + pd.Timedelta(days=1)
        )

        rfm = (
            X.groupby("CustomerId")
            .agg(
                recency=(
                    "TransactionStartTime",
                    lambda x:
                    (
                        snapshot_date -
                        pd.to_datetime(x).max()
                    ).days
                ),

                frequency=(
                    "TransactionId",
                    "count"
                ),

                monetary=(
                    "Amount",
                    "sum"
                )
            )
            .reset_index()
        )

        X = X.merge(
            rfm,
            on="CustomerId",
            how="left"
        )

        return X
class HighRiskLabel(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        rfm = (
            X[
                [
                    "CustomerId",
                    "recency",
                    "frequency",
                    "monetary"
                ]
            ]
            .drop_duplicates()
        )

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            rfm[
                [
                    "recency",
                    "frequency",
                    "monetary"
                ]
            ]
        )

        km = KMeans(
            n_clusters=3,
            random_state=42
        )

        rfm["cluster"] = km.fit_predict(
            scaled
        )

        cluster_summary = (
            rfm.groupby("cluster")
            [
                [
                    "recency",
                    "frequency",
                    "monetary"
                ]
            ]
            .mean()
        )

        high_risk_cluster = (
            cluster_summary
            .sort_values(
                ["frequency","monetary"]
            )
            .index[0]
        )

        rfm["is_high_risk"] = (
            rfm["cluster"] ==
            high_risk_cluster
        ).astype(int)

        X = X.merge(
            rfm[
                [
                    "CustomerId",
                    "is_high_risk"
                ]
            ],
            on="CustomerId",
            how="left"
        )

        return X
feature_pipeline = Pipeline([
    ("aggregate", AggregateFeatures()),
    ("date", DateFeatures()),
    ("rfm", RFMFeatures()),
    ("risk", HighRiskLabel())
])
categorical_features = [
    "ProviderId",
    "ProductCategory",
    "ChannelId",
    "ProductId"
]
numerical_features = [
    "Amount",
    "Value",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "recency",
    "frequency",
    "monetary"
]

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer",
             SimpleImputer(strategy="median")),
            ("scaler",
             StandardScaler())
        ]),
        numerical_features
    ),

    (
        "cat",
        Pipeline([
            ("imputer",
             SimpleImputer(strategy="most_frequent")),
            ("encoder",
             OneHotEncoder(
                 handle_unknown="ignore"
             ))
        ]),
        categorical_features
    )
])
if __name__ == "__main__":

    df = pd.read_csv(
        "../data/data.csv"
    )

    df = feature_pipeline.fit_transform(df)

    df.to_csv(
        "../data/processed_data.csv",
        index=False
    )

    print(
        "Processed data saved successfully."
    )