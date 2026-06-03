import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

df = pd.read_csv("../data/processed_data.csv")

TARGET = "is_high_risk"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mlflow.set_experiment(
    "credit_risk_model"
)

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return metrics
with mlflow.start_run(run_name="LogisticRegression"):

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

    grid = GridSearchCV(
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    mlflow.log_params(
        grid.best_params_
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.sklearn.log_model(
        best_model,
        "logistic_regression_model"
    )

    print("Logistic Regression")
    print(metrics)
results = []

results.append(
    {
        "model": "LogisticRegression",
        "roc_auc": metrics["roc_auc"]
    }
)

results.append(
    {
        "model": "LogisticRegression",
        "roc_auc": metrics["roc_auc"]
    }
)

results.append(
    {
        "model": "RandomForest",
        "roc_auc": metrics["roc_auc"]
    }
)