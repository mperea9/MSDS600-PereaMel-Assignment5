import argparse

import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model

# Feature columns the model expects, in order.
FEATURE_COLUMNS = [
    "tenure", "PhoneService", "Contract", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "charge_per_tenure",
]

# Week 2 encodings, reused so the unmodified data can be prepared to match.
PHONE_MAP = {"Yes": 1, "No": 0}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP = {
    "Credit card (automatic)": 0,
    "Mailed check": 1,
    "Electronic check": 2,
    "Bank transfer (automatic)": 3,
}


class ChurnPredictor:
    """Holds the churn-prediction workflow: load, preprocess, predict."""

    def __init__(self, model_name="churn_model", training_data="prepared_churn_data.csv"):
        self.model = load_model(model_name)
        self.training_data = training_data
        self._train_probabilities = None

    # ------------------------------------------------------------------ #
    # Data loading / preprocessing
    # ------------------------------------------------------------------ #
    def load_data(self, filepath):
        """Load churn data into a DataFrame from a string filepath."""
        df = pd.read_csv(filepath, index_col="customerID")
        # If this looks like the raw/unmodified data, preprocess it.
        if df["PhoneService"].dtype == object or "charge_per_tenure" not in df.columns:
            df = self.preprocess(df)
        return df

    def preprocess(self, df):
        df = df.copy()
        if df["PhoneService"].dtype == object:
            df["PhoneService"] = df["PhoneService"].map(PHONE_MAP)
        if df["Contract"].dtype == object:
            df["Contract"] = df["Contract"].map(CONTRACT_MAP)
        if df["PaymentMethod"].dtype == object:
            df["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # For brand-new customers TotalCharges can be missing; fall back to one month.
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

        if "charge_per_tenure" not in df.columns:
            df["charge_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
            df["charge_per_tenure"] = df["charge_per_tenure"].fillna(df["MonthlyCharges"])

        return df[FEATURE_COLUMNS]

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _training_probabilities(self):
        """Cache and return churn probabilities on the training data."""
        if self._train_probabilities is None:
            train = pd.read_csv(self.training_data, index_col="customerID")
            train = train.drop(columns=["Churn"], errors="ignore")
            scored = predict_model(self.model, data=train, raw_score=True)
            self._train_probabilities = scored["prediction_score_1"].values
        return self._train_probabilities

    def make_predictions(self, df):
        """
        Predict churn for each row in df.

        Returns a DataFrame with:
          - churn_probability : probability of the churn class (1)
          - churn_prediction  : 'Churn' / 'No churn' at the default 0.5 threshold
          - train_percentile  : percentile of that probability within the
                                training-data probability distribution
        """
        scored = predict_model(self.model, data=df, raw_score=True)
        probs = scored["prediction_score_1"].values

        train_probs = self._training_probabilities()
        percentiles = [
            round(float((train_probs <= p).mean() * 100), 1) for p in probs
        ]

        result = pd.DataFrame(index=df.index)
        result["churn_probability"] = np.round(probs, 4)
        result["churn_prediction"] = scored["prediction_label"].map(
            {1: "Churn", 0: "No churn"}
        ).values
        result["train_percentile"] = percentiles
        return result


def main():
    parser = argparse.ArgumentParser(description="Predict churn for a CSV of customers.")
    parser.add_argument(
        "--file", "-f", default="new_churn_data.csv",
        help="Path to the CSV file to score (default: new_churn_data.csv). "
             "Both prepared and unmodified churn data are accepted.",
    )
    args = parser.parse_args()

    predictor = ChurnPredictor()
    df = predictor.load_data(args.file)
    predictions = predictor.make_predictions(df)
    print(f"predictions for {args.file}:")
    print(predictions)


if __name__ == "__main__":
    main()
