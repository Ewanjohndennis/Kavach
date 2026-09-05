import os
import pickle

import pandas as pd
import xgboost as xgb
import shap

from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "../eval/synthetic_disputes_train.csv"
)

MODEL_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

FEE_DISPUTE = 500.0
MIN_CONTEST_AMOUNT = 1500.0


FEATURES = [
    "amount_inr",
    "days_since_order",
    "has_delivery_signature",
    "device_hash_match",
    "avs_match",
    "customer_past_disputes",
    "payment_method",
    "is_weekend",
    "merchant_category"
]


def train_model():

    print("Loading training data...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}. "
            "Run data/generate.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Training disputes: {len(df)}"
    )

    categorical_cols = [
        "payment_method",
        "merchant_category"
    ]

    encoders = {}

    # ---------------------------------------------------------
    # Encode categorical features
    # ---------------------------------------------------------

    for col in categorical_cols:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col]
        )

        encoders[col] = encoder

    X = df[FEATURES]
    y = df["ground_truth_won"]

    # ---------------------------------------------------------
    # Train XGBoost
    # ---------------------------------------------------------

    print("Training XGBoost classifier...")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X, y)

    # ---------------------------------------------------------
    # SHAP explainer
    # ---------------------------------------------------------

    print("Creating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    # ---------------------------------------------------------
    # Save everything required for inference
    # ---------------------------------------------------------

    artifact = {
        "model": model,
        "encoders": encoders,
        "features": FEATURES,
        "explainer": explainer,
        "fee_dispute": FEE_DISPUTE,
        "min_contest_amount": MIN_CONTEST_AMOUNT
    }

    with open(
        MODEL_OUTPUT_PATH,
        "wb"
    ) as f:

        pickle.dump(
            artifact,
            f
        )

    print(
        f"\nModel artifact saved to:"
        f"\n{MODEL_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    train_model()