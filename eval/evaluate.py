import os
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "synthetic_disputes_test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "../classifier/model.pkl"
)

FEE_DISPUTE = 500.0
MIN_CONTEST_AMOUNT = 1500.0


def evaluate():

    print("Loading held-out test data...")

    test_df = pd.read_csv(TEST_PATH)

    with open(
        MODEL_PATH,
        "rb"
    ) as f:

        artifact = pickle.load(f)

    model = artifact["model"]
    encoders = artifact["encoders"]
    features = artifact["features"]

    df = test_df.copy()

    # ---------------------------------------------------------
    # Apply the SAME encoders used during training
    # ---------------------------------------------------------

    for col, encoder in encoders.items():

        df[col] = df[col].map(
            lambda value:
                encoder.transform([value])[0]
                if value in encoder.classes_
                else -1
        )

    X = df[features]

    probabilities = model.predict_proba(X)[:, 1]

    df["prob_win"] = probabilities

    # ---------------------------------------------------------
    # Expected Value & Dynamic Break-Even Probability
    # ---------------------------------------------------------

    df["expected_value"] = (
        df["prob_win"] * df["amount_inr"]
        -
        (1 - df["prob_win"]) * FEE_DISPUTE
    )

    df["break_even_prob"] = FEE_DISPUTE / (df["amount_inr"] + FEE_DISPUTE)

    # ---------------------------------------------------------
    # Three-way decision hierarchy (Aligned with decision.py)
    # ---------------------------------------------------------

    def decide(row):

        amount = row["amount_inr"]
        ev = row["expected_value"]
        probability = row["prob_win"]
        break_even = row["break_even_prob"]
        
        ABSOLUTE_MIN_WIN_PROB = 0.35

        if amount < MIN_CONTEST_AMOUNT:
            return "AUTO_ACCEPT"

        if ev <= 0 or probability <= break_even or probability < ABSOLUTE_MIN_WIN_PROB:
            return "AUTO_ACCEPT"

        if probability < 0.55:
            return "MANUAL_REVIEW"

        return "AUTO_CONTEST"

    df["decision"] = df.apply(
        decide,
        axis=1
    )

    # ---------------------------------------------------------
    # ML metrics
    # ---------------------------------------------------------

    predicted_contest = (
        df["decision"] == "AUTO_CONTEST"
    ).astype(int)

    actual_won = df["ground_truth_won"]

    precision = precision_score(
        actual_won,
        predicted_contest,
        zero_division=0
    )

    recall = recall_score(
        actual_won,
        predicted_contest,
        zero_division=0
    )

    # ---------------------------------------------------------
    # Financial simulation
    # ---------------------------------------------------------

    contested = df[
        df["decision"] == "AUTO_CONTEST"
    ]

    manual = df[
        df["decision"] == "MANUAL_REVIEW"
    ]

    accepted = df[
        df["decision"] == "AUTO_ACCEPT"
    ]

    revenue_recovered = contested.loc[
        contested["ground_truth_won"] == 1,
        "amount_inr"
    ].sum()

    failed_contest_amount = contested.loc[
        contested["ground_truth_won"] == 0,
        "amount_inr"
    ].sum()

    fees_incurred = (
        len(contested) * FEE_DISPUTE
    )

    net_recovery = (
        revenue_recovered
        - failed_contest_amount
        - fees_incurred
    )

    capital_at_risk = (
        failed_contest_amount
        + fees_incurred
    )

    kavach_roi = (
        net_recovery / capital_at_risk
        if capital_at_risk > 0
        else 0
    )

    false_positive_cost = (
        failed_contest_amount
        + fees_incurred
    )

    # ---------------------------------------------------------
    # Baseline: contest everything
    # ---------------------------------------------------------

    baseline_revenue = df.loc[
        df["ground_truth_won"] == 1,
        "amount_inr"
    ].sum()

    baseline_losses = df.loc[
        df["ground_truth_won"] == 0,
        "amount_inr"
    ].sum()

    baseline_fees = (
        len(df) * FEE_DISPUTE
    )

    baseline_net = (
        baseline_revenue
        - baseline_losses
        - baseline_fees
    )

    baseline_capital = (
        baseline_losses
        + baseline_fees
    )

    baseline_roi = (
        baseline_net / baseline_capital
        if baseline_capital > 0
        else 0
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 50)
    print("          KAVACH EVALUATION REPORT")
    print("=" * 50)

    print(
        f"Held-out test set:       {len(df)} disputes"
    )

    print(
        f"Contest precision:       {precision:.2f}"
    )

    print(
        f"Contest recall:          {recall:.2f}"
    )

    print("-" * 50)

    print(
        f"AUTO_CONTEST:            "
        f"{len(contested)} "
        f"({len(contested) / len(df) * 100:.1f}%)"
    )

    print(
        f"MANUAL_REVIEW:           "
        f"{len(manual)} "
        f"({len(manual) / len(df) * 100:.1f}%)"
    )

    print(
        f"AUTO_ACCEPT:             "
        f"{len(accepted)} "
        f"({len(accepted) / len(df) * 100:.1f}%)"
    )

    print("-" * 50)

    print(
        f"Kavach ROI:              "
        f"{kavach_roi:.2f}x"
    )

    print(
        f"Baseline ROI:            "
        f"{baseline_roi:.2f}x"
    )

    print(
        f"False-positive cost:     "
        f"₹{false_positive_cost:,.2f}"
    )

    print(
        f"Net recovery:            "
        f"₹{net_recovery:,.2f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    evaluate()