import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42

TRAIN_SIZE = 10000
TEST_SIZE = 2000

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../eval"
)

TRAIN_PATH = os.path.join(OUTPUT_DIR, "synthetic_disputes_train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "synthetic_disputes_test.csv")


def generate_disputes(n, seed):
    rng = np.random.default_rng(seed)

    amounts = rng.lognormal(
        mean=np.log(3500),
        sigma=1.0,
        size=n
    )

    amounts = np.clip(amounts, 200, 50000).round(2)

    days_since_order = rng.integers(1, 30, size=n)

    has_delivery_signature = rng.binomial(1, 0.55, size=n)
    device_hash_match = rng.binomial(1, 0.75, size=n)
    avs_match = rng.binomial(1, 0.72, size=n)

    customer_past_disputes = rng.poisson(
        lam=0.25,
        size=n
    )

    payment_method = rng.choice(
        ["Credit Card", "Debit Card", "UPI"],
        size=n,
        p=[0.45, 0.25, 0.30]
    )

    is_weekend = rng.binomial(1, 0.28, size=n)

    merchant_category = rng.choice(
        [
            "Electronics",
            "Fashion",
            "Food",
            "Travel",
            "Digital Goods"
        ],
        size=n,
        p=[0.20, 0.25, 0.20, 0.15, 0.20]
    )

    # ---------------------------------------------------------
    # Latent probability of successfully defending the dispute
    # ---------------------------------------------------------
    #
    # This is NOT exposed to the model.
    # It represents the underlying process that generated
    # the eventual dispute outcome.
    #

    score = (
        -0.8
        + 1.3 * has_delivery_signature
        + 1.0 * device_hash_match
        + 0.9 * avs_match
        - 0.65 * customer_past_disputes
        - 0.025 * days_since_order
        + 0.00002 * amounts
    )

    # Merchant/category effects
    score += np.where(
        merchant_category == "Electronics",
        0.20,
        0
    )

    score += np.where(
        payment_method == "Credit Card",
        0.15,
        0
    )

    # Unobservable factor
    arbitrator_leniency = rng.normal(
        0,
        0.7,
        size=n
    )

    score += arbitrator_leniency

    # Logistic transformation
    probability = 1 / (1 + np.exp(-score))

    ground_truth_won = rng.binomial(
        1,
        probability
    )

    df = pd.DataFrame({
        "amount_inr": amounts,
        "days_since_order": days_since_order,
        "has_delivery_signature": has_delivery_signature,
        "device_hash_match": device_hash_match,
        "avs_match": avs_match,
        "customer_past_disputes": customer_past_disputes,
        "payment_method": payment_method,
        "is_weekend": is_weekend,
        "merchant_category": merchant_category,
        "ground_truth_won": ground_truth_won
    })

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic dispute dataset...")

    train_df = generate_disputes(
        TRAIN_SIZE,
        RANDOM_SEED
    )

    test_df = generate_disputes(
        TEST_SIZE,
        RANDOM_SEED + 1
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print(f"Training set: {len(train_df)} disputes")
    print(f"Held-out test set: {len(test_df)} disputes")

    print(f"\nSaved:")
    print(TRAIN_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()