import pandas as pd
import numpy as np
import uuid
import sqlite3

np.random.seed(42)

def generate_disputes(n_samples=1000):
    data = []
    for _ in range(n_samples):
        # 1. Base Transaction (Lognormal skew typical for Indian digital payments)
        amount = np.random.lognormal(mean=7.0, sigma=1.0)
        amount = round(max(100, min(amount, 50000)), 2)
        
        # 2. True Signal Features
        delivery_signature = 1 if (amount > 10000 and np.random.rand() > 0.1) else np.random.choice([0, 1], p=[0.4, 0.6])
        time_delta_days = max(1, min(int(np.random.exponential(scale=15)), 120))
        device_hash_match = np.random.choice([0, 1], p=[0.3, 0.7])
        avs_match = np.random.choice([0, 1], p=[0.2, 0.8])
        past_disputes = np.random.poisson(lam=0.1)

        # 3. Noise Features (Tests the model's ability to ignore garbage)
        payment_method = np.random.choice(['UPI', 'Credit Card', 'Debit Card'], p=[0.5, 0.4, 0.1])
        is_weekend = np.random.choice([0, 1], p=[0.71, 0.29])
        merchant_category = np.random.choice(['Electronics', 'Apparel', 'Digital Goods'])

        # 4. Latent Variable: The human element (arbitrator mood, unrecorded customer history)
        arbitrator_leniency = np.random.normal(0, 0.15)
        
        # 5. Ground Truth Function
        base_win_prob = (0.2 + (0.4 * delivery_signature) + 
                        (0.2 * device_hash_match) + 
                        (0.1 * avs_match) - 
                        (0.05 * past_disputes) + 
                        arbitrator_leniency)
        
        base_win_prob = max(0.01, min(base_win_prob, 0.99))
        actual_win = np.random.binomial(1, base_win_prob)

        data.append({
            "dispute_id": f"dsp_{uuid.uuid4().hex[:12]}",
            "amount_inr": amount,
            "reason_code": "Visa_10.4",
            "days_since_order": time_delta_days,
            "has_delivery_signature": delivery_signature,
            "device_hash_match": device_hash_match,
            "avs_match": avs_match,
            "customer_past_disputes": past_disputes,
            "payment_method": payment_method,
            "is_weekend": is_weekend,
            "merchant_category": merchant_category,
            "ground_truth_won": actual_win
        })

    df = pd.DataFrame(data)
    
    # Export for the ML pipeline
    csv_path = "../eval/synthetic_disputes_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {n_samples} records to {csv_path}")
    
    # Export to SQLite to act as our "Feature Store" for the FastAPI app
    conn = sqlite3.connect("../data/feature_store.db")
    df.to_sql("disputes", conn, if_exists="replace", index=False)
    conn.close()
    print("Database seeded at ../data/feature_store.db")

if __name__ == "__main__":
    import os
    os.makedirs("../eval", exist_ok=True)
    generate_disputes(1000)