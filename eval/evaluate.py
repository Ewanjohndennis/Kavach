import os
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, classification_report

# Ensure paths work whether run from root or eval/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../classifier/model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "synthetic_disputes_eval.csv")

def run_evaluation():
    print("Loading test dataset and model artifact...")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Test dataset not found at {CSV_PATH}. Run data/generate.py first.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model artifact not found at {MODEL_PATH}. Run classifier/train.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Avg transaction: ₹{df['amount_inr'].mean():,.0f} | Median: ₹{df['amount_inr'].median():,.0f}")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
        
    model = artifact["model"]
    encoders = artifact["encoders"]
    feature_names = artifact["features"]
    fee_dispute = artifact.get("fee_dispute", 500.0)
    min_contest_amount = artifact.get("min_contest_amount", 1500.0)

    # 1. Transform categorical variables using saved encoders
    df_eval = df.copy()
    for col, encoder in encoders.items():
        if col in df_eval.columns:
            df_eval[col] = df_eval[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    X_test = df_eval[feature_names]
    y_test = df_eval['ground_truth_won']
    amounts = df_eval['amount_inr']

    # 2. Run Inference
    print("Running model inference on evaluation set...")
    prob_wins = model.predict_proba(X_test)[:, 1]
    
    df_eval['prob_win'] = prob_wins
    df_eval['actual_result'] = y_test

    # 3. Apply Multi-Layer Decision Rule
    df_eval['expected_value'] = (df_eval['prob_win'] * amounts) - ((1 - df_eval['prob_win']) * fee_dispute)
    df_eval['decision_contest'] = (
        (df_eval['expected_value'] > 0) & 
        (amounts >= min_contest_amount)
    ).astype(int)

    predicted_decisions = df_eval['decision_contest']

    # 4. Statistical Metrics
    precision = precision_score(y_test, predicted_decisions)
    recall = recall_score(y_test, predicted_decisions)

    # 5. Financial Metrics: Kavach Strategy & Splits
    contested_cases = df_eval[df_eval['decision_contest'] == 1]
    auto_accepted_count = len(df_eval) - len(contested_cases)
    
    revenue_recovered = contested_cases[contested_cases['actual_result'] == 1]['amount_inr'].sum()
    losses_on_failed = contested_cases[contested_cases['actual_result'] == 0]['amount_inr'].sum()
    fees_incurred = len(contested_cases) * fee_dispute
    
    net_recovery = revenue_recovered - losses_on_failed - fees_incurred
    capital_at_risk = losses_on_failed + fees_incurred
    kavach_roi = (net_recovery / capital_at_risk) if capital_at_risk > 0 else 0

    # 6. Financial Metrics: Baseline Strategy ("Contest Everything")
    baseline_revenue = df_eval[df_eval['actual_result'] == 1]['amount_inr'].sum()
    baseline_losses = df_eval[df_eval['actual_result'] == 0]['amount_inr'].sum()
    baseline_fees = len(df_eval) * fee_dispute
    
    baseline_net = baseline_revenue - baseline_losses - baseline_fees
    baseline_capital = baseline_losses + baseline_fees
    baseline_roi = (baseline_net / baseline_capital) if baseline_capital > 0 else 0

    # 7. Output Results Report
    print("\n" + "="*42)
    print("         KAVACH EVALUATION REPORT       ")
    print("="*42)
    print(f"Test Set Size:       {len(df_eval)} disputes")
    print(f"Model Precision:     {precision:.2f}")
    print(f"Model Recall:        {recall:.2f}")
    print("-" * 42)
    print(f"Cases Contested:     {len(contested_cases)} ({len(contested_cases)/len(df_eval)*100:.1f}%)")
    print(f"Cases Auto-Accepted: {auto_accepted_count} ({auto_accepted_count/len(df_eval)*100:.1f}%)")
    print("-" * 42)
    print(f"Baseline ROI:        {baseline_roi:.2f}x")
    print(f"Kavach ROI:          {kavach_roi:.2f}x")
    
    improvement = ((kavach_roi - baseline_roi) / abs(baseline_roi) * 100) if baseline_roi != 0 else 0
    print(f"Net Improvement:     {improvement:.1f}%")
    print("="*42 + "\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, predicted_decisions, target_names=["Auto-Accept (Loss)", "Contest (Win)"]))

if __name__ == "__main__":
    run_evaluation()