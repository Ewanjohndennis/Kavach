from agent.decision import RiskDecisionEngine

engine = RiskDecisionEngine()

dispute = {
    "dispute_id": "shap_test_strong",
    "amount_inr": 12500,
    "reason_code": "Visa_10.4",
    "days_since_order": 5,
    "has_delivery_signature": 1,
    "device_hash_match": 1,
    "avs_match": 1,
    "customer_past_disputes": 0,
    "payment_method": "Credit Card",
    "is_weekend": 0,
    "merchant_category": "Electronics"
}

result = engine.evaluate_dispute(dispute)

print("\n==============================")
print("KAVACH DECISION")
print("==============================")
print(f"Decision:   {result['decision']}")
print(f"P(win):     {result['prob_win']:.2%}")
print(f"EV:         ₹{result['expected_value_inr']:,.2f}")

print("\n==============================")
print("SHAP FACTORS")
print("==============================")

for factor in result["shap_factors"]:
    print(
        f"{factor['feature']:25s} "
        f"impact={factor['impact']:+.4f} "
        f"value={factor['value']}"
    )