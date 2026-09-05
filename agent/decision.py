import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../classifier/model.pkl")

class RiskDecisionEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Run train.py first.")
        
        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)
            
        self.model = artifact["model"]
        self.encoders = artifact["encoders"]
        self.feature_names = artifact["features"]
        self.explainer = artifact.get("explainer", None)
        self.fee_dispute = artifact.get("fee_dispute", 500.0)
        self.min_contest_amount = artifact.get("min_contest_amount", 1500.0)

    def evaluate_dispute(self, dispute_data: dict) -> dict:
        processed_data = dispute_data.copy()
        amount = float(dispute_data.get('amount_inr', 0.0))
        
        for col, encoder in self.encoders.items():
            if col in processed_data:
                val = processed_data[col]
                if val in encoder.classes_:
                    processed_data[col] = encoder.transform([val])[0]
                else:
                    processed_data[col] = -1 
        
        features = [processed_data[feat] for feat in self.feature_names]
        X = pd.DataFrame([features], columns=self.feature_names)
        prob_win = float(self.model.predict_proba(X)[0][1])
        
        # 1. Calculate Dynamic Break-Even Probability: Fee / (Amount + Fee)
        break_even_prob = self.fee_dispute / (amount + self.fee_dispute)

        # 2. Calculate Financial Expected Value (EV)
        expected_value = (prob_win * amount) - ((1 - prob_win) * self.fee_dispute)
        
        # 3. SHAP Feature Impact Extraction (Optional explainability layer)
        shap_summary = []
        if self.explainer:
            shap_values = self.explainer.shap_values(X)
            vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]
            feature_impacts = sorted(zip(self.feature_names, vals, features), key=lambda x: abs(x[1]), reverse=True)
            for feat, val, raw_val in feature_impacts:
                shap_summary.append({"feature": feat, "impact": float(val), "direction": "+" if val > 0 else "-", "value": raw_val})

        # 4. Three-Way Decision Hierarchy using Dynamic Break-Even & Risk Floors
        ABSOLUTE_MIN_WIN_PROB = 0.35

        # Operational Friction Floor
        if amount < self.min_contest_amount:
            decision = "AUTO_ACCEPT"
            reason = f"Amount (₹{amount:,.2f}) is below operational minimum threshold (₹{self.min_contest_amount:,.2f})."
        
        # Economically Unviable or Below Absolute Confidence Floor
        elif expected_value <= 0 or prob_win <= break_even_prob or prob_win < ABSOLUTE_MIN_WIN_PROB:
            decision = "AUTO_ACCEPT"
            reason = f"Negative EV, break-even, or P(win) ({prob_win*100:.1f}%) below minimum risk floor ({ABSOLUTE_MIN_WIN_PROB*100}%)."
        
        # Borderline Zone: Between absolute floor/break-even and 55% confidence for manual review
        elif prob_win < 0.55:
            decision = "MANUAL_REVIEW"
            reason = f"Positive EV (₹{expected_value:,.2f}), but P(win) ({prob_win*100:.1f}%) sits within the manual review uncertainty zone (< 55%)."
        
        # High Confidence Automation
        else:
            decision = "AUTO_CONTEST"
            reason = f"Positive EV (₹{expected_value:,.2f}) and high model confidence (P(win): {prob_win*100:.1f}% >= 55%)."

        return {
            "decision": decision,
            "prob_win": round(prob_win, 4),
            "expected_value_inr": round(expected_value, 2),
            "break_even_prob": round(break_even_prob, 4),
            "fee_assumed_inr": self.fee_dispute,
            "reason": reason,
            "shap_factors": shap_summary
        }