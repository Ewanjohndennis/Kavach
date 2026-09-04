import os
import pickle
import pandas as pd

class RiskDecisionEngine:
    def __init__(self, model_path="../classifier/model.pkl"):
        """
        Initializes the engine, loading the XGBoost model, Encoders, and thresholds once into memory.
        """
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}. Run train.py first.")
        
        print(f"Loading decision engine artifact from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)
            
        self.model = artifact["model"]
        self.encoders = artifact["encoders"]
        self.feature_names = artifact["features"]
        self.fee_dispute = artifact.get("fee_dispute", 500.0)
        self.min_contest_amount = artifact.get("min_contest_amount", 1500.0)

    def evaluate_dispute(self, dispute_data: dict) -> dict:
        """
        Scores the dispute and applies the multi-layer EV + Min Amount decision rules.
        """
        processed_data = dispute_data.copy()
        amount = float(dispute_data.get('amount_inr', 0.0))
        
        # 1. Safely encode categorical strings
        for col, encoder in self.encoders.items():
            if col in processed_data:
                val = processed_data[col]
                if val in encoder.classes_:
                    processed_data[col] = encoder.transform([val])[0]
                else:
                    processed_data[col] = -1 
        
        # 2. Enforce strict feature ordering
        try:
            features = [processed_data[feat] for feat in self.feature_names]
        except KeyError as e:
            raise ValueError(f"Missing required feature for inference: {e}")

        # 3. Model Inference
        X = pd.DataFrame([features], columns=self.feature_names)
        prob_win = self.model.predict_proba(X)[0][1]
        
        # 4. Expected Value (EV) Calculation
        expected_value = (prob_win * amount) - ((1 - prob_win) * self.fee_dispute)
        
        # 5. Multi-Layer Financial Decision (EV > 0 AND amount >= Minimum Friction Threshold)
        meets_ev = expected_value > 0
        meets_min_amount = amount >= self.min_contest_amount
        
        decision = "CONTEST" if (meets_ev and meets_min_amount) else "AUTO_ACCEPT"
        
        reason = "Passed EV and minimum threshold."
        if not meets_min_amount:
            reason = f"Auto-accepted: Amount (₹{amount}) is below operational minimum threshold (₹{self.min_contest_amount})."
        elif not meets_ev:
            reason = f"Auto-accepted: Negative expected value (EV: ₹{expected_value:.2f})."

        return {
            "decision": decision,
            "prob_win": round(float(prob_win), 4),
            "expected_value_inr": round(float(expected_value), 2),
            "fee_assumed_inr": self.fee_dispute,
            "reason": reason
        }