import os
import pickle
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../classifier/model.pkl")


class RiskDecisionEngine:

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run classifier/train.py first."
            )

        print(f"Loading decision engine artifact from {self.model_path}...")

        with open(self.model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.encoders = artifact["encoders"]
        self.feature_names = artifact["features"]

        # SHAP is used only for explanation.
        # It does NOT participate in the decision itself.
        self.explainer = artifact.get("explainer")

        # Business-policy parameters
        self.fee_dispute = float(
            artifact.get("fee_dispute", 500.0)
        )

        self.min_contest_amount = float(
            artifact.get("min_contest_amount", 1500.0)
        )

        self.confidence_threshold = 0.70

    # ---------------------------------------------------------
    # Feature preprocessing
    # ---------------------------------------------------------

    def _prepare_features(self, dispute_data: dict):
        processed_data = dispute_data.copy()

        for column, encoder in self.encoders.items():

            if column not in processed_data:
                continue

            value = processed_data[column]

            if value in encoder.classes_:
                processed_data[column] = encoder.transform([value])[0]
            else:
                # Unknown categorical value.
                # XGBoost cannot use the raw string, so encode it
                # as an explicit unknown category.
                processed_data[column] = -1

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in processed_data
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features for inference: "
                f"{missing_features}"
            )

        feature_values = [
            processed_data[feature]
            for feature in self.feature_names
        ]

        return pd.DataFrame(
            [feature_values],
            columns=self.feature_names
        )

    # ---------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------

    def _calculate_shap(self, X):
        if self.explainer is None:
            return []

        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP/XGBoost output formats.
        if isinstance(shap_values, list):
            values = shap_values[1][0]

        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[0, :, 1]

        else:
            values = shap_values[0]

        feature_impacts = sorted(
            zip(
                self.feature_names,
                values,
                X.iloc[0].tolist()
            ),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        shap_summary = []

        for feature, impact, value in feature_impacts:

            if impact > 0:
                direction = "positive"
            elif impact < 0:
                direction = "negative"
            else:
                direction = "neutral"

            shap_summary.append({
                "feature": feature,
                "impact": round(float(impact), 4),
                "direction": direction,
                "value": value
            })

        return shap_summary

    # ---------------------------------------------------------
    # Main decision pipeline
    # ---------------------------------------------------------

    def evaluate_dispute(self, dispute_data: dict) -> dict:

        amount = float(
            dispute_data.get("amount_inr", 0.0)
        )

        if amount <= 0:
            raise ValueError(
                "amount_inr must be greater than zero."
            )

        # -----------------------------------------------------
        # 1. MODEL
        # -----------------------------------------------------

        X = self._prepare_features(dispute_data)

        prob_win = float(
            self.model.predict_proba(X)[0][1]
        )

        # -----------------------------------------------------
        # 2. SHAP
        # -----------------------------------------------------

        shap_factors = self._calculate_shap(X)

        # -----------------------------------------------------
        # 3. FINANCIAL DECISION
        # -----------------------------------------------------

        expected_value = (
            prob_win * amount
            - (1 - prob_win) * self.fee_dispute
        )

        # -----------------------------------------------------
        # 4. DETERMINISTIC POLICY
        # -----------------------------------------------------

        if amount < self.min_contest_amount:

            decision = "AUTO_ACCEPT"

            reason = (
                f"Amount (₹{amount:,.2f}) is below the "
                f"operational minimum threshold "
                f"(₹{self.min_contest_amount:,.2f})."
            )

        elif expected_value <= 0:

            decision = "AUTO_ACCEPT"

            reason = (
                f"Contesting has negative expected value "
                f"(EV: ₹{expected_value:,.2f})."
            )

        elif prob_win >= self.confidence_threshold:

            decision = "AUTO_CONTEST"

            reason = (
                f"Positive EV and high model confidence "
                f"(P(win): {prob_win * 100:.1f}% >= "
                f"{self.confidence_threshold * 100:.0f}%)."
            )

        else:

            decision = "MANUAL_REVIEW"

            reason = (
                f"Positive EV (₹{expected_value:,.2f}) but "
                f"model confidence is below the automatic "
                f"contest threshold "
                f"(P(win): {prob_win * 100:.1f}% < "
                f"{self.confidence_threshold * 100:.0f}%)."
            )

        # -----------------------------------------------------
        # 5. STRUCTURED RESULT
        # -----------------------------------------------------

        return {
            "decision": decision,

            # Model layer
            "prob_win": round(prob_win, 4),

            # Financial layer
            "amount_inr": round(amount, 2),
            "expected_value_inr": round(expected_value, 2),
            "fee_assumed_inr": round(self.fee_dispute, 2),

            # Policy layer
            "confidence_threshold": self.confidence_threshold,
            "minimum_contest_amount_inr": self.min_contest_amount,

            "reason": reason,

            # Explainability layer
            "shap_factors": shap_factors
        }