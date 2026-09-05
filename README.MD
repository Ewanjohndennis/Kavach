# Kavach 🛡️

**Razorpay AI Buildathon — Track 02: AI Risk Manager**

Kavach is an AI-assisted **chargeback decisioning and representment system** for **Visa Reason Code 10.4 — Other Fraud, Card-Absent Environment**.

When a dispute arrives, Kavach:

1. Validates that the dispute is within its supported reason-code scope.
2. Uses an **XGBoost classifier** to estimate the probability of successfully defending the dispute.
3. Calculates the **expected financial value (EV)** of contesting the dispute.
4. Applies deterministic business rules to route the dispute to:
* `AUTO_ACCEPT`
* `MANUAL_REVIEW`
* `AUTO_CONTEST`


5. Uses **SHAP** to explain the model's strongest contributing features.
6. Generates an evidence-grounded representment draft for automatically contested disputes.
7. Persists the decision, model explanation, and representment draft in a local SQLite database.
8. Provides a **terminal-based workflow** for reviewing and handling disputes without requiring a frontend.

> **Important:** Kavach's current representment generator is deterministic and template-grounded. It does not use an LLM to invent evidence. All factual claims in the generated draft are derived from the merchant telemetry supplied with the dispute.

---

## 1. Problem Scope

Traditional fraud detection generally focuses on deciding whether a transaction should be approved or rejected at checkout.

Kavach addresses a different operational problem:

> **Once a chargeback has already been received, should the merchant accept it, manually review it, or spend resources contesting it?**

The system currently focuses exclusively on **Visa Reason Code 10.4** rather than attempting to support every possible chargeback reason code.

### What Kavach does NOT do

* It is **not a checkout fraud detection system**.
* It does **not** support every Visa chargeback reason code.
* It currently supports **Visa 10.4 only**.
* It does **not** directly submit representments to Visa or any card network.
* It does **not** treat SHAP values as evidence of transaction legitimacy.
* It does **not** use an LLM to fabricate evidence.
* Its training and evaluation data are **synthetic buildathon data**, not production merchant dispute history.
* Its current SQLite database is intended for the local MVP and is not a production-grade distributed datastore.

---

## 2. System Architecture

```text
                    Incoming Dispute
                           │
                           ▼
                 FastAPI Webhook API
                           │
                           ▼
                 Reason-Code Validation
                     Visa 10.4 only
                           │
                           ▼
                   ┌───────────────────┐
                   │   XGBoost Model   │
                   │                   │
                   │ Estimate P(win)   │
                   └─────────┬─────────┘
                           │
                           ▼
                    SHAP Explanation
                           │
                           ▼
                  Expected Value (EV)
                           │
                           ▼
                Deterministic Decision
                    Policy Layer
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      AUTO_ACCEPT    MANUAL_REVIEW    AUTO_CONTEST
                                           │
                                           ▼
                                  Representment Draft
                                           │
                                           ▼
                                  SQLite Persistence

```

The architecture intentionally separates **prediction** from **business decisioning**:

* **XGBoost:** Estimates probability
* **SHAP:** Explains model prediction
* **Python:** Applies financial/business policy
* **Generator:** Produces evidence-grounded draft
* **SQLite:** Persists decisions and records

---

## 3. Decision Policy

Kavach uses three main decision boundaries:

1. **Minimum transaction threshold:** Transactions below the operational minimum of **₹1,500** are automatically accepted (`AUTO_ACCEPT`).
2. **Expected Value & Dynamic Break-Even:** Kavach calculates the financial expected value using a dispute arbitration fee assumption of **₹500**:

$$\text{EV} = (P(\text{win}) \times \text{Amount}) - ((1 - P(\text{win})) \times \text{Fee})$$



It computes the dynamic break-even probability ($\text{Break-Even} = \frac{\text{Fee}}{\text{Amount} + \text{Fee}}$). If $\text{EV} \le 0$ or $P(\text{win})$ falls below break-even or an absolute risk floor of **35%**, the dispute is auto-accepted.
3. **Model Confidence & Uncertainty Buffer:** Positive EV cases with win probabilities sitting in the uncertainty zone between the absolute floor and **55%** are routed to `MANUAL_REVIEW`. Cases clearing the 55% confidence threshold trigger `AUTO_CONTEST`.

---

## 4. Machine Learning & Explainability

### XGBoost

Kavach uses an XGBoost classifier trained on features including transaction amount, days since order, delivery signature availability, device fingerprint match, AVS match, customer past disputes, payment method, weekend indicators, and merchant categories.

### SHAP

Kavach uses SHAP tree explainers to surface top contributing feature weights for every prediction, which are explicitly separated from factual representment evidence.

---

## 5. Representment Generation

For `AUTO_CONTEST` cases, Kavach generates a structured representment draft using a deterministic, template-grounded Python generator. It incorporates verified merchant telemetry (such as AVS matches, device fingerprint matches, and delivery signatures) and appends model explanations under a clear separation header without hallucinating tracking numbers or evidence.

---

## 6. Synthetic Data & Evaluation

Kavach uses a synthetic dataset of 12,000 disputes split into 10,000 training rows and 2,000 held-out test rows. Running `eval/evaluate.py` evaluates the complete decision policy against a contest-everything baseline, tracking contest precision, recall, net financial recovery, ROI, and false-positive costs.

---

## 7. Persistence & API

* **SQLite Persistence:** Processed disputes are saved locally in `data/kavach.db`.
* **API Endpoints:** Built with FastAPI, exposing `/health`, `/`, `/webhook/dispute` (returning `202 Accepted`), `/disputes/{dispute_id}`, `/disputes`, and `/api/dashboard`.

---

## 8. Installation & Execution Guide

### Step 1 — Clone and Setup Environment

```bash
git clone <repository-url>
cd Kavach
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### Step 2 — Generate Synthetic Data, Train, and Evaluate

```bash
python data/generate.py
python classifier/train.py
python eval/evaluate.py

```

### Step 3 — Run the API and Test Webhook

```bash
uvicorn agent.main:app --reload

```

In a separate terminal, trigger a test webhook:

```bash
curl -X POST http://127.0.0.1:8000/webhook/dispute \
-H "Content-Type: application/json" \
-d '{
  "dispute_id": "dsp_live_demo_01",
  "amount_inr": 12500.50,
  "reason_code": "Visa_10.4",
  "days_since_order": 5,
  "has_delivery_signature": 1,
  "device_hash_match": 1,
  "avs_match": 1,
  "customer_past_disputes": 0,
  "payment_method": "Credit Card",
  "is_weekend": 0,
  "merchant_category": "Electronics"
}'

```

### Step 4 — Run CLI Operator Interface & Tests

```bash
python cli.py
python test_decision.py

```

---

**Built by Ewan John Dennis for the Razorpay AI Buildathon — September 2026.**