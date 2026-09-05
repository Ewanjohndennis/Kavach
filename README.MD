# Kavach 🛡️

**Razorpay AI Buildathon — Track 02: AI Risk Manager**

Kavach is an AI-assisted **chargeback decisioning and representment system** for **Visa Reason Code 10.4 — Other Fraud, Card-Absent Environment**.

When a dispute arrives, Kavach:

1. Validates that the dispute is within its supported reason-code scope.
2. Uses an **XGBoost classifier** to estimate the probability of successfully defending the dispute.
3. Calculates the **expected financial value (EV)** of contesting the dispute.
4. Applies deterministic business rules to route the dispute to:
   - `AUTO_ACCEPT`
   - `MANUAL_REVIEW`
   - `AUTO_CONTEST`
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

This allows the evidence model, risk model, and decision policy to remain bounded and auditable.

### What Kavach does NOT do

- It is **not a checkout fraud detection system**.
- It does **not** support every Visa chargeback reason code.
- It currently supports **Visa 10.4 only**.
- It does **not** directly submit representments to Visa or any card network.
- It does **not** treat SHAP values as evidence of transaction legitimacy.
- It does **not** use an LLM to fabricate evidence.
- Its training and evaluation data are **synthetic buildathon data**, not production merchant dispute history.
- Its current SQLite database is intended for the local MVP and is not a production-grade distributed datastore.

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
                           │
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
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    AUTO_ACCEPT      MANUAL_REVIEW    AUTO_CONTEST
                                          │
                                          ▼
                              Representment Draft
                                          │
                                          ▼
                                   SQLite Persistence
````

The architecture intentionally separates **prediction** from **business decisioning**:

```text
XGBoost  →  estimates probability
SHAP     →  explains model prediction
Python   →  applies financial/business policy
Generator → produces evidence-grounded draft
SQLite   → persists decisions and records
```

This prevents the ML model or an LLM from directly determining the financial action.

---

## 3. Decision Policy

Kavach uses three main decision boundaries.

### Step 1 — Minimum transaction threshold

Transactions below the operational minimum of **₹1,500** are automatically accepted rather than sent through an expensive contest process.

```text
Amount < ₹1,500
       │
       ▼
AUTO_ACCEPT
```

### Step 2 — Expected Value

For transactions above the minimum threshold, Kavach calculates:

```text
EV = (P(win) × Transaction Amount)
     − ((1 − P(win)) × Contest Fee)
```

The current buildathon configuration assumes:

```text
Contest / arbitration cost = ₹500
```

This is a **modeling assumption**, not a claim that ₹500 is a universal Visa fee.

If:

```text
EV ≤ 0
```

the system chooses:

```text
AUTO_ACCEPT
```

### Step 3 — Model confidence

If EV is positive, the model probability is compared with the automatic contest threshold:

```text
P(win) ≥ 70%
        │
        ├── YES → AUTO_CONTEST
        │
        └── NO  → MANUAL_REVIEW
```

### Complete Decision Flow

```text
                         Dispute
                            │
                            ▼
                  Supported reason code?
                    Visa 10.4 only
                            │
                            ▼
                     Amount < ₹1,500?
                       /           \
                     YES            NO
                      │              │
                      ▼              ▼
                AUTO_ACCEPT      Calculate EV
                                     │
                                     ▼
                                  EV > 0?
                                /         \
                              NO           YES
                              │             │
                              ▼             ▼
                        AUTO_ACCEPT     P(win) ≥ 70%?
                                         /         \
                                       YES          NO
                                        │            │
                                        ▼            ▼
                                  AUTO_CONTEST  MANUAL_REVIEW
```

The thresholds are stored with the model configuration so that the evaluation pipeline can reproduce the same policy used by the application.

---

## 4. Machine Learning & Explainability

### XGBoost

Kavach uses an XGBoost classifier to estimate the probability that a dispute can be successfully defended.

The model considers:

* Transaction amount
* Days since order
* Delivery signature availability
* Device fingerprint match
* AVS match
* Customer's previous dispute count
* Payment method
* Weekend indicator
* Merchant category

Categorical fields are encoded before inference, and the trained model, encoders, feature list, SHAP explainer, and decision-policy parameters are bundled into the generated `classifier/model.pkl` artifact.

### SHAP

Kavach uses SHAP's tree-based explanation capability to identify the features that contributed most strongly to each prediction.

Example:

```text
P(win) = 92.8%

Strong positive model contributors:
- has_delivery_signature
- amount_inr
- avs_match
- days_since_order
```

These values explain **why the model produced its prediction**.

They are **not representment evidence** and are never treated as proof that a transaction is legitimate.

---

## 5. Representment Generation

For `AUTO_CONTEST` cases, Kavach generates a structured representment draft.

The generator is intentionally conservative.

It can reference verified transaction telemetry such as:

```text
AVS match
Device fingerprint match
Delivery signature
Days since order
```

It does **not** invent:

* Tracking numbers
* IP addresses
* Customer statements
* Order metadata not provided to the system
* Additional delivery records
* Payment-network evidence
* Any other unsupported facts

The generated document also distinguishes between:

```text
Verified transaction evidence
             vs.
Model explanation
```

This prevents SHAP model explanations from being incorrectly presented as merchant evidence.

The current generator is **deterministic**. A constrained LLM prose layer could be added later, but factual claims should remain bound to verified transaction data.

---

## 6. Synthetic Data & Evaluation

Kavach uses synthetic data for the buildathon.

The generator creates **12,000 synthetic disputes** and performs a deterministic train/test split:

```text
12,000 synthetic disputes
        │
        ├── 10,000 training rows
        │
        └── 2,000 held-out test rows
```

The classifier is trained only on the training split.

The held-out test set is never used during model fitting.

### Evaluation Metrics

`eval/evaluate.py` evaluates the complete decision policy rather than only the classifier.

It reports:

* Contest precision
* Contest recall
* `AUTO_CONTEST` count
* `MANUAL_REVIEW` count
* `AUTO_ACCEPT` count
* Net financial impact
* ROI
* False-positive cost
* Comparison against a contest-everything baseline
* Relative ROI improvement over the baseline

The financial evaluation considers:

```text
Successful contests
        +
Failed contests
        +
Contest fees
```

This is important because a model can achieve good classification metrics while still making poor financial decisions.

> Because the dataset is synthetic, the resulting metrics demonstrate the behavior of the system and decision policy. They should not be interpreted as production performance or real-world chargeback win rates.

---

## 7. Persistence & Dispute Memory

Kavach uses **SQLite** for local persistence.

The application stores processed disputes in:

```text
data/kavach.db
```

The database contains information including:

* Original dispute features
* Final decision
* Predicted win probability
* Expected value
* Assumed contest fee
* Decision rationale
* SHAP factors
* Generated representment draft
* Creation/update timestamps

The database is created automatically by the application if it does not already exist.

### Why `kavach.db` is not committed

The SQLite database is runtime state, not source code.

A fresh clone of Kavach creates its own local database:

```text
Clone repository
       │
       ▼
Run Kavach
       │
       ▼
data/kavach.db created
       │
       ▼
New disputes stored locally
```

This means different users or forks maintain their own independent dispute history.

The repository therefore contains the **database implementation**, but not the developer's local database contents.

---

## 8. Manual Review Workflow

Not every economically positive dispute should be automatically contested.

When:

```text
EV > 0
but
P(win) < 70%
```

Kavach routes the case to:

```text
MANUAL_REVIEW
```

Manual-review cases can be inspected through the terminal-based workflow.

The terminal interface is intended to provide the operator with access to:

* Pending manual-review disputes
* Dispute details
* Model probability
* Expected value
* SHAP explanations
* Decision rationale
* Available representment information

This creates a human-in-the-loop boundary between automated decisioning and uncertain cases.

```text
                  Model
                    │
                    ▼
             Confidence < 70%
                    │
                    ▼
             MANUAL_REVIEW
                    │
                    ▼
             Human Operator
                    │
             ┌──────┴──────┐
             ▼             ▼
         Contest        Accept
```

The current MVP deliberately keeps this interface terminal-based rather than introducing a frontend.

---

## 9. API

Kavach exposes a FastAPI application.

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "service": "kavach"
}
```

### Service Status

```http
GET /
```

Returns the current service status and version.

### Process a Dispute

```http
POST /webhook/dispute
```

Example request:

```json
{
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
}
```

The webhook returns:

```text
202 Accepted
```

and the dispute is processed by the application before the resulting decision is persisted.

### Retrieve a Dispute

```http
GET /disputes/{dispute_id}
```

Returns the persisted dispute, decision, SHAP explanation, and representment draft when one exists.

### Retrieve Recent Disputes

```http
GET /disputes?limit=10
```

Returns recently processed disputes.

### Dashboard Summary

```http
GET /api/dashboard
```

Provides aggregate information such as:

* Total disputes
* Decision counts
* Contest rate
* Total expected recovery
* Average win probability
* Recent disputes

---

## 10. Repository Structure

```text
kavach/
│
├── agent/
│   ├── main.py
│   │   # FastAPI application and dispute webhook
│   │
│   ├── decision.py
│   │   # XGBoost inference, SHAP calculation,
│   │   # EV calculation and decision policy
│   │
│   ├── letter_generator.py
│   │   # Deterministic, evidence-grounded
│   │   # representment draft generator
│   │
│   └── database.py
│       # SQLite persistence and dispute retrieval
│
├── classifier/
│   ├── train.py
│   │   # Model training and artifact generation
│   │
│   └── model.pkl
│       # Generated model artifact
│       # (ignored by Git)
│
├── data/
│   └── generate.py
│       # Generates the synthetic dataset
│
├── eval/
│   ├── evaluate.py
│   │   # Held-out model and financial evaluation
│   │
│   ├── synthetic_disputes_train.csv
│   │   # Generated training split
│   │
│   └── synthetic_disputes_test.csv
│       # Generated held-out test split
│
├── cli.py
│   # Terminal-based operator interface
│
├── test_decision.py
│   # Decision-engine tests
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Runtime-generated files

The following are intentionally not committed:

```text
classifier/model.pkl
data/kavach.db
```

They are generated locally as part of the setup/runtime process.

---

## 11. Installation & Execution

### Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd Kavach
```

### Step 2 — Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Generate the Synthetic Dataset

From the repository root:

```bash
python data/generate.py
```

This creates the training and held-out test datasets under:

```text
eval/
├── synthetic_disputes_train.csv
└── synthetic_disputes_test.csv
```

### Step 5 — Train the Model

```bash
python classifier/train.py
```

This creates:

```text
classifier/model.pkl
```

The artifact contains the trained model, categorical encoders, feature configuration, SHAP explainer, and decision-policy parameters.

### Step 6 — Run Evaluation

```bash
python eval/evaluate.py
```

This evaluates the trained model on the **held-out test dataset** and reports both classification and financial metrics.

### Step 7 — Start the API

```bash
uvicorn agent.main:app --reload
```

The service will be available at:

```text
http://127.0.0.1:8000
```

### Step 8 — Verify the Service

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "kavach"
}
```

### Step 9 — Submit a Test Dispute

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

The API should return:

```text
202 Accepted
```

The resulting decision and associated information will be persisted to:

```text
data/kavach.db
```

### Step 10 — Inspect the Persisted Decision

```bash
curl http://127.0.0.1:8000/disputes/dsp_live_demo_01
```

Recent disputes can also be inspected with:

```bash
curl "http://127.0.0.1:8000/disputes?limit=10"
```

### Step 11 — Launch the Terminal Operator Interface

```bash
python cli.py
```

Use the terminal interface to inspect the system and handle cases routed to manual review.

---

## 12. Example Decision

A strong-evidence dispute may produce:

```text
Amount: ₹12,500
P(win): 92.8%
EV: ₹11,564.66

Decision:
AUTO_CONTEST
```

The reasoning is:

```text
EV > 0
AND
P(win) ≥ 70%
```

A weaker case may instead be routed to:

```text
MANUAL_REVIEW
```

even if its expected value is positive, because the model does not have sufficient confidence for automatic contesting.

A low-value or negative-EV dispute can be:

```text
AUTO_ACCEPT
```

without consuming resources on arbitration.

---

## 13. Testing the Decision Engine

The repository includes decision-engine tests:

```bash
python test_decision.py
```

The test cases cover the core policy boundaries, including scenarios for:

* Strong contest candidates
* Manual-review candidates
* Low-value disputes
* Negative-EV disputes

These tests verify that the deterministic decision layer behaves consistently around its configured thresholds.

---

## 14. Production Considerations

Kavach is intentionally implemented as a compact buildathon MVP.

Before production deployment, the following areas would require additional engineering:

### Durable event processing

The current API uses FastAPI's application-level background processing.

A production system should use a durable queue or event-streaming architecture with retry and failure handling.

Potential technologies include:

```text
Kafka / RabbitMQ
        +
Worker processes
```

### Persistent database

The current MVP uses local SQLite.

A production deployment would use a managed database such as:

```text
PostgreSQL
```

with appropriate backup, concurrency, retention, and access controls.

### Webhook security

Production webhook ingestion should include:

* Authentication
* Signature verification
* Replay protection
* Request validation
* Rate limiting

### Model governance

A production ML pipeline should additionally include:

* Probability calibration
* Model drift monitoring
* Feature-distribution monitoring
* Temporal validation
* Versioned model artifacts
* Champion/challenger evaluation
* Controlled deployment and rollback

### Human oversight

Automatic contesting should remain bounded by clearly defined policy thresholds and operational controls.

Manual review should remain available for uncertain or exceptional cases.

### Network-specific validation

Before a representment draft is actually submitted, the evidence package must be validated against the applicable network and acquirer requirements.

Kavach does not currently perform that external submission or certification step.

---

## 15. Retraining & Feedback Loop

Kavach persists dispute decisions and associated information so that eventual merchant outcomes can be used as **candidate feedback data**.

The intended lifecycle is:

```text
Dispute
   │
   ▼
Kavach Decision
   │
   ▼
Merchant Action
   │
   ▼
Actual Outcome
   │
   ▼
Historical Feedback Dataset
   │
   ▼
Offline Evaluation
   │
   ▼
Candidate Model
   │
   ▼
Metric Comparison
   │
   ▼
Human Approval
   │
   ▼
Production Model
```

The system does **not** automatically retrain or replace the deployed model from individual dispute outcomes.

This separation is intentional: model changes should be evaluated offline before deployment.

---

## 16. Design Principles

Kavach is built around several principles:

### Bounded automation

Automate only where the model and economics justify it.

### Economic decisioning

A high probability of winning is not enough. The expected financial value of contesting must also be positive.

### Explainability

Every ML prediction can be accompanied by feature-level SHAP explanations.

### Evidence grounding

Generated representment content is restricted to supplied merchant telemetry.

### Human-in-the-loop

Uncertain cases are routed to manual review rather than being forced into an automated decision.

### Reproducibility

The synthetic dataset generation, model training, and held-out evaluation pipelines are deterministic and can be rerun from the repository.

### Separation of concerns

Machine learning estimates risk; deterministic business logic decides what action to take.

---

**Built for the Razorpay AI Buildathon — September 2026.**