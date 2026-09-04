# Kavach 🛡️

**Razorpay AI Buildathon (Track 02: AI Risk Manager)**

Kavach (*meaning "Armor" or "Shield" in Sanskrit*) is an autonomous defense engine that stops Indian merchants from losing revenue to unrepresented chargebacks. Instead of generically detecting fraud at checkout, it operates post-transaction—ingesting dispute webhooks, predicting win probabilities, calculating financial expected value in INR, and compiling network-compliant representment evidence.

---

## ⚖️ How This Project Maps to the Judging Criteria

### 1. Problem Taste (Why This Matters)

Most fraud models focus on the transaction gateway. But when fraud slips through, or when legitimate customers commit "friendly fraud," chargebacks quietly eat merchant margins. Merchants have a strict 7–14 day window to submit a formal Representment Package. Because compiling shipping logs, IP matches, and PDF evidence takes 30–45 minutes per case, **merchants abandon up to 70% of winnable disputes.**

Kavach targets this specific, bounded revenue-loss vector. It automates the defense of Visa Reason Code 10.4 (Card Absent Fraud) to recover revenue that has already been captured but is at risk.

### 2. Build Quality (Honest Metrics & Architecture)

This project is built on financial pragmatism, not just ML accuracy.

* **The Expected Value (EV) Rule:** We don't contest every dispute. Arbitration costs money (₹500 flat fee). The engine calculates: `EV = (P_win * Amount) - (P_loss * Fee)`. If the EV is negative, the system auto-accepts the loss, saving the merchant from throwing good money after bad.
* **Defensible Evaluation:** The evaluation script doesn't just print Precision/Recall. It calculates the **Net Financial ROI** compared to a baseline "contest everything" strategy.
* **Earned Precision:** The synthetic dataset includes noise features and unobservable latent variables (like arbitrator leniency). The ML model has to work to find the true signal, proving the metrics are honest.

### 3. AI Judgment (Knowing When *Not* to Use AI)

This project demonstrates strict restraint regarding where AI is deployed:

* **No LLM Math:** We do not use LLMs to calculate ROI or determine if a dispute is worth fighting. That is deterministic Python arithmetic.
* **No RAG Hallucinations:** We explicitly avoided using RAG to "search" for evidence. Card networks heavily penalize fabricated evidence. Instead, we use **Template-Grounded Generation**. Deterministic Python extracts verified telemetry (AVS, IPs, Signatures), and the LLM is strictly constrained to formatting those exact facts into formal representment prose.
* **Right Tool for the Job:** We use XGBoost—not an LLM—for tabular risk scoring because it is faster, cheaper, and statistically interpretable.

### 4. Failure Recovery & Scale Awareness

This repository is a functioning synchronous MVP, but we recognize what breaks at scale.

* **The Single-Instance Bottleneck:** At 1M merchants, end-of-month webhook spikes will crash a standard FastAPI instance. **Our fix in this MVP:** The webhook endpoint immediately returns a `202 Accepted` and passes the ML/LLM workload to FastAPI `BackgroundTasks`.
* **Production Path:** In a true production environment, those `BackgroundTasks` would be replaced by Kafka (ingestion queue) and Celery workers (async processing). The in-memory SQLite feature store would migrate to Postgres/Redis, and the XGBoost model would be served via BentoML or MLflow.

---

## 🏗️ Repository Structure

```text
kavach/
│
├── data/
│   ├── generate.py              # Generates realistic synthetic disputes with latent noise
│   └── feature_store.db         # Local SQLite DB acting as the mock feature store
│
├── classifier/
│   ├── train.py                 # Trains XGBoost and outputs model artifact
│   └── model.pkl                # Saved artifact (Model + LabelEncoders + Feature array)
│
├── agent/
│   ├── main.py                  # Async FastAPI app (webhook endpoint)
│   ├── decision.py              # Loads model and executes the EV threshold math
│   └── letter_generator.py      # Template-grounded LLM Visa 10.4 representment builder
│
├── eval/
│   ├── evaluate.py              # Standalone script for running precision/recall/ROI metrics
│   └── synthetic_disputes_eval.csv  # 1000-row test set for judge evaluation
│
├── requirements.txt
└── .gitignore

```

---

## 🚀 Step-by-Step Installation & Execution Guide for Judges

### Step 1: Clone and Set Up Virtual Environment

Open your terminal inside the project directory and run:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### Step 2: Generate the Synthetic Data & Feature Store

Run the data generator to seed the evaluation test set and the local SQLite feature store:

```bash
cd data
python generate.py
cd ..

```

### Step 3: Train the XGBoost Classifier

Train the model on the generated data. This will create the `model.pkl` artifact inside the `classifier/` directory:

```bash
cd classifier
python train.py
cd ..

```

### Step 4: Run the Standalone Evaluation Script

To verify the model's performance and see the financial comparison (Baseline ROI vs. Kavach ROI), execute the evaluation script:

```bash
python eval/evaluate.py

```

*(This will print the precision, recall, and net financial improvement straight to your terminal).*

### Step 5: Start the Defense Agent API

Spin up the FastAPI asynchronous server:

```bash
uvicorn agent.main:app --reload

```

### Step 6: Trigger a Live Dispute Webhook

Open a separate terminal window and send a mock webhook to test the end-to-end event loop. Watch the FastAPI terminal output to see the asynchronous ML evaluation, EV calculation, and template-grounded letter generation happen in real-time:

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

---

**Built by Ewan John Dennis** for the Razorpay AI Buildathon (September 2026).