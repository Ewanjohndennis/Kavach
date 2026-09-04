import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from agent.decision import RiskDecisionEngine
from agent.letter_generator import generate_representment_letter
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../classifier/model.pkl")

# Configure logging to simulate our "database/observability" for the hackathon demo
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Kavach Agent",
    description="Autonomous Chargeback Dispute & Defense Engine. Evaluates expected value of disputes and generates compliant representment packages.",
    version="1.0.0"
)

# Global engine instance (Loaded once on startup to save memory/latency)
try:
    engine = RiskDecisionEngine(model_path=MODEL_PATH)
    logger.info("RiskDecisionEngine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load RiskDecisionEngine: {e}")
    engine = None

# --- Pydantic Data Models ---
class DisputeEvent(BaseModel):
    dispute_id: str = Field(..., description="Unique identifier from the payment network")
    amount_inr: float = Field(..., description="Transaction amount in INR")
    reason_code: str = Field(..., description="Network reason code, e.g., Visa_10.4")
    
    # Telemetry & Risk Features
    days_since_order: int
    has_delivery_signature: int = Field(..., ge=0, le=1)
    device_hash_match: int = Field(..., ge=0, le=1)
    avs_match: int = Field(..., ge=0, le=1)
    customer_past_disputes: int
    payment_method: str = Field(..., description="E.g., 'UPI', 'Credit Card'")
    is_weekend: int = Field(..., ge=0, le=1)
    merchant_category: str = Field(..., description="E.g., 'Electronics'")

class WebhookResponse(BaseModel):
    status: str
    dispute_id: str
    message: str

# --- Core Async Processing Logic ---
def process_dispute_async(dispute: DisputeEvent):
    """
    Background worker function. 
    At scale, this sits in a Celery worker pulling from a Kafka queue.
    """
    logger.info(f"Processing Dispute {dispute.dispute_id} | Amount: ₹{dispute.amount_inr}")
    
    dispute_dict = dispute.model_dump()
    
    # 1. Run the Financial Decision Rule (XGBoost + EV calculation)
    try:
        evaluation = engine.evaluate_dispute(dispute_dict)
    except Exception as e:
        logger.error(f"[{dispute.dispute_id}] ML Engine failed: {e}")
        return

    logger.info(f"[{dispute.dispute_id}] Win Probability: {evaluation['prob_win']:.2f} | EV: ₹{evaluation['expected_value_inr']}")
    
    # 2. Action based on Expected Value
    if evaluation['decision'] == "AUTO_ACCEPT":
        logger.info(f"[{dispute.dispute_id}] DECISION: AUTO_ACCEPT. Saved ₹{evaluation['fee_assumed_inr']} in arbitration fees.")
        # Simulating saving to DB
        return

    # 3. Trigger LLM Evidence Compilation for CONTEST decisions
    logger.info(f"[{dispute.dispute_id}] DECISION: CONTEST. Generating evidence package...")
    letter = generate_representment_letter(dispute_dict)
    
    # Simulating uploading PDF to Visa/Mastercard APIs and saving to Postgres
    logger.info(f"[{dispute.dispute_id}] Representment Package Generated Successfully.\n\n--- PREVIEW ---\n{letter}\n--- END PREVIEW ---\n")

# --- API Endpoints ---
@app.get("/health")
def health_check():
    if not engine:
        raise HTTPException(status_code=503, detail="Model engine not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/webhook/dispute", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def handle_dispute_webhook(dispute: DisputeEvent, background_tasks: BackgroundTasks):
    """
    Ingests network disputes. 
    Returns 202 Accepted immediately to prevent webhook timeouts during spikes.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Service unavailable: Model uninitialized.")

    # Hand off the heavy ML/LLM lifting to the background task
    background_tasks.add_task(process_dispute_async, dispute)
    
    return WebhookResponse(
        status="accepted",
        dispute_id=dispute.dispute_id,
        message="Dispute queued for risk evaluation and defense processing."
    )