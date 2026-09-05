import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.decision import RiskDecisionEngine
from agent.letter_generator import generate_representment_letter
from agent.database import KavachDatabase


# ============================================================
# Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger("kavach")

SUPPORTED_REASON_CODES = {"Visa_10.4"}


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Kavach",
    description="AI-assisted chargeback decisioning and representment system",
    version="1.0.0"
)


# ============================================================
# Components
# ============================================================

decision_engine = RiskDecisionEngine()
database = KavachDatabase()


# ============================================================
# Request schema
# ============================================================

class DisputeRequest(BaseModel):
    dispute_id: str = Field(..., min_length=1)
    amount_inr: float = Field(..., ge=0)

    reason_code: str

    days_since_order: int = Field(..., ge=0)

    has_delivery_signature: int = Field(..., ge=0, le=1)
    device_hash_match: int = Field(..., ge=0, le=1)
    avs_match: int = Field(..., ge=0, le=1)

    customer_past_disputes: int = Field(..., ge=0)

    payment_method: str
    is_weekend: int = Field(..., ge=0, le=1)

    merchant_category: str


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kavach"
    }


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "kavach",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# Dispute webhook
# ============================================================

@app.post("/webhook/dispute", status_code=202)
def process_dispute(dispute: DisputeRequest):

    dispute_data = dispute.model_dump()

    dispute_id = dispute_data["dispute_id"]
    amount = dispute_data["amount_inr"]
    reason_code = dispute_data["reason_code"]

    logger.info(
        f"Processing Dispute {dispute_id} | "
        f"Amount: ₹{amount:,.2f}"
    )

    # --------------------------------------------------------
    # Reason-code boundary
    # --------------------------------------------------------

    if reason_code not in SUPPORTED_REASON_CODES:

        logger.warning(
            f"[{dispute_id}] Unsupported reason code: {reason_code}"
        )

        raise HTTPException(
            status_code=422,
            detail=(
                f"Reason code '{reason_code}' is outside "
                f"Kavach's automated scope."
            )
        )

    # --------------------------------------------------------
    # Risk evaluation
    # --------------------------------------------------------

    try:
        decision = decision_engine.evaluate_dispute(
            dispute_data
        )

    except (ValueError, KeyError) as exc:

        logger.error(
            f"[{dispute_id}] Decision evaluation failed: {exc}"
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    prob_win = decision["prob_win"]
    expected_value = decision["expected_value_inr"]

    logger.info(
        f"[{dispute_id}] Win Probability: "
        f"{prob_win * 100:.1f}% | "
        f"EV: ₹{expected_value:,.2f}"
    )

    logger.info(
        f"[{dispute_id}] RATIONALE: "
        f"{decision['reason']}"
    )

    # --------------------------------------------------------
    # Representment generation
    # --------------------------------------------------------

    representment_draft = None

    if decision["decision"] == "AUTO_CONTEST":

        logger.info(
            f"[{dispute_id}] ACTION: AUTO_CONTEST. "
            f"High confidence + Positive EV. "
            f"Compiling evidence..."
        )

        representment_draft = generate_representment_letter(
            dispute_data,
            decision.get("shap_factors")
        )

        logger.info(
            f"\n"
            f"================ REPRESENTMENT PACKAGE "
            f"[{dispute_id}] ================\n"
            f"{representment_draft}\n"
            f"========================================================================="
        )

    elif decision["decision"] == "MANUAL_REVIEW":

        logger.info(
            f"[{dispute_id}] ACTION: MANUAL_REVIEW REQUIRED. "
            f"Routed to human queue."
        )

    elif decision["decision"] == "AUTO_ACCEPT":

        logger.info(
            f"[{dispute_id}] ACTION: AUTO_ACCEPTED. "
            f"Bypassed arbitration to save fee costs."
        )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    try:

        database.save_dispute(
            dispute_data=dispute_data,
            decision_data=decision,
            representment_draft=representment_draft
        )

    except Exception as exc:

        logger.exception(
            f"[{dispute_id}] Failed to persist dispute: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to persist dispute decision."
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "status": "accepted",
        "dispute_id": dispute_id,
        "message": (
            "Dispute queued for three-way risk evaluation "
            "and terminal logging."
        )
    }


# ============================================================
# Retrieve a dispute
# ============================================================

@app.get("/disputes/{dispute_id}")
def get_dispute(dispute_id: str):

    dispute = database.get_dispute(dispute_id)

    if dispute is None:

        raise HTTPException(
            status_code=404,
            detail=f"Dispute '{dispute_id}' not found."
        )

    return {
        "status": "success",
        "dispute": dispute
    }


# ============================================================
# Retrieve recent disputes
# ============================================================

@app.get("/disputes")
def get_recent_disputes(limit: int = 20):

    if limit <= 0:
        raise HTTPException(
            status_code=400,
            detail="limit must be greater than zero."
        )

    if limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit cannot exceed 100."
        )

    disputes = database.get_recent_disputes(limit)

    return {
        "status": "success",
        "count": len(disputes),
        "disputes": disputes
    }
# ============================================================
# Dashboard summary
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():

    disputes = database.get_recent_disputes(limit=100)

    total_disputes = len(disputes)

    auto_contest = sum(
        1 for d in disputes
        if d.get("decision") == "AUTO_CONTEST"
    )

    manual_review = sum(
        1 for d in disputes
        if d.get("decision") == "MANUAL_REVIEW"
    )

    auto_accept = sum(
        1 for d in disputes
        if d.get("decision") == "AUTO_ACCEPT"
    )

    total_expected_recovery = sum(
        float(d.get("expected_value_inr") or 0)
        for d in disputes
    )

    average_win_probability = (
        sum(
            float(d.get("prob_win") or 0)
            for d in disputes
        ) / total_disputes
        if total_disputes > 0
        else 0
    )

    contest_rate = (
        auto_contest / total_disputes
        if total_disputes > 0
        else 0
    )

    return {
        "total_disputes": total_disputes,

        "decisions": {
            "auto_contest": auto_contest,
            "manual_review": manual_review,
            "auto_accept": auto_accept
        },

        "contest_rate": round(contest_rate, 4),

        "total_expected_recovery_inr": round(
            total_expected_recovery,
            2
        ),

        "average_win_probability": round(
            average_win_probability,
            4
        ),

        "recent_disputes": disputes[:10]
    }