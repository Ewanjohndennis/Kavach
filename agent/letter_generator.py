import json


SUPPORTED_REASON_CODE = "Visa_10.4"


def _format_evidence(facts: dict) -> list[str]:
    """
    Convert verified transaction facts into human-readable evidence lines.

    Only evidence that is actually present is described as available.
    Missing evidence is not represented as positive evidence.
    """

    evidence = []

    if facts["address_verification_match"]:
        evidence.append("Address Verification System (AVS): MATCHED")

    if facts["device_fingerprint_match"]:
        evidence.append("Device Fingerprint Match: MATCHED")

    if facts["delivery_signature_present"]:
        evidence.append("Delivery Signature: PRESENT")

    return evidence


def _build_evidence_summary(facts: dict) -> str:
    """
    Build a concise factual summary from verified evidence.
    """

    statements = []

    if facts["address_verification_match"]:
        statements.append(
            "The transaction record contains an AVS match."
        )

    if facts["device_fingerprint_match"]:
        statements.append(
            "The transaction record contains a matching device fingerprint."
        )

    if facts["delivery_signature_present"]:
        statements.append(
            "The merchant record contains a delivery signature."
        )

    if not statements:
        return (
            "No positive supporting evidence was identified from the "
            "available transaction fields."
        )

    return " ".join(statements)


def _build_shap_context(shap_factors: list | None) -> str:
    """
    SHAP values are model explanations, not representment evidence.

    They are therefore kept separate from the factual evidence section.
    """

    if not shap_factors:
        return ""

    top_factors = shap_factors[:4]

    lines = []

    for factor in top_factors:
        feature = factor.get("feature", "unknown")
        impact = float(factor.get("impact", 0.0))
        direction = factor.get("direction", "+")

        lines.append(
            f"- {feature}: impact={impact:.3f}, direction={direction}"
        )

    return (
        "\n\nMODEL EXPLANATION — NOT REPRESENTMENT EVIDENCE\n"
        "-------------------------------------------------\n"
        + "\n".join(lines)
    )


def generate_representment_letter(
    dispute_data: dict,
    shap_factors: list | None = None
) -> str:
    """
    Generate an evidence-grounded representment draft for Visa 10.4.

    Important design principles:
    - Only Visa 10.4 is supported.
    - Only supplied transaction facts are used as evidence.
    - SHAP explanations are kept separate from factual evidence.
    - The generator does not claim that any individual signal guarantees
      representment eligibility.
    - No unsupported evidence is invented.
    """

    if dispute_data.get("reason_code") != SUPPORTED_REASON_CODE:
        return (
            "Manual review required. "
            "Reason code outside automated scope."
        )

    verified_facts = {
        "dispute_id": dispute_data.get("dispute_id"),
        "amount_inr": float(dispute_data.get("amount_inr", 0.0)),
        "delivery_signature_present": bool(
            dispute_data.get("has_delivery_signature")
        ),
        "device_fingerprint_match": bool(
            dispute_data.get("device_hash_match")
        ),
        "address_verification_match": bool(
            dispute_data.get("avs_match")
        ),
        "days_since_order": int(
            dispute_data.get("days_since_order", 0)
        ),
    }

    evidence_lines = _format_evidence(verified_facts)
    evidence_summary = _build_evidence_summary(verified_facts)
    shap_context = _build_shap_context(shap_factors)

    if evidence_lines:
        evidence_section = "\n".join(
            f"• {line}" for line in evidence_lines
        )
    else:
        evidence_section = (
            "• No positive supporting evidence identified "
            "from the available transaction fields"
        )

    letter = f"""
CHARGEBACK REPRESENTMENT DRAFT
VISA REASON CODE 10.4

Dispute ID: {verified_facts["dispute_id"]}
Transaction Amount: INR {verified_facts["amount_inr"]:,.2f}

To the appropriate dispute review team:

We are submitting this transaction for review in connection with
Visa Reason Code 10.4.

The available merchant transaction records contain the following
verified information:

VERIFIED TRANSACTION EVIDENCE
-----------------------------
{evidence_section}

The transaction occurred {verified_facts["days_since_order"]} days
prior to the reported dispute.

EVIDENCE SUMMARY
----------------
{evidence_summary}

These transaction records are being provided as supporting
documentation for review. The evidence listed above reflects only
information currently available in the merchant's transaction
records.

The presence of these signals does not, by itself, establish that
the transaction satisfies every applicable representment
requirement. The supporting records should therefore be evaluated
against the applicable Visa requirements before submission.

This document is a system-generated draft. Any additional evidence
required for representment should be attached and validated by the
merchant or dispute operations team before submission.
{shap_context}

Sincerely,

Kavach Merchant Defense
"""

    return letter.strip()