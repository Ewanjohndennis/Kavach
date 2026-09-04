import json
import os
# from openai import OpenAI # Uncomment in production

def generate_representment_letter(dispute_data: dict) -> str:
    """
    Template-Grounded Generation for Visa 10.4.
    Enforces strict structural constraints on the LLM to prevent hallucination.
    """
    if dispute_data.get("reason_code") != "Visa_10.4":
        return "Manual review required. Reason code outside automated scope."

    # 1. Deterministic Fact Extraction
    # We strictly bound what the LLM is allowed to know about the case.
    verified_facts = {
        "dispute_id": dispute_data.get("dispute_id"),
        "amount": f"INR {dispute_data.get('amount_inr')}",
        "delivery_signature_present": "Yes" if dispute_data.get("has_delivery_signature") else "No",
        "device_fingerprint_match": "Yes" if dispute_data.get("device_hash_match") else "No",
        "address_verification_match": "Yes" if dispute_data.get("avs_match") else "No",
        "days_since_order": dispute_data.get("days_since_order")
    }

    # 2. Strict System Prompting
    system_prompt = (
        "You are an expert payment dispute specialist. Your job is to draft a formal "
        "chargeback representment letter for a Visa 10.4 (Other Fraud - Card Absent Environment) dispute.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. You must ONLY use the facts provided in the JSON.\n"
        "2. DO NOT invent tracking numbers, customer names, or dates.\n"
        "3. Keep the tone strictly professional, concise, and objective.\n"
        "4. Structure with a clear header, body referencing the specific evidence, and a conclusion requesting reversal."
    )
    user_prompt = f"Draft the representment letter using ONLY these verified facts:\n{json.dumps(verified_facts, indent=2)}"

    # 3. LLM Execution (Mocked for hackathon reliability/speed)
    # ---------------------------------------------------------
    # In production: 
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     temperature=0.0 # Zero temperature for compliance-based outputs
    # )
    # return response.choices[0].message.content
    # ---------------------------------------------------------

    # Simulated LLM Output based strictly on the structured prompt guidelines:
    letter = f"""
CHARGEBACK REPRESENTMENT: VISA REASON CODE 10.4
Dispute ID: {verified_facts['dispute_id']}
Transaction Amount: {verified_facts['amount']}

To the Arbitration Committee:

We are contesting the chargeback filed under Visa Reason Code 10.4 (Other Fraud - Card Absent Environment). 
We maintain that this transaction was legitimately authorized and fulfilled. We have compiled the following 
verified telemetry and fulfillment data for your review:

• Address Verification System (AVS): {'MATCHED' if verified_facts['address_verification_match'] == 'Yes' else 'UNAVAILABLE'}
• Device Fingerprint Validation: {'MATCHED' if verified_facts['device_fingerprint_match'] == 'Yes' else 'UNAVAILABLE'}
• Proof of Delivery (Signature): {'ACQUIRED' if verified_facts['delivery_signature_present'] == 'Yes' else 'NOT APPLICABLE'}

The transaction occurred {verified_facts['days_since_order']} days prior to the dispute filing. 
Based on the compelling evidence of matching device telemetry and valid fulfillment, we respectfully 
request that this chargeback be reversed and liability placed back on the issuer.

Sincerely,
Kavach Merchant Defense
"""
    return letter.strip()