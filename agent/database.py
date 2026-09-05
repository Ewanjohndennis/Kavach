import json
import os
import sqlite3
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../data/kavach.db")
)


class KavachDatabase:
    """
    Persistence layer for Kavach dispute decisions.

    Responsibilities:
    - Create and maintain dispute records.
    - Persist incoming dispute information.
    - Persist ML risk decisions.
    - Persist SHAP explanations.
    - Persist representment drafts.
    - Retrieve previously processed disputes.
    - Manage manual-review decisions.
    - Maintain an audit trail of human actions.

    This layer contains no ML or business-decision logic.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

        os.makedirs(
            os.path.dirname(os.path.abspath(self.db_path)),
            exist_ok=True
        )

        self._initialize_database()

    def _get_connection(self):
        """
        Create a new SQLite connection per operation.
        """

        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        """
        Create the required tables if they do not already exist.

        Existing databases are preserved.
        """

        with self._get_connection() as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS disputes (
                    dispute_id TEXT PRIMARY KEY,

                    amount_inr REAL NOT NULL,
                    reason_code TEXT NOT NULL,

                    days_since_order INTEGER,
                    has_delivery_signature INTEGER,
                    device_hash_match INTEGER,
                    avs_match INTEGER,
                    customer_past_disputes INTEGER,

                    payment_method TEXT,
                    is_weekend INTEGER,
                    merchant_category TEXT,

                    decision TEXT,
                    prob_win REAL,
                    expected_value_inr REAL,
                    fee_assumed_inr REAL,
                    decision_reason TEXT,

                    shap_factors TEXT,
                    representment_draft TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    dispute_id TEXT NOT NULL,

                    previous_decision TEXT,
                    new_decision TEXT NOT NULL,

                    actor TEXT NOT NULL,
                    reason TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (dispute_id)
                        REFERENCES disputes(dispute_id)
                )
                """
            )

            conn.commit()

    # ------------------------------------------------------------------
    # DISPUTE PERSISTENCE
    # ------------------------------------------------------------------

    def save_dispute(
        self,
        dispute_data: dict,
        decision_data: Optional[dict] = None,
        representment_draft: Optional[str] = None
    ):
        """
        Persist a dispute and its decision.

        Existing disputes are updated rather than duplicated.
        """

        dispute_id = dispute_data.get("dispute_id")

        if not dispute_id:
            raise ValueError(
                "dispute_id is required for database persistence."
            )

        decision_data = decision_data or {}

        shap_factors = decision_data.get(
            "shap_factors",
            []
        )

        with self._get_connection() as conn:

            conn.execute(
                """
                INSERT INTO disputes (
                    dispute_id,
                    amount_inr,
                    reason_code,
                    days_since_order,
                    has_delivery_signature,
                    device_hash_match,
                    avs_match,
                    customer_past_disputes,
                    payment_method,
                    is_weekend,
                    merchant_category,
                    decision,
                    prob_win,
                    expected_value_inr,
                    fee_assumed_inr,
                    decision_reason,
                    shap_factors,
                    representment_draft
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT(dispute_id)
                DO UPDATE SET
                    amount_inr = excluded.amount_inr,
                    reason_code = excluded.reason_code,
                    days_since_order = excluded.days_since_order,
                    has_delivery_signature = excluded.has_delivery_signature,
                    device_hash_match = excluded.device_hash_match,
                    avs_match = excluded.avs_match,
                    customer_past_disputes = excluded.customer_past_disputes,
                    payment_method = excluded.payment_method,
                    is_weekend = excluded.is_weekend,
                    merchant_category = excluded.merchant_category,

                    decision = excluded.decision,
                    prob_win = excluded.prob_win,
                    expected_value_inr = excluded.expected_value_inr,
                    fee_assumed_inr = excluded.fee_assumed_inr,
                    decision_reason = excluded.decision_reason,
                    shap_factors = excluded.shap_factors,
                    representment_draft = excluded.representment_draft,

                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    dispute_id,
                    float(
                        dispute_data.get(
                            "amount_inr",
                            0.0
                        )
                    ),
                    dispute_data.get("reason_code"),

                    int(
                        dispute_data.get(
                            "days_since_order",
                            0
                        )
                    ),

                    int(
                        bool(
                            dispute_data.get(
                                "has_delivery_signature",
                                0
                            )
                        )
                    ),

                    int(
                        bool(
                            dispute_data.get(
                                "device_hash_match",
                                0
                            )
                        )
                    ),

                    int(
                        bool(
                            dispute_data.get(
                                "avs_match",
                                0
                            )
                        )
                    ),

                    int(
                        dispute_data.get(
                            "customer_past_disputes",
                            0
                        )
                    ),

                    dispute_data.get("payment_method"),

                    int(
                        bool(
                            dispute_data.get(
                                "is_weekend",
                                0
                            )
                        )
                    ),

                    dispute_data.get("merchant_category"),

                    decision_data.get("decision"),
                    decision_data.get("prob_win"),
                    decision_data.get(
                        "expected_value_inr"
                    ),
                    decision_data.get("fee_assumed_inr"),
                    decision_data.get("reason"),

                    json.dumps(
                        shap_factors,
                        separators=(",", ":")
                    ),

                    representment_draft
                )
            )

            conn.commit()

    # ------------------------------------------------------------------
    # SINGLE DISPUTE
    # ------------------------------------------------------------------

    def get_dispute(self, dispute_id: str):
        """
        Retrieve a dispute by dispute_id.

        Returns:
            dict | None
        """

        with self._get_connection() as conn:

            conn.row_factory = sqlite3.Row

            row = conn.execute(
                """
                SELECT *
                FROM disputes
                WHERE dispute_id = ?
                """,
                (dispute_id,)
            ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["shap_factors"] = self._decode_shap(
            result.get("shap_factors")
        )

        return result

    # ------------------------------------------------------------------
    # RECENT DISPUTES
    # ------------------------------------------------------------------

    def get_recent_disputes(self, limit: int = 20):
        """
        Retrieve the most recently processed disputes.
        """

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        with self._get_connection() as conn:

            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT *
                FROM disputes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        results = []

        for row in rows:

            result = dict(row)

            result["shap_factors"] = self._decode_shap(
                result.get("shap_factors")
            )

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # MANUAL REVIEW QUEUE
    # ------------------------------------------------------------------

    def get_manual_reviews(self):
        """
        Retrieve all disputes currently waiting for human review.
        """

        with self._get_connection() as conn:

            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT *
                FROM disputes
                WHERE decision = 'MANUAL_REVIEW'
                ORDER BY created_at ASC
                """
            ).fetchall()

        results = []

        for row in rows:

            result = dict(row)

            result["shap_factors"] = self._decode_shap(
                result.get("shap_factors")
            )

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # HUMAN DECISION
    # ------------------------------------------------------------------

    def record_human_decision(
        self,
        dispute_id: str,
        new_decision: str,
        reason: str,
        actor: str = "human_reviewer"
    ):
        """
        Record a human decision on a manually reviewed dispute.

        Valid decisions:
            HUMAN_CONTEST
            HUMAN_ACCEPT

        The previous decision must be MANUAL_REVIEW.
        """

        valid_decisions = {
            "HUMAN_CONTEST",
            "HUMAN_ACCEPT"
        }

        if new_decision not in valid_decisions:
            raise ValueError(
                f"Invalid human decision: {new_decision}"
            )

        if not reason or not reason.strip():
            raise ValueError(
                "A reason is required for a human decision."
            )

        with self._get_connection() as conn:

            conn.row_factory = sqlite3.Row

            row = conn.execute(
                """
                SELECT decision
                FROM disputes
                WHERE dispute_id = ?
                """,
                (dispute_id,)
            ).fetchone()

            if row is None:
                raise ValueError(
                    f"Dispute '{dispute_id}' not found."
                )

            previous_decision = row["decision"]

            if previous_decision != "MANUAL_REVIEW":
                raise ValueError(
                    f"Dispute '{dispute_id}' is not in "
                    f"MANUAL_REVIEW state."
                )

            conn.execute(
                """
                UPDATE disputes
                SET
                    decision = ?,
                    decision_reason = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE dispute_id = ?
                """,
                (
                    new_decision,
                    reason.strip(),
                    dispute_id
                )
            )

            conn.execute(
                """
                INSERT INTO decision_audit (
                    dispute_id,
                    previous_decision,
                    new_decision,
                    actor,
                    reason
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dispute_id,
                    previous_decision,
                    new_decision,
                    actor,
                    reason.strip()
                )
            )

            conn.commit()

    # ------------------------------------------------------------------
    # AUDIT HISTORY
    # ------------------------------------------------------------------

    def get_decision_history(self, dispute_id: str):
        """
        Retrieve the decision history for a dispute.
        """

        with self._get_connection() as conn:

            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT
                    id,
                    dispute_id,
                    previous_decision,
                    new_decision,
                    actor,
                    reason,
                    created_at
                FROM decision_audit
                WHERE dispute_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (dispute_id,)
            ).fetchall()

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_shap(value):
        """
        Safely decode stored SHAP JSON.
        """

        if not value:
            return []

        try:
            return json.loads(value)

        except (
            json.JSONDecodeError,
            TypeError
        ):
            return []