import sys
from colorama import init, Fore, Back, Style

# Initialise colorama (handles Windows ANSI support automatically)
init(autoreset=True)

# ---------------------------------------------------------------------------
# RAZORPAY BLADE COLOUR TOKENS
# ---------------------------------------------------------------------------
# Primary
RZ_NAVY       = "\033[38;2;12;38;81m"        # #0C2651  Space Cadet
RZ_BLUE       = "\033[38;2;13;148;251m"       # #0D94FB  Dodger Blue
RZ_WHITE      = "\033[38;2;255;255;255m"      # #FFFFFF

# Backgrounds
BG_DARK       = "\033[48;2;8;18;38m"          # near-black navy canvas
BG_NAVY       = "\033[48;2;12;38;81m"         # #0C2651  header band
BG_CARD       = "\033[48;2;16;28;56m"         # slightly lighter card surface

# Semantic status (Blade pills)
RZ_GREEN      = "\033[38;2;39;174;96m"        # SUCCESS  — Auto Contest
RZ_ORANGE     = "\033[38;2;242;153;74m"       # PENDING  — Manual Review
RZ_RED        = "\033[38;2;235;87;87m"        # FAILED   — Auto Accept
RZ_PURPLE     = "\033[38;2;104;34;204m"       # HUMAN override (#6822CC)
RZ_MUTED      = "\033[38;2;120;140;180m"      # de-emphasised text

# Utility
RESET         = Style.RESET_ALL
BOLD          = Style.BRIGHT
DIM           = Style.DIM

# ---------------------------------------------------------------------------
# DECISION COLOUR MAP
# ---------------------------------------------------------------------------
DECISION_COLOR = {
    "AUTO_CONTEST":  RZ_GREEN,
    "MANUAL_REVIEW": RZ_ORANGE,
    "AUTO_ACCEPT":   RZ_RED,
    "HUMAN_CONTEST": RZ_PURPLE,
    "HUMAN_ACCEPT":  RZ_RED,
}

DECISION_LABEL = {
    "AUTO_CONTEST":  "AUTO CONTEST",
    "MANUAL_REVIEW": "MANUAL REVIEW",
    "AUTO_ACCEPT":   "AUTO ACCEPT",
    "HUMAN_CONTEST": "HUMAN CONTEST",
    "HUMAN_ACCEPT":  "HUMAN ACCEPT",
}

# ---------------------------------------------------------------------------
# LAYOUT HELPERS
# ---------------------------------------------------------------------------
WIDTH = 72

def clear():
    print("\033[2J\033[H", end="")


def rule(char="─", color=RZ_MUTED):
    print(f"{color}{char * WIDTH}{RESET}")


def thick_rule(char="═", color=RZ_BLUE):
    print(f"{color}{char * WIDTH}{RESET}")


def header(title: str):
    """Full-width Razorpay navy banner with centred title."""
    thick_rule()
    banner = f"{BG_NAVY}{RZ_WHITE}{BOLD}  {'KAVACH':^{WIDTH - 4}}  {RESET}"
    subtitle = f"{RZ_BLUE}  {'AI-Assisted Chargeback Decisioning & Review System':^{WIDTH - 4}}  {RESET}"
    print(banner)
    print(subtitle)
    thick_rule()
    if title:
        print(f"\n{RZ_BLUE}{BOLD}  {title}{RESET}\n")


def section(title: str):
    """Thin section divider with label."""
    print()
    rule()
    print(f"{RZ_BLUE}{BOLD}  {title}{RESET}")
    rule()


def kv(label: str, value: str, value_color: str = RZ_WHITE):
    """Key-value row, left-aligned label, coloured value."""
    print(f"  {RZ_MUTED}{label:<26}{RESET}{value_color}{value}{RESET}")


def currency(value) -> str:
    return f"₹{float(value):>12,.2f}"


def decision_badge(decision: str) -> str:
    color = DECISION_COLOR.get(decision, RZ_MUTED)
    label = DECISION_LABEL.get(decision, decision)
    return f"{color}{BOLD}[ {label} ]{RESET}"


def win_bar(prob: float, width: int = 20) -> str:
    """Simple ASCII progress bar for win probability."""
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    if prob >= 0.70:
        color = RZ_GREEN
    elif prob >= 0.40:
        color = RZ_ORANGE
    else:
        color = RZ_RED
    return f"{color}{bar}{RESET}  {prob * 100:.1f}%"


def prompt(text: str) -> str:
    return input(f"\n  {RZ_BLUE}›{RESET} {text} ").strip()


# ---------------------------------------------------------------------------
# LAZY IMPORTS  (keeps startup fast; avoids circular issues)
# ---------------------------------------------------------------------------
def _db():
    from agent.database import KavachDatabase
    return KavachDatabase()


def _letter():
    from agent.letter_generator import generate_representment_letter
    return generate_representment_letter


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
def show_dashboard():
    database = _db()
    disputes = database.get_recent_disputes(limit=100_000)

    total         = len(disputes)
    auto_contest  = sum(d.get("decision") == "AUTO_CONTEST"  for d in disputes)
    manual_review = sum(d.get("decision") == "MANUAL_REVIEW" for d in disputes)
    auto_accept   = sum(d.get("decision") == "AUTO_ACCEPT"   for d in disputes)
    human_contest = sum(d.get("decision") == "HUMAN_CONTEST" for d in disputes)
    human_accept  = sum(d.get("decision") == "HUMAN_ACCEPT"  for d in disputes)

    contested     = auto_contest + human_contest
    contest_rate  = contested / total if total > 0 else 0

    expected_recovery = sum(
        float(d.get("expected_value_inr") or 0)
        for d in disputes
        if d.get("decision") in {"AUTO_CONTEST", "HUMAN_CONTEST"}
    )

    probs = [float(d["prob_win"]) for d in disputes if d.get("prob_win") is not None]
    avg_prob = sum(probs) / len(probs) if probs else 0

    header("DASHBOARD")

    # Summary cards
    print(f"  {RZ_MUTED}Total Disputes{RESET}           "
          f"{RZ_WHITE}{BOLD}{total}{RESET}")
    print()
    print(f"  {RZ_GREEN}● Auto Contest{RESET}           {auto_contest}")
    print(f"  {RZ_PURPLE}● Human Contest{RESET}          {human_contest}")
    print(f"  {RZ_ORANGE}● Manual Review{RESET}          {manual_review}")
    print(f"  {RZ_RED}● Auto Accept{RESET}            {auto_accept}")
    print(f"  {RZ_RED}● Human Accept{RESET}           {human_accept}")
    print()
    rule()
    print()
    print(f"  {RZ_MUTED}Contest Rate{RESET}             "
          f"{win_bar(contest_rate, 16)}")
    print(f"  {RZ_MUTED}Expected Recovery{RESET}        "
          f"{RZ_GREEN}{BOLD}{currency(expected_recovery)}{RESET}")
    print(f"  {RZ_MUTED}Avg Win Probability{RESET}      "
          f"{win_bar(avg_prob, 16)}")

    section("RECENT DISPUTES")

    if not disputes:
        print(f"  {RZ_MUTED}No disputes recorded yet.{RESET}")
        return

    # Table header
    print(f"  {RZ_MUTED}{'DISPUTE ID':<26}{'AMOUNT':>14}{'WIN %':>8}  {'DECISION':<20}{RESET}")
    rule("─")

    for d in disputes[:10]:
        did      = (d.get("dispute_id") or "")[:25]
        amt      = currency(d.get("amount_inr", 0))
        prob     = float(d.get("prob_win", 0)) * 100
        dec      = d.get("decision", "UNKNOWN")
        color    = DECISION_COLOR.get(dec, RZ_MUTED)
        label    = DECISION_LABEL.get(dec, dec)

        if prob >= 70:
            prob_color = RZ_GREEN
        elif prob >= 40:
            prob_color = RZ_ORANGE
        else:
            prob_color = RZ_RED

        print(
            f"  {RZ_WHITE}{did:<26}{RESET}"
            f"{RZ_MUTED}{amt:>14}{RESET}"
            f"{prob_color}{prob:>7.1f}%{RESET}  "
            f"{color}{BOLD}{label:<20}{RESET}"
        )


# ---------------------------------------------------------------------------
# VIEW SINGLE DISPUTE
# ---------------------------------------------------------------------------
def show_dispute():
    database = _db()
    header("VIEW DISPUTE")
    did = prompt("Enter dispute ID:")

    if not did:
        print(f"\n  {RZ_RED}Dispute ID cannot be empty.{RESET}")
        return

    dispute = database.get_dispute(did)
    if dispute is None:
        print(f"\n  {RZ_RED}No dispute found:{RESET} {did}")
        return

    _render_dispute(dispute)


def _render_dispute(dispute):
    print()
    kv("Dispute ID",     dispute["dispute_id"])
    kv("Amount",         currency(dispute["amount_inr"]), RZ_WHITE)
    kv("Reason Code",    dispute["reason_code"],          RZ_BLUE)
    kv("Days Since Order", str(dispute["days_since_order"]))

    section("TRANSACTION EVIDENCE")

    def yesno(val, good_is_yes=True):
        yes = bool(val)
        if good_is_yes:
            color = RZ_GREEN if yes else RZ_RED
        else:
            color = RZ_RED if yes else RZ_GREEN
        return f"{color}{'YES' if yes else 'NO'}{RESET}"

    kv("Delivery Signature",  yesno(dispute["has_delivery_signature"]))
    kv("Device Hash Match",   yesno(dispute["device_hash_match"]))
    kv("AVS Match",           yesno(dispute["avs_match"]))
    kv("Past Disputes",
       f"{RZ_RED if dispute['customer_past_disputes'] > 0 else RZ_GREEN}"
       f"{dispute['customer_past_disputes']}{RESET}")
    kv("Payment Method",      dispute["payment_method"])
    kv("Weekend Transaction", yesno(dispute["is_weekend"], good_is_yes=False))
    kv("Merchant Category",   dispute["merchant_category"])

    section("RISK DECISION")

    dec     = dispute.get("decision", "UNKNOWN")
    prob    = float(dispute.get("prob_win", 0))
    ev      = float(dispute.get("expected_value_inr", 0))
    fee     = float(dispute.get("fee_assumed_inr", 500))

    print(f"\n  Decision         {decision_badge(dec)}")
    print(f"  Win Probability  {win_bar(prob)}")
    kv("Expected Value",
       f"{'₹' if ev >= 0 else '-₹'}{abs(ev):,.2f}",
       RZ_GREEN if ev >= 0 else RZ_RED)
    kv("Arbitration Fee", f"₹{fee:,.2f}", RZ_MUTED)

    print(f"\n  {RZ_MUTED}Reason:{RESET}")
    print(f"  {dispute.get('decision_reason', '—')}")

    # SHAP
    section("MODEL EXPLANATION — SHAP")
    shap_factors = dispute.get("shap_factors") or []

    if not shap_factors:
        print(f"  {RZ_MUTED}No SHAP explanation available.{RESET}")
    else:
        print(f"  {RZ_MUTED}{'FEATURE':<32}{'IMPACT':>12}  DIRECTION{RESET}")
        rule("─")
        for f in shap_factors:
            feat   = f.get("feature", "unknown")
            impact = float(f.get("impact", 0))
            dirn   = f.get("direction", "")
            color  = RZ_GREEN if dirn == "positive" else RZ_RED
            sign   = "+" if dirn == "positive" else "-"
            bar_w  = min(int(abs(impact) * 12), 16)
            bar    = "▌" * bar_w
            print(
                f"  {RZ_WHITE}{feat:<32}{RESET}"
                f"{color}{sign}{abs(impact):.4f}  {bar}{RESET}"
            )

    # Representment draft
    draft = dispute.get("representment_draft")
    if draft:
        section("REPRESENTMENT DRAFT")
        print()
        for line in draft.splitlines():
            print(f"  {RZ_MUTED}{line}{RESET}")


# ---------------------------------------------------------------------------
# MANUAL REVIEW QUEUE
# ---------------------------------------------------------------------------
def show_manual_review_queue():
    database = _db()

    while True:
        clear()
        header("MANUAL REVIEW QUEUE")

        reviews = database.get_manual_reviews()

        if not reviews:
            print(f"  {RZ_GREEN}✓  No disputes waiting for manual review.{RESET}\n")
            input(f"  {RZ_MUTED}Press Enter to return...{RESET}")
            return

        print(f"  {RZ_ORANGE}{BOLD}{len(reviews)} dispute(s) require human review.{RESET}\n")
        print(f"  {RZ_MUTED}{'#':<4}{'DISPUTE ID':<26}{'AMOUNT':>14}{'WIN %':>8}{'EV':>16}{RESET}")
        rule("─")

        for i, d in enumerate(reviews, 1):
            prob  = float(d["prob_win"]) * 100
            ev    = float(d.get("expected_value_inr", 0))
            print(
                f"  {RZ_ORANGE}{i:<4}{RESET}"
                f"{RZ_WHITE}{d['dispute_id'][:25]:<26}{RESET}"
                f"{RZ_MUTED}{currency(d['amount_inr']):>14}{RESET}"
                f"{RZ_ORANGE}{prob:>7.1f}%{RESET}"
                f"{RZ_GREEN}{currency(ev):>16}{RESET}"
            )

        choice = prompt("Select dispute # to review (B to go back):")

        if choice.lower() == "b":
            return

        try:
            idx = int(choice)
            if not (1 <= idx <= len(reviews)):
                raise ValueError
        except ValueError:
            print(f"\n  {RZ_RED}Invalid selection.{RESET}")
            input(f"  {RZ_MUTED}Press Enter...{RESET}")
            continue

        _review_dispute(reviews[idx - 1], database)


def _review_dispute(dispute, database):
    while True:
        clear()
        header(f"MANUAL REVIEW:  {dispute['dispute_id']}")

        full = database.get_dispute(dispute["dispute_id"])
        _render_dispute(full)

        section("HUMAN DECISION")
        print(f"  {RZ_GREEN}1.{RESET} Contest dispute")
        print(f"  {RZ_RED}2.{RESET} Accept dispute")
        print(f"  {RZ_MUTED}3.{RESET} Keep pending")
        print(f"  {RZ_MUTED}4.{RESET} Back")

        choice = prompt("Select an action:")

        if choice == "1":
            _human_contest(dispute["dispute_id"], database)
            return
        elif choice == "2":
            _human_accept(dispute["dispute_id"], database)
            return
        elif choice == "3":
            print(f"\n  {RZ_MUTED}Dispute remains in MANUAL REVIEW.{RESET}")
            input(f"  {RZ_MUTED}Press Enter...{RESET}")
            return
        elif choice == "4":
            return
        else:
            print(f"\n  {RZ_RED}Invalid option.{RESET}")
            input(f"  {RZ_MUTED}Press Enter...{RESET}")


def _human_contest(dispute_id: str, database):
    generate = _letter()
    print(f"\n  {RZ_ORANGE}You are overriding the automated MANUAL REVIEW decision.{RESET}")
    reason = prompt("Enter reason for contesting:")

    if not reason:
        print(f"\n  {RZ_RED}A reason is required.{RESET}")
        input(f"  {RZ_MUTED}Press Enter...{RESET}")
        return

    dispute = database.get_dispute(dispute_id)
    if dispute is None:
        print(f"\n  {RZ_RED}Dispute not found.{RESET}")
        input(f"  {RZ_MUTED}Press Enter...{RESET}")
        return

    representment = generate(dispute, dispute.get("shap_factors"))

    try:
        database.record_human_decision(
            dispute_id=dispute_id,
            new_decision="HUMAN_CONTEST",
            reason=reason,
        )
        with database._get_connection() as conn:
            conn.execute(
                "UPDATE disputes SET representment_draft = ?, updated_at = CURRENT_TIMESTAMP WHERE dispute_id = ?",
                (representment, dispute_id),
            )
            conn.commit()
    except ValueError as exc:
        print(f"\n  {RZ_RED}Unable to record decision: {exc}{RESET}")
        input(f"  {RZ_MUTED}Press Enter...{RESET}")
        return

    thick_rule()
    print(f"{RZ_GREEN}{BOLD}  ✓  HUMAN CONTEST RECORDED — {dispute_id}{RESET}")
    thick_rule()
    print(f"\n  {RZ_MUTED}Representment draft generated:{RESET}\n")
    for line in representment.splitlines():
        print(f"  {RZ_MUTED}{line}{RESET}")
    print()
    input(f"  {RZ_MUTED}Press Enter to return...{RESET}")


def _human_accept(dispute_id: str, database):
    print(f"\n  {RZ_ORANGE}You are accepting despite a positive expected value.{RESET}")
    reason = prompt("Enter reason for accepting:")

    if not reason:
        print(f"\n  {RZ_RED}A reason is required.{RESET}")
        input(f"  {RZ_MUTED}Press Enter...{RESET}")
        return

    try:
        database.record_human_decision(
            dispute_id=dispute_id,
            new_decision="HUMAN_ACCEPT",
            reason=reason,
        )
    except ValueError as exc:
        print(f"\n  {RZ_RED}Unable to record decision: {exc}{RESET}")
        input(f"  {RZ_MUTED}Press Enter...{RESET}")
        return

    thick_rule()
    print(f"{RZ_RED}{BOLD}  ✗  HUMAN ACCEPT RECORDED — {dispute_id}{RESET}")
    thick_rule()
    print()
    input(f"  {RZ_MUTED}Press Enter to return...{RESET}")


# ---------------------------------------------------------------------------
# RECENT DISPUTES
# ---------------------------------------------------------------------------
def show_recent_disputes():
    database = _db()
    header("RECENT DISPUTES")

    disputes = database.get_recent_disputes(limit=20)

    if not disputes:
        print(f"  {RZ_MUTED}No disputes recorded yet.{RESET}\n")
        return

    print(f"  {RZ_MUTED}{'#':<4}{'DISPUTE ID':<26}{'AMOUNT':>14}{'WIN %':>8}  {'DECISION'}{RESET}")
    rule("─")

    for i, d in enumerate(disputes, 1):
        dec   = d.get("decision", "UNKNOWN")
        color = DECISION_COLOR.get(dec, RZ_MUTED)
        label = DECISION_LABEL.get(dec, dec)
        prob  = float(d.get("prob_win", 0)) * 100

        print(
            f"  {RZ_MUTED}{i:<4}{RESET}"
            f"{RZ_WHITE}{(d.get('dispute_id') or '')[:25]:<26}{RESET}"
            f"{RZ_MUTED}{currency(d.get('amount_inr', 0)):>14}{RESET}"
            f"{RZ_MUTED}{prob:>7.1f}%{RESET}  "
            f"{color}{BOLD}{label}{RESET}"
        )


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------
MENU_ITEMS = [
    ("1", "Dashboard",           show_dashboard),
    ("2", "Manual Review Queue", show_manual_review_queue),
    ("3", "View Dispute",        show_dispute),
    ("4", "Recent Disputes",     show_recent_disputes),
    ("5", "Exit",                None),
]

def main():
    while True:
        clear()
        thick_rule()
        print(f"{BG_NAVY}{RZ_WHITE}{BOLD}{'':2}{'KAVACH':^{WIDTH - 4}}{'':2}{RESET}")
        print(f"{BG_NAVY}{RZ_BLUE}{'':2}{'AI-Assisted Chargeback Decisioning & Review System':^{WIDTH - 4}}{'':2}{RESET}")
        thick_rule()
        print()

        for key, label, _ in MENU_ITEMS:
            if key == "5":
                print(f"  {RZ_MUTED}{key}.{RESET}  {RZ_MUTED}{label}{RESET}")
            else:
                print(f"  {RZ_BLUE}{key}.{RESET}  {RZ_WHITE}{label}{RESET}")

        print()
        choice = prompt("Select an option:")

        if choice == "5":
            print(f"\n  {RZ_MUTED}Exiting Kavach. Goodbye.{RESET}\n")
            sys.exit(0)

        action_map = {key: fn for key, _, fn in MENU_ITEMS if fn}

        if choice in action_map:
            clear()
            action_map[choice]()
            print()
            input(f"  {RZ_MUTED}Press Enter to return to menu...{RESET}")
        else:
            print(f"\n  {RZ_RED}Invalid option. Try 1–5.{RESET}")
            input(f"  {RZ_MUTED}Press Enter...{RESET}")


if __name__ == "__main__":
    main()