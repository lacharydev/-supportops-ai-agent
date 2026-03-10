"""
Pattern detector — identifies recurring error types across a batch of tickets.
Surfaces systemic issues before they become widespread outages.

This is the proactive support layer: instead of waiting for every customer
to file a ticket, this runs periodically over the ticket queue and flags
when multiple customers are hitting the same root cause.
"""

from collections import defaultdict, Counter
from app.models import EscalationRequest, PatternResult, PatternResponse


# Keyword → error category mapping
CATEGORY_SIGNALS = {
    "rate_limit":        ["429", "rate limit", "too many requests", "throttl", "rpm", "tpm"],
    "auth_failure":      ["401", "403", "unauthorized", "invalid key", "authentication", "api key"],
    "context_overflow":  ["context length", "token limit", "maximum context", "too long", "4096", "8192", "128k"],
    "server_error":      ["500", "503", "server error", "internal error", "service unavailable"],
    "invalid_request":   ["400", "bad request", "invalid", "malformed", "missing field"],
    "integration_failure": ["timeout", "connection", "network", "dns", "refused", "unreachable"],
}

RECOMMENDED_ACTIONS = {
    "rate_limit":        "Send proactive rate limit guide to all affected customers. Review tier upgrades.",
    "auth_failure":      "Check for platform-side auth issues. Verify API key generation is working.",
    "context_overflow":  "Publish context management best practices doc. Consider adding token counter to SDK.",
    "server_error":      "Escalate to engineering immediately if >3 customers affected. Check status page.",
    "invalid_request":   "Review recent API schema changes. Update documentation if breaking change.",
    "integration_failure": "Check network/DNS infrastructure. Notify SRE team if widespread.",
}


def detect_category(ticket: EscalationRequest) -> str:
    """
    Classify a single ticket into an error category using keyword matching.
    Checks both the description and error log lines.
    """
    search_text = (
        ticket.description.lower() + " " +
        " ".join(ticket.error_logs).lower()
    )

    scores = defaultdict(int)
    for category, signals in CATEGORY_SIGNALS.items():
        for signal in signals:
            if signal in search_text:
                scores[category] += 1

    if not scores:
        return "unknown"

    return max(scores, key=scores.__getitem__)


def detect_patterns(tickets: list[EscalationRequest]) -> PatternResponse:
    """
    Analyze a batch of tickets to surface systemic patterns.

    Algorithm:
    1. Classify each ticket into an error category
    2. Group by category
    3. Flag categories with >= 2 affected customers as patterns
    4. Generate recommended bulk action per pattern

    Args:
        tickets: List of escalation tickets to analyze.

    Returns:
        PatternResponse with detected patterns, affected customers, and recommendations.
    """
    # Classify all tickets
    category_to_customers = defaultdict(set)
    category_counts = Counter()

    for ticket in tickets:
        category = detect_category(ticket)
        category_to_customers[category].add(ticket.customer_id)
        category_counts[category] += 1

    # Build pattern results (only flag if >= 2 customers affected)
    patterns = []
    for category, customer_set in category_to_customers.items():
        if len(customer_set) >= 2:  # systemic threshold
            patterns.append(PatternResult(
                pattern_type=category,
                count=category_counts[category],
                affected_customers=sorted(list(customer_set)),
                recommended_action=RECOMMENDED_ACTIONS.get(
                    category,
                    "Manual investigation required."
                )
            ))

    # Sort by count descending — most common pattern first
    patterns.sort(key=lambda p: p.count, reverse=True)

    top_category = category_counts.most_common(1)[0][0] if category_counts else "unknown"
    systemic = any(len(p.affected_customers) >= 2 for p in patterns)

    return PatternResponse(
        total_tickets=len(tickets),
        patterns_found=patterns,
        top_error_category=top_category,
        systemic_issue_detected=systemic
    )
