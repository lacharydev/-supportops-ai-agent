"""
Unit tests for SupportOps AI Agent.
All tests run in mock mode — no API key needed.
Run with: pytest tests/ -v
"""

import pytest
from app.models import EscalationRequest, ActionRequest, SentimentLabel, Priority
from app.sentiment import classify_sentiment_mock
from app.summarizer import summarize_escalation, _mock_summary
from app.pattern_detector import detect_patterns, detect_category
from app.agent import get_next_best_action


# ── Sentiment Tests ────────────────────────────────────────────────────────

class TestSentiment:
    def test_frustrated_customer(self):
        text = "This is terrible! My app has been down for hours and nobody is responding. Critical issue!"
        result = classify_sentiment_mock(text)
        assert result.label == SentimentLabel.FRUSTRATED
        assert result.frustration_level == "high"

    def test_neutral_customer(self):
        text = "Getting some timeout errors occasionally."
        result = classify_sentiment_mock(text)
        assert result.label in [SentimentLabel.NEUTRAL, SentimentLabel.SATISFIED]

    def test_satisfied_customer(self):
        text = "Just wanted to check on the status of my request."
        result = classify_sentiment_mock(text)
        assert result.frustration_level in ["low", "medium"]

    def test_score_range(self):
        result = classify_sentiment_mock("My API is broken and still not working!")
        assert 0.0 <= result.score <= 1.0


# ── Summarizer Tests ───────────────────────────────────────────────────────

class TestSummarizer:
    def _make_request(self, description, logs=None):
        return EscalationRequest(
            ticket_id="T-TEST",
            customer_id="test_corp",
            description=description,
            error_logs=logs or [],
            priority=Priority.P2
        )

    def test_detects_rate_limit(self):
        req = self._make_request(
            "We're getting 429 errors on every request",
            ["status=429", "status=429"]
        )
        from app.models import SentimentResponse
        sentiment = SentimentResponse(
            text="frustrated", label=SentimentLabel.FRUSTRATED,
            score=0.9, frustration_level="high"
        )
        result = _mock_summary(req, sentiment)
        assert result["error_category"] == "rate_limit"
        assert len(result["suggested_actions"]) > 0
        assert result["priority_score"] >= 7.0
        assert result["escalate"] == True

    def test_detects_auth_failure(self):
        req = self._make_request("Getting 401 unauthorized errors", ["status=401"])
        from app.models import SentimentResponse
        sentiment = SentimentResponse(
            text="neutral", label=SentimentLabel.NEUTRAL,
            score=0.6, frustration_level="medium"
        )
        result = _mock_summary(req, sentiment)
        assert result["error_category"] == "auth_failure"

    def test_summary_never_empty(self):
        req = self._make_request("Something is wrong")
        from app.models import SentimentResponse
        sentiment = SentimentResponse(
            text="neutral", label=SentimentLabel.NEUTRAL,
            score=0.5, frustration_level="low"
        )
        result = _mock_summary(req, sentiment)
        assert len(result["summary"]) > 10


# ── Pattern Detection Tests ────────────────────────────────────────────────

class TestPatternDetection:
    def _ticket(self, customer_id, description, logs=None):
        return EscalationRequest(
            ticket_id=f"T-{customer_id}",
            customer_id=customer_id,
            description=description,
            error_logs=logs or []
        )

    def test_detects_rate_limit_pattern(self):
        tickets = [
            self._ticket("acme", "Getting 429 too many requests", ["status=429"]),
            self._ticket("globex", "Rate limit errors all morning", ["status=429"]),
            self._ticket("initech", "API throttling our requests", ["status=429"]),
        ]
        result = detect_patterns(tickets)
        assert result.systemic_issue_detected == True
        assert result.top_error_category == "rate_limit"
        rate_limit_patterns = [p for p in result.patterns_found if p.pattern_type == "rate_limit"]
        assert len(rate_limit_patterns) > 0
        assert len(rate_limit_patterns[0].affected_customers) >= 2

    def test_single_customer_not_systemic(self):
        tickets = [
            self._ticket("acme", "Getting 429 errors", ["status=429"]),
        ]
        result = detect_patterns(tickets)
        assert result.systemic_issue_detected == False

    def test_mixed_errors(self):
        tickets = [
            self._ticket("acme", "429 rate limit", ["status=429"]),
            self._ticket("globex", "401 auth failure", ["status=401"]),
            self._ticket("initech", "429 too many requests", ["status=429"]),
        ]
        result = detect_patterns(tickets)
        assert result.total_tickets == 3
        assert result.top_error_category == "rate_limit"


# ── Agent Tests ────────────────────────────────────────────────────────────

class TestAgent:
    def test_mock_action_returns_response(self):
        request = ActionRequest(
            error_category="rate_limit",
            customer_tier="enterprise",
            sentiment_label=SentimentLabel.FRUSTRATED
        )
        result = get_next_best_action(request, mock=True)
        assert result.error_category == "rate_limit"
        assert len(result.recommended_actions) > 0
        assert len(result.customer_message_template) > 0

    def test_frustrated_enterprise_escalates(self):
        request = ActionRequest(
            error_category="server_error",
            customer_tier="enterprise",
            sentiment_label=SentimentLabel.FRUSTRATED
        )
        result = get_next_best_action(request, mock=True)
        assert result.escalate_to_engineering == True

    def test_all_error_categories(self):
        categories = ["rate_limit", "auth_failure", "context_overflow",
                      "server_error", "invalid_request"]
        for cat in categories:
            request = ActionRequest(
                error_category=cat,
                customer_tier="standard",
                sentiment_label=SentimentLabel.NEUTRAL
            )
            result = get_next_best_action(request, mock=True)
            assert result is not None
            assert result.error_category == cat
