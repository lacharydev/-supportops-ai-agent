"""
Sentiment classifier using DistilBERT (SST-2).
Scores customer frustration level to prioritize support queue.

Same model as sentiment-analysis-pytorch-cloudrun project —
here integrated directly into the support agent pipeline.
"""

from transformers import pipeline
from functools import lru_cache
from app.models import SentimentLabel, SentimentResponse
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_pipeline():
    """
    Load DistilBERT sentiment pipeline once and cache it.
    lru_cache ensures we don't reload the model on every request.
    First call takes ~3 seconds; subsequent calls are instant.
    """
    logger.info("Loading DistilBERT sentiment model...")
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )


def classify_sentiment(text: str) -> SentimentResponse:
    """
    Classify sentiment of customer ticket text.

    Maps DistilBERT's POSITIVE/NEGATIVE output to support-relevant labels:
    - NEGATIVE + high confidence → FRUSTRATED (priority escalation)
    - NEGATIVE + low confidence  → NEUTRAL
    - POSITIVE                   → SATISFIED

    Args:
        text: Raw customer ticket description or message.

    Returns:
        SentimentResponse with label, score, and frustration level.
    """
    classifier = _load_pipeline()
    result = classifier(text[:512])[0]  # truncate to model max length

    raw_label = result["label"]   # "POSITIVE" or "NEGATIVE"
    confidence = result["score"]  # 0.0 - 1.0

    # Map to support-relevant sentiment labels
    if raw_label == "NEGATIVE" and confidence > 0.75:
        label = SentimentLabel.FRUSTRATED
        frustration_level = "high"
    elif raw_label == "NEGATIVE":
        label = SentimentLabel.NEUTRAL
        frustration_level = "medium"
    else:
        label = SentimentLabel.SATISFIED
        frustration_level = "low"

    return SentimentResponse(
        text=text[:100] + "..." if len(text) > 100 else text,
        label=label,
        score=round(confidence, 4),
        frustration_level=frustration_level
    )


def batch_classify(texts: list[str]) -> list[SentimentResponse]:
    """Classify sentiment for multiple texts efficiently."""
    return [classify_sentiment(t) for t in texts]


# ── Mock mode (no model download needed for testing) ──────────────────────

def classify_sentiment_mock(text: str) -> SentimentResponse:
    """
    Mock classifier for unit tests and CI — no model download needed.
    Detects frustration via simple keyword heuristics.
    """
    frustration_keywords = [
        "urgent", "down", "broken", "failing", "hours", "critical",
        "still not", "nobody", "unacceptable", "disaster", "terrible"
    ]
    text_lower = text.lower()
    hit_count = sum(1 for kw in frustration_keywords if kw in text_lower)

    if hit_count >= 3:
        return SentimentResponse(
            text=text[:100],
            label=SentimentLabel.FRUSTRATED,
            score=0.91,
            frustration_level="high"
        )
    elif hit_count >= 1:
        return SentimentResponse(
            text=text[:100],
            label=SentimentLabel.NEUTRAL,
            score=0.62,
            frustration_level="medium"
        )
    else:
        return SentimentResponse(
            text=text[:100],
            label=SentimentLabel.SATISFIED,
            score=0.85,
            frustration_level="low"
        )
