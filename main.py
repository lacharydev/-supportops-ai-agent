"""
SupportOps AI Agent — FastAPI Application
Main entry point. Defines all REST endpoints.

Run with:
    uvicorn app.main:app --reload --port 8000

Interactive docs at: http://localhost:8000/docs
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.models import (
    EscalationRequest, EscalationResponse,
    TicketBatch, PatternResponse,
    ActionRequest, ActionResponse,
    SentimentRequest, SentimentResponse,
    HealthResponse
)
from app.sentiment import classify_sentiment, classify_sentiment_mock
from app.summarizer import summarize_escalation
from app.pattern_detector import detect_patterns
from app.agent import get_next_best_action

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use mock mode if no API key is set (safe for testing/demo)
MOCK_MODE = not bool(os.getenv("OPENAI_API_KEY"))
if MOCK_MODE:
    logger.warning("No OPENAI_API_KEY found — running in MOCK MODE. "
                   "All LLM calls will return deterministic mock responses.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle manager."""
    logger.info(f"SupportOps AI Agent starting up (mock_mode={MOCK_MODE})")
    yield
    logger.info("SupportOps AI Agent shutting down.")


app = FastAPI(
    title="SupportOps AI Agent",
    description=(
        "LLM-powered support copilot that summarizes escalations, "
        "detects patterns, and suggests next-best actions using "
        "OpenAI GPT-4o + LangChain + DistilBERT sentiment analysis."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — confirms API is running and which models are active."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "gpt4o_summarizer": not MOCK_MODE,
            "langchain_agent": not MOCK_MODE,
            "distilbert_sentiment": True,  # always available
        }
    )


# ── Sentiment ──────────────────────────────────────────────────────────────

@app.post("/sentiment", response_model=SentimentResponse, tags=["Sentiment"])
async def analyze_sentiment(request: SentimentRequest):
    """
    Classify sentiment of customer text using DistilBERT.
    Returns FRUSTRATED / NEUTRAL / SATISFIED with confidence score.
    """
    try:
        if MOCK_MODE:
            return classify_sentiment_mock(request.text)
        return classify_sentiment(request.text)
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Escalation Summarizer ──────────────────────────────────────────────────

@app.post("/summarize", response_model=EscalationResponse, tags=["Escalations"])
async def summarize_ticket(request: EscalationRequest):
    """
    Summarize a support escalation ticket using GPT-4o.

    Combines sentiment analysis + LLM summarization to produce:
    - Executive-ready RCA summary
    - Error category classification
    - Suggested resolution actions
    - Priority score and escalation flag
    """
    try:
        # Step 1: Score customer sentiment
        sentiment_fn = classify_sentiment_mock if MOCK_MODE else classify_sentiment
        sentiment = sentiment_fn(request.description)

        # Step 2: Summarize with GPT-4o (or mock)
        summary_data = summarize_escalation(request, sentiment, mock=MOCK_MODE)

        return EscalationResponse(
            ticket_id=request.ticket_id,
            summary=summary_data["summary"],
            sentiment_score=sentiment.score,
            sentiment_label=sentiment.label,
            suggested_actions=summary_data["suggested_actions"],
            priority_score=summary_data["priority_score"],
            escalate=summary_data["escalate"],
            error_category=summary_data["error_category"]
        )

    except Exception as e:
        logger.error(f"Summarize error for ticket {request.ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Pattern Detection ──────────────────────────────────────────────────────

@app.post("/detect-patterns", response_model=PatternResponse, tags=["Patterns"])
async def detect_ticket_patterns(batch: TicketBatch):
    """
    Analyze a batch of tickets to surface recurring error patterns.

    Flags systemic issues when >= 2 customers are hitting the same
    root cause — enabling proactive outreach before more tickets arrive.
    """
    try:
        return detect_patterns(batch.tickets)
    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Next-Best Action ───────────────────────────────────────────────────────

@app.post("/next-best-action", response_model=ActionResponse, tags=["Agent"])
async def next_best_action(request: ActionRequest):
    """
    LangChain agent recommends next-best action for a support scenario.

    Uses GPT-4o + custom tools (error playbooks, SLA policies, escalation rules)
    to generate specific, actionable guidance for the support engineer.
    """
    try:
        return get_next_best_action(request, mock=MOCK_MODE)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
