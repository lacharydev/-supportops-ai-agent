"""
LangChain agent for next-best-action recommendations.

Uses GPT-4o as the backbone with custom tools that simulate
the knowledge base a support engineer would consult:
  - Error category lookup
  - Customer tier policy lookup
  - Escalation decision tool
"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from app.models import ActionRequest, ActionResponse, SentimentLabel

logger = logging.getLogger(__name__)


# ── LangChain Tools ────────────────────────────────────────────────────────

@tool
def lookup_error_playbook(error_category: str) -> str:
    """
    Look up the standard resolution playbook for a given error category.
    Returns step-by-step resolution actions.
    """
    playbooks = {
        "rate_limit": (
            "1. Confirm whether it's RPM (requests/min) or TPM (tokens/min) limit. "
            "2. Implement exponential backoff: wait 1s, 2s, 4s between retries. "
            "3. Add jitter to prevent thundering herd. "
            "4. Review usage dashboard for spike patterns. "
            "5. If persistent: recommend tier upgrade or request limit increase."
        ),
        "auth_failure": (
            "1. Verify API key format starts with 'sk-'. "
            "2. Check Authorization header: must be 'Bearer sk-...'. "
            "3. Confirm key hasn't been rotated or deleted in dashboard. "
            "4. Check org billing status — suspended orgs get 401s. "
            "5. Generate new key and test with minimal curl request."
        ),
        "context_overflow": (
            "1. Count tokens in the request using tiktoken. "
            "2. Check model's context limit (gpt-4o: 128k tokens). "
            "3. Implement sliding window: drop oldest messages first. "
            "4. Consider summarizing conversation history instead of full replay. "
            "5. Use embeddings + retrieval for long document Q&A instead of stuffing."
        ),
        "server_error": (
            "1. Check status.openai.com for active incidents. "
            "2. Retry with exponential backoff — 5xx errors are often transient. "
            "3. If persistent >5 min, capture request ID from response headers. "
            "4. Escalate to engineering with request ID + timestamp. "
            "5. Consider fallback to secondary model endpoint."
        ),
        "invalid_request": (
            "1. Validate JSON payload structure against API docs. "
            "2. Check message role values: must be 'system', 'user', or 'assistant'. "
            "3. Verify model string is correct (e.g. 'gpt-4o' not 'gpt4o'). "
            "4. Ensure 'content' field is present in all message objects. "
            "5. Test with minimal request in Postman/curl to isolate issue."
        ),
    }
    return playbooks.get(
        error_category,
        "No playbook found. Escalate to L2 engineering for manual investigation."
    )


@tool
def get_customer_tier_policy(customer_tier: str) -> str:
    """
    Get SLA and escalation policy for a customer tier.
    Returns response time SLAs and escalation thresholds.
    """
    policies = {
        "enterprise": (
            "Enterprise SLA: 1-hour response, 4-hour resolution for P1. "
            "Dedicated CSM must be notified for any P1. "
            "Auto-escalate to L2 engineering if unresolved >2 hours."
        ),
        "business": (
            "Business SLA: 4-hour response, 24-hour resolution for P1. "
            "Escalate to L2 if unresolved >8 hours."
        ),
        "standard": (
            "Standard SLA: 24-hour response. "
            "Community forum + documentation recommended for L1 issues. "
            "Escalate only if confirmed platform-side bug."
        ),
        "free": (
            "Free tier: best-effort support via community forum. "
            "No SLA guarantees. Direct email support not included."
        ),
    }
    return policies.get(
        customer_tier.lower(),
        "Unknown tier. Apply standard SLA until tier is confirmed."
    )


@tool
def check_escalation_needed(priority_score: float, sentiment: str, tier: str) -> str:
    """
    Determine whether immediate escalation to engineering is needed.
    Returns escalation decision with reasoning.
    """
    score = float(priority_score)
    is_enterprise = tier.lower() == "enterprise"
    is_frustrated = sentiment.upper() == "FRUSTRATED"

    if score >= 9.0 or (is_enterprise and score >= 7.0):
        return "ESCALATE_IMMEDIATELY: Page on-call engineering. Notify CSM. Start incident bridge."
    elif score >= 7.0 or (is_frustrated and score >= 5.0):
        return "ESCALATE_TO_L2: Assign to senior support engineer within 30 minutes."
    elif score >= 5.0:
        return "MONITOR: Handle at L1. Escalate if no resolution within SLA window."
    else:
        return "NO_ESCALATION: Standard L1 handling. Point to documentation."


# ── Agent Builder ──────────────────────────────────────────────────────────

def build_agent() -> AgentExecutor:
    """
    Build the LangChain agent with tools.
    Called once at startup and reused for all requests.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    tools = [lookup_error_playbook, get_customer_tier_policy, check_escalation_needed]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior AI support engineer assistant.
         
Given an error category, customer tier, and sentiment, use your tools to:
1. Look up the resolution playbook for this error type
2. Check the customer tier SLA policy  
3. Determine if escalation is needed

Then synthesize a clear, actionable recommendation.
Be specific and technical. Your output will be read by a support engineer
who needs to act immediately."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=4)


def get_next_best_action(request: ActionRequest, mock: bool = False) -> ActionResponse:
    """
    Run the LangChain agent to get next-best-action recommendation.

    Args:
        request: ActionRequest with error_category, customer_tier, sentiment.
        mock: If True, returns deterministic mock without calling OpenAI.

    Returns:
        ActionResponse with recommended_actions, customer_message_template,
        escalation decision, and agent reasoning.
    """
    if mock:
        return _mock_action(request)

    agent = build_agent()

    query = (
        f"Error category: {request.error_category}. "
        f"Customer tier: {request.customer_tier}. "
        f"Sentiment: {request.sentiment_label}. "
        f"Priority score: 7.5. "
        f"Additional context: {request.additional_context or 'none'}. "
        f"What should I do right now?"
    )

    try:
        result = agent.invoke({"input": query})
        agent_output = result.get("output", "")

        # Parse agent output into structured response
        escalate = any(word in agent_output.upper() for word in ["ESCALATE", "PAGE", "P1"])

        return ActionResponse(
            error_category=request.error_category,
            recommended_actions=_extract_actions(agent_output),
            customer_message_template=_build_customer_template(request.error_category),
            escalate_to_engineering=escalate,
            agent_reasoning=agent_output
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return _mock_action(request)


def _extract_actions(agent_output: str) -> list[str]:
    """Extract numbered action items from agent output."""
    lines = agent_output.split("\n")
    actions = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            # Clean up formatting
            clean = line.lstrip("0123456789.-•) ").strip()
            if clean and len(clean) > 10:
                actions.append(clean)
    return actions[:5] if actions else [agent_output[:200]]


def _build_customer_template(error_category: str) -> str:
    """Pre-written customer message templates by error category."""
    templates = {
        "rate_limit": (
            "Hi [Customer], thank you for reaching out. I can see your application "
            "is hitting our rate limits. To resolve this immediately, please implement "
            "exponential backoff in your retry logic (1s → 2s → 4s between retries). "
            "I've also flagged your account for a limit review. Let me know if you need "
            "code examples for the backoff implementation."
        ),
        "auth_failure": (
            "Hi [Customer], I can see you're getting authentication errors. "
            "Please rotate your API key at platform.openai.com/api-keys and update "
            "your application. If the issue persists, please confirm your org billing "
            "status is active. Happy to do a quick call to verify your setup."
        ),
        "server_error": (
            "Hi [Customer], we're aware of elevated error rates and are investigating. "
            "Please check status.openai.com for real-time updates. In the meantime, "
            "implementing retry logic with backoff will help your application recover "
            "automatically when the issue resolves."
        ),
        "context_overflow": (
            "Hi [Customer], your request is exceeding the model's context window. "
            "Please review the total token count of your messages array — each model "
            "has a different limit. I can suggest a sliding window or summarization "
            "approach to manage long conversations. Would a code example help?"
        ),
    }
    return templates.get(
        error_category,
        "Hi [Customer], thank you for your patience. I'm investigating this issue "
        "and will have an update for you shortly."
    )


def _mock_action(request: ActionRequest) -> ActionResponse:
    """Mock response for testing."""
    return ActionResponse(
        error_category=request.error_category,
        recommended_actions=[
            f"Look up {request.error_category} playbook",
            f"Apply {request.customer_tier} tier SLA: respond within defined window",
            "Implement retry logic with exponential backoff",
            "Document resolution in knowledge base"
        ],
        customer_message_template=_build_customer_template(request.error_category),
        escalate_to_engineering=(request.sentiment_label == SentimentLabel.FRUSTRATED),
        agent_reasoning=f"[Mock] Processed {request.error_category} for {request.customer_tier} customer."
    )
