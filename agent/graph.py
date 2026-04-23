"""
graph.py — LangGraph-based conversational agent for AutoStream.

Architecture:
  - AgentState: typed dict that persists across all turns
  - Three nodes: intent_router → rag_responder | lead_collector | greeter
  - Conditional edges route based on detected intent
  - State includes full message history (memory across 5-6 turns)
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .rag import get_knowledge_context, get_full_context
from .tools import LeadData, extract_email, extract_platform, mock_lead_capture


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # latest classified intent
    lead: LeadData                             # lead collection progress
    lead_captured: bool                        # whether tool was already fired


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)


# ── Intent detection ──────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing tool.

Classify the user's latest message into exactly one of:
- greeting      : simple hello, hi, how are you, small talk
- product_query : asking about features, pricing, plans, policies, how things work
- high_intent   : ready to sign up, wants to try, buy, upgrade, or get started

Reply with ONLY one word: greeting, product_query, or high_intent."""


def detect_intent(state: AgentState) -> AgentState:
    llm = _get_llm()
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    resp = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_human),
    ])
    intent = resp.content.strip().lower()
    if intent not in ("greeting", "product_query", "high_intent"):
        intent = "product_query"  # safe default
    return {**state, "intent": intent}


# ── Router ────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> Literal["greeter", "rag_responder", "lead_collector"]:
    # If already in lead collection, stay there until complete
    lead: LeadData = state.get("lead") or LeadData()
    if lead.name or lead.email or lead.platform:
        return "lead_collector"
    intent = state.get("intent", "product_query")
    if intent == "greeting":
        return "greeter"
    if intent == "high_intent":
        return "lead_collector"
    return "rag_responder"


# ── Greeter node ──────────────────────────────────────────────────────────────

GREETER_SYSTEM = """You are a friendly sales assistant for AutoStream, an AI-powered video
editing SaaS for content creators. Keep greetings warm, brief (2-3 sentences), and invite
the user to ask about plans or features."""


def greeter(state: AgentState) -> AgentState:
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=GREETER_SYSTEM),
        *state["messages"],
    ])
    return {**state, "messages": [AIMessage(content=response.content)]}


# ── RAG responder node ────────────────────────────────────────────────────────

def rag_responder(state: AgentState) -> AgentState:
    llm = _get_llm()
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    context = get_knowledge_context(last_human)

    system = f"""You are a helpful sales assistant for AutoStream, an AI-powered video editing SaaS.

Answer the user's question accurately using ONLY the knowledge base below.
If the answer isn't in the knowledge base, say you'll check and get back to them.
Keep responses clear, helpful, and under 120 words.
End with a soft nudge toward signing up if appropriate.

KNOWLEDGE BASE:
{context}"""

    response = llm.invoke([
        SystemMessage(content=system),
        *state["messages"],
    ])
    return {**state, "messages": [AIMessage(content=response.content)]}


# ── Lead collector node ───────────────────────────────────────────────────────

def lead_collector(state: AgentState) -> AgentState:
    lead: LeadData = state.get("lead") or LeadData()

    # If tool already fired, just confirm
    if state.get("lead_captured"):
        msg = AIMessage(content=(
            "You're all set! Our team will reach out to you shortly. "
            "Is there anything else I can help you with? 😊"
        ))
        return {**state, "messages": [msg]}

    # Try to extract fields from the latest user message
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Attempt to fill missing fields from what user just said
    if not lead.name:
        # If intent just triggered, greet and ask for name
        pass
    if not lead.email:
        email = extract_email(last_human)
        if email:
            lead.email = email
    if not lead.platform:
        platform = extract_platform(last_human)
        if platform:
            lead.platform = platform

    # If no email yet, try to fill name from free text
    # (assume first message in lead flow that isn't an email or platform is a name)
    if not lead.name and not extract_email(last_human) and not extract_platform(last_human):
        # Simple heuristic: if the message is short (< 5 words) and looks like a name
        words = last_human.strip().split()
        if 1 <= len(words) <= 4 and all(w.replace("'", "").isalpha() for w in words):
            lead.name = last_human.strip().title()

    # Check if complete → fire tool
    if lead.is_complete():
        result = mock_lead_capture(lead.name, lead.email, lead.platform)
        msg = AIMessage(content=(
            f"Perfect, thank you {lead.name}! 🎉\n\n"
            f"I've passed your details to our team and you'll hear from us shortly at **{lead.email}**. "
            f"We'll make sure to tailor everything for your {lead.platform} workflow.\n\n"
            f"In the meantime, feel free to start your **7-day free Pro trial** — no credit card needed!"
        ))
        return {**state, "messages": [msg], "lead": lead, "lead_captured": True}

    # Otherwise ask for next missing field
    next_field = lead.next_missing_field()
    prompts = {
        "name": (
            "Awesome, I'd love to get you set up! 🚀\n\n"
            "Could I start with your **name**?"
        ),
        "email": (
            f"Great, {lead.name}! What's the best **email address** to reach you at?"
        ),
        "platform": (
            f"Almost there! Which platform do you primarily create for? "
            f"(e.g. YouTube, Instagram, TikTok)"
        ),
    }
    msg = AIMessage(content=prompts.get(next_field, "Could you share a bit more about yourself?"))
    return {**state, "messages": [msg], "lead": lead}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", detect_intent)
    builder.add_node("greeter", greeter)
    builder.add_node("rag_responder", rag_responder)
    builder.add_node("lead_collector", lead_collector)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges("intent_router", route)
    builder.add_edge("greeter", END)
    builder.add_edge("rag_responder", END)
    builder.add_edge("lead_collector", END)

    return builder.compile()
