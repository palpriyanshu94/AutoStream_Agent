"""
rag.py — Local knowledge retrieval for AutoStream agent.

Loads the knowledge base from a JSON file and provides a simple
retrieval function that returns relevant context as a formatted string.
No vector DB needed at this scale — keyword matching is sufficient
and keeps the project dependency-light.
"""

import json
import os
from pathlib import Path

_KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"


def _load_kb() -> dict:
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_knowledge_context(query: str) -> str:
    """
    Given a user query, return the most relevant knowledge base
    content as a formatted string to inject into the LLM prompt.
    """
    kb = _load_kb()
    query_lower = query.lower()

    sections = []

    # Always include company overview
    company = kb["company"]
    sections.append(
        f"Company: {company['name']}\n"
        f"Description: {company['description']}"
    )

    # Include pricing/plans if query is about price, plan, cost, features
    plan_keywords = {"price", "plan", "cost", "basic", "pro", "month",
                     "resolution", "video", "caption", "unlimited", "feature",
                     "4k", "720p", "how much", "cheap", "expensive", "upgrade"}
    if any(kw in query_lower for kw in plan_keywords) or not query_lower.strip():
        plan_lines = []
        for plan in kb["plans"]:
            feats = "\n    - ".join(plan["features"])
            plan_lines.append(
                f"  {plan['name']} — {plan['price']}\n    - {feats}"
            )
        sections.append("Pricing Plans:\n" + "\n".join(plan_lines))

    # Include policies if query is about refund, support, trial, policy
    policy_keywords = {"refund", "support", "cancel", "policy", "trial",
                       "free", "24/7", "platform", "youtube", "instagram",
                       "tiktok", "return", "money back"}
    if any(kw in query_lower for kw in policy_keywords):
        policy_lines = [
            f"  {p['topic']}: {p['detail']}" for p in kb["policies"]
        ]
        sections.append("Policies:\n" + "\n".join(policy_lines))

    return "\n\n".join(sections)


def get_full_context() -> str:
    """Return the entire knowledge base as formatted text."""
    return get_knowledge_context("")
