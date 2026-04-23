"""
tools.py — Tool execution layer for the AutoStream agent.

Contains the mock lead capture function and the LeadCollector
helper that manages collecting name/email/platform across turns.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Mock API ──────────────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock CRM API call.
    In production this would POST to a real CRM (HubSpot, Salesforce, etc.)
    """
    print(f"\n[TOOL CALLED] mock_lead_capture({name!r}, {email!r}, {platform!r})")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ── Lead State Collector ──────────────────────────────────────────────────────

@dataclass
class LeadData:
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None

    def is_complete(self) -> bool:
        return all([self.name, self.email, self.platform])

    def next_missing_field(self) -> Optional[str]:
        if not self.name:
            return "name"
        if not self.email:
            return "email"
        if not self.platform:
            return "platform"
        return None


def extract_email(text: str) -> Optional[str]:
    """Extract an email address from user message if present."""
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


KNOWN_PLATFORMS = [
    "youtube", "instagram", "tiktok", "twitter", "x",
    "facebook", "linkedin", "twitch", "snapchat"
]


def extract_platform(text: str) -> Optional[str]:
    """Extract a known platform name from user message if present."""
    text_lower = text.lower()
    for p in KNOWN_PLATFORMS:
        if p in text_lower:
            return p.capitalize()
    return None
