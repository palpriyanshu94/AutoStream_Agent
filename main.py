#!/usr/bin/env python3
"""
main.py — CLI entry point for the AutoStream conversational agent.

Usage:
    python main.py

The agent maintains full conversation state (memory) across all turns.
Type 'quit', 'exit', or 'q' to end the session.
"""

import os
import sys
from langchain_core.messages import HumanMessage
from agent.graph import build_graph, AgentState
from agent.tools import LeadData
from dotenv import load_dotenv
load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║       AutoStream — AI Video Editing Assistant        ║
║       Powered by Claude Haiku + LangGraph            ║
╚══════════════════════════════════════════════════════╝
  Type your message and press Enter.
  Type 'quit' to exit.
──────────────────────────────────────────────────────
"""


def check_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Set it with:")
        print("    export ANTHROPIC_API_KEY=your_key_here\n")
        sys.exit(1)


def main():
    check_env()
    graph = build_graph()

    # Initial state
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead": LeadData(),
        "lead_captured": False,
    }

    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Thanks for chatting with AutoStream. 👋\n")
            break

        # Append user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\n  [Error] Something went wrong: {e}\n")
            continue

        # Print last AI message
        from langchain_core.messages import AIMessage
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None
        )
        if last_ai:
            print(f"\nAutoStream: {last_ai.content}\n")
            print("─" * 54)

        # End session naturally if lead was just captured
        if state.get("lead_captured"):
            again = input("\nAnything else I can help you with? (yes/no): ").strip().lower()
            if again not in ("yes", "y"):
                print("\nThanks for your interest in AutoStream! Have a great day. 🎬\n")
                break
            # Reset lead state for next inquiry but keep message history
            state["lead"] = LeadData()
            state["lead_captured"] = False


if __name__ == "__main__":
    main()
