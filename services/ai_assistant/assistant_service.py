"""
AI Assistant Service
AI Pharmacovigilance Intelligence Platform

High-level service wrapper with conversation history management.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from services.ai_assistant.query_engine import LocalQueryEngine, LLMQueryEngine


class ConversationMessage:
    """Single turn in a conversation."""
    def __init__(self, role: str, content: str, data=None):
        self.role = role
        self.content = content
        self.data = data or []
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class AIAssistantService:
    """
    Conversational AI assistant for pharmacovigilance queries.

    Maintains conversation history and supports both local and LLM-enhanced modes.

    Usage
    -----
    assistant = AIAssistantService()
    response = assistant.chat("What are the adverse events for Metformin?")
    response = assistant.chat("Show me risk signals for it")  # context-aware
    """

    WELCOME_MESSAGE = (
        "Hello! I'm PharmAI, your pharmacovigilance AI assistant. "
        "I can help you analyse adverse event data, identify risk signals, "
        "and answer questions about drug safety. "
        "Try asking: 'What are the most common adverse events for Metformin?' "
        "or 'Show safety signals detected in the dataset.'"
    )

    def __init__(self, use_llm: bool = False, openai_api_key: Optional[str] = None) -> None:
        if use_llm and openai_api_key:
            self._engine = LLMQueryEngine(api_key=openai_api_key)
            logger.info("AIAssistantService initialised with LLM mode.")
        else:
            self._engine = LocalQueryEngine()
            logger.info("AIAssistantService initialised with local mode.")

        self._history: List[ConversationMessage] = []
        self._context: dict = {}

    def chat(self, message: str) -> dict:
        """
        Process a user message and return a response.

        Parameters
        ----------
        message : str
            User's natural language question.

        Returns
        -------
        dict with keys: answer, data, intent, confidence, timestamp
        """
        # Resolve pronouns from context (basic coreference)
        resolved_message = self._resolve_context(message)

        # Store user message
        self._history.append(ConversationMessage("user", message))

        # Query the engine
        result = self._engine.query(resolved_message)

        # Update context
        self._update_context(result)

        # Store assistant response
        self._history.append(
            ConversationMessage("assistant", result["answer"], result.get("data", []))
        )

        return result

    def get_history(self) -> List[dict]:
        """Return conversation history as a list of dicts."""
        return [msg.to_dict() for msg in self._history]

    def clear_history(self) -> None:
        """Reset conversation history and context."""
        self._history = []
        self._context = {}
        logger.info("Conversation history cleared.")

    def get_suggested_queries(self) -> List[str]:
        """Return example queries for UI display."""
        return [
            "What are the most common adverse events for Metformin?",
            "Show risk signals detected in the dataset",
            "Top 10 most reported drugs",
            "How many reports of nausea are there?",
            "What is the seriousness rate for Warfarin?",
            "Show me the top 10 adverse events",
            "Give me a platform summary",
            "What adverse events are associated with SSRIs?",
            "Show signals for Adalimumab",
        ]

    # ------------------------------------------------------------------
    # Context resolution helpers
    # ------------------------------------------------------------------

    def _resolve_context(self, message: str) -> str:
        """Replace pronouns with context from previous turns."""
        if not self._history:
            return message

        message_lower = message.lower()
        # Replace "it" / "that drug" with last mentioned drug
        if any(p in message_lower for p in ["for it", "about it", "its ", "for that"]):
            if last_drug := self._context.get("last_drug"):
                message = re.sub(r"\b(it|that\s+drug)\b", last_drug, message, flags=re.IGNORECASE)

        return message

    def _update_context(self, result: dict) -> None:
        """Extract context from result for future reference."""
        import re
        data = result.get("data", [])
        intent = result.get("intent", "")

        if intent in ("adverse_events_for_drug", "signals_for_drug", "seriousness_rate"):
            # Try to extract drug from the question in history
            if self._history:
                last_user = next(
                    (m for m in reversed(self._history) if m.role == "user"), None
                )
                if last_user:
                    # Simple drug extraction from question
                    match = re.search(r"for\s+([\w\s]+?)(?:\?|$)", last_user.content, re.IGNORECASE)
                    if match:
                        self._context["last_drug"] = match.group(1).strip()



