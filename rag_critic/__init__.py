"""
rag-critic
~~~~~~~~~~
Lightweight adversarial hallucination auditor and citation grounding engine
for LangGraph and production RAG pipelines.

Author: Shaheer Ul Hassan (https://shaheerulhassan.com/)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Shaheer Ul Hassan"

from .core import RAGCritic
from .models import (
    Claim,
    ClaimStatus,
    CriticResult,
    ChunkContext
)
from .integrations.langgraph import (
    create_critic_node,
    critic_router
)

__all__ = [
    "__version__",
    "__author__",
    "RAGCritic",
    "Claim",
    "ClaimStatus",
    "CriticResult",
    "ChunkContext",
    "create_critic_node",
    "critic_router",
]
