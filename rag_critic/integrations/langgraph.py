"""
rag_critic.integrations.langgraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Native LangGraph state machine node and conditional routing helpers.
"""

from typing import Dict, Any, Callable, Optional
from ..core import RAGCritic
from ..models import CriticResult


def create_critic_node(
    critic: Optional[RAGCritic] = None,
    response_key: str = "draft",
    context_key: str = "chunks",
    result_key: str = "critic_result",
    feedback_key: str = "revision_feedback"
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate a LangGraph state graph node that executes adversarial verification.
    
    :param critic: Pre-configured RAGCritic instance (defaults to heuristic if None).
    :param response_key: Key in LangGraph state containing the generated draft response.
    :param context_key: Key in LangGraph state containing retrieved reference chunks (List[str]).
    :param result_key: Key in state where CriticResult will be stored.
    :param feedback_key: Key in state where diagnostic re-writer instructions will be saved.
    :return: A state-transition node function.
    """
    evaluator = critic or RAGCritic(provider="heuristic")

    def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
        draft = state.get(response_key, "")
        chunks = state.get(context_key, [])
        
        # Normalize chunks if state contains document objects
        normalized_chunks = []
        for c in chunks:
            if isinstance(c, str):
                normalized_chunks.append(c)
            elif hasattr(c, "page_content"):
                normalized_chunks.append(getattr(c, "page_content"))
            elif isinstance(c, dict) and "text" in c:
                normalized_chunks.append(c["text"])
            elif isinstance(c, dict) and "content" in c:
                normalized_chunks.append(c["content"])
            else:
                normalized_chunks.append(str(c))

        result: CriticResult = evaluator.evaluate(draft, normalized_chunks)

        return {
            result_key: result,
            feedback_key: result.revision_feedback,
            "is_grounded": result.is_grounded,
            "grounding_score": result.grounding_score
        }

    return critic_node


def critic_router(
    state: Dict[str, Any],
    result_key: str = "critic_result"
) -> str:
    """
    Conditional edge router for LangGraph state machines.
    
    Returns:
        - "approved": When Grounding Score >= threshold.
        - "hallucination_detected": When ungrounded assertions require re-writing.
    """
    result = state.get(result_key)
    if isinstance(result, CriticResult):
        return "approved" if result.is_grounded else "hallucination_detected"
        
    is_grounded = state.get("is_grounded", False)
    return "approved" if is_grounded else "hallucination_detected"
