"""
rag_critic.core
~~~~~~~~~~~~~~~
Main RAGCritic evaluator orchestrating atomic decomposition, verification, and scoring.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Optional, Callable, Dict, Any, Union
from .models import CriticResult, Claim, ClaimStatus
from .decomposer import decompose_heuristic, decompose_with_llm
from .verifier import (
    verify_heuristic,
    verify_with_llm,
    compute_metrics,
    build_feedback
)


class RAGCritic:
    """
    Adversarial Hallucination Auditor & Factual Grounding Engine.
    
    Evaluates whether an LLM's draft RAG response is strictly backed by the
    retrieved reference context chunks, producing actionable diagnostic feedback.
    """
    
    def __init__(
        self,
        provider: str = "heuristic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        threshold: float = 0.80,
        custom_llm: Optional[Callable[[str, str], str]] = None
    ):
        """
        Initialize RAGCritic.
        
        :param provider: 'heuristic' (offline), 'groq', 'openai', 'ollama', or 'custom'.
        :param api_key: API key for the chosen provider (defaults to env vars).
        :param model: Specific model string (e.g. 'llama-3.3-70b-versatile', 'gpt-4o-mini').
        :param base_url: Custom API endpoint URL for self-hosted LLMs (vLLM, Ollama).
        :param threshold: Minimum Grounding Score S_ground required to pass (default 0.80).
        :param custom_llm: Custom callable taking (system_prompt, user_prompt) -> str.
        """
        self.provider = provider.lower()
        self.threshold = float(threshold)
        self.custom_llm = custom_llm
        
        # Configure API endpoints
        if self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
            self.base_url = base_url or "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or "llama-3.3-70b-versatile"
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.base_url = base_url or "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4o-mini"
        elif self.provider == "ollama":
            self.api_key = api_key or "ollama"
            self.base_url = base_url or "http://localhost:11434/v1/chat/completions"
            self.model = model or "llama3.2"
        else:
            self.api_key = api_key or ""
            self.base_url = base_url
            self.model = model or "default"

    def _call_http_api(self, system_prompt: str, user_prompt: str) -> str:
        """Lightweight HTTP requester for OpenAI-compatible endpoints using standard library."""
        if not self.base_url or not self.api_key:
            raise ValueError(f"Provider '{self.provider}' requires a valid API key or base_url.")
            
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"} if "gpt" in self.model or "llama-3.3" in self.model else None
        }
        
        # Filter None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "rag-critic-python/0.1.0"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed with HTTP {e.code}: {err_msg}")

    def _get_llm_callable(self) -> Optional[Callable[[str, str], str]]:
        """Return the appropriate callable for decomposition and verification."""
        if self.custom_llm:
            return self.custom_llm
        if self.provider in ("groq", "openai", "ollama") and (self.api_key or self.provider == "ollama"):
            return self._call_http_api
        return None

    def evaluate(
        self,
        draft_response: str,
        retrieved_chunks: List[str]
    ) -> CriticResult:
        """
        Evaluate a draft response against a list of retrieved reference chunks.
        
        :param draft_response: The generated candidate answer.
        :param retrieved_chunks: List of context passages retrieved by the RAG pipeline.
        :return: CriticResult with Grounding Score, Hallucination Index, and itemized claims.
        """
        if not draft_response or not draft_response.strip():
            return CriticResult(
                grounding_score=1.0,
                hallucination_index=0.0,
                is_grounded=True,
                status="APPROVED",
                total_claims=0,
                supported_claims=0,
                ungrounded_claims=0,
                contradicted_claims=0,
                claims=[]
            )
            
        if not retrieved_chunks:
            # If no context was provided, any factual claim is ungrounded
            claims = decompose_heuristic(draft_response)
            for c in claims:
                c.status = ClaimStatus.UNGROUNDED
                c.explanation = "No reference context passages were provided."
                
            return CriticResult(
                grounding_score=0.0,
                hallucination_index=1.0,
                is_grounded=False,
                status="REJECTED_HALLUCINATION",
                total_claims=len(claims),
                supported_claims=0,
                ungrounded_claims=len(claims),
                contradicted_claims=0,
                claims=claims,
                revision_feedback="All assertions are ungrounded because zero source context passages were provided."
            )

        llm_caller = self._get_llm_callable()
        
        # Step 1: Decompose draft into atomic factual claims
        if llm_caller:
            claims = decompose_with_llm(draft_response, llm_caller)
        else:
            claims = decompose_heuristic(draft_response)
            
        # Step 2: Verify claims against retrieved context chunks
        if llm_caller:
            verified_claims = verify_with_llm(claims, retrieved_chunks, llm_caller)
        else:
            verified_claims = verify_heuristic(claims, retrieved_chunks)
            
        # Step 3: Compute mathematical scores
        grounding_score, hallucination_index, supp, unground, contra = compute_metrics(verified_claims)
        
        is_grounded = grounding_score >= self.threshold
        status = "APPROVED" if is_grounded else "REJECTED_HALLUCINATION"
        
        # Step 4: Build citations list
        citations = []
        for c in verified_claims:
            if c.status == ClaimStatus.SUPPORTED and c.citation_chunk_index is not None:
                citations.append({
                    "claim_id": c.id,
                    "claim_text": c.text,
                    "chunk_index": c.citation_chunk_index,
                    "confidence": c.confidence
                })
                
        # Step 5: Formulate feedback for re-writers
        feedback = build_feedback(verified_claims, retrieved_chunks) if not is_grounded else None
        
        # Step 6: Create auto-redacted clean response (stripping ungrounded sentences)
        clean_resp = None
        if not is_grounded:
            supported_texts = [c.text for c in verified_claims if c.status == ClaimStatus.SUPPORTED]
            clean_resp = " ".join(supported_texts) if supported_texts else ""

        return CriticResult(
            grounding_score=grounding_score,
            hallucination_index=hallucination_index,
            is_grounded=is_grounded,
            status=status,
            total_claims=len(verified_claims),
            supported_claims=supp,
            ungrounded_claims=unground,
            contradicted_claims=contra,
            claims=verified_claims,
            citations=citations,
            revision_feedback=feedback,
            clean_response=clean_resp
        )
