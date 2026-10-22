"""
rag_critic.verifier
~~~~~~~~~~~~~~~~~~~
Entailment matrix verification and mathematical factual grounding scoring.
"""

import json
import re
from typing import List, Dict, Any, Callable, Tuple, Optional
from .models import Claim, ClaimStatus, ChunkContext
from .prompts import VERIFICATION_SYSTEM_PROMPT


def format_context_chunks(chunks: List[str]) -> str:
    """Format raw chunk strings into indexed context blocks."""
    formatted = []
    for idx, c in enumerate(chunks):
        formatted.append(f"[Chunk {idx}]\n{c.strip()}")
    return "\n\n".join(formatted)


def verify_heuristic(claims: List[Claim], chunks: List[str]) -> List[Claim]:
    """
    Deterministic lexical & n-gram overlap baseline for offline/local environments.
    Checks fuzzy containment, numeric matching, and negative keyword detection.
    """
    verified_claims: List[Claim] = []
    
    for claim in claims:
        c_text = claim.text.lower()
        # Extract numeric tokens (dates, percentages, figures)
        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', c_text)
        claim_words = set(re.findall(r'\b[a-z]{3,}\b', c_text))
        
        best_match_idx = None
        best_overlap_ratio = 0.0
        is_contradicted = False
        
        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            
            # Check for direct contradictions (e.g., "was not", "failed", "denied")
            if "not " + c_text[:20] in chunk_lower or "never " + c_text[:20] in chunk_lower:
                is_contradicted = True
                best_match_idx = idx
                break
                
            chunk_words = set(re.findall(r'\b[a-z]{3,}\b', chunk_lower))
            if not claim_words:
                continue
                
            overlap = len(claim_words.intersection(chunk_words)) / len(claim_words)
            
            # If claim has specific numbers, ensure chunk also has those numbers
            has_matching_numbers = True
            if claim_numbers:
                has_matching_numbers = any(num in chunk_lower for num in claim_numbers)
                
            if overlap > best_overlap_ratio and (not claim_numbers or has_matching_numbers):
                best_overlap_ratio = overlap
                best_match_idx = idx
                
        # Classify based on lexical thresholds
        if is_contradicted:
            status = ClaimStatus.CONTRADICTED
            conf = 0.85
            expl = f"Direct negation detected in Chunk {best_match_idx}."
        elif best_overlap_ratio >= 0.65:
            status = ClaimStatus.SUPPORTED
            conf = min(1.0, round(best_overlap_ratio, 2))
            expl = f"Substantial lexical and entity overlap with Chunk {best_match_idx}."
        else:
            status = ClaimStatus.UNGROUNDED
            conf = round(1.0 - best_overlap_ratio, 2)
            expl = "Insufficient grounding found across any retrieved context chunks."
            best_match_idx = None
            
        verified_claims.append(
            Claim(
                id=claim.id,
                text=claim.text,
                status=status,
                citation_chunk_index=best_match_idx,
                confidence=conf,
                explanation=expl
            )
        )
        
    return verified_claims


def verify_with_llm(
    claims: List[Claim],
    chunks: List[str],
    call_llm: Callable[[str, str], str]
) -> List[Claim]:
    """
    Evaluate claims using an LLM reasoning engine against formatted context chunks.
    """
    context_str = format_context_chunks(chunks)
    claims_str = "\n".join([f"Claim {c.id}: {c.text}" for c in claims])
    
    prompt = (
        f"REFERENCE CONTEXT CHUNKS:\n"
        f"=======================\n"
        f"{context_str}\n\n"
        f"CLAIMS TO VERIFY:\n"
        f"=================\n"
        f"{claims_str}\n\n"
        f"Output valid JSON matching the schema."
    )
    
    response_str = call_llm(VERIFICATION_SYSTEM_PROMPT, prompt)
    
    try:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_str)
        json_str = match.group(1) if match else response_str.strip()
        data = json.loads(json_str)
        evals = {e.get("claim_id"): e for e in data.get("evaluations", [])}
        
        verified: List[Claim] = []
        for c in claims:
            ev = evals.get(c.id, {})
            raw_status = ev.get("status", "UNGROUNDED").upper()
            status = getattr(ClaimStatus, raw_status, ClaimStatus.UNGROUNDED)
            
            verified.append(
                Claim(
                    id=c.id,
                    text=c.text,
                    status=status,
                    citation_chunk_index=ev.get("citation_chunk_index"),
                    confidence=float(ev.get("confidence", 0.9)),
                    explanation=ev.get("explanation", "")
                )
            )
        return verified
    except Exception:
        # Fallback to deterministic heuristic verifier on parsing failure
        return verify_heuristic(claims, chunks)


def compute_metrics(claims: List[Claim]) -> Tuple[float, float, int, int, int]:
    """
    Compute Grounding Score S_ground, Hallucination Index H, and tally counts.
    """
    if not claims:
        return 1.0, 0.0, 0, 0, 0
        
    total = len(claims)
    supported = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
    ungrounded = sum(1 for c in claims if c.status == ClaimStatus.UNGROUNDED)
    contradicted = sum(1 for c in claims if c.status == ClaimStatus.CONTRADICTED)
    
    grounding_score = round(supported / total, 4)
    hallucination_penalty = (ungrounded * 1.0) + (contradicted * 2.0)
    hallucination_index = round(min(1.0, hallucination_penalty / total), 4)
    
    return grounding_score, hallucination_index, supported, ungrounded, contradicted


def build_feedback(claims: List[Claim], chunks: List[str]) -> Optional[str]:
    """
    Generate actionable diagnostic feedback for re-writers when ungrounded claims occur.
    """
    flagged = [c for c in claims if c.status != ClaimStatus.SUPPORTED]
    if not flagged:
        return None
        
    feedback_lines = [
        "ADVERSARIAL CRITIC AUDIT: Hallucinations or ungrounded assertions detected.",
        "Please revise the answer to satisfy these constraints:"
    ]
    
    for c in flagged:
        if c.status == ClaimStatus.CONTRADICTED:
            feedback_lines.append(
                f"- [CONTRADICTED] Remove or correct claim: \"{c.text}\". (Reason: {c.explanation})"
            )
        else:
            feedback_lines.append(
                f"- [UNGROUNDED] Remove claim not found in reference passages: \"{c.text}\"."
            )
            
    feedback_lines.append("\nStrict requirement: Do not state any facts, metrics, or dates not explicitly present in the context.")
    return "\n".join(feedback_lines)
