"""
rag_critic.decomposer
~~~~~~~~~~~~~~~~~~~~~
Extracts atomic, indivisible factual propositions from draft responses.
"""

import json
import re
from typing import List, Callable, Optional, Dict, Any
from .models import Claim
from .prompts import DECOMPOSITION_SYSTEM_PROMPT


def decompose_heuristic(text: str) -> List[Claim]:
    """
    Fallback deterministic sentence and clause decomposition when no LLM is provided.
    Splits on sentence delimiters, semicolons, and independent coordinating clauses.
    """
    cleaned = text.strip()
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Split on sentence terminals
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    claims: List[Claim] = []
    claim_id = 1
    
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 5:
            continue
            
        # Filter obvious conversational filler
        if re.match(r'^(here is|sure,|certainly,|in summary|as an ai|based on)', s_clean, re.I):
            continue
            
        # Split compound sentences on semicolons
        sub_clauses = re.split(r';\s*', s_clean)
        for clause in sub_clauses:
            clause = clause.strip().rstrip('.')
            if len(clause) > 8:
                claims.append(Claim(id=claim_id, text=clause + "."))
                claim_id += 1
                
    return claims


def decompose_with_llm(text: str, call_llm: Callable[[str, str], str]) -> List[Claim]:
    """
    Decompose draft text into atomic propositions using an LLM.
    
    :param text: The draft response to decompose.
    :param call_llm: A callable taking (system_prompt, user_content) and returning a string response.
    """
    prompt = f"Decompose the following text into atomic factual propositions:\n\n{text}"
    response_str = call_llm(DECOMPOSITION_SYSTEM_PROMPT, prompt)
    
    try:
        # Extract JSON block if surrounded by markdown code fences
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_str)
        json_str = match.group(1) if match else response_str.strip()
        
        data = json.loads(json_str)
        raw_claims = data.get("claims", [])
        
        claims: List[Claim] = []
        for i, c in enumerate(raw_claims):
            cid = c.get("id", i + 1)
            ctext = c.get("text", "").strip()
            if ctext:
                claims.append(Claim(id=cid, text=ctext))
        return claims
    except Exception:
        # Fallback to heuristic parser if JSON parsing fails
        return decompose_heuristic(text)
