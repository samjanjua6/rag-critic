# rag-critic

[![PyPI version](https://img.shields.io/pypi/v/rag-critic.svg?color=blue)](https://pypi.org/project/rag-critic/)
[![Python versions](https://img.shields.io/pypi/pyversions/rag-critic.svg)](https://pypi.org/project/rag-critic/)
[![Downloads](https://img.shields.io/pypi/dm/rag-critic.svg?color=emerald)](https://pypistats.org/packages/rag-critic)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Shaheer%20Ul%20Hassan-8F63DD)](https://shaheerulhassan.com/)
[![GitHub stars](https://img.shields.io/github/stars/samjanjua6/rag-critic?style=social)](https://github.com/samjanjua6/rag-critic)

> **Lightweight adversarial hallucination auditor and citation grounding engine for LangGraph and production RAG pipelines.**

> [!TIP]
> **Support Open-Source**: If you find `rag-critic` useful in your RAG pipelines or LangGraph evaluation loops, please consider giving it a ⭐ on GitHub — it helps other engineers discover it!

Stop deploying unchecked RAG applications. `rag-critic` provides an adversarial verification loop that decomposes LLM draft answers into atomic factual claims, checks them against retrieved source passages, computes continuous grounding scores, and outputs actionable revision instructions for re-writer agents.

---

## ⚡ The Problem: Unchecked RAG in Production

Most production RAG architectures follow a naive, linear pipeline:
```
[User Query] ──> [Vector DB Retrieval] ──> [LLM Generation] ──> [Emit to User]
```

In production, single-pass LLMs frequently:
- **Invent ungrounded statistics, numbers, and dates** absent from the source files.
- **Extrapolate beyond retrieved context** without acknowledging uncertainty.
- **Fabricate phantom citations** and misattribute quotes.

---

## 🛡️ The Solution: The `rag-critic` Loop

```
[User Query] ──> [Retriever] ──> [LLM Draft] ──> [rag-critic Node]
                                                        │
                      ┌─────────────────────────────────┴──────────────────┐
                      ▼                                                    ▼
             [Grounding >= 0.85]                                  [Hallucination Detected]
                      │                                                    │
                      ▼                                                    ▼
         [Verified Output + Citations]                             [Re-writer Agent]
                                                                           │
                                                                           └──> [Regenerate]
```

`rag-critic` acts as an automated critic node:
1. **Atomic Decomposition**: Deconstructs compound sentences into indivisible factual claims.
2. **Entailment Matrix**: Classifies each claim as `SUPPORTED`, `CONTRADICTED`, or `UNGROUNDED`.
3. **Mathematical Scoring**: Computes the continuous Grounding Score $S_{\text{ground}} \in [0.0, 1.0]$ and Hallucination Index.
4. **Targeted Diagnostic Feedback**: Delivers specific, itemized instructions to re-writer nodes explaining exactly what failed and why.

---

## 📦 Installation

```bash
pip install rag-critic
```

Zero heavy dependencies. Pure Python with Pydantic v2.

---

## 🚀 Quickstart

### 1. Standalone Python / FastAPI

```python
from rag_critic import RAGCritic

# Use Groq (sub-second LPU speed), OpenAI, Ollama, or offline heuristic mode
critic = RAGCritic(provider="groq", model="llama-3.3-70b-versatile")

draft_answer = (
    "DeepTrace uses Error Level Analysis at Q=95 to detect document tampering. "
    "The platform raised $50 million in Series A funding in London."
)

retrieved_context = [
    "DeepTrace is an open-source forensic platform using Error Level Analysis (calibrated at Q=95) to audit bank statement tampering."
]

result = critic.evaluate(draft_answer, retrieved_context)

print("Status:         ", result.status)            # REJECTED_HALLUCINATION
print("Grounding Score:", f"{result.grounding_score:.1%}") # 50.0%
print("Ungrounded:     ", result.ungrounded_claims) # 1 ($50M Series A claim flagged!)

# Auto-generated diagnostic feedback for your re-writer
print(result.revision_feedback)
# -> [UNGROUNDED] Remove claim not found in reference passages: "The platform raised $50 million in Series A funding in London."
```

---

### 2. LangGraph Native Integration

Add an adversarial hallucination guardrail to any LangGraph workflow in 3 lines:

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from rag_critic import RAGCritic, create_critic_node, critic_router

class RAGState(TypedDict):
    query: str
    chunks: List[str]
    draft: str
    revision_feedback: str
    critic_result: Any

workflow = StateGraph(RAGState)

# Define your pipeline nodes
workflow.add_node("retrieve", my_retriever_node)
workflow.add_node("generate", my_generator_node)

# Add rag-critic adversarial node
critic = RAGCritic(provider="groq", threshold=0.85)
workflow.add_node("critic", create_critic_node(critic))

# Define conditional edge
workflow.add_conditional_edges(
    "critic",
    critic_router,
    {
        "approved": END,
        "hallucination_detected": "generate",  # Loops back with exact diagnostic feedback
    }
)
```

---

## 📊 Factual Verification States

| Status | Meaning | Action Taken |
| :--- | :--- | :--- |
| `SUPPORTED` | Context explicitly confirms claim. | Chunk index and citation mapped. |
| `CONTRADICTED` | Context contradicts the claim. | Flagged with severity penalty; revision generated. |
| `UNGROUNDED` | Context lacks evidence for claim. | Flagged as hallucination; removed or re-written. |

### Mathematical Formulation

$$\text{Grounding Score } S_{\text{ground}} = \frac{\sum \text{SUPPORTED}}{\text{Total Claims}} \in [0.0, \; 1.0]$$

$$\text{Hallucination Index } H = \frac{\sum (\text{UNGROUNDED} + 2 \cdot \text{CONTRADICTED})}{\text{Total Claims}}$$

---

## 🛠️ CLI Usage

Audit responses directly from the command line or CI/CD test suites:

```bash
rag-critic \
  --draft "DeepTrace was built in 2026." \
  --context "DeepTrace is a forensic platform built by Shaheer Ul Hassan in 2026." \
  --threshold 0.80
```

---

## 🌐 Supported Model Providers

- **Groq** (`provider="groq"`, default `llama-3.3-70b-versatile` — Recommended for sub-500ms audit latency)
- **OpenAI** (`provider="openai"`, default `gpt-4o-mini`)
- **Ollama** (`provider="ollama"`, default `llama3.2` — 100% offline local inference)
- **Heuristic Mode** (`provider="heuristic"` — Deterministic n-gram & entity overlap; zero API keys required)
- **Custom Callable** (`custom_llm=my_custom_inference_fn`)

---

## 👤 Author & Citations

Engineered by **Shaheer Ul Hassan**  
- **Portfolio & Case Studies**: [shaheerulhassan.com](https://shaheerulhassan.com/)  
- **GitHub**: [@samjanjua6](https://github.com/samjanjua6)  
- **LinkedIn**: [Shaheer Ul Hassan](https://www.linkedin.com/in/muhammad-shaheer-ul-hassan-b3b7213b3/)  
- **PyPI Profile**: [pypi.org/user/samjanjua6/](https://pypi.org/user/samjanjua6/)

---

## 📄 License

MIT License. Open source and free for commercial and academic use.
