"""
rag_critic.cli
~~~~~~~~~~~~~~
Command-line interface for evaluating factual grounding from terminal & CI/CD.
"""

import sys
import argparse
import json
from .core import RAGCritic


def main():
    parser = argparse.ArgumentParser(
        prog="rag-critic",
        description="Adversarial hallucination auditor & grounding evaluator for RAG."
    )
    parser.add_argument(
        "--draft",
        type=str,
        help="Draft response text to evaluate."
    )
    parser.add_argument(
        "--context",
        type=str,
        nargs="+",
        help="One or more reference context chunk strings."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="heuristic",
        choices=["heuristic", "groq", "openai", "ollama"],
        help="LLM provider or heuristic mode (default: heuristic)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum grounding score threshold (default: 0.80)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON evaluation report."
    )
    parser.add_argument(
        "--version",
        action="version",
        version="rag-critic 0.1.0"
    )

    args = parser.parse_args()

    if not args.draft:
        parser.print_help()
        sys.exit(1)

    chunks = args.context or []
    critic = RAGCritic(provider=args.provider, threshold=args.threshold)
    result = critic.evaluate(args.draft, chunks)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print("=" * 60)
        print(f"RAG-CRITIC AUDIT RESULT: {result.status}")
        print("=" * 60)
        print(f"Grounding Score:     {result.grounding_score:.1%}")
        print(f"Hallucination Index: {result.hallucination_index:.1%}")
        print(f"Total Claims:        {result.total_claims}")
        print(f"Supported Claims:    {result.supported_claims}")
        print(f"Ungrounded Claims:   {result.ungrounded_claims}")
        print(f"Contradicted Claims: {result.contradicted_claims}")
        print("-" * 60)
        print("ITEMIZED PROPOSITIONS:")
        for c in result.claims:
            icon = "✓" if c.status == "SUPPORTED" else "✗"
            print(f"  [{icon}] Claim {c.id}: {c.text} -> {c.status}")
            if c.citation_chunk_index is not None:
                print(f"      (Cited Chunk: {c.citation_chunk_index})")
                
        if result.revision_feedback:
            print("-" * 60)
            print("REVISION INSTRUCTIONS:")
            print(result.revision_feedback)
            
    sys.exit(0 if result.is_grounded else 1)


if __name__ == "__main__":
    main()
