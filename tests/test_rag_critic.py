"""
Unit tests for rag-critic engine, decomposer, verifier, and LangGraph integration.
"""

import unittest
from rag_critic import (
    RAGCritic,
    ClaimStatus,
    create_critic_node,
    critic_router
)


class TestRAGCritic(unittest.TestCase):

    def setUp(self):
        self.critic = RAGCritic(provider="heuristic", threshold=0.75)
        self.sample_context = [
            "Shaheer Ul Hassan is an AI Systems Engineer based in Islamabad.",
            "DeepTrace is a digital document forensics engine that uses Error Level Analysis at Q=95 to catch bank statement tampering."
        ]

    def test_perfect_grounding(self):
        draft = "Shaheer Ul Hassan is an AI Systems Engineer in Islamabad. DeepTrace uses Error Level Analysis at Q=95 for bank statement tampering."
        result = self.critic.evaluate(draft, self.sample_context)

        self.assertTrue(result.is_grounded)
        self.assertEqual(result.status, "APPROVED")
        self.assertGreaterEqual(result.grounding_score, 0.75)
        self.assertEqual(result.ungrounded_claims, 0)
        self.assertIsNone(result.revision_feedback)

    def test_hallucination_detection(self):
        # Contains an ungrounded claim (raised $50M) not in context
        draft = (
            "Shaheer Ul Hassan is an AI Systems Engineer. "
            "DeepTrace raised $50 million in Series A funding from Google Ventures in London."
        )
        result = self.critic.evaluate(draft, self.sample_context)

        self.assertFalse(result.is_grounded)
        self.assertEqual(result.status, "REJECTED_HALLUCINATION")
        self.assertGreater(result.ungrounded_claims, 0)
        self.assertIsNotNone(result.revision_feedback)
        self.assertIn("ADVERSARIAL CRITIC AUDIT", result.revision_feedback)

    def test_empty_draft(self):
        result = self.critic.evaluate("", self.sample_context)
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.total_claims, 0)

    def test_no_context_flags_everything(self):
        draft = "The system achieved 99% accuracy."
        result = self.critic.evaluate(draft, [])
        self.assertFalse(result.is_grounded)
        self.assertEqual(result.grounding_score, 0.0)
        self.assertEqual(result.status, "REJECTED_HALLUCINATION")

    def test_langgraph_node_and_router(self):
        node_fn = create_critic_node(self.critic)
        
        # Test Approved state
        state_pass = {
            "draft": "Shaheer Ul Hassan is an AI Systems Engineer in Islamabad.",
            "chunks": self.sample_context
        }
        res_pass = node_fn(state_pass)
        self.assertTrue(res_pass["is_grounded"])
        self.assertEqual(critic_router(res_pass), "approved")

        # Test Rejected state
        state_fail = {
            "draft": "DeepTrace raised $50 million from Google Ventures in London.",
            "chunks": self.sample_context
        }
        res_fail = node_fn(state_fail)
        self.assertFalse(res_fail["is_grounded"])
        self.assertEqual(critic_router(res_fail), "hallucination_detected")


if __name__ == "__main__":
    unittest.main()
