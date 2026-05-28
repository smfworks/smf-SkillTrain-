"""
Generic Skill Scorer — Starter Template

Copy this file and customize it for your own workflow.
Replace the gates below with checks that make sense for your task.

Each gate is a simple check: did the agent do this thing correctly?
"""

from skilltrain import Gate, Scorer


class GenericScorer(Scorer):
    """Replace with your own scoring logic."""

    def score(self, result: dict, task: dict) -> "ScoringResult":
        gates = [
            # Gate: Did execution produce output?
            Gate("has_output", bool(result),
                 "" if result else "No output produced", weight=1.0),

            # Gate: Were there errors?
            Gate("no_errors", not result.get("had_errors", False),
                 result.get("error_msg", "Execution had errors"), weight=2.0),

            # Gate: ADD YOUR GATES HERE
            # Gate("your_gate_name", condition, reason_if_failed, weight=1.0),
        ]

        return self.evaluate(gates)
