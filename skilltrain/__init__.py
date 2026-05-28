"""
SkillTrain — Train Your Agent Skills Like Neural Networks

A lightweight, OpenClaw-native skill optimizer. Point it at a SKILL.md,
give it task data and success criteria, and it produces a measurably
better skill — validated, auditable, and deployment-ready.

Usage:
    from skilltrain import SkillTrain, Gate, Scorer

    class MyScorer(Scorer):
        def score(self, result, task):
            return self.evaluate([
                Gate("file_created", bool(result.get("path")), "" if result.get("path") else "No file")
            ])

    trainer = SkillTrain(
        skill_path="~/.openclaw/skills/my-skill/SKILL.md",
        scorer=MyScorer(),
        train_data=["data/train.json", "data/val.json"],
    )
    trainer.run(epochs=4, steps_per_epoch=4)
    trainer.save("best_skill.md")
"""

__version__ = "1.0.0"
__author__ = "SMF Works"
__license__ = "MIT"

from skilltrain.harness import SkillTrain, TrainingResult
from skilltrain.scoring import Gate, ScoringResult, Scorer

__all__ = ["SkillTrain", "TrainingResult", "Gate", "ScoringResult", "Scorer"]
