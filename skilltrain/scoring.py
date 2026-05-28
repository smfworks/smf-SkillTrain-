"""
SkillTrain Scoring — Define what success looks like for your workflow.

Every custom scorer inherits from Scorer and returns scored gate results.
The base class handles aggregation, partial credit, and formatting.

Usage:
    from skilltrain import Gate, ScoringResult, Scorer

    class MyScorer(Scorer):
        def score(self, result, task):
            return self.evaluate([
                Gate("output_exists", bool(result.get("output")), "" if result.get("output") else "No output produced", weight=1.0),
                Gate("no_errors", not result.get("had_errors", False), "Execution had errors", weight=2.0),
            ])
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any


# ── Data Types ───────────────────────────────────────────────────────

@dataclass
class Gate:
    """One check in a scoring run."""
    name: str                      # e.g. "image_exists", "posted_to_x"
    passed: bool
    reason: str = ""               # empty if passed; explanation if failed
    weight: float = 1.0            # relative importance (2.0 = double weight)

@dataclass
class ScoringResult:
    """Complete scoring output for one task execution."""
    hard_score: float                  # 0.0-1.0, overall pass/fail
    soft_score: float                  # 0.0-1.0, quality-weighted partial credit
    gates: list[Gate] = field(default_factory=list)
    fail_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Scorer:
    """
    Base class for all workflow-specific scorers.

    Override score() to define your gates.
    Use evaluate() to compute hard_score and soft_score from gates.
    Override compute_soft_score() for quality-weighted scoring.
    """

    def score(self, result: Any, task: dict) -> ScoringResult:
        """Override: inspect the agent's result and return scored gates."""
        raise NotImplementedError(
            "Override score() to define your workflow-specific gates. "
            "Return self.evaluate([Gate(...), Gate(...)])"
        )

    def evaluate(self, gates: list[Gate]) -> ScoringResult:
        """Compute scores from a list of gates."""
        total_weight = sum(g.weight for g in gates)
        if total_weight == 0:
            return ScoringResult(hard_score=0.0, soft_score=0.0, gates=gates)

        passed_weight = sum(g.weight for g in gates if g.passed)
        hard = passed_weight / total_weight

        failed = [g for g in gates if not g.passed]
        fail_reason = "; ".join(f"{g.name}: {g.reason}" for g in failed) if failed else ""

        return ScoringResult(
            hard_score=hard,
            soft_score=hard,  # default: same as hard; override for partial credit
            gates=gates,
            fail_reason=fail_reason,
        )

    def compute_soft_score(self, gates: list[Gate], result: Any, task: dict) -> float:
        """Override for quality-weighted partial credit scoring."""
        return self.evaluate(gates).hard_score


# ── Shared Utilities ──────────────────────────────────────────────────

def file_exists(path: str) -> bool:
    """True if file exists AND has non-zero size."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False

def url_ok(url: str, timeout: int = 10) -> bool:
    """True if the URL returns HTTP 200."""
    try:
        import requests
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False

def ffprobe_has_video(path: str) -> bool:
    """True if ffprobe reports a video stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return any(s.get("codec_type") == "video" for s in data.get("streams", []))
    except Exception:
        return False

def frontmatter_field(path: str, field: str) -> tuple[bool, str]:
    """Check frontmatter YAML for a field. Returns (exists, value)."""
    try:
        with open(path) as f:
            content = f.read()
        if not content.startswith("---"):
            return False, ""
        parts = content.split("---", 2)
        if len(parts) < 3:
            return False, ""
        import yaml
        fm = yaml.safe_load(parts[1])
        if fm and field in fm:
            return True, str(fm[field])
        return False, ""
    except Exception:
        return False, ""
