"""
SkillTrain Rollouts — Agent Execution Engine

Executes tasks with a given SKILL.md and returns scored results.
Supports multiple execution modes:

1. "ollama" — Direct Ollama API call (no gateway dependency)
2. "mock" — Synthetic data for testing without a model
3. "subprocess" — Shell command execution (for CLI-based agents)

The rollouts module is pluggable: swap in your own executor by
implementing execute_batch(skill, tasks, scorer).

Usage:
    from skilltrain.rollouts import RolloutRunner
    runner = RolloutRunner(mode="ollama", model="deepseek-v4-pro:cloud")
    results = runner.execute_batch(skill_content, tasks, scorer)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any


class RolloutRunner:
    """Execute agent tasks and return scored results."""

    def __init__(
        self,
        mode: str = "ollama",
        model: str = "deepseek-v4-pro:cloud",
        api_endpoint: str = "http://127.0.0.1:11434",
        timeout: int = 300,
    ):
        self.mode = mode
        self.model = model
        self.api_endpoint = api_endpoint
        self.timeout = timeout

    def execute_batch(
        self,
        skill_content: str,
        tasks: list[dict],
        scorer,
        verbose: bool = True,
    ) -> list[dict]:
        """Run all tasks in a batch and return rollout results."""
        results = []
        for i, task in enumerate(tasks, 1):
            if verbose:
                name = task.get("instruction", task.get("id", ""))[:80]
                print(f"    [{i}/{len(tasks)}] {name}...")
            result = self._execute_one(task, skill_content)
            scored = scorer.score(result, task)

            results.append({
                "task": task,
                "output": result,
                "hard_score": scored.hard_score,
                "soft_score": scored.soft_score,
                "fail_reason": scored.fail_reason,
                "trajectory_summary": json.dumps(result)[:500] if result else "",
                "gates": [g for g in scored.gates],
            })
        return results

    def _execute_one(self, task: dict, skill: str) -> dict:
        """Execute a single task with the given skill."""
        if self.mode == "ollama":
            return self._via_ollama(task, skill)
        elif self.mode == "mock":
            return self._mock(task)
        elif self.mode == "subprocess":
            return self._via_subprocess(task, skill)
        else:
            raise ValueError(f"Unknown rollout mode: {self.mode}")

    def _via_ollama(self, task: dict, skill: str) -> dict:
        """Execute via Ollama API with skill as system context."""
        instruction = task.get("instruction", task.get("id", ""))

        system = f"""{skill}

Execute the task below. Follow the skill's procedures exactly.
Respond with a JSON object containing the output of each step."""

        user = f"TASK: {instruction}\nSection: {task.get('section', '')}\nTopic: {task.get('topic', '')}\n\nRespond with a structured JSON result."

        try:
            payload = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            })

            result = subprocess.run(
                ["curl", "-s", f"{self.api_endpoint}/api/chat", "-d", payload],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                return {"_error": f"API error: {result.returncode}"}

            data = json.loads(result.stdout)
            content = data.get("message", {}).get("content", "")
            parsed = self._parse_json(content)
            return parsed if parsed else {"_raw": content}

        except subprocess.TimeoutExpired:
            return {"_error": "Timeout"}
        except Exception as e:
            return {"_error": str(e)}

    def _via_subprocess(self, task: dict, skill: str) -> dict:
        """Execute via shell command — for CLI-based agent pipelines."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(skill)
            skill_path = f.name

        try:
            cmd = task.get("command", "echo 'no command specified'")
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.timeout, cwd=task.get("workdir", "."),
                env={**os.environ, "SKILL_PATH": skill_path, "TASK_ID": task.get("id", "")},
            )
            return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
        except Exception as e:
            return {"_error": str(e)}
        finally:
            if os.path.exists(skill_path):
                os.unlink(skill_path)

    def _mock(self, task: dict) -> dict:
        """Mock execution for testing — returns synthetic success data."""
        return {
            "slug": task.get("id", "test"),
            "excerpt": "A concise excerpt under 160 characters.",
            "categories": [task.get("section", "blog").title(), "Test"],
            "file_path": f"/mock/{task.get('section','output')}/{task.get('id','test')}.md",
            "deployed_url": f"https://example.com/{task.get('section','')}/{task.get('id','test')}",
            "is_valid": True,
            "_source": "mock",
        }

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from model response."""
        if not text:
            return None
        if "```json" in text:
            s = text.index("```json") + 7; e = text.index("```", s)
            text = text[s:e].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "{" in text and "}" in text:
                try:
                    return json.loads(text[text.index("{"):text.rindex("}") + 1])
                except json.JSONDecodeError:
                    pass
        return None

import os
