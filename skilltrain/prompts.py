"""
SkillTrain Prompts — Optimizer Prompt Templates

The optimizer model receives structured prompts for the reflect/merge/select
phases. These are workflow-agnostic and configurable per domain.

Usage:
    from skilltrain.prompts import build_reflect_prompt
    system, user = build_reflect_prompt(skill, failures, successes, budget)
"""

from __future__ import annotations

from typing import Any

# ── Reflect: Error Analyst ─────────────────────────────────────────

REFLECT_SYSTEM = """You are a skill optimizer. Your job is to improve a SKILL.md document
based on execution evidence. You are NOT rewriting the skill — you are making
surgical edits that address documented failure patterns.

You will receive:
- The current SKILL.md content
- A batch of FAILED executions (what the agent did, what went wrong, which gates failed)
- A batch of SUCCESSFUL executions (preserve these behaviors)
- Previously rejected edits (do NOT repeat these)
- An edit budget of L={edit_budget} edits

RULES:
1. Address PATTERNS across multiple failures, not single instances.
2. Edits must be SPECIFIC and ACTIONABLE. "Be more careful" is not an edit.
3. PRESERVE existing working behavior from success patterns.
4. Do NOT repeat edits from the rejected buffer.
5. If a failure pattern is already in the skill (the agent just skipped it),
   strengthen the gate — add verification steps, not reminders.
6. Every edit must be a structural improvement to the SKILL.md content.

Output as JSON:
{{"edits": [{{"op": "append|replace|delete", "content": "...", "target": "..."}}], "reasoning": "..."}}"""

REFLECT_USER = """## Current SKILL.md

{skill_content}

## Failed Executions ({n_failures})

{failures_text}

## Successful Executions ({n_successes})

{successes_text}

## Rejected Edits (DO NOT REPEAT)

{rejected_text}

## Budget: L = {edit_budget}

Propose at most {edit_budget} edits. Output as JSON."""


def build_reflect_prompt(
    skill_content: str,
    failures: list,
    successes: list,
    edit_budget: int,
    rejected_edits: list | None = None,
    max_skill_chars: int = 4000,
) -> tuple[str, str]:
    """Build the system and user prompts for the reflect stage.

    Args:
        skill_content: Current SKILL.md text
        failures: List of RolloutResult (hard_score < 1.0)
        successes: List of RolloutResult (hard_score >= 1.0)
        edit_budget: Maximum edits to propose
        rejected_edits: Previously rejected edit dicts
        max_skill_chars: Truncate skill to this length for context

    Returns:
        (system_prompt, user_prompt)
    """
    # Format failures
    parts = []
    for f in failures:
        gates = getattr(f, 'gates', [])
        gate_str = ", ".join(
            f"{g.name}({'PASS' if g.passed else 'FAIL'})" for g in gates
        )
        parts.append(
            f"- {getattr(f, 'task_id', '?')}: hard={getattr(f, 'hard_score', 0):.2f}\n"
            f"  Gates: {gate_str}\n"
            f"  Fail: {getattr(f, 'fail_reason', 'unknown')}"
        )
    failures_text = "\n".join(parts) if parts else "No failures this batch."

    # Format successes
    s_parts = []
    for s in successes:
        s_parts.append(
            f"- {getattr(s, 'task_id', '?')}: hard={getattr(s, 'hard_score', 0):.2f}"
        )
    successes_text = "\n".join(s_parts) if s_parts else "No successes this batch."

    # Format rejected edits
    if rejected_edits:
        rej_parts = []
        for r in rejected_edits[:20]:
            rej_parts.append(f"- {r.get('op','?')}: {r.get('content','')[:120]}")
        rejected_text = "\n".join(rej_parts)
    else:
        rejected_text = "No previously rejected edits."

    system = REFLECT_SYSTEM.format(edit_budget=edit_budget)
    user = REFLECT_USER.format(
        skill_content=skill_content[:max_skill_chars],
        n_failures=len(failures),
        failures_text=failures_text,
        n_successes=len(successes),
        successes_text=successes_text,
        rejected_text=rejected_text,
        edit_budget=edit_budget,
    )
    return system, user
