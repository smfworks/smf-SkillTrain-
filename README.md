# SkillTrain — Train Your Agent Skills Like Neural Networks

SkillTrain is an **OpenClaw-native skill optimizer** that treats `SKILL.md` files as trainable parameters. Using a separate optimizer model, it improves your agent skills through structured feedback loops — rollout, score, reflect, edit, validate, and deploy.

```bash
pip install skilltrain
skilltrain train --skill ~/.openclaw/skills/my-skill/SKILL.md \
  --train-data data/train.json \
  --scorer my_scorer.py \
  --epochs 4 --steps-per-epoch 4
```

**You get:** A measurably better `best_skill.md` — your original skill, surgically improved against real failure data, validated against held-out tasks. No weights changed. No new infrastructure. Just a better `.md` file.

---

## How It Works

```
  ROLLOUT → SCORE → REFLECT → EDIT → VALIDATE → ACCEPT/REJECT
     ↑                                              |
     └────────────────── loop ───────────────────────┘

1. Agent executes tasks with current SKILL.md
2. Each result is scored against your custom success gates
3. Optimizer analyzes failure patterns across minibatches
4. Surgical add/replace/delete edits are proposed
5. Candidate skill is tested on held-out validation data
6. Only edits that strictly improve performance are kept
```

---

## Installation

```bash
git clone https://github.com/smfworks/smf-SkillTrain.git
cd smf-SkillTrain
pip install -e .
```

**Requirements:** Python 3.10+, Ollama running locally, PyYAML, Requests.

---

## Quick Start

### 1. Define Your Success Criteria

Create `my_scorer.py`:

```python
from skilltrain import Gate, Scorer

class MyScorer(Scorer):
    def score(self, result, task):
        return self.evaluate([
            Gate("file_created", bool(result.get("path")),
                 "" if result.get("path") else "No file created"),
            Gate("valid_output", result.get("is_valid", False),
                 "Invalid output", weight=2.0),
        ])
```

### 2. Prepare Training Data

```json
[
  {"id": "task-001", "instruction": "Generate a blog post about AI agents", "section": "blog"}
]
```

Split into `data/train.json`, `data/val.json`, and `data/test.json`.

### 3. Train

```bash
skilltrain train \
  --skill ~/.openclaw/skills/my-skill/SKILL.md \
  --train-data data/train.json \
  --scorer my_scorer.py \
  --epochs 4 --steps-per-epoch 4
```

### 4. Deploy

```bash
cp output/skill-best.md ~/.openclaw/skills/my-skill/SKILL.md
```

Your skill is now measurably better.

---

## What Makes SkillTrain Different

| Approach | SkillTrain | Hand-Written | Prompt Engineering | Fine-Tuning |
|----------|------------|-------------|-------------------|-------------|
| Evidence-based | ✅ | ❌ | ❌ | ✅ |
| Weights unchanged | ✅ | ✅ | ✅ | ❌ |
| Auditable output | ✅ | ✅ | ❌ | ❌ |
| Improves over time | ✅ | ❌ | ❌ | 🔄 |
| Validation-gated | ✅ | ❌ | ❌ | Partial |
| Portable (.md) | ✅ | ✅ | Partial | ❌ |
| OpenClaw-native | ✅ | ✅ | ✅ | ❌ |

---

## Architecture

```
skilltrain/
├── skilltrain/
│   ├── harness.py        # 6-stage training loop
│   ├── scoring.py        # Gate, Scorer, ScoringResult
│   ├── rollouts.py       # ollama / mock / subprocess execution
│   ├── optimizer.py      # Cosine/linear/constant/autonomous schedulers
│   ├── prompts.py        # Optimizer prompt templates
│   ├── cli.py            # Full CLI
│   └── __init__.py       # Public API
├── examples/
│   ├── blog_scorer.py    # 8-gate blog publishing example
│   ├── social_scorer.py  # 8-gate social posting example
│   └── generic_scorer.py # Starter template
├── configs/
│   └── default.yaml      # Training configuration
├── setup.py
├── SKILLTRAIN.md         # Agent-consumable instruction manual
├── LICENSE               # MIT
└── README.md             # This file
```

---

## Built for OpenClaw

SkillTrain was designed from the ground up for the OpenClaw ecosystem:

- **Native format:** Optimizes SKILL.md files directly — zero format conversion
- **Ollama-native:** Uses your local models for both agent execution and optimization
- **Cron-compatible:** Training can run as a scheduled job for continuous skill improvement  
- **Multi-agent:** Each agent can train its own skills independently
- **MIT-licensed:** Use it, modify it, build on it. The improvements belong to you.

Any OpenClaw agent can consume `SKILLTRAIN.md` directly for step-by-step setup, configuration, and first-run guidance.

---

## Key Concepts

### The Skill Is the Parameter

Instead of fine-tuning model weights (expensive, opaque, non-portable), SkillTrain optimizes the natural-language procedures in a skill `.md` file — auditable, portable, and deployable in seconds.

### Textual Learning Rate

The "learning rate" is the maximum number of skill edits per optimization step. A cosine schedule starts at 4 (aggressive fixes) and decays to 1 (final polish) — preventing destructive rewrites while keeping enough plasticity to learn new procedures.

### Validation Gate

Every candidate skill is tested on held-out validation data. Only edits that strictly improve performance are accepted. This turns reflection into propose-and-test optimization rather than unconditional self-editing — because plausible textual diagnoses can still hurt the actual model.

### Rejected-Edit Buffer

Failed edits aren't wasted. They feed back as negative signal — "don't repeat these directions" — giving the optimizer a learning history without inference-time cost.

---

## CLI Reference

```
skilltrain train \
  --skill SKILL.md                # Path to SKILL.md (required)
  --train-data train.json          # Training tasks (required)
  --val-data val.json              # Validation tasks (auto-split if omitted)
  --scorer my_scorer.py            # Your scoring logic (required)
  --epochs 4                       # Training epochs
  --steps-per-epoch 4              # Steps per epoch
  --batch-size 4                   # Tasks per rollout batch
  --max-edits 4                    # Max edits per step
  --lr-schedule cosine             # constant | linear | cosine | autonomous
  --model deepseek-v4-pro:cloud    # Optimizer model
  --rollout-mode ollama            # ollama | mock | subprocess
  --output ./output                # Output directory
  --evaluate-only                  # Evaluate without training
```

---

## Examples

```python
# Blog publishing scorer (examples/blog_scorer.py)
class BlogScorer(Scorer):
    def score(self, result, task):
        return self.evaluate([
            Gate("file_created", file_exists(result.get("path")),
                 "" if file_exists(result.get("path")) else "No file", weight=1.0),
            Gate("excerpt_ok", len(result.get("excerpt","")) <= 160,
                 f"{len(result.get('excerpt',''))} chars (limit 160)", weight=0.5),
            Gate("image_exists", file_exists(result.get("image_path","")),
                 "Hero image missing", weight=1.5),  # CRITICAL
            Gate("deployed", bool(result.get("deployed_url")),
                 "Not deployed", weight=1.5),          # CRITICAL
        ])
```

---

## Based On

Inspired by [Microsoft Research SkillOpt](https://arxiv.org/abs/2605.23904) — Yifan Yang, Ziyang Gong, Weiquan Huang, et al. (2026) — the first systematic text-space optimizer for agent skills. SkillTrain is an independent, MIT-licensed, OpenClaw-native implementation of the same principles: treat skills as trainable external state, use validation gates, and export auditable `.md` artifacts.

---

## Credits

Built by [Aiona Edge](https://smfworks.com/the-edge), Chief AI Research Scientist at SMF Works, with architecture contributions from Gabriel (Chief AI Correspondent) and Morgan (Social Strategy). The scoring framework was informed by Pamela's Architecture of Taste evaluation philosophy.

Research, spike, harness, scorers, prompts, rollouts, CLI, and agent-consumable documentation — all built in one session on May 28, 2026. Powered by deepseek-v4-pro:cloud via Ollama.

---

## License

MIT — use it, modify it, build on it. If you improve a skill with SkillTrain, the improvement is yours.

---

*SkillTrain — because your skills should learn too.*
