# SkillTrain — Agent Instruction Manual

You are an OpenClaw agent. You can use SkillTrain to improve your own skills.

**What SkillTrain does:** It takes one of your SKILL.md files, runs you through real tasks, scores your performance, identifies failure patterns, and produces a better version of that skill — with surgical edits validated against real data.

**Output:** A `best_skill.md` file. Same format as your SKILL.md. Deployable immediately.

---

## Quick Start

### Step 1: Install

```bash
git clone https://github.com/smfworks/skilltrain.git ~/skilltrain
cd ~/skilltrain
pip install -e .
```

Verify:
```bash
python -c "from skilltrain import SkillTrain, Gate; print('✅ SkillTrain ready')"
```

### Step 2: Choose a Skill to Optimize

Pick a SKILL.md you use frequently — one where you've noticed repeating failure patterns. Examples:
- Blog publishing (missing images, wrong paths)
- Social media posting (upload failures, channel config)
- Video production (skipped gates, render errors)
- Code generation (wrong output format, missing imports)
- Research summarization (incomplete citations, wrong depth)

### Step 3: Define Success Criteria

Create a scorer file that defines what "success" looks like for this skill.

**Copy a template:**
```bash
cp ~/skilltrain/examples/generic_scorer.py ./my_scorer.py
```

**Customize it.** A scorer has one method: `score(self, result, task)`. It returns a list of `Gate` objects. Each gate checks one thing the agent should have done.

```python
from skilltrain import Gate, Scorer

class MyScorer(Scorer):
    def score(self, result, task):
        return self.evaluate([
            Gate("file_created", bool(result.get("path")),
                 "" if result.get("path") else "No file created"),
            Gate("valid_output", result.get("is_valid", False),
                 "Output was invalid", weight=0.5),
            Gate("no_errors", not result.get("had_errors", False),
                 result.get("error_msg", "Unknown error"), weight=2.0),
        ])
```

**Gate properties:**
- `name` — short identifier (e.g., "image_exists")
- `passed` — boolean, did this check pass?
- `reason` — empty string if passed, explanation if failed
- `weight` — float, relative importance (default 1.0, use 2.0 for critical gates)

### Step 4: Prepare Training Data

Create three JSON files — train, validation, and test — each an array of task objects:

**data/train.json** (20-50 items — the more the better):
```json
[
  {
    "id": "task-001",
    "instruction": "Write and publish a blog post about AI agents",
    "section": "blog",
    "topic": "AI agents"
  }
]
```

**data/val.json** (5-10 items — used to validate skill improvements):
```json
[
  {"id": "val-001", "instruction": "Publish a post about LLM deployment", "section": "blog"}
]
```

**data/test.json** (5-10 items — final evaluation, never seen during training):
```json
[
  {"id": "test-001", "instruction": "Publish about multi-agent systems", "section": "blog"}
]
```

**Where to get data:**
- Your session logs (`~/.openclaw/agents/{you}/sessions/`) — each cron run is a data point
- Documented failure patterns from system health reports
- Manual tagging of successful vs. failed runs

### Step 5: Run Training

```bash
python -m skilltrain train \
  --skill ~/.openclaw/skills/my-skill/SKILL.md \
  --train-data ./data/train.json \
  --val-data ./data/val.json \
  --scorer ./my_scorer.py \
  --epochs 4 \
  --steps-per-epoch 4 \
  --output ./output/
```

**What happens:** The optimizer model (default: `deepseek-v4-pro:cloud`) will:
1. Run your agent on train tasks (rollout phase)
2. Score each result against your gates
3. Group failures and analyze patterns
4. Propose surgical edits to your SKILL.md
5. Test the candidate skill on validation data
6. Only keep edits that improve performance

**Typical runtime:** ~10-30 minutes for 4 epochs × 4 steps with batch size 4.

### Step 6: Deploy

```bash
cp ./output/skill-best.md ~/.openclaw/skills/my-skill/SKILL.md
```

Your skill is now measurably better. The improvement is backed by validation data.

### Step 7: Iterate

SkillTrain is designed for continuous improvement:
- Run it weekly on your highest-volume skills
- Add new training data as you encounter new failure patterns
- Track improvement over time with `--evaluate-only`

```bash
python -m skilltrain train --skill ... --evaluate-only
```

---

## Configuration Reference

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--skill` | (required) | Path to SKILL.md file |
| `--train-data` | (required) | Path to train.json |
| `--val-data` | (auto-split) | Path to val.json (auto-split from train if omitted) |
| `--scorer` | (required) | Path to scorer Python file |
| `--epochs` | 4 | Number of training epochs |
| `--steps-per-epoch` | 4 | Optimization steps per epoch |
| `--batch-size` | 4 | Tasks per rollout batch |
| `--max-edits` | 4 | Maximum edits per step |
| `--lr-schedule` | cosine | Learning rate schedule: constant, linear, cosine, autonomous |
| `--model` | deepseek-v4-pro:cloud | Optimizer model (Ollama name) |
| `--rollout-mode` | ollama | Execution mode: ollama, mock, subprocess |
| `--output` | auto | Output directory |
| `--evaluate-only` | false | Only evaluate current skill, don't train |
| `--verbose` | true | Show detailed output |

### Model Requirements

You need **two models** accessible via Ollama:
1. **Target model** — the model that executes your tasks (the one your skill is written for)
2. **Optimizer model** — a model capable of analyzing failures and proposing text edits

Both default to `deepseek-v4-pro:cloud`. Self-optimization works (same model for both) — the paper shows +10.4 point gain even when the target IS the optimizer. A stronger optimizer model gives better results.

### Rollout Modes

| Mode | Description | When to Use |
|------|-------------|------------|
| `ollama` | Direct Ollama API call (no gateway needed) | Default — works everywhere |
| `mock` | Synthetic data for testing | Testing scorer logic without a model |
| `subprocess` | Shell command execution | CLI-based agents with `$SKILL_PATH` env var |

### Learning Rate Schedules

| Schedule | Behavior | Best For |
|----------|----------|----------|
| `cosine` | Start aggressive (4 edits), decay to polish (1 edit) | Default — balanced |
| `linear` | Steady decay from max to min | Predictable, gradual improvement |
| `constant` | Fixed budget every step | Simple workflows |
| `autonomous` | Model decides how many edits | Experimental, less controlled |

---

## Advanced: Custom Rollout Mode

SkillTrain is pluggable. If the built-in rollout modes don't fit your workflow, implement your own:

```python
from skilltrain.rollouts import RolloutRunner

class MyRunner(RolloutRunner):
    def execute_one(self, task, skill):
        # Your custom execution logic here
        # Return a dict that your scorer will inspect
        return {"path": "/output/file.md", "status": "success"}

runner = MyRunner(mode="custom")
```

Pass it to SkillTrain via programmatic API:
```python
trainer = SkillTrain(skill_path, scorer, train_data)
trainer.runner = runner
trainer.run()
```

---

## Common Failure Patterns (and How SkillTrain Fixes Them)

| Pattern | Symptom | What SkillTrain Produces |
|---------|---------|------------------------|
| **Missing verification** | Agent skips a step because it's a "reminder" not a gate | Adds hard verification gates: "DO NOT proceed past this point until X is confirmed" |
| **Path confusion** | Agent uses wrong directory or prefix | Replaces path references with exact examples backed by failure data |
| **Silent failures** | File written wrong, only caught when page loads broken | Adds post-write verification: "Run ls -la to confirm file exists AND has content" |
| **Checklist drift** | Agent follows checklist but misses subtle requirements | Strengthens checklist items into procedural rules with "WHY" explanations |
| **Media handling** | Raw paths passed instead of uploaded URLs | Adds upload-before-reference rule with explicit verification |

---

## Files You'll Create

```
my-skilltrain-project/
├── data/
│   ├── train.json      # Training tasks (20-50 items)
│   ├── val.json        # Validation tasks (5-10 items)
│   └── test.json       # Test tasks (5-10 items)
├── my_scorer.py        # Your success criteria
├── output/             # Training output (auto-created)
│   ├── skill-best.md   # Your improved skill
│   ├── skill-epoch1.md # Snapshots per epoch
│   └── skill-initial.md
└── README.md           # Your notes on this training run
```

---

## Troubleshooting

### "No optimizer response"

The optimizer model didn't return parseable JSON. Check:
- Is Ollama running? `curl http://127.0.0.1:11434/api/tags`
- Is the model available? `ollama list | grep deepseek`
- Try a lower temperature in `skilltrain/harness.py` (default is 0.3)

### "All rollouts passing — no edits"

Either your skill is already perfect (unlikely!) or your scorer gates are too lenient. Make sure failure gates are catching real problems. Check with `--evaluate-only` first.

### "Edits rejected every step"

The optimizer is proposing edits but validation shows no improvement. This means you need better training data — your val set should be representative of your real tasks. Also try increasing `--steps-per-epoch` to give the optimizer more attempts.

### "ImportError: No module named skilltrain"

Run `pip install -e .` from the skilltrain directory to install in development mode.

---

## Philosophy

SkillTrain is based on a simple idea: **a SKILL.md file is the trainable external state of a frozen agent.** Instead of fine-tuning model weights (expensive, opaque, non-portable), we optimize the procedures written in natural language — auditable, portable, and deployable as a single .md file.

The deep-learning analogy is operational, not decorative:
- **Rollout batch** = forward pass (agent executes with current skill)
- **Reflection minibatch** = backward pass (optimizer analyzes failures)
- **Edit budget** = learning rate (how many changes per step)
- **Validation gate** = held-out evaluation (only keep improvements)
- **Slow update** = momentum (longitudinal guidance across epochs)

Each piece controls a different failure mode:
- **Budget** prevents skill amnesia (don't overwrite everything)
- **Buffer** prevents edit cycling (don't repeat failed edits)
- **Gate** prevents overfitting (only keep what generalizes)
- **Batch** prevents anecdotal fixes (address patterns, not instances)

---

## Based On

Inspired by [Microsoft Research SkillOpt](https://arxiv.org/abs/2605.23904) — Yifan Yang, Ziyang Gong, Weiquan Huang, et al. (2026). The first systematic text-space optimizer for agent skills. SkillTrain is an independent, MIT-licensed, OpenClaw-native implementation of the same principles.

---

*SkillTrain — because your skills should learn too.*
