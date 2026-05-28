#!/usr/bin/env python3
"""
SkillTrain CLI — Train and evaluate SKILL.md files.

Usage:
    python -m skilltrain train --skill SKILL.md --train-data data/train.json --scorer my_scorer.py
    python -m skilltrain train --skill SKILL.md --evaluate-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime


def load_scorer(path: str):
    """Load a scorer class from a Python file."""
    path = os.path.expanduser(os.path.abspath(path))
    spec = importlib.util.spec_from_file_location("user_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find the first Scorer subclass
    from skilltrain.scoring import Scorer
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, Scorer) and obj is not Scorer:
            return obj()
    raise ValueError(f"No Scorer subclass found in {path}")


def cmd_train(args):
    """Run the full SkillTrain training loop."""
    from skilltrain import SkillTrain

    scorer = load_scorer(args.scorer)

    trainer = SkillTrain(
        skill_path=args.skill,
        scorer=scorer,
        train_data=args.train_data,
        val_data=args.val_data,
        model=args.model,
        batch_size=args.batch_size,
        max_edits=args.max_edits,
        min_edits=args.min_edits,
        lr_schedule=args.lr_schedule,
        rollout_mode=args.rollout_mode,
        output_dir=args.output,
        verbose=not args.quiet,
    )

    if args.evaluate_only:
        from skilltrain import SkillTrain
        print(f"\n{'='*60}")
        print(f"SkillTrain — Evaluation Only")
        print(f"{'='*60}")
        print(f"Skill: {trainer.skill_path} ({len(trainer.current_skill)} chars)")
        print(f"{'='*60}\n")

        # Evaluate val set
        val = trainer._evaluate(trainer.val_tasks)
        print(f"Validation ({len(trainer.val_tasks)} tasks): hard={val['avg_hard']:.3f} soft={val['avg_soft']:.3f}")

        # Evaluate test set (if exists)
        test_path = args.val_data.replace("val.json", "test.json") if args.val_data else ""
        if test_path and os.path.exists(test_path):
            test_tasks = trainer._load(test_path)
            test = trainer._evaluate(test_tasks)
            print(f"Test ({len(test_tasks)} tasks): hard={test['avg_hard']:.3f} soft={test['avg_soft']:.3f}")

        return

    result = trainer.run(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
    trainer.save(os.path.join(trainer.output_dir, "best_skill.md"))

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Best val: {result.best_val_score:.3f}")
    print(f"Epochs: {result.total_epochs}")
    print(f"Edits: +{result.total_edits_applied}/-{result.total_edits_rejected}")
    print(f"Output: {result.output_dir}")
    for ep in result.epochs:
        print(f"  E{ep['epoch']}: hard={ep['avg_hard']:.3f} fail={ep['failure_rate']:.1%} "
              f"edits=+{ep['edits_applied']}/-{ep['edits_rejected']} "
              f"imp={'✅' if ep['improved'] else '❌'}")


def main():
    parser = argparse.ArgumentParser(description="SkillTrain — Train Your Agent Skills Like Neural Networks")
    subs = parser.add_subparsers(dest="command")

    # train subcommand
    p = subs.add_parser("train", help="Train a SKILL.md")
    p.add_argument("--skill", "-s", required=True, help="Path to SKILL.md file")
    p.add_argument("--train-data", "-t", required=True, help="Path to train.json")
    p.add_argument("--val-data", "-v", help="Path to val.json (auto-split if omitted)")
    p.add_argument("--scorer", "-c", required=True, help="Path to scorer Python file")
    p.add_argument("--epochs", "-e", type=int, default=4)
    p.add_argument("--steps-per-epoch", type=int, default=4)
    p.add_argument("--batch-size", "-b", type=int, default=4)
    p.add_argument("--max-edits", type=int, default=4)
    p.add_argument("--min-edits", type=int, default=1)
    p.add_argument("--lr-schedule", default="cosine", choices=["constant", "linear", "cosine", "autonomous"])
    p.add_argument("--model", "-m", default="deepseek-v4-pro:cloud")
    p.add_argument("--rollout-mode", default="ollama", choices=["ollama", "mock", "subprocess"])
    p.add_argument("--output", "-o")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
