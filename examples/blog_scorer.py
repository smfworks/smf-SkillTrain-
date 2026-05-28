"""
Example Blog Publishing Scorer

Copy this file and customize it for your own workflow.
Shows how to define gates for a typical content publishing pipeline.

Usage:
    from skilltrain import SkillTrain
    from examples.blog_scorer import BlogScorer

    trainer = SkillTrain(
        skill_path="~/.openclaw/skills/my-blog-skill/SKILL.md",
        scorer=BlogScorer(),
        train_data="data/train.json",
    )
    trainer.run(epochs=4, steps_per_epoch=4)
"""

from skilltrain import Gate, Scorer
from skilltrain.scoring import file_exists, url_ok, frontmatter_field


class BlogScorer(Scorer):
    """Score blog publishing tasks with 8 gates."""

    def score(self, result: dict, task: dict) -> "ScoringResult":
        gates = []

        # Gate: File created
        md_path = result.get("md_file_path", result.get("path", ""))
        md_ok = file_exists(md_path)
        gates.append(Gate("file_created", md_ok, "" if md_ok else f"No file at {md_path}"))

        # Gate: Slug matches filename
        slug = result.get("slug", "")
        slug_ok = slug and os.path.basename(md_path).replace(".md", "") == slug
        gates.append(Gate("slug_match", slug_ok, "" if slug_ok else f"Slug '{slug}' != filename"))

        # Gate: SEO excerpt under 160 chars
        excerpt = result.get("excerpt", "")
        ex_ok = len(excerpt) <= 160
        gates.append(Gate("excerpt_ok", ex_ok, "" if ex_ok else f"Excerpt is {len(excerpt)} chars (limit 160)", weight=0.5))

        # Gate: Hero image exists (CRITICAL — weight 1.5)
        img_path = result.get("image_path", result.get("img", ""))
        img_ok = file_exists(img_path)
        gates.append(Gate("image_exists", img_ok, "" if img_ok else "Hero image missing", weight=1.5))

        # Gate: Image path uses /images/blog/ prefix
        img_ref = result.get("hero_image_ref", result.get("image_url", ""))
        path_ok = img_ref.startswith("/images/blog/")
        gates.append(Gate("image_path_correct", path_ok, "" if path_ok else f"Wrong prefix: {img_ref[:40]}"))

        # Gate: Categories present
        cats = result.get("categories", [])
        cats_ok = len(cats) > 0
        gates.append(Gate("categories", cats_ok, "" if cats_ok else "No categories", weight=0.5))

        # Gate: Deployed (CRITICAL — weight 1.5)
        url = result.get("deployed_url", "")
        dep_ok = bool(url)
        gates.append(Gate("deployed", dep_ok, "" if dep_ok else "Not deployed", weight=1.5))

        # Gate: Git pulled before writing
        pulled = result.get("did_git_pull", False)
        gates.append(Gate("git_pull", pulled, "" if pulled else "No git pull"))

        return self.evaluate(gates)


import os
