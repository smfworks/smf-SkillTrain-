"""
Example Social Media Posting Scorer

Copy this file and customize it for your workflow.
Shows how to define gates for social media posting.

Usage:
    from skilltrain import SkillTrain
    from examples.social_scorer import SocialScorer

    trainer = SkillTrain(
        skill_path="~/.openclaw/skills/postiz/SKILL.md",
        scorer=SocialScorer(),
        train_data="data/train.json",
    )
    trainer.run(epochs=4, steps_per_epoch=4)
"""

from skilltrain import Gate, Scorer


class SocialScorer(Scorer):
    """Score social media posting tasks with 8 gates."""

    def score(self, result: dict, task: dict) -> "ScoringResult":
        gates = []

        # Gate: Media uploaded to Postiz before referencing (CRITICAL)
        uploaded = result.get("media_uploaded", False)
        gates.append(Gate(
            "media_uploaded", uploaded,
            "" if uploaded else "Media not uploaded — raw filesystem path passed instead",
            weight=1.5,
        ))

        # Gate: Postiz-verified URL used
        url = result.get("postiz_url", "")
        url_ok = bool(url) and ("postiz.com" in url or "cdn" in url)
        gates.append(Gate("media_url_valid", url_ok, "" if url_ok else "URL not Postiz-verified", weight=1.5))

        # Gate: POSTED — THE critical gate
        posted = result.get("posted", False)
        gates.append(Gate("posted", posted, "" if posted else "Postiz did not confirm delivery", weight=2.0))

        # Gate: Content matches what we intended
        intended = result.get("intended_content", "")
        actual = result.get("posted_content", "")
        content_ok = bool(actual) and (actual == intended or (intended and intended[:80] in actual))
        gates.append(Gate("content_correct", content_ok, "" if content_ok else "Content differs from intended", weight=1.5))

        # Gate: Posted within scheduled window (±15 min)
        scheduled = result.get("scheduled_time", "")
        actual_time = result.get("actual_post_time", "")
        on_time = True
        if scheduled and actual_time:
            try:
                from datetime import datetime
                s = datetime.fromisoformat(scheduled)
                a = datetime.fromisoformat(actual_time)
                on_time = abs((a - s).total_seconds()) <= 900
            except Exception:
                on_time = True
        gates.append(Gate("on_time", on_time, "" if on_time else "Posted off schedule", weight=0.5))

        # Gate: Platform character limits respected
        platform = result.get("platform", task.get("platform", "x"))
        limits = {"x": 280, "linkedin": 3000, "bluesky": 300, "tiktok": 2200}
        chars = result.get("char_count", result.get("chars", 0))
        limit = limits.get(platform, 5000)
        fmt_ok = chars <= limit if chars > 0 else True
        gates.append(Gate("format_ok", fmt_ok, "" if fmt_ok else f"{chars} chars > {platform} limit {limit}", weight=1.0))

        # Gate: Link present when required
        has_link = result.get("has_link", False)
        link_needed = task.get("link_required", False)
        link_ok = has_link if link_needed else True
        gates.append(Gate("link_present", link_ok, "" if link_ok else "Link required but missing", weight=0.5))

        # Gate: Not a duplicate of recent post
        duplicate = result.get("duplicate_check", False)
        gates.append(Gate("no_duplicate", not duplicate, "" if not duplicate else "Duplicate of recent post", weight=0.5))

        return self.evaluate(gates)
