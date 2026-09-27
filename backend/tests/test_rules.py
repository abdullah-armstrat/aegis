"""Tests for the deterministic fusion rules.

Each rule is tested across all three honest outcomes where applicable. Bundles are
constructed directly (no extractors run), so these are fast, deterministic, and pin the rule
logic precisely — exactly the reproducibility the rules layer exists to provide.
"""

from app.fusion.rules import (
    CAPTION_SCENE_OVERLAP_THRESHOLD,
    MIN_MARKERS_TO_FIRE,
    caption_scene_mismatch_rule,
    emotional_framing_rule,
    find_manipulation_markers,
    recycled_context_rule,
    run_rules,
)
from app.models import (
    EvidenceBundle,
    FlagStatus,
    FlagType,
    Meta,
    Modality,
    SceneDescription,
    Sentiment,
    WebMatch,
)


def _bundle(**kwargs) -> EvidenceBundle:
    kwargs.setdefault("meta", Meta(modality=Modality.IMAGE, source_ref="t.jpg"))
    return EvidenceBundle(**kwargs)


# --- emotional framing ---

def test_emotional_framing_fires_on_a_combination_of_markers():
    """Fires on a combination of deterministic markers, and names them."""
    b = _bundle(caption="ABSOLUTELY SHOCKING!! Share this before they delete it!!")
    flag = emotional_framing_rule(b)
    assert flag.type == FlagType.EMOTIONAL_FRAMING
    assert flag.status == FlagStatus.FIRED
    assert flag.what_to_check
    # The evidence must report the specific markers found, not just a score.
    assert "ALL-CAPS" in flag.evidence
    assert "exclamation" in flag.evidence
    assert "manipulation markers present" in flag.evidence


def test_emotional_framing_clear_on_neutral_wording():
    b = _bundle(caption="The committee published its quarterly transport figures on Tuesday.")
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert "None of the 4" in flag.evidence


def test_emotional_framing_requires_a_combination_not_a_single_marker():
    """A single marker must not fire — the rule fires only on a combination. This is what
    stops one stray urgency word from labelling an ordinary post as manipulative."""
    b = _bundle(caption="Urgent: the residents meeting has moved to Tuesday evening.")
    flag = emotional_framing_rule(b)
    assert len(find_manipulation_markers(b.caption)) < MIN_MARKERS_TO_FIRE
    assert flag.status == FlagStatus.CLEAR
    assert "Only 1 of 4" in flag.evidence


def test_emotional_framing_does_not_fire_on_sober_but_negative_text():
    """Regression for the construct-validity gap the marker rule closes: critical, negative,
    measured prose (the kind of critical news post the old rule flagged) carries no manipulation markers and must
    now clear, where the old sentiment-polarity proxy fired on it."""
    b = _bundle(
        caption=(
            "Residents say they are concerned about the proposed parking changes, and have "
            "asked the council to verify the wording and context before acting."
        ),
        # A strongly negative reading is present and must be IGNORED by this rule now.
        sentiment=Sentiment(label="negative", score=0.9994, source="caption"),
        extractor_status={"sentiment": FlagStatus.FIRED},
    )
    flag = emotional_framing_rule(b)
    assert flag.status == FlagStatus.CLEAR
    assert "manipulation" not in flag.plain_explanation


def test_emotional_framing_not_assessed_without_caption_text():
    """Markers are properties of text: no caption means the check could not run."""
    assert emotional_framing_rule(_bundle()).status == FlagStatus.NOT_ASSESSED
    assert emotional_framing_rule(_bundle(caption="   ")).status == FlagStatus.NOT_ASSESSED


def test_marker_detector_units():
    """Each of the four markers is detectable on its own."""
    assert any("ALL-CAPS" in m for m in find_manipulation_markers("THIS IS ALL SHOUTED TEXT"))
    assert any("exclamation" in m for m in find_manipulation_markers("Look at this!!"))
    assert any("urgency" in m for m in find_manipulation_markers("URGENT: read this"))
    assert any(
        "in-/out-group" in m
        for m in find_manipulation_markers("They are hiding the truth about it")
    )
    # Bare pronouns are deliberately not markers — too common in ordinary reporting.
    assert find_manipulation_markers("They said the changes affect us in March.") == []


# --- recycled context ---

def test_recycled_context_fires_with_matches():
    b = _bundle(
        web_matches=[WebMatch(url="https://e.com/2019", published_date="2019-03-04")],
        extractor_status={"reverse_image": FlagStatus.FIRED},
    )
    flag = recycled_context_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert "2019-03-04" in flag.evidence  # earliest date surfaced


def test_recycled_context_clear_when_searched_empty():
    b = _bundle(web_matches=[], extractor_status={"reverse_image": FlagStatus.CLEAR})
    assert recycled_context_rule(b).status == FlagStatus.CLEAR


def test_recycled_context_not_assessed_when_unknown():
    b = _bundle(web_matches=[], extractor_status={"reverse_image": FlagStatus.NOT_ASSESSED})
    assert recycled_context_rule(b).status == FlagStatus.NOT_ASSESSED


def test_recycled_context_not_assessed_when_status_missing():
    """Regression: a bundle where the reverse-image extractor never ran (no status key) and
    has no matches must be NOT_ASSESSED, never CLEAR — empty must not read as 'searched, none
    recycled'. This bug was caught by the scorecard 'nothing fires' test."""
    b = _bundle(web_matches=[], extractor_status={})
    assert recycled_context_rule(b).status == FlagStatus.NOT_ASSESSED


# --- recycled context: the five status paths of the date-gated rule ---

def _matched(posted_date=None, dates=("2019-03-04", "2021-08-17")):
    """A bundle whose lookup found the image, with appearances on the given dates."""
    return _bundle(
        web_matches=[
            WebMatch(url=f"https://e.com/{i}", published_date=d, hash_distance=2)
            for i, d in enumerate(dates)
        ],
        extractor_status={"reverse_image": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref="t.jpg", posted_date=posted_date),
    )


def test_recycled_fires_when_an_appearance_predates_the_posting_date():
    flag = recycled_context_rule(_matched(posted_date="2026-09-01"))
    assert flag.status == FlagStatus.FIRED
    assert "2019-03-04" in flag.evidence and "2026-09-01" in flag.evidence
    assert "2 of 2" in flag.evidence
    assert "Closest match differs by 2 of 64" in flag.evidence


def test_recycled_clear_when_every_appearance_is_after_the_posting_date():
    """The image is out there, but not from before this post: consistent with it being first."""
    flag = recycled_context_rule(_matched(posted_date="2018-12-31"))
    assert flag.status == FlagStatus.CLEAR
    assert "on or after" in flag.evidence


def test_recycled_same_day_appearance_is_not_earlier():
    """An appearance on the posting date itself may be the post being checked."""
    flag = recycled_context_rule(_matched(posted_date="2019-03-04", dates=("2019-03-04",)))
    assert flag.status == FlagStatus.CLEAR


def test_recycled_fires_without_a_posting_date_and_says_why():
    flag = recycled_context_rule(_matched(posted_date=None))
    assert flag.status == FlagStatus.FIRED
    assert "earliest known appearance is dated 2019-03-04" in flag.evidence
    assert "posting date" in flag.plain_explanation


def test_recycled_undated_appearances_never_read_as_clear():
    """Nothing dated is earlier, but one appearance has no date: the comparison is incomplete,
    so it must not be reported clear. This case is outside the rule's status table."""
    flag = recycled_context_rule(_matched(posted_date="2018-12-31", dates=("2019-03-04", None)))
    assert flag.status == FlagStatus.FIRED
    assert "carry no date" in flag.evidence


def test_recycled_without_posting_date_and_no_dates_at_all():
    flag = recycled_context_rule(_matched(posted_date=None, dates=(None,)))
    assert flag.status == FlagStatus.FIRED
    assert "None of the appearances is dated" in flag.evidence


# --- caption <-> scene mismatch ---

def test_caption_scene_fires_on_low_overlap():
    b = _bundle(
        caption="Massive flood devastates the city centre this morning",
        scene_descriptions=[SceneDescription(text="a dry empty street on a sunny day")],
    )
    flag = caption_scene_mismatch_rule(b)
    assert flag.type == FlagType.CAPTION_CONTENT_MISMATCH
    assert flag.status == FlagStatus.FIRED


def test_caption_scene_clear_on_high_overlap():
    b = _bundle(
        caption="a dry empty street on a sunny day with parked cars",
        scene_descriptions=[SceneDescription(text="a dry empty street on a sunny day")],
    )
    assert caption_scene_mismatch_rule(b).status == FlagStatus.CLEAR


def test_caption_scene_not_assessed_without_scene():
    """Before the captioner is wired (no scene_descriptions) this is NOT_ASSESSED, not a
    silent pass — the honest behaviour."""
    b = _bundle(caption="anything")
    assert caption_scene_mismatch_rule(b).status == FlagStatus.NOT_ASSESSED


# --- the worked mismatch example end-to-end through the rules ---

def test_run_rules_on_built_mismatch_example():
    """The canonical demo case: recycled flood image + wrong fresh caption. Rules alone should
    raise the caption↔scene mismatch AND the recycled-context flag (the rules-only baseline)."""
    b = _bundle(
        caption="URGENT: massive flood hitting the city right now, share immediately!",
        scene_descriptions=[SceneDescription(text="a calm dry residential street, parked cars")],
        sentiment=Sentiment(label="negative", score=0.97, source="caption"),
        web_matches=[
            WebMatch(url="https://news.example.com/2019/flood", published_date="2019-03-04")
        ],
        extractor_status={
            "sentiment": FlagStatus.FIRED,
            "reverse_image": FlagStatus.FIRED,
        },
    )
    flags = run_rules(b)
    fired = {f.type for f in flags if f.status == FlagStatus.FIRED}
    assert FlagType.CAPTION_CONTENT_MISMATCH in fired
    assert FlagType.RECYCLED_CONTEXT in fired
    assert FlagType.EMOTIONAL_FRAMING in fired
    # Exactly one flag per rule, always.
    assert len(flags) == 3


def test_recycled_evidence_explains_a_keypoint_match():
    """A match found by keypoints has no hash distance; the evidence must say how it was found."""
    b = _bundle(
        web_matches=[WebMatch(url="https://e.com/0", published_date="2019-03-04", keypoint_inliers=40)],
        extractor_status={"reverse_image": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, source_ref="t.jpg", posted_date="2026-09-01"),
    )
    flag = recycled_context_rule(b)
    assert flag.status == FlagStatus.FIRED
    assert "40 image details that line up" in flag.evidence
    assert "hash bits" not in flag.evidence
