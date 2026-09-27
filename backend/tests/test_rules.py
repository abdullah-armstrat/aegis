"""Tests for the deterministic fusion rules.

Each rule is tested across all three honest outcomes where applicable. Bundles are
constructed directly (no extractors run), so these are fast, deterministic, and pin the rule
logic precisely — exactly the reproducibility the rules layer exists to provide.
"""

import pytest

from app.config import get_settings
from app.fusion.rules import (
    CAPTION_CHECK_OFF_REASON,
    CAPTION_IMAGE_ONLY_THRESHOLD,
    CAPTION_IMAGE_SIMILARITY_THRESHOLD,
    CAPTION_SCENE_OVERLAP_THRESHOLD,
    CAPTION_TEXT_SIMILARITY_THRESHOLD,
    MIN_MARKERS_TO_FIRE,
    caption_scene_mismatch_rule,
    emotional_framing_rule,
    find_manipulation_markers,
    recycled_context_rule,
    run_rules,
)
from app.models import (
    CaptionMatch,
    EvidenceBundle,
    FlagStatus,
    FlagType,
    Meta,
    Modality,
    SceneDescription,
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
    assert "words in capitals" in flag.evidence
    assert "exclamation" in flag.evidence
    assert "style markers present" in flag.evidence
    assert "shouting style" in flag.plain_explanation
    assert not any(w in flag.plain_explanation.lower() for w in ("manipulat", "pressure", "sensational"))


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
    assert any("words in capitals" in m for m in find_manipulation_markers("THIS IS ALL SHOUTED TEXT"))
    assert any("exclamation" in m for m in find_manipulation_markers("Look at this!!"))
    assert any("urgency" in m for m in find_manipulation_markers("URGENT: read this"))
    assert any(
        "listed phrases" in m
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
    assert "found on a page dated 2019-03-04 (https://e.com/0)" in flag.evidence
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
    assert "None of the pages is dated" in flag.evidence


@pytest.mark.parametrize("posted, dates", [
    ("2026-09-01", ("2019-03-04", "2021-08-17")),   # a page dated earlier
    (None, ("2019-03-04", "2021-08-17")),           # no posting date
    ("2018-12-31", ("2019-03-04", None)),           # an undated page
    ("2018-12-31", ("2019-03-04", "2021-08-17")),   # every page dated after the post
])
def test_a_page_date_is_never_called_the_image_s_first_appearance(posted, dates):
    """A page's date is the page's: a Wikipedia article can be created years before a photo is added."""
    flag = recycled_context_rule(_matched(posted_date=posted, dates=dates))
    text = " ".join((flag.evidence, flag.plain_explanation, flag.what_to_check)).lower()
    assert "appeared on" not in text and "first appeared" not in text
    assert "the page may be older or newer than the image on it" in flag.plain_explanation
    if flag.status == FlagStatus.FIRED and posted and dates[1]:
        assert flag.evidence.startswith("Found on a page dated 2019-03-04 (https://e.com/0)")


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


# --- caption <-> picture by meaning ---

@pytest.fixture
def by_meaning(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", "meaning")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _scored(image: float | None, text: float | None, **kwargs) -> EvidenceBundle:
    kwargs.setdefault("caption", "Protesters fill the main square")
    kwargs.setdefault("scene_descriptions", [SceneDescription(text="a bowl of soup on a table")])
    return _bundle(caption_match=CaptionMatch(image_similarity=image, text_similarity=text), **kwargs)


LOW_IMAGE = CAPTION_IMAGE_SIMILARITY_THRESHOLD - 0.05
HIGH_IMAGE = CAPTION_IMAGE_SIMILARITY_THRESHOLD + 0.05
LOW_TEXT = CAPTION_TEXT_SIMILARITY_THRESHOLD - 0.05
HIGH_TEXT = CAPTION_TEXT_SIMILARITY_THRESHOLD + 0.05


def test_meaning_fires_only_when_both_similarities_are_low(by_meaning):
    flag = caption_scene_mismatch_rule(_scored(LOW_IMAGE, LOW_TEXT))
    assert flag.type == FlagType.CAPTION_CONTENT_MISMATCH
    assert flag.status == FlagStatus.FIRED
    assert "different kind of scene" in flag.plain_explanation
    # The evidence says what the numbers mean, and quotes what the picture appears to show.
    assert flag.evidence.count("weak match") == 2
    assert "a bowl of soup on a table" in flag.evidence


@pytest.mark.parametrize("image, text", [(LOW_IMAGE, HIGH_TEXT), (HIGH_IMAGE, LOW_TEXT), (HIGH_IMAGE, HIGH_TEXT)])
def test_meaning_clear_when_either_similarity_is_reasonable(by_meaning, image, text):
    flag = caption_scene_mismatch_rule(_scored(image, text))
    assert flag.status == FlagStatus.CLEAR
    assert "reasonable match" in flag.evidence


@pytest.mark.parametrize("image, text", [(LOW_IMAGE, LOW_TEXT), (HIGH_IMAGE, HIGH_TEXT)])
def test_meaning_always_states_what_it_cannot_catch(by_meaning, image, text):
    """A clear result must not read as 'the caption is accurate': the check sees only the kind
    of scene, never a wrong name, place or date."""
    flag = caption_scene_mismatch_rule(_scored(image, text))
    assert "cannot catch a wrong name, place or date" in flag.plain_explanation


def test_meaning_not_assessed_when_a_similarity_is_missing(by_meaning):
    b = _scored(LOW_IMAGE, None)
    b.caption_match.detail = "There was no scene description or on-screen text to compare with the caption."
    flag = caption_scene_mismatch_rule(b)
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert "no scene description" in flag.evidence


def test_meaning_not_assessed_when_models_did_not_run(by_meaning):
    b = _bundle(caption="anything", scene_descriptions=[SceneDescription(text="a street")])
    flag = caption_scene_mismatch_rule(b)
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert "did not run" in flag.evidence


def test_meaning_not_assessed_without_caption(by_meaning):
    flag = caption_scene_mismatch_rule(_scored(LOW_IMAGE, LOW_TEXT, caption=None))
    assert flag.status == FlagStatus.NOT_ASSESSED


def test_meaning_says_when_the_caption_was_cut(by_meaning):
    b = _scored(LOW_IMAGE, LOW_TEXT)
    b.caption_match.caption_truncated = True
    assert "only its beginning was compared" in caption_scene_mismatch_rule(b).evidence


def test_overlap_method_ignores_the_meaning_scores():
    """Word overlap stays selectable, and reads only the words, whatever the similarities say."""
    b = _scored(LOW_IMAGE, LOW_TEXT, caption="a bowl of soup on a table",
                 scene_descriptions=[SceneDescription(text="a bowl of soup on a table")])
    assert get_settings().caption_match_method == "overlap"
    assert caption_scene_mismatch_rule(b).status == FlagStatus.CLEAR


# --- caption <-> picture, picture similarity only ---

def _use_method(monkeypatch, method: str) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", method)
    get_settings.cache_clear()


def _image_scored(image: float | None, **kwargs) -> EvidenceBundle:
    kwargs.setdefault("caption", "Protesters fill the main square")
    return _bundle(caption_match=CaptionMatch(image_similarity=image), **kwargs)


def test_image_method_fires_below_its_threshold(monkeypatch):
    _use_method(monkeypatch, "image")
    flag = caption_scene_mismatch_rule(_image_scored(CAPTION_IMAGE_ONLY_THRESHOLD - 0.02))
    assert flag.status == FlagStatus.FIRED
    assert "weak match" in flag.evidence
    assert "different kind of scene" in flag.plain_explanation
    assert "cannot catch a wrong name, place or date" in flag.plain_explanation
    get_settings.cache_clear()


def test_image_method_clear_above_its_threshold_and_needs_no_scene(monkeypatch):
    """The picture score alone decides: no scene description is needed, none is read."""
    _use_method(monkeypatch, "image")
    flag = caption_scene_mismatch_rule(_image_scored(CAPTION_IMAGE_ONLY_THRESHOLD + 0.02))
    assert flag.status == FlagStatus.CLEAR
    assert "reasonable match" in flag.evidence
    assert "cannot catch a wrong name, place or date" in flag.plain_explanation
    get_settings.cache_clear()


def test_image_method_not_assessed_without_a_picture_score(monkeypatch):
    _use_method(monkeypatch, "image")
    b = _image_scored(None)
    b.caption_match.detail = "Caption-match models could not run: CLIP ViT-B/32 weights are not in the cache"
    flag = caption_scene_mismatch_rule(b)
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert "weights are not in" in flag.evidence
    get_settings.cache_clear()


def test_switched_off_check_is_not_assessed_and_says_why(monkeypatch):
    """Off never means clear: the scorecard shows the check and the reason it did not run."""
    _use_method(monkeypatch, "off")
    b = _scored(LOW_IMAGE, LOW_TEXT)
    flag = caption_scene_mismatch_rule(b)
    assert flag.type == FlagType.CAPTION_CONTENT_MISMATCH
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert flag.evidence == CAPTION_CHECK_OFF_REASON
    assert len(run_rules(b)) == 3
    get_settings.cache_clear()


# --- the worked mismatch example end-to-end through the rules ---

def test_run_rules_on_built_mismatch_example():
    """The canonical demo case: recycled flood image + wrong fresh caption. Rules alone should
    raise the caption↔scene mismatch AND the recycled-context flag (the rules-only baseline)."""
    b = _bundle(
        caption="URGENT: massive flood hitting the city right now, share immediately!",
        scene_descriptions=[SceneDescription(text="a calm dry residential street, parked cars")],
        web_matches=[
            WebMatch(url="https://news.example.com/2019/flood", published_date="2019-03-04")
        ],
        extractor_status={"reverse_image": FlagStatus.FIRED},
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
