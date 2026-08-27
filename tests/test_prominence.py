import pytest

from stressmark.engine import analyze
from stressmark.render import _render_rich_text


def _words(text, nuclear_only=False):
    raw_tokens, results = analyze(text, nuclear_only=nuclear_only)
    return [
        (token.lower(), result)
        for (is_word, token), result in zip(raw_tokens, results)
        if is_word
    ]


def _occurrences(text, word, nuclear_only=False):
    return [
        result
        for token, result in _words(text, nuclear_only=nuclear_only)
        if token == word.lower()
    ]


def _nuclear_words(text):
    return [token for token, result in _words(text) if result.tier == "nuclear"]


def test_simple_rightmost_nuclear_and_cross_sentence_givenness_stay_intact():
    words = _words("John bought a car. John washed the car.")

    assert [token for token, result in words if result.tier == "nuclear"] == ["car", "washed"]
    assert [result.tier for token, result in words if token == "john"] == ["prominent", "given"]
    assert [result.tier for token, result in words if token == "car"] == ["nuclear", "given"]


@pytest.mark.parametrize("boundary", [".", "!", "?", ";", ":"])
def test_every_documented_major_boundary_starts_a_new_nuclear_phrase(boundary):
    text = f"John arrived{boundary} Mary departed."

    assert _nuclear_words(text) == ["arrived", "departed"]
    assert _occurrences(text, "Mary")[0].tier == "prominent"


def test_comma_creates_an_intermediate_phrase_with_its_own_nuclear_accent():
    assert _nuclear_words("John arrived, Mary departed.") == ["arrived", "departed"]


def test_discourse_marker_starts_an_intermediate_phrase_without_becoming_its_nucleus():
    words = _words("We expected rain however the sun appeared.")

    assert [token for token, result in words if result.tier == "nuclear"] == ["rain", "appeared"]
    assert next(result for token, result in words if token == "however").tier == "given"


@pytest.mark.parametrize("particle", ["Only", "Exactly", "Especially"])
def test_focus_particle_associate_overrides_rightmost_default(particle):
    text = f"{particle} John came."

    assert _occurrences(text, "John")[0].tier == "nuclear"
    assert _occurrences(text, "came")[0].tier != "nuclear"


def test_not_x_but_y_marks_both_contrasts_and_makes_y_nuclear():
    words = _words("They selected not red but blue paint yesterday.")

    assert next(result for token, result in words if token == "red").tier == "prominent"
    assert next(result for token, result in words if token == "blue").tier == "nuclear"


def test_wh_answer_receives_nuclear_accent_and_following_material_is_deaccented():
    words = _words("Who came? John came yesterday.")

    assert [result.tier for token, result in words if token == "john"] == ["nuclear"]
    assert [result.tier for token, result in words if token == "came"] == ["nuclear", "given"]
    assert next(result for token, result in words if token == "yesterday").tier == "given"


def test_givenness_uses_lemmas_instead_of_exact_inflected_spellings():
    words = _words("A dog barked. The dogs slept.")

    assert next(result for token, result in words if token == "dogs").tier == "given"
    assert next(result for token, result in words if token == "slept").tier == "nuclear"


def test_exact_repetition_stays_given_even_when_the_pos_tag_changes():
    words = _words("I caught you prompting. The prompting was the problem.")

    assert [result.tier for token, result in words if token == "prompting"] == [
        "nuclear",
        "given",
    ]


def test_known_heteronym_senses_are_not_falsely_treated_as_the_same_referent():
    words = _words("They object to the strange object.")

    assert [result.tier for token, result in words if token == "object"] == [
        "pre-nuclear",
        "nuclear",
    ]


def test_pos_tagging_does_not_leak_context_across_major_boundaries():
    words = _words("Rain fell. Dogs bark loudly.")
    bark = next(result for token, result in words if token == "bark")

    assert bark.tag == "VBP"
    assert bark.cls == "content"


def test_temporal_noun_after_a_noun_is_not_mistaken_for_a_compound_tail():
    words = _words("Mary bought apples yesterday.")
    yesterday = next(result for token, result in words if token == "yesterday")

    assert yesterday.cls == "content"
    assert yesterday.tier == "nuclear"


def test_compound_noun_rule_remains_active_and_is_explainable():
    words = _words("The tennis ball bounced.")
    tennis = next(result for token, result in words if token == "tennis")
    ball = next(result for token, result in words if token == "ball")

    assert tennis.rule == 8
    assert ball.cls == "compound-tail"
    assert ball.tier == "suppressed"


def test_repeat_inside_one_phrase_is_given_before_nuclear_selection():
    words = _words("John likes John.")

    assert [result.tier for token, result in words if token == "john"] == ["prominent", "given"]
    assert next(result for token, result in words if token == "likes").tier == "nuclear"


def test_focus_is_local_to_the_cued_occurrence_and_does_not_leak_globally():
    words = _words("John slept. Only John returned. John yawned.")

    assert [result.tier for token, result in words if token == "john"] == [
        "prominent",
        "nuclear",
        "given",
    ]


def test_nuclear_only_keeps_the_correct_shifted_focus_and_demotes_every_other_content_peak():
    raw_tokens, results = analyze("Only John came.", nuclear_only=True)
    words = [
        (token.lower(), result)
        for (is_word, token), result in zip(raw_tokens, results)
        if is_word
    ]

    assert next(result for token, result in words if token == "john").primary >= 0
    assert all(
        result.primary < 0
        for token, result in words
        if token != "john" and result.cls in ("content", "compound-adj")
    )
    rendered = _render_rich_text(raw_tokens, results)
    assert rendered.plain == "only JOHN came."
    assert [span.style for span in rendered.spans] == [
        "dim",
        "bold reverse yellow",
        "dim",
    ]
