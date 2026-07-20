from stressmark.engine import _POS_VOCAB_TO_TAG, resolve_word_by_pos
from stressmark.render import render_word

# The complete, real set of distinct `pos` values that appear in revdict's
# built index metadata (extracted directly from a real built index, not
# guessed) -- this is the actual input space revdict.models.stress.mark()
# can ever call resolve_word_by_pos() with. A contract test against this
# exact list catches drift between revdict's POS vocabulary and this
# module's tag mapping at CI time instead of silently, someday.
_REAL_REVDICT_POS_VALUES = [
    "adjective", "adv_phrase", "adverb", "article", "character", "circumfix",
    "conj", "contraction", "det", "infix", "interfix", "intj", "name", "noun",
    "num", "particle", "phrase", "postp", "prefix", "prep", "prep_phrase",
    "pron", "proverb", "punct", "suffix", "symbol", "verb",
]


def test_resolve_word_by_pos_never_raises_for_any_real_revdict_pos_value():
    for pos in _REAL_REVDICT_POS_VALUES:
        result = resolve_word_by_pos("record", pos)
        render_word(result)  # must also render cleanly, not just resolve


def test_resolve_word_by_pos_recognizes_exactly_the_four_revdict_pos_names():
    # If revdict's normalization (WordNet's _POS_NAMES, Wiktionary's
    # _POS_NORMALIZATION) or this module's _POS_VOCAB_TO_TAG ever drifts out
    # of sync, this is the test that catches it -- it pins the exact four
    # recognized keys and their Penn Treebank targets.
    assert _POS_VOCAB_TO_TAG == {
        "noun": "NN",
        "verb": "VB",
        "adjective": "JJ",
        "adverb": "RB",
    }


def test_resolve_word_by_pos_resolves_heteronym_as_noun():
    result = resolve_word_by_pos("record", "noun")

    assert result.primary == 0


def test_resolve_word_by_pos_resolves_heteronym_as_verb():
    result = resolve_word_by_pos("record", "verb")

    assert result.primary == 1


def test_resolve_word_by_pos_handles_a_plain_dictionary_word():
    result = resolve_word_by_pos("happy", "adjective")

    assert result.syllables == ["hap", "py"]
    assert result.primary == 0


def test_resolve_word_by_pos_falls_back_to_noun_tag_for_an_unrecognized_pos_string():
    # revdict's POS vocabulary includes things beyond noun/verb/adjective/adverb
    # (e.g. WordNet's "name" for proper nouns, or raw Wiktionary POS strings
    # like "article"/"prefix") -- any of these should fall back to treating
    # the word as a common noun rather than raising or guessing wildly.
    result = resolve_word_by_pos("record", "name")

    assert result.primary == 0  # noun reading


def test_resolve_word_by_pos_predicts_words_absent_from_the_dictionary():
    result = resolve_word_by_pos("kubernetes", "noun")

    assert result.confidence == "predicted"
    assert len(result.syllables) > 1


def test_render_word_uppercases_primary_and_styles_the_rest():
    result = resolve_word_by_pos("happy", "adjective")

    text = render_word(result)

    assert text.plain == "HAPpy"
    spans = {(s.start, s.end): s.style for s in text.spans}
    assert spans[(0, 3)] == "bold yellow"
    assert spans[(3, 5)] == "grey62"


def test_render_word_marks_predicted_words_with_the_confidence_symbol():
    result = resolve_word_by_pos("kubernetes", "noun")

    text = render_word(result)

    assert text.plain == "kuBERnetes≈"


def test_resolve_word_by_pos_exposes_phonemes_for_a_dictionary_word():
    result = resolve_word_by_pos("cat", "noun")
    assert result.phonemes == ["K", "AE1", "T"]


def test_resolve_word_by_pos_exposes_the_heteronym_resolved_phonemes():
    """record(noun) and record(verb) must expose DIFFERENT phoneme lists --
    proves .phonemes reflects the POS-resolved variant, not a naive
    first-CMUdict-entry lookup that would return the same phonemes
    regardless of POS."""
    noun = resolve_word_by_pos("record", "noun")
    verb = resolve_word_by_pos("record", "verb")
    assert noun.phonemes == ["R", "EH1", "K", "ER0", "D"]
    assert verb.phonemes == ["R", "IH0", "K", "AO1", "R", "D"]
    assert noun.phonemes != verb.phonemes


def test_resolve_word_by_pos_exposes_phonemes_for_a_g2p_predicted_word():
    """A word with essentially no chance of being in CMUdict -- proves the
    G2P-prediction branch also populates .phonemes, not just the two
    dictionary-lookup branches."""
    result = resolve_word_by_pos("zxqvorplitude", "noun")
    assert isinstance(result.phonemes, list)
    assert len(result.phonemes) > 0


def test_resolve_word_by_pos_gives_an_empty_phonemes_list_for_a_reducible_word():
    """Function words never get a phoneme lookup at all -- .phonemes must
    default to [] rather than being missing or None, so callers can always
    do `if result.phonemes:` without a hasattr/None check. Uses "is" (a
    BE_FORMS word, classified as reducible regardless of the POS argument)
    rather than "the" -- resolve_word_by_pos's coarse POS-vocabulary
    mapping (noun/verb/adjective/adverb -> NN/VB/JJ/RB) has no way to
    express "this is a determiner", so an article like "the" would not
    actually classify as reducible via this entrypoint without a
    production-code change this task doesn't need to make."""
    result = resolve_word_by_pos("is", "verb")
    assert result.cls == "reducible"
    assert result.phonemes == []
