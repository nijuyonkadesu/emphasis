"""Shared domain vocabulary for stressmark's engine and output layers."""

from enum import IntEnum, StrEnum

APP_NAME = "stressmark"


class PartOfSpeech(StrEnum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"


class DiscourseKeyType(StrEnum):
    HETERONYM = "heteronym"
    SURFACE = "surface"
    LEMMA = "lemma"


class WordClass(StrEnum):
    REDUCIBLE = "reducible"
    WEAK = "weak"
    CONTENT = "content"
    COMPOUND_ADJECTIVE = "compound-adj"
    COMPOUND_TAIL = "compound-tail"


class StressLevel(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NONE = "none"


class ProminenceTier(StrEnum):
    NUCLEAR = "nuclear"
    PROMINENT = "prominent"
    PRE_NUCLEAR = "pre-nuclear"
    SECONDARY = StressLevel.SECONDARY
    GIVEN = "given"
    REDUCED = "reduced"
    SUPPRESSED = "suppressed"


class Confidence(StrEnum):
    DICTIONARY = "dict"
    POS_RESOLVED = "dict-pos-resolved"
    AMBIGUOUS = "dict-flagged"
    PREDICTED = "predicted"
    COMPOUND_ADJECTIVE_RULE = "rule-9"
    REDUCIBLE = WordClass.REDUCIBLE


class StressRule(IntEnum):
    def __new__(cls, value, description):
        member = int.__new__(cls, value)
        member._value_ = value
        member.description = description
        return member

    TWO_SYLLABLE_NOUN = (1, "2-syllable noun -> stress 1st syllable")
    TWO_SYLLABLE_ADJECTIVE = (2, "2-syllable adjective -> stress 1st syllable")
    TWO_SYLLABLE_VERB = (3, "2-syllable verb -> stress 2nd syllable")
    IC_ENDING = (4, "-ic ending -> stress syllable before 'ic'")
    SION_TION_ENDING = (
        5,
        "-sion/-tion ending -> stress syllable before the ending",
    )
    CY_TY_PHY_GY_ENDING = (
        6,
        "-cy/-ty/-phy/-gy ending -> stress antepenultimate syllable",
    )
    AL_ENDING = (7, "-al ending -> stress antepenultimate syllable")
    COMPOUND_NOUN = (8, "compound noun -> stress 1st part")
    COMPOUND_ADJECTIVE = (9, "compound adjective -> stress 2nd part")


class OutputFormat(StrEnum):
    TERMINAL = "terminal"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
