"""
stressmark.engine
Core pipeline: text -> per-word stress analysis (primary/secondary/unstressed),
sentence-level nuclear-stress tiering, and confidence tagging.
"""
import re

import nltk
import pyphen
from g2p_en import G2p
from nltk import pos_tag
from nltk.corpus import cmudict
from nltk.stem import WordNetLemmatizer

from stressmark.model import (
    Confidence,
    DiscourseKeyType,
    PartOfSpeech,
    ProminenceTier,
    StressRule,
    WordClass,
)


def _ensure_nltk_data():
    needed = [
        ("corpora/cmudict", "cmudict"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(f"{path}.zip")
            except LookupError:
                nltk.download(pkg, quiet=True)

_ensure_nltk_data()

_dic = pyphen.Pyphen(lang='en_US')
_cmu = cmudict.dict()
_g2p = G2p()
_lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Prominence tiers (AM theory + information structure)
# nuclear > prominent > pre-nuclear > given > reduced
# ---------------------------------------------------------------------------
TIER_ORDER = [tier.value for tier in ProminenceTier]

BACKWARD_FOCUS_PARTICLE = "alone"
NEGATION_MARKER = "not"
CONTRAST_CONJUNCTION = "but"

# Focus particles that trigger contrastive focus on their associate
FOCUS_PARTICLES = {"only", "even", "just", "also", BACKWARD_FOCUS_PARTICLE, "merely", "simply",
                    "exactly", "precisely", "specifically", "particularly",
                    "especially", "mostly", "mainly", "largely", "exclusively",
                    "solely"}

# Structural contrast markers handled by the local not-X-but-Y scan.
CONTRAST_MARKERS = {NEGATION_MARKER, CONTRAST_CONJUNCTION}

# Discourse markers that often signal IP boundaries
DISCOURSE_MARKERS = {"however", "therefore", "moreover", "furthermore", "nevertheless",
                      "meanwhile", "consequently", "thus", "hence", "indeed", "actually",
                      "basically", "essentially", "frankly", "honestly", "obviously"}

# ---------------------------------------------------------------------------
# Heteronyms: words whose stress pattern depends on part of speech.
# Auto-detected from CMUdict (any word with >1 stress-distinct pronunciation),
# then hand-resolved: which variant is the noun reading, which is the verb.
# ---------------------------------------------------------------------------

def _heteronym(noun_pronunciation, verb_pronunciation):
    return {
        PartOfSpeech.NOUN: noun_pronunciation,
        PartOfSpeech.VERB: verb_pronunciation,
    }


HETERONYMS = {
    "record": _heteronym(['R','EH1','K','ER0','D'], ['R','IH0','K','AO1','R','D']),
    "object": _heteronym(['AA1','B','JH','EH0','K','T'], ['AH0','B','JH','EH1','K','T']),
    "export": _heteronym(['EH1','K','S','P','AO0','R','T'], ['IH0','K','S','P','AO1','R','T']),
    "import": _heteronym(['IH1','M','P','AO0','R','T'], ['IH0','M','P','AO1','R','T']),
    "contract": _heteronym(['K','AA1','N','T','R','AE0','K','T'], ['K','AH0','N','T','R','AE1','K','T']),
    "present": _heteronym(['P','R','EH1','Z','AH0','N','T'], ['P','R','IH0','Z','EH1','N','T']),
    "produce": _heteronym(['P','R','OW1','D','UW2','S'], ['P','R','AH0','D','UW1','S']),
    "project": _heteronym(['P','R','AA1','JH','EH0','K','T'], ['P','R','AH0','JH','EH1','K','T']),
    "progress": _heteronym(['P','R','AA1','G','R','EH2','S'], ['P','R','AH0','G','R','EH1','S']),
    "conduct": _heteronym(['K','AA1','N','D','AH0','K','T'], ['K','AH0','N','D','AH1','K','T']),
    "increase": _heteronym(['IH1','N','K','R','IY2','S'], ['IH0','N','K','R','IY1','S']),
    "decrease": _heteronym(['D','IY1','K','R','IY2','S'], ['D','IH0','K','R','IY1','S']),
    "permit": _heteronym(['P','ER1','M','IH0','T'], ['P','ER0','M','IH1','T']),
    "address": _heteronym(['AE1','D','R','EH2','S'], ['AH0','D','R','EH1','S']),
    "contrast": _heteronym(['K','AA1','N','T','R','AE2','S','T'], ['K','AH0','N','T','R','AE1','S','T']),
    "subject": _heteronym(['S','AH1','B','JH','IH0','K','T'], ['S','AH0','B','JH','EH1','K','T']),
    "extract": _heteronym(['EH1','K','S','T','R','AE0','K','T'], ['IH0','K','S','T','R','AE1','K','T']),
    "insult": _heteronym(['IH1','N','S','AH0','L','T'], ['IH0','N','S','AH1','L','T']),
    "perfect": _heteronym(['P','ER1','F','IH0','K','T'], ['P','ER0','F','EH1','K','T']),
    "suspect": _heteronym(['S','AH1','S','P','EH0','K','T'], ['S','AH0','S','P','EH1','K','T']),
}

_POS_VOCAB_TO_TAG = {
    PartOfSpeech.NOUN: "NN",
    PartOfSpeech.VERB: "VB",
    PartOfSpeech.ADJECTIVE: "JJ",
    PartOfSpeech.ADVERB: "RB",
}
if set(_POS_VOCAB_TO_TAG) != set(PartOfSpeech):
    raise ValueError("Part-of-speech tag mapping is incomplete")
PAST_PARTICIPLE_TAG = "VBN"
NOUN_TAGS = {_POS_VOCAB_TO_TAG[PartOfSpeech.NOUN], "NNS", "NNP", "NNPS"}
VERB_TAGS = {
    _POS_VOCAB_TO_TAG[PartOfSpeech.VERB],
    "VBD",
    "VBG",
    PAST_PARTICIPLE_TAG,
    "VBP",
    "VBZ",
}

# ---------------------------------------------------------------------------
# Word classification: reducible (fully suppressible function word) /
# weak (function-class but keeps its own lexical stress) / content
# ---------------------------------------------------------------------------

CLOSED_CLASS_TAGS = {"DT", "IN", "CC", "TO", "PRP$", "MD", "EX"}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DO_FORMS = {"do", "does", "did"}
HAVE_FORMS = {"have", "has", "had"}
TEMPORAL_ADVERBIALS = {"today", "tomorrow", "tonight", "yesterday"}

SCHWA_MAP = {
    "a": "ə", "an": "ən", "the": "thə", "to": "tə", "of": "əv", "and": "ən",
    "for": "fər", "your": "yər", "are": "ər", "was": "wəz", "were": "wər",
    "can": "kən", "could": "kəd", "would": "wəd", "should": "shəd", CONTRAST_CONJUNCTION: "bət",
    "that": "thət", "than": "thən", "as": "əz", "from": "frəm", "or": "ər",
    "our": "ər", "their": "thər", "has": "həz", "have": "həv", "had": "həd",
    "am": "əm", "some": "səm", "his": "ihz",
}

def syllable_count(word):
    h = _dic.inserted(word.lower())
    return max(1, h.count("-") + 1)

def classify(word, tag, sent_tags_after, lower):
    """Return 'reducible' | 'weak' | 'content' for a word given its POS tag
    and the tags immediately following it within the same clause."""
    if CONTRACTION_FRAGMENT.match(word):
        return WordClass.REDUCIBLE
    if lower in BE_FORMS:
        return WordClass.REDUCIBLE
    if lower in DO_FORMS:
        return (
            WordClass.REDUCIBLE
            if _POS_VOCAB_TO_TAG[PartOfSpeech.VERB] in sent_tags_after[:4]
            else WordClass.CONTENT
        )
    if lower in HAVE_FORMS:
        return (
            WordClass.REDUCIBLE
            if PAST_PARTICIPLE_TAG in sent_tags_after[:4]
            else WordClass.CONTENT
        )
    if tag in CLOSED_CLASS_TAGS:
        return WordClass.REDUCIBLE if syllable_count(word) <= 1 else WordClass.WEAK
    return WordClass.CONTENT

# ---------------------------------------------------------------------------
# Syllabification: split the WRITTEN word into syllables (pyphen, TeX-style
# hyphenation patterns) -- far more reliable than ad hoc vowel-grouping.
# ---------------------------------------------------------------------------

ONSETS2 = {"bl","br","cl","cr","dr","fl","fr","gl","gr","pl","pr","sc","sk","sl",
           "sm","sn","sp","st","sw","tr","tw","wr","ch","sh","th","ph","wh","qu"}
ONSETS3 = {"scr","spl","spr","str","squ","thr","shr"}

def _consonant_cut(gap_start, gap_end, lower):
    gap_len = gap_end - gap_start
    if gap_len <= 1:
        return gap_start
    gap = lower[gap_start:gap_end]
    if gap_len == 2:
        return gap_start if gap in ONSETS2 else gap_start + 1
    if gap_len == 3:
        if gap in ONSETS3:
            return gap_start
        if gap[1:] in ONSETS2:
            return gap_start + 1
        return gap_start + 1
    return gap_end - 2

def _vowel_group_syllabify(word):
    """Fallback orthographic syllabifier (vowel-grouping with onset-cluster
    awareness), used only when pyphen refuses to hyphenate a word at all."""
    lower = word.lower()
    spans = [(m.start(), m.end()) for m in re.finditer(r"[aeiouy]+", lower)]
    if not spans:
        return [word]
    if len(spans) >= 2:
        last = spans[-1]
        is_final_single_e = (last[1] == len(word) and last[1] - last[0] == 1
                              and lower[last[0]] == "e")
        if is_final_single_e and not re.search(r"[bcdfgklmnprstvz]le$", lower):
            spans = spans[:-1]
    if not spans:
        return [word]
    boundaries = [0]
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        boundaries.append(_consonant_cut(e1, s2, lower))
    boundaries.append(len(word))
    out = [word[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    return [s for s in out if s]

def syllabify(word, min_syllables=1):
    h = _dic.inserted(word)
    parts = h.split("-")
    if len(parts) >= min_syllables and len(parts) > 1:
        return parts
    if min_syllables > 1:
        fallback = _vowel_group_syllabify(word)
        if len(fallback) >= len(parts):
            return fallback
    return parts if parts else [word]

def map_stress_index(phonetic_idx, phonetic_count, ortho_count):
    if ortho_count == phonetic_count:
        return max(0, min(ortho_count - 1, phonetic_idx))
    if ortho_count <= 1 or phonetic_count <= 1:
        return 0
    ratio = phonetic_idx / (phonetic_count - 1)
    idx = round(ratio * (ortho_count - 1))
    return max(0, min(ortho_count - 1, idx))

STRESS_BEARING_PREFIXES = (
    "anti", "multi", "non", "semi", "auto", "inter", "super", "ultra",
    "counter", "pseudo", "quasi", "extra", "proto", "neo", "post", "pre",
    "over", "under", "sub", "co",
)

def _collapse_secondary_clashes(secondary, primary, word=""):
    """CMUdict marks vowel reduction, not strictly rhythmic prominence, so it
    can mark several adjacent non-primary syllables as level-2 in a row.
    Two distinct real phenomena produce this:

    1. Stress-bearing prefixes (anti-, multi-, non-...) keep their OWN fixed
       stress no matter where the primary stress later falls -- this is a
       lexical fact about the prefix, not a rhythmic effect. If the word
       starts with one of these and syllable 0 is in the clash, syllable 0
       wins (verified against anticancer/AN, antidepressants/AN, multinational/MUL).

    2. Otherwise (plain suffix-driven stress shift, e.g. exploitation,
       acceleration, absolutism), English rhythmic alternation means the real
       secondary beat sits an EVEN number of syllables from the primary;
       verified against acceleration/CEL, exploitation/EX, absolutism/LU.
    """
    if len(secondary) <= 1:
        return set(secondary)
    has_prefix = any(word.lower().startswith(p) for p in STRESS_BEARING_PREFIXES)
    ordered = sorted(secondary)
    clusters, cur = [], [ordered[0]]
    for x in ordered[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    kept = set()
    for cluster in clusters:
        if has_prefix and 0 in cluster:
            kept.add(0)
            continue
        even = [i for i in cluster if (primary - i) % 2 == 0]
        kept |= set(even) if even else {max(cluster, key=lambda i: abs(i - primary))}
    return kept

def stress_positions_for_pron(pron, ortho_count, word=""):
    """Given an ARPAbet pronunciation (list of phonemes) and the number of
    written syllables, return (primary_idx, set_of_secondary_idx) in
    orthographic-syllable space."""
    stresses = [p[-1] for p in pron if p[-1].isdigit()]
    phonetic_count = len(stresses)
    if phonetic_count == 0:
        return 0, set()
    primary = stresses.index("1") if "1" in stresses else 0
    secondary = [i for i, s in enumerate(stresses) if s == "2"]
    secondary = _collapse_secondary_clashes(secondary, primary, word)
    primary_ortho = map_stress_index(primary, phonetic_count, ortho_count)
    secondary_ortho = {map_stress_index(i, phonetic_count, ortho_count) for i in secondary}
    secondary_ortho.discard(primary_ortho)
    return primary_ortho, secondary_ortho

# ---------------------------------------------------------------------------
# The 9 named rules -- explanatory layer, not the decision mechanism.
# ---------------------------------------------------------------------------

RULES = {rule: rule.description for rule in StressRule}

def explain_rule(word, tag, ortho_sylls, primary_idx):
    lower = word.lower()
    n = len(ortho_sylls)
    if n == 2:
        if tag in NOUN_TAGS and primary_idx == 0:
            return StressRule.TWO_SYLLABLE_NOUN
        if tag == _POS_VOCAB_TO_TAG[PartOfSpeech.ADJECTIVE] and primary_idx == 0:
            return StressRule.TWO_SYLLABLE_ADJECTIVE
        if tag in VERB_TAGS and primary_idx == 1:
            return StressRule.TWO_SYLLABLE_VERB
    if lower.endswith("ic") and primary_idx == n - 2:
        return StressRule.IC_ENDING
    if (lower.endswith("sion") or lower.endswith("tion")) and primary_idx == n - 2:
        return StressRule.SION_TION_ENDING
    if any(lower.endswith(s) for s in ("cy", "ty", "phy", "gy")) and primary_idx == max(0, n - 3):
        return StressRule.CY_TY_PHY_GY_ENDING
    if lower.endswith("al") and primary_idx == max(0, n - 3):
        return StressRule.AL_ENDING
    return None

# ---------------------------------------------------------------------------
# Main per-word resolver
# ---------------------------------------------------------------------------

class WordResult:
    def __init__(self, raw):
        self.raw = raw
        self.is_word = False
        self.is_heteronym = False
        self.syllables = []
        self.primary = -1
        self.secondary = set()
        self.confidence = None
        self.tier = None
        self.rule = None
        self.cls = None
        self.tag = None
        self.compound_parts = None
        self.phonemes = []

def _primary_positions_with_one(prons):
    """Primary-stress index for each pronunciation variant that actually
    marks a primary stress (excludes degenerate fully-reduced fast-speech
    variants that mark no '1' at all)."""
    out = []
    for p in prons:
        stresses = [ph[-1] for ph in p if ph[-1].isdigit()]
        if "1" in stresses:
            out.append(stresses.index("1"))
    return out

def resolve_word(raw, tag, sent_tags_after):
    lower = raw.lower()
    r = WordResult(raw)
    r.is_word = True
    r.is_heteronym = lower in HETERONYMS
    r.cls = classify(raw, tag, sent_tags_after, lower)

    if r.cls == WordClass.REDUCIBLE:
        r.syllables = syllabify(raw)
        r.confidence = Confidence.REDUCIBLE
        return r

    # --- heteronym check (resolved by POS) ---
    if lower in HETERONYMS:
        want_verb = tag in VERB_TAGS
        variant = (
            HETERONYMS[lower][PartOfSpeech.VERB]
            if want_verb
            else HETERONYMS[lower][PartOfSpeech.NOUN]
        )
        phonetic_n = sum(1 for p in variant if p[-1].isdigit())
        ortho = syllabify(raw, min_syllables=phonetic_n)
        r.syllables = ortho
        primary, secondary = stress_positions_for_pron(variant, len(ortho), lower)
        r.primary, r.secondary, r.confidence = (
            primary,
            secondary,
            Confidence.POS_RESOLVED,
        )
        r.phonemes = variant
        return r

    # --- plain dictionary lookup ---
    if lower in _cmu:
        prons = _cmu[lower]
        pron = prons[0]
        phonetic_n = sum(1 for p in pron if p[-1].isdigit())
        ortho = syllabify(raw, min_syllables=phonetic_n)
        r.syllables = ortho
        primary, secondary = stress_positions_for_pron(pron, len(ortho), lower)
        r.primary, r.secondary = primary, secondary
        # Only flag genuine ambiguity: multiple variants that disagree on
        # WHERE the primary stress falls (ignore variants with no primary
        # marked at all -- those are just reduced fast-speech allophones,
        # and ignore disagreement when the word is a single syllable, since
        # there's nothing to disambiguate).
        primaries = set(_primary_positions_with_one(prons))
        r.confidence = (
            Confidence.AMBIGUOUS
            if (len(ortho) > 1 and len(primaries) > 1)
            else Confidence.DICTIONARY
        )
        r.phonemes = pron
        return r

    # --- not in dictionary: consult G2P first to find the real phonetic
    # syllable count, THEN syllabify with that as a hint. (Checking pyphen's
    # unhinted output first is wrong -- it sometimes refuses to hyphenate
    # unfamiliar words at all, which would falsely look monosyllabic.)
    try:
        phones = [p for p in _g2p(raw) if p != " "]
        phonetic_n = sum(1 for p in phones if p[-1].isdigit())
    except Exception:
        phones, phonetic_n = [], 1
    r.phonemes = phones

    ortho = syllabify(raw, min_syllables=max(phonetic_n, 1))
    r.syllables = ortho

    if phonetic_n <= 1 or len(ortho) == 1:
        r.primary, r.confidence = 0, (
            Confidence.DICTIONARY
            if phonetic_n <= 1
            else Confidence.PREDICTED
        )
        return r

    primary, secondary = stress_positions_for_pron(phones, len(ortho), lower)
    r.primary, r.secondary, r.confidence = primary, secondary, Confidence.PREDICTED
    return r


def resolve_word_by_pos(word, pos):
    """Resolve a single word's stress pattern given an ALREADY-KNOWN part of
    speech, bypassing this module's own POS-tagging and heteronym-guessing
    entirely. Intended for callers (like revdict) that already know the
    correct sense-specific POS from their own dictionary data -- both
    faster (no POS-tagger model needed for this path) and more accurate
    (no context-free guessing on an isolated word, which is exactly where
    heteronym resolution like record/object needs real context).

    `pos` uses revdict's vocabulary ("noun"/"verb"/"adjective"/"adverb");
    anything else (e.g. WordNet's "name" for proper nouns, or a raw
    Wiktionary POS string) falls back to treating the word as a common
    noun.
    """
    tag = _POS_VOCAB_TO_TAG.get(pos, "NN")
    return resolve_word(word, tag, [])

# ---------------------------------------------------------------------------
# Sentence-level pass: tokenize, tag, classify, resolve, then apply
# compound rules (8/9), nuclear-stress rule, and given/repeat backgrounding.
# ---------------------------------------------------------------------------

CLAUSE_BOUNDARY = re.compile(r"[.!?;:,]")
MAJOR_IP_BOUNDARY = re.compile(r"[.!?;:]")
INTERMEDIATE_IP_BOUNDARY = re.compile(r",")
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|'(?:s|re|ve|ll|d|m)\b|n't\b", re.IGNORECASE)
CONTRACTION_FRAGMENT = re.compile(r"^'?(s|re|ve|ll|d|m|t)$", re.IGNORECASE)

def tokenize_with_spans(text):
    """Tokenize while keeping the literal separator text between tokens so
    we can reconstruct the original spacing/punctuation exactly. Splits
    contraction/possessive suffixes ('s, n't, 're...) the way NLTK's own
    tokenizer and POS tagger expect."""
    tokens = []
    pos = 0
    for m in WORD_PATTERN.finditer(text):
        if m.start() > pos:
            tokens.append((False, text[pos:m.start()]))
        tokens.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        tokens.append((False, text[pos:]))
    return tokens


def _separator_after_word(raw_tokens, word_indices, position):
    """Return literal separator text after one entry in ``word_indices``."""
    idx = word_indices[position]
    end = word_indices[position + 1] if position + 1 < len(word_indices) else len(raw_tokens)
    return "".join(raw_tokens[k][1] for k in range(idx + 1, end) if not raw_tokens[k][0])


def _split_word_indices(raw_tokens, word_indices, boundary):
    """Split word-token indices after separators matching ``boundary``."""
    groups = []
    current = []
    for position, idx in enumerate(word_indices):
        current.append(idx)
        if boundary.search(_separator_after_word(raw_tokens, word_indices, position)):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _discourse_key(word, tag):
    """Return a conservative POS-aware lemma for discourse-givenness tracking."""
    lower = word.lower()
    if tag.startswith("N"):
        wordnet_pos = "n"
    elif tag.startswith("V"):
        wordnet_pos = "v"
    elif tag.startswith("J"):
        wordnet_pos = "a"
    elif tag.startswith("R"):
        wordnet_pos = "r"
    else:
        return lower
    try:
        return _lemmatizer.lemmatize(lower, wordnet_pos)
    except LookupError:
        # Analysis remains usable in an offline/minimal NLTK installation.
        return lower


def _discourse_keys(word, tag):
    """Return surface and lemma keys without conflating known heteronym senses."""
    lower = word.lower()
    if lower in HETERONYMS:
        sense = PartOfSpeech.VERB if tag in VERB_TAGS else PartOfSpeech.NOUN
        return {(DiscourseKeyType.HETERONYM, lower, sense)}
    return {
        (DiscourseKeyType.SURFACE, lower),
        (DiscourseKeyType.LEMMA, _discourse_key(word, tag)),
    }


def analyze(text, nuclear_only=False):
    raw_tokens = tokenize_with_spans(text)
    word_indices = [i for i, (isw, _) in enumerate(raw_tokens) if isw]
    # Tag each major phrase independently. Feeding punctuation-stripped words
    # from the entire document to one tagger pass leaks context across sentence
    # boundaries and can turn a new sentence's verb into a compound-noun tail.
    tag_by_idx = {}
    for sentence_indices in _split_word_indices(
        raw_tokens, word_indices, MAJOR_IP_BOUNDARY
    ):
        sentence_words = [raw_tokens[i][1] for i in sentence_indices]
        sentence_tags = pos_tag(sentence_words)
        tag_by_idx.update(
            (idx, sentence_tags[position][1])
            for position, idx in enumerate(sentence_indices)
        )

    results = [None] * len(raw_tokens)
    for i, (isw, txt) in enumerate(raw_tokens):
        if not isw:
            results[i] = WordResult(txt)
            continue
        tag = tag_by_idx[i]
        later_word_idx = [w for w in word_indices if w > i]
        later_tags = []
        for w in later_word_idx:
            sep = "".join(raw_tokens[k][1] for k in range(i + 1, w) if not raw_tokens[k][0])
            if CLAUSE_BOUNDARY.search(sep):
                break
            later_tags.append(tag_by_idx[w])
        results[i] = resolve_word(txt, tag, later_tags)
        results[i].tag = tag

    # --- compound noun detection (Rule 8): adjacent NN* NN*, no punctuation between
    for a, b in zip(word_indices, word_indices[1:]):
        between = "".join(raw_tokens[k][1] for k in range(a + 1, b) if not raw_tokens[k][0])
        if (
            between.strip() == ""
            and results[a].tag in NOUN_TAGS
            and results[b].tag in NOUN_TAGS
            and raw_tokens[b][1].lower() not in TEMPORAL_ADVERBIALS
        ):
            results[b].cls = WordClass.COMPOUND_TAIL
            results[a].rule = StressRule.COMPOUND_NOUN

    # --- compound adjective detection (Rule 9): hyphenated token tagged JJ.
    # Resolve each hyphen-part on its own, then stitch back together: the
    # LAST part carries primary stress, earlier parts' own primary stress
    # is kept but demoted to secondary (matches real prosody, e.g. ,old-'fashioned).
    for i in word_indices:
        res = results[i]
        if (
            res.tag == _POS_VOCAB_TO_TAG[PartOfSpeech.ADJECTIVE]
            and "-" in res.raw
            and res.cls == WordClass.CONTENT
        ):
            parts = res.raw.split("-")
            sub_results = [
                resolve_word(p, _POS_VOCAB_TO_TAG[PartOfSpeech.ADJECTIVE], [])
                for p in parts
            ]
            res.compound_parts = sub_results
            res.cls = WordClass.COMPOUND_ADJECTIVE

            combined_syllables = []
            combined_secondary = set()
            combined_primary = -1
            for pi, sub in enumerate(sub_results):
                offset = len(combined_syllables)
                combined_syllables.extend(sub.syllables)
                is_last = (pi == len(sub_results) - 1)
                if sub.primary >= 0:
                    if is_last:
                        combined_primary = offset + sub.primary
                    else:
                        combined_secondary.add(offset + sub.primary)
                combined_secondary |= {offset + s for s in sub.secondary}
                if pi < len(sub_results) - 1:
                    combined_syllables.append("-")
            combined_secondary.discard(combined_primary)
            res.syllables = combined_syllables
            res.primary = combined_primary if combined_primary >= 0 else 0
            res.secondary = combined_secondary
            res.confidence = Confidence.COMPOUND_ADJECTIVE_RULE
            res.rule = StressRule.COMPOUND_ADJECTIVE

    # -------------------------------------------------------------------------
    # Prominence assignment: multi-level, information-structure aware
    # -------------------------------------------------------------------------
    # Tiers (most to least prominent):
    #   nuclear        - nuclear pitch accent (IP-final new/contrastive)
    #   prominent      - early ip-initial, contrastive focus, wh-correspondent
    #   pre-nuclear    - regular new information, compressed
    #   given          - de-accented (old information)
    #   suppressed     - compound tails
    #   secondary      - polysyllabic weak function words
    #   reducible      - fully reduced function words
    #
    # Information status per discourse referent:
    #   new        - first mention
    #   contrastive - explicit contrast / correction / focus particle associate
    #   given      - explicitly mentioned
    # -------------------------------------------------------------------------

    # Track mentioned discourse referents across the full text. Keys are
    # POS-aware lemmas, while focus and contrast are attached to exact token
    # occurrences so a cue cannot leak to every matching word in the document.
    seen_keys = set()
    focus_associates = set()
    contrastive_indices = set()
    cue_words = FOCUS_PARTICLES | CONTRAST_MARKERS | DISCOURSE_MARKERS

    def is_prominence_candidate(i):
        return (
            results[i].cls in (WordClass.CONTENT, WordClass.COMPOUND_ADJECTIVE)
            and raw_tokens[i][1].lower() not in cue_words
        )

    def is_trackable(i):
        return results[i].cls in (
            WordClass.CONTENT,
            WordClass.COMPOUND_ADJECTIVE,
            WordClass.COMPOUND_TAIL,
            WordClass.WEAK,
        )

    word_position = {idx: position for position, idx in enumerate(word_indices)}
    major_ips = _split_word_indices(raw_tokens, word_indices, MAJOR_IP_BOUNDARY)

    # Find focus and contrast associates within their own major phrase.
    for ip in major_ips:
        for position, idx in enumerate(ip):
            lower = raw_tokens[idx][1].lower()
            if lower in FOCUS_PARTICLES:
                search_positions = (
                    range(position - 1, -1, -1)
                    if lower == BACKWARD_FOCUS_PARTICLE
                    else range(position + 1, len(ip))
                )
                for associate_position in search_positions:
                    associate = ip[associate_position]
                    if is_prominence_candidate(associate):
                        focus_associates.add(associate)
                        break

            if lower != NEGATION_MARKER:
                continue

            but_position = next(
                (
                    later_position
                    for later_position in range(position + 1, len(ip))
                    if raw_tokens[ip[later_position]][1].lower()
                    == CONTRAST_CONJUNCTION
                ),
                None,
            )
            first_half_end = but_position if but_position is not None else len(ip)
            negated = next(
                (
                    ip[later_position]
                    for later_position in range(position + 1, first_half_end)
                    if is_prominence_candidate(ip[later_position])
                ),
                None,
            )
            if negated is not None:
                contrastive_indices.add(negated)

            if but_position is not None:
                alternative = next(
                    (
                        ip[later_position]
                        for later_position in range(but_position + 1, len(ip))
                        if is_prominence_candidate(ip[later_position])
                    ),
                    None,
                )
                if alternative is not None:
                    contrastive_indices.add(alternative)

    def split_into_intermediate_phrases(ip):
        """Split one major IP at commas and before discourse markers."""
        phrases = []
        current = []
        for idx in ip:
            lower = raw_tokens[idx][1].lower()
            if lower in DISCOURSE_MARKERS and current:
                phrases.append(current)
                current = []
            current.append(idx)
            separator = _separator_after_word(
                raw_tokens, word_indices, word_position[idx]
            )
            if INTERMEDIATE_IP_BOUNDARY.search(separator):
                phrases.append(current)
                current = []
        if current:
            phrases.append(current)
        return phrases

    pending_wh_answer = False
    wh_tags = {"WDT", "WP", "WP$", "WRB"}
    wh_words = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how"}

    for ip in major_ips:
        if not ip:
            continue
        intermediate_phrases = split_into_intermediate_phrases(ip)
        answering_wh = pending_wh_answer
        wh_answer_assigned = False

        for phrase in intermediate_phrases:
            candidates = [i for i in phrase if is_prominence_candidate(i)]

            # Cue words organize focus but should not compete with their own
            # associate for an accent.
            for i in phrase:
                if (
                    results[i].cls
                    in (WordClass.CONTENT, WordClass.COMPOUND_ADJECTIVE)
                    and i not in candidates
                ):
                    results[i].tier = ProminenceTier.GIVEN

            if not candidates:
                for i in phrase:
                    if is_trackable(i):
                        seen_keys.update(_discourse_keys(raw_tokens[i][1], results[i].tag))
                continue

            occurrence_is_given = {}
            working_seen = set(seen_keys)
            for i in phrase:
                if i in candidates:
                    keys = _discourse_keys(raw_tokens[i][1], results[i].tag)
                    occurrence_is_given[i] = not keys.isdisjoint(working_seen)
                if is_trackable(i):
                    working_seen.update(_discourse_keys(raw_tokens[i][1], results[i].tag))

            explicit_focus = [
                i for i in candidates
                if i in focus_associates or i in contrastive_indices
            ]
            if explicit_focus:
                nuclear_i = explicit_focus[-1]
                shifted_nuclear = True
            else:
                new_candidates = [i for i in candidates if not occurrence_is_given[i]]
                nuclear_i = new_candidates[-1] if new_candidates else candidates[-1]
                shifted_nuclear = False

            # In a wh-answer phrase, the first new constituent is the likely
            # correspondent; recording it as a focus shift de-accents following
            # material.
            if answering_wh and not wh_answer_assigned:
                new_candidates = [i for i in candidates if not occurrence_is_given[i]]
                if new_candidates:
                    # The answer constituent is normally the first material
                    # not already supplied by the question ("Who came? JOHN
                    # came yesterday", "What did Mary buy? ... APPLES").
                    nuclear_i = new_candidates[0]
                    shifted_nuclear = True
                wh_answer_assigned = True

            for candidate_position, i in enumerate(candidates):
                if i == nuclear_i:
                    results[i].tier = ProminenceTier.NUCLEAR
                elif i in explicit_focus:
                    results[i].tier = ProminenceTier.PROMINENT
                elif occurrence_is_given[i] or (shifted_nuclear and i > nuclear_i):
                    results[i].tier = ProminenceTier.GIVEN
                elif candidate_position <= 1:
                    results[i].tier = ProminenceTier.PROMINENT
                else:
                    results[i].tier = ProminenceTier.PRE_NUCLEAR

            seen_keys = working_seen

            # Resolve clashes inside this phrase only. Recompute the previous
            # state after a demotion so a three-accent run alternates rather
            # than cascading every later accent downward.
            previous_is_prominent = False
            for i in phrase:
                is_prominent = results[i].tier in (
                    ProminenceTier.NUCLEAR,
                    ProminenceTier.PROMINENT,
                )
                if (
                    is_prominent
                    and previous_is_prominent
                    and results[i].tier == ProminenceTier.PROMINENT
                ):
                    results[i].tier = ProminenceTier.PRE_NUCLEAR
                previous_is_prominent = results[i].tier in (
                    ProminenceTier.NUCLEAR,
                    ProminenceTier.PROMINENT,
                )

        ending_separator = _separator_after_word(
            raw_tokens, word_indices, word_position[ip[-1]]
        )
        pending_wh_answer = (
            "?" in ending_separator
            and any(
                results[i].tag in wh_tags or raw_tokens[i][1].lower() in wh_words
                for i in ip
            )
        )

    # Weak / compound-tail get fixed tiers
    for i in word_indices:
        if results[i].cls == WordClass.WEAK:
            results[i].tier = ProminenceTier.SECONDARY
        if results[i].cls == WordClass.COMPOUND_TAIL:
            results[i].tier = ProminenceTier.SUPPRESSED

    # Nuclear-only mode: retain lexical stress data only on nuclear words and
    # turn every other content-word primary into a secondary visual cue.
    if nuclear_only:
        for i in word_indices:
            res = results[i]
            if (
                res.cls in (WordClass.CONTENT, WordClass.COMPOUND_ADJECTIVE)
                and res.tier != ProminenceTier.NUCLEAR
                and res.primary >= 0
            ):
                res.secondary.add(res.primary)
                res.primary = -1

    # Rule explanation
    for i in word_indices:
        res = results[i]
        if (
            res.cls in (WordClass.CONTENT, WordClass.WEAK)
            and res.primary >= 0
            and res.rule is None
        ):
            res.rule = explain_rule(res.raw, res.tag, res.syllables, res.primary)

    return raw_tokens, results
