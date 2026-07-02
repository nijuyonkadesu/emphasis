"""
stressmark.engine
Core pipeline: text -> per-word stress analysis (primary/secondary/unstressed),
sentence-level nuclear-stress tiering, and confidence tagging.
"""
import re
import pyphen
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import cmudict
from g2p_en import G2p

def _ensure_nltk_data():
    needed = [
        ("corpora/cmudict", "cmudict"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()

_dic = pyphen.Pyphen(lang='en_US')
_cmu = cmudict.dict()
_g2p = G2p()

# ---------------------------------------------------------------------------
# Heteronyms: words whose stress pattern depends on part of speech.
# Auto-detected from CMUdict (any word with >1 stress-distinct pronunciation),
# then hand-resolved: which variant is the noun reading, which is the verb.
# ---------------------------------------------------------------------------

def _stress_seq(pron):
    return tuple(p[-1] for p in pron if p[-1].isdigit())

def find_heteronym_candidates():
    """Scan CMUdict for words with multiple pronunciations that differ in
    stress pattern (not just phoneme quality). Returns {word: [stress_seqs]}"""
    out = {}
    for w, prons in _cmu.items():
        if not re.match(r"^[a-z]+$", w):
            continue
        seqs = {_stress_seq(p) for p in prons}
        if len(seqs) > 1:
            out[w] = prons
    return out

# Hand-resolved: word -> {"noun": pron_index, "verb": pron_index} into _cmu[word]
# Verified against published English heteronym lists. Only common, unambiguous
# cases are listed; anything else found by find_heteronym_candidates() but not
# listed here gets surfaced with a warning instead of guessed.
HETERONYMS = {
    "record":   {"noun": ['R','EH1','K','ER0','D'],         "verb": ['R','IH0','K','AO1','R','D']},
    "object":   {"noun": ['AA1','B','JH','EH0','K','T'],     "verb": ['AH0','B','JH','EH1','K','T']},
    "export":   {"noun": ['EH1','K','S','P','AO0','R','T'],  "verb": ['IH0','K','S','P','AO1','R','T']},
    "import":   {"noun": ['IH1','M','P','AO0','R','T'],      "verb": ['IH0','M','P','AO1','R','T']},
    "contract": {"noun": ['K','AA1','N','T','R','AE0','K','T'], "verb": ['K','AH0','N','T','R','AE1','K','T']},
    "present":  {"noun": ['P','R','EH1','Z','AH0','N','T'],  "verb": ['P','R','IH0','Z','EH1','N','T']},
    "produce":  {"noun": ['P','R','OW1','D','UW2','S'],      "verb": ['P','R','AH0','D','UW1','S']},
    "project":  {"noun": ['P','R','AA1','JH','EH0','K','T'], "verb": ['P','R','AH0','JH','EH1','K','T']},
    "progress": {"noun": ['P','R','AA1','G','R','EH2','S'],  "verb": ['P','R','AH0','G','R','EH1','S']},
    "conduct":  {"noun": ['K','AA1','N','D','AH0','K','T'],  "verb": ['K','AH0','N','D','AH1','K','T']},
    "increase": {"noun": ['IH1','N','K','R','IY2','S'],      "verb": ['IH0','N','K','R','IY1','S']},
    "decrease": {"noun": ['D','IY1','K','R','IY2','S'],      "verb": ['D','IH0','K','R','IY1','S']},
    "permit":   {"noun": ['P','ER1','M','IH0','T'],          "verb": ['P','ER0','M','IH1','T']},
    "address":  {"noun": ['AE1','D','R','EH2','S'],          "verb": ['AH0','D','R','EH1','S']},
    "contrast": {"noun": ['K','AA1','N','T','R','AE2','S','T'], "verb": ['K','AH0','N','T','R','AE1','S','T']},
    "subject":  {"noun": ['S','AH1','B','JH','IH0','K','T'], "verb": ['S','AH0','B','JH','EH1','K','T']},
    "extract":  {"noun": ['EH1','K','S','T','R','AE0','K','T'], "verb": ['IH0','K','S','T','R','AE1','K','T']},
    "insult":   {"noun": ['IH1','N','S','AH0','L','T'],      "verb": ['IH0','N','S','AH1','L','T']},
    "perfect":  {"noun": ['P','ER1','F','IH0','K','T'],      "verb": ['P','ER0','F','EH1','K','T']},
    "suspect":  {"noun": ['S','AH1','S','P','EH0','K','T'],  "verb": ['S','AH0','S','P','EH1','K','T']},
}

NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

# ---------------------------------------------------------------------------
# Word classification: reducible (fully suppressible function word) /
# weak (function-class but keeps its own lexical stress) / content
# ---------------------------------------------------------------------------

CLOSED_CLASS_TAGS = {"DT", "IN", "CC", "TO", "PRP$", "MD", "EX"}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DO_FORMS = {"do", "does", "did"}
HAVE_FORMS = {"have", "has", "had"}

SCHWA_MAP = {
    "a": "ə", "an": "ən", "the": "thə", "to": "tə", "of": "əv", "and": "ən",
    "for": "fər", "your": "yər", "are": "ər", "was": "wəz", "were": "wər",
    "can": "kən", "could": "kəd", "would": "wəd", "should": "shəd", "but": "bət",
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
        return "reducible"
    if lower in BE_FORMS:
        return "reducible"
    if lower in DO_FORMS:
        return "reducible" if "VB" in sent_tags_after[:4] else "content"
    if lower in HAVE_FORMS:
        return "reducible" if "VBN" in sent_tags_after[:4] else "content"
    if tag in CLOSED_CLASS_TAGS:
        return "reducible" if syllable_count(word) <= 1 else "weak"
    return "content"

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

RULES = {
    1: "2-syllable noun -> stress 1st syllable",
    2: "2-syllable adjective -> stress 1st syllable",
    3: "2-syllable verb -> stress 2nd syllable",
    4: "-ic ending -> stress syllable before 'ic'",
    5: "-sion/-tion ending -> stress syllable before the ending",
    6: "-cy/-ty/-phy/-gy ending -> stress antepenultimate syllable",
    7: "-al ending -> stress antepenultimate syllable",
    8: "compound noun -> stress 1st part",
    9: "compound adjective -> stress 2nd part",
}

def explain_rule(word, tag, ortho_sylls, primary_idx):
    lower = word.lower()
    n = len(ortho_sylls)
    if n == 2:
        if tag in ("NN", "NNS", "NNP", "NNPS") and primary_idx == 0:
            return 1
        if tag in ("JJ",) and primary_idx == 0:
            return 2
        if tag in VERB_TAGS and primary_idx == 1:
            return 3
    if lower.endswith("ic") and primary_idx == n - 2:
        return 4
    if (lower.endswith("sion") or lower.endswith("tion")) and primary_idx == n - 2:
        return 5
    if any(lower.endswith(s) for s in ("cy", "ty", "phy", "gy")) and primary_idx == max(0, n - 3):
        return 6
    if lower.endswith("al") and primary_idx == max(0, n - 3):
        return 7
    return None

# ---------------------------------------------------------------------------
# Main per-word resolver
# ---------------------------------------------------------------------------

class WordResult:
    def __init__(self, raw):
        self.raw = raw
        self.is_word = False
        self.syllables = []
        self.primary = -1
        self.secondary = set()
        self.confidence = None
        self.tier = None
        self.rule = None
        self.cls = None
        self.tag = None
        self.compound_parts = None

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
    r.cls = classify(raw, tag, sent_tags_after, lower)

    if r.cls == "reducible":
        r.syllables = syllabify(raw)
        r.confidence = "reducible"
        return r

    # --- heteronym check (resolved by POS) ---
    if lower in HETERONYMS:
        want_verb = tag in VERB_TAGS
        variant = HETERONYMS[lower]["verb"] if want_verb else HETERONYMS[lower]["noun"]
        phonetic_n = sum(1 for p in variant if p[-1].isdigit())
        ortho = syllabify(raw, min_syllables=phonetic_n)
        r.syllables = ortho
        primary, secondary = stress_positions_for_pron(variant, len(ortho), lower)
        r.primary, r.secondary, r.confidence = primary, secondary, "dict-pos-resolved"
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
        r.confidence = "dict-flagged" if (len(ortho) > 1 and len(primaries) > 1) else "dict"
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

    ortho = syllabify(raw, min_syllables=max(phonetic_n, 1))
    r.syllables = ortho

    if phonetic_n <= 1 or len(ortho) == 1:
        r.primary, r.confidence = 0, ("dict" if phonetic_n <= 1 else "predicted")
        return r

    primary, secondary = stress_positions_for_pron(phones, len(ortho), lower)
    r.primary, r.secondary, r.confidence = primary, secondary, "predicted"
    return r

# ---------------------------------------------------------------------------
# Sentence-level pass: tokenize, tag, classify, resolve, then apply
# compound rules (8/9), nuclear-stress rule, and given/repeat backgrounding.
# ---------------------------------------------------------------------------

CLAUSE_BOUNDARY = re.compile(r"[.!?;:,]")
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

def analyze(text):
    raw_tokens = tokenize_with_spans(text)
    word_indices = [i for i, (isw, _) in enumerate(raw_tokens) if isw]
    words_only = [raw_tokens[i][1] for i in word_indices]
    tags = pos_tag(words_only) if words_only else []
    tag_by_idx = {idx: tags[j][1] for j, idx in enumerate(word_indices)}

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
        if between.strip() == "" and results[a].tag in NOUN_TAGS and results[b].tag in NOUN_TAGS:
            results[b].cls = "compound-tail"

    # --- compound adjective detection (Rule 9): hyphenated token tagged JJ.
    # Resolve each hyphen-part on its own, then stitch back together: the
    # LAST part carries primary stress, earlier parts' own primary stress
    # is kept but demoted to secondary (matches real prosody, e.g. ,old-'fashioned).
    for i in word_indices:
        res = results[i]
        if res.tag == "JJ" and "-" in res.raw and res.cls == "content":
            parts = res.raw.split("-")
            sub_results = [resolve_word(p, "JJ", []) for p in parts]
            res.compound_parts = sub_results
            res.cls = "compound-adj"

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
            res.confidence = "rule-9"
            res.rule = 9

    # --- nuclear stress + given/repeat pass, scoped per clause ---
    seen = set()
    clause_word_idxs = []

    def finalize(clause_idxs):
        contentish = [i for i in clause_idxs if results[i].cls in ("content", "compound-adj")]
        if not contentish:
            return
        nuclear_i = contentish[-1]
        for i in reversed(contentish):
            lw = raw_tokens[i][1].lower()
            if lw in HETERONYMS or lw not in seen:
                nuclear_i = i
                break
        for i in contentish:
            lw = raw_tokens[i][1].lower()
            if i == nuclear_i:
                results[i].tier = "nuclear"
            elif lw not in HETERONYMS and lw in seen:
                results[i].tier = "given"
            else:
                results[i].tier = "secondary"
        for i in contentish:
            lw = raw_tokens[i][1].lower()
            if lw not in HETERONYMS:
                seen.add(lw)

    for i in word_indices:
        clause_word_idxs.append(i)
        sep_after = ""
        idx_in_list = word_indices.index(i)
        if idx_in_list + 1 < len(word_indices):
            nxt = word_indices[idx_in_list + 1]
            sep_after = "".join(raw_tokens[k][1] for k in range(i + 1, nxt) if not raw_tokens[k][0])
        else:
            sep_after = "".join(raw_tokens[k][1] for k in range(i + 1, len(raw_tokens)) if not raw_tokens[k][0])
        if CLAUSE_BOUNDARY.search(sep_after):
            finalize(clause_word_idxs)
            clause_word_idxs = []
    finalize(clause_word_idxs)

    # weak / compound-tail get a fixed tier
    for i in word_indices:
        if results[i].cls in ("weak",):
            results[i].tier = "secondary"
        if results[i].cls == "compound-tail":
            results[i].tier = "suppressed"

    # rule explanation
    for i in word_indices:
        res = results[i]
        if res.cls in ("content", "weak") and res.primary >= 0:
            res.rule = explain_rule(res.raw, res.tag, res.syllables, res.primary)

    return raw_tokens, results
