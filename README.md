# stressmark

A sentence-aware English stress marker. Paste in an article or transcript;
it marks primary stress, secondary stress, and reduced/unstressed syllables
on every word — using a real pronunciation dictionary and part-of-speech
tagging, not hand-written suffix-rule guesses.

## Install

```
uv sync --all-extras
ln -sf "$(pwd)/.venv/bin/stressmark" ~/.local/bin/stressmark
```

First run will auto-download the NLTK data it needs (CMU Pronouncing
Dictionary, POS tagger). No spaCy model is required — see "Why NLTK and not
spaCy" below.

## Usage

```
stressmark transcript.txt
cat article.txt | stressmark
stressmark transcript.txt --format html -o out.html
stressmark transcript.txt --format json -o out.json
stressmark transcript.txt --explain          # show which of the 9 rules applied
stressmark transcript.txt --flag-heteronyms  # mark record/object/export-type words
```

Terminal output: 
- **CAPS** = primary stress, 
- underline = secondary stress,
- dim = unstressed/reduced, 
- reverse-video = nuclear (loudest word in its clause), 
- `≈` = predicted (word not in the dictionary), 
- `⚠` = ambiguous in the dictionary and not resolved.

## How it works

```
text → tokenize (NLTK, contraction-aware) → POS-tag (NLTK)
     → classify each word: reducible / weak / content
     → heteronym resolution (record/object/... by POS)
     → compound-noun (Rule 8) / compound-adjective (Rule 9) detection
     → dictionary lookup (full CMUdict) or G2P prediction for unknown words
     → nuclear-stress + given/repeat tiering, per clause
     → rule-explainer annotation (--explain)
     → render (terminal / html / json)
```

| Piece | Tool | Why |
|---|---|---|
| Dictionary | `nltk.corpus.cmudict` | Full ~125k-entry CMU dictionary, all pronunciation variants, primary **and** secondary stress |
| POS tagging | NLTK averaged perceptron tagger | Used to resolve heteronyms (record/object) and detect real auxiliary-verb usage. ~95-97% accurate on standard benchmarks |
| Orthographic syllable splitting | `pyphen` (TeX/Hunspell hyphenation patterns), with a vowel-grouping fallback for the handful of short words pyphen refuses to hyphenate (e.g. "record", "open") | Real letter-pattern based splitting instead of naive vowel-counting |
| Unknown words / proper nouns | `g2p_en` (neural G2P trained on CMUdict) | Predicts pronunciation *and* stress for anything not in the dictionary |
| Compound detection | POS-tag adjacency (NN NN) and hyphen+JJ pattern | Rules 8 and 9 |

### Why NLTK and not spaCy

The original plan was spaCy for POS tagging and dependency parsing.
spaCy's actual trained model files are only published as GitHub release
assets, which are served from a separate, untrusted-by-default subdomain
that this build environment's network policy blocks. NLTK's data (including
the POS tagger) downloads from a different, allowed host, so the build
uses NLTK throughout. This means compound-noun detection uses POS-tag
adjacency instead of a real dependency parse — works well for common
patterns like "tennis ball", less robust for syntactically complex compounds.

## Measured accuracy (not assumed)

**Heteronym resolution** (record/object/export-type words, picking the
right CMUdict pronunciation variant by POS tag): **19/20 (95%)** on a
hand-built test set covering 10 heteronyms in both noun and verb context.
The one miss was a genuine POS-tagger error ("the disease will contract
the muscle" — tagger mistakenly read "contract" as a noun there), not a
bug in the resolution logic itself. See `test_heteronyms.py`.

**G2P fallback on words actually inside CMUdict**: ~99.9% — but this number
is close to meaningless on its own, because `g2p_en`'s model was *trained
on* CMUdict. Testing it against CMUdict words is closer to a training-
accuracy check than a generalization check.

**G2P fallback on words confirmed absent from CMUdict** (brand names,
character names, invented words — the category that actually matters for
real transcripts): roughly **9/16 correct** by hand-verification (see
`benchmark_g2p.py` for the held-out-CMUdict number, and the conversation
that produced this tool for the hand-checked OOV list). Misses included
Anthropic, Kubernetes, Pikachu, Mjölnir, Hermione, and the deliberately
brutal Cholmondeley. This is exactly why every OOV prediction is flagged
with `≈` in the output rather than presented as fact — for proper nouns
specifically, treat the tool's guess as a starting point to verify, not
a final answer.

## Known limitations

- **POS tagging is ~95-97%, not 100%.** Heteronym resolution and
  auxiliary-verb detection inherit whatever error rate the tagger has.
- **G2P is weak on truly novel proper nouns**, especially ones with
  irregular, historically-derived English spelling (Cholmondeley-class
  words). No G2P system handles these well — the correct pronunciation is
  lexically arbitrary, not predictable from spelling.
- **CMUdict sometimes marks two adjacent syllables as secondary stress**
  in a row (it tracks vowel reduction, not strictly rhythmic prominence).
  This is corrected for: stress-bearing prefixes (anti-, multi-, non-...)
  keep their own fixed stress; otherwise the real secondary beat is chosen
  by rhythmic alternation (an even number of syllables from the primary).
  Verified against exploitation/EX, acceleration/CEL, absolutism/LU,
  antidepressants/AN, multinational/MUL — see `test_secondary_stress.py`.
- **CMUdict and pedagogical/textbook IPA sometimes disagree on whether a
  syllable carries secondary stress at all** — not a clash to resolve, a
  genuine difference in convention. Example: CMUdict marks the final
  syllable of "apocalypse" (-lypse) as non-reduced (level 2); many
  textbook sources mark the word with no secondary stress at all. CMUdict
  is tracking vowel reduction; your textbook is tracking rhythmic
  prominence. Both are internally consistent, they're just answering
  slightly different questions. This build follows CMUdict's actual
  encoding rather than silently overriding it, since "silently make it
  match what I assume you want" is a worse failure mode than "tell you
  plainly where the two systems diverge."
- **Only one CMUdict pronunciation variant is used** for ordinary (non-
  heteronym) words with multiple dialectal/allophonic variants — the tool
  doesn't pick a variant by regional accent.
- **Compound-noun detection (Rule 8) is a POS-tag-adjacency heuristic**,
  not a real dependency parse. It will occasionally flag two adjacent
  nouns that aren't really a semantic compound, and miss multi-word
  compounds that aren't simple NN-NN pairs.
- **Compound-adjective detection (Rule 9) only catches hyphenated tokens**
  tagged as adjectives. Open (non-hyphenated) compound adjectives aren't
  detected.
- **Orthographic syllable boundaries can land one consonant off** at
  cluster edges (e.g. "alcoholism" splits as "al-co-ho-lism" rather than
  "al-co-hol-ism") — the chosen stress *position* is still correct, just
  the displayed substring can be a letter short. Cosmetic, not positional.
- **The heteronym table is hand-curated and finite** (20 common pairs).
  `find_heteronym_candidates()` in `stressmark_engine.py` will surface
  more candidates mechanically from CMUdict if you want to extend it —
  each one still needs a human to confirm which variant is which.

## Files

- `src/stressmark/cli.py` — CLI entry point
- `src/stressmark/engine.py` — the analysis pipeline; also exposes
  `resolve_word_by_pos(word, pos)`, a single-word entry point for callers
  that already know the correct part of speech (see "Used as a library"
  below)
- `src/stressmark/render.py` — terminal / HTML / JSON renderers; also
  exposes `render_word(result)`, a single-word Rich `Text` renderer
- `test_engine.py` — exercises every rule and feature
- `test_heteronyms.py` — heteronym resolution accuracy test
- `test_secondary_stress.py` — secondary-stress clash-collapse accuracy test
- `benchmark_g2p.py` — G2P accuracy benchmark (see the caveat above about
  what this number does and doesn't tell you)
- `tests/test_integration.py` — pytest coverage for the two library entry
  points above

## Used as a library

`stressmark` is also installable as a plain Python package (`import
stressmark.engine`, `import stressmark.render`) for callers that want a
single word's stress pattern without running the whole sentence-level
pipeline. `resolve_word_by_pos(word, pos)` skips this project's own
POS-tagging/heteronym-guessing entirely — pass in a part of speech you
already know (`"noun"/"verb"/"adjective"/"adverb"`) and it resolves
directly, which is both faster and more accurate for an isolated word
than re-guessing POS from no context. `render_word(result)` renders that
single word's stress-highlighted syllables as a Rich `Text` object.

The `revdict` project (a sibling local project) uses exactly this to show
stress-marked pronunciation on dictionary lookups, entirely as an optional
plugin — `stressmark` is not a declared dependency of `revdict`, so
nothing about `revdict`'s own install is affected whether or not this
project is present.
