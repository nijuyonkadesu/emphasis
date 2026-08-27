# Meaning-Based Prominence Guide

## The Problem You Faced

The default stressmark output highlights **every lexical primary stress** (bold CAPS) on every content word. This gives a pronunciation dictionary view — accurate for *word-level* stress, but misleading for *sentence-level* prosody.

**Natural speech doesn't work that way.** Native speakers use a hierarchy:

```
Nuclear (IP focus) > Prominent (contrastive/wh/early) > Pre-nuclear (new info) > Given (de-accented) > Reduced (function words)
```

---

## The Fix: Multi-Level Prominence System

The tool implements a **text-derived AM (Autosegmental-Metrical) approximation** with information-structure heuristics:

| Tier | When | Visual | Speech Realization |
|------|------|--------|-------------------|
| **nuclear** | Rightmost new/contrastive in IP | `REVERSE CAPS` (gold, glow) | Full pitch accent, highest prominence |
| **prominent** | Non-nuclear contrast, early IP/ip-initial material | `BOLD CAPS` (yellow) | Strong pitch accent, slightly compressed |
| **pre-nuclear** | Regular new information | `BOLD YELLOW3 CAPS` | Pitch accent present but compressed |
| **given** | Repeated/old information | `dim` (gray) | No accent, reduced vowels, low/flat pitch |
| **suppressed** | Compound tails | `dim` | Fully reduced |
| **secondary** | Polysyllabic weak function words | `underline` | Lexical beat without phrase prominence |
| **reducible** | Function words (a, the, to, of...) | `dim` | Maximum reduction |

---

## Usage

```bash
# Default: full multi-tier prominence view
stressmark text.txt

# Nuclear-only practice view (all other content peaks become secondary cues)
stressmark text.txt --nuclear-only

# HTML output with full tier legend
stressmark text.txt --format html -o out.html
```

---

## What Drives Prominence (Fact-Based)

### 1. Nuclear Accent = Focus (Pierrehumbert 1980, Ladd 2008)
- **One per intonation phrase (IP)** — roughly a clause
- Default: **rightmost non-given content word** (nuclear stress rule)
- Shifts left for: contrast, correction, focus particles (*only, even, just*), *wh*-answers

### 2. Prominent Pre-Nuclear = Early/Ip-Initial / Contrastive / Wh
- **IP-initial content words** often get strong accents (topic/comment)
- **Contrastive focus**: "not RED but **BLUE**" → both get prominent
- **Focus particles**: "only **JOHN** came" → associate gets prominent
- **Wh-correspondents**: "Who came?" → "**JOHN** came" → wh-word gets prominent

### 3. Pre-Nuclear = New Information
- Regular new content words between prominent and nuclear
- Compressed pitch range, still have lexical stress

### 4. Given = De-Accented (Schwarzschild 1999)
- **Explicitly mentioned** → no pitch accent
- **Inferrable/accessible** (bridging) requires manual interpretation; the engine has no coreference model
- Givenness tracked **across full discourse**, not per sentence

### 5. Rhythm: Stress Clash Avoidance (Liberman & Prince 1977)
- English avoids adjacent strong beats
- If two prominent accents would be adjacent → second demoted to pre-nuclear
- This is **automatic** in the engine

---

## Intonation Phrasing (IP / ip)

| Boundary | Trigger | Effect |
|----------|---------|--------|
| **IP** (major) | `. ! ? ; :` | New nuclear accent; discourse givenness remains available |
| **ip** (intermediate) | `,`, discourse markers (*however, therefore*) | Pre-nuclear accents allowed, own nuclear |

The engine splits at punctuation for IPs, and at commas/discourse markers for intermediate phrases (ips).

---

## Practical Workflow for Speaking Practice

### 1. Run with Native-Like Prominence
```bash
cat your_text.txt | stressmark --nuclear-only
```

### 2. Read the Output Legend
- **REVERSE GOLD** = Nuclear — hit this HARD, highest pitch
- **BOLD YELLOW** = Prominent — strong accent, contrastive/early
- **BOLD ORANGE** = Pre-nuclear — accent but compressed
- **underline** = Secondary stress (lexical)
- **dim gray** = Given/reduced — flat, low, de-accented

### 3. Mental Corrections (The Tool Isn't Perfect)

| Situation | Tool Output | Your Adjustment |
|-----------|-------------|-----------------|
| "not X but **Y**" | Prominent X, nuclear Y | ✓ Correct |
| "**X** not Y" (correction) | Nuclear on Y | → Force prominent on X |
| "Only **JOHN** came" | Nuclear on JOHN; following material de-accented | ✓ Correct |
| Q: "Who came?" A: "**JOHN** came" | Nuclear on JOHN | ✓ Correct |
| "The **BIG** dog" (contrast) | Often early-prominent, but contrast is not inferable | Verify manually |
| Topic: "**JOHN**, he left" | Nuclear on JOHN in its comma-delimited ip | ✓ Correct |

### 4. Practice Rhythm
- **Nuclear** = full pitch movement (H* or L+H*)
- **Prominent** = strong but slightly lower range
- **Pre-nuclear** = compressed, "shoulder taps"
- **Given** = no pitch movement, just segments

---

## Architecture (For Extending)

### Core Files
- `src/stressmark/engine.py` — `analyze(text, nuclear_only=False)` returns `(raw_tokens, results)`
- `src/stressmark/render.py` — terminal/PDF/HTML/JSON renderers
- `src/stressmark/cli.py` — CLI

### Key Data Structures
```python
WordResult:
  .raw           # original token
  .syllables     # orthographic syllables
  .primary       # primary stress syllable index
  .secondary     # set of secondary stress indices
  .tier          # "nuclear" | "prominent" | "pre-nuclear" | "given" | "secondary" | "suppressed" | None
  .cls           # "content" | "weak" | "reducible" | "compound-adj" | "compound-tail"
  .confidence    # "dict" | "dict-pos-resolved" | "predicted" | "rule-9" | "reducible"
  .rule          # which of 9 stress rules (1-9)
  .tag           # NLTK POS tag
```

### Prominence Pipeline (in `analyze()`)
1. **Tokenize + POS tag** (NLTK)
2. **Per-word stress** (CMUdict → G2P fallback)
3. **Compound detection** (NN-NN, hyphenated JJ)
4. **Discourse scan** — detect:
   - Focus particles (*only, even, just, also...*) + associates
   - Contrast patterns (*not X but Y*, *X not Y*)
5. **IP segmentation** (punctuation)
6. **ip segmentation** (commas, discourse markers)
7. **Information status tracking** (surface form + WordNet lemma, across discourse)
8. **Tier assignment per ip**:
   - Nuclear = explicit contrast/focus/wh-correspondent, otherwise rightmost non-given
   - Prominent = IP/ip-initial or non-nuclear contrast
   - Pre-nuclear = other new info
   - Given = seen before
9. **Clash resolution** — demote adjacent prominents
10. **Optional demotion** (`--nuclear-only`: every non-nuclear content primary → secondary only)

---

## Known Limitations (Honest)

| Limitation | Why | Workaround |
|------------|-----|------------|
| No dependency parse | NLTK only, no spaCy model | Compound detection uses POS adjacency |
| POS tagger ~95% | Errors on "contract" (verb→NN) | Heteronym table catches common cases |
| No acoustic/prosodic model | Text-only input | Use `--explain` to see rule triggers |
| Givenness = surface/WordNet lemma match | No synonym, coreference, or bridging inference | Manual mental correction |
| Focus particles only lexicalized | No syntactic focus projection | Add to `FOCUS_PARTICLES` set |
| English only | CMUdict + G2P_en | N/A |

---

## Extending It Yourself

### Add Focus Particles
```python
# In stressmark_engine.py, extend FOCUS_PARTICLES:
FOCUS_PARTICLES = {"only", "even", "just", "also", "alone", "merely",
                    "exactly", "precisely", "specifically", "particularly",
                    "especially", "mostly", "mainly", "largely",
                    "exclusively", "solely"}
```

### Add Contrast Patterns
```python
# In the discourse scan loop, add patterns:
if lw == "rather" and prev_word == "not":
    contrastive_indices.add(next_content_index)
```

### Better IP Boundaries
Replace the punctuation/discourse-marker heuristics with a syntactic or prosodic parser.

### Cross-Sentence Givenness
Already implemented — the local `seen_keys` discourse set persists across the full text.

---

## Recommended Reading (No Hallucinations)

| Source | What It Covers |
|--------|----------------|
| Ladd, *Intonational Phonology* (2008) Ch. 3–5 | Nuclear/pre-nuclear, focus, AM theory |
| Pierrehumbert, *Phonology of English Intonation* (1980) | Original AM framework |
| Schwarzschild, *Givenness, AvoidF* (1999) | Formal de-accenting theory |
| Rooth, *Focus Interpretation* (1992) | Focus semantics → prosody |
| Calhoun, *Centrality of Metrical Structure* (2010) | Clash avoidance evidence |
| Krifka, *Basic Notions of IS* (2008) | Information structure survey |

---

## Quick Reference: Output Legend

### Terminal
```
REVERSE GOLD = nuclear (IP focus, loudest)
BOLD YELLOW  = prominent (contrast/wh/early)
BOLD ORANGE  = pre-nuclear (new info, compressed)
underline    = secondary stress (lexical)
dim gray     = given / reduced / suppressed
≈            = predicted (not in CMUdict)
⚠            = ambiguous in dictionary
```

### HTML Classes
- `.nuclear` — gold, glow, thick underline
- `.prominent` — bright yellow, solid underline
- `.pre-nuclear` — orange-yellow, solid underline, slight opacity
- `.secondary` — dotted underline
- `.reduced` — very dim
- `.unstressed` — muted

---

## TL;DR

```bash
# Your new daily driver
cat text.txt | stressmark --nuclear-only
```

**Only REVERSE GOLD words are your true speaking peaks.**  
**BOLD YELLOW = strong but not peak.**  
**BOLD ORANGE = "I'm here but not the point."**  
**dim = background noise.**

This approximates common native-speaker prominence patterns from text alone;
context, intended contrast, and delivery can still justify a different reading.
