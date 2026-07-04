# Meaning-Based Prominence Guide

## The Problem You Faced

The default stressmark output highlights **every lexical primary stress** (bold CAPS) on every content word. This gives a pronunciation dictionary view — accurate for *word-level* stress, but misleading for *sentence-level* prosody.

**Natural speech doesn't work that way.** Native speakers use a hierarchy:

```
Nuclear (IP focus) > Prominent (contrastive/wh/early) > Pre-nuclear (new info) > Given (de-accented) > Reduced (function words)
```

---

## The Fix: Multi-Level Prominence System

The tool now implements a **proper AM (Autosegmental-Metrical) model** with information structure:

| Tier | When | Visual | Speech Realization |
|------|------|--------|-------------------|
| **nuclear** | Rightmost new/contrastive in IP | `REVERSE CAPS` (gold, glow) | Full pitch accent, highest prominence |
| **prominent** | Contrastive focus, wh-correspondent, focus particle associate, early IP-initial | `BOLD CAPS` (yellow) | Strong pitch accent, slightly compressed |
| **pre-nuclear** | Regular new information | `BOLD YELLOW3 CAPS` | Pitch accent present but compressed |
| **given** | Repeated/old information | `dim` (gray) | No accent, reduced vowels, low/flat pitch |
| **suppressed** | Compound tails, weak function words | `dim` | Fully reduced |
| **reducible** | Function words (a, the, to, of...) | `dim` | Maximum reduction |

---

## Usage

```bash
# Default: full lexical stress (all primary = bold CAPS)
python3 stressmark.py text.txt

# Native-like prominence (recommended for speaking practice)
python3 stressmark.py text.txt --nuclear-only

# HTML output with full tier legend
python3 stressmark.py text.txt --format html -o out.html
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
- **Inferrable/accessible** (bridging) → may stay pre-nuclear
- Givenness tracked **across full discourse**, not per sentence

### 5. Rhythm: Stress Clash Avoidance (Liberman & Prince 1977)
- English avoids adjacent strong beats
- If two prominent accents would be adjacent → second demoted to pre-nuclear
- This is **automatic** in the engine

---

## Intonation Phrasing (IP / ip)

| Boundary | Trigger | Effect |
|----------|---------|--------|
| **IP** (major) | `. ! ? ; :` | New nuclear accent, givenness reset possible |
| **ip** (intermediate) | `,` heavy NP, discourse markers (*however, therefore*) | Pre-nuclear accents allowed, own nuclear |

The engine splits at punctuation for IPs, and at commas/discourse markers for intermediate phrases (ips).

---

## Practical Workflow for Speaking Practice

### 1. Run with Native-Like Prominence
```bash
cat your_text.txt | python3 stressmark.py --nuclear-only
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
| "not X but **Y**" | Nuclear on Y | ✓ Correct |
| "**X** not Y" (correction) | Nuclear on Y | → Force prominent on X |
| "Only **JOHN** came" | Prominent on JOHN | ✓ Correct |
| Q: "Who came?" A: "**JOHN** came" | Prominent on JOHN | ✓ Correct |
| "The **BIG** dog" (contrast) | Pre-nuclear on BIG | → Force prominent on BIG |
| Topic: "**JOHN**, he left" | Prominent on JOHN | ✓ Correct (IP-initial) |

### 4. Practice Rhythm
- **Nuclear** = full pitch movement (H* or L+H*)
- **Prominent** = strong but slightly lower range
- **Pre-nuclear** = compressed, "shoulder taps"
- **Given** = no pitch movement, just segments

---

## Architecture (For Extending)

### Core Files
- `stressmark_engine.py` — `analyze(text, nuclear_only=False)` returns `(raw_tokens, results)`
- `stressmark_render.py` — `render_terminal/html/json`
- `stressmark.py` — CLI

### Key Data Structures
```python
WordResult:
  .raw           # original token
  .syllables     # orthographic syllables
  .primary       # primary stress syllable index
  .secondary     # set of secondary stress indices
  .tier          # "nuclear" | "prominent" | "pre-nuclear" | "given" | "suppressed" | None
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
7. **Information status tracking** (per lemma, across discourse)
8. **Tier assignment per ip**:
   - Nuclear = rightmost non-given / contrastive / focus-associate
   - Prominent = IP-initial, contrastive, focus-associate, wh-correspondent
   - Pre-nuclear = other new info
   - Given = seen before
9. **Clash resolution** — demote adjacent prominents
10. **Optional demotion** (`--nuclear-only`: pre-nuclear → secondary only)

---

## Known Limitations (Honest)

| Limitation | Why | Workaround |
|------------|-----|------------|
| No dependency parse | NLTK only, no spaCy model | Compound detection uses POS adjacency |
| POS tagger ~95% | Errors on "contract" (verb→NN) | Heteronym table catches common cases |
| No acoustic/prosodic model | Text-only input | Use `--explain` to see rule triggers |
| Givenness = exact lemma match | No synonym/bridging inference | Manual mental correction |
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
                    "particularly", "exclusively", "solely"}
```

### Add Contrast Patterns
```python
# In the discourse scan loop, add patterns:
if lw == "rather" and prev_word == "not":
    contrastive_lemmas.add(next_content_word)
```

### Better IP Boundaries
Replace `IP_BOUNDARY` regex with spaCy dependency parse (needs model download).

### Cross-Sentence Givenness
Already implemented — `info_status` dict persists across full text.

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
cat text.txt | python3 stressmark.py --nuclear-only
```

**Only REVERSE GOLD words are your true speaking peaks.**  
**BOLD YELLOW = strong but not peak.**  
**BOLD ORANGE = "I'm here but not the point."**  
**dim = background noise.**

This matches how native speakers actually distribute prominence.