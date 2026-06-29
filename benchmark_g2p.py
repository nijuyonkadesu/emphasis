import random
import re
from nltk.corpus import cmudict
from g2p_en import G2p

random.seed(42)
d = cmudict.dict()

# Build a clean test set: common-ish words, single-token, alphabetic only,
# first pronunciation as ground truth.
words = [w for w in d.keys() if re.match(r"^[a-z]+$", w) and len(w) >= 3]
random.shuffle(words)
sample = words[:1500]

g2p = G2p()

def stress_seq(phonemes):
    return [p[-1] for p in phonemes if p[-1].isdigit()]

def primary_idx(stress):
    return stress.index('1') if '1' in stress else -1

exact_phoneme_match = 0
exact_stress_seq_match = 0
primary_stress_match = 0
total = 0
mismatches = []

for w in sample:
    gold_prons = d[w]  # list of pronunciation variants
    pred = g2p(w)
    pred = [p for p in pred if p != ' ']  # g2p_en sometimes inserts spaces for multi-word expansion
    pred_stress = stress_seq(pred)
    if not pred_stress:
        continue
    total += 1
    pred_primary = primary_idx(pred_stress)

    # compare against best-matching gold variant (some words have multiple accepted pronunciations)
    best_primary_match = False
    best_exact = False
    best_stress_exact = False
    for gold in gold_prons:
        gold_stress = stress_seq(gold)
        if primary_idx(gold_stress) == pred_primary:
            best_primary_match = True
        if gold == pred:
            best_exact = True
        if gold_stress == pred_stress:
            best_stress_exact = True

    if best_primary_match: primary_stress_match += 1
    if best_exact: exact_phoneme_match += 1
    if best_stress_exact: exact_stress_seq_match += 1
    if not best_primary_match:
        mismatches.append((w, gold_prons, pred))

print(f"Sample size: {total}")
print(f"Primary stress position match: {primary_stress_match}/{total} = {primary_stress_match/total*100:.1f}%")
print(f"Exact stress sequence match (incl. secondary): {exact_stress_seq_match}/{total} = {exact_stress_seq_match/total*100:.1f}%")
print(f"Exact full phoneme match: {exact_phoneme_match}/{total} = {exact_phoneme_match/total*100:.1f}%")
print()
print("Sample of primary-stress mismatches (first 15):")
for w, gold, pred in mismatches[:15]:
    print(f"  {w}: gold={gold[0]} pred={pred}")
