import sys
sys.path.insert(0, '/home/claude/stressmark')
from stressmark_engine import resolve_word

# CMUdict marks vowel reduction (full vs. schwa), not strictly rhythmic
# prominence, so it sometimes marks two ADJACENT syllables as level-2 ("secondary")
# in a row. Real English only has one independent secondary beat in these cases.
# Two distinct mechanisms resolve which one is real, both verified against
# real-world pronunciation independent of CMUdict:
#
#   1. Stress-bearing prefixes (anti-, multi-, non-...) keep their OWN fixed
#      stress regardless of where the primary stress falls elsewhere in the word.
#   2. Otherwise, English rhythmic alternation puts the real secondary stress an
#      EVEN number of syllables from the primary.
#
# (word, expected_secondary_syllable_substring)
cases = [
    ("exploitation", "ex"),       # rhythmic alternation: EVEN distance from primary
    ("acceleration", "cel"),
    ("absolutism", "lu"),
    ("anticancer", "an"),         # prefix keeps its own stress
    ("antidepressants", "an"),
    ("antifungal", "an"),
    ("multinational", "mul"),
]

correct = 0
for word, expected in cases:
    r = resolve_word(word, "NN", [])
    secs = [r.syllables[i].lower().strip("-") for i in r.secondary]
    ok = expected in secs
    correct += int(ok)
    mark = "OK " if ok else "FAIL"
    print(f"{mark}  {word:18s} primary={r.syllables[r.primary]!r:10s} secondary={secs}  expected~={expected}")

print(f"\n{correct}/{len(cases)} correct ({correct/len(cases)*100:.0f}%)")
