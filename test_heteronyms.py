import sys
sys.path.insert(0, '/home/claude/stressmark')
from stressmark_engine import analyze

# (sentence, target_word, expected_primary_syllable_index)
cases = [
    ("I want to record this song.", "record", 1),       # verb: re-CORD
    ("Put on the record, please.", "record", 0),         # noun: RE-cord
    ("He will object to this plan.", "object", 1),        # verb: ob-JECT
    ("I found a strange object.", "object", 0),            # noun: OB-ject
    ("We need to export this data.", "export", 1),         # verb: ex-PORT
    ("Check the export settings.", "export", 0),           # noun: EX-port
    ("They will import the goods.", "import", 1),           # verb: im-PORT
    ("This is an important import.", "import", 0),          # noun: IM-port
    ("Please sign the contract now.", "contract", 0),       # noun: CON-tract
    ("The disease will contract the muscle.", "contract", 1), # verb: con-TRACT
    ("Let me present the award.", "present", 1),            # verb: pre-SENT
    ("Here is your present.", "present", 0),                # noun: PRE-sent
    ("Factories produce more goods.", "produce", 1),        # verb: pro-DUCE
    ("Buy fresh produce at the market.", "produce", 0),      # noun: PRO-duce
    ("I will project the image.", "project", 1),             # verb: pro-JECT
    ("This is a school project.", "project", 0),             # noun: PRO-ject
    ("Prices will increase next year.", "increase", 1),       # verb: in-CREASE
    ("There was a sharp increase.", "increase", 0),            # noun: IN-crease
    ("Please address the issue.", "address", 1),               # verb: a-DDRESS
    ("What is your home address?", "address", 0),               # noun: AD-dress
]

correct = 0
for sent, target, expected in cases:
    raw_tokens, results = analyze(sent)
    found = None
    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if isw and txt.lower() == target:
            found = res
            break
    ok = found is not None and found.primary == expected
    correct += int(ok)
    mark = "OK " if ok else "FAIL"
    print(f"{mark}  {sent:45s} {target:10s} expected={expected} got={found.primary if found else None}  (tag={found.tag if found else None})")

print(f"\n{correct}/{len(cases)} correct ({correct/len(cases)*100:.0f}%)")
