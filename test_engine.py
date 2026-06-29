import sys
sys.path.insert(0, '/home/claude/stressmark')
from stressmark_engine import analyze

def show(text):
    raw_tokens, results = analyze(text)
    out = []
    for tok, res in zip(raw_tokens, results):
        isw, txt = tok
        if not isw:
            out.append(txt)
            continue
        if res.cls == "reducible":
            out.append(txt.lower())
            continue
        if res.tier == "given" or res.tier == "suppressed":
            out.append(txt.lower())
            continue
        sylls = res.syllables
        pieces = []
        for i, s in enumerate(sylls):
            if i == res.primary:
                tag = "NUC" if res.tier == "nuclear" else "PRI"
                pieces.append(f"[{s.upper()}:{tag}]")
            elif i in res.secondary:
                pieces.append(f"[{s}:SEC]")
            else:
                pieces.append(s.lower())
        conf = f"<{res.confidence}>" if res.confidence not in ("dict",) else ""
        rule = f"(R{res.rule})" if res.rule else ""
        out.append("".join(pieces) + conf + rule)
    print("IN :", text)
    print("OUT:", "".join(out))
    print()

tests = [
    "How did I end up here?",
    "Before we do a sweet intro, I want to do an even sweeter intro.",
    "I caught you prompting right before this big intro, and I think the prompting was the problem.",
    "I want to record this conversation, then play back the record.",
    "He will object to this plan because of that strange object.",
    "The organization needs better education and conversation.",
    "It was a well-known, old-fashioned bookcase near the tennis ball.",
    "Music, heroic, graphic, magnetic, electric, and electronic devices.",
    "Democracy, visibility, photography, and psychology are antepenultimate.",
    "Physical, critical, magical, and hysterical reactions.",
    "Picture, mirror, bottle, and cupboard versus provide, believe, decide, and begin.",
]

for t in tests:
    show(t)
