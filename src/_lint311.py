"""Flag f-strings that only Python 3.12 accepts.

A regex cannot do this: the offending quote is exactly the character a regex
would treat as the end of the string. So scan, tracking brace depth, and treat a
quote as a terminator only at depth zero.
"""
import sys, pathlib

def scan(src):
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "fF" and i + 1 < n and src[i+1] in "\"'" and (i == 0 or not (src[i-1].isalnum() or src[i-1] == "_")):
            q = src[i+1]
            trip = src[i+1:i+4] == q*3
            delim = q*3 if trip else q
            j = i + 1 + len(delim)
            depth, inner = 0, False
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src[j] == "{":
                    if src[j:j+2] == "{{": j += 2; continue
                    depth += 1; j += 1; continue
                if src[j] == "}":
                    depth = max(0, depth - 1); j += 1; continue
                if depth > 0 and src[j] == q:
                    inner = True; j += 1; continue
                if depth == 0 and src.startswith(delim, j):
                    break
                j += 1
            if inner:
                out.append((src[:i].count("\n") + 1, q, src[i:min(j+len(delim), i+110)]))
            i = j + len(delim); continue
        if c in "\"'":                      # ordinary string, skip it
            q = c; trip = src[i:i+3] == q*3
            delim = q*3 if trip else q
            j = i + len(delim)
            while j < n:
                if src[j] == "\\": j += 2; continue
                if src.startswith(delim, j): break
                j += 1
            i = j + len(delim); continue
        if c == "#":
            i = src.find("\n", i); i = n if i < 0 else i + 1; continue
        i += 1
    return out

bad = 0
for f in sys.argv[1:]:
    p = pathlib.Path(f)
    if not p.exists(): continue
    hits = scan(p.read_text(encoding="utf-8"))
    bad += len(hits)
    print(f"  {p.name}: {len(hits)} 处" + (" ✓" if not hits else " ✗"))
    for ln, q, frag in hits:
        print(f"     第{ln}行  {frag.strip()[:96]}")
sys.exit(1 if bad else 0)
