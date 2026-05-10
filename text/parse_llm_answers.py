import re, json, csv, sys, unicodedata
from pathlib import Path

# ── Configure these paths for your environment ─────────────────────────────────
IN_DIR  = Path("CONFIGURE_ME/question_answers")   # directory with {PID}_transcription.json files
OUT_CSV = Path("CONFIGURE_ME/features/llm_question_answers.csv")
# ──────────────────────────────────────────────────────────────────────────────

FNPAT   = re.compile(r"(?P<pid>\d{3,4})_transcription\.json$", re.I)

def norm_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip().strip('"').strip()

QHDR = re.compile(r"(?m)^\s*(\d{1,2})\s*[\.\)\-:]*\s")

def split_into_question_blocks(text: str):
    text = norm_text(text)
    ms = list(QHDR.finditer(text))
    for i, m in enumerate(ms):
        try: q = int(m.group(1))
        except ValueError: continue
        start = m.start()
        end = ms[i+1].start() if i+1 < len(ms) else len(text)
        yield q, text[start:end]

def spaced(word: str) -> str:
    out = []
    for ch in word:
        if ch.isalpha():
            out.append(ch + r"\s*")
        elif ch in "- ":
            out.append(r"[\s-]*")
        elif ch in "/\\.":
            out.append(r"[\/\.]?\s*")
        else:
            out.append(r"\s*" + re.escape(ch) + r"\s*")
    return "".join(out)

YES_WORDS = [
    "yes","y","yeah","yep","affirmative","true","ja","j","stimmt","korrekt","zutreffend"
]

NO_WORDS  = [
    "no","n","nope","negative","false","nein","trifft nicht zu","keineswegs"
]
NM_WORDS  = [
    "not mentioned","not specified","not stated","not reported","not discussed",
    "no mention","no information","no info","no data","no details","none","n/?a","na","n.a.","n/a",
    "unknown","undisclosed","unclear","cannot tell","cannot be determined","no answer",
    "keine angabe","keine information","keine info","unbekannt","nicht erwähnt","nicht genannt","nicht angegeben",
    "k.a.","k angabe","k. angabe","ohne angabe"
]

EXTENT_WORDS = [
    "to some extent","some extent","partly","partial","partially","somewhat","in part","more or less",
    "teilweise","zum teil","in gewissem maße","in gewissem mass","in gewissem ausmaß","in gewissem ausmass"
]

YES_RE = re.compile(r"\b(?:" + "|".join(spaced(w) for w in YES_WORDS) + r")\b", re.I)
NO_RE  = re.compile(r"\b(?:" + "|".join(spaced(w) for w in NO_WORDS)  + r")\b", re.I)
NM_RE  = re.compile(r"\b(?:" + "|".join(spaced(w) for w in NM_WORDS)  + r")\b", re.I)
EXTENT_RE = re.compile(r"\b(?:" + "|".join(spaced(w) for w in EXTENT_WORDS) + r")\b", re.I)

DIGIT_YES = re.compile(r"^\s*1\s*[\.\)]?\s*$")
DIGIT_NO  = re.compile(r"^\s*0\s*[\.\)]?\s*$")

ANS_HDR = re.compile(
    r"(?mi)^\s*(?:\[\s*)?(?:a\s*n\s*s(?:\s*w\s*e\s*r)?|a\s*n\s*t\s*w\s*o\s*r\s*t)\s*(?:\]\s*)?[:\-]?\s*(.*)$"
)
BRACKET_TOKEN = re.compile(r"[\[\(]\s*([^\]\)\n\r]+)\s*[\]\)]")

def is_qhdr_line(t: str) -> bool: return QHDR.match(t) is not None
def is_short_token_like(t: str) -> bool:
    t = t.strip()
    return bool(t) and not is_qhdr_line(t) and len(t) <= 40 and len(t.split()) <= 4

def detect_label(s: str) -> str:
    """
    Strict detection:
      - yes / no / to some extent / not mentioned (EN/DE; case/spacing tolerant)
      - '1' -> yes, '0' -> no (only when line-like)
      - else return '' (missing)
    """
    if not s: return ""
    t = s.strip().strip("[](){}.:;,'\" ").lower()
    if YES_RE.search(t):    return "yes"
    if NO_RE.search(t):     return "no"
    if EXTENT_RE.search(t): return "to some extent"
    if NM_RE.search(t):     return "not mentioned"
    if DIGIT_YES.match(t):  return "yes"
    if DIGIT_NO.match(t):   return "no"
    return ""

def nonempty_lines_after(lines, i0):
    for j in range(i0+1, len(lines)):
        t = lines[j].strip()
        if t:
            yield t, j

def extract_from_block(block: str) -> str:
    if not block or not block.strip(): return ""
    lines = block.splitlines()

    # Explicit Answer/Antwort header
    for i, line in enumerate(lines):
        m = ANS_HDR.match(line)
        if m:
            tail = m.group(1).strip()
            c = detect_label(tail)
            if c: return c
            for t, _ in nonempty_lines_after(lines, i):
                c = detect_label(t)
                if c: return c
            rest = "\n".join(lines[i+1:]) if i+1 < len(lines) else ""
            c = detect_label(rest)
            if c: return c
            break

    # Headerless: scan first few non-empty lines after the first line
    scanned = 0
    for t, _ in nonempty_lines_after(lines, 0):
        if is_short_token_like(t):
            c = detect_label(t)
            if c: return c
        c = detect_label(t)
        if c: return c
        scanned += 1
        if scanned >= 16: break

    # Bracketed token anywhere
    b = BRACKET_TOKEN.search(block)
    if b:
        c = detect_label(b.group(1))
        if c: return c

    # Weighted global search (earliest wins; yes/no/extent before NM)
    hits = []
    for pat, lab, pri in [
        (YES_RE,"yes",0),
        (NO_RE,"no",1),
        (EXTENT_RE,"to some extent",2),
        (NM_RE,"not mentioned",3)
    ]:
        for m in pat.finditer(block):
            hits.append((m.start(), pri, lab))
    if hits:
        hits.sort(key=lambda x: (x[0], x[1]))
        return hits[0][2]

    return ""

def parse_qa(text: str) -> dict[int, str]:
    out = {}
    for qnum, block in split_into_question_blocks(text):
        out[qnum] = extract_from_block(block)
    return out

def main():
    files = [p for p in IN_DIR.iterdir() if p.is_file() and FNPAT.search(p.name)]
    files.sort(key=lambda p: int(FNPAT.search(p.name).group("pid")))
    total, processed, errors = len(files), 0, 0
    rows = []
    missing_report = []

    for p in files:
        pid = FNPAT.search(p.name).group("pid")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            qa_map = parse_qa(data.get("question_answers", ""))
            row = {"patient_id": pid}
            missing = []
            for q in range(1, 12):
                val = qa_map.get(q, "")
                row[f"q{q}"] = val
                if val == "":
                    missing.append(f"q{q}")
            if missing:
                missing_report.append((pid, missing))
            rows.append(row)
            processed += 1
        except Exception:
            errors += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id"] + [f"q{i}" for i in range(1, 12)])
        w.writeheader()
        w.writerows(rows)

    print(f"Total JSON files matched: {total}")
    print(f"Processed: {processed}")
    print(f"Unprocessed/Errors: {errors}")
    print(f"Wrote: {OUT_CSV}")

    if missing_report:
        print("\nRows with at least one missing answer:")
        for pid, miss in missing_report:
            print(f"- {pid}: {', '.join(miss)}")
    else:
        print("\nNo missing answers detected.")

if __name__ == "__main__":
    sys.exit(main())
