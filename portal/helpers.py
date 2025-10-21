# portal/helpers.py
import json, re, csv
from datetime import datetime
from io import TextIOWrapper, StringIO, BytesIO

def detect_alphabet(seq: str) -> str:
    s = re.sub(r"\s", "", seq).upper()
    if re.fullmatch(r"[ACGTURYSWKMBDHVN\-\*]+", s):
        return "RNA" if "U" in s else "DNA"
    return "protein"

def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_json_or_none(txt: str):
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None

def session_get_draft(request):
    return request.session.get("submit_draft", {})

def session_set_draft(request, data: dict):
    request.session["submit_draft"] = data
    request.session.modified = True


DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
RNA_COMP = str.maketrans("ACGUacguNn", "UGCAugcaNn")

def clean_seq(s: str) -> str:
    return re.sub(r"[\s\r\n\t]", "", s or "")

def detect_alphabet_simple(seq: str) -> str:
    s = seq.upper()
    if re.search(r"[EFILPQZ]", s):
        return "protein"
    if "U" in s and "T" not in s:
        return "RNA"
    if set(s) <= set("ACGTN"):
        return "DNA"
    if set(s) <= set("ACGUN"):
        return "RNA"
    return "DNA"

def gc_content(seq: str) -> float:
    s = clean_seq(seq).upper().replace("U", "T")
    if not s:
        return 0.0
    gc = sum(1 for c in s if c in ("G", "C"))
    return round(100.0 * gc / len(s), 2)

def reverse_complement(seq: str) -> str:
    s = clean_seq(seq)
    if "U" in s.upper() and "T" not in s.upper():
        return s.translate(RNA_COMP)[::-1]  # RNA
    return s.translate(DNA_COMP)[::-1]      # DNA

def parse_fasta(file_bytes) -> list[tuple[str, str]]:
    text = TextIOWrapper(file_bytes, encoding="utf-8", errors="ignore")
    name, seq = None, []
    out = []
    for line in text:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq)))
            name = line[1:].strip()
            seq = []
        else:
            seq.append(line)
    if name is not None:
        out.append((name, "".join(seq)))
    return out

def parse_csv_sequences(file_bytes, column: str) -> list[tuple[str, str]]:
    text = TextIOWrapper(file_bytes, encoding="utf-8", errors="ignore")
    reader = csv.DictReader(text)
    out = []
    for i, row in enumerate(reader, start=1):
        s = row.get(column)
        if s:
            out.append((row.get("id") or row.get("name") or f"row{i}", s))
    return out

def split_lines_sequences(text: str) -> list[tuple[str, str]]:
    t = (text or "").strip()
    if ">" in t:
        # try FASTA
        bio = StringIO(t).read().encode("utf-8")
        return parse_fasta(BytesIO(bio))
    out = []
    for i, line in enumerate(t.splitlines(), start=1):
        s = line.strip()
        if s:
            out.append((f"seq{i}", s))
    return out