# views.py
import json, os, csv
import numpy as np
from io import StringIO, BytesIO
from itertools import product
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.http import require_GET
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from .forms import FeatureExtractorForm
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from django.utils.safestring import mark_safe
from sklearn.neighbors import NearestNeighbors
from .helpers import (
    gc_content, clean_seq, detect_alphabet_simple,
    parse_fasta, parse_csv_sequences, split_lines_sequences
)

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_DIR, "data", "database.json")
DATABASE_CSV_FILE = os.path.join(APP_DIR, "data", "database.csv")
REF_TSNE_FILE = os.path.join(APP_DIR, "data", "embeddings", "refs_tsne.npy")
REF_EMB_FILE  = os.path.join(APP_DIR, "data", "embeddings", "refs_embeddings.npy")
REF_META_FILE = os.path.join(APP_DIR, "data", "embeddings", "refs_meta.json")
_RNAFM_SINGLETON = {"model": None, "alphabet": None, "emb_len": 640, "device": "cpu"}

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def _lexicokeys(alpha_letters: str, k: int):
    return [''.join(p) for p in product(sorted(alpha_letters), repeat=k)]

def _alpha_letters(alpha: str) -> str:
    return "ACGT" if alpha == "DNA" else "ACGU"

def home(request):
    return render(request, "home.html")

def _assign_colors(groups: list[str]) -> dict[str, str]:
    seen = []
    for g in groups:
        if g not in seen:
            seen.append(g)
    return {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(seen)}

def help(request):
    # submit or search topics
    faqs = [
        {
            "title": "Feature Extractor",
            "body": "Use the Feature Extractor to compute or load features (k-mers, GC%, skews, MNC) and visualize them.",
            "tags": ["features", "kmer", "umap"],
        },
        {
            "title": "Feature Explorer",
            "body": "Upload sequences (Text/FASTA/CSV) and compare to reference siRNA/miRNA/piRNA embeddings with t-SNE.",
            "tags": ["explorer", "tsne", "rn a-fm"],
        },
    ]
    q = (request.GET.get("q") or "").strip().lower()
    if q:
        faqs = [
            f for f in faqs
            if q in f["title"].lower()
            or q in f["body"].lower()
            or any(q in t for t in f.get("tags", []))
        ]
    return render(request, "help.html", {"faqs": faqs, "q": q})

def help_getting_started(request):
    return render(request, "help_getting_started.html")

def about(request):
    return render(request, "about.html")

def _round_feature_row(row: dict, nd=4) -> dict:
    out = {}
    for k, v in row.items():
        if k in ("name", "sequence", "alphabet", "length"):
            out[k] = v
            continue
        try:
            if v is None:
                out[k] = v
            else:
                fv = float(v)
                out[k] = round(fv, nd)
        except (TypeError, ValueError):
            out[k] = v
    return out


def effective_length(seq: dict) -> int:
    s = seq.get("sequence") or {}
    L = s.get("length")
    if isinstance(L, int) and L > 0:
        return L
    strands = s.get("strands") or {}
    def _strand_len(strand: dict) -> int:
        if not isinstance(strand, dict):
            return 0
        L = strand.get("length")
        if isinstance(L, int) and L > 0:
            return L
        r = (strand.get("clean") or strand.get("raw") or "")
        if r:
            return len("".join(str(r).split()))
        return 0
    ls = _strand_len(strands.get("sense") or {})
    la = _strand_len(strands.get("antisense") or {})
    if ls or la:
        return ls + la
    raw = (s.get("raw") or "")
    if raw:
        return len("".join(str(raw).split()))
    return 0

# CSV database reader for Feature Extractor 
def _iter_database_csv_rows(selected_types: set[str]) -> list[tuple[str, str, str, dict]]:

    if not os.path.exists(DATABASE_CSV_FILE):
        return []

    items = []
    with open(DATABASE_CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rna_type = (row.get("RNA Type") or "").strip()
            if selected_types and rna_type and rna_type not in selected_types:
                continue

            name = row.get("ID") or row.get("id") or "seq"
            seq = (row.get("Sequence") or row.get("sequence") or "").strip()
            if not seq:
                continue

            # Building pre-computed features from CSV columns
            def _float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            feats = {
                "length": int(float(row.get("length", 0) or 0)) if (row.get("length") not in (None, "")) else None,
                "gc_content": _float(row.get("gc_content")),
                "gc_skew": _float(row.get("gc_skew")),
                "au_or_at_skew": _float(row.get("au_at_skew")),
                "mnc": {},
                "kmer2": {},
                "kmer3": {},
            }

            # MNC (A/C/G/U)
            for base in ("A", "C", "G", "U"):
                key = f"mnc_{base}"
                if key in row:
                    val = _float(row.get(key))
                    if val is not None:
                        feats["mnc"][base] = val

            # k=2
            for a in "ACGU":
                for b in "ACGU":
                    col = f"kmer_2_{a}{b}"
                    if col in row:
                        val = _float(row.get(col))
                        if val is not None:
                            feats["kmer2"][f"{a}{b}"] = val

            # k=3
            for a in "ACGU":
                for b in "ACGU":
                    for c in "ACGU":
                        col = f"kmer_3_{a}{b}{c}"
                        if col in row:
                            val = _float(row.get(col))
                            if val is not None:
                                feats["kmer3"][f"{a}{b}{c}"] = val

            alpha = "RNA"
            items.append((name, seq, alpha, feats))

    return items

def _parse_text_sequences_input(raw_text: str):
    raw_text = raw_text or ""
    has_fasta = any(line.lstrip().startswith(">") for line in raw_text.splitlines())
    if has_fasta:
        buf = BytesIO(raw_text.encode("utf-8"))
        return list(parse_fasta(buf))  # concatenating multiline FASTA entries
    else:
        return list(split_lines_sequences(raw_text))
    
def _save_refs_tsne(X_ref: np.ndarray, Y_ref: np.ndarray, counts: dict):
    os.makedirs(os.path.dirname(REF_TSNE_FILE), exist_ok=True)
    np.save(REF_EMB_FILE, X_ref.astype(np.float32))
    np.save(REF_TSNE_FILE, Y_ref.astype(np.float32))
    with open(REF_META_FILE, "w") as f:
        json.dump(counts, f)

def _load_refs_tsne():
    try:
        if not (os.path.exists(REF_EMB_FILE) and os.path.exists(REF_TSNE_FILE) and os.path.exists(REF_META_FILE)):
            return None, None, None
        X_ref = np.load(REF_EMB_FILE).astype(np.float32)
        Y_ref = np.load(REF_TSNE_FILE).astype(np.float32)
        with open(REF_META_FILE, "r") as f:
            counts = json.load(f)
        # sanity check
        if X_ref.shape[0] != Y_ref.shape[0]:
            return None, None, None
        total = sum(int(v) for v in counts.values())
        if total != X_ref.shape[0]:
            return None, None, None
        return X_ref, Y_ref, counts
    except Exception:
        return None, None, None

def _build_or_load_joint_refs(si_arr, mi_arr, pi_arr):
    # Check first into cache
    X_ref_cached, Y_ref_cached, counts_cached = _load_refs_tsne()
    if X_ref_cached is not None:
        return X_ref_cached, Y_ref_cached, counts_cached

    # else compute once and cache it
    n_si, n_mi, n_pi = si_arr.shape[0], mi_arr.shape[0], pi_arr.shape[0]
    parts = [p for p in (si_arr, mi_arr, pi_arr) if p.size]
    if not parts:
        return np.zeros((0, _RNAFM_SINGLETON["emb_len"]), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {"siRNA":0,"miRNA":0,"piRNA":0}

    X_ref = np.vstack(parts).astype(np.float32)
    xs, ys = _tsne_2d(X_ref.tolist())
    Y_ref = np.column_stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)])
    counts = {"siRNA": n_si, "miRNA": n_mi, "piRNA": n_pi}
    _save_refs_tsne(X_ref, Y_ref, counts)
    return X_ref, Y_ref, counts

def _knn_project(X_ref: np.ndarray, Y_ref: np.ndarray, X_new: np.ndarray, k: int = 50, eps: float = 1e-6):

    if X_new.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    k = min(k, max(1, X_ref.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(X_ref)
    dists, idxs = nn.kneighbors(X_new)

    # inverse-distance weights
    w = 1.0 / (dists + eps)
    w = w / w.sum(axis=1, keepdims=True)

    # barycentric mapping in the 2D space
    Y_neighbors = Y_ref[idxs]  # (N_new, k, 2)
    Y_new = (w[..., None] * Y_neighbors).sum(axis=1)
    return Y_new.astype(np.float32)

@csrf_protect
@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def feature_extractor(request):

    if request.method == "GET":
        form = FeatureExtractorForm()
        return render(request, "tools/feature_extractor_form.html", {"form": form})

    form = FeatureExtractorForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "tools/feature_extractor_form.html", {"form": form})

    cd = form.cleaned_data
    mode = cd["input_mode"]
    selected_db_types = set(cd.get("db_types") or [])

    items = []

    if mode == "database":
        items = _iter_database_csv_rows(selected_db_types)

    elif mode == "text":
        for name, seq in _parse_text_sequences_input(cd["sequences_text"]):
            alpha = detect_alphabet_simple(seq)
            items.append((name, seq, alpha, None))

    elif mode == "fasta":
        for name, seq in parse_fasta(request.FILES["file"].file):
            alpha = detect_alphabet_simple(seq)
            items.append((name, seq, alpha, None))

    else:
        for name, seq in parse_csv_sequences(request.FILES["file"].file, "sequence"):
            alpha = detect_alphabet_simple(seq)
            items.append((name, seq, alpha, None))

    if not items:
        messages.error(request, "No sequences found in the provided input.")
        return render(request, "tools/feature_extractor_form.html", {"form": form})

    rows = []
    for name, seq, alpha, pre in items:
        feats = pre if pre else _compute_all_features(seq)
        if feats.get("length") is None:
            feats["length"] = len(clean_seq(seq))
        row = _flatten_feature_row(name, seq, alpha, feats)
        rows.append(row)

    rows = [_round_feature_row(r, nd=4) for r in rows]

    first_alpha = None
    for _, _, alpha, _ in items:
        first_alpha = alpha
        if first_alpha:
            break
    letters = _alpha_letters(first_alpha or "DNA")  # "ACGT" or "ACGU"

    k2_labels = _lexicokeys(letters, 2)
    k3_labels = _lexicokeys(letters, 3)

    for r in rows:
        r["k2_values_str"] = "[" + ", ".join(f"{float(r.get('k2_'+k, 0.0)):.4f}" for k in k2_labels) + "]"
        r["k3_values_str"] = "[" + ", ".join(f"{float(r.get('k3_'+k, 0.0)):.4f}" for k in k3_labels) + "]"

    mnc_cols = [f"mnc_{b}" for b in letters]

    columns = [
        "name", "sequence",
        "length", "gc_pct", "gc_skew", "at_au_skew",
        *mnc_cols,
        "k2_values_str", "k3_values_str",
    ]

    headers = [
        "Name", "Sequence",
        "Length", "GC %", "GC skew", "AT/AU skew",
        *[f"MNC {b}" for b in letters],
        "K-mer (k=2)", "K-mer (k=3)",
    ]

    k2_labels_str = "[" + ", ".join(k2_labels) + "]"
    k3_labels_str = "[" + ", ".join(k3_labels) + "]"

    default_feature_keys = [c for c in columns if c not in ("name", "sequence")]

    umap_payload = {
        "rows": rows,
        "columns": columns,
        "default_feature_keys": default_feature_keys,
        "k2_labels": k2_labels,
        "k3_labels": k3_labels,
    }

    request.session["extractor_results"] = rows
    request.session["extractor_csv_meta"] = {
        "letters": letters,
        "k2_labels": k2_labels,
        "k3_labels": k3_labels,
        "base_headers": ["Name", "Sequence", "Length", "GC %", "GC skew", "AT/AU skew"],
        "mnc_headers": [f"MNC {b}" for b in letters],
    }

    return render(
        request,
        "tools/feature_extractor_results.html",
        {
            "results": rows,
            "columns": columns,
            "headers": headers,
            "count": len(rows),
            "engine": "Built-in",
            "umap_payload_json": json.dumps(umap_payload),
            "k2_labels_str": k2_labels_str,
            "k3_labels_str": k3_labels_str,
        }
    )

@csrf_protect
@ensure_csrf_cookie
@require_http_methods(["GET"])
def feature_extractor_download(request):
    rows = request.session.get("extractor_results") or []
    meta = request.session.get("extractor_csv_meta") or {}
    if not rows:
        return HttpResponse("No results to download.", status=400)

    letters   = meta.get("letters") or ("ACGU" if any("mnc_U" in r for r in rows) else "ACGT")
    k2_labels = meta.get("k2_labels") or _lexicokeys(letters, 2)
    k3_labels = meta.get("k3_labels") or _lexicokeys(letters, 3)

    base_headers = meta.get("base_headers") or ["Name", "Sequence", "Length", "GC %", "GC skew", "AT/AU skew"]
    mnc_headers  = meta.get("mnc_headers")  or [f"MNC {b}" for b in letters]

    base_key_map = {
        "Name": "name",
        "Sequence": "sequence",
        "Length": "length",
        "GC %": "gc_pct",
        "GC skew": "gc_skew",
        "AT/AU skew": "at_au_skew",
    }
    mnc_key_map = {f"MNC {b}": f"mnc_{b}" for b in letters}

    fieldnames = base_headers + mnc_headers + k2_labels + k3_labels

    out = StringIO()
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()

    for r in rows:
        rec = {}
        for h in base_headers:
            rec[h] = r.get(base_key_map[h], "")
        for h in mnc_headers:
            rec[h] = r.get(mnc_key_map[h], "")
        for lab in k2_labels:
            rec[lab] = r.get(f"k2_{lab}", "")
        for lab in k3_labels:
            rec[lab] = r.get(f"k3_{lab}", "")
        w.writerow(rec)

    resp = HttpResponse(out.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="features.csv"'
    return resp

# Feature calculation helpers

DNA_SET = set("ACGT")
RNA_SET = set("ACGU")

def _alphabet_set(alpha):
    return DNA_SET if alpha == "DNA" else RNA_SET

def _count_chars(seq, alphabet_set):
    counts = {b: 0 for b in alphabet_set}
    for ch in seq:
        if ch in alphabet_set:
            counts[ch] += 1
    return counts

def _mononucleotide_composition(seq, alpha):
    pool = _alphabet_set(alpha)
    counts = _count_chars(seq, pool)
    total = sum(counts.values()) or 1
    return {k: counts[k] / total for k in sorted(counts)}

def _kmer_freq(seq, k, alpha):
    pool = _alphabet_set(alpha)
    from itertools import product
    all_kmers = [''.join(p) for p in product(sorted(pool), repeat=k)]
    counts = {km: 0 for km in all_kmers}
    n = len(seq)
    if n >= k:
        for i in range(n - k + 1):
            win = seq[i:i+k]
            if all(c in pool for c in win):
                counts[win] += 1
    denom = max(1, (n - k + 1))
    return {km: counts[km] / denom for km in all_kmers}

def _gc_and_at_au_skew(seq, alpha):
    pool = _alphabet_set(alpha)
    if alpha == "DNA":
        a, t, g, c = 'A', 'T', 'G', 'C'
    else:
        a, t, g, c = 'A', 'U', 'G', 'C'
    counts = _count_chars(seq, pool)
    g_plus_c = counts[g] + counts[c]
    a_plus_tu = counts[a] + counts[t]
    gc_skew = (counts[g]-counts[c]) / g_plus_c if g_plus_c else 0.0
    at_au_skew = (counts[a]-counts[t]) / a_plus_tu if a_plus_tu else 0.0
    return gc_skew, at_au_skew

def _safe_gc_pct(seq, alpha):
    try:
        return gc_content(seq) if alpha in ("DNA", "RNA") else None
    except Exception:
        return None

def _compute_all_features(seq_raw: str):
    alpha = detect_alphabet_simple(seq_raw)
    clean = clean_seq(seq_raw).upper()
    length = len(clean)
    gc_pct = _safe_gc_pct(clean, alpha)
    gc_skew, at_au_skew = _gc_and_at_au_skew(clean, alpha)
    mnc = _mononucleotide_composition(clean, alpha)
    k2 = _kmer_freq(clean, 2, alpha)
    k3 = _kmer_freq(clean, 3, alpha)
    return {
        "length": length,
        "gc_content": gc_pct,
        "gc_skew": gc_skew,
        "au_or_at_skew": at_au_skew,
        "mnc": mnc,
        "kmer2": k2,
        "kmer3": k3,
    }

def _flatten_feature_row(name: str, seq: str, alpha: str, feats: dict):
    row = {
        "name": name,
        "sequence": seq,
        "alphabet": alpha,
        "length": feats.get("length"),
        "gc_pct": feats.get("gc_content"),
        "gc_skew": feats.get("gc_skew"),
        "at_au_skew": feats.get("au_or_at_skew"),
    }
    mnc = feats.get("mnc") or {}
    for k in sorted(mnc):
        row[f"mnc_{k}"] = mnc[k]
    k2 = feats.get("kmer2") or {}
    for k in sorted(k2):
        row[f"k2_{k}"] = k2[k]
    k3 = feats.get("kmer3") or {}
    for k in sorted(k3):
        row[f"k3_{k}"] = k3[k]
    return row

def _collect_sequences_for_explorer(request, cleaned):
    mode = cleaned["input_mode"]
    items = []

    if mode == "database":
        return []

    if mode == "text":
        for name, seq in _parse_text_sequences_input(cleaned["sequences_text"]):
            items.append((name, seq, "Sequence"))

    elif mode == "fasta":
        for name, seq in parse_fasta(request.FILES["file"].file):
            items.append((name, seq, "FASTA"))

    elif mode == "csv":
        for name, seq in parse_csv_sequences(request.FILES["file"].file, "sequence"):
            items.append((name, seq, "CSV"))

    return items

def _load_rnafm():
    if _RNAFM_SINGLETON["model"] is not None:
        return _RNAFM_SINGLETON["model"], _RNAFM_SINGLETON["alphabet"], _RNAFM_SINGLETON["device"]
    device = os.getenv("RNAFM_DEVICE", "cpu")
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    model.to(device)
    _RNAFM_SINGLETON["model"] = model
    _RNAFM_SINGLETON["alphabet"] = alphabet
    _RNAFM_SINGLETON["device"] = device
    return model, alphabet, device

def compute_rnafm_embeddings(seqs_or_pairs, chunk_size=16):
    if not seqs_or_pairs:
        return []
    print(f"[RNA-FM] Incoming sequences: N={len(seqs_or_pairs)}")
    model, alphabet, device = _load_rnafm()
    batch_converter = alphabet.get_batch_converter()

    cleaned_pairs = []
    for i, item in enumerate(seqs_or_pairs):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, raw = item[0], item[1]
        else:
            label, raw = f"seq_{i}", item
        s = "".join(ch for ch in str(raw).upper() if ch.isalpha())
        s = s.replace("T", "U")
        if s:
            cleaned_pairs.append((str(label), s))

    if not cleaned_pairs:
        return []
    print(f"[RNA-FM] Incoming sequences: N={len(cleaned_pairs)}")
    emb_len = _RNAFM_SINGLETON["emb_len"]
    embeddings = np.zeros((len(cleaned_pairs), emb_len), dtype=np.float32)
    import torch
    with torch.no_grad():
        start = 0
        while start < len(cleaned_pairs):
            batch = cleaned_pairs[start : start + chunk_size]
            labels, seqs, toks = batch_converter(batch)
            toks = toks.to(device)
            out = model(toks, repr_layers=[12])
            rep = out["representations"][12].detach().cpu().numpy()
            for j, seq in enumerate(seqs):
                L = len(seq)
                vec = rep[j, 1:1+L, :].mean(axis=0)
                embeddings[start + j, :] = vec
            start += chunk_size
    print(embeddings.shape)
    return embeddings.tolist()

def _tsne_2d(embeddings, perplexity=30, seed=42):
    X = np.asarray(embeddings, dtype=float)
    if len(X) <= 1:
        return [0.0], [0.0]
    if len(X) == 2:
        return [-0.5, 0.5], [0.0, 0.0]
    from sklearn.manifold import TSNE
    perp = max(5, min(perplexity, len(X)//2))
    tsne = TSNE(n_components=2, perplexity= perp, random_state=seed)
    Y = tsne.fit_transform(X)
    return Y[:,0].tolist(), Y[:,1].tolist()

@csrf_protect
@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def feature_explorer(request):
    if request.method == "GET":
        form = FeatureExtractorForm()
        return render(request, "tools/feature_explorer_form.html", {"form": form})

    form = FeatureExtractorForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "tools/feature_explorer_form.html", {"form": form})

    if form.cleaned_data.get("input_mode") == "database":
        messages.error(request, "Database input is not available in Explorer. Use Text/FASTA/CSV.")
        return render(request, "tools/feature_explorer_form.html", {"form": form})

    items = _collect_sequences_for_explorer(request, form.cleaned_data)
    if not items:
        messages.error(request, "No sequences found from the chosen input.")
        return render(request, "tools/feature_explorer_form.html", {"form": form})

    names  = [n for (n, _, _) in items]
    seqs   = [s for (_, s, _) in items]
    groups = [g for (_, _, g) in items]

    pairs = list(zip(names, seqs))
    user_embeddings = compute_rnafm_embeddings(pairs)
    user_embeddings_arr = np.asarray(user_embeddings, dtype=np.float32) if user_embeddings else np.zeros((0, _RNAFM_SINGLETON["emb_len"]), dtype=np.float32)

    request.session["feature_explorer_user_embeddings"] = user_embeddings_arr.tolist()

    emb_dir = os.path.join(APP_DIR, "data", "embeddings")
    emb_dim = _RNAFM_SINGLETON["emb_len"]

    def _safe_load_first_existing(candidates):
        for name in candidates:
            path = os.path.join(emb_dir, name)
            if os.path.exists(path):
                try:
                    arr = np.load(path)
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    print(f"[refs] Loaded {arr.shape[0]} rows from {name}")
                    return arr
                except Exception as e:
                    print(f"[refs] Failed to load {name}: {e}")
                    return np.zeros((0, emb_dim), dtype=np.float32)
        print(f"[refs] None of {candidates} found under {emb_dir}")
        return np.zeros((0, emb_dim), dtype=np.float32)

    # si_arr = _safe_load_first_existing([
    #     "siRNA_embeddings.npy", "sirna_embeddings.npy",
    #     "siRNA_emebeddings.npy", "sirna_emebeddings.npy",
    # ])
    # mi_arr = _safe_load_first_existing([
    #     "miRNA_embeddings.npy", "mirna_embeddings.npy",
    #     "miRNA_emebeddings.npy", "mirna_emebeddings.npy",
    # ])
    # pi_arr = _safe_load_first_existing([
    #     "piRNA_embeddings.npy", "pirna_embeddings.npy",
    #     "piRNA_emebeddings.npy", "pirna_emebeddings.npy",
    # ])

    # refs = {"siRNA": si_arr, "miRNA": mi_arr, "piRNA": pi_arr}
    # n_si, n_mi, n_pi = si_arr.shape[0], mi_arr.shape[0], pi_arr.shape[0]

    # X_parts = [refs["siRNA"], refs["miRNA"], refs["piRNA"], user_embeddings_arr]
    # X = np.vstack([p for p in X_parts if p.size]) if any(p.size for p in X_parts) else np.zeros((0, _RNAFM_SINGLETON["emb_len"]), dtype=np.float32)
    # n_user = int(user_embeddings_arr.shape[0])

    # if X.shape[0] == 0:
    #     messages.error(request, "No embeddings to plot.")
    #     return render(request, "tools/feature_explorer_form.html", {"form": form})

    # xs_all, ys_all = _tsne_2d(X.tolist())

    # i0 = 0
    # i1 = i0 + n_si
    # i2 = i1 + n_mi
    # i3 = i2 + n_pi
    # i4 = i3 + n_user

    # xs_si, ys_si = xs_all[i0:i1], ys_all[i0:i1]
    # xs_mi, ys_mi = xs_all[i1:i2], ys_all[i1:i2]
    # xs_pi, ys_pi = xs_all[i2:i3], ys_all[i2:i3]
    # xs_user, ys_user = xs_all[i3:i4], ys_all[i3:i4]

    si_arr = _safe_load_first_existing([
        "siRNA_embeddings.npy", "sirna_embeddings.npy",
        "siRNA_emebeddings.npy", "sirna_emebeddings.npy",
    ])
    mi_arr = _safe_load_first_existing([
        "miRNA_embeddings.npy", "mirna_embeddings.npy",
        "miRNA_emebeddings.npy", "mirna_emebeddings.npy",
    ])
    pi_arr = _safe_load_first_existing([
        "piRNA_embeddings.npy", "pirna_embeddings.npy",
        "piRNA_emebeddings.npy", "pirna_emebeddings.npy",
    ])

    X_ref, Y_ref, counts = _build_or_load_joint_refs(si_arr, mi_arr, pi_arr)
    n_si, n_mi, n_pi = counts.get("siRNA", 0), counts.get("miRNA", 0), counts.get("piRNA", 0)
    n_user = int(user_embeddings_arr.shape[0])
    
    if X_ref.shape[0] > 0:
        Y_user = _knn_project(X_ref, Y_ref, user_embeddings_arr, k=50)
    else:
        if user_embeddings_arr.shape[0] == 0:
            messages.error(request, "No embeddings to plot.")
            return render(request, "tools/feature_explorer_form.html", {"form": form})
        xs_u, ys_u = _tsne_2d(user_embeddings_arr.tolist())
        Y_user = np.column_stack([np.asarray(xs_u, dtype=np.float32), np.asarray(ys_u, dtype=np.float32)])
        Y_ref = np.zeros((0, 2), dtype=np.float32)
        n_si = n_mi = n_pi = 0

    i0 = 0
    i1 = i0 + n_si
    i2 = i1 + n_mi
    i3 = i2 + n_pi

    Y_si = Y_ref[i0:i1]
    Y_mi = Y_ref[i1:i2]
    Y_pi = Y_ref[i2:i3]

    xs_si, ys_si = (Y_si[:,0].tolist(), Y_si[:,1].tolist()) if n_si else ([], [])
    xs_mi, ys_mi = (Y_mi[:,0].tolist(), Y_mi[:,1].tolist()) if n_mi else ([], [])
    xs_pi, ys_pi = (Y_pi[:,0].tolist(), Y_pi[:,1].tolist()) if n_pi else ([], [])

    xs_user, ys_user = Y_user[:,0].tolist(), Y_user[:,1].tolist()

    palette = _assign_colors(groups)
    colors = [palette[g] for g in groups]

    payload = {
        "user": {
            "names": names,
            "xs": xs_user,
            "ys": ys_user,
            "n": n_user,
            "groups": groups,
            "colors": colors,
            "group_palette": palette,
            "meta": {
                "source": form.cleaned_data["input_mode"],
                "db_types": [],
            },
        },
        "refs": {
            "siRNA": {"xs": xs_si, "ys": ys_si, "n": n_si},
            "miRNA": {"xs": xs_mi, "ys": ys_mi, "n": n_mi},
            "piRNA": {"xs": xs_pi, "ys": ys_pi, "n": n_pi},
        },
    }

    request.session["feature_explorer_payload"] = payload
    return render(request, "tools/feature_explorer_results.html", {
        "payload_json": mark_safe(json.dumps(payload))
    })

@require_GET
def feature_explorer_results(request):
    payload = request.session.get("feature_explorer_payload")
    if not payload:
        messages.warning(request, "No explorer payload found. Start again.")
        return redirect("feature_explorer")
    return render(request, "tools/feature_explorer_results.html", {
        "payload_json": mark_safe(json.dumps(payload))
    })

@require_GET
def feature_explorer_download_embeddings(request):
    data = request.session.get("feature_explorer_user_embeddings")
    if not data:
        return HttpResponse("No user embeddings to download.", status=400)

    arr = np.asarray(data, dtype=np.float32)
    from io import BytesIO
    buf = BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    resp = HttpResponse(buf.read(), content_type="application/octet-stream")
    resp["Content-Disposition"] = 'attachment; filename="user_embeddings.npy"'
    return resp