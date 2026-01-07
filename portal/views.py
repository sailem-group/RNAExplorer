# views.py (only the parts used by Feature Lab + help/about + downloads)

import json, os, csv, glob, re, math
import numpy as np
from io import StringIO, BytesIO
from itertools import product
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET, require_http_methods
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from django.utils.safestring import mark_safe
from sklearn.neighbors import NearestNeighbors
from functools import lru_cache
from typing import Optional

from .forms import FeatureExtractorForm
from .helpers import (
    gc_content, clean_seq, detect_alphabet_simple,
    parse_fasta, parse_csv_sequences, split_lines_sequences
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_CSV_FILE = os.path.join(APP_DIR, "data", "database.csv")
DATABASE_PART_DIR = os.path.join(APP_DIR, "data")
EMB_DIR = os.path.join(APP_DIR, "data", "embeddings")

REF_TSNE_FILE = os.path.join(EMB_DIR, "refs_tsne.npy")
REF_EMB_FILE  = os.path.join(EMB_DIR, "refs_embeddings.npy")
REF_META_FILE = os.path.join(EMB_DIR, "refs_meta.json")

FEAT_REF_EMB_FILE  = os.path.join(EMB_DIR, "interp_features.npy")
FEAT_REF_TSNE_FILE = os.path.join(EMB_DIR, "interp_tsne.npy")
FEAT_REF_META_FILE = os.path.join(EMB_DIR, "interp_meta.json")

REFS_PER_TYPE_LIMIT = 23_116
MAX_MATCH_ROWS_TO_SEND = 2000 
MOST_MATCHING_SEQ_LIMIT = 100

_RNAFM_SINGLETON = {"model": None, "alphabet": None, "emb_len": 640, "device": "cpu"}

def _is_ajax(request):
    return request.POST.get("__ajax__") == "1" or request.headers.get("x-requested-with") == "XMLHttpRequest"

def help(request):
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

def _safe_float(v):
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None

def _distance_to_similarity(d, sigma=1.0):
    try:
        d = float(d)
        return round(math.exp(-d / sigma), 3)
    except Exception:
        return None

def _safe_int(v):
    try:
        if v in (None, ""):
            return None
        return int(float(v))
    except Exception:
        return None

def _row_props_from_db_row(row: dict, cleaned_seq: Optional[str] = None) -> dict:
    seq_clean = cleaned_seq if cleaned_seq is not None else _clean_rna((row.get("Sequence") or row.get("sequence") or "").strip())
    rna_type = (row.get("RNA Type") or row.get("rna_type") or row.get("type") or "").strip() or "Unknown"
    link = (row.get("link") or row.get("Link") or "").strip() or ""
    species = (row.get("species") or row.get("Species") or "").strip()

    length = _safe_int(row.get("length"))
    if length is None and seq_clean:
        length = len(seq_clean)

    gc_pct = _safe_float(row.get("gc_content"))
    if gc_pct is None:
        gc_pct = _safe_float(row.get("gc_pct"))

    if gc_pct is not None and gc_pct <= 1.0:
        gc_pct *= 100.0

    if gc_pct is not None:
        gc_pct = round(gc_pct, 2)

    at_au_skew = _safe_float(row.get("au_or_at_skew"))
    if at_au_skew is None:
        at_au_skew = _safe_float(row.get("au_at_skew"))
    if at_au_skew is None:
        at_au_skew = _safe_float(row.get("at_au_skew"))

    return {
        "type": rna_type,
        "sequence": seq_clean,
        "link": link,
        "length": length,
        "gc_pct": gc_pct, 
        "gc_skew": _safe_float(row.get("gc_skew")),
        "at_au_skew": at_au_skew,
        "mnc_A": _safe_float(row.get("mnc_A")),
        "mnc_C": _safe_float(row.get("mnc_C")),
        "mnc_G": _safe_float(row.get("mnc_G")),
        "mnc_U": _safe_float(row.get("mnc_U")),
        "species": species, 
    }

def _feature_vector_for_query(q_raw: str, feat_names: list[str]) -> np.ndarray:
    feats = _compute_all_features(q_raw)
    flat = _flatten_feature_row("query", q_raw, detect_alphabet_simple(q_raw), feats)
    vec = []
    for k in feat_names:
        try:
            vec.append(float(flat.get(k, 0.0)))
        except Exception:
            vec.append(0.0)
    return np.asarray(vec, dtype=np.float32)


def _numeric_suffix_key(fp: str):
    m = re.search(r'\.part(\d+)\.', os.path.basename(fp))
    return int(m.group(1)) if m else 10**9


def _load_sharded_npy(dir_path, base_name):
    pattern = os.path.join(dir_path, f"{base_name}.part*.npy")
    files = sorted(glob.glob(pattern), key=_numeric_suffix_key)
    if not files:
        return None
    parts = []
    for fp in files:
        arr = np.load(fp).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        parts.append(arr)
    return np.vstack(parts)


def _load_sharded_csv_parts(base_dir, base_name="database"):
    pattern = os.path.join(base_dir, f"{base_name}.part*.csv")
    files = sorted(glob.glob(pattern), key=_numeric_suffix_key)
    for fp in files:
        with open(fp, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def _tsne_2d(embeddings, perplexity=30, seed=42):
    X = np.asarray(embeddings, dtype=float)
    if len(X) <= 1:
        return [0.0], [0.0]
    if len(X) == 2:
        return [-0.5, 0.5], [0.0, 0.0]
    from sklearn.manifold import TSNE
    perp = max(5, min(perplexity, len(X)//2))
    Y = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(X)
    return Y[:, 0].tolist(), Y[:, 1].tolist()

def _normalize_tsne_coords(Y: np.ndarray, target_half_range: float = 40.0) -> np.ndarray:
    if Y is None:
        return Y
    Y = np.asarray(Y, dtype=np.float32)
    if Y.size == 0:
        return Y

    mins = Y.min(axis=0)
    maxs = Y.max(axis=0)
    center = (maxs + mins) / 2.0
    span = (maxs - mins).max()

    if span <= 0:
        return Y - center

    scale = (2.0 * target_half_range) / span
    return (Y - center) * scale


def _alpha_letters(alpha: str) -> str:
    return "ACGT" if alpha == "DNA" else "ACGU"


def _lexicokeys(alpha_letters: str, k: int):
    from itertools import product
    return [''.join(p) for p in product(sorted(alpha_letters), repeat=k)]


def _round_feature_row(row: dict, nd=4) -> dict:
    out = {}
    for k, v in row.items():
        if k in ("name", "sequence", "alphabet", "length"):
            out[k] = v
            continue
        try:
            out[k] = (None if v is None else round(float(v), nd))
        except (TypeError, ValueError):
            out[k] = v
    return out


def _flatten_feature_row(name: str, seq: str, alpha: str, feats: dict):
    row = {
        "name": name, "sequence": seq, "alphabet": alpha,
        "length": feats.get("length"),
        "gc_pct": feats.get("gc_content"),
        "gc_skew": feats.get("gc_skew"),
        "at_au_skew": feats.get("au_or_at_skew"),
    }
    for k, v in (feats.get("mnc") or {}).items():
        row[f"mnc_{k}"] = v
    for k, v in (feats.get("kmer2") or {}).items():
        row[f"k2_{k}"] = v
    for k, v in (feats.get("kmer3") or {}).items():
        row[f"k3_{k}"] = v
    return row


def _save_refs_tsne(X_ref, Y_ref, counts):
    os.makedirs(os.path.dirname(REF_TSNE_FILE), exist_ok=True)
    np.save(REF_EMB_FILE, X_ref.astype(np.float32))
    np.save(REF_TSNE_FILE, Y_ref.astype(np.float32))
    with open(REF_META_FILE, "w") as f:
        json.dump(counts, f)


def _load_refs_tsne():
    try:
        X_ref = _load_sharded_npy(EMB_DIR, "refs_embeddings")
        Y_ref = np.load(REF_TSNE_FILE).astype(np.float32) if os.path.exists(REF_TSNE_FILE) else None

        if X_ref is None and os.path.exists(REF_EMB_FILE):
            X_ref = np.load(REF_EMB_FILE).astype(np.float32)
        if Y_ref is None:
            return None, None, None

        with open(REF_META_FILE, "r") as f:
            counts = json.load(f)

        if X_ref.shape[0] != Y_ref.shape[0]:
            return None, None, None
        return X_ref, Y_ref, counts
    except Exception as e:
        print("Error loading ref shards:", e)
        return None, None, None


def _build_or_load_interpretable_refs():
    X, Y, feat_names, labels, counts = None, None, None, None, None

    if os.path.exists(FEAT_REF_EMB_FILE):
        try:
            X = np.load(FEAT_REF_EMB_FILE).astype(np.float32)
            Y = np.load(FEAT_REF_TSNE_FILE).astype(np.float32)
            with open(FEAT_REF_META_FILE, "r") as f:
                meta = json.load(f)
            feat_names = meta["feature_names"]
            labels = meta["labels"]
            counts = meta["counts"]
            csv_indices = meta.get("csv_indices")

            return X, Y, feat_names, labels, counts, csv_indices
        except Exception:
            pass

    rows = []
    parts_found = glob.glob(os.path.join(DATABASE_PART_DIR, "database.part*.csv"))
    reader_iter = _load_sharded_csv_parts(DATABASE_PART_DIR) if parts_found else csv.DictReader(open(DATABASE_CSV_FILE))

    csv_idx = 0

    for row in reader_iter:
        seq = (row.get("Sequence") or row.get("sequence") or "").strip()

        if not seq:
            csv_idx += 1
            continue

        rec = {}
        for ksrc, kdst in [
            ("length", "length"),
            ("gc_content", "gc_pct"),
            ("gc_skew", "gc_skew"),
            ("au_at_skew", "at_au_skew"),
            ("au_or_at_skew", "at_au_skew"),
        ]:
            v = row.get(ksrc)
            if v not in (None, ""):
                try:
                    rec[kdst] = float(v)
                except:
                    pass

        for b in "ACGU":
            v = row.get(f"mnc_{b}")
            if v not in (None, ""):
                try:
                    rec[f"mnc_{b}"] = float(v)
                except:
                    pass

        rec["_label"] = (row.get("RNA Type") or "").strip() or "Unknown"
        rec["_csv_idx"] = csv_idx
        rows.append(rec)
        csv_idx += 1

    if not rows:
        return np.zeros((0, 0), np.float32), np.zeros((0, 2), np.float32), [], [], {}

    feat_names = sorted({k for r in rows for k in r.keys() if k != "_label"})
    X = np.asarray([[float(r.get(k, 0.0)) for k in feat_names] for r in rows], np.float32)
    labels = [r["_label"] for r in rows]
    from sklearn.manifold import TSNE
    Y = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    counts = {lab: labels.count(lab) for lab in set(labels)}

    csv_indices = [r["_csv_idx"] for r in rows]

    np.save(FEAT_REF_EMB_FILE, X)
    np.save(FEAT_REF_TSNE_FILE, Y)
    with open(FEAT_REF_META_FILE, "w") as f:
        json.dump(
            {
                "feature_names": feat_names,
                "labels": labels,
                "counts": counts,
                "csv_indices": csv_indices,
            },
            f,
        )

    return X, Y, feat_names, labels, counts, csv_indices


def _clean_rna(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalpha()).replace("T", "U")


def _load_ref_sequences_by_type():
    seqs = {"siRNA": [], "miRNA": [], "piRNA": []}
    links = {"siRNA": [], "miRNA": [], "piRNA": []}
    parts_found = glob.glob(os.path.join(DATABASE_PART_DIR, "database.part*.csv"))
    reader_iter = _load_sharded_csv_parts(DATABASE_PART_DIR) if parts_found else csv.DictReader(open(DATABASE_CSV_FILE))

    for row in reader_iter:
        rna_type = (row.get("RNA Type") or "").strip()
        seq = (row.get("Sequence") or row.get("sequence") or "").strip()
        link = (row.get("link") or row.get("Link") or "").strip()
        if seq and rna_type in seqs:
            seqs[rna_type].append(_clean_rna(seq))
            links[rna_type].append(link)
    return seqs, links


def _build_or_load_joint_refs(si_arr, mi_arr, pi_arr):
    Xc, Yc, Cc = _load_refs_tsne()
    if Xc is not None:
        return Xc, Yc, Cc
    n_si, n_mi, n_pi = si_arr.shape[0], mi_arr.shape[0], pi_arr.shape[0]
    parts = [p for p in (si_arr, mi_arr, pi_arr) if p.size]
    if not parts:
        return (
            np.zeros((0, _RNAFM_SINGLETON["emb_len"]), np.float32),
            np.zeros((0, 2), np.float32),
            {"siRNA": 0, "miRNA": 0, "piRNA": 0},
        )
    X_ref = np.vstack(parts).astype(np.float32)
    xs, ys = _tsne_2d(X_ref.tolist())
    Y_ref = np.column_stack(
        [np.asarray(xs, np.float32), np.asarray(ys, np.float32)]
    )

    Y_ref = _normalize_tsne_coords(Y_ref, target_half_range=40.0)

    counts = {"siRNA": n_si, "miRNA": n_mi, "piRNA": n_pi}
    _save_refs_tsne(X_ref, Y_ref, counts)
    return X_ref, Y_ref, counts


def _load_rnafm():
    if _RNAFM_SINGLETON["model"] is not None:
        return _RNAFM_SINGLETON["model"], _RNAFM_SINGLETON["alphabet"], _RNAFM_SINGLETON["device"]
    device = os.getenv("RNAFM_DEVICE", "cpu")
    import fm, torch
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    model.to(device)
    _RNAFM_SINGLETON.update({"model": model, "alphabet": alphabet, "device": device})
    return model, alphabet, device


def _knn_project(X_ref: np.ndarray, Y_ref: np.ndarray, X_new: np.ndarray, k=50, eps=1e-6):
    if X_new.size == 0:
        return np.zeros((0, 2), np.float32)
    k = min(k, max(1, X_ref.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean").fit(X_ref)
    dists, idxs = nn.kneighbors(X_new)
    w = 1.0 / (dists + eps)
    w = w / w.sum(axis=1, keepdims=True)
    Y_neighbors = Y_ref[idxs]
    return (w[..., None] * Y_neighbors).sum(axis=1).astype(np.float32)


def compute_rnafm_embeddings(seqs_or_pairs, chunk_size=16):
    if not seqs_or_pairs:
        return []
    model, alphabet, device = _load_rnafm()
    batch_converter = alphabet.get_batch_converter()
    cleaned = []
    for i, it in enumerate(seqs_or_pairs):
        label, raw = (it if (isinstance(it, (list, tuple)) and len(it) >= 2) else (f"seq_{i}", it))
        s = "".join(ch for ch in str(raw).upper() if ch.isalpha()).replace("T", "U")
        if s:
            cleaned.append((str(label), s))
    if not cleaned:
        return []
    emb_len = _RNAFM_SINGLETON["emb_len"]
    out = np.zeros((len(cleaned), emb_len), np.float32)
    import torch
    with torch.no_grad():
        start = 0
        while start < len(cleaned):
            batch = cleaned[start:start+chunk_size]
            labels, seqs, toks = batch_converter(batch)
            toks = toks.to(_RNAFM_SINGLETON["device"])
            rep = model(toks, repr_layers=[12])["representations"][12].detach().cpu().numpy()
            for j, seq in enumerate(seqs):
                L = len(seq)
                out[start+j, :] = rep[j, 1:1+L, :].mean(axis=0)
            start += chunk_size
    return out.tolist()


def _compute_all_features(seq_raw: str):
    alpha = detect_alphabet_simple(seq_raw)
    clean = clean_seq(seq_raw).upper()
    length = len(clean)

    def _kmer_freq(seq, k, alpha):
        pool = set(_alpha_letters(alpha))
        from itertools import product
        all_kmers = [''.join(p) for p in product(sorted(pool), repeat=k)]
        counts = {km: 0 for km in all_kmers}
        n = len(seq)
        if n >= k:
            for i in range(n-k+1):
                w = seq[i:i+k]
                if all(c in pool for c in w):
                    counts[w] += 1
        denom = max(1, n-k+1)
        return {km: counts[km]/denom for km in all_kmers}

    def _mnc(seq, alpha):
        pool = set(_alpha_letters(alpha))
        d = {b: 0 for b in pool}
        for ch in seq:
            if ch in pool:
                d[ch] += 1
        tot = sum(d.values()) or 1
        return {k: d[k]/tot for k in sorted(d)}

    def _skews(seq, alpha):
        if alpha == "DNA":
            a, t, g, c = 'A', 'T', 'G', 'C'
        else:
            a, t, g, c = 'A', 'U', 'G', 'C'
        d = {a: 0, t: 0, g: 0, c: 0}
        for ch in seq:
            if ch in d:
                d[ch] += 1
        gc = d[g] + d[c]
        at = d[a] + d[t]
        return ((d[g]-d[c])/gc if gc else 0.0, (d[a]-d[t])/at if at else 0.0)

    gc_pct = gc_content(clean) if alpha in ("DNA", "RNA") else None
    gc_skew, at_au_skew = _skews(clean, alpha)
    k2 = _kmer_freq(clean, 2, alpha)
    k3 = _kmer_freq(clean, 3, alpha)
    return {
        "length": length,
        "gc_content": gc_pct,
        "gc_skew": gc_skew,
        "au_or_at_skew": at_au_skew,
        "mnc": _mnc(clean, alpha),
        "kmer2": k2,
        "kmer3": k3,
    }


@lru_cache(maxsize=128)
def _compute_refs_tsne_for_subset(key: tuple, Xi_shape0: int, perplexity=30):
    return None


def _tsne_refs_from_subset(Xi_sub: np.ndarray, seed=42):
    from sklearn.manifold import TSNE
    n = Xi_sub.shape[0]
    if n <= 1:
        return np.zeros((n, 2), np.float32)
    perp = max(5, min(30, n//2))
    Yi = TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(Xi_sub)
    return Yi.astype(np.float32)

def _interp_refs_payload_from_embedding(Yi: np.ndarray, feat_labels: list[str], interp_csv_indices: list[int],):
    payload = {}
    if Yi is None or Yi.shape[0] == 0:
        return payload

    parts_found = glob.glob(os.path.join(DATABASE_PART_DIR, "database.part*.csv"))
    reader_iter = (
        _load_sharded_csv_parts(DATABASE_PART_DIR)
        if parts_found
        else csv.DictReader(open(DATABASE_CSV_FILE))
    )
    seqs_all = []
    links_all = []
    for row in reader_iter:
        seq = (row.get("Sequence") or row.get("sequence") or "").strip()
        link = (row.get("link") or row.get("Link") or "").strip()
        seqs_all.append(_clean_rna(seq) if seq else "")
        links_all.append(link)

    from collections import defaultdict
    idxs = defaultdict(list)
    for interp_i, lab in enumerate(feat_labels):
        idxs[lab].append(interp_i)

    for lab, interp_ids in idxs.items():
        total = len(interp_ids)
        ids_shown = interp_ids[:REFS_PER_TYPE_LIMIT]

        payload[lab] = {
            "xs": [float(Yi[i, 0]) for i in ids_shown],
            "ys": [float(Yi[i, 1]) for i in ids_shown],
            "n": len(ids_shown),
            "total": total,
            "seqs": [
                seqs_all[interp_csv_indices[i]]
                if interp_csv_indices[i] < len(seqs_all)
                else ""
                for i in ids_shown
            ],
            "links": [
                links_all[interp_csv_indices[i]]
                if interp_csv_indices[i] < len(links_all)
                else ""
                for i in ids_shown
            ],
        }

    return payload

def _interp_refs_payload_from_embedding_subset(Yi: np.ndarray, feat_labels_sub: list[str],
                                               global_row_indices: list[int], full_counts: dict):
    payload = {}
    if Yi is None or Yi.shape[0] == 0:
        return payload

    parts_found = glob.glob(os.path.join(DATABASE_PART_DIR, "database.part*.csv"))
    reader_iter = _load_sharded_csv_parts(DATABASE_PART_DIR) if parts_found else csv.DictReader(open(DATABASE_CSV_FILE))
    seqs_all = []
    links_all = []
    for row in reader_iter:
        seq  = (row.get("Sequence") or row.get("sequence") or "").strip()
        link = (row.get("link") or row.get("Link") or "").strip()
        seqs_all.append(_clean_rna(seq) if seq else "")
        links_all.append(link)

    from collections import defaultdict
    idxs_by_lab = defaultdict(list)
    for local_i, lab in enumerate(feat_labels_sub):
        idxs_by_lab[lab].append(local_i)

    for lab, local_ids in idxs_by_lab.items():
        total = len(local_ids)
        ids_shown = local_ids[:REFS_PER_TYPE_LIMIT]
        payload[lab] = {
            "xs": [float(Yi[i, 0]) for i in ids_shown],
            "ys": [float(Yi[i, 1]) for i in ids_shown],
            "n": len(ids_shown),
            "total": int(full_counts.get(lab, 0)),
            "seqs": [
                (seqs_all[global_row_indices[i]] if global_row_indices[i] < len(seqs_all) else "")
                for i in ids_shown
            ],
            "links": [
                (links_all[global_row_indices[i]] if global_row_indices[i] < len(links_all) else "")
                for i in ids_shown
            ],
        }
    return payload

@csrf_protect
@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def feature_lab(request):
    """Single page: unified input -> overlays on both Deep & Interpretable; plus Query sequence."""

    def _collect_items(req, mode):
        items = []
        if mode == "text":
            raw = req.POST.get("sequences_text", "") or ""
            if any(line.lstrip().startswith(">") for line in raw.splitlines()):
                for name, seq in parse_fasta(BytesIO(raw.encode("utf-8"))):
                    items.append((name, seq))
            else:
                for name, seq in split_lines_sequences(raw):
                    items.append((name, seq))
        elif mode == "fasta":
            f = req.FILES.get("file")
            if f:
                for name, seq in parse_fasta(f.file):
                    items.append((name, seq))
        elif mode == "csv":
            f = req.FILES.get("file")
            if f:
                for name, seq in parse_csv_sequences(f.file, "sequence"):
                    items.append((name, seq))
        return items[:10], False

    def _make_deep_refs_payload():
        emb_dir = os.path.join(APP_DIR, "data", "embeddings")
        emb_dim = _RNAFM_SINGLETON["emb_len"]

        def _load_first(cands):
            for nm in cands:
                p = os.path.join(emb_dir, nm)
                if os.path.exists(p):
                    arr = np.load(p).astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr
            return np.zeros((0, emb_dim), np.float32)

        si = _load_first(["siRNA_embeddings.npy", "sirna_embeddings.npy"])
        mi = _load_first(["miRNA_embeddings.npy", "mirna_embeddings.npy"])
        pi = _load_first(["piRNA_embeddings.npy", "pirna_embeddings.npy"])

        Xr, Yr, counts = _build_or_load_joint_refs(si, mi, pi)
        n_si, n_mi, n_pi = counts.get("siRNA", 0), counts.get("miRNA", 0), counts.get("piRNA", 0)
        i0, i1 = 0, n_si
        i2, i3 = i1 + n_mi, i1 + n_mi + n_pi

        ref_seqs_by_type, ref_links_by_type = _load_ref_sequences_by_type()

        # Desired limits based on embeddings
        lim_si_raw = min(n_si, REFS_PER_TYPE_LIMIT)
        lim_mi_raw = min(n_mi, REFS_PER_TYPE_LIMIT)
        lim_pi_raw = min(n_pi, REFS_PER_TYPE_LIMIT)

        def _limit_for_type(n_type, lim_raw, seqs_for_type):
            if n_type == 0:
                return 0
            if seqs_for_type:
                return min(lim_raw, len(seqs_for_type))
            return lim_raw

        lim_si = _limit_for_type(n_si, lim_si_raw, ref_seqs_by_type["siRNA"])
        lim_mi = _limit_for_type(n_mi, lim_mi_raw, ref_seqs_by_type["miRNA"])
        lim_pi = _limit_for_type(n_pi, lim_pi_raw, ref_seqs_by_type["piRNA"])

        refs = {
            "siRNA": {
                "xs": Yr[i0:i1, 0].tolist()[:lim_si] if n_si else [],
                "ys": Yr[i0:i1, 1].tolist()[:lim_si] if n_si else [],
                "n": lim_si,
                "total": n_si,
            },
            "miRNA": {
                "xs": Yr[i1:i2, 0].tolist()[:lim_mi] if n_mi else [],
                "ys": Yr[i1:i2, 1].tolist()[:lim_mi] if n_mi else [],
                "n": lim_mi,
                "total": n_mi,
            },
            "piRNA": {
                "xs": Yr[i2:i3, 0].tolist()[:lim_pi] if n_pi else [],
                "ys": Yr[i2:i3, 1].tolist()[:lim_pi] if n_pi else [],
                "n": lim_pi,
                "total": n_pi,
            },
        }

        # Attach sequences + links per type where we have them
        if n_si and ref_seqs_by_type["siRNA"]:
            refs["siRNA"]["seqs"] = ref_seqs_by_type["siRNA"][:lim_si]
            refs["siRNA"]["links"] = ref_links_by_type["siRNA"][:lim_si]
        if n_mi and ref_seqs_by_type["miRNA"]:
            refs["miRNA"]["seqs"] = ref_seqs_by_type["miRNA"][:lim_mi]
            refs["miRNA"]["links"] = ref_links_by_type["miRNA"][:lim_mi]
        if n_pi and ref_seqs_by_type["piRNA"]:
            refs["piRNA"]["seqs"] = ref_seqs_by_type["piRNA"][:lim_pi]
            refs["piRNA"]["links"] = ref_links_by_type["piRNA"][:lim_pi]

        return refs, Xr, Yr, (n_si, n_mi, n_pi)

    def _make_interp_refs_payload():
        Xr, Yr, feat_names, feat_labels, feat_counts, interp_csv_indices = _build_or_load_interpretable_refs()
        if Yr is not None and Yr.shape[0]:
            payload = _interp_refs_payload_from_embedding(Yr, feat_labels,interp_csv_indices)
        else:
            payload = {}
        return payload, Xr, Yr, feat_names, interp_csv_indices

    form = FeatureExtractorForm() if request.method == "GET" else FeatureExtractorForm(request.POST, request.FILES)
    selected_feats = request.POST.getlist("features")

    def _respond(deep_payload, interp_payload, has_user=False, banner=None):
        if _is_ajax(request):
            return JsonResponse({
                "deep": deep_payload,
                "interp": interp_payload,
                "has_user": bool(has_user),
                "banner": banner or ""
            })
        ctx = {
            "form": form,
            "deep_payload_json": mark_safe(json.dumps(deep_payload)),
            "interp_payload_json": mark_safe(json.dumps(interp_payload)),
            "has_user": has_user,
        }
        if banner:
            ctx["inline_banner"] = banner
        return render(request, "tools/feature_lab.html", ctx)

    deep_refs, Xd, Yd, (n_si, n_mi, n_pi) = _make_deep_refs_payload()
    interp_refs, Xi, Yi, feat_names, interp_csv_indices = _make_interp_refs_payload()

    if request.method == "GET":
        request.session["has_user_overlaid"] = False
        return _respond({"refs": deep_refs}, {"refs": interp_refs}, has_user=False)

    panel = (request.POST.get("__panel__") or "").strip().lower()
    if panel == "query":
        request.session["has_user_overlaid"] = False
        q_raw = (request.POST.get("query_seq") or "").strip()
        q = _clean_rna(q_raw)
        deep_payload = {"refs": deep_refs}

        if not q:
            return _respond(
                {"refs": deep_refs}, {"refs": interp_refs}, has_user=False,
                banner="Please paste a single sequence to query."
            )

        ref_seqs, ref_links = _load_ref_sequences_by_type()
        ok = True
        if len(ref_seqs["siRNA"]) and len(ref_seqs["siRNA"]) != n_si:
            ok = False
        if len(ref_seqs["miRNA"]) and len(ref_seqs["miRNA"]) != n_mi:
            ok = False
        if len(ref_seqs["piRNA"]) and len(ref_seqs["piRNA"]) != n_pi:
            ok = False

        seq_to_emb = {}

        i0, i1 = 0, n_si
        i2, i3 = i1 + n_mi, i1 + n_mi + n_pi

        def _fill_seq_emb(blockX, seqs):
            m = min(blockX.shape[0], len(seqs))
            for j in range(m):
                s_clean = _clean_rna(seqs[j])
                if s_clean not in seq_to_emb:
                    seq_to_emb[s_clean] = blockX[j]

        if n_si and ref_seqs["siRNA"]:
            _fill_seq_emb(Xd[i0:i1], ref_seqs["siRNA"])
        if n_mi and ref_seqs["miRNA"]:
            _fill_seq_emb(Xd[i1:i2], ref_seqs["miRNA"])
        if n_pi and ref_seqs["piRNA"]:
            _fill_seq_emb(Xd[i2:i3], ref_seqs["piRNA"])

        xs_q, ys_q, seqs_q, links_q = [], [], [], []
        if Xd.shape[0]:
            i0, i1 = 0, n_si
            i2, i3 = i1 + n_mi, i1 + n_mi + n_pi

            def _append(blockY, seqs, links):
                nonlocal xs_q, ys_q, seqs_q, links_q
                m = min(blockY.shape[0], len(seqs))
                for j in range(m):
                    s_clean = _clean_rna(seqs[j])
                    if q in s_clean:
                        xs_q.append(float(blockY[j, 0]))
                        ys_q.append(float(blockY[j, 1]))
                        seqs_q.append(s_clean)
                        links_q.append((links[j] if j < len(links) else "") or "")

            if n_si and ref_seqs["siRNA"]:
                _append(Yd[i0:i1], ref_seqs["siRNA"], ref_links["siRNA"])
            if n_mi and ref_seqs["miRNA"]:
                _append(Yd[i1:i2], ref_seqs["miRNA"], ref_links["miRNA"])
            if n_pi and ref_seqs["piRNA"]:
                _append(Yd[i2:i3], ref_seqs["piRNA"], ref_links["piRNA"])

        deep_nearest_seq_set = set()
        deep_nearest_dist = {}

        if xs_q and ys_q:
            query_embs = []

            for s in seqs_q:
                emb = seq_to_emb.get(s)
                if emb is not None:
                    query_embs.append(emb)

            if not query_embs:
                return _respond(
                    {"refs": deep_refs}, {"refs": interp_refs},
                    has_user=False,
                    banner="Unable to compute embedding-based similarity."
                )

            q_center = np.mean(np.asarray(query_embs), axis=0)

            candidates = []  # (dist, seq_clean)

            i0, i1 = 0, n_si
            i2, i3 = i1 + n_mi, i1 + n_mi + n_pi

            def _add_candidates(seqs):
                for s_clean in seqs:
                    emb = seq_to_emb.get(_clean_rna(s_clean))
                    if emb is None:
                        continue
                    d = float(np.linalg.norm(emb - q_center))
                    candidates.append((d, _clean_rna(s_clean)))

            if n_si and ref_seqs["siRNA"]:
                _add_candidates(ref_seqs["siRNA"])
            if n_mi and ref_seqs["miRNA"]:
                _add_candidates(ref_seqs["miRNA"])
            if n_pi and ref_seqs["piRNA"]:
                _add_candidates(ref_seqs["piRNA"])

            candidates.sort(key=lambda t: t[0])

            for d, s_clean in candidates:
                if s_clean in deep_nearest_seq_set:
                    continue
                deep_nearest_seq_set.add(s_clean)
                deep_nearest_dist[s_clean] = float(d)
                if len(deep_nearest_seq_set) >= MOST_MATCHING_SEQ_LIMIT:
                    break

        xs_q_i, ys_q_i, seqs_q_i, links_q_i = [], [], [], []
        nearest_rows = []
        match_rows = []
        match_total = 0

        # dedupe sets
        seen_match_seq = set()
        seen_match_plot = set()
        seen_nearest_seq = set()
        csv_to_interp = {csv_i: i for i, csv_i in enumerate(interp_csv_indices)}
        if Yi is not None and isinstance(Yi, np.ndarray) and Yi.shape[0] > 0:
            parts_found = glob.glob(os.path.join(DATABASE_PART_DIR, "database.part*.csv"))
            reader_iter = _load_sharded_csv_parts(DATABASE_PART_DIR) if parts_found else csv.DictReader(open(DATABASE_CSV_FILE))

            idx = 0
            for row in reader_iter:
                seq = (row.get("Sequence") or row.get("sequence") or "").strip()
                link = (row.get("link") or row.get("Link") or "").strip()

                if not seq:
                    idx += 1
                    continue

                seq_clean = _clean_rna(seq)
                is_exact_match = (seq_clean == q)
                is_fragment_match = (q in seq_clean and not is_exact_match)
                is_match = is_exact_match or is_fragment_match
                interp_i = csv_to_interp.get(idx)

                if interp_i is not None and is_match and seq_clean not in seen_match_plot:
                    xs_q_i.append(float(Yi[interp_i, 0]))
                    ys_q_i.append(float(Yi[interp_i, 1]))
                    seqs_q_i.append(seq_clean)
                    links_q_i.append(link)
                    seen_match_plot.add(seq_clean)

                    if is_match and seq_clean not in seen_match_seq:
                        seen_match_seq.add(seq_clean)
                        match_total += 1
                        if len(match_rows) < MAX_MATCH_ROWS_TO_SEND:
                            match_rows.append(
                                _row_props_from_db_row(row, cleaned_seq=seq_clean)
                            )

                if seq_clean in deep_nearest_seq_set and seq_clean not in seen_nearest_seq:
                    dist = float(deep_nearest_dist.get(seq_clean, 1e18))
                    rec = _row_props_from_db_row(row, cleaned_seq=seq_clean)
                    rec["similarity"] = _distance_to_similarity(dist)
                    nearest_rows.append(rec)
                    seen_nearest_seq.add(seq_clean)

                idx += 1

        if not xs_q and not xs_q_i and not nearest_rows:
            return _respond(
                {"refs": deep_refs}, {"refs": interp_refs}, has_user=False,
                banner="No matches found in the reference."
            )

        deep_payload = {
            "refs": deep_refs,
            "query": {
                "xs": xs_q,
                "ys": ys_q,
                "n": len(xs_q),
                "seqs": seqs_q,
                "links": links_q,
            },
        }

        interp_payload = {
            "refs": interp_refs,
            "query": {
                "xs": xs_q_i,
                "ys": ys_q_i,
                "n": len(xs_q_i),
                "seqs": seqs_q_i,
                "links": links_q_i,
            },
            "tables": {
                "query_seq": q,
                "nearest": {
                    "rows": sorted(nearest_rows, key=lambda r: r.get("similarity", 0), reverse=True)[:MOST_MATCHING_SEQ_LIMIT],
                    "k": MOST_MATCHING_SEQ_LIMIT,
                },
                "matches": {
                    "rows": match_rows,
                    "total": int(match_total),  # now unique count
                    "truncated": bool(match_total > len(match_rows)),
                    "sent": len(match_rows),
                }
            }
        }

        return _respond(deep_payload, interp_payload, has_user=False)


    elif panel == "interp":
        selected_feats = request.POST.getlist("features")

        Xi_all, Yi_all, feat_names, feat_labels, feat_counts , interp_csv_indices= _build_or_load_interpretable_refs()

        if selected_feats:
            sel_idx = [i for i, k in enumerate(feat_names) if k in selected_feats]
            if not sel_idx:
                sel_idx = list(range(len(feat_names)))
        else:
            sel_idx = list(range(len(feat_names)))

        Xi_use_all = Xi_all[:, sel_idx]

        from collections import defaultdict
        lab_to_idxs = defaultdict(list)
        for i, lab in enumerate(feat_labels):
            lab_to_idxs[lab].append(i)

        rng = np.random.default_rng(42)
        subset_rows = []
        for lab, idxs in lab_to_idxs.items():
            if len(idxs) > REFS_PER_TYPE_LIMIT:
                pick = rng.choice(idxs, REFS_PER_TYPE_LIMIT, replace=False)
                subset_rows.extend(sorted(pick.tolist()))
            else:
                subset_rows.extend(idxs)
        subset_rows = sorted(subset_rows)

        Xi_sub = Xi_use_all[subset_rows, :]
        feat_labels_sub = [feat_labels[i] for i in subset_rows]

        Yi_use = _tsne_refs_from_subset(Xi_sub)

        interp_refs = _interp_refs_payload_from_embedding_subset(Yi_use, feat_labels_sub, subset_rows, feat_counts)

        has_user = bool(request.session.get("has_user_overlaid"))
        # interp_payload = {"refs": interp_refs}
        interp_payload = {
            "refs": interp_refs,
            "query": {
                "xs": [],
                "ys": [],
                "n": 0,
                "seqs": [],
                "links": [],
            },
            "tables": {
                "query_seq": "",
                "nearest": {
                    "rows": [],
                    "k": 0,
                },
                "matches": {
                    "rows": [],
                    "total": 0,
                    "truncated": False,
                    "sent": 0,
                },
            },
        }

        if has_user:
            rows = request.session.get("extractor_results") or []
            if rows:
                def vec_sel(r):
                    v = []
                    for i in sel_idx:
                        k = feat_names[i]
                        try:
                            v.append(float(r.get(k, 0.0)))
                        except:
                            v.append(0.0)
                    return v
                X_new = np.asarray([vec_sel(r) for r in rows], np.float32)
                Yu_feat = _knn_project(Xi_sub, Yi_use, X_new, k=50)

                interp_payload["user"] = {
                    "names": [r.get("name", "") for r in rows],
                    "seqs": [r.get("sequence", "") for r in rows],
                    "xs": Yu_feat[:, 0].tolist(),
                    "ys": Yu_feat[:, 1].tolist(),
                    "n": len(rows),
                    "groups": ["Sequence"]*len(rows),
                    "colors": ["#FFD700"]*len(rows),
                    "group_palette": {"Sequence": "#FFD700"},
                    "meta": {"source": "feature_selection"},
                }

        return JsonResponse({"interp": interp_payload, "deep": {"refs": deep_refs,},"has_user": has_user})

    # panel == "both"
    input_mode = request.POST.get("input_mode", "text")
    items, _ = _collect_items(request, input_mode)
    if not items:
        if _is_ajax(request):
            return JsonResponse({
                "deep": {"refs": deep_refs},
                "interp": {"refs": interp_refs},
                "has_user": False,
                "banner": "No sequences found. Please provide up to 10 sequences.",
            })
        messages.error(request, "No sequences found. Please provide up to 10 sequences.")
        return redirect("feature_lab")

    names = [n for (n, _) in items]
    seqs = [s for (_, s) in items]
    pairs = list(items)
    U_emb = compute_rnafm_embeddings(pairs)
    U_arr = np.asarray(U_emb, np.float32) if U_emb else np.zeros((0, _RNAFM_SINGLETON["emb_len"]), np.float32)
    request.session["feature_explorer_user_embeddings"] = U_arr.tolist()

    if Xd.shape[0] > 0:
        Yu = _knn_project(Xd, Yd, U_arr, k=50)
    else:
        if U_arr.shape[0] == 0:
            messages.error(request, "No embeddings to plot.")
            return redirect("feature_lab")
        xsu, ysu = _tsne_2d(U_arr.tolist())
        Yu = np.column_stack([np.asarray(xsu, np.float32), np.asarray(ysu, np.float32)])

    deep_payload = {
        "refs": deep_refs,
        "user": {
            "names": names,
            "seqs": seqs,
            "xs": Yu[:, 0].tolist(),
            "ys": Yu[:, 1].tolist(),
            "n": len(names),
            "groups": ["Sequence"]*len(names),
            "colors": ["#FFD700"]*len(names),
            "group_palette": {"Sequence": "#FFD700"},
            "meta": {"source": input_mode},
        }
    }

    rows = []
    for name, seq in items:
        feats = _compute_all_features(seq)
        if feats.get("length") is None:
            feats["length"] = len(clean_seq(seq))
        rows.append(_round_feature_row(_flatten_feature_row(name, seq, detect_alphabet_simple(seq), feats), nd=4))

    letters = "ACGU" if any("U" in (r.get("sequence", "").upper()) for r in rows) else "ACGT"
    request.session["extractor_results"] = rows
    request.session["extractor_csv_meta"] = {
        "letters": letters,
        "k2_labels": _lexicokeys(letters, 2),
        "k3_labels": _lexicokeys(letters, 3),
        "base_headers": ["Name", "Sequence", "Length", "GC %", "GC skew", "AT/AU skew"],
        "mnc_headers": [f"MNC {b}" for b in letters],
    }

    if Xi is not None and Xi.shape[0] and feat_names:
        if selected_feats:
            sel_idx = [i for i, k in enumerate(feat_names) if k in selected_feats]
            if not sel_idx:
                sel_idx = list(range(len(feat_names)))
        else:
            sel_idx = list(range(len(feat_names)))

        Xi_use = Xi[:, sel_idx]

        def vec_sel(r):
            v = []
            for i in sel_idx:
                k = feat_names[i]
                try:
                    v.append(float(r.get(k, 0.0)))
                except:
                    v.append(0.0)
            return v

        X_new = np.asarray([vec_sel(r) for r in rows], np.float32)
        Yu_feat = _knn_project(Xi_use, Yi, X_new, k=50)
    else:
        numeric = [k for k in rows[0].keys() if k not in ("name", "sequence", "alphabet")]
        X_new = np.asarray([[float(r.get(k, 0.0)) for k in numeric] for r in rows], np.float32)
        xsu, ysu = _tsne_2d(X_new.tolist())
        Yu_feat = np.column_stack([np.asarray(xsu, np.float32), np.asarray(ysu, np.float32)])

    interp_payload = {
        "refs": interp_refs,
        "user": {
            "names": names,
            "seqs": seqs,
            "xs": Yu_feat[:, 0].tolist(),
            "ys": Yu_feat[:, 1].tolist(),
            "n": len(names),
            "groups": ["Sequence"]*len(names),
            "colors": ["#FFD700"]*len(names),
            "group_palette": {"Sequence": "#FFD700"},
            "meta": {"source": input_mode},
        }
    }
    request.session["has_user_overlaid"] = True
    return _respond(deep_payload, interp_payload, has_user=True)

@require_GET
def feature_explorer_download_embeddings(request):
    data = request.session.get("feature_explorer_user_embeddings")
    if not data:
        return HttpResponse("No user embeddings to download.", status=400)
    arr = np.asarray(data, dtype=np.float32)
    buf = BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    resp = HttpResponse(buf.read(), content_type="application/octet-stream")
    resp["Content-Disposition"] = 'attachment; filename="user_embeddings.npy"'
    return resp


@require_GET
def feature_extractor_download(request):
    rows = request.session.get("extractor_results") or []
    meta = request.session.get("extractor_csv_meta") or {}
    if not rows:
        return HttpResponse("No results to download.", status=400)

    letters = meta.get("letters") or ("ACGU" if any("mnc_U" in r for r in rows) else "ACGT")
    k2_labels = meta.get("k2_labels") or _lexicokeys(letters, 2)
    k3_labels = meta.get("k3_labels") or _lexicokeys(letters, 3)

    base_headers = meta.get("base_headers") or ["Name", "Sequence", "Length", "GC %", "GC skew", "AT/AU skew"]
    mnc_headers = meta.get("mnc_headers") or [f"MNC {b}" for b in letters]

    base_key = {
        "Name": "name",
        "Sequence": "sequence",
        "Length": "length",
        "GC %": "gc_pct",
        "GC skew": "gc_skew",
        "AT/AU skew": "at_au_skew",
    }
    mnc_key = {f"MNC {b}": f"mnc_{b}" for b in letters}

    fieldnames = base_headers + mnc_headers + k2_labels + k3_labels
    out = StringIO()
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        rec = {h: r.get(base_key[h], "") for h in base_headers}
        rec.update({h: r.get(mnc_key[h], "") for h in mnc_headers})
        for lab in k2_labels:
            rec[lab] = r.get(f"k2_{lab}", "")
        for lab in k3_labels:
            rec[lab] = r.get(f"k3_{lab}", "")
        w.writerow(rec)
    resp = HttpResponse(out.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="features.csv"'
    return resp
