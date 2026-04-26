#!/bin/bash
#SBATCH --job-name=mrag-retrieval-test
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_retrieval_test_%j.out
#SBATCH --error=slurm_retrieval_test_%j.err

# =============================================================================
# MRAG-DB Retrieval Test  (self-contained — all Python inlined, no .py needed)
# Tests the hybrid (text.faiss + image.faiss) + MMR retrieval pipeline
# built on top of SigLIP embeddings over mrag-db.
#
# 5 tests:
#   1. Nature / outdoor landscape   (text query)
#   2. Food / object                (text query)
#   3. Sports / event               (text query)
#   4. Abstract / concept           (text query)
#   5. Query-by-example             (image query — uses images/000000.webp)
# =============================================================================

MINICONDA_PATH="/media02/nthuy/miniconda3"
NEXUS_ROOT="/media02/nthuy/ndbao"
if [ -n "${DB_PATH:-}" ]; then
    DB_PATH="${DB_PATH}"
elif [ -d "${NEXUS_ROOT}/Nexus-Gen/mrag-db" ]; then
    DB_PATH="${NEXUS_ROOT}/Nexus-Gen/mrag-db"
elif [ -d "${NEXUS_ROOT}/Nexus-Gen/mrag-db-orgcap" ]; then
    DB_PATH="${NEXUS_ROOT}/Nexus-Gen/mrag-db-orgcap"
elif [ -d "${NEXUS_ROOT}/Nexus-Gen/mrag-db_orgcap" ]; then
    DB_PATH="${NEXUS_ROOT}/Nexus-Gen/mrag-db_orgcap"
else
    DB_PATH="${NEXUS_ROOT}/Nexus-Gen/mrag-db"
fi

# ── Banner (printed before anything that could hang) ──────────────────────────
echo "============================================================"
echo "  MRAG-DB Retrieval Test"
echo "  Node    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Date    : $(date)"
echo "  DB path : ${DB_PATH}"
echo "============================================================"
echo ""

# ── Environment ───────────────────────────────────────────────────────────────
echo "[ENV] Purging modules..."
module purge
echo "[ENV] Activating conda env nexus_mrag..."
source "${MINICONDA_PATH}/bin/activate" nexus_mrag
echo "[ENV] Python      : $(which python3)"
echo "[ENV] Env         : ${CONDA_DEFAULT_ENV}"
echo "[ENV] torch       : $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "[ENV] CUDA avail  : $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "[ENV] faiss       : $(python3 -c 'import faiss; print(faiss.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "[ENV] transformers: $(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo ""

# ── DB sanity check ───────────────────────────────────────────────────────────
echo "[DB] Checking DB files..."
for f in "${DB_PATH}/image.faiss" "${DB_PATH}/text.faiss"; do
    if [ -f "$f" ]; then
        echo "[DB]   FOUND   : $f  ($(du -sh "$f" | cut -f1))"
    else
        echo "[DB]   MISSING : $f"
    fi
done
if [ -f "${DB_PATH}/metadata.json" ]; then
    echo "[DB]   FOUND   : ${DB_PATH}/metadata.json  ($(du -sh "${DB_PATH}/metadata.json" | cut -f1))"
elif [ -f "${DB_PATH}/metadata.jsonl" ]; then
    echo "[DB]   FOUND   : ${DB_PATH}/metadata.jsonl  ($(du -sh "${DB_PATH}/metadata.jsonl" | cut -f1))"
else
    echo "[DB]   MISSING : ${DB_PATH}/metadata.json and ${DB_PATH}/metadata.jsonl"
fi
echo "[DB] Image files : $(ls "${DB_PATH}/images/" 2>/dev/null | wc -l) files in ${DB_PATH}/images/"
echo ""

# ── Inline Python test suite ──────────────────────────────────────────────────
echo "[RUN] Starting Python test suite..."
echo "------------------------------------------------------------"

python3 -u - "${DB_PATH}" <<'PYEOF'
# ============================================================================
#  MRAG-DB Retriever + 5-test smoke suite  (inlined into SLURM script)
# ============================================================================

import os, sys, json, textwrap, time
from typing import Dict, List, Union, Literal

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = sys.argv[1]
MODEL_NAME = "google/siglip-so400m-patch14-384"
K          = 3
LAMBDA_MMR = 0.9
K_CANDS    = 100
TEXT_W     = 0.5   # weight for text-index in hybrid (image-index = 1 - TEXT_W)

YELLOW = "\033[93m"; GREEN = "\033[92m"; RED = "\033[91m"
RESET  = "\033[0m";  BOLD  = "\033[1m";  CYAN = "\033[96m"

def _log(stage, msg):
    print(f"{CYAN}[{stage}]{RESET} {msg}", flush=True)

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

def _minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = s.min(), s.max()
    return np.ones_like(s) if hi - lo < 1e-9 else (s - lo) / (hi - lo)

# ── Imports ───────────────────────────────────────────────────────────────────
_log("IMPORT", "importing faiss...")
import faiss
_log("IMPORT", f"faiss {faiss.__version__}")

_log("IMPORT", "importing torch...")
import torch
_log("IMPORT", f"torch {torch.__version__}  |  CUDA={torch.cuda.is_available()}")

_log("IMPORT", "importing PIL...")
from PIL import Image
_log("IMPORT", "PIL OK")

_log("IMPORT", "importing transformers...")
from transformers import AutoProcessor, AutoModel
_log("IMPORT", "transformers OK")
print(flush=True)

# ── Load FAISS indices ────────────────────────────────────────────────────────
_log("FAISS", f"loading image index  →  {DB_PATH}/image.faiss")
t0 = time.time()
img_index = faiss.read_index(os.path.join(DB_PATH, "image.faiss"))
_log("FAISS", f"image index: {img_index.ntotal:,} vectors  ({time.time()-t0:.1f}s)")

_log("FAISS", f"loading text index   →  {DB_PATH}/text.faiss")
t0 = time.time()
txt_index = faiss.read_index(os.path.join(DB_PATH, "text.faiss"))
_log("FAISS", f"text index : {txt_index.ntotal:,} vectors  ({time.time()-t0:.1f}s)")

# ── Load metadata ─────────────────────────────────────────────────────────────
meta_json = os.path.join(DB_PATH, "metadata.json")
meta_jsonl = os.path.join(DB_PATH, "metadata.jsonl")
if os.path.isfile(meta_json):
    _log("META", f"loading metadata  →  {meta_json}")
elif os.path.isfile(meta_jsonl):
    _log("META", f"loading metadata  →  {meta_jsonl}")
else:
    raise FileNotFoundError(
        f"No metadata file found in {DB_PATH}; expected metadata.json or metadata.jsonl"
    )
t0 = time.time()
if os.path.isfile(meta_json):
    with open(meta_json) as f:
        meta_list = json.load(f)
else:
    with open(meta_jsonl) as f:
        meta_list = [json.loads(line) for line in f if line.strip()]
metadata = {m["id"]: m for m in meta_list}
_log("META", f"{len(metadata):,} entries loaded  ({time.time()-t0:.1f}s)")
print(flush=True)

# ── Load SigLIP ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
_log("MODEL", f"loading SigLIP processor  ({MODEL_NAME})...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
_log("MODEL", f"processor ready  ({time.time()-t0:.1f}s)")

_log("MODEL", f"loading SigLIP model  →  {device}...")
t0 = time.time()
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
_log("MODEL", f"model ready  ({time.time()-t0:.1f}s)  "
              f"params={sum(p.numel() for p in model.parameters()):,}")
print(flush=True)

# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_texts(texts):
    inputs = processor(text=texts, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return _normalize(feats.cpu().numpy())

def embed_images(images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return _normalize(feats.cpu().numpy())

# ── Hybrid search ─────────────────────────────────────────────────────────────
def hybrid_search(query_emb):
    txt_raw, txt_ids = txt_index.search(query_emb, K_CANDS)
    img_raw, img_ids = img_index.search(query_emb, K_CANDS)
    txt_raw = txt_raw[0]; txt_ids = txt_ids[0]
    img_raw = img_raw[0]; img_ids = img_ids[0]

    _log("HYBRID", f"  text-index  raw scores: min={txt_raw.min():.4f}  max={txt_raw.max():.4f}  mean={txt_raw.mean():.4f}")
    _log("HYBRID", f"  image-index raw scores: min={img_raw.min():.4f}  max={img_raw.max():.4f}  mean={img_raw.mean():.4f}")
    _log("HYBRID", f"  applying per-modality min-max normalisation (absorbs scale gap)...")

    txt_norm = _minmax(txt_raw)
    img_norm = _minmax(img_raw)
    scores: Dict[int, float] = {}
    for s, i in zip(txt_norm, txt_ids):
        if i >= 0: scores[i] = scores.get(i, 0.0) + TEXT_W * float(s)
    for s, i in zip(img_norm, img_ids):
        if i >= 0: scores[i] = scores.get(i, 0.0) + (1 - TEXT_W) * float(s)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    _log("HYBRID", f"  unique candidates after merge: {len(ranked)}")
    return ranked

# ── MMR re-ranking ────────────────────────────────────────────────────────────
def mmr_rerank(candidates, query_emb):
    if len(candidates) <= K:
        return candidates
    cand_ids  = [c[0] for c in candidates]
    score_map = {c[0]: c[1] for c in candidates}

    valid_ids, vecs = [], []
    for idx in cand_ids:
        try:
            v = img_index.reconstruct(int(idx))
            valid_ids.append(idx); vecs.append(v)
        except Exception:
            pass
    if not vecs:
        return candidates[:K]

    emb_matrix = _normalize(np.stack(vecs))   # (N, D)
    q_vec      = _normalize(query_emb)[0]     # (D,)
    rel_sims   = emb_matrix @ q_vec           # (N,)

    _log("MMR", f"  reconstructed {len(valid_ids)} candidate embeddings")
    _log("MMR", f"  relevance scores: min={rel_sims.min():.4f}  max={rel_sims.max():.4f}")
    _log("MMR", f"  running greedy MMR (λ={LAMBDA_MMR})...")

    selected, remaining = [], list(range(len(valid_ids)))
    while len(selected) < K and remaining:
        if not selected:
            best = max(remaining, key=lambda i: rel_sims[i])
        else:
            sel_embs = emb_matrix[selected]
            best, best_val = None, -1e9
            for i in remaining:
                score = (LAMBDA_MMR * float(rel_sims[i])
                         - (1 - LAMBDA_MMR) * float(np.max(emb_matrix[i] @ sel_embs.T)))
                if score > best_val:
                    best_val = score; best = i
        _log("MMR", f"    selected rank {len(selected)+1}: db_id={valid_ids[best]}  "
                    f"rel={rel_sims[best]:.4f}  hybrid={score_map[valid_ids[best]]:.5f}")
        selected.append(best); remaining.remove(best)

    return [(valid_ids[i], score_map[valid_ids[i]]) for i in selected]

# ── Top-level retrieve ────────────────────────────────────────────────────────
def retrieve(query, query_type="auto"):
    if query_type == "auto":
        query_type = "image" if isinstance(query, Image.Image) else "text"
    _log("ENCODE", f"query_type={query_type}  encoding with SigLIP...")
    t0 = time.time()
    query_emb = embed_texts([query]) if query_type == "text" else embed_images([query])
    _log("ENCODE", f"embedding shape={query_emb.shape}  norm={float(np.linalg.norm(query_emb)):.4f}  ({time.time()-t0:.2f}s)")

    _log("HYBRID", "searching both FAISS indices...")
    t0 = time.time()
    candidates = hybrid_search(query_emb)
    _log("HYBRID", f"hybrid search done  ({time.time()-t0:.2f}s)")

    _log("MMR", f"re-ranking {len(candidates)} candidates → top-{K}...")
    t0 = time.time()
    final = mmr_rerank(candidates, query_emb)
    _log("MMR", f"MMR done  ({time.time()-t0:.2f}s)")

    img_base = os.path.join(DB_PATH, "images")
    return [{
        "id":              item_id,
        "filename":        metadata.get(int(item_id), {}).get("filename"),
        "image_path":      os.path.join(img_base, metadata.get(int(item_id), {}).get("filename", "")),
        "caption":         metadata.get(int(item_id), {}).get("caption"),
        "url":             metadata.get(int(item_id), {}).get("url"),
        "aesthetic_score": metadata.get(int(item_id), {}).get("score"),
        "retrieval_score": round(score, 6),
    } for item_id, score in final]

# ── Pretty-print results ──────────────────────────────────────────────────────
def print_results(tag, query_str, results):
    sep = "─" * 70
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}{YELLOW}[{tag}]{RESET}")
    print(f"  Query   : {str(query_str)[:100]}")
    print(f"  Returns : {len(results)} result(s)  (top-{K}, λ={LAMBDA_MMR}, hybrid text_w={TEXT_W})")
    print(sep)
    for i, r in enumerate(results, 1):
        cap   = textwrap.shorten(r["caption"] or "(no caption)", width=75)
        exist = f"{GREEN}YES{RESET}" if os.path.isfile(r["image_path"]) else f"{RED}NO{RESET}"
        print(f"  {GREEN}#{i}{RESET}  id={r['id']:>6}  "
              f"hybrid_score={r['retrieval_score']:.5f}  "
              f"aesthetic={r['aesthetic_score']:.2f}")
        print(f"       caption : {cap}")
        print(f"       url     : {(r['url'] or 'N/A')[:80]}")
        print(f"       file    : {r['image_path']}")
        print(f"       on disk : {exist}")
    print(sep, flush=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Run 5 tests
# ═════════════════════════════════════════════════════════════════════════════
passed = 0; failed = 0

def run_test(tag, query, query_type="auto"):
    global passed, failed
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Running {tag}{RESET}")
    print(f"{'='*70}", flush=True)
    try:
        t0      = time.time()
        results = retrieve(query, query_type=query_type)
        elapsed = time.time() - t0
        qs      = "<PIL.Image>" if isinstance(query, Image.Image) else query
        print_results(tag, qs, results)

        assert results,                            "empty result list"
        assert len(results) <= K,                  f"got {len(results)} > k={K}"
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)),          "MMR returned duplicate ids"
        for r in results:
            assert os.path.isfile(r["image_path"]), f"file missing: {r['image_path']}"

        print(f"\n{GREEN}[PASS]{RESET} {tag}  (total={elapsed:.1f}s)")
        passed += 1
    except Exception as exc:
        import traceback
        print(f"\n{RED}[FAIL]{RESET} {tag}: {exc}")
        traceback.print_exc()
        failed += 1

# ── Test 1 ── nature / landscape ──────────────────────────────────────────────
run_test("TEST-01  nature/landscape",
         "wooden bridge over a calm lake surrounded by trees in summer")

# ── Test 2 ── food / object ───────────────────────────────────────────────────
run_test("TEST-02  food/object",
         "beautiful bouquet of pink peonies and roses")

# ── Test 3 ── sports / event ──────────────────────────────────────────────────
run_test("TEST-03  sports/event",
         "soccer player celebrating championship trophy victory")

# ── Test 4 ── abstract / concept ─────────────────────────────────────────────
run_test("TEST-04  abstract/concept",
         "warm candle light glowing in winter darkness holiday atmosphere")

# ── Test 5 ── image-as-query ──────────────────────────────────────────────────
sample = os.path.join(DB_PATH, "images", "000000.webp")
if os.path.isfile(sample):
    run_test("TEST-05  image-as-query (000000.webp)",
             Image.open(sample).convert("RGB"), query_type="image")
else:
    print(f"\n{YELLOW}[SKIP]{RESET} TEST-05: {sample} not found")

# ── Summary ───────────────────────────────────────────────────────────────────
total  = passed + failed
colour = GREEN if failed == 0 else (YELLOW if passed > 0 else RED)
print(f"\n{BOLD}{'='*70}{RESET}")
print(f"{colour}{BOLD}RESULTS: {passed}/{total} passed  |  {failed} failed{RESET}")
print(f"{BOLD}{'='*70}{RESET}\n", flush=True)

sys.exit(0 if failed == 0 else 1)
PYEOF

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  All retrieval tests PASSED — exit code 0"
else
    echo "  One or more tests FAILED — exit code ${EXIT_CODE}"
fi
echo "  Date: $(date)"
echo "============================================================"

exit $EXIT_CODE
