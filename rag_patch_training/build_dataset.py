#!/usr/bin/env python3
"""
Build training JSONL for RAG Patch Training via MMR retrieval.
==============================================================

For every image in the MRAG database, retrieves k reference images using
**Maximal Marginal Relevance (MMR)** with a **hybrid score** that blends
SigLIP image similarity (70 %) and text/caption similarity (30 %).

Retrieval settings (configurable via CLI):
    --mmr_lambda   0.9      (high relevance, mild diversity)
    --cosine_k     50       (initial shortlist size for MMR re-ranking)
    --image_weight 0.7      (70 % image, 30 % text in the hybrid score)
    --top_k        3        (number of final references per query)

Output
------
A JSONL file where each line is:
    {"image": "<abs_path>", "prompt": "<caption>", "rag_images": ["<abs1>", ...]}

Usage
-----
    conda activate nexus          # or rag_patch — needs faiss, transformers, torch
    python rag_patch_training/build_dataset.py \
        --db_dir  Nexus-Gen/mrag-db \
        --output  rag_patch_training/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import faiss
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────
#  MMR implementation
# ─────────────────────────────────────────────────────────────────────────

def mmr(
    query_vec: np.ndarray,          # (d,)     — query embedding
    candidate_vecs: np.ndarray,     # (N, d)   — candidate embeddings
    candidate_ids: np.ndarray,      # (N,)     — original DB ids
    top_k: int = 3,
    lam: float = 0.9,
) -> list[int]:
    """
    Maximal Marginal Relevance selection.

    Returns `top_k` DB indices from `candidate_ids` that maximise:
        MMR = λ · sim(query, doc) − (1 − λ) · max_{selected} sim(doc, doc_j)

    All similarities are cosine (vectors assumed L2-normalised).
    """
    # Cosine similarity between query and all candidates
    q_sim = candidate_vecs @ query_vec                       # (N,)
    # Pre-compute pairwise sims among candidates for diversity term
    cand_sim = candidate_vecs @ candidate_vecs.T             # (N, N)

    selected_idx: list[int] = []   # indices into candidate_vecs
    remaining = set(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        best_score = -1e9
        best_i = -1
        for i in remaining:
            relevance = q_sim[i]
            if selected_idx:
                diversity = max(cand_sim[i, j] for j in selected_idx)
            else:
                diversity = 0.0
            score = lam * relevance - (1.0 - lam) * diversity
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == -1:
            break
        selected_idx.append(best_i)
        remaining.discard(best_i)

    return [int(candidate_ids[i]) for i in selected_idx]


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAG Patch training dataset via MMR retrieval.",
    )
    parser.add_argument(
        "--db_dir", type=str,
        default="Nexus-Gen/mrag-db",
        help="Path to the MRAG database directory.",
    )
    parser.add_argument(
        "--output", type=str,
        default="rag_patch_training/dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of reference images per query.")
    parser.add_argument("--cosine_k", type=int, default=50,
                        help="Initial shortlist size for MMR re-ranking.")
    parser.add_argument("--mmr_lambda", type=float, default=0.9,
                        help="MMR trade-off (1=pure relevance, 0=pure diversity).")
    parser.add_argument("--image_weight", type=float, default=0.7,
                        help="Weight for image similarity (1 − this = text weight).")
    parser.add_argument("--max_entries", type=int, default=0,
                        help="Process at most this many DB entries (0 = all).")
    args = parser.parse_args()

    text_weight = 1.0 - args.image_weight

    db_dir = os.path.abspath(args.db_dir)
    images_dir = os.path.join(db_dir, "images")
    output_path = os.path.abspath(args.output)

    # ── Load metadata ────────────────────────────────────────────────────
    print("Loading metadata...")
    meta_jsonl = os.path.join(db_dir, "metadata.jsonl")
    meta_json  = os.path.join(db_dir, "metadata.json")
    if os.path.isfile(meta_jsonl):
        with open(meta_jsonl) as f:
            metadata: list[dict] = [json.loads(line) for line in f if line.strip()]
        print(f"  {len(metadata)} entries in metadata.jsonl")
    else:
        with open(meta_json) as f:
            metadata: list[dict] = json.load(f)
        print(f"  {len(metadata)} entries in metadata.json")

    # Build id → metadata lookup (metadata ids may not be contiguous)
    id_to_meta = {entry["id"]: entry for entry in metadata}

    # ── Load FAISS indices ───────────────────────────────────────────────
    print("Loading FAISS indices...")
    img_index = faiss.read_index(os.path.join(db_dir, "image.faiss"))
    txt_index = faiss.read_index(os.path.join(db_dir, "text.faiss"))
    n_db = img_index.ntotal
    d = img_index.d
    print(f"  {n_db} vectors, dim={d}")
    assert txt_index.ntotal == n_db, "Image/text index size mismatch"
    assert txt_index.d == d, "Image/text index dimension mismatch"

    # ── Reconstruct all vectors (needed for MMR pairwise sim) ────────────
    print("Reconstructing all embeddings from FAISS...")
    all_img_vecs = np.zeros((n_db, d), dtype=np.float32)
    all_txt_vecs = np.zeros((n_db, d), dtype=np.float32)
    for i in range(n_db):
        all_img_vecs[i] = img_index.reconstruct(i)
        all_txt_vecs[i] = txt_index.reconstruct(i)
    # Ensure they are L2-normalised (they should be, but just in case)
    norms_img = np.linalg.norm(all_img_vecs, axis=1, keepdims=True)
    norms_img[norms_img < 1e-8] = 1.0
    all_img_vecs /= norms_img
    norms_txt = np.linalg.norm(all_txt_vecs, axis=1, keepdims=True)
    norms_txt[norms_txt < 1e-8] = 1.0
    all_txt_vecs /= norms_txt
    print("  Done.")

    # ── Iterate: for each DB entry, retrieve references via MMR ──────────
    print(f"Building dataset (cosine_k={args.cosine_k}, "
          f"mmr_lambda={args.mmr_lambda}, top_k={args.top_k}, "
          f"image_weight={args.image_weight})...")

    # We search cosine_k + 1 because the query itself will be in the results
    search_k = args.cosine_k + 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    skipped = 0

    with open(output_path, "w") as fout:
        query_range = range(n_db)
        if args.max_entries > 0:
            query_range = range(min(args.max_entries, n_db))
        for query_id in tqdm(query_range, desc="Retrieval"):
            # Skip entries without metadata
            if query_id not in id_to_meta:
                skipped += 1
                continue

            meta = id_to_meta[query_id]
            img_path = os.path.join(images_dir, meta["filename"])
            if not os.path.isfile(img_path):
                skipped += 1
                continue
            caption = meta.get("caption", "")

            # ── Hybrid shortlist: blend image + text FAISS scores ────────
            q_img = all_img_vecs[query_id : query_id + 1]     # (1, d)
            q_txt = all_txt_vecs[query_id : query_id + 1]     # (1, d)

            img_scores, img_ids = img_index.search(q_img, search_k)
            txt_scores, txt_ids = txt_index.search(q_txt, search_k)

            # Merge candidates into a set (union of both shortlists)
            candidate_set: dict[int, float] = {}
            for rank in range(search_k):
                cid = int(img_ids[0, rank])
                if cid != query_id:
                    s = float(img_scores[0, rank]) * args.image_weight
                    candidate_set[cid] = candidate_set.get(cid, 0.0) + s

                cid = int(txt_ids[0, rank])
                if cid != query_id:
                    s = float(txt_scores[0, rank]) * text_weight
                    candidate_set[cid] = candidate_set.get(cid, 0.0) + s

            if len(candidate_set) == 0:
                skipped += 1
                continue

            # Sort by hybrid score, take top cosine_k
            sorted_cands = sorted(candidate_set.items(),
                                  key=lambda x: x[1], reverse=True)
            sorted_cands = sorted_cands[: args.cosine_k]

            cand_ids = np.array([c[0] for c in sorted_cands])
            # Build hybrid query vector and candidate vectors for MMR
            # Use the same image+text weighting for the MMR similarity
            q_hybrid = (q_img[0] * args.image_weight +
                        q_txt[0] * text_weight)
            q_hybrid /= max(np.linalg.norm(q_hybrid), 1e-8)

            cand_vecs = (all_img_vecs[cand_ids] * args.image_weight +
                         all_txt_vecs[cand_ids] * text_weight)
            cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            cand_norms[cand_norms < 1e-8] = 1.0
            cand_vecs /= cand_norms

            # ── MMR re-ranking ───────────────────────────────────────────
            selected_ids = mmr(
                query_vec=q_hybrid,
                candidate_vecs=cand_vecs,
                candidate_ids=cand_ids,
                top_k=args.top_k,
                lam=args.mmr_lambda,
            )

            # Resolve file paths for the selected references
            rag_paths: list[str] = []
            for ref_id in selected_ids:
                if ref_id in id_to_meta:
                    ref_file = os.path.join(
                        images_dir, id_to_meta[ref_id]["filename"]
                    )
                    if os.path.isfile(ref_file):
                        rag_paths.append(ref_file)
            if len(rag_paths) == 0:
                skipped += 1
                continue

            entry = {
                "image": img_path,
                "prompt": caption,
                "rag_images": rag_paths,
            }
            fout.write(json.dumps(entry) + "\n")
            written += 1

    print(f"\nDone!  wrote={written}  skipped={skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
