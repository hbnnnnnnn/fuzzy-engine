"""
MRAG-DB Retriever
=================
Hybrid retrieval over mrag-db-orgcap using:
  - Dual FAISS indices (image embeddings + text/caption embeddings)
  - SigLIP (google/siglip-so400m-patch14-384) for query encoding
  - Score normalization to handle the scale gap between
      text-text  (cosine ~0.5-1.0)  and  image-text  (cosine ~0.1-0.4)
  - MMR (Maximal Marginal Relevance) re-ranking for final top-k selection

Usage:
    from retrieve_mrag import MRAGRetriever

    retriever = MRAGRetriever()
    results = retriever.retrieve("a golden retriever playing in the snow", k=3)
    for r in results:
        print(r["caption"], r["retrieval_score"])
"""

import os
import json
from typing import List, Union, Literal, Dict, Tuple, Any

import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "mrag-db-orgcap")
MODEL_SIGLIP    = "google/siglip-so400m-patch14-384"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows of a 2D array (no-op if already unit-norm)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors / norms


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Map an array to [0, 1] via min-max scaling."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class MRAGRetriever:
    """
    Hybrid FAISS + MMR retriever for mrag-db-orgcap.

    Parameters
    ----------
    db_path : str
        Directory that contains ``image.faiss``, ``text.faiss`` and
        ``metadata.json``.
    model_name : str
        HuggingFace model ID for SigLIP (must match the one used at build
        time).
    device : str | None
        ``"cuda"`` or ``"cpu"``.  Autodetected when *None*.
    text_weight : float
        Weight for the text-index score in the hybrid combination
        (image-index weight = ``1 - text_weight``).  Both scores are
        min-max-normalised before weighting so the scale gap between
        modalities is absorbed.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        model_name: str = MODEL_SIGLIP,
        device: str | None = None,
        text_weight: float = 0.5,
    ) -> None:
        self.db_path     = db_path
        self.text_weight = text_weight
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- FAISS indices ------------------------------------------------
        img_idx_path = os.path.join(db_path, "image.faiss")
        txt_idx_path = os.path.join(db_path, "text.faiss")
        print(f"[MRAGRetriever] Loading FAISS indices from {db_path} …")
        self.img_index: faiss.Index = faiss.read_index(img_idx_path)
        self.txt_index: faiss.Index = faiss.read_index(txt_idx_path)
        print(
            f"[MRAGRetriever] image index: {self.img_index.ntotal} vectors | "
            f"text index: {self.txt_index.ntotal} vectors"
        )

        # ---- Metadata (supports both metadata.json and metadata.jsonl) -----
        meta_json  = os.path.join(db_path, "metadata.json")
        meta_jsonl = os.path.join(db_path, "metadata.jsonl")
        if os.path.isfile(meta_json):
            with open(meta_json, "r") as f:
                meta_list = json.load(f)
        elif os.path.isfile(meta_jsonl):
            with open(meta_jsonl, "r") as f:
                meta_list = [json.loads(line) for line in f if line.strip()]
        else:
            raise FileNotFoundError(
                f"Neither metadata.json nor metadata.jsonl found in {db_path}"
            )
        self.metadata: Dict[int, Dict] = {m["id"]: m for m in meta_list}

        # ---- SigLIP model -------------------------------------------------
        print(f"[MRAGRetriever] Loading SigLIP ({model_name}) on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print("[MRAGRetriever] Ready.")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised SigLIP text embeddings, shape (N, D)."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return _normalize(feats.cpu().numpy())

    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return L2-normalised SigLIP image embeddings, shape (N, D)."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return _normalize(feats.cpu().numpy())

    def _encode_query(
        self,
        query: Union[str, Image.Image],
        query_type: Literal["text", "image"],
    ) -> np.ndarray:
        if query_type == "text":
            return self._embed_texts([query])   # (1, D)
        else:
            return self._embed_images([query])  # (1, D)

    # ------------------------------------------------------------------
    # Candidate reconstruction (for MMR pairwise similarity)
    # ------------------------------------------------------------------
    def _reconstruct_batch(self, ids: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Reconstruct image embeddings from the FAISS index for a list of IDs.
        Returns only the IDs for which reconstruction succeeded.
        """
        valid_ids, vecs = [], []
        for idx in ids:
            try:
                v = self.img_index.reconstruct(int(idx))  # (D,)
                valid_ids.append(idx)
                vecs.append(v)
            except Exception:
                pass
        if vecs:
            return valid_ids, _normalize(np.stack(vecs))  # (N, D)
        return [], np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------
    def _hybrid_search(
        self,
        query_emb: np.ndarray,
        k_candidates: int,
    ) -> List[Tuple[int, float]]:
        """
        Search both FAISS indices, normalise their scores individually
        (to correct for the image-text vs text-text scale mismatch), then
        combine with ``text_weight`` and return a ranked candidate list.

        Parameters
        ----------
        query_emb : np.ndarray, shape (1, D)
        k_candidates : int   number of results to fetch from each index

        Returns
        -------
        List of (db_id, combined_score) sorted descending.
        """
        # ---- query both indices -----------------------------------------
        txt_scores_raw, txt_ids = self.txt_index.search(query_emb, k_candidates)
        img_scores_raw, img_ids = self.img_index.search(query_emb, k_candidates)

        txt_scores_raw = txt_scores_raw[0]   # (k,)
        txt_ids        = txt_ids[0]
        img_scores_raw = img_scores_raw[0]
        img_ids        = img_ids[0]

        # ---- per-modality min-max normalisation -------------------------
        # This maps each modality's raw cosine scores to [0, 1] so that
        # the naturally-higher text-text values don't dominate the image-text
        # values after combining.
        txt_scores_norm = _minmax(txt_scores_raw)
        img_scores_norm = _minmax(img_scores_raw)

        # ---- accumulate per item ----------------------------------------
        scores_map: Dict[int, float] = {}

        for score, idx in zip(txt_scores_norm, txt_ids):
            if idx < 0:
                continue
            scores_map[idx] = scores_map.get(idx, 0.0) + self.text_weight * float(score)

        img_w = 1.0 - self.text_weight
        for score, idx in zip(img_scores_norm, img_ids):
            if idx < 0:
                continue
            scores_map[idx] = scores_map.get(idx, 0.0) + img_w * float(score)

        # ---- sort --------------------------------------------------------
        ranked = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        return ranked  # [(id, score), …]

    # ------------------------------------------------------------------
    # MMR re-ranking
    # ------------------------------------------------------------------
    def _mmr_rerank(
        self,
        candidates: List[Tuple[int, float]],
        query_emb: np.ndarray,
        k: int,
        lambda_mmr: float,
    ) -> List[Tuple[int, float]]:
        """
        Maximal Marginal Relevance selection.

        MMR score:
            λ · Sim(dᵢ, q)  −  (1−λ) · max_{dⱼ ∈ S} Sim(dᵢ, dⱼ)

        Parameters
        ----------
        candidates  : List[(id, combined_score)]  pre-ranked by hybrid search
        query_emb   : np.ndarray (1, D)
        k           : int  final number of results
        lambda_mmr  : float  0 = max diversity, 1 = max relevance (default 0.9)
        """
        if len(candidates) <= k:
            return candidates

        cand_ids   = [c[0] for c in candidates]
        score_map  = {c[0]: c[1] for c in candidates}

        valid_ids, emb_matrix = self._reconstruct_batch(cand_ids)  # (N, D)

        if len(valid_ids) == 0:
            return candidates[:k]

        # Relevance = cosine sim between each candidate and the query
        q_vec    = _normalize(query_emb)[0]          # (D,)
        rel_sims = emb_matrix @ q_vec                # (N,)  cosine similarity

        id_to_pos = {vid: i for i, vid in enumerate(valid_ids)}

        selected_pos: List[int] = []
        remaining_pos: List[int] = list(range(len(valid_ids)))

        while len(selected_pos) < k and remaining_pos:
            if not selected_pos:
                # first pick: highest relevance
                best = max(remaining_pos, key=lambda i: rel_sims[i])
            else:
                sel_embs = emb_matrix[selected_pos]  # (S, D)
                best, best_score = None, -1e9

                for i in remaining_pos:
                    relevance = float(rel_sims[i])
                    # max cosine sim to already-selected
                    diversity_penalty = float(np.max(emb_matrix[i] @ sel_embs.T))
                    mmr = lambda_mmr * relevance - (1.0 - lambda_mmr) * diversity_penalty
                    if mmr > best_score:
                        best_score = mmr
                        best = i

            selected_pos.append(best)
            remaining_pos.remove(best)

        return [(valid_ids[i], score_map[valid_ids[i]]) for i in selected_pos]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: Union[str, Image.Image],
        k: int = 3,
        lambda_mmr: float = 0.9,
        k_candidates: int = 100,
        query_type: Literal["text", "image", "auto"] = "auto",
        image_dir_override: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k images from the MRAG database.

        Parameters
        ----------
        query          : str  or  PIL.Image.Image
        k              : int        number of final results (default 3)
        lambda_mmr     : float      MMR relevance/diversity trade-off (0.9)
        k_candidates   : int        initial candidate pool size
        query_type     : "text" | "image" | "auto"
                         "auto" infers from the query type.
        image_dir_override : override the image directory path if needed

        Returns
        -------
        List of result dicts with keys:
            id, filename, image_path, caption, url, aesthetic_score,
            retrieval_score
        """
        # ---- infer query type -------------------------------------------
        if query_type == "auto":
            query_type = "image" if isinstance(query, Image.Image) else "text"

        # ---- encode query -----------------------------------------------
        query_emb = self._encode_query(query, query_type)  # (1, D)

        # ---- hybrid retrieval -------------------------------------------
        candidates = self._hybrid_search(query_emb, k_candidates)
        if not candidates:
            return []

        # ---- MMR re-ranking ---------------------------------------------
        final = self._mmr_rerank(candidates, query_emb, k=k, lambda_mmr=lambda_mmr)

        # ---- build result list ------------------------------------------
        img_base = image_dir_override or os.path.join(self.db_path, "images")
        results  = []
        for item_id, score in final:
            meta = self.metadata.get(int(item_id), {})
            results.append(
                {
                    "id":              item_id,
                    "filename":        meta.get("filename"),
                    "image_path":      os.path.join(img_base, meta.get("filename", "")),
                    "caption":         meta.get("caption"),
                    "url":             meta.get("url"),
                    "aesthetic_score": meta.get("score"),
                    "retrieval_score": round(score, 6),
                }
            )
        return results


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="MRAG-DB text query retrieval")
    parser.add_argument("query", help="Text query string")
    parser.add_argument("--db",    default=DEFAULT_DB_PATH)
    parser.add_argument("--k",     type=int,   default=3)
    parser.add_argument("--lambda_mmr", type=float, default=0.9)
    parser.add_argument("--k_candidates", type=int, default=100)
    args = parser.parse_args()

    retriever = MRAGRetriever(db_path=args.db)
    results   = retriever.retrieve(
        args.query,
        k=args.k,
        lambda_mmr=args.lambda_mmr,
        k_candidates=args.k_candidates,
    )
    pprint.pprint(results)
