"""
Dataset for RAG Patch Training.

Each sample provides:
    - image       : (3, H, W) float32 tensor in [0, 1]   — target image  (JourneyDB)
    - prompt      : str                                   — text prompt   (JourneyDB)
    - rag_images  : (k, 3, H, W) float32 tensor in [0, 1] — reference images (mrag-db)

Data flow:
    1. Image & prompt are sourced from JourneyDB (journeydb_dir/metadata.jsonl).
    2. Each (image, prompt) pair is embedded with SigLIP and queried against
       the mrag-db FAISS indices to retrieve the top-k RAG reference images.
    3. Results are cached to ``cache_path`` so the expensive SigLIP + FAISS
       retrieval only runs once.

Cached JSONL format (one per line):
    {"image": "/abs/path.jpg", "prompt": "...", "rag_images": ["/abs/path1.jpg", ...]}
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────
#  MMR implementation  (same as build_dataset.py)
# ─────────────────────────────────────────────────────────────────────────

def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int = 3,
    lam: float = 0.9,
) -> list[int]:
    """Maximal Marginal Relevance selection (cosine, L2-normalised vectors)."""
    q_sim = candidate_vecs @ query_vec
    cand_sim = candidate_vecs @ candidate_vecs.T
    selected_idx: list[int] = []
    remaining = set(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        best_score, best_i = -1e9, -1
        for i in remaining:
            relevance = q_sim[i]
            diversity = max((cand_sim[i, j] for j in selected_idx), default=0.0)
            score = lam * relevance - (1.0 - lam) * diversity
            if score > best_score:
                best_score, best_i = score, i
        if best_i == -1:
            break
        selected_idx.append(best_i)
        remaining.discard(best_i)

    return [int(candidate_ids[i]) for i in selected_idx]


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    vecs /= norms
    return vecs


# ─────────────────────────────────────────────────────────────────────────
#  Dataset builder  (JourneyDB → SigLIP → FAISS query → JSONL cache)
# ─────────────────────────────────────────────────────────────────────────

def _build_dataset(
    journeydb_dir: str,
    rag_db_dir: str,
    num_rag_images: int = 3,
    cosine_k: int = 50,
    mmr_lambda: float = 0.9,
    image_weight: float = 0.7,
    siglip_model: str = "google/siglip-so400m-patch14-384",
    siglip_batch_size: int = 64,
    max_entries: int = 0,
) -> list[dict]:
    """
    Embed every JourneyDB entry with SigLIP, query the mrag-db FAISS
    indices for the top-k RAG references, and return the resulting dataset
    entries (ready to be serialised as JSONL).
    """
    import faiss
    from transformers import AutoProcessor, AutoModel

    text_weight = 1.0 - image_weight
    journeydb_dir = os.path.abspath(journeydb_dir)
    rag_db_dir = os.path.abspath(rag_db_dir)

    # ── Load JourneyDB metadata ─────────────────────────────────────────
    jdb_meta_path = os.path.join(journeydb_dir, "metadata.jsonl")
    if not os.path.isfile(jdb_meta_path):
        jdb_meta_path = os.path.join(journeydb_dir, "train_anno_realease_repath.jsonl")
    jdb_images_dir = os.path.join(journeydb_dir, "images")
    if not os.path.isdir(jdb_images_dir):
        jdb_images_dir = os.path.join(journeydb_dir, "imgs")
    print(f"Loading JourneyDB metadata from {jdb_meta_path} ...")
    jdb_entries = _load_jsonl(jdb_meta_path)
    if max_entries > 0:
        jdb_entries = jdb_entries[:max_entries]
    print(f"  {len(jdb_entries)} JourneyDB entries")

    # ── Load mrag-db metadata & FAISS indices ────────────────────────────
    rag_meta_path = os.path.join(rag_db_dir, "metadata.jsonl")
    rag_images_dir = os.path.join(rag_db_dir, "images")
    print(f"Loading mrag-db metadata from {rag_meta_path} ...")
    rag_metadata = _load_jsonl(rag_meta_path)
    rag_id_to_meta = {e["id"]: e for e in rag_metadata}

    print("Loading FAISS indices ...")
    img_index = faiss.read_index(os.path.join(rag_db_dir, "image.faiss"))
    txt_index = faiss.read_index(os.path.join(rag_db_dir, "text.faiss"))
    n_db = img_index.ntotal
    d = img_index.d
    print(f"  {n_db} vectors, dim={d}")

    # Reconstruct DB vectors (needed for MMR pairwise similarity)
    print("Reconstructing mrag-db embeddings for MMR ...")
    all_img_vecs = np.zeros((n_db, d), dtype=np.float32)
    all_txt_vecs = np.zeros((n_db, d), dtype=np.float32)
    for i in range(n_db):
        all_img_vecs[i] = img_index.reconstruct(i)
        all_txt_vecs[i] = txt_index.reconstruct(i)
    _l2_normalise(all_img_vecs)
    _l2_normalise(all_txt_vecs)

    # ── Load SigLIP ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP ({siglip_model}) on {device} ...")
    processor = AutoProcessor.from_pretrained(siglip_model)
    model = AutoModel.from_pretrained(siglip_model).to(device).eval()

    # ── Embed JourneyDB entries in batches ───────────────────────────────
    print("Embedding JourneyDB images & prompts with SigLIP ...")
    jdb_img_vecs_list: list[np.ndarray] = []
    jdb_txt_vecs_list: list[np.ndarray] = []
    valid_entries: list[dict] = []

    batch_images: list[Image.Image] = []
    batch_prompts: list[str] = []
    batch_entries: list[dict] = []

    def _flush_batch() -> None:
        if not batch_images:
            return
        # Image embeddings
        img_inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(**img_inputs).cpu().numpy()
        # Text embeddings
        txt_inputs = processor(
            text=batch_prompts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=64,
        ).to(device)
        with torch.no_grad():
            txt_emb = model.get_text_features(**txt_inputs).cpu().numpy()
        jdb_img_vecs_list.append(img_emb)
        jdb_txt_vecs_list.append(txt_emb)
        valid_entries.extend(batch_entries)

    for entry in tqdm(jdb_entries, desc="Preparing batches"):
        img_path = os.path.join(jdb_images_dir, entry.get("image_path") or entry.get("img_path", ""))
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        batch_images.append(img)
        batch_prompts.append(entry["prompt"])
        batch_entries.append({"image": img_path, "prompt": entry["prompt"]})
        if len(batch_images) >= siglip_batch_size:
            _flush_batch()
            batch_images, batch_prompts, batch_entries = [], [], []

    _flush_batch()  # remaining

    # Free SigLIP from GPU
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    jdb_img_vecs = _l2_normalise(np.concatenate(jdb_img_vecs_list, axis=0))
    jdb_txt_vecs = _l2_normalise(np.concatenate(jdb_txt_vecs_list, axis=0))
    print(f"  Embedded {len(valid_entries)} entries")

    # ── Retrieve RAG images for each JourneyDB entry ────────────────────
    print(f"Querying mrag-db (cosine_k={cosine_k}, mmr_lambda={mmr_lambda}, "
          f"top_k={num_rag_images}, image_weight={image_weight}) ...")
    search_k = cosine_k + 1
    dataset: list[dict] = []

    for i in tqdm(range(len(valid_entries)), desc="Retrieval"):
        q_img = jdb_img_vecs[i : i + 1]   # (1, d)
        q_txt = jdb_txt_vecs[i : i + 1]   # (1, d)

        img_scores, img_ids = img_index.search(q_img, search_k)
        txt_scores, txt_ids = txt_index.search(q_txt, search_k)

        # Merge candidates (hybrid score)
        candidate_set: dict[int, float] = {}
        for rank in range(search_k):
            cid = int(img_ids[0, rank])
            candidate_set[cid] = candidate_set.get(cid, 0.0) + float(img_scores[0, rank]) * image_weight

            cid = int(txt_ids[0, rank])
            candidate_set[cid] = candidate_set.get(cid, 0.0) + float(txt_scores[0, rank]) * text_weight

        if not candidate_set:
            continue

        sorted_cands = sorted(candidate_set.items(), key=lambda x: x[1], reverse=True)[:cosine_k]
        cand_ids = np.array([c[0] for c in sorted_cands])

        # Hybrid query vector for MMR
        q_hybrid = q_img[0] * image_weight + q_txt[0] * text_weight
        q_hybrid /= max(np.linalg.norm(q_hybrid), 1e-8)

        cand_vecs = all_img_vecs[cand_ids] * image_weight + all_txt_vecs[cand_ids] * text_weight
        _l2_normalise(cand_vecs)

        selected_ids = _mmr(q_hybrid, cand_vecs, cand_ids, num_rag_images, mmr_lambda)

        # Resolve file paths
        rag_paths: list[str] = []
        for ref_id in selected_ids:
            if ref_id in rag_id_to_meta:
                ref_file = os.path.join(rag_images_dir, rag_id_to_meta[ref_id]["filename"])
                if os.path.isfile(ref_file):
                    rag_paths.append(ref_file)

        if not rag_paths:
            continue

        dataset.append({
            "image": valid_entries[i]["image"],
            "prompt": valid_entries[i]["prompt"],
            "rag_images": rag_paths,
        })

    print(f"  Built {len(dataset)} entries")
    return dataset


# ─────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────

class RAGPatchDataset(Dataset):
    """
    Load (image, prompt, rag_images) triples.

    - **image** and **prompt** come from JourneyDB
      (``journeydb_dir/metadata.jsonl``).
    - **rag_images** are retrieved from the mrag-db via SigLIP + FAISS
      similarity search.
    - A cached JSONL file (``cache_path``) is produced on first run so
      the expensive embedding / retrieval step is only performed once.
    """

    def __init__(
        self,
        journeydb_dir: str,
        rag_db_dir: str,
        cache_path: str = "rag_patch_training/dataset.jsonl",
        num_rag_images: int = 3,
        image_size: int = 512,
        steps_per_epoch: int = 1000,
        center_crop: bool = True,
        random_flip: bool = False,
        # Retrieval settings (used only when building the cache)
        cosine_k: int = 50,
        mmr_lambda: float = 0.9,
        image_weight: float = 0.7,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        siglip_batch_size: int = 64,
        max_entries: int = 0,
    ) -> None:
        super().__init__()
        self.num_rag_images = num_rag_images
        self.steps_per_epoch = steps_per_epoch

        # ── Load or build the dataset manifest ───────────────────────────
        if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
            print(f"Loading cached dataset from {cache_path}")
            self.data = _load_jsonl(cache_path)
        else:
            print(f"Cache not found at {cache_path}, building dataset ...")
            self.data = _build_dataset(
                journeydb_dir=journeydb_dir,
                rag_db_dir=rag_db_dir,
                num_rag_images=num_rag_images,
                cosine_k=cosine_k,
                mmr_lambda=mmr_lambda,
                image_weight=image_weight,
                siglip_model=siglip_model,
                siglip_batch_size=siglip_batch_size,
                max_entries=max_entries,
            )
            # Persist cache
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as f:
                for entry in self.data:
                    f.write(json.dumps(entry) + "\n")
            print(f"Cached {len(self.data)} entries → {cache_path}")

        assert len(self.data) > 0, f"Empty dataset (cache: {cache_path})"

        # ── Image transforms ─────────────────────────────────────────────
        xforms = [transforms.Resize(image_size)]
        if center_crop:
            xforms.append(transforms.CenterCrop(image_size))
        else:
            xforms.append(transforms.RandomCrop(image_size))
        if random_flip:
            xforms.append(transforms.RandomHorizontalFlip())
        xforms.append(transforms.ToTensor())  # → [0, 1]
        self.transform = transforms.Compose(xforms)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # (3, H, W) in [0, 1]

    def __getitem__(self, idx: int) -> dict:
        entry = self.data[idx % len(self.data)]

        image = self._load_image(entry["image"])
        prompt = entry["prompt"]

        # Load k reference images  (pad with repeats if fewer available)
        rag_paths = entry.get("rag_images", [])
        if len(rag_paths) == 0:
            # Fallback: use the target image itself as a dummy reference
            rag_paths = [entry["image"]]
        while len(rag_paths) < self.num_rag_images:
            rag_paths.append(random.choice(rag_paths))
        rag_paths = rag_paths[: self.num_rag_images]

        rag_imgs = torch.stack([self._load_image(p) for p in rag_paths])
        # rag_imgs : (k, 3, H, W)

        return {
            "image": image,          # (3, H, W)
            "prompt": prompt,        # str
            "rag_images": rag_imgs,  # (k, 3, H, W)
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that stacks tensors and collects prompts as a list."""
    return {
        "image": torch.stack([b["image"] for b in batch]),         # (B, 3, H, W)
        "prompt": [b["prompt"] for b in batch],                     # list[str]
        "rag_images": torch.stack([b["rag_images"] for b in batch]), # (B, k, 3, H, W)
    }
