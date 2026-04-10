"""
build_journeydb_dataset.py
==========================
Download and prepare 100k samples from JourneyDB for training.

Strategy:
  - Discord CDN URLs in train_anno.jsonl are 100% expired (verified 0/50 alive).
  - The actual images live in HuggingFace shards: data/train/imgs/000.tgz, 001.tgz, ...
  - Each annotation's img_path field (e.g. "./000/filename.jpg") tells us which shard.
  - Annotation archive (~small) fetched via hf_hub_download (already cached).
  - Image shards (~15 GB each) fetched via wget -c (resumable, robust for large files).
  - Images are streamed out of each shard with tarfile — no full extraction to disk.
  - Shard 000 alone has ~84k annotated images; shard 001 covers the rest to 100k.

Env vars:
  JOURNEYDB_OUT   output directory (default: ./journeydb_dataset)
  MAX_SHARDS      max shards to process (default: 50; set to 1 for a quick test)
  SHARD_CACHE     where to store downloaded shard tarballs
                  (default: ~/.cache/huggingface/journeydb_shards)
"""

import io
import json
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_REPO   = "JourneyDB/JourneyDB"
ANNO_FILE      = "data/train/train_anno_realease_repath.jsonl.tgz"

TARGET_SAMPLES = 100_000
MAX_SHARDS     = int(os.environ.get("MAX_SHARDS", "50"))
OUTPUT_DIR     = Path(os.environ.get("JOURNEYDB_OUT", "./journeydb_dataset"))
IMAGE_DIR      = OUTPUT_DIR / "images"
META_FILE      = OUTPUT_DIR / "metadata.jsonl"
CSV_FILE       = OUTPUT_DIR / "train.csv"

SHARD_CACHE    = Path(os.environ.get("SHARD_CACHE",
                      str(Path.home() / ".cache/huggingface/journeydb_shards")))

SAVE_EXT       = ".jpg"
JPEG_QUALITY   = 95
LOG_EVERY      = 1_000

HF_TOKEN_FILE  = Path.home() / ".cache/huggingface/token"
HF_TOKEN       = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else ""

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_annotations(tgz_path: str):
    """Yield parsed JSON dicts from train_anno.jsonl inside the annotation .tgz."""
    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".jsonl"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def build_anno_index(anno_tgz: str, shard_id: str) -> dict:
    """
    Return dict: bare_filename -> prompt
    for all annotations whose img_path belongs to shard_id (e.g. "000").
    """
    index = {}
    for record in iter_annotations(anno_tgz):
        img_path = record.get("img_path", "")
        prompt   = record.get("prompt", "")
        if not img_path or not isinstance(prompt, str) or not prompt.strip():
            continue
        # img_path looks like "./000/<uuid>.jpg" (repath version)
        parts = Path(img_path.lstrip("./")).parts  # ('000', 'uuid.jpg')
        if len(parts) == 2 and parts[0] == shard_id:
            index[parts[1]] = prompt.strip()
    return index


def wget_download(url: str, dest: Path) -> bool:
    """Download url to dest using wget -c (resumable). Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-c",               # resume partial downloads
        "--tries=5",
        "--timeout=120",
        "--show-progress",
        "-O", str(dest),
        url,
    ]
    if HF_TOKEN:
        cmd += ["--header", f"Authorization: Bearer {HF_TOKEN}"]

    log.info("wget -> %s", dest)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("wget failed (exit %d) for %s", result.returncode, url)
        return False
    return True


def shard_url(shard_id: str) -> str:
    """Return the HuggingFace resolve URL for an image shard."""
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}"
        f"/resolve/main/data/train/imgs/{shard_id}.tgz"
    )


def process_shard(shard_tgz: Path, anno_index: dict,
                  meta_fh, csv_fh, collected: int) -> int:
    """
    Stream images from shard_tgz, match against anno_index, save as JPEG.
    Returns updated collected count.
    """
    shard_name = shard_tgz.stem
    log.info("Streaming images from shard %s ...", shard_name)

    with tarfile.open(shard_tgz, "r:gz") as tf:
        for member in tf:
            if collected >= TARGET_SAMPLES:
                break
            if not member.isfile():
                continue

            bare = Path(member.name).name
            prompt = anno_index.get(bare)
            if prompt is None:
                continue

            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                img.load()
                if img.size[0] < 64 or img.size[1] < 64:
                    continue
            except (UnidentifiedImageError, Exception):
                continue

            safe_name = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in bare
            )
            if not safe_name.lower().endswith(SAVE_EXT):
                safe_name = safe_name.rsplit(".", 1)[0] + SAVE_EXT

            out_shard = collected // 10_000
            rel_path  = f"{out_shard:02d}/{safe_name}"
            dest      = IMAGE_DIR / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            try:
                img.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            except Exception:
                continue

            meta_fh.write(json.dumps({"image_path": rel_path, "prompt": prompt},
                                     ensure_ascii=False) + "\n")
            csv_prompt = '"' + prompt.replace('"', '""') + '"'
            csv_fh.write(f"{rel_path},{csv_prompt}\n")

            collected += 1
            if collected % LOG_EVERY == 0:
                log.info("Progress: %d / %d collected", collected, TARGET_SAMPLES)

    return collected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_CACHE.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  JourneyDB dataset preparation")
    log.info("  Target     : %d samples", TARGET_SAMPLES)
    log.info("  Max shards : %d", MAX_SHARDS)
    log.info("  Output     : %s", OUTPUT_DIR.resolve())
    log.info("  Shard cache: %s", SHARD_CACHE)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: annotation archive (already cached from previous run)
    # ------------------------------------------------------------------
    log.info("Fetching annotation archive (uses cache if present)...")
    try:
        anno_tgz = hf_hub_download(
            repo_id=DATASET_REPO, filename=ANNO_FILE, repo_type="dataset"
        )
        log.info("Annotation archive: %s", anno_tgz)
    except Exception as e:
        log.error("Failed to fetch annotation archive: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: iterate shards
    # ------------------------------------------------------------------
    collected = 0

    meta_fh = META_FILE.open("w", encoding="utf-8")
    csv_fh  = CSV_FILE.open("w", encoding="utf-8")
    csv_fh.write("image_path,prompt\n")

    try:
        for shard_idx in range(MAX_SHARDS):
            if collected >= TARGET_SAMPLES:
                break

            shard_id = f"{shard_idx:03d}"

            # Build annotation index for this shard only
            log.info("Building annotation index for shard %s ...", shard_id)
            anno_index = build_anno_index(anno_tgz, shard_id)
            log.info("  -> %d annotated images in shard %s", len(anno_index), shard_id)
            if not anno_index:
                continue

            # Download shard via wget (resumable, skips if already fully downloaded)
            shard_local = SHARD_CACHE / f"{shard_id}.tgz"
            shard_ok = False
            if shard_local.exists() and shard_local.stat().st_size > 1_000_000:
                # Quick gzip integrity check to catch truncated downloads
                result = subprocess.run(["gzip", "-t", str(shard_local)],
                                        capture_output=True)
                if result.returncode == 0:
                    log.info("Shard %s already cached (%s), skipping download.",
                             shard_id, shard_local)
                    shard_ok = True
                else:
                    log.warning("Shard %s cache is corrupt, re-downloading.", shard_id)
                    shard_local.unlink()

            if not shard_ok:
                log.info("Downloading shard %s via wget (~15 GB)...", shard_id)
                if not wget_download(shard_url(shard_id), shard_local):
                    log.error("Skipping shard %s due to download failure.", shard_id)
                    continue

            # Process shard
            try:
                collected = process_shard(shard_local, anno_index, meta_fh, csv_fh, collected)
            except EOFError:
                log.error("Shard %s is truncated/corrupt — deleting and skipping.", shard_id)
                shard_local.unlink(missing_ok=True)
                continue
            log.info("After shard %s: %d / %d collected", shard_id, collected, TARGET_SAMPLES)

    finally:
        meta_fh.close()
        csv_fh.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  Done!")
    log.info("  Samples collected : %d", collected)
    log.info("  Metadata JSONL    : %s", META_FILE)
    log.info("  CSV               : %s", CSV_FILE)
    log.info("=" * 60)

    if collected < TARGET_SAMPLES:
        log.warning("Only collected %d / %d samples.", collected, TARGET_SAMPLES)
        sys.exit(2)


if __name__ == "__main__":
    main()
