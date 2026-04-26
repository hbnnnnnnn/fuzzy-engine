#!/bin/bash
#SBATCH --job-name=rag-patch-test
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/media02/nthuy/ndbao/logs/slurm_rag_patch_test_%A_%a.out
#SBATCH --error=/media02/nthuy/ndbao/logs/slurm_rag_patch_test_%A_%a.err
#SBATCH --array=0-4%2

# Controlled rag_patch smoke test.
# Compares the base model against the trained rag_patch checkpoint on the same
# five prompts and the same retrieved references.

set -euo pipefail

module purge
source ~/miniconda3/bin/activate
conda activate nexus

echo "Installing dependencies..."
pip install faiss-gpu scikit-learn --quiet
echo "Dependencies installed."

cd /media02/nthuy/ndbao

BASE_ROOT="/media02/nthuy/ndbao/src/Nexus-Gen"
if [ -d "${BASE_ROOT}/mrag-db" ]; then
    MRAG_DB_PATH="${BASE_ROOT}/mrag-db"
elif [ -d "${BASE_ROOT}/mrag-db-orgcap" ]; then
    MRAG_DB_PATH="${BASE_ROOT}/mrag-db-orgcap"
elif [ -d "${BASE_ROOT}/mrag-db_orgcap" ]; then
    MRAG_DB_PATH="${BASE_ROOT}/mrag-db_orgcap"
else
    MRAG_DB_PATH="${BASE_ROOT}/mrag-db"
fi
RAG_PATCH_ROOT="/media02/nthuy/ndbao/experiments/workdirs/rag_patch"
RAG_PATCH_EXPORT_DIR="${RAG_PATCH_ROOT}/checkpoints/epoch=4-step=625"
RAG_PATCH_ZERO_DIR="${RAG_PATCH_ROOT}/lightning_logs/version_64043/checkpoints/epoch=4-step=625.ckpt"
OUTPUT_DIR="rag_patch_results_controlled"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${OUTPUT_DIR}/ground_truth"
mkdir -p "${OUTPUT_DIR}/references"
mkdir -p "${OUTPUT_DIR}/reports"
mkdir -p "${OUTPUT_DIR}/baseline"
mkdir -p "${OUTPUT_DIR}/patched"

case $TASK_ID in
  0)
    DB_IMAGE_FILENAME="007824.webp"
    PROMPT="A couple embracing on a tropical beach at sunset, golden light reflecting on the ocean waves, palm trees silhouetted against an orange and pink sky"
    SEED=100
    ;;
  1)
    DB_IMAGE_FILENAME="000281.webp"
    PROMPT="A beautiful three-tiered wedding cake with pastel colors, decorated with fresh flowers and elegant gold leaf accents, on a white tablecloth"
    SEED=101
    ;;
  2)
    DB_IMAGE_FILENAME="016425.webp"
    PROMPT="A snowy mountain landscape with snow-covered peaks under a clear blue sky, alpine trees in the foreground, crisp winter sunlight"
    SEED=102
    ;;
  3)
    DB_IMAGE_FILENAME="001457.webp"
    PROMPT="A golden retriever dog with a fluffy coat sitting in a green grassy field, bright eyes, happy expression, warm sunlight"
    SEED=103
    ;;
  4)
    DB_IMAGE_FILENAME="007406.webp"
    PROMPT="A cozy wooden cabin in a dense forest with warm lights glowing from the windows, a front porch, surrounded by tall pine trees"
    SEED=104
    ;;
esac

SAFE_NAME=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-40)

echo "============================================================"
echo "  RAG Patch Controlled Test"
echo "  Task  : ${TASK_ID} / 5"
echo "  Prompt: ${PROMPT}"
echo "  DB    : ${DB_IMAGE_FILENAME}"
echo "  Seed  : ${SEED}"
echo "============================================================"

python3 -u - "${BASE_ROOT}" "${MRAG_DB_PATH}" "${RAG_PATCH_ZERO_DIR}" "${RAG_PATCH_EXPORT_DIR}" "${OUTPUT_DIR}" "${DB_IMAGE_FILENAME}" "${PROMPT}" "${SEED}" "${TASK_ID}" <<'PYEOF'
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

BASE_ROOT = Path(sys.argv[1])
MRAG_DB_PATH = Path(sys.argv[2])
RAG_PATCH_ZERO_DIR = Path(sys.argv[3])
RAG_PATCH_EXPORT_DIR = Path(sys.argv[4])
OUTPUT_DIR = Path(sys.argv[5])
DB_IMAGE_FILENAME = sys.argv[6]
PROMPT = sys.argv[7]
SEED = int(sys.argv[8])
TASK_ID = int(sys.argv[9])

# Path setup — Nexus-Gen modules + the ndbao/src parent so we can import
# rag_patch_training.{model,pipeline}.
sys.path.insert(0, str(BASE_ROOT))
sys.path.insert(0, str(BASE_ROOT / "DiffSynth-Studio"))
sys.path.insert(0, str(BASE_ROOT.parent))

from retrieve_mrag import MRAGRetriever
from rag_patch_training.model import RAGPatchTrainer, NEXUS_GEN_EN_TEMPLATE
from rag_patch_training.pipeline import (
    NexusGenRAGPatchPipeline,
    load_rag_patch_state_dict,
)


# ──────────────────────────────────────────────────────────────────────
#  AR forward — Qwen2.5-VL → 81-token image_embed
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_image_embed(trainer: RAGPatchTrainer, prompt: str, device: str) -> torch.Tensor:
    """Run Qwen2.5-VL.generate + frozen adapter to produce a (1, 81, 4096)
    image embedding for the given text prompt.  Mirrors the AR call in
    ``trainer.forward`` and ``Nexus-Gen/image_generation.py``.
    """
    trainer.qwen_ar.to(device)
    trainer.ar_adapter.to(device)
    formatted = NEXUS_GEN_EN_TEMPLATE.format(prompt)
    messages = [{"role": "user", "content": [{"type": "text", "text": formatted}]}]
    text = trainer.qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ar_inputs = trainer.qwen_processor(
        text=[text], padding=True, return_tensors="pt",
    ).to(device)
    grid_thw = torch.tensor([[1, 18, 18]], device=device)
    ar_outputs = trainer.qwen_ar.generate(
        **ar_inputs,
        max_new_tokens=1024,
        return_dict_in_generate=True,
        generation_image_grid_thw=grid_thw,
    )
    image_embed = trainer.ar_adapter(
        ar_outputs["output_image_embeddings"].to(dtype=trainer.pipe.torch_dtype)
    )                                                            # (1, 81, 4096)
    # Offload AR back to CPU; adapter is small but offload too for parity.
    trainer.qwen_ar.cpu()
    trainer.ar_adapter.cpu()
    torch.cuda.empty_cache()
    return image_embed


# ──────────────────────────────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────────────────────────────

device = "cuda:0" if torch.cuda.is_available() else "cpu"
retriever = MRAGRetriever(db_path=str(MRAG_DB_PATH), device=device)

print("[INIT] Instantiating RAGPatchTrainer (this loads Qwen, FLUX DiT, VAE, CLIP, VGG19) …")
model = RAGPatchTrainer(
    nexgen_decoder_path="models/Nexus-GenV2/generation_decoder.bin",
    pretrained_text_encoder_path="models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    pretrained_vae_path="models/FLUX/FLUX.1-dev/ae.safetensors",
    nexus_gen_root=str(BASE_ROOT),
    torch_dtype_str="bf16",
)
model.eval()
model.to(device)

# Try to load the trained patch weights.  If the DeepSpeed ZeRO directory
# exists we use it; otherwise the run still proceeds with patch_dit at
# its untrained init (LoRA delta = 0), which is useful as a sanity check
# (patched output should equal base output bit-exactly).
load_mode = "untrained"
if (RAG_PATCH_ZERO_DIR / "latest").is_file():
    print(f"[LOAD] Reconstructing DeepSpeed checkpoint from {RAG_PATCH_ZERO_DIR}")
    info = load_rag_patch_state_dict(model, str(RAG_PATCH_ZERO_DIR))
    print(f"[LOAD] missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    load_mode = "deepspeed_zero"
elif RAG_PATCH_EXPORT_DIR.is_dir():
    # Fallback: exported per-epoch checkpoint with separate adapter.safetensors
    # + style_projectors.pt.  NOTE: this format pre-dates F3's per-layer
    # ModuleList style_to_context, so projector keys may not match — we
    # load with strict=False and warn.
    import safetensors.torch
    adapter_path = RAG_PATCH_EXPORT_DIR / "lora" / "adapter_model.safetensors"
    projectors_path = RAG_PATCH_EXPORT_DIR / "style_projectors.pt"
    if adapter_path.is_file():
        adapter_state = safetensors.torch.load_file(str(adapter_path), device="cpu")
        model.patch_dit.load_state_dict(adapter_state, strict=False)
        print(f"[LOAD] LoRA adapters from {adapter_path}")
    if projectors_path.is_file():
        projector_state = torch.load(projectors_path, map_location="cpu")
        sc = model.style_to_context.load_state_dict(
            projector_state.get("style_to_context", {}), strict=False,
        )
        sp = model.style_to_pooled.load_state_dict(
            projector_state.get("style_to_pooled", {}), strict=False,
        )
        print(f"[LOAD] style projectors  | sc.missing={len(sc.missing_keys)} sp.missing={len(sp.missing_keys)}")
    load_mode = "export_compat"
else:
    print("[LOAD] No checkpoint found — running with untrained patch (LoRA delta ≈ 0).")

# Build the inference pipeline.  Shares modules with the trainer.
pipe = NexusGenRAGPatchPipeline.from_trainer(model, strength=1.0)


# ──────────────────────────────────────────────────────────────────────
#  Retrieve refs + prepare tensors
# ──────────────────────────────────────────────────────────────────────

target_path = MRAG_DB_PATH / "images" / DB_IMAGE_FILENAME
target_image = Image.open(target_path).convert("RGB")

raw_refs = retriever.retrieve(PROMPT, k=5, lambda_mmr=0.95, k_candidates=50)
refs = []
for item in raw_refs:
    if item.get("filename") == DB_IMAGE_FILENAME:
        continue
    refs.append(item)
    if len(refs) == 3:
        break
if len(refs) < 3:
    raise RuntimeError(f"Need 3 references, got {len(refs)} for prompt: {PROMPT}")

# Resize refs to match the target image's resolution; convert to a tensor
# in [0, 1] of shape (1, k, 3, H, W) — the format the patch pipeline and
# trainer.forward expect for ``rag_images``.
TARGET_SIZE = target_image.size  # (W, H)

def to_unit_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32)
    arr = arr.view(image.size[1], image.size[0], 3).permute(2, 0, 1) / 255.0
    return arr  # (3, H, W) in [0, 1]

ref_tensors = torch.stack(
    [to_unit_tensor(Image.open(item["image_path"])) for item in refs], dim=0,
).unsqueeze(0)                                                 # (1, k, 3, H, W)

# Persist artifacts
ground_truth_dir = OUTPUT_DIR / "ground_truth"
references_dir = OUTPUT_DIR / "references" / f"task{TASK_ID}_{Path(DB_IMAGE_FILENAME).stem}"
baseline_dir = OUTPUT_DIR / "baseline"
patched_dir = OUTPUT_DIR / "patched"
references_dir.mkdir(parents=True, exist_ok=True)

target_save_path = ground_truth_dir / f"task{TASK_ID}_{Path(DB_IMAGE_FILENAME).stem}.png"
target_image.save(target_save_path)
for index, ref in enumerate(refs, start=1):
    Image.open(ref["image_path"]).convert("RGB").save(
        references_dir / f"ref{index}_{ref['filename']}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Generate AR image_embed once, then run baseline + patched
# ──────────────────────────────────────────────────────────────────────

print(f"[AR] Computing 81-token image_embed for prompt …")
image_embed = compute_image_embed(model, PROMPT, device)
print(f"[AR] image_embed.shape = {tuple(image_embed.shape)}")

W, H = TARGET_SIZE
print(f"[GEN] baseline (no rag_images) seed={SEED} {W}x{H}")
baseline_image = pipe(
    prompt="",                # ignored when image_embed is provided
    image_embed=image_embed,
    rag_images=None,
    height=H, width=W,
    num_inference_steps=30,
    cfg_scale=3.0,
    embedded_guidance=3.5,
    seed=SEED,
)

print(f"[GEN] patched (with {len(refs)} rag_images) seed={SEED}")
patched_image = pipe(
    prompt="",
    image_embed=image_embed,
    rag_images=ref_tensors,
    height=H, width=W,
    num_inference_steps=30,
    cfg_scale=3.0,
    embedded_guidance=3.5,
    seed=SEED,
)

baseline_path = baseline_dir / f"task{TASK_ID}_{Path(DB_IMAGE_FILENAME).stem}.png"
patched_path = patched_dir / f"task{TASK_ID}_{Path(DB_IMAGE_FILENAME).stem}.png"
baseline_image.save(baseline_path)
patched_image.save(patched_path)


# ──────────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────────

# Pixel-space delta gives a quick scalar to confirm the patch is doing
# *something* without having to eyeball the images.  L1 in [0,1] across
# all pixels.  Used as a sanity number, not a quality metric.
import numpy as np
b_arr = np.asarray(baseline_image, dtype=np.float32) / 255.0
p_arr = np.asarray(patched_image, dtype=np.float32) / 255.0
mean_abs_diff = float(np.mean(np.abs(b_arr - p_arr)))
gate_value = float(model.correction_gate.detach().cpu().item())

report_path = OUTPUT_DIR / "reports" / f"task{TASK_ID}_{Path(DB_IMAGE_FILENAME).stem}.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    "\n".join(
        [
            "==============================================================",
            f"RAG Patch Controlled Test - Task {TASK_ID}",
            "==============================================================",
            f"Load mode:           {load_mode}",
            f"Date:                {datetime.now()}",
            f"Prompt:              {PROMPT}",
            f"DB Source Image:     {DB_IMAGE_FILENAME}",
            f"Seed:                {SEED}",
            f"Resolution (WxH):    {W}x{H}",
            f"Target image:        {target_save_path}",
            f"Reference dir:       {references_dir}",
            f"Baseline image:      {baseline_path}",
            f"Patched image:       {patched_path}",
            f"correction_gate:     {gate_value:.6f}",
            f"strength:            {pipe.strength:.6f}",
            f"|baseline-patched|:  {mean_abs_diff:.6f}  (mean per-pixel L1 in [0,1])",
            "==============================================================",
        ]
    )
)

print(f"[RESULT] baseline saved to : {baseline_path}")
print(f"[RESULT] patched  saved to : {patched_path}")
print(f"[RESULT] mean |Δpixel|     : {mean_abs_diff:.6f}")
print(f"[RESULT] report            : {report_path}")
PYEOF

echo "============================================================"
echo "  DONE: Task ${TASK_ID} complete."
echo "============================================================"