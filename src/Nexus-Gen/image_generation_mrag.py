"""
MRAG-Enhanced Image Generation using FLUX DiT (text-to-image, no AR step)
=========================================================================

Architecture
------------
  Baseline (--no_mrag --no_dual_stream):
      text → FLUX T5/CLIP encoding → FLUX DiT denoising → image

  MRAG + DualStream (--enable_mrag --use_dual_stream):
      text → FLUX T5/CLIP encoding              → base stream (e_0)
      retrieved images → VGG-19 style → projector → correction stream (S_phi)
      noise_pred = e_0 + correction_gate * S_phi → image

The DiT weights come from Nexus-Gen's generation_decoder.bin (the fine-tuned
FLUX DiT), and the LoRA correction weights from the RAG patch training.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure Nexus-Gen and DiffSynth are on the Python path
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFSYNTH = os.path.join(_DIR, "DiffSynth-Studio")
for _p in [_DIR, _DIFFSYNTH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffsynth import ModelManager                                       # noqa: E402
from diffsynth.models.flux_dit import FluxDiT                           # noqa: E402
from diffsynth.models.utils import load_state_dict as _load_state_dict  # noqa: E402
from diffsynth.pipelines.flux_image import lets_dance_flux              # noqa: E402
from modeling.decoder.generation_decoder import state_dict_converter    # noqa: E402
from modeling.decoder.pipelines import NexusGenGenerationPipeline       # noqa: E402

# ---------------------------------------------------------------------------
# VGG-19 style constants (must match rag_patch_training/model.py)
# ---------------------------------------------------------------------------
_VGG_LAYER_INDICES = {"relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20}
VGG_STYLE_DIM = (64 + 128 + 256 + 512) * 2   # 1920


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRAG-Enhanced Image Generation")

    # Core
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_path", type=str, default="result.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--embedded_guidance", type=float, default=3.5)
    parser.add_argument("--fp8_quantization", action="store_true", default=False)
    parser.add_argument("--enable_cpu_offload", action="store_true", default=True)
    parser.add_argument("--generation_decoder_path", type=str,
                        default="models/Nexus-GenV2/generation_decoder.bin")
    parser.add_argument("--flux_path", type=str, default="models")

    # MRAG retrieval
    parser.add_argument("--enable_mrag", action="store_true", default=False)
    parser.add_argument("--no_mrag", dest="enable_mrag", action="store_false")
    parser.add_argument("--mrag_db_path", type=str, default="mrag-db")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--mmr_lambda", type=float, default=0.9)
    parser.add_argument("--top_k_candidates", type=int, default=50)
    parser.add_argument("--image_weight", type=float, default=0.7,
                        help="Weight for image modality in retrieval (text_weight = 1 - image_weight)")
    parser.add_argument("--text_weight", type=float, default=None,
                        help="Weight for text modality in retrieval. Overrides --image_weight if set.")

    # Dual-stream correction
    parser.add_argument("--use_dual_stream", action="store_true", default=False)
    parser.add_argument("--no_dual_stream", dest="use_dual_stream", action="store_false")
    parser.add_argument("--stream2_weight", type=float, default=None,
                        help="Override correction_gate. Uses trained value if None.")
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None,
                        help="Path to LoRA checkpoint dir (contains lora/ and style_projectors.pt). "
                             "Defaults to latest workdirs/rag_patch/checkpoints/epoch=* dir.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# VGG-19 style extractor
# ---------------------------------------------------------------------------
class VGG19StyleExtractor(nn.Module):
    """Frozen VGG-19 that returns feature maps at relu{1,2,3,4}_1."""

    def __init__(self) -> None:
        super().__init__()
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        )
        max_idx = max(_VGG_LAYER_INDICES.values())
        self.slices = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        self.target_indices = sorted(_VGG_LAYER_INDICES.values())
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        features = []
        for i, layer in enumerate(self.slices):
            x = layer(x)
            if i in self.target_indices:
                features.append(x)
        return features


def _extract_style_embedding(
    vgg19: VGG19StyleExtractor,
    images: torch.Tensor,       # (k, 3, H, W) in [0, 1]
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:              # (1, 1920)
    """Average VGG-19 style stats over k reference images."""
    with torch.no_grad():
        feats = vgg19(images.to(device=device, dtype=torch.float32))
    stats: list[torch.Tensor] = []
    for feat in feats:
        stats.append(feat.mean(dim=[2, 3]))
        stats.append(feat.std(dim=[2, 3], unbiased=False))
    style_vec = torch.cat(stats, dim=-1)           # (k, 1920)
    return style_vec.mean(dim=0, keepdim=True).to(dtype=dtype)  # (1, 1920)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def load_pipeline(args: argparse.Namespace) -> NexusGenGenerationPipeline:
    """Load FLUX generation pipeline (text-to-image, no AR)."""
    torch_dtype = torch.bfloat16
    mm_device = "cpu" if args.enable_cpu_offload else args.device

    model_manager = ModelManager(torch_dtype=torch_dtype, device=mm_device)
    model_manager.load_models([
        f"{args.flux_path}/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
        f"{args.flux_path}/FLUX/FLUX.1-dev/text_encoder_2",
        f"{args.flux_path}/FLUX/FLUX.1-dev/ae.safetensors",
    ])

    # Load Nexus-Gen fine-tuned DiT weights from generation_decoder.bin
    state_dict = _load_state_dict(args.generation_decoder_path)
    dit_sd = {
        k.replace("pipe.dit.", ""): v
        for k, v in state_dict.items()
        if not k.startswith("adapter.")
    }
    del state_dict

    FluxDiT.state_dict_converter = staticmethod(state_dict_converter)
    model_manager.load_model_from_single_file(
        args.generation_decoder_path,
        state_dict=dit_sd,
        model_names=["flux_dit"],
        model_classes=[FluxDiT],
        model_resource="diffusers",
    )
    del dit_sd

    dit_dtype = torch.float8_e4m3fn if (args.fp8_quantization and not args.use_dual_stream) else torch_dtype
    if args.fp8_quantization and args.use_dual_stream:
        print("[Setup] FP8 disabled for dual-stream (LoRA requires bfloat16 matmul)")
    model_manager.model[-1].to(dtype=dit_dtype)

    pipe = NexusGenGenerationPipeline.from_model_manager(
        model_manager, device=args.device
    )
    if args.enable_cpu_offload:
        pipe.enable_cpu_offload()
    if args.fp8_quantization and not args.use_dual_stream:
        pipe.dit.quantize()

    return pipe


# ---------------------------------------------------------------------------
# Dual-stream LoRA loading
# ---------------------------------------------------------------------------
def _find_latest_checkpoint() -> str | None:
    """Return the most recent epoch checkpoint dir under workdirs/rag_patch/."""
    pattern = os.path.join(
        _DIR, "..", "workdirs", "rag_patch", "checkpoints", "epoch=*"
    )
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def load_dual_stream_components(
    pipe: NexusGenGenerationPipeline,
    checkpoint_dir: str,
    stream2_weight: float | None,
    device: str,
    dtype: torch.dtype,
):
    """Wrap pipe.dit with LoRA and load style projectors.

    Returns (patch_dit, style_to_context, style_to_pooled, correction_gate).
    """
    from peft import PeftModel

    lora_dir = os.path.join(checkpoint_dir, "lora")
    style_proj_path = os.path.join(checkpoint_dir, "style_projectors.pt")

    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")
    if not os.path.isfile(style_proj_path):
        raise FileNotFoundError(f"style_projectors.pt not found: {style_proj_path}")

    print(f"[MRAG] Loading LoRA adapters from {lora_dir} …")
    patch_dit = PeftModel.from_pretrained(pipe.dit, lora_dir)
    patch_dit.eval()

    print(f"[MRAG] Loading style projectors from {style_proj_path} …")
    proj_sd = torch.load(style_proj_path, map_location="cpu")

    style_to_context = nn.Sequential(
        nn.Linear(VGG_STYLE_DIM, 4096), nn.SiLU(), nn.Linear(4096, 4096)
    ).to(device=device, dtype=dtype)
    style_to_pooled = nn.Sequential(
        nn.Linear(VGG_STYLE_DIM, 768), nn.SiLU(), nn.Linear(768, 768)
    ).to(device=device, dtype=dtype)
    style_to_context.load_state_dict(proj_sd["style_to_context"])
    style_to_pooled.load_state_dict(proj_sd["style_to_pooled"])

    if stream2_weight is not None:
        correction_gate = torch.tensor([stream2_weight], dtype=dtype, device=device)
        print(f"[MRAG] correction_gate overridden to {stream2_weight}")
    else:
        # Try style_projectors.pt first (saved by new LoRASaveCallback)
        if "correction_gate" in proj_sd:
            gate_val = float(proj_sd["correction_gate"].squeeze())
            correction_gate = torch.tensor([gate_val], dtype=dtype, device=device)
            print(f"[MRAG] correction_gate loaded from style_projectors.pt: {gate_val:.6f}")
        else:
            # Fall back to Lightning checkpoint
            ckpt_glob = os.path.join(
                _DIR, "..", "workdirs", "rag_patch_v2", "logs",
                "version_*", "checkpoints", "epoch=*.ckpt"
            )
            ckpt_files = sorted(glob.glob(ckpt_glob))
            if ckpt_files:
                sd = torch.load(ckpt_files[-1], map_location="cpu").get("state_dict", {})
                gate_val = float(sd.get("correction_gate", torch.tensor([1.0]))[0])
                correction_gate = torch.tensor([gate_val], dtype=dtype, device=device)
                print(f"[MRAG] correction_gate loaded from Lightning ckpt: {gate_val:.6f}")
            else:
                correction_gate = torch.tensor([1.0], dtype=dtype, device=device)
                print("[MRAG] correction_gate not found; defaulting to 1.0")

    return patch_dit, style_to_context, style_to_pooled, correction_gate


# ---------------------------------------------------------------------------
# MRAG retrieval
# ---------------------------------------------------------------------------
def retrieve_references(args: argparse.Namespace) -> list[Image.Image]:
    """Load MRAGRetriever and return top-k PIL images for the prompt."""
    from retrieve_mrag import MRAGRetriever

    text_weight = args.text_weight if args.text_weight is not None else (1.0 - args.image_weight)
    print(f"[MRAG] Loading retriever from {args.mrag_db_path} …")
    retriever = MRAGRetriever(db_path=args.mrag_db_path, text_weight=text_weight)

    print(f"[MRAG] Retrieving top-{args.top_k} references for prompt …")
    results = retriever.retrieve(
        args.prompt,
        k=args.top_k,
        lambda_mmr=args.mmr_lambda,
        k_candidates=args.top_k_candidates,
    )

    # Save RAG reference images next to the result
    rag_ref_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.result_path)), "mrag_references"
    )
    os.makedirs(rag_ref_dir, exist_ok=True)

    images: list[Image.Image] = []
    for i, r in enumerate(results):
        img_path = r["image_path"]
        if not os.path.isfile(img_path):
            print(f"[MRAG]   WARNING: file not found: {img_path}")
            continue
        images.append(Image.open(img_path).convert("RGB"))
        dst = os.path.join(rag_ref_dir, f"ref_{i:02d}_{r['filename']}")
        shutil.copy(img_path, dst)
        cap = (r["caption"] or "")[:60]
        print(f"[MRAG]   #{i+1} score={r['retrieval_score']:.4f}  {r['filename']}  {cap!r}")

    print(f"[MRAG] {len(images)} reference image(s) loaded.")
    return images


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _images_to_vgg_tensor(
    images: list[Image.Image],
    size: int = 224,
) -> torch.Tensor:
    """Resize PIL images and stack to (k, 3, size, size) float32 tensor in [0,1]."""
    import torchvision.transforms.functional as TF
    tensors = [TF.to_tensor(img.resize((size, size))) for img in images]
    return torch.stack(tensors)   # (k, 3, 224, 224)


def generate_baseline(
    args: argparse.Namespace,
    pipe: NexusGenGenerationPipeline,
) -> Image.Image:
    """Standard FLUX text-to-image generation."""
    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            embedded_guidance=args.embedded_guidance,
            progress_bar_cmd=tqdm,
        )
    return image


def generate_mrag(
    args: argparse.Namespace,
    pipe: NexusGenGenerationPipeline,
    retrieved_images: list[Image.Image],
) -> Image.Image:
    """Dual-stream MRAG generation using LoRA correction."""
    dtype = torch.bfloat16
    device = args.device

    ckpt_dir = args.lora_checkpoint_dir or _find_latest_checkpoint()
    if ckpt_dir is None:
        print("[MRAG] WARNING: no LoRA checkpoint found; falling back to baseline")
        return generate_baseline(args, pipe)
    print(f"[MRAG] Using LoRA checkpoint: {ckpt_dir}")

    patch_dit, style_to_context, style_to_pooled, correction_gate = \
        load_dual_stream_components(pipe, ckpt_dir, args.stream2_weight, device, dtype)

    # --- Prepare style conditioning from retrieved images ---
    vgg19 = VGG19StyleExtractor().to(device=device)
    rag_tensors = _images_to_vgg_tensor(retrieved_images)       # (k, 3, 224, 224)
    style_emb = _extract_style_embedding(vgg19, rag_tensors, device, dtype)  # (1, 1920)

    with torch.no_grad():
        style_context = style_to_context(style_emb).unsqueeze(1)   # (1, 1, 4096)
        style_pooled = style_to_pooled(style_emb)                  # (1, 768)
    style_text_ids = torch.zeros(1, 1, 3, device=device, dtype=dtype)

    # --- Prepare denoising ---
    pipe.scheduler.set_timesteps(args.num_inference_steps, 1.0)
    latents, _ = pipe.prepare_latents(None, args.height, args.width, args.seed, False, 128, 64)

    # Text embeddings for Stream 1 (base)
    pipe.load_models_to_device(["text_encoder_1", "text_encoder_2"])
    text_emb = pipe.encode_prompt(args.prompt, t5_sequence_length=512)
    pipe.load_models_to_device([])
    torch.cuda.empty_cache()

    extra_input = pipe.prepare_extra_input(latents, guidance=args.embedded_guidance)

    # Load DiT (including LoRA weights that were merged into pipe.dit in-place)
    pipe.load_models_to_device(["dit"])

    print("[MRAG] Running dual-stream denoising …")
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
        t = timestep.unsqueeze(0).to(device)

        with torch.no_grad():
            # Stream 1 — base DiT (LoRA disabled)
            with patch_dit.disable_adapter():
                e_0 = lets_dance_flux(
                    dit=patch_dit,
                    hidden_states=latents,
                    timestep=t,
                    **text_emb,
                    **extra_input,
                )

            # Stream 2 — LoRA-patched DiT with style conditioning
            S_phi = lets_dance_flux(
                dit=patch_dit,
                hidden_states=latents,
                timestep=t,
                prompt_emb=style_context,
                pooled_prompt_emb=style_pooled,
                text_ids=style_text_ids,
                **extra_input,
            )

            # Additive correction: e_0 + gate * S_phi
            # Matches the training fusion formula (model.py: v_pred = e_0 + gate * S_phi).
            # S_phi is supervised to predict the residual velocity (v* - e_0),
            # so noise_pred = e_0 + gate*(v* - e_0) → v* as gate → 1.
            noise_pred = e_0 + correction_gate * S_phi

            latents = pipe.scheduler.step(
                noise_pred.detach(), pipe.scheduler.timesteps[progress_id], latents
            )

    pipe.load_models_to_device(["vae_decoder"])
    with torch.no_grad():
        image = pipe.decode_image(latents.detach())
    pipe.load_models_to_device([])
    return image


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    result_dir = os.path.dirname(os.path.abspath(args.result_path))
    os.makedirs(result_dir, exist_ok=True)

    print(f"[Setup] Prompt      : {args.prompt[:80]}")
    print(f"[Setup] enable_mrag : {args.enable_mrag}")
    print(f"[Setup] dual_stream : {args.use_dual_stream}")
    print(f"[Setup] result_path : {args.result_path}")

    # --- Load FLUX pipeline ---
    print("[Setup] Loading FLUX pipeline …")
    pipe = load_pipeline(args)

    # --- MRAG retrieval ---
    retrieved_images: list[Image.Image] = []
    if args.enable_mrag:
        retrieved_images = retrieve_references(args)

    # --- Generate ---
    if args.use_dual_stream and retrieved_images:
        print("[Generate] MRAG dual-stream generation …")
        image = generate_mrag(args, pipe, retrieved_images)
    else:
        if args.use_dual_stream and not retrieved_images:
            print("[Generate] WARNING: dual_stream requested but no references retrieved — using baseline")
        print("[Generate] Baseline generation …")
        image = generate_baseline(args, pipe)

    image.save(args.result_path)
    print(f"[Done] Image saved to {args.result_path}")
