"""
RAG-Conditioned Additive Patch Training for FLUX DiT  (Stream 2)
=================================================================

Trains a LoRA-based correction stream (patch_dit) on top of a frozen
base FLUX text-to-image diffusion model (base_dit), conditioned on
VGG-19 style features extracted from retrieved reference images (RAG).

Architecture
------------
  Stream 1 (frozen):  base_dit  →  e_0   (base velocity prediction)
  Stream 2 (LoRA):    patch_dit →  S_phi (additive residual)
  Train loss:         MSE(S_phi, v* − e_0.detach())   [residual regression]
  Inference fusion:   v_pred = e_0 + strength · S_phi  (strength is a buffer,
                                                        default 1.0; not trained)

Important implementation notes vs. task description
----------------------------------------------------
1. FLUX uses *flow matching* with velocity prediction (v = ε − x₀),
   **not** standard ε-prediction.  The correct training target is the
   velocity v* = ε − x₀, not raw noise ε.

2. Nexus-GenV2 conditions the DiT on an **81-token** visual embedding
   produced by a Qwen2.5-VL autoregressive model + a frozen adapter
   (Linear(3584→4096) → LN → ReLU → Linear(4096→4096) → LN).  This is
   the conditioning distribution the DiT was fine-tuned on, so the
   base forward (Stream 1) MUST use this path — not FLUX's native
   T5-XXL + CLIP-L encoders — for  e_0  to match what the Nexus-Gen
   pipeline produces at inference.  CLIP-L is still used for the
   pooled conditioning slot with an empty string (matches
   ``train/decoder/generation_trainer.py``).

3. Stream 2 conditioning is **multi-token** (F3): per VGG layer per
   reference image, mean+std stats are projected through a per-layer
   trainable MLP to a 4096-dim token, giving K = 4·k context tokens.
   These are concatenated with the same 81-token base ``image_embed``
   that Stream 1 receives, so the patch DiT sees strictly ≥ Stream
   1's information (semantic content + retrieved style/structure).
   The pooled slot is also augmented (style_to_pooled added to the
   CLIP-pooled-empty vector).

4. The training objective is **residual regression**:  S_phi is trained
   to predict (v* − e_0.detach()) directly, NOT (v_pred − v*) with a
   learnable gate.  Reasoning: with the additive-gate formulation the
   loss decomposes to ‖e_0 − v*‖² + 2g⟨S_phi, e_0 − v*⟩ + g²‖S_phi‖²;
   when Stream 1 is unbiased the linear term vanishes in expectation
   and the optimum is g → 0 (gate collapse).  Residual regression
   side-steps this by giving S_phi a non-degenerate target.

   ``correction_gate`` is retained as a non-trainable buffer, init to
   1.0, used only at inference as a user-facing strength slider.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import get_constant_schedule_with_warmup

# ---------------------------------------------------------------------------
# Ensure Nexus-Gen modules are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NEXUS_GEN_ROOT = os.path.join(_SCRIPT_DIR, "..", "Nexus-Gen")
_DIFFSYNTH_ROOT = os.path.join(_NEXUS_GEN_ROOT, "DiffSynth-Studio")
for _p in (_NEXUS_GEN_ROOT, _DIFFSYNTH_ROOT):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffsynth import ModelManager                           # noqa: E402
from diffsynth.models.flux_dit import FluxDiT                # noqa: E402
from diffsynth.models.utils import load_state_dict as _load_state_dict  # noqa: E402
from modeling.decoder.pipelines import NexusGenGenerationPipeline  # noqa: E402
from modeling.ar.modeling_qwen2_5_vl import (                # noqa: E402
    Qwen2_5_VLForConditionalGeneration,
)
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor  # noqa: E402
from transformers import AutoConfig                          # noqa: E402


# English prompt template used by Nexus-Gen's image_generation.py.
# Must match exactly — the Qwen AR model was trained to produce image
# embeddings conditional on this wrapping.
NEXUS_GEN_EN_TEMPLATE = (
    "Generate an image according to the following description: {}"
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      VGG-19 Feature Extractor                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Layer-index mapping inside ``torchvision.models.vgg19().features``
VGG_LAYER_INDICES = {
    "relu1_1": 1,    # 64  channels
    "relu2_1": 6,    # 128 channels
    "relu3_1": 11,   # 256 channels
    "relu4_1": 20,   # 512 channels
}
VGG_STYLE_DIM = (64 + 128 + 256 + 512) * 2   # 1920  (mean + std per channel)

# Per-VGG-layer dimensions for mean+std stats (used by the multi-token
# style projector in F3: one projector per layer maps 2·C_ℓ → 4096).
VGG_LAYER_CHANNELS = (64, 128, 256, 512)              # relu{1,2,3,4}_1
VGG_LAYER_STAT_DIMS = tuple(2 * c for c in VGG_LAYER_CHANNELS)  # (128, 256, 512, 1024)


class VGG19StyleExtractor(nn.Module):
    """Frozen VGG-19 that returns feature maps at relu{1,2,3,4}_1."""

    def __init__(self) -> None:
        super().__init__()
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1,
        )
        max_idx = max(VGG_LAYER_INDICES.values())
        # Keep only up to relu4_1 (saves memory)
        self.slices = nn.Sequential(*list(vgg.features.children())[: max_idx + 1])
        self.target_indices = sorted(VGG_LAYER_INDICES.values())

        # ImageNet normalisation constants (registered as buffers)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Freeze everything
        self.requires_grad_(False)
        self.eval()

    # keep eval mode even when parent calls .train()
    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, 3, H, W)  images in **[0, 1]** range.

        Returns
        -------
        List of 4 feature-map tensors at relu{1,2,3,4}_1.
        """
        # Cast input to match VGG weight dtype.
        # DeepSpeed bf16 mode casts all registered nn.Module parameters
        # (including VGG conv weights/biases) to bfloat16, so we must match
        # the input dtype to the actual weight dtype rather than hardcoding float32.
        weight_dtype = next(self.slices.parameters()).dtype
        x = x.to(dtype=weight_dtype)
        x = (x - self.mean.to(dtype=weight_dtype)) / self.std.to(dtype=weight_dtype)
        features: list[torch.Tensor] = []
        for idx, layer in enumerate(self.slices):
            x = layer(x)
            if idx in self.target_indices:
                features.append(x)
        return features


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      RAG Patch Trainer Module                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class RAGPatchTrainer(pl.LightningModule):
    """
    Two-stream diffusion trainer:

    * **Stream 1** — frozen base FLUX DiT (Qwen-conditioned)  →  e_0
    * **Stream 2** — LoRA-wrapped FLUX DiT (style-conditioned) →  S_phi
    * **Train**   — MSE(S_phi, (v* − e_0.detach())) · BSMNTW weight
    * **Infer**   — v_pred = e_0 + strength · S_phi (buffer; default 1.0)
    """

    def __init__(
        self,
        # ---- model paths (relative to Nexus-Gen root) ----
        nexgen_decoder_path: str = "models/Nexus-GenV2/generation_decoder.bin",
        pretrained_text_encoder_path: str = "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
        # The Qwen2.5-VL AR checkpoint lives in the same HF folder as the
        # Nexus-GenV2 weights — config.json + 4-shard safetensors at
        # ``models/Nexus-GenV2/``.  Its output (81 × 3584) passes through
        # the frozen adapter embedded in ``generation_decoder.bin`` to
        # produce the DiT's conditioning input.
        qwen_ckpt_path: str = "models/Nexus-GenV2",
        # Kept for backwards compat with older configs; unused now that
        # Stream 1 uses the Qwen path instead of T5-XXL.
        pretrained_t5_path: str | None = None,
        pretrained_vae_path: str = "models/FLUX/FLUX.1-dev/ae.safetensors",
        nexus_gen_root: str | None = None,
        # ---- LoRA ----
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        # ---- training ----
        learning_rate: float = 1e-5,
        lr_warmup_steps: int = 100,
        use_gradient_checkpointing: bool = True,
        t5_sequence_length: int = 512,
        guidance_scale: float = 3.5,
        torch_dtype_str: str = "bf16",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.t5_sequence_length = t5_sequence_length
        self.guidance_scale = guidance_scale

        dtype_map = {"bf16": torch.bfloat16, "16": torch.float16, "32": torch.float32}
        torch_dtype = dtype_map[torch_dtype_str]

        # Resolve model paths
        if nexus_gen_root is None:
            nexus_gen_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Nexus-Gen")
            )
        _abs = lambda p: p if os.path.isabs(p) else os.path.join(nexus_gen_root, p)

        # ==================================================================
        # 1.  LOAD FROZEN BASE PIPELINE  (VAE, text encoders, Nexus-GenV2 DiT)
        # ==================================================================

        # 1a. VAE + CLIP-L text encoder via ModelManager
        # CLIP-L is still loaded because the Nexus-Gen pipeline uses it
        # for the *pooled* conditioning slot with an empty string prompt
        # (see ``train/decoder/generation_trainer.py`` training_step).
        # T5-XXL is NOT loaded — Stream 1 uses the Qwen2.5-VL AR path
        # for the main (sequential) conditioning instead, matching the
        # distribution the DiT was actually fine-tuned on.
        model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
        model_manager.load_models(
            [
                _abs(pretrained_text_encoder_path),
                _abs(pretrained_vae_path),
            ]
        )

        # 1b. Load Nexus-GenV2 generation_decoder.bin → extract DiT weights
        #     AND the frozen adapter (Linear 3584→4096 → LN → ReLU → Linear
        #     4096→4096 → LN) that bridges Qwen AR output into the DiT's
        #     conditioning space.  Pattern mirrors
        #     ``modeling/decoder/generation_decoder.py``.
        print("Loading Nexus-GenV2 generation decoder DiT + adapter weights …")
        nexgen_bin_path = _abs(nexgen_decoder_path)
        nexgen_state = _load_state_dict(nexgen_bin_path)
        dit_state_dict = {
            k.replace("pipe.dit.", ""): v
            for k, v in nexgen_state.items()
            if not k.startswith("adapter.")
        }
        adapter_state_dict = {
            k.replace("adapter.", ""): v
            for k, v in nexgen_state.items()
            if k.startswith("adapter.")
        }
        del nexgen_state  # free original references

        # Passthrough converter (keys already match FluxDiT parameter names)
        class _PassthroughConverter:
            def from_diffusers(self, sd):
                return sd

        FluxDiT.state_dict_converter = staticmethod(
            lambda: _PassthroughConverter()
        )
        model_manager.load_model_from_single_file(
            nexgen_bin_path,
            state_dict=dit_state_dict,
            model_names=["flux_dit"],
            model_classes=[FluxDiT],
            model_resource="diffusers",
        )
        model_manager.model[-1].to(dtype=torch_dtype)

        # Free dit_state_dict NOW — it is no longer needed after
        # load_model_from_single_file copied the values into FluxDiT.
        # This frees ~22 GB before building the pipeline.
        del dit_state_dict

        # 1c. Build pipeline + scheduler
        # IMPORTANT: use object.__setattr__ instead of self.pipe = ...
        # BasePipeline inherits from torch.nn.Module, so a normal assignment
        # would register the entire pipeline (VAE + text encoders) as a
        # sub-module of RAGPatchTrainer.  DeepSpeed would then include them
        # in its initial self.module.to(device) call, wasting GPU memory.
        # Bypassing nn.Module.__setattr__ hides self.pipe from DS3 so it
        # only sees patch_dit + VGG19 + style projectors.  We manage the
        # pipeline's sub-models (VAE, text encoders) manually in forward().
        # NOTE: self.pipe.dit is the SAME object as self.patch_dit's inner
        # model (weight-sharing).  DS3 manages it via self.patch_dit.
        pipe = NexusGenGenerationPipeline.from_model_manager(model_manager)
        pipe.scheduler.set_timesteps(1000, training=True)
        object.__setattr__(self, "pipe", pipe)

        # 1d. Freeze the entire pipeline
        # NOTE: ``self.pipe`` is NOT an nn.Module, so Lightning does not
        # track its parameters.  We freeze and manage devices manually.
        for attr in ("dit", "vae_encoder", "vae_decoder"):
            model = getattr(self.pipe, attr, None)
            if model is not None:
                model.requires_grad_(False)
                model.eval()

        # ==================================================================
        # 1e.  QWEN2.5-VL AR MODEL + ADAPTER  (frozen)
        # ==================================================================
        # Stream 1 conditioning path.  The prompt is wrapped with
        # NEXUS_GEN_EN_TEMPLATE, tokenised by the Qwen processor, and the
        # AR model's ``generate`` call — with ``generation_image_grid_thw
        # = [[1, 18, 18]]`` (see image_generation.py:56) — produces an
        # ``output_image_embeddings`` tensor of shape (B, 81, 3584).  The
        # frozen adapter then projects that to (B, 81, 4096) for the DiT.
        #
        # BOTH the Qwen model and the adapter are hidden from nn.Module
        # via ``object.__setattr__`` so DeepSpeed does not try to shard
        # or stream them — they are frozen, large, and managed manually
        # (shuttled to GPU only during ``forward``).
        print("Loading Qwen2.5-VL AR model + processor …")
        qwen_ckpt_abs = _abs(qwen_ckpt_path)
        qwen_config = AutoConfig.from_pretrained(qwen_ckpt_abs, trust_remote_code=True)
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_ckpt_abs,
            config=qwen_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        qwen_model.eval()
        qwen_model.requires_grad_(False)
        qwen_processor = Qwen2_5_VLProcessor.from_pretrained(qwen_ckpt_abs)
        object.__setattr__(self, "qwen_ar", qwen_model)
        object.__setattr__(self, "qwen_processor", qwen_processor)

        # Build adapter module and load weights extracted from
        # generation_decoder.bin at step 1b.  Dimensions fixed by the
        # Nexus-GenV2 design (see train/decoder/generation_trainer.py:48).
        ar_adapter = nn.Sequential(
            nn.Linear(3584, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
        )
        ar_adapter.load_state_dict(adapter_state_dict)
        ar_adapter.to(dtype=torch_dtype)
        ar_adapter.eval()
        ar_adapter.requires_grad_(False)
        object.__setattr__(self, "ar_adapter", ar_adapter)
        del adapter_state_dict

        # ==================================================================
        # 2.  VGG-19 FEATURE EXTRACTOR  (frozen)
        # ==================================================================
        self.vgg19 = VGG19StyleExtractor()

        # ==================================================================
        # 3.  WRAP DiT with LoRA  (weight-sharing — NO deepcopy)
        # ==================================================================
        # Instead of deepcopy (which consumed an extra ~22 GB of RAM and
        # caused OOM under --mem=64G), we wrap self.pipe.dit IN-PLACE with
        # LoRA.  The base model and the patch model share the same weights.
        #
        # For the base forward  (Stream 1, frozen):  we use the PEFT
        #   context manager ``self.patch_dit.disable_adapter()`` which
        #   makes the model behave as the original DiT without LoRA.
        # For the patch forward (Stream 2, trainable): LoRA is active.
        #
        # This saves ~22 GB of CPU RAM, keeping steady-state at ~30 GB
        # (well within --mem=64G).  Both forwards go through DS3's
        # parameter-streaming hooks, so no manual .to(device)/.cpu().
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                # -- Joint blocks (FluxJointAttention) ---------------------
                "a_to_qkv",   # image Q/K/V  (fused, Linear 3072→9216)
                "b_to_qkv",   # context Q/K/V (fused, Linear 3072→9216)
                "a_to_out",   # image output projection
                "b_to_out",   # context output projection
                # -- Single blocks (FluxSingleTransformerBlock) ------------
                "to_qkv_mlp", # fused Q/K/V + MLP
                "proj_out",   # output projection
            ],
            bias="none",
        )
        self.patch_dit = get_peft_model(self.pipe.dit, lora_config)
        self.patch_dit.print_trainable_parameters()

        # ==================================================================
        # 4.  STYLE PROJECTION LAYERS  (trainable — bridges VGG → FLUX)
        # ==================================================================
        # F3: emit ONE 4096-dim context token per (VGG layer, reference)
        # rather than collapsing all four layers and k references into a
        # single token.  Stacked across k refs and 4 layers → K = 4·k
        # tokens (e.g. K = 12 for k = 3).  This gives the patch DiT real
        # multi-token structure to attend to and avoids the 1-token
        # OOD-context regime that crippled the previous design.
        #
        # Each layer ℓ has its own projector (input dim 2·C_ℓ varies by
        # layer, so we cannot share weights across layers).
        self.style_to_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 4096),
                nn.SiLU(),
                nn.Linear(4096, 4096),
            )
            for in_dim in VGG_LAYER_STAT_DIMS
        ])
        # VGG style stats  (1920-dim, ref-averaged) →  FLUX pooled  (B, 768)
        # The pooled slot remains a single vector; we add it (F3) to the
        # CLIP-pooled-empty vector that Stream 1 uses, so Stream 2's
        # pooled conditioning carries both the base prior and the style
        # signal.
        self.style_to_pooled = nn.Sequential(
            nn.Linear(VGG_STYLE_DIM, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
        )

        # ==================================================================
        # 5.  CORRECTION GATE  (inference-only strength buffer; NOT trained)
        # ==================================================================
        # The training objective is residual regression on S_phi directly,
        # so no learnable gate is needed during training.  We keep this as
        # a non-trainable buffer initialised to 1.0 so the inference
        # pipeline (F4) can read it as the default fusion strength
        # (v_pred = e_0 + correction_gate · S_phi) and so users can
        # override it via state_dict for stylistic exaggeration / damping.
        # Registered as a buffer so DeepSpeed does not place it in the
        # optimizer; saved by state_dict() unconditionally.
        self.register_buffer(
            "correction_gate",
            torch.ones(1, dtype=torch_dtype),
            persistent=True,
        )

        # ==================================================================
        # 6.  CAST TRAINABLE MODULES TO torch_dtype  (must be last)
        # ==================================================================
        # PEFT initialises LoRA layers in float32.  Cast them to bf16 so
        # they match the base model dtype.  DS3 expects a uniform dtype
        # when bf16-mixed precision is enabled.
        #
        # VGG-19 is not explicitly cast here: DeepSpeed bf16 will cast its
        # weights to bf16 automatically (it is a registered nn.Module).
        # VGG19StyleExtractor.forward() dynamically reads the actual weight
        # dtype and casts the input to match, so no dtype mismatch occurs.
        self.patch_dit.to(torch_dtype)
        self.style_to_context.to(torch_dtype)
        self.style_to_pooled.to(torch_dtype)

    # ──────────────────────────────────────────────────────────────────────
    #  FORWARD
    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        prompts: list[str],
        rag_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (B, 3, H, W)  ground-truth target images in [0, 1].
        prompts    : list[str] of length B — text prompts.
        rag_images : (B, k, 3, H', W')  retrieved reference images in [0, 1].

        Returns
        -------
        loss : scalar training loss.
        """
        B, k = x.shape[0], rag_images.shape[1]
        device = x.device
        dtype = self.pipe.torch_dtype

        # ==============================================================
        # STEP 1 — Latent & Noise Preparation  (no gradients)
        # ==============================================================
        with torch.no_grad():
            # --- Move VAE encoder to the active device ---
            self.pipe.vae_encoder.to(device)

            # Encode ground-truth images → latent  z_0
            z_0 = self.pipe.vae_encoder(
                x.to(dtype=dtype)
            )                                                      # (B, 16, H/8, W/8)

            # Move VAE encoder back to CPU to free GPU memory
            self.pipe.vae_encoder.cpu()
            torch.cuda.empty_cache()

            # Sample one random timestep *per sample* in the batch.
            # The scheduler stores `timesteps` as a 1-D tensor of length
            # num_train_timesteps.  We draw integer indices uniformly then
            # index into it — this gives the correct sigma-shifted values.
            # Using shape (B,) instead of (1,) ensures every sample in the
            # batch sees a different noise level, producing richer gradients
            # especially when accumulate_grad_batches > 1.
            timestep_ids = torch.randint(
                0, self.pipe.scheduler.num_train_timesteps, (B,)
            )
            # timesteps is on CPU (scheduler never moves it to GPU)
            t = self.pipe.scheduler.timesteps[timestep_ids].to(device)  # (B,)

            # Sample Gaussian noise  ε
            epsilon = torch.randn_like(z_0)

            # Noisy latent per sample: z_t[i] = (1 − σ_i) z_0[i] + σ_i ε[i]
            # scheduler.add_noise accepts only a scalar/shape-(1,) timestep, so
            # we vectorise the sigma lookup ourselves using the same formula.
            sched = self.pipe.scheduler
            # Vectorised index lookup: (B,) vs (num_train_timesteps,)
            diffs = (sched.timesteps.unsqueeze(0) - t.cpu().unsqueeze(1)).abs()
            tids = diffs.argmin(dim=1)                        # (B,)
            sigmas = sched.sigmas[tids].to(device=device, dtype=dtype)  # (B,)
            # Broadcast sigmas over (C, H, W) dimensions
            sigmas_4d = sigmas.view(B, 1, 1, 1)
            z_t = (1 - sigmas_4d) * z_0 + sigmas_4d * epsilon

        # ==============================================================
        # STEP 2 — Stream 1: Base Generation  (frozen, no gradients)
        # ==============================================================
        with torch.no_grad():
            # --- Semantic conditioning via Qwen2.5-VL AR + adapter ---
            # This is the ONLY conditioning path the Nexus-GenV2 DiT was
            # fine-tuned to accept.  Produces image_embed shape
            # (B, 81, 4096).  Mirrors image_generation.py:56-66 and
            # train/decoder/generation_trainer.py:86-89.
            self.pipe.device = device
            self.qwen_ar.to(device)
            self.ar_adapter.to(device)

            # Qwen.generate is autoregressive and does not accept a
            # batch of prompts with different lengths cleanly at this
            # template wrapping, so we call it per-prompt.  B is small
            # (batch_size=1 in config.yaml) so the loop cost is minor
            # relative to the DiT forward passes below.
            grid_thw = torch.tensor([[1, 18, 18]], device=device)
            ar_embeds: list[torch.Tensor] = []
            for prompt in prompts:
                formatted = NEXUS_GEN_EN_TEMPLATE.format(prompt)
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": formatted}]}
                ]
                text = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                ar_inputs = self.qwen_processor(
                    text=[text], padding=True, return_tensors="pt",
                ).to(device)
                ar_outputs = self.qwen_ar.generate(
                    **ar_inputs,
                    max_new_tokens=1024,
                    return_dict_in_generate=True,
                    generation_image_grid_thw=grid_thw,
                )
                # (1, 81, 3584)
                ar_embeds.append(ar_outputs["output_image_embeddings"])

            output_image_embeddings = torch.cat(ar_embeds, dim=0)       # (B, 81, 3584)
            prompt_emb = self.ar_adapter(
                output_image_embeddings.to(dtype=dtype)
            )                                                            # (B, 81, 4096)

            # --- Pooled conditioning: CLIP-L on empty string ---
            # Matches generation_trainer.py:86 which uses
            # ``encode_prompt("", positive=True, clip_only=True)``.  The
            # DiT's pooled slot was trained with an empty-prompt pooled
            # vector when the main context came from the AR adapter.
            prompter = self.pipe.prompter
            text_encoder_1 = self.pipe.text_encoder_1
            if text_encoder_1 is None:
                raise RuntimeError(
                    "CLIP text encoder (text_encoder_1) is None — "
                    "check that model_manager loaded the CLIP safetensors file."
                )
            text_encoder_1.to(device)
            pooled_prompt_emb = prompter.encode_prompt_using_clip(
                [""] * B, text_encoder_1, prompter.tokenizer_1, 77, device,
            )                                                            # (B, 768)

            text_ids = torch.zeros(
                B, prompt_emb.shape[1], 3,
                device=device, dtype=prompt_emb.dtype,
            )                                                            # (B, 81, 3)
            batch_prompt_emb = {
                "prompt_emb": prompt_emb,
                "pooled_prompt_emb": pooled_prompt_emb,
                "text_ids": text_ids,
            }

            # --- Offload all Stream 1 conditioning modules back to CPU ---
            self.qwen_ar.cpu()
            self.ar_adapter.cpu()
            text_encoder_1.cpu()
            torch.cuda.empty_cache()

            # --- Extra inputs  (image position IDs + guidance) ---
            extra_input = self.pipe.prepare_extra_input(
                z_t, guidance=self.guidance_scale,
            )

            # --- Base DiT forward  →  velocity prediction  e_0 ---
            # Use the SAME weights as patch_dit but with LoRA adapters
            # disabled — equivalent to the original frozen base DiT.
            # DS3 handles parameter streaming automatically; no manual
            # .to(device)/.cpu() needed.
            with self.patch_dit.disable_adapter():
                e_0 = self.patch_dit(
                    z_t,
                    timestep=t,
                    **batch_prompt_emb,
                    **extra_input,
                )                                                  # (B, 16, H/8, W/8)

        # ==============================================================
        # STEP 3 — Stream 2: RAG Patch  (trainable LoRA + projectors)
        # ==============================================================

        # 3a. Feature Extraction — VGG-19  (frozen, no gradients)
        B_r, k_r, C_r, H_r, W_r = rag_images.shape
        rag_flat = rag_images.reshape(B_r * k_r, C_r, H_r, W_r)

        with torch.no_grad():
            vgg_feats = self.vgg19(
                rag_flat.to(device=device, dtype=torch.float32)
            )
            # vgg_feats: list of 4 tensors at relu{1,2,3,4}_1
            #   shapes: (B*k, 64, …), (B*k, 128, …), (B*k, 256, …), (B*k, 512, …)

        # 3b. Per-layer mean+std stats — one tensor per VGG depth.
        # Use unbiased=False (population std): the unbiased correction
        # (÷ n-1 vs ÷ n) is negligible for large maps but prevents
        # division-by-zero edge cases at relu4_1 where H·W is small.
        # Each entry has shape (B*k, 2·C_ℓ).
        layer_stats: list[torch.Tensor] = []
        for feat in vgg_feats:
            ch_mean = feat.mean(dim=[2, 3])                    # (B*k, C_ℓ)
            ch_std  = feat.std(dim=[2, 3], unbiased=False)     # (B*k, C_ℓ)
            layer_stats.append(torch.cat([ch_mean, ch_std], dim=-1))  # (B*k, 2·C_ℓ)

        # 3c. Multi-token style projection (F3)
        # Each (layer, ref) pair → one 4096-dim token.  Stacked over the
        # 4 VGG depths and k references → K = 4·k context tokens per
        # batch sample.  No averaging across refs here — refs differ in
        # style and the DiT's attention can decide which to weight.
        layer_tokens = []
        for layer_idx, stat in enumerate(layer_stats):
            token = self.style_to_context[layer_idx](
                stat.to(dtype=dtype)
            )                                                   # (B*k, 4096)
            layer_tokens.append(token)
        # (B*k, 4_layers, 4096) → (B, k, 4, 4096) → (B, K=4k, 4096)
        style_context_tokens = torch.stack(layer_tokens, dim=1)
        style_context_tokens = style_context_tokens.view(B, k * len(layer_stats), -1)

        # 3d. Pooled style: ref-averaged 1920-dim vector → 768-dim.
        # This is the same recipe as the previous (collapsed) design,
        # used only for the pooled-conditioning slot.  Reuse the per-
        # layer stats already computed above to avoid recomputation.
        flat_stats = torch.cat(layer_stats, dim=-1)             # (B*k, 1920)
        style_embedding = flat_stats.view(B, k, -1).mean(dim=1) # (B, 1920)
        style_pooled = self.style_to_pooled(
            style_embedding.to(dtype=dtype)
        )                                                       # (B, 768)

        # 3e. Stream 2 conditioning fusion (F3)
        # Concatenate the base 81-token Qwen image_embed (semantic
        # context) with the K style tokens, so the patch DiT sees
        # strictly ≥ Stream 1's information.  Without the semantic
        # context the patch can only learn content-agnostic global
        # style deltas; with it, the patch can learn content-aware
        # localized style residuals.
        base_prompt_emb = batch_prompt_emb["prompt_emb"]        # (B, 81, 4096)
        patch_prompt_emb = torch.cat(
            [base_prompt_emb, style_context_tokens.to(dtype=base_prompt_emb.dtype)],
            dim=1,
        )                                                       # (B, 81+K, 4096)

        # Pooled: sum CLIP-pooled-empty (Stream 1) + style-pooled.
        patch_pooled_prompt_emb = (
            batch_prompt_emb["pooled_prompt_emb"]
            + style_pooled.to(dtype=batch_prompt_emb["pooled_prompt_emb"].dtype)
        )                                                       # (B, 768)

        # text_ids: zeros of matching length (no positional info; matches
        # the convention used by Stream 1 and by the Nexus-Gen pipeline
        # for AR-produced embeddings).
        patch_text_ids = torch.zeros(
            B, patch_prompt_emb.shape[1], 3,
            device=device, dtype=patch_prompt_emb.dtype,
        )

        # --- Extra inputs for patch DiT  (same geometry & guidance) ---
        patch_extra = self.pipe.prepare_extra_input(
            z_t, guidance=self.guidance_scale,
        )

        # 3f. Residual Prediction — LoRA-wrapped patch DiT
        #     Autograd tracks only LoRA parameters + style projectors.
        S_phi = self.patch_dit(
            z_t,
            timestep=t,
            prompt_emb=patch_prompt_emb,
            pooled_prompt_emb=patch_pooled_prompt_emb,
            text_ids=patch_text_ids,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            **patch_extra,
        )                                                       # (B, 16, H/8, W/8)

        # ==============================================================
        # STEP 4 — Residual Regression Loss
        # ==============================================================
        # Train  S_phi  to predict the *base residual*  r* := v* − e_0,
        # where v* = ε − z_0 is the flow-matching velocity target.  This
        # avoids the gate-collapse pathology of the additive-gate
        # formulation: the residual target is non-degenerate (it is by
        # definition exactly what the base gets wrong), so there is no
        # equilibrium that pushes ‖S_phi‖ to zero.  At inference, fusion
        # is applied as  v_pred = e_0 + correction_gate · S_phi  with
        # correction_gate = 1.0 by default.
        #
        # e_0 is already produced under torch.no_grad() so it carries
        # no graph; we still call .detach() defensively to make the
        # mathematical statement explicit and to guard against future
        # refactors that re-enable autograd on Stream 1.

        # Flow-matching velocity target  (B, 16, H/8, W/8)
        v_target = (epsilon - z_0).to(dtype=S_phi.dtype)

        # Per-sample MSE between S_phi and the residual target.
        # F.mse_loss(reduction="none") gives (B, C, H, W); we mean over C,H,W.
        residual_target = v_target - e_0.detach().to(dtype=S_phi.dtype)
        per_sample_mse = F.mse_loss(
            S_phi.float(), residual_target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])                                  # (B,)

        # BSMNTW importance weighting — one weight per sample's timestep.
        # scheduler.linear_timesteps_weights is a 1-D tensor of length
        # num_train_timesteps on CPU.  We use the same vectorised index lookup
        # computed above for the sigmas (tids is already in scope).
        bsmntw_weights = sched.linear_timesteps_weights[tids].to(
            device=device, dtype=S_phi.dtype
        )                                                      # (B,)

        # Weighted mean over the batch → scalar loss
        loss = (per_sample_mse * bsmntw_weights).mean()

        # ==============================================================
        # F6: diagnostic logging (training only)
        # ==============================================================
        # These three signals together falsify or confirm the F2
        # gate-collapse argument under the new objective:
        #   * ‖S_phi‖   — should rise from ~0 as projectors warm up.
        #   * residual_mse_unweighted — should fall (unweighted mirror
        #     of the loss for direct comparison across BSMNTW changes).
        #   * rho_t (cosine of S_phi with the residual target r*)
        #     — should drift away from 0 toward +1; near-zero running
        #     mean means the patch is making no useful progress.
        if self.training and self.trainer is not None:
            with torch.no_grad():
                s_flat = S_phi.detach().float().flatten(1)               # (B, D)
                r_flat = residual_target.detach().float().flatten(1)     # (B, D)
                e_flat = e_0.detach().float().flatten(1)                 # (B, D)
                v_flat = v_target.detach().float().flatten(1)            # (B, D)

                s_norm = s_flat.norm(dim=1)                              # (B,)
                r_norm = r_flat.norm(dim=1)                              # (B,)
                base_err = (e_flat - v_flat).norm(dim=1)                 # (B,)

                eps = 1e-8
                rho = (s_flat * r_flat).sum(dim=1) / (s_norm * r_norm + eps)

                self.log("diag/S_phi_norm",        s_norm.mean(),
                         prog_bar=False, on_step=True, on_epoch=False)
                self.log("diag/base_err_norm",     base_err.mean(),
                         prog_bar=False, on_step=True, on_epoch=False)
                self.log("diag/residual_mse",      per_sample_mse.detach().mean(),
                         prog_bar=False, on_step=True, on_epoch=False)
                self.log("diag/rho_S_phi_target",  rho.mean(),
                         prog_bar=False, on_step=True, on_epoch=False)

        return loss

    # ──────────────────────────────────────────────────────────────────────
    #  Lightning Hooks
    # ──────────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.forward(
            x=batch["image"],
            prompts=batch["prompt"],
            rag_images=batch["rag_images"],
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        # Collect only *trainable* parameters:
        #   - LoRA adapters inside patch_dit  (base weights are frozen)
        #   - Style projection layers
        # NOTE: ``correction_gate`` is now a non-trainable buffer (F2) so
        # it is intentionally NOT included here.  See the residual-loss
        # formulation in ``forward``.
        trainable_params: list[torch.Tensor] = []
        trainable_params += [
            p for p in self.patch_dit.parameters() if p.requires_grad
        ]
        trainable_params += list(self.style_to_context.parameters())
        trainable_params += list(self.style_to_pooled.parameters())

        # DeepSpeedCPUAdam is required when offload_optimizer=True (ZeRO-Offload).
        # PyTorch AdamW has no fused CPU kernel, so DeepSpeed rejects it outright.
        # DeepSpeedCPUAdam is a drop-in replacement with a fused C++ CPU Adam kernel
        # purpose-built for ZeRO-Offload and accepts the same arguments.
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(trainable_params, lr=self.learning_rate)

        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.lr_warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return only the trainable weights (LoRA adapters + style projectors)
        plus the ``correction_gate`` inference buffer.

        Lightning calls this automatically when saving checkpoints, so the
        standard checkpoint dict (epoch, global_step, lr_schedulers, …) is
        preserved while the state_dict key contains only our small set of
        trained weights — NOT the frozen base DiT.

        DeepSpeed ZeRO Stage 3 shards parameters across ranks, so we must
        gather each parameter before reading .data.  deepspeed.zero.GatheredParameters
        temporarily reassembles the full tensor on every rank inside its context;
        outside the context the param reverts to its shard.  We only run the
        body on rank 0 to avoid redundant file I/O.
        """
        import deepspeed

        sd = {}

        # Only gather + save on rank 0 (or when called outside training, e.g.
        # during initial save_hyperparameters).  GatheredParameters must only
        # be enabled when DeepSpeed ZeRO-3 has already initialised its engine;
        # calling it earlier (e.g. during __init__) raises an error.
        # Use the official DeepSpeedStrategy.zero_stage_3 property (bool) rather
        # than checking for "config" attr, which is unreliable across versions.
        is_ds_active = (
            self.trainer is not None
            and hasattr(self.trainer, "strategy")
            and getattr(self.trainer.strategy, "zero_stage_3", False)
        )

        all_params = (
            list(self.patch_dit.parameters())
            + list(self.style_to_context.parameters())
            + list(self.style_to_pooled.parameters())
        )

        with deepspeed.zero.GatheredParameters(all_params, enabled=is_ds_active):
            is_rank0 = (self.trainer is None) or (self.trainer.global_rank == 0)
            if is_rank0:
                # LoRA adapter weights only (no base model weights)
                # Keys: "base_model.model.<block>.<proj>.lora_A.weight", etc.
                adapter_sd = get_peft_model_state_dict(self.patch_dit)
                for k, v in adapter_sd.items():
                    sd[f"patch_dit.{k}"] = v.cpu()

                # Style projection layers
                for prefix in ("style_to_context", "style_to_pooled"):
                    module = getattr(self, prefix)
                    for name, param in module.named_parameters():
                        sd[f"{prefix}.{name}"] = param.data.cpu()

        # ``correction_gate`` is a non-trainable buffer (F2) — DeepSpeed
        # does not shard it, so save outside the GatheredParameters scope.
        # Save on rank 0 only to match the rest of the dict.
        if (self.trainer is None) or (self.trainer.global_rank == 0):
            sd["correction_gate"] = self.correction_gate.data.cpu()

        return sd
