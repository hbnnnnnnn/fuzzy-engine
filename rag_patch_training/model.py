"""
RAG-Conditioned Additive Patch Training for FLUX DiT  (Stream 2)
=================================================================

Trains a LoRA-based correction stream (patch_dit) on top of a frozen
base FLUX text-to-image diffusion model (base_dit), conditioned on
VGG-19 style features extracted from retrieved reference images (RAG).

Architecture
------------
  Stream 1 (frozen):  base_dit  →  e_0   (base velocity prediction)
  Stream 2 (LoRA):    patch_dit →  S_phi (additive correction)
  Fusion:             v_pred = e_0 + correction_gate · S_phi
  Loss:               MSE(v_pred, velocity_target)

Important implementation notes vs. task description
----------------------------------------------------
1. FLUX uses *flow matching* with velocity prediction (v = ε − x₀),
   **not** standard ε-prediction.  The correct training target is the
   velocity v* = ε − x₀, not raw noise ε.

2. FLUX has TWO text encoders (CLIP-L for pooled embedding, T5-XXL for
   sequential tokens).  The pipeline's ``encode_prompt()`` wraps both.

3. VGG style_embedding (1920-dim) requires **trainable projection
   layers** to match FLUX's expected input dimensions (4096 for
   context, 768 for pooled conditioning).

4. At initialisation the LoRA contribution is zero, so S_phi ≈ e_0.
   A zero-initialised ``correction_gate`` scalar ensures v_pred = e_0
   at the start (analogous to ControlNet zero-conv initialisation).
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

    * **Stream 1** — frozen base FLUX DiT (text-conditioned)  →  e_0
    * **Stream 2** — LoRA-wrapped FLUX DiT (style-conditioned) →  S_phi
    * **Fusion**  — v_pred = e_0 + correction_gate · S_phi
    * **Loss**    — MSE(v_pred, velocity_target) · BSMNTW weight
    """

    def __init__(
        self,
        # ---- model paths (relative to Nexus-Gen root) ----
        nexgen_decoder_path: str = "models/Nexus-GenV2/generation_decoder.bin",
        pretrained_text_encoder_path: str = "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
        pretrained_t5_path: str = "models/FLUX/FLUX.1-dev/text_encoder_2",
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

        # 1a. VAE + text encoders via ModelManager
        # T5-XXL lives in a HuggingFace folder (config.json + shard files).
        # load_model() bails out with a warning for directory paths before it
        # ever reaches ModelDetectorFromHuggingfaceFolder, so we must call
        # load_model_from_huggingface_folder() directly for the T5 directory.
        model_manager = ModelManager(torch_dtype=torch_dtype, device="cpu")
        model_manager.load_models(
            [
                _abs(pretrained_text_encoder_path),
                _abs(pretrained_vae_path),
            ]
        )
        # Load T5-XXL from HuggingFace folder directly (bypasses the isfile check).
        # load_model() bails out for directory paths before reaching any detector,
        # so we call load_model_from_huggingface_folder with explicit names/classes.
        from diffsynth.models.flux_text_encoder import FluxTextEncoder2
        model_manager.load_model_from_huggingface_folder(
            _abs(pretrained_t5_path),
            model_names=["flux_text_encoder_2"],
            model_classes=[FluxTextEncoder2],
        )

        # 1b. Load Nexus-GenV2 generation_decoder.bin → extract DiT weights
        #     (follows the same pattern as modeling/decoder/generation_decoder.py)
        print("Loading Nexus-GenV2 generation decoder DiT weights …")
        nexgen_bin_path = _abs(nexgen_decoder_path)
        nexgen_state = _load_state_dict(nexgen_bin_path)
        dit_state_dict = {
            k.replace("pipe.dit.", ""): v
            for k, v in nexgen_state.items()
            if not k.startswith("adapter.")
        }
        del nexgen_state  # free adapter keys + original references

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
        # VGG style stats  (1920-dim) →  FLUX context path  (B, N, 4096)
        self.style_to_context = nn.Sequential(
            nn.Linear(VGG_STYLE_DIM, 4096),
            nn.SiLU(),
            nn.Linear(4096, 4096),
        )
        # VGG style stats  (1920-dim) →  FLUX pooled conditioning  (B, 768)
        self.style_to_pooled = nn.Sequential(
            nn.Linear(VGG_STYLE_DIM, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
        )

        # ==================================================================
        # 5.  ZERO-INITIALISED CORRECTION GATE  (ControlNet-style)
        # ==================================================================
        # Ensures  v_pred = e_0  at initialisation  (gate starts at 0).
        # Initialise in torch_dtype so it matches e_0 / S_phi dtype in forward;
        # avoids an implicit float32 upcast that would break DS3 bf16 mode.
        self.correction_gate = nn.Parameter(torch.zeros(1, dtype=torch_dtype))

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
            # --- Text encoding  (CLIP pooled  +  T5 sequential) ---
            self.pipe.device = device
            # Move text encoders to device
            if hasattr(self.pipe, "text_encoder_1") and self.pipe.text_encoder_1 is not None:
                self.pipe.text_encoder_1.to(device)
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.to(device)

            # Encode the full batch of prompts in one tokenizer call.
            # encode_prompt_using_t5 / _clip both pass `prompt` straight to
            # the HuggingFace tokenizer which handles a list[str] natively,
            # returning (B, T) input_ids — so we avoid B serial forward passes.
            prompter = self.pipe.prompter
            text_encoder_1 = self.pipe.text_encoder_1
            text_encoder_2 = self.pipe.text_encoder_2
            if text_encoder_2 is None:
                raise RuntimeError(
                    "T5 text encoder (text_encoder_2) is None — "
                    "check that load_model_from_huggingface_folder succeeded for T5."
                )
            if text_encoder_1 is None:
                raise RuntimeError(
                    "CLIP text encoder (text_encoder_1) is None — "
                    "check that model_manager loaded the CLIP safetensors file."
                )
            prompt_emb = prompter.encode_prompt_using_t5(
                prompts, text_encoder_2, prompter.tokenizer_2,
                self.t5_sequence_length, device,
            )                                                      # (B, T, 4096)
            pooled_prompt_emb = prompter.encode_prompt_using_clip(
                prompts, text_encoder_1, prompter.tokenizer_1,
                77, device,
            )                                                      # (B, 768)
            text_ids = torch.zeros(
                B, prompt_emb.shape[1], 3,
                device=device, dtype=prompt_emb.dtype,
            )                                                      # (B, T, 3)
            batch_prompt_emb = {
                "prompt_emb": prompt_emb,
                "pooled_prompt_emb": pooled_prompt_emb,
                "text_ids": text_ids,
            }

            # Move text encoders back to CPU
            if hasattr(self.pipe, "text_encoder_1") and self.pipe.text_encoder_1 is not None:
                self.pipe.text_encoder_1.cpu()
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self.pipe.text_encoder_2.cpu()
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

        # 3b. Style Pooling — channel-wise mean & std over spatial (H, W)
        # Use unbiased=False (population std) because spatial maps at relu1_1
        # are 512×512/2 = large, but at relu4_1 may be as small as 32×32 = 1024
        # elements per channel. The unbiased correction (÷ n-1 vs ÷ n) is
        # negligible for large maps but prevents division-by-zero edge cases.
        style_stats: list[torch.Tensor] = []
        for feat in vgg_feats:
            ch_mean = feat.mean(dim=[2, 3])                    # (B*k, C_l)
            ch_std  = feat.std(dim=[2, 3], unbiased=False)     # (B*k, C_l)
            style_stats.extend([ch_mean, ch_std])

        style_vec = torch.cat(style_stats, dim=-1)             # (B*k, 1920)

        # 3c. Aggregation — average over the k reference images
        style_vec = style_vec.view(B, k, -1)                   # (B, k, 1920)
        style_embedding = style_vec.mean(dim=1)                # (B, 1920)

        # 3d. Embedding projection  (trainable)
        style_context = self.style_to_context(
            style_embedding.to(dtype=dtype)
        ).unsqueeze(1)                                         # (B, 1, 4096)

        style_pooled = self.style_to_pooled(
            style_embedding.to(dtype=dtype)
        )                                                      # (B, 768)

        style_text_ids = torch.zeros(
            B, 1, 3, device=device, dtype=dtype,
        )                                                      # (B, 1, 3)

        # --- Extra inputs for patch DiT  (same geometry & guidance) ---
        patch_extra = self.pipe.prepare_extra_input(
            z_t, guidance=self.guidance_scale,
        )

        # 3e. Correction Prediction — LoRA-wrapped patch DiT
        #     Autograd tracks only LoRA parameters + style projectors.
        S_phi = self.patch_dit(
            z_t,
            timestep=t,
            prompt_emb=style_context,
            pooled_prompt_emb=style_pooled,
            text_ids=style_text_ids,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            **patch_extra,
        )                                                      # (B, 16, H/8, W/8)

        # ==============================================================
        # STEP 4 — Fusion & Loss
        # ==============================================================

        # Training target — flow matching velocity:  v* = ε − z_0
        training_target = (epsilon - z_0).to(dtype=S_phi.dtype)

        # ── Fusion loss: trains correction_gate ────────────────────────
        v_pred = e_0 + self.correction_gate * S_phi
        per_sample_mse = F.mse_loss(
            v_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])                                  # (B,)

        bsmntw_weights = sched.linear_timesteps_weights[tids].to(
            device=device, dtype=v_pred.dtype
        )                                                      # (B,)
        loss_fusion = (per_sample_mse * bsmntw_weights).mean()

        # ── Residual supervision: trains LoRA + style projectors ───────
        # S_phi should learn to predict the velocity residual (target − e_0).
        # This loss gives S_phi a non-zero gradient even when gate ≈ 0,
        # fixing the zero-init gate problem that suppressed all LoRA/projector
        # gradients and kept the correction stream untrained.
        residual_target = (training_target - e_0.detach()).to(dtype=S_phi.dtype)
        per_sample_residual = F.mse_loss(
            S_phi.float(), residual_target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])                                  # (B,)
        loss_patch = (per_sample_residual * bsmntw_weights).mean()

        # Total loss: alpha=0.5 balances residual supervision vs fusion.
        loss = loss_fusion + 0.5 * loss_patch

        return loss

    # ──────────────────────────────────────────────────────────────────────
    #  Lightning Hooks
    # ──────────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        print(f"[step {batch_idx}] training_step start", flush=True)
        loss = self.forward(
            x=batch["image"],
            prompts=batch["prompt"],
            rag_images=batch["rag_images"],
        )
        print(f"[step {batch_idx}] loss={loss.item():.4f}", flush=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        # Collect only *trainable* parameters:
        #   - LoRA adapters inside patch_dit  (base weights are frozen)
        #   - Style projection layers
        #   - Correction gate
        trainable_params: list[torch.Tensor] = []
        trainable_params += [
            p for p in self.patch_dit.parameters() if p.requires_grad
        ]
        trainable_params += list(self.style_to_context.parameters())
        trainable_params += list(self.style_to_pooled.parameters())
        trainable_params += [self.correction_gate]

        # Use DeepSpeedCPUAdam when offload_optimizer is active (ZeRO Stage 2/3
        # with CPU offload).  Fall back to standard AdamW for DDP / single-GPU
        # runs where DeepSpeed ops may not be available.
        try:
            trainer_strategy = self.trainer.strategy.__class__.__name__ if hasattr(self, "trainer") and self.trainer else ""
        except Exception:
            trainer_strategy = ""
        is_deepspeed = "DeepSpeed" in trainer_strategy
        if is_deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(trainable_params, lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

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
        """Return only the trainable weights (LoRA adapters + style projectors + gate).

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
            + [self.correction_gate]
        )

        # Only import deepspeed when the ZeRO-3 engine is actually active.
        # Importing deepspeed unconditionally is expensive (8-20s on login nodes
        # and can hang on GPU nodes while it tries to JIT-compile CUDA extensions).
        if is_ds_active:
            import deepspeed
            ctx = deepspeed.zero.GatheredParameters(all_params, enabled=True)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
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

                # Correction gate scalar
                sd["correction_gate"] = self.correction_gate.data.cpu()

        return sd
