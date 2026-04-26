"""
RAG-Patch Inference Pipeline (F4)
=================================

Subclasses ``NexusGenGenerationPipeline`` to fuse the trained RAG-Patch
LoRA branch into the denoising loop.  At each timestep the dual-stream
prediction

    v_pred  =  e_0  +  correction_gate · strength · S_phi

is computed, where:

* ``e_0``  — base velocity from the frozen Nexus-GenV2 DiT, conditioned
  on the 81-token Qwen2.5-VL ``image_embed`` (LoRA disabled).
* ``S_phi`` — RAG-Patch residual from the same DiT with LoRA enabled,
  conditioned on ``[image_embed, K style tokens]`` where the style
  tokens come from VGG-19 stats over the retrieved reference images.
* ``correction_gate`` — non-trainable buffer baked into the trainer
  (default 1.0); shipped alongside the LoRA weights.
* ``strength`` — user-facing multiplier (default 1.0) for A/B sweeps.

This module is intentionally kept inside ``rag_patch_training/`` so that
``src/Nexus-Gen/`` remains untouched.

Typical use:

    trainer = RAGPatchTrainer(...)
    load_rag_patch_state_dict(trainer, ckpt_dir)
    trainer.eval().to("cuda")

    pipe = NexusGenRAGPatchPipeline.from_trainer(trainer, strength=1.0)
    image = pipe(
        prompt="A cat on the moon",
        image_embed=image_embed,            # (1, 81, 4096) from AR + adapter
        rag_images=ref_tensor,              # (1, k, 3, H, W) in [0, 1]
        height=512, width=512, seed=42,
    )
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch
from tqdm import tqdm

# --- Make sure Nexus-Gen + DiffSynth-Studio are importable ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NEXUS_GEN_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "Nexus-Gen"))
_DIFFSYNTH_ROOT = os.path.join(_NEXUS_GEN_ROOT, "DiffSynth-Studio")
for _p in (_NEXUS_GEN_ROOT, _DIFFSYNTH_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffsynth.pipelines.flux_image import lets_dance_flux, TeaCache       # noqa: E402
from modeling.decoder.pipelines import NexusGenGenerationPipeline          # noqa: E402


# ──────────────────────────────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────────────────────────────

class NexusGenRAGPatchPipeline(NexusGenGenerationPipeline):
    """Drop-in replacement for ``NexusGenGenerationPipeline`` that runs
    the trained RAG-Patch branch alongside the base DiT.

    Construct via :meth:`from_trainer`; the bare ``__init__`` does not
    load any weights of its own — it reuses the modules already held by
    the trainer (the in-place LoRA design means base and patch DiT
    share the same FluxDiT object).
    """

    # Patch-specific fields (set by :meth:`from_trainer`)
    patch_dit = None
    style_to_context = None
    style_to_pooled = None
    vgg19 = None
    correction_gate = None
    strength: float = 1.0

    @staticmethod
    def from_trainer(trainer, strength: float = 1.0) -> "NexusGenRAGPatchPipeline":
        """Wrap an existing :class:`RAGPatchTrainer` (already ``.eval()``-ed
        and weights loaded) into an inference pipeline.

        The pipeline shares submodules with the trainer — the trainer
        instance must remain alive for the pipeline's lifetime.
        """
        # The trainer's frozen base pipeline has all the static models
        # (VAE, CLIP-L, scheduler, prompter, base DiT).  We re-use it
        # rather than constructing a fresh one.
        base_pipe = trainer.pipe  # type: NexusGenGenerationPipeline

        pipe = NexusGenRAGPatchPipeline(
            device=base_pipe.device,
            torch_dtype=base_pipe.torch_dtype,
        )
        # Carry over every submodel the parent class touches.
        pipe.dit = base_pipe.dit
        pipe.vae_encoder = getattr(base_pipe, "vae_encoder", None)
        pipe.vae_decoder = getattr(base_pipe, "vae_decoder", None)
        pipe.text_encoder_1 = getattr(base_pipe, "text_encoder_1", None)
        pipe.text_encoder_2 = getattr(base_pipe, "text_encoder_2", None)
        pipe.scheduler = base_pipe.scheduler
        pipe.prompter = base_pipe.prompter
        pipe.controlnet = getattr(base_pipe, "controlnet", None)
        # Match the parent's model_names list so load_models_to_device works.
        for attr in ("model_names", "in_iteration_models"):
            if hasattr(base_pipe, attr):
                setattr(pipe, attr, getattr(base_pipe, attr))

        # Patch-side modules — shared with the trainer.
        pipe.patch_dit = trainer.patch_dit
        pipe.style_to_context = trainer.style_to_context
        pipe.style_to_pooled = trainer.style_to_pooled
        pipe.vgg19 = trainer.vgg19
        pipe.correction_gate = trainer.correction_gate
        pipe.strength = float(strength)
        return pipe

    # ──────────────────────────────────────────────────────────────────
    #  Style conditioning  (mirror trainer.forward steps 3b–3e)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_style_conditioning(
        self, rag_images: torch.Tensor, target_dtype: torch.dtype,
    ):
        """Build (K-token context, pooled) style conditioning from a
        batch of retrieved reference images.

        Parameters
        ----------
        rag_images : Tensor of shape (B, k, 3, H, W) in [0, 1].

        Returns
        -------
        style_context_tokens : (B, K=4·k, 4096)
        style_pooled         : (B, 768)
        """
        from rag_patch_training.model import VGG_LAYER_STAT_DIMS  # noqa: F401  (assert)

        device = self.device
        B, k, C, H, W = rag_images.shape
        rag_flat = rag_images.reshape(B * k, C, H, W).to(
            device=device, dtype=torch.float32,
        )

        vgg_feats = self.vgg19(rag_flat)
        # vgg_feats: list of 4 tensors at relu{1,2,3,4}_1.

        # Per-layer mean+std → list of (B*k, 2·C_ℓ).
        layer_stats = []
        for feat in vgg_feats:
            ch_mean = feat.mean(dim=[2, 3])
            ch_std = feat.std(dim=[2, 3], unbiased=False)
            layer_stats.append(torch.cat([ch_mean, ch_std], dim=-1))

        # Project each (layer, ref) → 4096-dim token.
        layer_tokens = []
        for layer_idx, stat in enumerate(layer_stats):
            token = self.style_to_context[layer_idx](
                stat.to(dtype=target_dtype)
            )
            layer_tokens.append(token)
        style_context_tokens = torch.stack(layer_tokens, dim=1)        # (B*k, 4, 4096)
        style_context_tokens = style_context_tokens.view(
            B, k * len(layer_stats), -1,
        )                                                              # (B, K, 4096)

        # Pooled: ref-averaged 1920-dim flat → 768.
        flat_stats = torch.cat(layer_stats, dim=-1)                    # (B*k, 1920)
        style_embedding = flat_stats.view(B, k, -1).mean(dim=1)        # (B, 1920)
        style_pooled = self.style_to_pooled(
            style_embedding.to(dtype=target_dtype)
        )                                                              # (B, 768)
        return style_context_tokens, style_pooled

    # ──────────────────────────────────────────────────────────────────
    #  Denoising loop  (override)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 3.5,
        t5_sequence_length: int = 512,
        # Conditioning
        image_embed=None,
        rag_images: Optional[torch.Tensor] = None,
        # Image
        input_image=None,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        seed=None,
        # Steps
        num_inference_steps: int = 30,
        # TeaCache (ignored when patch path is active — it would cache
        # the wrong DiT output across the disable_adapter / enabled flips)
        tea_cache_l1_thresh=None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)
        latents, _ = self.prepare_latents(
            input_image, height, width, seed, tiled, tile_size, tile_stride,
        )

        prompt_emb_posi, prompt_emb_nega = self.prepare_prompts(
            prompt, image_embed, t5_sequence_length, negative_prompt, cfg_scale,
        )
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        patch_active = (
            rag_images is not None
            and self.patch_dit is not None
            and self.style_to_context is not None
        )

        if patch_active:
            # Pre-compute style tokens once (they do not depend on t).
            style_ctx, style_pool = self._compute_style_conditioning(
                rag_images.to(self.device), target_dtype=self.torch_dtype,
            )
            # Build patch-side conditioning by concatenation, mirroring
            # trainer.forward step 3e exactly.
            base_emb = prompt_emb_posi["prompt_emb"]
            patch_prompt_emb = {
                "prompt_emb": torch.cat(
                    [base_emb, style_ctx.to(base_emb.dtype)], dim=1,
                ),
                "pooled_prompt_emb": (
                    prompt_emb_posi["pooled_prompt_emb"]
                    + style_pool.to(prompt_emb_posi["pooled_prompt_emb"].dtype)
                ),
            }
            patch_prompt_emb["text_ids"] = torch.zeros(
                patch_prompt_emb["prompt_emb"].shape[0],
                patch_prompt_emb["prompt_emb"].shape[1],
                3,
                device=self.device,
                dtype=patch_prompt_emb["prompt_emb"].dtype,
            )
            tea_cache_kwargs = {"tea_cache": None}  # disable for patch path
        else:
            tea_cache_kwargs = {
                "tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh)
                if tea_cache_l1_thresh is not None else None
            }

        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            if patch_active:
                # Stream 1: base velocity (LoRA disabled).  Because the
                # in-place LoRA design makes self.dit and self.patch_dit
                # the same FluxDiT, we run patch_dit here too — the
                # context manager temporarily zeroes the LoRA delta.
                with self.patch_dit.disable_adapter():
                    e_0 = lets_dance_flux(
                        dit=self.patch_dit,
                        hidden_states=latents,
                        timestep=timestep,
                        **prompt_emb_posi,
                        **tiler_kwargs,
                        **extra_input,
                        **tea_cache_kwargs,
                    )
                # Stream 2: residual (LoRA enabled), full conditioning.
                S_phi = lets_dance_flux(
                    dit=self.patch_dit,
                    hidden_states=latents,
                    timestep=timestep,
                    **patch_prompt_emb,
                    **tiler_kwargs,
                    **extra_input,
                    **tea_cache_kwargs,
                )
                gate = self.correction_gate.to(e_0.dtype)
                noise_pred_posi = e_0 + (self.strength * gate) * S_phi
            else:
                noise_pred_posi = lets_dance_flux(
                    dit=self.dit,
                    hidden_states=latents,
                    timestep=timestep,
                    **prompt_emb_posi,
                    **tiler_kwargs,
                    **extra_input,
                    **tea_cache_kwargs,
                )

            if cfg_scale != 1.0:
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit,
                    controlnet=self.controlnet,
                    hidden_states=latents,
                    timestep=timestep,
                    **prompt_emb_nega,
                    **tiler_kwargs,
                    **extra_input,
                    **tea_cache_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], latents,
            )

            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, **tiler_kwargs)
        self.load_models_to_device([])
        return image


# ──────────────────────────────────────────────────────────────────────
#  Checkpoint loader
# ──────────────────────────────────────────────────────────────────────

def _load_zero_to_fp32(checkpoint_dir: str):
    """Dynamic import of the ``zero_to_fp32.py`` helper that DeepSpeed
    emits next to its checkpoint.  Returns the
    ``get_fp32_state_dict_from_zero_checkpoint`` function.
    """
    helper_path = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    spec = importlib.util.spec_from_file_location("rag_patch_zero_to_fp32", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_fp32_state_dict_from_zero_checkpoint


def load_rag_patch_state_dict(trainer, checkpoint_dir: str, tag: str = "checkpoint"):
    """Reconstruct the trained state (LoRA adapters + style projectors +
    correction_gate buffer) from a DeepSpeed ZeRO checkpoint directory
    and load it into ``trainer``.

    Mirrors the loading pattern in
    ``scripts/eval/test_rag_patch_controlled.sh``.
    """
    helper = _load_zero_to_fp32(checkpoint_dir)
    state_dict = helper(checkpoint_dir, tag=tag, exclude_frozen_parameters=True)
    missing, unexpected = trainer.load_state_dict(state_dict, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}
