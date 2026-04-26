"""
Training entry-point for RAG Patch Training.

Usage
-----
    PYTHONPATH=/path/to/Nexus-Gen:/path/to/Nexus-Gen/DiffSynth-Studio \
    python rag_patch_training/train.py --config rag_patch_training/config.yaml

All heavy-lifting is in ``model.py``  (RAGPatchTrainer)  and
``dataset.py``  (RAGPatchDataset).
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger

# Ensure Nexus-Gen imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NEXUS_GEN_ROOT = os.path.join(_SCRIPT_DIR, "..", "Nexus-Gen")
_DIFFSYNTH_ROOT = os.path.join(_NEXUS_GEN_ROOT, "DiffSynth-Studio")
for _p in (_NEXUS_GEN_ROOT, _DIFFSYNTH_ROOT):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_patch_training.model import RAGPatchTrainer     # noqa: E402
from rag_patch_training.dataset import (                   # noqa: E402
    RAGPatchDataset,
    collate_fn,
)


# ──────────────────────────────────────────────────────────────────────────
#  CLI / Config
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA correction patch for FLUX DiT "
                    "conditioned on RAG style features.",
    )
    # -- config file (overrides defaults) --
    parser.add_argument(
        "--config", type=str,
        default="rag_patch_training/config.yaml",
        help="Path to YAML config (values override CLI defaults).",
    )

    # -- paths --
    parser.add_argument("--nexus_gen_root", type=str, default=None)
    parser.add_argument("--journeydb_dir", type=str, default=None,
                        help="Path to JourneyDB dataset directory (contains metadata.jsonl + images/).")
    parser.add_argument("--rag_db_dir", type=str, default=None,
                        help="Path to mrag-db directory (contains image.faiss, text.faiss, metadata.jsonl, images/).")
    parser.add_argument("--cache_path", type=str,
                        default="rag_patch_training/dataset.jsonl",
                        help="Path to cached dataset JSONL (built automatically on first run).")
    parser.add_argument("--output_path", type=str, default="workdirs/rag_patch")
    parser.add_argument(
        "--nexgen_decoder_path", type=str,
        default="models/Nexus-GenV2/generation_decoder.bin",
    )
    parser.add_argument(
        "--pretrained_text_encoder_path", type=str,
        default="models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    )
    parser.add_argument(
        "--pretrained_t5_path", type=str,
        default="models/FLUX/FLUX.1-dev/text_encoder_2",
    )
    parser.add_argument(
        "--pretrained_vae_path", type=str,
        default="models/FLUX/FLUX.1-dev/ae.safetensors",
    )

    # -- LoRA --
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # -- training hyper-parameters --
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--t5_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--torch_dtype_str", type=str, default="bf16",
                        choices=["bf16", "16", "32"])

    # -- data --
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_rag_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--random_flip", action="store_true", default=False)

    # -- Lightning trainer --
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_2",
                        choices=["auto", "ddp", "deepspeed_stage_2",
                                 "deepspeed_stage_3"])
    parser.add_argument("--num_devices", type=int, default=None,
                        help="Number of GPUs to use. Defaults to all visible GPUs "
                             "(torch.cuda.device_count()).")

    args = parser.parse_args()

    # Override with YAML config if present
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            setattr(args, key, value)

    return args


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

class LoRASaveCallback(Callback):
    """Save LoRA adapter weights + style projectors at the end of every epoch.

    Uses PEFT's save_pretrained so the checkpoint is small (only LoRA deltas)
    and can be loaded directly for inference without the full DeepSpeed state.
    Only rank-0 writes to disk.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        step = trainer.global_step
        save_dir = os.path.join(self.output_dir, f"epoch={epoch}-step={step}")
        os.makedirs(save_dir, exist_ok=True)

        # LoRA adapter weights (patch_dit is a PeftModel)
        pl_module.patch_dit.save_pretrained(os.path.join(save_dir, "lora"))

        # Style projectors + correction_gate buffer.
        # correction_gate is a non-trainable buffer (F2) but it is the user-facing
        # strength multiplier that the inference pipeline reads, so persisting it
        # alongside the per-epoch export lets `pipeline.load_rag_patch_state_dict`
        # restore the right value when bypassing the DeepSpeed ZeRO checkpoint.
        torch.save(
            {
                "style_to_context": pl_module.style_to_context.state_dict(),
                "style_to_pooled": pl_module.style_to_pooled.state_dict(),
                "correction_gate": pl_module.correction_gate.data.cpu(),
            },
            os.path.join(save_dir, "style_projectors.pt"),
        )
        print(
            f"[LoRASaveCallback] Saved checkpoint to {save_dir} "
            f"(correction_gate={float(pl_module.correction_gate.item()):.6f})"
        )


def main() -> None:
    import torch
    # Use TF32 on A100 Tensor Cores for float32 matmuls (free ~10% speedup)
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    print("=" * 60)
    print("  RAG Patch Training — Configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:35s}: {v}")
    print("=" * 60)

    # ---- Resolve Nexus-Gen root ----
    nexus_gen_root = args.nexus_gen_root
    if nexus_gen_root is None:
        nexus_gen_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Nexus-Gen")
        )

    # ---- Model ----
    model = RAGPatchTrainer(
        nexgen_decoder_path=args.nexgen_decoder_path,
        pretrained_text_encoder_path=args.pretrained_text_encoder_path,
        pretrained_t5_path=args.pretrained_t5_path,
        pretrained_vae_path=args.pretrained_vae_path,
        nexus_gen_root=nexus_gen_root,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        t5_sequence_length=args.t5_sequence_length,
        guidance_scale=args.guidance_scale,
        torch_dtype_str=args.torch_dtype_str,
    )

    # ---- Dataset ----
    assert args.journeydb_dir is not None, (
        "Please set --journeydb_dir or journeydb_dir in config.yaml"
    )
    assert args.rag_db_dir is not None, (
        "Please set --rag_db_dir or rag_db_dir in config.yaml"
    )
    dataset = RAGPatchDataset(
        journeydb_dir=args.journeydb_dir,
        rag_db_dir=args.rag_db_dir,
        cache_path=args.cache_path,
        num_rag_images=args.num_rag_images,
        image_size=args.image_size,
        steps_per_epoch=args.steps_per_epoch,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    # SLURM compatibility (mirrors fuzzy-engine pattern):
    #   * pin_memory=False — pin_memory's background thread can deadlock
    #     when CUDA was already initialised in the parent before fork.
    #   * persistent_workers=True (when num_workers > 0) — avoids re-spawning
    #     workers every epoch (each spawn re-imports the model module).
    #   * multiprocessing_context="spawn" — sidesteps the classic
    #     fork+CUDA hang on Linux DataLoaders.
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=(args.dataloader_num_workers > 0),
        multiprocessing_context=("spawn" if args.dataloader_num_workers > 0 else None),
    )

    # ---- DeepSpeed config (CPU offloading for memory) ----
    strategy = args.training_strategy
    if strategy == "deepspeed_stage_3":
        from lightning.pytorch.strategies import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,      # offload Adam states to CPU
            offload_parameters=True,     # offload params to CPU (stream to GPU)
            cpu_checkpointing=True,
            # Exclude frozen parameters from checkpoint saves (reduces checkpoint
            # size by not writing the frozen base DiT/VGG weights to disk).
            # Note: this does NOT prevent DeepSpeed from casting frozen params to
            # bf16 at engine init — that is handled in VGG19StyleExtractor.forward()
            # by dynamically matching input dtype to the actual weight dtype.
            exclude_frozen_parameters=True,
        )
    elif strategy == "deepspeed_stage_2":
        from lightning.pytorch.strategies import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
        )

    # ---- Logger (CSV — F6 diag/* metrics need somewhere to land) ----
    # Without an explicit logger, every self.log(...) call from RAGPatchTrainer
    # is silently dropped, so the F6 diagnostics (diag/S_phi_norm,
    # diag/rho_S_phi_target, etc.) would be invisible.  CSVLogger writes to
    # ${output_path}/logs/version_<N>/metrics.csv on rank 0 only.
    csv_logger = CSVLogger(save_dir=args.output_path, name="logs")

    # ---- Resolve number of GPUs ----
    n_gpus = args.num_devices if args.num_devices is not None else torch.cuda.device_count()
    n_gpus = max(1, n_gpus)

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=n_gpus,
        precision=args.precision,
        strategy=strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            LoRASaveCallback(output_dir=os.path.join(args.output_path, "checkpoints")),
        ],
        logger=csv_logger,
        log_every_n_steps=1,
        enable_progress_bar=False,   # SLURM has no TTY; tqdm spam wastes log file
        gradient_clip_val=1.0,
    )

    # ---- Launch ----
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
