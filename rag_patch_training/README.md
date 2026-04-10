# RAG Patch Training — Full Code Walkthrough

## 1. Big Picture (What & Why)

This project trains a **LoRA-based additive correction module** ("RAG Patch") for the **FLUX DiT** (Diffusion Transformer) text-to-image model used inside **Nexus-GenV2**. The goal: improve generated image quality by conditioning on **style features extracted from retrieved reference images** (RAG = Retrieval-Augmented Generation).

The core idea is a **two-stream architecture**:

| Stream | Model | Role | Trainable? |
|--------|-------|------|------------|
| Stream 1 | Base FLUX DiT | Text-conditioned velocity prediction (`e_0`) | Frozen |
| Stream 2 | Same DiT + LoRA adapters | Style-conditioned correction signal (`S_phi`) | Yes (LoRA only) |

Final prediction: `v_pred = e_0 + correction_gate * S_phi`, where `correction_gate` starts at **zero** (ControlNet-style zero-init) so training begins from baseline performance and gradually learns the correction.

---

## 2. File-by-File Breakdown

### `train_rag_patch.sh` — SLURM Launch Script

- Submits a SLURM job: 2 GPUs, 64 GB RAM, 16 CPUs, 2-day time limit.
- Activates the `nexus` conda environment (installs `peft`, `lightning`, `deepspeed`, `torchvision` if missing).
- Sets `PYTHONPATH` to include `Nexus-Gen/` and `DiffSynth-Studio/`.
- Launches training via `srun python rag_patch_training/train.py --config rag_patch_training/config.yaml`.

### `config.yaml` — Configuration

Key settings:

- **Models**: Nexus-GenV2 DiT, FLUX CLIP-L text encoder, T5-XXL, FLUX VAE.
- **Dataset**: 9,999-entry JSONL, 512×512 images, 3 RAG references per sample.
- **LoRA**: rank 16, alpha 16, no dropout.
- **Training**: lr=1e-5, 100 warmup steps, bf16-mixed precision, gradient checkpointing.
- **Trainer**: batch_size=1, accumulate_grad_batches=8, 2 GPUs → effective batch size = 16. DeepSpeed Stage 2 with optimizer offload. 100 epochs × 1000 steps/epoch.

### `build_dataset.py` — Dataset Builder

Builds the training JSONL by retrieving reference images from the MRAG database:

1. **Loads** metadata (`metadata.jsonl`) and two FAISS indices (`image.faiss`, `text.faiss`) containing SigLIP embeddings.
2. For each image in the DB, creates a **hybrid query** blending image similarity (70%) and text/caption similarity (30%).
3. Retrieves a shortlist of 50 candidates from both FAISS indices.
4. Re-ranks with **Maximal Marginal Relevance (MMR)** (λ=0.9 — mostly relevance, slight diversity) to select the top 3 references.
5. Writes one JSONL line per entry: `{"image": "...", "prompt": "...", "rag_images": [...]}`.

The actual dataset has **9,999 entries**, each with 3 retrieved reference images from the MRAG database of ~100k images.

### `dataset.py` — PyTorch Dataset

- `RAGPatchDataset`: Reads the JSONL, loads images as `[0,1]` tensors with resize/crop/flip transforms.
- `__len__` returns `steps_per_epoch` (1000), cycling through data via modular indexing.
- Each sample returns: `{image: (3,512,512), prompt: str, rag_images: (3,3,512,512)}`.
- `collate_fn`: Stacks tensors, collects prompts as a list of strings.

### `model.py` — Core Model (685 lines)

#### `VGG19StyleExtractor` (frozen)

- Uses a pretrained VGG-19 (ImageNet weights), truncated to `relu4_1`.
- Extracts feature maps at `relu{1,2,3,4}_1` (64, 128, 256, 512 channels).
- In the training loop these are reduced to a **1920-dim style vector** via channel-wise mean + std pooling (`(64+128+256+512) × 2 = 1920`).

#### `RAGPatchTrainer` (PyTorch Lightning Module)

The heart of the system.

**`__init__`** sets up:

1. **Frozen base pipeline**: Loads FLUX VAE, CLIP-L text encoder, T5-XXL text encoder, and the Nexus-GenV2 DiT via `ModelManager` + `NexusGenGenerationPipeline`.
2. **VGG-19 feature extractor** (frozen).
3. **LoRA wrapping**: Wraps the DiT **in-place** (no deepcopy — saves ~22 GB) with `peft.get_peft_model()`. Target modules: `a_to_qkv`, `b_to_qkv`, `a_to_out`, `b_to_out` (joint attention), `to_qkv_mlp`, `proj_out` (single blocks).
4. **Style projection layers** (trainable): Two small MLPs mapping VGG's 1920-dim style vector into FLUX's conditioning spaces — `1920 → 4096` (context tokens) and `1920 → 768` (pooled conditioning).
5. **Correction gate**: A single zero-initialized scalar parameter.

**`forward(x, prompts, rag_images)`** — the training step:

| Step | Description | Grad? |
|------|-------------|-------|
| **1. Latent prep** | VAE encodes target image → `z_0`. Random timestep `t` sampled per item. Noisy latent computed: `z_t = (1−σ)z_0 + σε`. VAE moved to GPU then back to CPU. | No |
| **2. Stream 1** | Text encoders (CLIP + T5) encode prompts → embeddings. Base DiT forward with **LoRA disabled** → `e_0`. Text encoders moved to GPU then CPU. | No |
| **3. Stream 2** | VGG-19 extracts features from RAG images. Mean+std pooling → 1920-dim style vec. Trainable MLPs project to FLUX conditioning format. **LoRA-enabled** DiT forward with style context → `S_phi`. | **Yes** |
| **4. Fusion & Loss** | `v_pred = e_0 + gate * S_phi`. Target = `ε − z_0` (flow-matching velocity). Per-sample MSE weighted by BSMNTW timestep importance weights → scalar loss. | Yes |

**Memory management**: VAE and text encoders are manually shuttled between CPU/GPU. The pipeline is hidden from DeepSpeed via `object.__setattr__` to prevent DS from trying to manage it. Only LoRA params + projectors + gate are tracked by the optimizer.

**`configure_optimizers`**: Uses `DeepSpeedCPUAdam` (required for ZeRO-Offload) with constant LR + 100-step warmup.

**`state_dict`**: Saves only trainable weights (LoRA adapters, style projectors, correction gate) — not the frozen 22 GB base model.

### `train.py` — Entry Point

- Parses CLI args, overrides with YAML config.
- Instantiates `RAGPatchTrainer` and `RAGPatchDataset`.
- Configures PyTorch Lightning `Trainer` with DeepSpeed Stage 2, 2 GPUs, bf16-mixed, gradient clipping (1.0), checkpoints every epoch.
- Calls `trainer.fit()`.

---

## 3. Training Flow Summary

```
MRAG DB (100k images + FAISS indices)
        │
        ▼  build_dataset.py (MMR retrieval)
dataset.jsonl (9,999 entries, each with 3 reference images)
        │
        ▼  dataset.py (load & transform)
    ┌──────────────────────────────────────────────────┐
    │  Per training step:                              │
    │                                                  │
    │  target_image ──► VAE ──► z_0 ──► add noise ──► z_t │
    │  prompt ──────► CLIP + T5 ──► text embeddings    │
    │  z_t + text ──► Base DiT (frozen) ──► e_0       │
    │                                                  │
    │  rag_images ──► VGG-19 (frozen) ──► style_vec   │
    │  style_vec ──► MLP projectors (trainable) ──►   │
    │         style conditioning                       │
    │  z_t + style ──► LoRA DiT (trainable) ──► S_phi │
    │                                                  │
    │  v_pred = e_0 + gate · S_phi                    │
    │  loss = MSE(v_pred, ε − z_0) × BSMNTW weights  │
    └──────────────────────────────────────────────────┘
```

## 4. Key Design Decisions

- **In-place LoRA** (no deepcopy): The base DiT and patch DiT share the same underlying weights. Stream 1 runs with `disable_adapter()`, Stream 2 runs with LoRA active. Saves ~22 GB RAM.
- **Zero-init correction gate**: At the start, `gate=0` so `v_pred = e_0` exactly — training begins from baseline quality and gradually learns to correct.
- **CPU offloading**: DeepSpeed Stage 2 offloads optimizer states to CPU. VAE/text encoders are manually shuttled to keep GPU memory within ~30 GB per GPU.
- **BSMNTW weighting**: Timestep importance weighting for the MSE loss — down-weights easy timesteps and up-weights informative ones.
- **MMR retrieval**: Reference images are selected to be relevant *and* diverse (λ=0.9 favors relevance with mild diversity).
