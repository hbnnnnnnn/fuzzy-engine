# RAG Patch — F1-F6 Revision

End-to-end rework of [`src/rag_patch_training/`](../src/rag_patch_training/) to align the implementation with [`docs/approach.md`](approach.md) and fix the root causes behind the observed "baseline ≡ patched" generation outcome.

Plan file: [`/media02/nthuy/.claude/plans/in-docs-approach-md-is-my-clever-hamming.md`](../../.claude/plans/in-docs-approach-md-is-my-clever-hamming.md).
Diagnostic write-up that motivated the rework: [`revision.md`](../revision.md).

---

## 1. Problem recap

Three intertwined defects caused the trained patch to be visually indistinguishable from the base:

1. **Wrong Stream-1 conditioning.** Code used FLUX's CLIP-L + T5-XXL path. Nexus-GenV2's DiT was actually fine-tuned on the **81-token Qwen2.5-VL output** routed through a `Linear(3584→4096) → LN → ReLU → Linear(4096→4096) → LN` adapter. So `e_0` during training did not match what the deployed pipeline produces.
2. **Gate-collapse objective.** With $v_\text{pred} = e_0 + g\cdot S_\varphi$ and Stream 1 receiving the ground-truth conditioning,
    $$\mathcal{L} = \|e_0-v^*\|^2 + 2g\langle S_\varphi, e_0-v^*\rangle + g^2\|S_\varphi\|^2.$$
   The expected linear term vanishes; the optimum is $g\to 0$. Observed $g = 0.017$ ⇒ patch contributes ~1.7% ⇒ images look identical.
3. **Missing inference wiring.** No script actually computed $e_0 + g\cdot S_\varphi$ at every denoising step.

The plan fixed each in numbered steps F1–F6.

---

## 2. Changes by step

All new code lives inside [`src/rag_patch_training/`](../src/rag_patch_training/); nothing under `src/Nexus-Gen/` was modified.

### F1 — Stream 1 uses Qwen2.5-VL → adapter

**File**: [`src/rag_patch_training/model.py`](../src/rag_patch_training/model.py)

- Added imports: `Qwen2_5_VLForConditionalGeneration`, `Qwen2_5_VLProcessor`, `AutoConfig`, plus the `NEXUS_GEN_EN_TEMPLATE` constant.
- New `__init__` parameter `qwen_ckpt_path` (default `"models/Nexus-GenV2"`); `pretrained_t5_path` retained as deprecated/unused for backward compat.
- Removed the T5-XXL `load_model_from_huggingface_folder` call. `ModelManager` only loads CLIP-L + VAE now.
- Extract the adapter weights from `generation_decoder.bin` (keys prefixed with `adapter.`) before the state dict is freed.
- New section **1e**: load Qwen AR model + processor (`Qwen2_5_VLProcessor.from_pretrained`); rebuild the adapter `nn.Sequential` and `load_state_dict` it. Both stored via `object.__setattr__` so DeepSpeed does not try to manage them.
- `forward` Stream 1 rewritten: per prompt, format with `NEXUS_GEN_EN_TEMPLATE`, call `qwen_ar.generate(..., generation_image_grid_thw=[[1,18,18]])`, take `output_image_embeddings` (shape `(1, 81, 3584)`), pass through `ar_adapter` → `(1, 81, 4096)`. Concatenate across the batch. Pooled conditioning still via CLIP-L on `[""] * B` (matches [`Nexus-Gen/train/decoder/generation_trainer.py:86`](../src/Nexus-Gen/train/decoder/generation_trainer.py#L86)). `text_ids = zeros(B, 81, 3)`.

### F2 — Residual-regression objective (no learnable gate)

**File**: [`src/rag_patch_training/model.py`](../src/rag_patch_training/model.py)

- `correction_gate` converted from `nn.Parameter(zeros(1))` to `register_buffer("correction_gate", torch.ones(1), persistent=True)`. Default value 1.0 (was 0.0). Not in optimizer; saved by `state_dict()` outside the `GatheredParameters` scope.
- Loss block replaced. Old:
    $$\mathcal{L} = w(t)\cdot \|(e_0 + g\,S_\varphi) - v^*\|^2$$
  New:
    $$\mathcal{L}_\text{res} = w(t)\cdot \big\|S_\varphi - \big(v^* - e_0^{\,\text{detach}}\big)\big\|_2^2,
       \qquad v^* = \varepsilon - z_0.$$
  Target $v^* - e_0$ is non-degenerate by construction; `S_\varphi \to 0` is no longer the loss minimum.
- `configure_optimizers` no longer includes `correction_gate` in trainable params. `state_dict` saves it as a buffer alongside the LoRA adapters and style projectors.
- Module + class docstrings updated.

### F3 — Multi-token + base-embedded Stream 2 conditioning

**File**: [`src/rag_patch_training/model.py`](../src/rag_patch_training/model.py)

- New module-level constants `VGG_LAYER_CHANNELS = (64, 128, 256, 512)` and `VGG_LAYER_STAT_DIMS = (128, 256, 512, 1024)`.
- `style_to_context` is now an `nn.ModuleList` of 4 per-VGG-layer projectors: `Linear(2C_ℓ, 4096) → SiLU → Linear(4096, 4096)`. Per-layer weights since input dims differ.
- `style_to_pooled` unchanged in shape; output is now **summed with** the CLIP-pooled-empty vector (instead of replacing it).
- Stream 2 forward (steps 3b–3f) rewritten:
  - Per `(layer, ref)` pair → one 4096-dim token. Stacked → `(B, K=4·k, 4096)` (K = 12 for k = 3).
  - Pooled: ref-averaged 1920-dim flat → `(B, 768)`. Reuses per-layer stats; no recomputation.
  - **Patch DiT conditioning** = concat`([base image_embed (81 tokens), style_context_tokens (K)])` → `(B, 81+K, 4096)`. Pooled = CLIP-pooled-empty + style-pooled. `text_ids = zeros(B, 81+K, 3)`.
  - The patch DiT now sees strictly ≥ Stream 1's information (semantic Qwen embed + retrieval-derived style/structure), so under the F2 objective $S_\varphi$ has a learnable, non-degenerate target.

### F4 — Inference pipeline that actually fuses the patch

**File**: [`src/rag_patch_training/pipeline.py`](../src/rag_patch_training/pipeline.py) (new, ~280 lines)

- `class NexusGenRAGPatchPipeline(NexusGenGenerationPipeline)`. Static `from_trainer(trainer, strength=1.0)` reuses the trainer's already-loaded modules (no duplicate weight loading; trainer must outlive pipeline).
- `_compute_style_conditioning(rag_images, target_dtype)` mirrors trainer steps 3b–3d exactly. Computed **once per call**, not per timestep.
- `__call__` override accepts a new `rag_images: Optional[Tensor]` of shape `(B, k, 3, H, W)` in $[0,1]$. When provided, the denoising loop computes per timestep:
    1. `e_0 = lets_dance_flux(self.patch_dit, ..., **prompt_emb_posi)` under `with self.patch_dit.disable_adapter()` (LoRA delta = 0; equivalent to base DiT because of in-place LoRA wrapping).
    2. `S_phi = lets_dance_flux(self.patch_dit, ..., **patch_prompt_emb)` (LoRA active; full conditioning).
    3. `noise_pred_posi = e_0 + (self.strength * self.correction_gate) * S_phi`.
  When `rag_images is None`, falls back to the parent's single-stream behavior (useful for A/B with identical seeds). TeaCache is disabled when the patch path is active (would cache stale outputs across LoRA enable/disable flips).
- `load_rag_patch_state_dict(trainer, ckpt_dir, tag)` dynamically imports the `zero_to_fp32.py` helper DeepSpeed emits next to checkpoints, reconstructs the trained state, and `strict=False`-loads it into the trainer.

### F5 — Controlled eval generates real images

**File**: [`scripts/eval/test_rag_patch_controlled.sh`](../scripts/eval/test_rag_patch_controlled.sh)

- SBATCH header + 5-prompt array unchanged.
- New `mkdir -p ${OUTPUT_DIR}/{baseline,patched}` directories.
- Python heredoc completely rewritten:
  - Helper `compute_image_embed(trainer, prompt, device)` runs Qwen + adapter once per prompt to produce `image_embed (1, 81, 4096)`. Mirrors trainer Stream 1.
  - Instantiate `RAGPatchTrainer`, then `load_rag_patch_state_dict(...)` from the DeepSpeed ZeRO checkpoint. Falls back to legacy export-dir format with `strict=False` (warns on missing keys — pre-F3 checkpoints have a different `style_to_context` key shape).
  - Build `pipe = NexusGenRAGPatchPipeline.from_trainer(model, strength=1.0)`.
  - Retrieve 3 refs via `MRAGRetriever`, resize to target res, stack to `(1, 3, 3, H, W)` in $[0,1]$.
  - Generate **twice with the same seed**: baseline (`rag_images=None`) and patched (`rag_images=ref_tensors`).
  - Save PNGs to `baseline/` and `patched/`. Compute mean per-pixel $|baseline - patched|$ in $[0,1]$ as a quick sanity scalar. Report includes `correction_gate`, `strength`, and the pixel delta.

### F6 — Training budget + diagnostic logging

**Files**: [`src/rag_patch_training/config.yaml`](../src/rag_patch_training/config.yaml), [`src/rag_patch_training/model.py`](../src/rag_patch_training/model.py)

`config.yaml`:
- Added `qwen_ckpt_path: models/Nexus-GenV2`.
- Removed active `pretrained_t5_path` line (kept comment marking it deprecated since F1).
- `max_epochs: 5 → 10`. Comment notes: with the current per-epoch step count this is still ~1.25k optimizer steps; to hit the plan's ≥ 10k target, raise `steps_per_epoch` to ~8000 or `max_epochs` to ~80 (Qwen.generate per step is the new bottleneck).

`model.py` — new diagnostics block in `forward` (before `return loss`), gated by `self.training and self.trainer is not None`:

| Log key | Quantity | What to look for |
|---|---|---|
| `diag/S_phi_norm` | $\mathbb{E}\|S_\varphi\|_2$ | Should rise from ~0 as projectors warm up. Stays at 0 ⇒ patch dead. |
| `diag/base_err_norm` | $\mathbb{E}\|e_0 - v^*\|_2$ | Headroom for the residual learner. Near 0 ⇒ base saturated, nothing to learn. |
| `diag/residual_mse` | unweighted $\mathbb{E}\|S_\varphi - r^*\|_2^2$ | Direct comparable across BSMNTW changes; should fall. |
| `diag/rho_S_phi_target` | $\mathbb{E}\frac{\langle S_\varphi, r^*\rangle}{\|S_\varphi\|\,\|r^*\|}$ | **Key falsifier**: under F2/F3 should drift toward +1. Stays near 0 ⇒ Stream 2 uninformative even after F3. |

Where $r^* = v^* - e_0^\text{detach}$ is the residual target.

### Path audit (post-F6)

The repo was reorganised under a `src/` root sometime between the original code and the F1–F6 work. After F6 several launcher paths still referenced the old layout. Fixed in [`src/rag_patch_training/train_rag_patch.sh`](../src/rag_patch_training/train_rag_patch.sh): `NEXUS_DIR="${NDBAO_DIR}/Nexus-Gen"` → `${SRC_DIR}/Nexus-Gen` (with `SRC_DIR="${NDBAO_DIR}/src"`); `srun python rag_patch_training/...` → `srun python src/rag_patch_training/...`; `PYTHONPATH` rebased on `SRC_DIR`. Fixed in [`src/rag_patch_training/config.yaml`](../src/rag_patch_training/config.yaml): `journeydb_dir: /media02/nthuy/ndbao/journeydb_dataset` → `/media02/nthuy/ndbao/data/journeydb_dataset`; `rag_db_dir` → `src/Nexus-Gen/mrag-db`; `cache_path` → `src/rag_patch_training/dataset.jsonl`; `output_path: workdirs/rag_patch` → `experiments/workdirs/rag_patch` (matches existing checkpoint root).

---

## 2b. Addendum — runtime fix + fuzzy-engine infra integration

After the path-audit changes the first SLURM training run still failed at step 0 with `FileNotFoundError: '/media02/nthuy/ndbao/journeydb_dataset/images/00/...'`. Diagnosis: the on-disk JSONL cache at [`src/rag_patch_training/dataset.jsonl`](../src/rag_patch_training/dataset.jsonl) (49 MB) stored **absolute** paths frozen from the pre-`src/` layout for both `image` and `rag_images`. [`dataset.py:184, 250`](../src/rag_patch_training/dataset.py#L184) write paths absolute via `os.path.join(jdb_images_dir, …)`; [`dataset.py:306-308`](../src/rag_patch_training/dataset.py#L306-L308) reuses the cache unconditionally if non-empty. The corrected `config.yaml` paths therefore had no effect — the loader read the stale cache.

A separate review pulled in [`ndbao/fuzzy-engine/`](../fuzzy-engine/) — the snapshot from the previous cluster that produced the original gate-collapse training. Architecturally fuzzy-engine is **pre-F1/F2/F3** (T5/CLIP Stream 1, single-token style, learnable gate with hybrid `loss_fusion + 0.5·loss_patch`); the F1–F6 work strictly supersedes it and was kept. Operationally fuzzy-engine had several SLURM-stability fixes that hadn't made it into the current code.

### A1. Invalidate stale dataset cache

`mv src/rag_patch_training/dataset.jsonl{,.stale-pre-pathfix}`. On the next run, [`dataset.py:306`](../src/rag_patch_training/dataset.py#L306) sees the cache missing and rebuilds via `build_dataset_jsonl(...)` against the corrected `config.yaml` paths. The renamed file is preserved as a fallback. Cost: one full retrieval pass (FAISS over mrag-db × ~10k JourneyDB samples → minutes).

### A2. SLURM logs to `logs/rag_patch_training/`

`mkdir -p /media02/nthuy/ndbao/logs/rag_patch_training`. In [`train_rag_patch.sh`](../src/rag_patch_training/train_rag_patch.sh):

```diff
- #SBATCH --output=slurm_rag_patch_%j.out
- #SBATCH --error=slurm_rag_patch_%j.err
+ #SBATCH --output=/media02/nthuy/ndbao/logs/rag_patch_training/train_%j.out
+ #SBATCH --error=/media02/nthuy/ndbao/logs/rag_patch_training/train_%j.err
```

Absolute paths because SBATCH headers are parsed before any `cd` in the script body.

### A3. CSVLogger so F6's `diag/*` metrics land somewhere

The original [`train.py`](../src/rag_patch_training/train.py) passed `logger=None` to `pl.Trainer`, so every `self.log(...)` from F6 — including the four `diag/*` falsifiers — was silently dropped. Fixed:

```python
from lightning.pytorch.loggers import CSVLogger
csv_logger = CSVLogger(save_dir=args.output_path, name="logs")
trainer = pl.Trainer(
    ...
    logger=csv_logger,
    log_every_n_steps=1,            # (was 5; F6 metrics need denser sampling)
    enable_progress_bar=False,      # SLURM has no TTY
    ...
)
```

Output lands at `experiments/workdirs/rag_patch/logs/version_<N>/metrics.csv` on rank 0.

### A4. SLURM-safe DataLoader (avoid CUDA-fork deadlock)

Three changes in the `DataLoader(...)` call:

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    ...
    pin_memory=False,                                   # was True
    persistent_workers=(args.dataloader_num_workers > 0),
    multiprocessing_context=(
        "spawn" if args.dataloader_num_workers > 0 else None
    ),
)
```

`pin_memory=False` because the pin_memory background thread can deadlock when CUDA is initialised in the parent before fork; `multiprocessing_context="spawn"` sidesteps the classic fork+CUDA hang on Linux; `persistent_workers` avoids re-spawning each epoch. Mirrors [`fuzzy-engine/rag_patch_training/train.py`](../fuzzy-engine/rag_patch_training/train.py).

### A5. `--num_devices` CLI flag

Replaced hardcoded `devices=2` in `pl.Trainer(...)`:

```python
parser.add_argument("--num_devices", type=int, default=None)
...
n_gpus = args.num_devices if args.num_devices is not None else torch.cuda.device_count()
n_gpus = max(1, n_gpus)
trainer = pl.Trainer(..., devices=n_gpus, ...)
```

Lets a single-GPU debug run work without editing source: `--num_devices 1`.

### A6. Persist `correction_gate` in `LoRASaveCallback`

The per-epoch export at `experiments/workdirs/rag_patch/checkpoints/epoch=*/style_projectors.pt` previously held only `style_to_context` and `style_to_pooled`. Added `"correction_gate": pl_module.correction_gate.data.cpu()` to the dict, and a one-line print of its current value. Pairs with [`pipeline.load_rag_patch_state_dict`](../src/rag_patch_training/pipeline.py) so the export-format fallback in [`scripts/eval/test_rag_patch_controlled.sh`](../scripts/eval/test_rag_patch_controlled.sh) can restore strength even without the full DeepSpeed ZeRO directory.

### A7. Offline + DeepSpeed-quiet env vars

Appended to [`train_rag_patch.sh`](../src/rag_patch_training/train_rag_patch.sh) right after the existing `TRITON_CACHE_DIR` block:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
```

`TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE` prevent the first `from_pretrained` call from timing out on offline SLURM compute nodes. `DS_BUILD_OPS=0` skips DeepSpeed's CUDA-extension JIT compile at import (the ZeRO-2 weight sharding path doesn't need it).

### Things deliberately NOT pulled from fuzzy-engine

- T5/CLIP Stream-1 path — superseded by F1.
- Hybrid `loss_fusion + 0.5·loss_patch` — F2's pure residual loss is strictly better (no gate-collapse term in the loss at all).
- 1-token style projection — superseded by F3.
- `learning_rate: 3e-5` from `config_v2.yaml` — gradient magnitudes differ between fuzzy-engine's hybrid and F2's residual loss; the optimal LR likely differs. Keep `1e-5` as default; revisit if `diag/residual_mse` plateaus.

### Files touched in the addendum

| File | Actions |
|---|---|
| [`src/rag_patch_training/train_rag_patch.sh`](../src/rag_patch_training/train_rag_patch.sh) | A2 (SBATCH paths), A7 (env vars) |
| [`src/rag_patch_training/train.py`](../src/rag_patch_training/train.py) | A3 (CSVLogger), A4 (DataLoader), A5 (`--num_devices`), A6 (`correction_gate` save) |
| Renamed: `src/rag_patch_training/dataset.jsonl` → `dataset.jsonl.stale-pre-pathfix` | A1 |
| Created: `/media02/nthuy/ndbao/logs/rag_patch_training/` | A2 |

No edits to `model.py`, `pipeline.py`, `config.yaml`, or anywhere under `src/Nexus-Gen/`.

### Verification (post-addendum)

Smoke checks all green:
- `dataset.jsonl.stale-pre-pathfix` is the only cache on disk; live `dataset.jsonl` absent, so next run rebuilds.
- `logs/rag_patch_training/` exists.
- `bash -n train_rag_patch.sh` — OK.
- `train.py` PARSE OK; structural assertions for `CSVLogger`, `persistent_workers`, `multiprocessing_context`, `pin_memory=False`, `num_devices`, `enable_progress_bar=False`, `log_every_n_steps=1`, `correction_gate` all present; full module IMPORT OK in the `nexus` conda env.

Still pending: actual `sbatch` resubmission. Watch `logs/rag_patch_training/train_<jobid>.out` for `Cache not found at .../dataset.jsonl, building dataset ...` followed by `Cached <N> entries → ...`. Then `metrics.csv` should accumulate the four `diag/*` columns.

---

## 3. How to run

### 3.1 Train

The launcher script is unchanged; the new config is picked up automatically.

```bash
sbatch /media02/nthuy/ndbao/src/rag_patch_training/train_rag_patch.sh
```

Or interactively:

```bash
cd /media02/nthuy/ndbao
source ~/miniconda3/bin/activate nexus
export PYTHONPATH=src/Nexus-Gen:src/Nexus-Gen/DiffSynth-Studio:src
python src/rag_patch_training/train.py --config src/rag_patch_training/config.yaml
```

**SLURM stdout/stderr** land under `/media02/nthuy/ndbao/logs/rag_patch_training/train_<jobid>.{out,err}` (per addendum A2).

**Lightning metrics** land under `experiments/workdirs/rag_patch/logs/version_<N>/metrics.csv` — the four F6 `diag/*` keys plus `train_loss` and `lr` are written every step (CSVLogger wired in A3).

**Per-epoch checkpoint exports** land under `experiments/workdirs/rag_patch/checkpoints/epoch=<E>-step=<S>/` and now include `correction_gate` inside `style_projectors.pt` (A6) so the eval script's fallback loader can restore the strength multiplier without needing the DeepSpeed ZeRO directory.

### 3.2 Inference / controlled A/B

```bash
sbatch /media02/nthuy/ndbao/scripts/eval/test_rag_patch_controlled.sh
```

The SLURM array runs all 5 prompts. Outputs land at:

- `rag_patch_results_controlled/baseline/task<N>_<stem>.png`
- `rag_patch_results_controlled/patched/task<N>_<stem>.png`
- `rag_patch_results_controlled/reports/task<N>_<stem>.txt`

The report file lists `correction_gate`, `strength`, and the mean per-pixel L1 delta between baseline and patched. Eyeball the two PNGs side-by-side.

### 3.3 Single-prompt programmatic use

```python
import sys
sys.path += ["src/Nexus-Gen", "src/Nexus-Gen/DiffSynth-Studio", "src"]

import torch
from rag_patch_training.model import RAGPatchTrainer, NEXUS_GEN_EN_TEMPLATE
from rag_patch_training.pipeline import (
    NexusGenRAGPatchPipeline, load_rag_patch_state_dict,
)

device = "cuda:0"

trainer = RAGPatchTrainer(
    nexus_gen_root="src/Nexus-Gen",
    torch_dtype_str="bf16",
)
trainer.eval().to(device)

# Optional: load a trained checkpoint
load_rag_patch_state_dict(trainer, "experiments/workdirs/rag_patch/.../checkpoint")

# 1. AR forward to get the 81-token image_embed
trainer.qwen_ar.to(device); trainer.ar_adapter.to(device)
prompt = "A snowy mountain landscape …"
text = trainer.qwen_processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "text", "text": NEXUS_GEN_EN_TEMPLATE.format(prompt)}
    ]}],
    tokenize=False, add_generation_prompt=True,
)
ar_inputs = trainer.qwen_processor(text=[text], padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    ar = trainer.qwen_ar.generate(
        **ar_inputs, max_new_tokens=1024, return_dict_in_generate=True,
        generation_image_grid_thw=torch.tensor([[1, 18, 18]], device=device),
    )
    image_embed = trainer.ar_adapter(
        ar["output_image_embeddings"].to(dtype=trainer.pipe.torch_dtype)
    )
trainer.qwen_ar.cpu(); trainer.ar_adapter.cpu()

# 2. Build the pipeline + run
pipe = NexusGenRAGPatchPipeline.from_trainer(trainer, strength=1.0)

# rag_images: (1, k, 3, H, W) in [0, 1]; rag_images=None → baseline
image = pipe(
    prompt="", image_embed=image_embed, rag_images=rag_tensor,
    height=512, width=512, num_inference_steps=30,
    cfg_scale=3.0, embedded_guidance=3.5, seed=42,
)
image.save("out.png")
```

---

## 4. Verification ladder (recommended order)

Per the plan §Verification:

1. **Shape sanity (~minutes).** Instantiate trainer, B=1 forward, assert `e_0.shape == S_phi.shape == z_0.shape`, `image_embed.shape == (B, 81, 4096)`, all bf16.
2. **Diagnostic run (1 epoch ≈ 125 steps).** Watch `diag/S_phi_norm` rise, `diag/residual_mse` fall, `diag/rho_S_phi_target` drift toward +1. If all three move, the F2/F3 reframing worked. If `rho` stays ≈ 0, Stream 2 is uninformative even with semantic conditioning — that points at the dataset (refs near-circular with target).
3. **Full training (≥ 10k optimizer steps).** Bump `steps_per_epoch` or `max_epochs` accordingly; mind the per-step Qwen.generate cost.
4. **Generation A/B** via [`scripts/eval/test_rag_patch_controlled.sh`](../scripts/eval/test_rag_patch_controlled.sh). Pass criterion: visible, coherent differences between baseline and patched outputs that reflect the retrieved refs' style traits.
5. **Strength sweep.** With fixed prompt/seed, generate at `strength ∈ {0.0, 0.5, 1.0, 1.5}`. Pass criterion: monotonic, interpretable progression; `strength = 0` reproduces baseline bit-exactly (set `pipe.strength = 0.0` → `e_0 + 0·S_phi = e_0`).
6. **Benchmark.** Run a Nexus-Gen-supported eval (`benchmarks/geneval`, `benchmarks/T2I-CompBench`) base vs. patched.

---

## 5. Caveats

- **Per-step Qwen.generate is autoregressive and expensive.** F1 calls it inline on every training step (one prompt at a time). For batch=1 that's a single AR run per step, but with tens-of-thousands-of-steps budget this dominates wall clock. If it becomes a bottleneck, pre-cache image embeddings per prompt (the dataset is fixed, so this is a one-time cost) and switch the dataset to return cached `image_embed` tensors.
- **Pre-F3 checkpoints don't load cleanly into the F3 model** — `style_to_context` was a single `nn.Sequential`; it's now a 4-element `nn.ModuleList` with different per-layer input dims. The F5 fallback path uses `strict=False` and reports missing keys, but the loaded weights will be effectively garbage. Train fresh after F3.
- **Dataset circularity not addressed.** Per user's call, retrieval still uses 70% image / 30% text MMR ($\lambda = 0.9$). If `diag/rho_S_phi_target` plateaus near 0 after F2/F3 are in place, that's the next thing to revisit.
- **VRAM**: peak per-GPU bumps from ~28-30 GB to ~30-32 GB after F1's transient Qwen-on-GPU spike. Still within 40 GB. CPU RAM usage rises ~15 GB when Qwen is offloaded — within the existing `--mem=64G` SLURM budget.
- The `correction_gate` buffer defaults to **1.0** (was 0.0). Pre-F2 checkpoints with `correction_gate ≈ 0.017` saved as a parameter will load fine into the buffer, but during inference `noise_pred_posi = e_0 + (strength * 0.017) * S_phi` makes the patch nearly invisible — set `pipe.strength = 1.0 / 0.017 ≈ 58.8` if you want to debug an old checkpoint at full effect, or just retrain.
