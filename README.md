# fuzzy-engine — Nexus-Gen + RAG Patch Workspace

Research workspace for improving the **generation** and **editing** performance of the [Nexus-GenV2](src/Nexus-Gen/) unified VLM by training a lightweight **RAG-conditioned additive patch** on top of its frozen FLUX DiT.

The patch is a LoRA + style-projector branch that runs alongside the frozen base DiT at every denoising step:

$$
v_\text{pred}(z_t, t) \;=\; \epsilon_\theta(z_t,\,t,\,\text{prompt}) \;+\; \text{correction\_gate} \cdot \text{strength} \cdot S_\varphi(z_t,\,t,\,\text{prompt},\,\text{retrieved refs})
$$

with $S_\varphi$ trained by **residual regression** against $v^* - e_0^{\text{detach}}$ (no learnable gate during training; gate is a buffer used only at inference as a strength multiplier).

---

## Workspace layout

```
fuzzy-engine/
├── src/
│   ├── Nexus-Gen/              # upstream Nexus-GenV2 (read-only; do NOT edit)
│   │   ├── DiffSynth-Studio/   # FLUX DiT + scheduler + lets_dance_flux
│   │   ├── modeling/{ar,decoder}/
│   │   ├── models/Nexus-GenV2/ # Qwen2.5-VL + adapter weights (4 shards + bin)
│   │   └── retrieve_mrag.py    # MRAGRetriever (FAISS hybrid + MMR)
│   ├── rag_patch_training/     # all trainable code lives here
│   │   ├── model.py            # RAGPatchTrainer (Lightning module)
│   │   ├── pipeline.py         # NexusGenRAGPatchPipeline (inference fusion)
│   │   ├── dataset.py          # RAGPatchDataset + JSONL cache builder
│   │   ├── build_dataset.py    # standalone retrieval-cache builder
│   │   ├── train.py            # entry-point
│   │   ├── train_rag_patch.sh  # SLURM launcher
│   │   ├── config.yaml         # active hyperparameters
│   │   └── README.md           # full code walkthrough
│   └── mmdetection/            # used by some benchmarks
│
├── scripts/
│   ├── setup/                  # env install + benchmark setup
│   ├── building/               # dataset / mrag-db construction
│   └── eval/                   # controlled tests + benchmark runs
│
├── benchmarks/                 # geneval, T2I-CompBench, ImgEdit, drawbench, tifa, ...
├── data/                       # datasets (not tracked); journeydb_dataset/ lives here
├── experiments/                # one folder per training run; checkpoints + lightning_logs
├── logs/                       # SLURM .out/.err archive (per-script subfolders)
├── rag_patch_results_controlled/  # outputs of test_rag_patch_controlled.sh
├── docs/                       # design + revision notes
└── requirements.txt
```

Imports assume `fuzzy-engine/` is on `PYTHONPATH` along with `src/Nexus-Gen/` and `src/Nexus-Gen/DiffSynth-Studio/`. The training launcher sets these automatically.

---

## What's where to read first

| If you want to … | Read |
|---|---|
| Understand the patch end-to-end | [src/rag_patch_training/README.md](src/rag_patch_training/README.md) — full code walkthrough |
| Understand **why** the original training collapsed and what was changed | [docs/revision.md](docs/revision.md) — diagnostic write-up with the gate-collapse derivation |
| Understand the F1–F6 fixes and the post-F6 addendum | [docs/rag_patch_revision.md](docs/rag_patch_revision.md) — what changed, how to run, verification ladder, caveats |
| Track or add a new training run | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — per-run procedure + log table |
| Understand the workspace reorg history | [docs/REFACTOR.md](docs/REFACTOR.md) |

---

## Architecture summary (post-F1–F6)

| Stream | Conditioning | Weights | Output |
|---|---|---|---|
| **1 (frozen)** | Qwen2.5-VL → 81-token `image_embed` via the trained adapter | LoRA disabled | $e_0$ |
| **2 (trainable)** | `[image_embed (81 tokens), 4·k VGG-style tokens]` (k=3 ⇒ 12 style tokens) | LoRA enabled | $S_\varphi$ |

- **Training loss**: $\mathcal{L}_\text{res} = \mathbb{E}\big[w(t)\cdot \|S_\varphi - (v^* - e_0^{\text{detach}})\|^2\big]$ where $v^* = \varepsilon - z_0$ (flow-matching velocity).
- **Inference fusion**: per denoising step, `noise_pred = e_0 + (strength · correction_gate) · S_phi` via [pipeline.NexusGenRAGPatchPipeline](src/rag_patch_training/pipeline.py).
- **Diagnostics** logged to CSV every step: `train_loss`, `lr`, plus `diag/{S_phi_norm, base_err_norm, residual_mse, rho_S_phi_target}` — the cosine alignment $\rho_t$ is the key falsifier of gate-collapse-style pathology under the new objective.

---

## Quick start

### Environment

```bash
# One-time
bash scripts/setup/setup_nexus_env.sh
# OR pip install from the lockfile in an existing venv
pip install -r requirements.txt
```

### Build dataset cache (one-time, ~minutes)

The first training run rebuilds the cache automatically. To pre-build:

```bash
PYTHONPATH=src:src/Nexus-Gen:src/Nexus-Gen/DiffSynth-Studio \
python src/rag_patch_training/build_dataset.py --config src/rag_patch_training/config.yaml
```

Cache lands at `src/rag_patch_training/dataset.jsonl`. (If `dataset.jsonl.stale-pre-pathfix` is sitting next to it, that's the pre-path-audit cache preserved as a fallback; the live cache is the one without the suffix.)

### Train

```bash
sbatch src/rag_patch_training/train_rag_patch.sh
```

- **SLURM stdout/stderr** → `logs/rag_patch_training/train_<jobid>.{out,err}`
- **Lightning metrics** → `experiments/workdirs/rag_patch/logs/version_<N>/metrics.csv`
- **Per-epoch checkpoints** → `experiments/workdirs/rag_patch/checkpoints/epoch=<E>-step=<S>/`
- **DeepSpeed ZeRO checkpoint** (full state) → `experiments/workdirs/rag_patch/lightning_logs/version_<N>/checkpoints/.../`

To run on a single GPU for debugging: `srun python src/rag_patch_training/train.py --config src/rag_patch_training/config.yaml --num_devices 1`.

### Controlled A/B generation

```bash
sbatch scripts/eval/test_rag_patch_controlled.sh    # SLURM array of 5 prompts
```

Outputs to `rag_patch_results_controlled/{baseline,patched,reports}/`. Reports include `correction_gate`, `strength`, and the mean per-pixel L1 between baseline and patched.

### Larger benchmarks

`scripts/eval/run_nexus_test.sh` and the per-benchmark setups under `benchmarks/` (e.g. `geneval`, `T2I-CompBench`, `ImgEdit`).

---

## Key conventions

- **Don't modify `src/Nexus-Gen/`.** Treat it as an upstream dependency; everything new goes in `src/rag_patch_training/`. The inference pipeline subclasses `NexusGenGenerationPipeline` rather than editing it.
- **Per-experiment artifacts** live in `experiments/<run-name>/` (see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)). Don't dump SLURM logs at the workspace root.
- **SLURM logs** route to `logs/<script-name>/` (e.g. `logs/rag_patch_training/`, `logs/setup/`).
- **Datasets** under `data/` (e.g. `data/journeydb_dataset/`); large files stay out of git.

---

## Status

| | |
|---|---|
| Architecture rework (F1–F6) | ✅ shipped — Qwen Stream 1, residual loss, multi-token Stream 2, inference pipeline, diagnostics |
| Path audit (post-F6) | ✅ shipped — launcher + config paths under `src/` |
| Runtime addendum (A1–A7) | ✅ shipped — stale-cache invalidated, SLURM logs redirected, CSVLogger wired, spawn-safe DataLoader, `--num_devices`, `correction_gate` in per-epoch export, offline + DS-quiet env vars |
| First post-rework training run | ⏳ pending resubmission |
| Generation A/B + strength sweep | ⏳ pending training |

Open caveats are tracked in [docs/rag_patch_revision.md §5](docs/rag_patch_revision.md): per-step Qwen.generate cost, pre-F3 checkpoint incompatibility, dataset-circularity ceiling, VRAM impact, and the `correction_gate` default flip from 0.0 → 1.0.
