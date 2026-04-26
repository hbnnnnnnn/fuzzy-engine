# T2I Model Evaluation Suite

Generate images and evaluate across **all benchmarks** (GenEval, TIFA, DrawBench,
T2I-CompBench) for models from the Nexus-Gen paper (Table 2) + SDXL.

**Resource-aware**: Scripts automatically detect GPU VRAM and skip models that
exceed available resources or lack public weights.

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8 GB VRAM (24 GB recommended for all models)
- **Disk**: ~80 GB for all model weights (stored in `model_cache/`)

## Model Compatibility Matrix

| Model | HuggingFace ID | VRAM (est.) | Precision | Status |
|-------|---------------|-------------|-----------|--------|
| **SDXL** | `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | fp16 | ✅ Ready |
| **Janus 1.3B** | `deepseek-ai/Janus-1.3B` | ~3 GB | bf16 | ✅ Ready |
| **Janus-Pro 1.5B** | `deepseek-ai/Janus-Pro-1B` | ~3 GB | bf16 | ✅ Ready |
| **Janus-Pro 7B** | `deepseek-ai/Janus-Pro-7B` | ~14 GB | bf16 | ✅ Ready |
| **Show-O** | `showlab/show-o-512x512` | ~6 GB | bf16 | ✅ Ready |
| **Emu3-Gen** | `BAAI/Emu3-Gen` | ~17 GB | bf16 | ✅ Ready |
| **Nexus-Gen V2** | `DiffSynth-Studio/Nexus-GenV2` | ~24 GB | fp8 quant | ✅ Ready (fp8+offload) |
| Transfusion | — | — | — | ❌ No public weights |
| MetaQuery-XL | — | — | — | ❌ No public release |
| TokenFlow-XL | — | — | — | ❌ No public release |
| SEED-X | `AILab-CVC/SEED-X` | ~34 GB | — | ❌ Too large for 24 GB |

> **Note**: Nexus-Gen (Ours) and Nexus-Gen\* (Ours) in Table 2 correspond to
> Nexus-Gen V1 (BLIP-3o fine-tuned) and V2 (long-short caption training),
> respectively. This suite uses V2 by default.

## Quick Start

### 1. Install environments & download weights

```bash
bash setup_all_models.sh
```

This creates 5 conda environments (`geneval`, `janus`, `showo`, `emu3`, `nexusgen`)
and pre-downloads all model weights.

### 2. Run all generations + evaluation

```bash
bash run_all.sh
```

### 3. Run a subset of models

```bash
bash run_all.sh --models=sdxl,janus_pro_7b,nexus_gen
```

### 4. Generation only (skip evaluation)

```bash
bash run_all.sh --generate-only
```

### 5. Evaluation only (images must already exist)

```bash
bash run_all.sh --evaluate-only
```

## File Structure

```
geneval_models/
├── setup_all_models.sh          # Install envs + download weights
├── run_all.sh                   # Master generate + evaluate script
├── generate_sdxl.py             # SDXL generation
├── generate_janus.py            # Janus / Janus-Pro generation
├── generate_showo.py            # Show-O generation
├── generate_emu3.py             # Emu3-Gen generation
├── generate_nexusgen.py         # Nexus-Gen V2 generation
├── outputs/                     # Generated images (per model)
│   ├── sdxl/
│   ├── janus/
│   ├── janus_pro_1b/
│   ├── janus_pro_7b/
│   ├── show_o/
│   ├── emu3_gen/
│   └── nexus_gen/
└── results/                     # GenEval results (JSONL)
    ├── sdxl.jsonl
    ├── janus.jsonl
    └── ...
```

Each model output follows GenEval format:
```
outputs/<model>/<index:05d>/
    metadata.jsonl          # prompt metadata
    samples/
        00000.png           # sample 0
        00001.png           # sample 1
        ...
```

## Running Individual Models

Each script can be run standalone:

```bash
# SDXL
conda activate geneval
python generate_sdxl.py ../geneval/prompts/evaluation_metadata.jsonl \
    --outdir outputs/sdxl --n_samples 4

# Janus-Pro 7B
conda activate janus
python generate_janus.py ../geneval/prompts/evaluation_metadata.jsonl \
    --model deepseek-ai/Janus-Pro-7B \
    --outdir outputs/janus_pro_7b --n_samples 4

# Nexus-Gen V2 (with fp8 to fit 24 GB)
conda activate nexusgen
python generate_nexusgen.py ../geneval/prompts/evaluation_metadata.jsonl \
    --outdir outputs/nexus_gen --fp8_quantization --enable_cpu_offload
```

## Evaluation

After generating images, evaluate with the GenEval detector:

```bash
conda activate geneval
python ../geneval/evaluation/evaluate_images.py \
    outputs/sdxl/ \
    --model-path ../geneval/models/ \
    --outfile results/sdxl.jsonl

python ../geneval/evaluation/summary_scores.py results/sdxl.jsonl
```

## Notes

- **Nexus-Gen V2** requires `--fp8_quantization` and `--enable_cpu_offload` to fit
  in 24 GB. The AR model (Qwen2.5-VL 7B) and FLUX decoder alternate on the GPU.
- **Emu3-Gen** benefits from `flash-attn` for speed; falls back to SDPA if unavailable.
- **Show-O** uses iterative mask-based denoising (not standard diffusion) — generation
  is slower but fits easily in memory.
- All generation scripts default to 4 samples per prompt and seed 42 for reproducibility.
