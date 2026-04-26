#!/bin/bash
#SBATCH --job-name=eval-setup-models
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_eval_setup_%j.out
#SBATCH --error=slurm_eval_setup_%j.err

# =============================================================================
# Setup script for T2I evaluation models
#
# Models from Nexus-Gen paper (Table 2) + SDXL.
# Automatically detects GPU VRAM, RAM, and disk and SKIPS models that
# exceed available resources or lack public weights.
#
# All models from the table:
#   ✅ SDXL             (~7 GB VRAM, fp16)         — diffusers pipeline
#   ✅ Janus 1.3B       (~3 GB VRAM, bf16)         — deepseek-ai/Janus-1.3B
#   ✅ Janus-Pro 1.5B   (~3 GB VRAM, bf16)         — deepseek-ai/Janus-Pro-1B
#   ✅ Janus-Pro 7B     (~14 GB VRAM, bf16)        — deepseek-ai/Janus-Pro-7B
#   ✅ Show-O           (~6 GB VRAM, bf16)         — showlab/show-o-512x512
#   ✅ Emu3-Gen         (~17 GB VRAM, bf16)        — BAAI/Emu3-Gen
#   ✅ Nexus-Gen V2     (~24 GB VRAM, fp8 quant)   — DiffSynth-Studio/Nexus-GenV2
#      (Nexus-Gen* in Table 2 = V2 with long-short caption training)
#
# Skipped (no public weights or too large):
#   ❌ Transfusion   (Meta — weights not released)
#   ❌ MetaQuery-XL  (no public release)
#   ❌ TokenFlow-XL  (ByteDance — no public release)
#   ❌ SEED-X        (~34 GB VRAM, exceeds single-GPU limit)
#
# Benchmarks (must be downloaded first via setup_benchmarks.sh):
#   GenEval, TIFA v1.0, DrawBench, T2I-CompBench
# =============================================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "============================================================"
    echo " SLURM setup job"
    echo " Job ID : ${SLURM_JOB_ID}"
    echo " Host   : $(hostname)"
    echo " GPUs   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ' || echo none)"
    echo " Date   : $(date)"
    echo "============================================================"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INSTALL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-${DEFAULT_INSTALL_DIR}}"
MODELS_CACHE="${INSTALL_DIR}/model_cache"

# Respect user-provided CONDA_BASE; otherwise detect from current shell.
if [[ -z "${CONDA_BASE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
fi
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"

# Redirect caches to user-configurable locations.
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${HOME}/.conda/pkgs}"
export HF_HOME="${MODELS_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${MODELS_CACHE}/torch"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME"

if [[ ! -f "${CONDA_BASE}/bin/activate" ]]; then
    echo "ERROR: Could not find conda activate script at ${CONDA_BASE}/bin/activate"
    echo "Set CONDA_BASE to your conda installation path, then re-run."
    exit 1
fi

source "${CONDA_BASE}/bin/activate"

# ============================================================
# Resource Detection — detect GPU, RAM, disk and decide which
# models can run on this machine.
# ============================================================
detect_resources() {
    echo "============================================================"
    echo " Resource Detection"
    echo "============================================================"

    # --- GPU VRAM (MiB) ---
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        GPU_VRAM_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
        if [[ -n "$GPU_VRAM_RAW" ]]; then
            GPU_VRAM_MIB="$GPU_VRAM_RAW"
        else
            GPU_VRAM_MIB=0
        fi
    else
        GPU_NAME="none"
        GPU_VRAM_MIB=0
    fi
    [[ -z "${GPU_NAME}" ]] && GPU_NAME="none"
    GPU_VRAM_GB=$(awk "BEGIN {printf \"%.1f\", ${GPU_VRAM_MIB:-0}/1024}")

    # --- System RAM (MiB) ---
    RAM_TOTAL_MIB=$(free -m | awk '/^Mem:/ {print $2}')
    RAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", ${RAM_TOTAL_MIB}/1024}")

    # --- Disk free (GB) on install dir ---
    DISK_FREE_GB=$(df --output=avail -BG "$INSTALL_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')

    echo " GPU            : ${GPU_NAME}"
    echo " GPU VRAM       : ${GPU_VRAM_GB} GB (${GPU_VRAM_MIB} MiB)"
    echo " System RAM     : ${RAM_TOTAL_GB} GB"
    echo " Disk free      : ${DISK_FREE_GB} GB"
    echo " Install dir    : ${INSTALL_DIR}"
    echo " Model cache    : ${MODELS_CACHE}"
    echo "============================================================"
}

# ============================================================
# Model compatibility matrix
# Each model: (name, vram_mib_required, public, reason_if_skipped)
# ============================================================
declare -A MODEL_VRAM_MIB  # VRAM required in MiB (with safety margin)
declare -A MODEL_PUBLIC    # 1 = public weights available, 0 = not
declare -A MODEL_SKIP_MSG  # reason for skipping if not public

MODEL_VRAM_MIB[sdxl]=8000;          MODEL_PUBLIC[sdxl]=1
MODEL_VRAM_MIB[janus_1.3b]=4000;    MODEL_PUBLIC[janus_1.3b]=1
MODEL_VRAM_MIB[janus_pro_1b]=4000;  MODEL_PUBLIC[janus_pro_1b]=1
MODEL_VRAM_MIB[janus_pro_7b]=15000; MODEL_PUBLIC[janus_pro_7b]=1
MODEL_VRAM_MIB[showo]=7000;         MODEL_PUBLIC[showo]=1
MODEL_VRAM_MIB[emu3]=18000;         MODEL_PUBLIC[emu3]=1
MODEL_VRAM_MIB[nexusgen]=22000;     MODEL_PUBLIC[nexusgen]=1  # runs with fp8 + cpu offload

MODEL_VRAM_MIB[transfusion]=0;      MODEL_PUBLIC[transfusion]=0; MODEL_SKIP_MSG[transfusion]="Meta — weights not publicly released"
MODEL_VRAM_MIB[metaquery_xl]=0;     MODEL_PUBLIC[metaquery_xl]=0; MODEL_SKIP_MSG[metaquery_xl]="No public release found"
MODEL_VRAM_MIB[tokenflow_xl]=0;     MODEL_PUBLIC[tokenflow_xl]=0; MODEL_SKIP_MSG[tokenflow_xl]="ByteDance — no public release found"
MODEL_VRAM_MIB[seed_x]=35000;       MODEL_PUBLIC[seed_x]=1; MODEL_SKIP_MSG[seed_x]="~34 GB VRAM required, complex multi-model pipeline"

# Check if a model can run on this hardware
can_run_model() {
    local model="$1"
    local required_vram="${MODEL_VRAM_MIB[$model]}"
    local is_public="${MODEL_PUBLIC[$model]}"

    if [[ "$is_public" -eq 0 ]]; then
        echo "SKIP_NO_WEIGHTS"
        return
    fi

    # Nexus-Gen uses CPU offload + fp8, so it can run with slightly less VRAM
    # than the full model size. Allow it if GPU >= 22 GB.
    if [[ "$model" == "nexusgen" && "${GPU_VRAM_MIB:-0}" -ge 22000 ]]; then
        echo "OK"
        return
    fi

    if [[ "${GPU_VRAM_MIB:-0}" -lt "$required_vram" ]]; then
        echo "SKIP_VRAM"
        return
    fi

    echo "OK"
}

# Print compatibility report
print_compatibility_report() {
    echo ""
    echo "============================================================"
    echo " Model Compatibility Report"
    echo "============================================================"
    printf "  %-20s %-12s %-10s %s\n" "Model" "VRAM Req" "Status" "Note"
    printf "  %-20s %-12s %-10s %s\n" "-----" "--------" "------" "----"

    local all_models=("sdxl" "janus_1.3b" "janus_pro_1b" "janus_pro_7b" "showo" "emu3" "nexusgen" "transfusion" "metaquery_xl" "tokenflow_xl" "seed_x")

    MODELS_TO_SETUP=()

    for model in "${all_models[@]}"; do
        local vram="${MODEL_VRAM_MIB[$model]}"
        local status
        status=$(can_run_model "$model")
        local vram_str
        if [[ "$vram" -gt 0 ]]; then
            vram_str="~$(awk "BEGIN {printf \"%.0f\", ${vram}/1024}") GB"
        else
            vram_str="N/A"
        fi

        case "$status" in
            OK)
                printf "  %-20s %-12s %-10s %s\n" "$model" "$vram_str" "✅ READY" ""
                MODELS_TO_SETUP+=("$model")
                ;;
            SKIP_NO_WEIGHTS)
                printf "  %-20s %-12s %-10s %s\n" "$model" "$vram_str" "❌ SKIP" "${MODEL_SKIP_MSG[$model]}"
                ;;
            SKIP_VRAM)
                printf "  %-20s %-12s %-10s %s\n" "$model" "$vram_str" "❌ SKIP" "Exceeds ${GPU_VRAM_GB} GB VRAM"
                ;;
        esac
    done
    echo "============================================================"
    echo " Models to set up: ${MODELS_TO_SETUP[*]}"
    echo "============================================================"
    echo ""
}

# ============================================================
# Helper: create or activate environment
# ============================================================
ensure_env() {
    local env_name="$1"
    local python_ver="${2:-3.10}"
    if conda env list | grep -qE "^${env_name}\s"; then
        echo "[env] '${env_name}' already exists."
    else
        echo "[env] Creating '${env_name}' (Python ${python_ver})..."
        conda create -y -n "${env_name}" python="${python_ver}"
    fi
    conda activate "${env_name}"
}

install_torch_cu121() {
    python3 -m pip install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
}

# ============================================================
# Run resource detection and compatibility check
# ============================================================
detect_resources
print_compatibility_report

# Check minimum requirements
if [[ "${GPU_VRAM_MIB:-0}" -eq 0 ]]; then
    echo "ERROR: No NVIDIA GPU detected. Cannot set up T2I models."
    exit 1
fi

if [[ "${DISK_FREE_GB:-0}" -lt 50 ]]; then
    echo "WARNING: Only ${DISK_FREE_GB} GB disk free. At least 80 GB recommended for all model weights."
fi

# Helper to check if a model is in the setup list
should_setup() {
    local model="$1"
    for m in "${MODELS_TO_SETUP[@]}"; do
        [[ "$m" == "$model" ]] && return 0
    done
    return 1
}

# ============================================================
# 1. SDXL  (reuses the geneval env — diffusers-based)
# ============================================================
if should_setup "sdxl"; then
    echo ""
    echo ">>> [1/7] SDXL — environment: geneval"
    echo "------------------------------------------------------------"
    ensure_env "geneval" "3.9"
    # SDXL just needs diffusers — likely already installed
    python3 -m pip install --quiet diffusers transformers accelerate safetensors xformers Pillow tqdm

    echo "Pre-downloading SDXL weights..."
    python3 -c "
from diffusers import StableDiffusionXLPipeline
StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype='auto', use_safetensors=True, variant='fp16'
)
print('SDXL weights cached.')
" || echo "  (SDXL download skipped — run manually if needed)"
else
    echo ">>> [1/7] SDXL — SKIPPED (resource check failed)"
fi

# ============================================================
# 2–4. Janus family (shared env)
# ============================================================
if should_setup "janus_1.3b" || should_setup "janus_pro_1b" || should_setup "janus_pro_7b"; then
    echo ""
    echo ">>> [2/7] Janus family — environment: janus"
    echo "------------------------------------------------------------"
    ensure_env "janus" "3.10"
    install_torch_cu121

    # Install Janus from the DeepSeek repo
    if [ ! -d "${INSTALL_DIR}/Janus" ]; then
        echo "Cloning Janus..."
        git clone https://github.com/deepseek-ai/Janus.git "${INSTALL_DIR}/Janus"
    else
        echo "Janus directory already exists — skipping clone."
    fi

    python3 -m pip install --quiet -e "${INSTALL_DIR}/Janus"
    python3 -m pip install --quiet transformers accelerate safetensors Pillow tqdm numpy

    echo "Pre-downloading Janus model weights..."
    JANUS_MODELS=""
    should_setup janus_1.3b   && JANUS_MODELS="${JANUS_MODELS},deepseek-ai/Janus-1.3B"
    should_setup janus_pro_1b && JANUS_MODELS="${JANUS_MODELS},deepseek-ai/Janus-Pro-1B"
    should_setup janus_pro_7b && JANUS_MODELS="${JANUS_MODELS},deepseek-ai/Janus-Pro-7B"
    JANUS_MODELS="${JANUS_MODELS#,}"  # strip leading comma
    python3 -c "
from transformers import AutoModelForCausalLM
import sys
models = '${JANUS_MODELS}'.split(',')
for m in models:
    m = m.strip()
    if not m: continue
    print(f'Downloading {m}...')
    try:
        AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)
        print(f'  {m} cached.')
    except Exception as e:
        print(f'  {m} download issue: {e}')
" || echo "  (Some Janus downloads skipped — run manually if needed)"
else
    echo ">>> [2/7] Janus family — SKIPPED (resource check failed)"
fi

# ============================================================
# 5. Show-O
# ============================================================
if should_setup "showo"; then
    echo ""
    echo ">>> [3/7] Show-O — environment: showo"
    echo "------------------------------------------------------------"
    ensure_env "showo" "3.10"
    install_torch_cu121

    if [ ! -d "${INSTALL_DIR}/Show-o" ]; then
        echo "Cloning Show-o..."
        git clone https://github.com/showlab/Show-o.git "${INSTALL_DIR}/Show-o"
    else
        echo "Show-o directory already exists — skipping clone."
    fi

    # Filter out pycurl (not used by Show-o, requires libcurl-dev to build)
    grep -iv '^pycurl' "${INSTALL_DIR}/Show-o/requirements.txt" | \
        python3 -m pip install --quiet -r /dev/stdin
    python3 -m pip install --quiet transformers accelerate safetensors Pillow tqdm numpy omegaconf wandb

    echo "Pre-downloading Show-O model weights..."
    python3 -c "
from huggingface_hub import snapshot_download
for repo in ['showlab/show-o-512x512', 'showlab/magvitv2']:
    print(f'Downloading {repo}...')
    snapshot_download(repo)
    print(f'  {repo} cached.')
" || echo "  (Show-O downloads skipped — run manually)"
else
    echo ">>> [3/7] Show-O — SKIPPED (resource check failed)"
fi

# ============================================================
# 6. Emu3-Gen
# ============================================================
if should_setup "emu3"; then
    echo ""
    echo ">>> [4/7] Emu3-Gen — environment: emu3"
    echo "------------------------------------------------------------"
    ensure_env "emu3" "3.10"
    install_torch_cu121

    python3 -m pip install --quiet transformers accelerate safetensors Pillow tqdm numpy
    python3 -m pip install --quiet flash-attn --no-build-isolation 2>/dev/null \
        || echo "  (flash-attn build failed — Emu3 will fall back to sdpa)"

    echo "Pre-downloading Emu3-Gen model weights..."
    python3 -c "
from huggingface_hub import snapshot_download
for repo in ['BAAI/Emu3-Gen', 'BAAI/Emu3-VisionTokenizer']:
    print(f'Downloading {repo}...')
    snapshot_download(repo, trust_remote_code=True)
    print(f'  {repo} cached.')
" || echo "  (Emu3 downloads skipped — run manually)"
else
    echo ">>> [4/7] Emu3-Gen — SKIPPED (resource check failed)"
fi

# ============================================================
# 7. Nexus-Gen V2 (= "Nexus-Gen*" in Table 2)
#    Repo: https://github.com/modelscope/Nexus-Gen
#    Uses CPU offload + fp8 quantization for 24 GB GPUs
# ============================================================
if should_setup "nexusgen"; then
    echo ""
    echo ">>> [5/7] Nexus-Gen V2 — environment: nexusgen"
    echo "------------------------------------------------------------"
    ensure_env "nexusgen" "3.10"
    install_torch_cu121

    # 7a. Install DiffSynth-Studio (required backend)
    if [ ! -d "${INSTALL_DIR}/DiffSynth-Studio" ]; then
        echo "Cloning DiffSynth-Studio..."
        git clone https://github.com/modelscope/DiffSynth-Studio.git "${INSTALL_DIR}/DiffSynth-Studio"
    else
        echo "DiffSynth-Studio already exists — skipping clone."
    fi
    python3 -m pip install --quiet -e "${INSTALL_DIR}/DiffSynth-Studio"

    # 7b. Clone Nexus-Gen repo
    if [ ! -d "${INSTALL_DIR}/Nexus-Gen" ]; then
        echo "Cloning Nexus-Gen..."
        git clone https://github.com/modelscope/Nexus-Gen.git "${INSTALL_DIR}/Nexus-Gen"
    else
        echo "Nexus-Gen already exists — skipping clone."
    fi

    # 7c. Install Nexus-Gen requirements
    python3 -m pip install --quiet -r "${INSTALL_DIR}/Nexus-Gen/requirements.txt"
    python3 -m pip install --quiet transformers==4.49.0 accelerate safetensors Pillow tqdm numpy
    python3 -m pip install --quiet qwen_vl_utils
python3 -m pip install --quiet flash-attn --no-build-isolation 2>/dev/null \
    || echo "  (flash-attn build failed — will fall back to sdpa)"

# 7d. Download model weights using modelscope + HF
echo "Downloading Nexus-Gen V2 model weights..."
cd "${INSTALL_DIR}/Nexus-Gen"
python3 -c "
from modelscope import snapshot_download

print('Downloading Nexus-GenV2 checkpoint...')
snapshot_download('DiffSynth-Studio/Nexus-GenV2', local_dir='models/Nexus-GenV2')

print('Downloading FLUX.1-dev components...')
snapshot_download('black-forest-labs/FLUX.1-dev',
    allow_file_pattern=[
        'text_encoder/model.safetensors',
        'text_encoder_2/*',
        'ae.safetensors',
    ],
    local_dir='models/FLUX/FLUX.1-dev')
print('All Nexus-Gen V2 weights downloaded.')
" || echo "  (Nexus-Gen download issue — run download_models.py manually)"
cd "${INSTALL_DIR}"
else
    echo ">>> [5/7] Nexus-Gen V2 — SKIPPED (resource check failed)"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo " Model Setup Complete!"
echo ""
echo " Detected hardware:"
echo "   GPU  : ${GPU_NAME} (${GPU_VRAM_GB} GB)"
echo "   RAM  : ${RAM_TOTAL_GB} GB"
echo "   Disk : ${DISK_FREE_GB} GB free"
echo ""
echo " Models set up:"
for m in "${MODELS_TO_SETUP[@]}"; do
    echo "   ✅ $m"
done
echo ""
echo " Models SKIPPED (from Nexus-Gen paper Table 2):"
SKIPPED_ANY=false
for model in "transfusion" "metaquery_xl" "tokenflow_xl" "seed_x"; do
    status=$(can_run_model "$model")
    if [[ "$status" != "OK" ]]; then
        SKIPPED_ANY=true
        reason="${MODEL_SKIP_MSG[$model]:-Exceeds VRAM}"
        echo "   ❌ $model — $reason"
    fi
done
# Also check if any setup-able models were skipped due to VRAM
for model in "sdxl" "janus_1.3b" "janus_pro_1b" "janus_pro_7b" "showo" "emu3" "nexusgen"; do
    if ! should_setup "$model"; then
        SKIPPED_ANY=true
        echo "   ❌ $model — Exceeds ${GPU_VRAM_GB} GB VRAM"
    fi
done
if ! $SKIPPED_ANY; then
    echo "   (none)"
fi
echo ""
echo " Benchmarks expected (from setup_benchmarks.sh):"
echo "   GenEval, TIFA v1.0, DrawBench, T2I-CompBench"
echo ""
echo " Generation scripts in: ${INSTALL_DIR}/eval_models/"
echo " Run all with:          bash ${INSTALL_DIR}/eval_models/run_all.sh"
echo "============================================================"
