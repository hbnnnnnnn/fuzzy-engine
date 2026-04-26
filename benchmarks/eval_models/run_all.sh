#!/bin/bash
#SBATCH --job-name=eval-run-all
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_eval_run_all_%j.out
#SBATCH --error=slurm_eval_run_all_%j.err

# =============================================================================
# Master script: generate & evaluate across ALL benchmarks × ALL models
#
# Automatically detects GPU VRAM and skips models that exceed available
# resources or lack public weights.
#
# Benchmarks:
#   1. GenEval          (553 prompts, n_samples=4)
#   2. TIFA v1.0        (4081 prompts, n_samples=1)
#   3. DrawBench        (200 prompts, n_samples=1)
#   4. T2I-CompBench    (7 categories × ~300 prompts, n_samples=10)
#
# Models (from Nexus-Gen paper Table 2 + SDXL):
#   1. SDXL             (env: geneval,  ~7 GB)
#   2. Janus 1.3B       (env: janus,    ~3 GB)
#   3. Janus-Pro 1B     (env: janus,    ~3 GB)
#   4. Janus-Pro 7B     (env: janus,    ~14 GB)
#   5. Show-O           (env: showo,    ~6 GB)
#   6. Emu3-Gen         (env: emu3,     ~17 GB)
#   7. Nexus-Gen V2     (env: nexusgen, ~24 GB fp8+offload)
#
# Usage:
#   bash run_all.sh               # run everything
#   bash run_all.sh --gen-only    # generate only, skip evaluation
#   bash run_all.sh --eval-only   # evaluate only, skip generation
#   bash run_all.sh --model sdxl  # run only one model
#   bash run_all.sh --bench geneval  # run only one benchmark
# =============================================================================
# NOTE: no set -e — we handle errors per-task so one failure doesn't abort
# the whole pipeline. Already-generated outputs are auto-skipped on re-run.

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SLURM job: ${SLURM_JOB_ID}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Host: $(hostname)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ' || echo none)"
fi

# ---------------------------------------------------------------------------
# Auto-logging: re-execute with tee so output always goes to a log file.
# Skip if already inside a logged invocation (_LOGGED=1).
# ---------------------------------------------------------------------------
if [[ -z "$_LOGGED" ]]; then
    SCRIPT_SELF="$(readlink -f "$0")"
    LOG_DIR="$(dirname "$SCRIPT_SELF")"
    LOG_FILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Logging to: $LOG_FILE"
    _LOGGED=1 bash "$SCRIPT_SELF" "$@" 2>&1 | tee "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

# Respect user-provided CONDA_BASE; otherwise detect from current shell.
if [[ -z "${CONDA_BASE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
fi
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"

# Benchmark data paths
GENEVAL_PROMPTS="${BASE_DIR}/geneval/prompts/evaluation_metadata.jsonl"
TIFA_PROMPTS="${BASE_DIR}/tifa/tifa_v1.0/tifa_v1.0_text_inputs.json"
TIFA_QA="${BASE_DIR}/tifa/tifa_v1.0/tifa_v1.0_question_answers.json"
DRAWBENCH_PROMPTS="${BASE_DIR}/drawbench/drawbench_prompts.json"
T2I_DATASET="${BASE_DIR}/T2I-CompBench/examples/dataset"
GENEVAL_DETECTOR="${BASE_DIR}/geneval/models"

# Cache dirs
# NOTE: TMPDIR MUST be on local disk (non-NFS) because Python's multiprocessing
# creates temp directories there. NFS locks prevent cleanup, causing "Device or
# resource busy" errors. Use /tmp (local) for temp files, keep caches on NFS.
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${HOME}/.conda/pkgs}"
export HF_HOME="${BASE_DIR}/model_cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.huggingface/token 2>/dev/null || true)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${BASE_DIR}/tifa:${PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$OUTPUT_ROOT"

if [[ ! -f "${CONDA_BASE}/bin/activate" ]]; then
    echo "ERROR: Could not find conda activate script at ${CONDA_BASE}/bin/activate"
    echo "Set CONDA_BASE to your conda installation path, then re-run."
    exit 1
fi

for p in "$GENEVAL_PROMPTS" "$TIFA_PROMPTS" "$TIFA_QA" "$DRAWBENCH_PROMPTS" "$T2I_DATASET" "$GENEVAL_DETECTOR"; do
    if [[ ! -e "$p" ]]; then
        echo "ERROR: Missing required path: $p"
        exit 1
    fi
done

source "${CONDA_BASE}/bin/activate"

# ---------------------------------------------------------------------------
# Helpers (defined early so resource detection can use log())
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# ---------------------------------------------------------------------------
# Resource detection — detect GPU VRAM and skip incompatible models
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_VRAM_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    if [[ -n "$GPU_VRAM_RAW" ]]; then
        GPU_VRAM_MIB="$GPU_VRAM_RAW"
    else
        GPU_VRAM_MIB=0
    fi
else
    GPU_NAME="none"; GPU_VRAM_MIB=0; GPU_VRAM_GB="0.0"
fi
[[ -z "${GPU_NAME}" ]] && GPU_NAME="none"
GPU_VRAM_GB=$(awk "BEGIN {printf \"%.1f\", ${GPU_VRAM_MIB:-0}/1024}")
log "GPU: ${GPU_NAME} (${GPU_VRAM_GB} GB VRAM)"

# VRAM requirements per model (MiB, with safety margin)
declare -A VRAM_REQ
VRAM_REQ[sdxl]=8000
VRAM_REQ[janus_1.3b]=4000
VRAM_REQ[janus_pro_1b]=4000
VRAM_REQ[janus_pro_7b]=15000
VRAM_REQ[showo]=7000
VRAM_REQ[emu3]=18000
VRAM_REQ[nexusgen]=22000  # fp8 + cpu offload

gpu_can_run() {
    local model="$1"
    local req="${VRAM_REQ[$model]:-999999}"
    [[ "${GPU_VRAM_MIB:-0}" -ge "$req" ]]
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DO_GEN=true
DO_EVAL=true
FILTER_MODEL=""
FILTER_BENCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gen-only)   DO_EVAL=false; shift;;
        --eval-only)  DO_GEN=false; shift;;
        --model)      FILTER_MODEL="$2"; shift 2;;
        --bench)      FILTER_BENCH="$2"; shift 2;;
        *)            echo "Unknown arg: $1"; exit 1;;
    esac
done

# ---------------------------------------------------------------------------
# More helpers
# ---------------------------------------------------------------------------
should_run_model() {
    # Check user filter
    [[ -n "$FILTER_MODEL" && "$FILTER_MODEL" != "$1" ]] && return 1
    # Check GPU compatibility
    if ! gpu_can_run "$1"; then
        log "SKIP $1 — requires ~$((${VRAM_REQ[$1]}/1024)) GB VRAM, have ${GPU_VRAM_GB} GB"
        return 1
    fi
    return 0
}
should_run_bench() { [[ -z "$FILTER_BENCH" || "$FILTER_BENCH" == "$1" ]]; }

run_gen() {
    local env="$1" script="$2" bench="$3" prompt_file="$4" outdir="$5" n_samples="$6"
    shift 6
    local extra_args="$*"
    local done_flag="${outdir}/.done"
    local status

    # Skip only if a previous run completed successfully
    if [[ -f "$done_flag" ]]; then
        log "SKIP (complete): $outdir"
        return 0
    fi

    # Log resuming vs fresh start
    local n_existing
    n_existing=$(find "$outdir" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l)
    if [[ "$n_existing" -gt 0 ]]; then
        log "RESUME ($n_existing images already exist): $outdir"
    else
        log "GEN: env=$env script=$script bench=$bench -> $outdir"
    fi

    PYTHONNOUSERSITE=1 conda run -n "$env" python "${SCRIPT_DIR}/${script}" \
        --benchmark "$bench" \
        --prompt_file "$prompt_file" \
        --outdir "$outdir" \
        --n_samples "$n_samples" \
        $extra_args
    status=$?

    # Retry once if CUDA temporarily vanished (common after transient driver hiccups).
    if [[ "$status" -ne 0 ]] && ! PYTHONNOUSERSITE=1 conda run -n "$env" python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        log "GEN retry (CUDA unavailable): env=$env script=$script bench=$bench"
        sleep 10
        PYTHONNOUSERSITE=1 conda run -n "$env" python "${SCRIPT_DIR}/${script}" \
            --benchmark "$bench" \
            --prompt_file "$prompt_file" \
            --outdir "$outdir" \
            --n_samples "$n_samples" \
            $extra_args
        status=$?
    fi

    if [[ "$status" -eq 0 ]]; then
        touch "$done_flag"   # mark as fully complete
        log "GEN OK: $outdir"
    else
        rm -f "$done_flag"   # ensure .done is absent so next run retries
        local count
        count=$(find "$outdir" -type f 2>/dev/null | wc -l)
        log "GEN FAILED (exit $status): env=$env script=$script bench=$bench — will resume on next run ($count files kept)"
    fi
}

# ===========================================================================
# T2I-CompBench category list
# ===========================================================================
COMPBENCH_CATS=("color" "shape" "texture" "spatial" "non_spatial" "complex" "numeracy")

# Map category -> prompt file
declare -A CAT2FILE
CAT2FILE[color]="${T2I_DATASET}/color_val.txt"
CAT2FILE[shape]="${T2I_DATASET}/shape_val.txt"
CAT2FILE[texture]="${T2I_DATASET}/texture_val.txt"
CAT2FILE[spatial]="${T2I_DATASET}/spatial_val.txt"
CAT2FILE[non_spatial]="${T2I_DATASET}/non_spatial_val.txt"
CAT2FILE[complex]="${T2I_DATASET}/complex_val.txt"
CAT2FILE[numeracy]="${T2I_DATASET}/numeracy_val.txt"

# ===========================================================================
# Model definitions: (name, conda_env, script, extra_args)
# ===========================================================================
declare -a MODEL_NAMES=("sdxl" "janus_1.3b" "janus_pro_1b" "janus_pro_7b" "showo" "emu3" "nexusgen")
declare -A MODEL_ENV MODEL_SCRIPT MODEL_EXTRA

MODEL_ENV[sdxl]="geneval";        MODEL_SCRIPT[sdxl]="generate_sdxl.py";       MODEL_EXTRA[sdxl]=""
MODEL_ENV[janus_1.3b]="janus";    MODEL_SCRIPT[janus_1.3b]="generate_janus.py"; MODEL_EXTRA[janus_1.3b]="--model deepseek-ai/Janus-1.3B"
MODEL_ENV[janus_pro_1b]="janus";  MODEL_SCRIPT[janus_pro_1b]="generate_janus.py"; MODEL_EXTRA[janus_pro_1b]="--model deepseek-ai/Janus-Pro-1B"
MODEL_ENV[janus_pro_7b]="janus";  MODEL_SCRIPT[janus_pro_7b]="generate_janus.py"; MODEL_EXTRA[janus_pro_7b]="--model deepseek-ai/Janus-Pro-7B"
MODEL_ENV[showo]="showo";         MODEL_SCRIPT[showo]="generate_showo.py";     MODEL_EXTRA[showo]=""
MODEL_ENV[emu3]="emu3";           MODEL_SCRIPT[emu3]="generate_emu3.py";       MODEL_EXTRA[emu3]=""
MODEL_ENV[nexusgen]="nexusgen";   MODEL_SCRIPT[nexusgen]="generate_nexusgen.py"; MODEL_EXTRA[nexusgen]="--fp8_quantization"

check_required_envs() {
    local required_envs=("geneval" "janus" "showo" "emu3" "nexusgen" "tifa" "t2icomp")
    local missing=()
    local env
    for env in "${required_envs[@]}"; do
        if ! conda env list | awk 'NF && $1 !~ /^#/' | awk '{print $1}' | grep -Fxq "$env"; then
            missing+=("$env")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        log "ERROR: Missing conda env(s): ${missing[*]}"
        log "Run setup_all_models.sh (after updating its path settings) or create these envs manually."
        exit 1
    fi
}

check_required_envs

# ===========================================================================
#  GENERATION
# ===========================================================================
if $DO_GEN; then
    log "========== GENERATION =========="
    log "Compatible models on ${GPU_NAME} (${GPU_VRAM_GB} GB):"
    for model in "${MODEL_NAMES[@]}"; do
        if gpu_can_run "$model"; then
            log "  ✅ $model (~$((${VRAM_REQ[$model]}/1024)) GB)"
        else
            log "  ❌ $model (~$((${VRAM_REQ[$model]}/1024)) GB) — exceeds VRAM"
        fi
    done

    for model in "${MODEL_NAMES[@]}"; do
        should_run_model "$model" || continue
        env="${MODEL_ENV[$model]}"
        script="${MODEL_SCRIPT[$model]}"
        extra="${MODEL_EXTRA[$model]}"

        # --- GenEval ---
        if should_run_bench "geneval"; then
            run_gen "$env" "$script" "geneval" "$GENEVAL_PROMPTS" \
                "${OUTPUT_ROOT}/geneval/${model}" 4 $extra
        fi

        # --- TIFA ---
        if should_run_bench "tifa"; then
            run_gen "$env" "$script" "tifa" "$TIFA_PROMPTS" \
                "${OUTPUT_ROOT}/tifa/${model}" 1 $extra
        fi

        # --- DrawBench ---
        if should_run_bench "drawbench"; then
            run_gen "$env" "$script" "drawbench" "$DRAWBENCH_PROMPTS" \
                "${OUTPUT_ROOT}/drawbench/${model}" 1 $extra
        fi

        # --- T2I-CompBench (each category) ---
        for cat in "${COMPBENCH_CATS[@]}"; do
            if should_run_bench "$cat" || should_run_bench "t2i_compbench"; then
                run_gen "$env" "$script" "$cat" "${CAT2FILE[$cat]}" \
                    "${OUTPUT_ROOT}/t2i_compbench/${cat}/${model}" 10 $extra
            fi
        done
    done

    log "========== GENERATION COMPLETE =========="
fi

# ===========================================================================
#  EVALUATION
# ===========================================================================
if $DO_EVAL; then
    log "========== EVALUATION =========="

    for model in "${MODEL_NAMES[@]}"; do
        should_run_model "$model" || continue

        # --- GenEval ---
        if should_run_bench "geneval"; then
            GDIR="${OUTPUT_ROOT}/geneval/${model}"
            if [[ -d "$GDIR" && -n "$(find "$GDIR" -type f -name '*.png' 2>/dev/null | head -1)" ]]; then
                log "EVAL GenEval: ${model}"
                PYTHONNOUSERSITE=1 conda run -n geneval \
                    bash -c "cd '${BASE_DIR}/geneval' && \
                        python evaluation/evaluate_images.py \
                            '${GDIR}' \
                            --outfile '${GDIR}/results.jsonl' \
                            --model-path '${GENEVAL_DETECTOR}' && \
                        python evaluation/summary_scores.py '${GDIR}/results.jsonl' \
                            | tee '${GDIR}/summary.txt'"
            fi
        fi

        # --- TIFA ---
        if should_run_bench "tifa"; then
            TDIR="${OUTPUT_ROOT}/tifa/${model}"
            if [[ -d "$TDIR" && -f "${TDIR}/id2img.json" ]]; then
                log "EVAL TIFA: ${model}"
                PYTHONNOUSERSITE=1 conda run -n tifa \
                    python "${SCRIPT_DIR}/eval_tifa.py" \
                        --qa_file "${TIFA_QA}" \
                        --id2img "${TDIR}/id2img.json" \
                        --output "${TDIR}/tifa_results.json"
            fi
        fi

        # --- DrawBench (CLIPScore) ---
        if should_run_bench "drawbench"; then
            DDIR="${OUTPUT_ROOT}/drawbench/${model}"
            if [[ -d "$DDIR" && -n "$(find "$DDIR" -type f -name '*.png' 2>/dev/null | head -1)" ]]; then
                log "EVAL DrawBench (CLIPScore): ${model}"
                PYTHONNOUSERSITE=1 conda run -n tifa \
                    python "${SCRIPT_DIR}/eval_drawbench.py" \
                        --image_dir "${DDIR}/images" \
                        --prompts_file "${DRAWBENCH_PROMPTS}" \
                        --output "${DDIR}/drawbench_results.json"
            fi
        fi

        # --- T2I-CompBench categories ---
        for cat in "${COMPBENCH_CATS[@]}"; do
            if should_run_bench "$cat" || should_run_bench "t2i_compbench"; then
                CDIR="${OUTPUT_ROOT}/t2i_compbench/${cat}/${model}"
                if [[ -d "$CDIR" && -n "$(find "$CDIR" -type f -name '*.png' 2>/dev/null | head -1)" ]]; then
                    log "EVAL T2I-CompBench ${cat}: ${model}"

                    case "$cat" in
                        color|shape|texture)
                            PYTHONNOUSERSITE=1 conda run -n t2icomp \
                                bash -c "cd '${BASE_DIR}/T2I-CompBench' && \
                                    python BLIPvqa_eval/BLIP_vqa.py --out_dir '${CDIR}'"
                            ;;
                        spatial)
                            PYTHONNOUSERSITE=1 conda run -n t2icomp \
                                bash -c "cd '${BASE_DIR}/T2I-CompBench/UniDet_eval' && \
                                    python 2D_spatial_eval.py --outpath '${CDIR}'"
                            ;;
                        non_spatial)
                            PYTHONNOUSERSITE=1 conda run -n t2icomp \
                                bash -c "cd '${BASE_DIR}/T2I-CompBench' && \
                                    python CLIPScore_eval/CLIP_similarity.py --outpath '${CDIR}'"
                            ;;
                        complex)
                            PYTHONNOUSERSITE=1 conda run -n t2icomp \
                                bash -c "cd '${BASE_DIR}/T2I-CompBench' && \
                                    python BLIPvqa_eval/BLIP_vqa.py --out_dir '${CDIR}' && \
                                    cd UniDet_eval && \
                                    python 2D_spatial_eval.py --outpath '${CDIR}' --complex True && \
                                    cd .. && \
                                    python CLIPScore_eval/CLIP_similarity.py --outpath '${CDIR}' --complex True && \
                                    python 3_in_1_eval/3_in_1.py --outpath '${CDIR}' \
                                        --data_path '${T2I_DATASET}'"
                            ;;
                        numeracy)
                            PYTHONNOUSERSITE=1 conda run -n t2icomp \
                                bash -c "cd '${BASE_DIR}/T2I-CompBench/UniDet_eval' && \
                                    python numeracy_eval.py --outpath '${CDIR}'"
                            ;;
                    esac
                fi
            fi
        done
    done

    # --- Collect all scores into a summary table ---
    log "Collecting summary scores..."
    PYTHONNOUSERSITE=1 conda run -n geneval \
        python "${SCRIPT_DIR}/collect_scores.py" --output_root "${OUTPUT_ROOT}"

    log "========== EVALUATION COMPLETE =========="
fi

log "All done! Results in: ${OUTPUT_ROOT}/"
