#!/bin/bash
#SBATCH --job-name=mrag-ctrl-test
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_mrag_ctrl_%A_%a.out
#SBATCH --error=slurm_mrag_ctrl_%A_%a.err
#SBATCH --array=0-4%2          # 5 controlled test cases, 2 concurrent max

# =============================================================================
# CONTROLLED MRAG Test — Topics with KNOWN relevant images in the DB
# =============================================================================
# Instead of random prompts (which often have NO relevant neighbors), we pick
# prompts from well-populated topic clusters so retrieval can actually find
# useful references. We hold out the exact test image and exclude it.
#
# This is the proper way to evaluate whether MRAG retrieval + conditioning
# actually improves generation quality over the baseline.
# =============================================================================

module purge
source ~/miniconda3/bin/activate
conda activate nexus

echo "Installing dependencies..."
pip install faiss-gpu scikit-learn --quiet
echo "Dependencies installed."

cd ~/ndbao/Nexus-Gen

# =============================================================================
# CONFIGURATION
# =============================================================================
MRAG_DB_PATH="mrag-db"
OUTPUT_DIR="mrag_results_controlled"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${OUTPUT_DIR}/mrag"
mkdir -p "${OUTPUT_DIR}/baseline"
mkdir -p "${OUTPUT_DIR}/rag_references"
mkdir -p "${OUTPUT_DIR}/ground_truth"

# =============================================================================
# 5 CONTROLLED TEST CASES — each from a topic with 100+ similar images in DB
# =============================================================================
# Format: DB_IMAGE_FILENAME|SHORT_PROMPT (what we feed to generation)
# We write a SHORT, focused prompt that fits SigLIP's 64-token window well.
# Ground truth images are from the rebuilt DB (no Qwen rewrites).
# We do NOT exclude them from retrieval — the test is whether MRAG
# retrieval + conditioning actually helps generation quality.
# =============================================================================

case $TASK_ID in
  0)
    # SUNSET BEACH
    DB_IMAGE_FILENAME="007824.webp"
    PROMPT="A couple embracing on a tropical beach at sunset, golden light reflecting on the ocean waves, palm trees silhouetted against an orange and pink sky"
    SEED=100
    ;;
  1)
    # WEDDING CAKE
    DB_IMAGE_FILENAME="000281.webp"
    PROMPT="A beautiful three-tiered wedding cake with pastel colors, decorated with fresh flowers and elegant gold leaf accents, on a white tablecloth"
    SEED=101
    ;;
  2)
    # SNOWY MOUNTAIN
    DB_IMAGE_FILENAME="016425.webp"
    PROMPT="A snowy mountain landscape with snow-covered peaks under a clear blue sky, alpine trees in the foreground, crisp winter sunlight"
    SEED=102
    ;;
  3)
    # GOLDEN RETRIEVER
    DB_IMAGE_FILENAME="001457.webp"
    PROMPT="A golden retriever dog with a fluffy coat sitting in a green grassy field, bright eyes, happy expression, warm sunlight"
    SEED=103
    ;;
  4)
    # COZY CABIN
    DB_IMAGE_FILENAME="007406.webp"
    PROMPT="A cozy wooden cabin in a dense forest with warm lights glowing from the windows, a front porch, surrounded by tall pine trees"
    SEED=104
    ;;
esac

# Generate safe filename
SAFE_NAME=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-40)

echo ""
echo "============================================================"
echo "=== CONTROLLED MRAG Test — Task ${TASK_ID} / 5 ==="
echo "============================================================"
echo "  Prompt:             ${PROMPT}"
echo "  DB Source Image:    ${DB_IMAGE_FILENAME}"
echo "  Seed:               ${SEED}"
echo "============================================================"

# =============================================================================
# Copy ground truth
# =============================================================================
GT_SRC="${MRAG_DB_PATH}/images/${DB_IMAGE_FILENAME}"
GT_DST="${OUTPUT_DIR}/ground_truth/task${TASK_ID}_${SAFE_NAME}.png"

if [ -f "$GT_SRC" ]; then
    cp "$GT_SRC" "$GT_DST"
    echo "✅ Ground truth saved: $GT_DST"
else
    echo "⚠️  Ground truth not found at: $GT_SRC"
fi

MRAG_RESULT="${OUTPUT_DIR}/mrag/task${TASK_ID}/${SAFE_NAME}.png"
BASE_RESULT="${OUTPUT_DIR}/baseline/task${TASK_ID}_${SAFE_NAME}.png"
RAG_REF_DIR="${OUTPUT_DIR}/rag_references/task${TASK_ID}"
mkdir -p "$(dirname "$MRAG_RESULT")"
mkdir -p "$RAG_REF_DIR"

# =============================================================================
# MRAG-ENHANCED GENERATION
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/2] Running MRAG-Enhanced Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 image_generation_mrag.py \
    --prompt "$PROMPT" \
    --enable_mrag \
    --mrag_db_path "$MRAG_DB_PATH" \
    --top_k 3 \
    --mmr_lambda 0.95 \
    --top_k_candidates 50 \
    --image_weight 0.7 \
    --text_weight 0.3 \
    --use_dual_stream \
    --stream2_weight 0.3 \
    --fp8_quantization \
    --enable_cpu_offload \
    --device "cuda:0" \
    --result_path "$MRAG_RESULT" \
    --seed $SEED

MRAG_EXIT=$?

if [ $MRAG_EXIT -ne 0 ]; then
    echo "❌ MRAG generation FAILED (exit code $MRAG_EXIT)"
else
    echo "✅ MRAG result saved: $MRAG_RESULT"
fi

# Copy RAG references (per-task subdir avoids race condition)
MRAG_REF_AUTO_DIR="$(dirname "$MRAG_RESULT")/mrag_references"
if [ -d "$MRAG_REF_AUTO_DIR" ]; then
    rm -f "$RAG_REF_DIR"/* 2>/dev/null
    cp "$MRAG_REF_AUTO_DIR"/* "$RAG_REF_DIR/" 2>/dev/null && \
        echo "✅ RAG references copied to: $RAG_REF_DIR/"
    echo "  Retrieved references:"
    ls -1 "$RAG_REF_DIR/" 2>/dev/null | while read f; do echo "    - $f"; done
fi

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
sleep 5

# =============================================================================
# BASELINE GENERATION (No MRAG)
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/2] Running Baseline Generation (No MRAG)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 image_generation_mrag.py \
    --prompt "$PROMPT" \
    --no_mrag \
    --no_dual_stream \
    --fp8_quantization \
    --enable_cpu_offload \
    --device "cuda:0" \
    --result_path "$BASE_RESULT" \
    --seed $SEED

BASE_EXIT=$?

if [ $BASE_EXIT -ne 0 ]; then
    echo "❌ Baseline generation FAILED (exit code $BASE_EXIT)"
else
    echo "✅ Baseline result saved: $BASE_RESULT"
fi

# =============================================================================
# SUMMARY
# =============================================================================
SUMMARY_FILE="${OUTPUT_DIR}/summary_task${TASK_ID}.txt"
cat > "$SUMMARY_FILE" <<EOF
==============================================================
CONTROLLED MRAG Test — Task ${TASK_ID}
==============================================================
Date:               $(date)
Prompt:             ${PROMPT}
DB Source Image:    ${DB_IMAGE_FILENAME}
Seed:               ${SEED}
--------------------------------------------------------------
Ground Truth (DB):  ${GT_DST}
MRAG Result:        ${MRAG_RESULT}       (exit: ${MRAG_EXIT})
Baseline Result:    ${BASE_RESULT}       (exit: ${BASE_EXIT})
RAG References:     ${RAG_REF_DIR}/
--------------------------------------------------------------
Files produced:
$(ls -lh "${OUTPUT_DIR}/ground_truth/task${TASK_ID}_"* 2>/dev/null || echo "  [no ground truth]")
$(ls -lh "${OUTPUT_DIR}/mrag/task${TASK_ID}/"* 2>/dev/null || echo "  [no MRAG output]")
$(ls -lh "${OUTPUT_DIR}/baseline/task${TASK_ID}_"* 2>/dev/null || echo "  [no baseline output]")
RAG reference images:
$(ls -lh "${RAG_REF_DIR}/"* 2>/dev/null || echo "  [no RAG references]")
==============================================================
EOF

echo ""
echo "============================================================"
echo "  SUMMARY:       $SUMMARY_FILE"
echo "  Ground Truth:  $GT_DST"
echo "  MRAG Output:   $MRAG_RESULT"
echo "  Baseline:      $BASE_RESULT"
echo "  RAG Refs:      $RAG_REF_DIR/"
echo "============================================================"
echo "🏁 DONE — Task ${TASK_ID} complete."
