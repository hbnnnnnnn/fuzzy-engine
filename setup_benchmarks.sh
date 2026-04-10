#!/bin/bash
#SBATCH --job-name=setup_benchmarks
#SBATCH --output=setup_benchmarks_%j.out
#SBATCH --error=setup_benchmarks_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

# =============================================================================
# Setup script for T2I evaluation benchmarks
#   TIFA v1.0   : https://github.com/Yushi-Hu/tifa
#   DrawBench   : prompt dataset — downloaded from HuggingFace
#   T2I-CompBench (PartiPrompts): https://github.com/Karine-Huang/T2I-CompBench
#
# GenEval is already set up via setup_imgedit_geneval.sh.
# =============================================================================

set -e  # exit on error

# ---------------------------------------------------------------------------
# Configuration — adjust paths as needed
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/media02/nthuy/ndbao}"
CONDA_ENV_TIFA="tifa"
CONDA_ENV_T2ICOMP="t2icomp"

module purge

mkdir -p "$INSTALL_DIR"

echo "============================================================"
echo "Install directory    : $INSTALL_DIR"
echo "TIFA env             : $CONDA_ENV_TIFA"
echo "T2I-CompBench env    : $CONDA_ENV_T2ICOMP"
echo "============================================================"

# ---------------------------------------------------------------------------
# Initialise conda inside a non-interactive shell
# ---------------------------------------------------------------------------
source /media02/nthuy/miniconda3/bin/activate

# ============================================================
# Part 1 — TIFA v1.0
# ============================================================
echo ""
echo ">>> [1/3] Setting up TIFA v1.0"
echo "------------------------------------------------------------"

# 1-1. Clone the repository
if [ ! -d "${INSTALL_DIR}/tifa" ]; then
    echo "Cloning TIFA..."
    git clone https://github.com/Yushi-Hu/tifa.git "${INSTALL_DIR}/tifa"
else
    echo "TIFA directory already exists — skipping clone."
fi

# 1-2. Create conda environment
if conda env list | grep -qE "^${CONDA_ENV_TIFA}\s"; then
    echo "Conda env '${CONDA_ENV_TIFA}' already exists — skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_TIFA}' (Python 3.9)..."
    conda create -y -n "${CONDA_ENV_TIFA}" python=3.9
fi

conda activate "${CONDA_ENV_TIFA}"

# 1-3. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 1-4. Install TIFA package and its dependencies
echo "Installing TIFA..."
pip install --quiet -e "${INSTALL_DIR}/tifa"

# 1-5. Install VQA model backends used by TIFA
#      mPLUG-Owl2 (default recommended backend) + BLIP fallback
echo "Installing VQA backends..."
pip install --quiet \
    transformers \
    accelerate \
    sentencepiece \
    Pillow \
    requests \
    tqdm \
    openai \
    spacy

# Download spaCy English model (used for question parsing)
python3 -m spacy download en_core_web_sm

# 1-6. Download the pre-generated TIFA v1.0 benchmark question-answer file
#      (4,081 prompts, 25,829 QA pairs — no GPT API required for evaluation)
TIFA_DATA_DIR="${INSTALL_DIR}/tifa/tifa_v1.0"
mkdir -p "${TIFA_DATA_DIR}"
if [ ! -f "${TIFA_DATA_DIR}/tifa_v1.0_question_answers.json" ]; then
    echo "Downloading TIFA v1.0 QA pairs..."
    wget -q -O "${TIFA_DATA_DIR}/tifa_v1.0_question_answers.json" \
        "https://huggingface.co/datasets/tifa-benchmark/tifav1.0/resolve/main/tifa_v1.0_question_answers.json"
else
    echo "TIFA v1.0 QA file already exists — skipping download."
fi

echo "TIFA environment setup complete."

# ============================================================
# Part 2 — DrawBench (prompt dataset only)
# ============================================================
echo ""
echo ">>> [2/3] Setting up DrawBench prompts"
echo "------------------------------------------------------------"
# DrawBench is a prompt dataset, not executable code — no conda env needed.
# Evaluation is done by running your model on the prompts then scoring with
# CLIP or TIFA using the TIFA env above.

DRAWBENCH_DIR="${INSTALL_DIR}/drawbench"
mkdir -p "${DRAWBENCH_DIR}"

# Download the official DrawBench prompts from HuggingFace
if [ ! -f "${DRAWBENCH_DIR}/drawbench_prompts.json" ]; then
    echo "Downloading DrawBench prompts..."
    python3 - <<'PYEOF'
import json, urllib.request, os

url = "https://huggingface.co/datasets/shunk031/DrawBench/resolve/main/DrawBench_prompts.json"
out = os.path.join(os.environ.get("DRAWBENCH_DIR", "/media02/nthuy/ndbao/drawbench"),
                   "drawbench_prompts.json")
urllib.request.urlretrieve(url, out)
print(f"Saved to {out}")
PYEOF
else
    echo "DrawBench prompts already exist — skipping download."
fi

echo "DrawBench prompt data ready."

# ============================================================
# Part 3 — T2I-CompBench (PartiPrompts / compositional eval)
# ============================================================
echo ""
echo ">>> [3/3] Setting up T2I-CompBench"
echo "------------------------------------------------------------"

# 3-1. Clone the repository
if [ ! -d "${INSTALL_DIR}/T2I-CompBench" ]; then
    echo "Cloning T2I-CompBench..."
    git clone https://github.com/Karine-Huang/T2I-CompBench.git "${INSTALL_DIR}/T2I-CompBench"
else
    echo "T2I-CompBench directory already exists — skipping clone."
fi

# 3-2. Create conda environment
if conda env list | grep -qE "^${CONDA_ENV_T2ICOMP}\s"; then
    echo "Conda env '${CONDA_ENV_T2ICOMP}' already exists — skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_T2ICOMP}' (Python 3.9)..."
    conda create -y -n "${CONDA_ENV_T2ICOMP}" python=3.9
fi

conda activate "${CONDA_ENV_T2ICOMP}"

# 3-3. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available:', torch.cuda.is_available())"

# 3-4. Install T2I-CompBench dependencies
echo "Installing T2I-CompBench dependencies..."
pip install --quiet \
    transformers \
    accelerate \
    diffusers \
    huggingface_hub \
    Pillow \
    opencv-python \
    einops \
    timm \
    tqdm \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    matplotlib \
    sentencepiece \
    clip-benchmark \
    open_clip_torch \
    pycocotools

# CLIP (OpenAI) for CLIPScore
pip install --quiet git+https://github.com/openai/CLIP.git

# 3-5. Download UniDet expert weights (needed for PA and CA evaluation)
#      UniDet = unified multi-dataset object detector used for spatial/counting alignment
UNIDET_WEIGHTS="${INSTALL_DIR}/T2I-CompBench/UniDet_eval/experts/expert_weights"
mkdir -p "${UNIDET_WEIGHTS}"

# Unified learned detector (for OCIM object detection)
UNIDET_MODEL="${UNIDET_WEIGHTS}/Unified_learned_OCIM_RS200_6x+2x.pth"
if [ ! -f "${UNIDET_MODEL}" ]; then
    echo "Downloading UniDet model weights..."
    wget -q -O "${UNIDET_MODEL}" \
        "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth"
else
    echo "UniDet model already exists — skipping."
fi

# DPT depth estimation weights (used by the depth-based spatial metric)
DPT_MODEL="${UNIDET_WEIGHTS}/dpt_hybrid-midas-501f0c75.pt"
if [ ! -f "${DPT_MODEL}" ]; then
    echo "Downloading DPT depth model..."
    wget -q -O "${DPT_MODEL}" \
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
else
    echo "DPT model already exists — skipping."
fi

# 3-6. Install detectron2 (required by UniDet)
echo "Installing detectron2..."
pip install --quiet \
    'git+https://github.com/facebookresearch/detectron2.git'

# 3-7. Download PartiPrompts dataset from HuggingFace
PARTI_DIR="${INSTALL_DIR}/T2I-CompBench/parti_prompts"
mkdir -p "${PARTI_DIR}"
if [ ! -f "${PARTI_DIR}/PartiPrompts.tsv" ]; then
    echo "Downloading PartiPrompts dataset..."
    python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

dest_dir = os.path.join(os.environ.get("INSTALL_DIR", "/media02/nthuy/ndbao"),
                        "T2I-CompBench", "parti_prompts")
path = hf_hub_download(
    repo_id="nateraw/parti-prompts",
    filename="PartiPrompts.tsv",
    repo_type="dataset",
    local_dir=dest_dir,
)
print(f"PartiPrompts saved to {path}")
PYEOF
else
    echo "PartiPrompts already exists — skipping download."
fi

echo "T2I-CompBench environment setup complete."

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Setup finished successfully!"
echo ""
echo "--- TIFA v1.0 ---"
echo "  conda activate ${CONDA_ENV_TIFA}"
echo "  cd ${INSTALL_DIR}/tifa"
echo "  # Score images against the pre-generated QA pairs:"
echo "  python3 tifa/tifa_score.py \\"
echo "      --model mplug-large \\"
echo "      --question_answer_file tifa_v1.0/tifa_v1.0_question_answers.json \\"
echo "      --image_file <your_images.json>"
echo ""
echo "--- DrawBench ---"
echo "  Prompts: ${INSTALL_DIR}/drawbench/drawbench_prompts.json"
echo "  Generate images from those prompts, then score with TIFA or CLIP."
echo ""
echo "--- T2I-CompBench (PA / CA / Attribute Binding) ---"
echo "  conda activate ${CONDA_ENV_T2ICOMP}"
echo "  cd ${INSTALL_DIR}/T2I-CompBench"
echo "  # Positional Alignment:"
echo "  python3 UniDet_eval/pos_alignment.py --image_dir <images/> --output_dir <out/>"
echo "  # Counting Alignment:"
echo "  python3 UniDet_eval/counting_alignment.py --image_dir <images/> --output_dir <out/>"
echo "  # Attribute Binding (BLIP-VQA):"
echo "  python3 BLIP_vqa/blip_vqa_eval.py --image_dir <images/> --output_dir <out/>"
echo "============================================================"
