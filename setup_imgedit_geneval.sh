#!/bin/bash
#SBATCH --job-name=setup_eval
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

# =============================================================================
# Setup script for ImgEdit and GenEval
#   ImgEdit: https://github.com/PKU-YuanGroup/ImgEdit
#   GenEval: https://github.com/djghosh13/geneval
# =============================================================================

set -e  # exit on error

# ---------------------------------------------------------------------------
# Configuration — adjust these paths as needed
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/media02/nthuy/ndbao}"
CONDA_ENV_IMGEDIT="imgedit"
CONDA_ENV_GENEVAL="geneval"
OBJECT_DETECTOR_DIR="${INSTALL_DIR}/geneval/models"

module purge

mkdir -p "$INSTALL_DIR"

echo "============================================================"
echo "Install directory : $INSTALL_DIR"
echo "ImgEdit env       : $CONDA_ENV_IMGEDIT"
echo "GenEval env       : $CONDA_ENV_GENEVAL"
echo "============================================================"

# ---------------------------------------------------------------------------
# Initialise conda inside a non-interactive shell (same approach as build_mrag_db.sh)
# ---------------------------------------------------------------------------
source /media02/nthuy/miniconda3/bin/activate

# ============================================================
# Part 1 — ImgEdit
# ============================================================
echo ""
echo ">>> [1/2] Setting up ImgEdit"
echo "------------------------------------------------------------"

# 1-1. Clone the repository
if [ ! -d "${INSTALL_DIR}/ImgEdit" ]; then
    echo "Cloning ImgEdit..."
    git clone https://github.com/PKU-YuanGroup/ImgEdit.git "${INSTALL_DIR}/ImgEdit"
else
    echo "ImgEdit directory already exists — skipping clone."
fi

# 1-2. Create conda environment
if conda env list | grep -qE "^${CONDA_ENV_IMGEDIT}\s"; then
    echo "Conda env '${CONDA_ENV_IMGEDIT}' already exists — skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_IMGEDIT}' (Python 3.10)..."
    conda create -y -n "${CONDA_ENV_IMGEDIT}" python=3.10
fi

conda activate "${CONDA_ENV_IMGEDIT}"

# 1-3. Install PyTorch (CUDA 11.8; adjust --index-url for other CUDA versions)
echo "Installing PyTorch..."
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 1-4. Install Hugging Face stack + vision dependencies
echo "Installing Hugging Face + vision dependencies..."
pip install --quiet \
    transformers \
    diffusers \
    accelerate \
    huggingface_hub \
    Pillow \
    opencv-python \
    einops \
    timm \
    pyyaml \
    tqdm \
    sentencepiece \
    tokenizers

# 1-5. Install Qwen2.5-VL (used by ImgEdit-Judge)
echo "Installing Qwen2.5-VL dependencies..."
pip install --quiet qwen-vl-utils

# 1-6. Install SAM2
echo "Installing SAM2..."
pip install --quiet git+https://github.com/facebookresearch/sam2.git

# 1-7. Install YOLO-World (optional; needed for bounding-box pipeline)
echo "Installing YOLO-World (inference only)..."
pip install --quiet ultralytics

# 1-8. Install any extras referenced by ImgEdit tools/benchmark scripts
pip install --quiet \
    matplotlib \
    scipy \
    scikit-image \
    pandas \
    pycocotools

# 1-9. (Optional) Download ImgEdit-Judge checkpoint from HuggingFace
# Uncomment and set your HF token if required:
# echo "Downloading ImgEdit-Judge checkpoint..."
# huggingface-cli download --repo-type model sysuyy/ImgEdit_Judge \
#     --local-dir "${INSTALL_DIR}/ImgEdit/checkpoints/ImgEdit_Judge" \
#     --token "${HF_TOKEN}"

echo "ImgEdit environment setup complete."

# ============================================================
# Part 2 — GenEval
# ============================================================
echo ""
echo ">>> [2/2] Setting up GenEval"
echo "------------------------------------------------------------"

# 2-1. Clone the repository
if [ ! -d "${INSTALL_DIR}/geneval" ]; then
    echo "Cloning GenEval..."
    git clone https://github.com/djghosh13/geneval.git "${INSTALL_DIR}/geneval"
else
    echo "GenEval directory already exists — skipping clone."
fi

# 2-2. Patch environment.yml — strip legacy NVIDIA pip wheels that are no
#      longer reliably available on public PyPI (nvidia-cublas-cu11 2022.x,
#      nvidia-cuda-runtime-cu11, etc.).  Also remove the pinned PyTorch
#      1.12.1 / torchvision 0.13.1 and cudatoolkit entries so we can install
#      a modern, cluster-compatible stack in the next step.
ORIG_YML="${INSTALL_DIR}/geneval/environment.yml"
PATCHED_YML="${INSTALL_DIR}/geneval/environment_patched.yml"

echo "Patching environment.yml to remove legacy NVIDIA + old PyTorch pins..."
grep -vE \
    '^\s*-\s*(nvidia-|torch==|torchvision==|torchaudio==|pytorch==|pytorch-mutex|cudatoolkit|mmcv-full)' \
    "${ORIG_YML}" > "${PATCHED_YML}"
echo "  -> written to ${PATCHED_YML}"

# 2-3. Create conda environment from the patched yml
if conda env list | grep -qE "^${CONDA_ENV_GENEVAL}\s"; then
    echo "Conda env '${CONDA_ENV_GENEVAL}' already exists — skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_GENEVAL}' from patched environment.yml..."
    conda env create -y -f "${PATCHED_YML}" -n "${CONDA_ENV_GENEVAL}"
fi

conda activate "${CONDA_ENV_GENEVAL}"

# 2-4. Install modern PyTorch (replaces the yanked 1.12.1 pin).
#      Targets CUDA 12.1 — change cu121 -> cu118 if your cluster uses CUDA 11.8.
echo "Installing modern PyTorch (CUDA 12.1)..."
pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available:', torch.cuda.is_available())"

# 2-5. Install compatible mmcv for mmdetection 2.x + the chosen PyTorch.
#      mim (from openmim) resolves the right pre-built wheel automatically.
echo "Installing openmim + mmcv..."
pip install --quiet openmim
# mmcv 1.7.x is the last series compatible with mmdetection 2.x
mim install -y "mmcv==1.7.2"

# 2-6. Install MMDetection (2.x branch required by GenEval)
if [ ! -d "${INSTALL_DIR}/mmdetection" ]; then
    echo "Cloning MMDetection (2.x branch) into ${INSTALL_DIR}/mmdetection..."
    git clone https://github.com/open-mmlab/mmdetection.git "${INSTALL_DIR}/mmdetection"
    git -C "${INSTALL_DIR}/mmdetection" checkout 2.x
else
    echo "MMDetection directory already exists — skipping clone."
fi

echo "Installing MMDetection..."
pip install --quiet -v -e "${INSTALL_DIR}/mmdetection"

# 2-7. Download object-detector models required for evaluation
echo "Downloading GenEval object-detector models to ${OBJECT_DETECTOR_DIR}..."
mkdir -p "${OBJECT_DETECTOR_DIR}"
bash "${INSTALL_DIR}/geneval/evaluation/download_models.sh" "${OBJECT_DETECTOR_DIR}/"

echo "GenEval environment setup complete."

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Setup finished successfully!"
echo ""
echo "To use ImgEdit:"
echo "  source /media02/nthuy/miniconda3/bin/activate"
echo "  conda activate ${CONDA_ENV_IMGEDIT}"
echo "  cd ${INSTALL_DIR}/ImgEdit"
echo ""
echo "To use GenEval:"
echo "  source /media02/nthuy/miniconda3/bin/activate"
echo "  conda activate ${CONDA_ENV_GENEVAL}"
echo "  cd ${INSTALL_DIR}/geneval"
echo "  # Run evaluation:"
echo "  python evaluation/evaluate_images.py \\"
echo "      <IMAGE_FOLDER>/ <PROMPT_FOLDER>/ \\"
echo "      --model-path ${OBJECT_DETECTOR_DIR}"
echo "============================================================"
