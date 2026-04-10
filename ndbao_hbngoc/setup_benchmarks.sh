#!/bin/bash

# =============================================================================
# Setup script for all T2I / image-editing evaluation benchmarks
#   TIFA v1.0        : https://github.com/Yushi-Hu/tifa
#   DrawBench        : prompt dataset — downloaded from HuggingFace
#   T2I-CompBench    : https://github.com/Karine-Huang/T2I-CompBench
#   ImgEdit          : https://github.com/PKU-YuanGroup/ImgEdit
#   GenEval          : https://github.com/djghosh13/geneval
# =============================================================================

set -e  # exit on error

# ---------------------------------------------------------------------------
# Configuration — adjust paths as needed
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc}"
CONDA_ENV_TIFA="tifa"
CONDA_ENV_T2ICOMP="t2icomp"
CONDA_ENV_IMGEDIT="imgedit"
CONDA_ENV_GENEVAL="geneval"
OBJECT_DETECTOR_DIR="${INSTALL_DIR}/geneval/models"

mkdir -p "$INSTALL_DIR"

echo "============================================================"
echo "Install directory    : $INSTALL_DIR"
echo "TIFA env             : $CONDA_ENV_TIFA"
echo "T2I-CompBench env    : $CONDA_ENV_T2ICOMP"
echo "ImgEdit env          : $CONDA_ENV_IMGEDIT"
echo "GenEval env          : $CONDA_ENV_GENEVAL"
echo "============================================================"

# ---------------------------------------------------------------------------
# Redirect pip cache, conda packages, and temp files away from the full
# root filesystem (/dev/nvme1n1p2) to the NAS mount which has ~700 GB free.
# NOTE: /media is on the SAME root filesystem — use /mnt/mmlab2024nas instead.
# ---------------------------------------------------------------------------
export TMPDIR="/mnt/mmlab2024nas/ldtuan/.tmp"
export PIP_CACHE_DIR="/mnt/mmlab2024nas/ldtuan/.pip_cache"
export CONDA_PKGS_DIRS="/mnt/mmlab2024nas/ldtuan/.conda_pkgs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# ---------------------------------------------------------------------------
# Initialise conda inside a non-interactive shell
# ---------------------------------------------------------------------------
source /mnt/mmlab2024nas/ldtuan/miniconda3/bin/activate

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
python3 -m pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 1-4. Install fairseq & modelscope dependencies for TIFA.
#
#  Three interrelated problems make "python3 -m pip install tifascore" fail on pip>=24.1:
#
#  (a) tifascore → modelscope[multi-modal] → fairseq==0.12.2
#      fairseq 0.12.2's sdist fails to build (setup.py can't find
#      fairseq/version.txt under build-isolation) and its metadata pins
#      omegaconf<2.1 whose wheels also have broken metadata.
#
#  (b) tifascore → modelscope[multi-modal] → pytorch-lightning<=1.7.7
#      EVERY version 1.7.0–1.7.7 declares "torch (>=1.9.*)" which is
#      invalid PEP 440.  pip>=24.1 rejects them all, making the dep tree
#      unresolvable.  This is the error you see:
#        "has invalid metadata: .* suffix can only be used with == or !="
#
#  (c) Root filesystem (/) is nearly full.  pip's build-isolation and cache
#      can spill onto / even with TMPDIR set, causing "No space left" errors.
#
#  Fix strategy:
#    - Use fairseq-fixed==0.12.3.1 (community fork with working build;
#      modelscope>=1.30.0 already switched to this).
#    - Strip modelscope[multi-modal] from tifascore's setup.py and install
#      plain modelscope + only the multi-modal deps tifascore actually uses.
#      This avoids the unresolvable pytorch-lightning<=1.7.7 constraint entirely.
#    - tifascore's runtime code only imports modelscope.pipelines,
#      modelscope.utils.constant, modelscope.outputs, and
#      modelscope.preprocessors.multi_modal — none of which need
#      pytorch-lightning or fairseq at import time.

echo "Installing fairseq-fixed (community fork with working build)..."
python3 -m pip install --quiet fairseq-fixed==0.12.3.1

echo "Installing modelscope (base, without [multi-modal] extras)..."
python3 -m pip install --quiet modelscope

# Install the subset of modelscope[multi-modal] extras that tifascore
# actually needs at runtime, deliberately omitting pytorch-lightning and fairseq.
echo "Installing modelscope multi-modal runtime deps..."
python3 -m pip install --quiet \
    addict attrs einops scipy simplejson sortedcontainers \
    accelerate cloudpickle "diffusers>=0.25.0" "ftfy>=6.0.3" \
    opencv-python pycocoevalcap pycocotools safetensors timm \
    tokenizers "transformers>=4.27.1" unicodedata2 Pillow \
    "omegaconf>=2.1"

# 1-5. Install TIFA.
#      Patch tifascore's setup.py to remove modelscope[multi-modal] (replaced
#      with plain modelscope above) and fairseq (replaced with fairseq-fixed).
echo "Installing TIFA..."
sed -i "s/'modelscope\[multi-modal\]'/'modelscope'/g" "${INSTALL_DIR}/tifa/setup.py" 2>/dev/null || true
sed -i "s/'fairseq[^']*'[,]*//" "${INSTALL_DIR}/tifa/setup.py" 2>/dev/null || true
sed -i 's/modelscope\[multi-modal\]/modelscope/' "${INSTALL_DIR}/tifa/requirements.txt" 2>/dev/null || true
sed -i '/fairseq/d' "${INSTALL_DIR}/tifa/requirements.txt" 2>/dev/null || true
python3 -m pip install --quiet -e "${INSTALL_DIR}/tifa"

# 1-5. Install VQA model backends used by TIFA
#      mPLUG-Owl2 (default recommended backend) + BLIP fallback
echo "Installing VQA backends..."
python3 -m pip install --quiet \
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

# Download DrawBench prompts directly from the Google Sheets CSV that the
# shunk031/DrawBench HF dataset loader uses as its source.
if [ ! -f "${DRAWBENCH_DIR}/drawbench_prompts.json" ]; then
    echo "Downloading DrawBench prompts..."
    python3 - <<'PYEOF'
import csv, json, os, urllib.request, io

url = "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/gviz/tq?tqx=out:csv"
data = urllib.request.urlopen(url).read().decode("utf-8")
reader = csv.DictReader(io.StringIO(data))
rows = [{"prompt": r.get("Prompts", r.get("prompts", "")).strip(),
         "category": r.get("Category", r.get("category", "")).strip()}
        for r in reader]
rows = [r for r in rows if r["prompt"]]  # drop empty rows

out = os.path.join(os.environ.get("DRAWBENCH_DIR",
                   "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/drawbench"),
                   "drawbench_prompts.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved {len(rows)} prompts to {out}")
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
python3 -m pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available:', torch.cuda.is_available())"

# 3-4. Install T2I-CompBench dependencies
echo "Installing T2I-CompBench dependencies..."
python3 -m pip install --quiet \
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
python3 -m pip install --quiet git+https://github.com/openai/CLIP.git

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
#      detectron2's setup.py does `import torch` at build time to detect
#      CUDA/torch versions.  --no-build-isolation lets it find installed torch.
#
#      System nvcc is CUDA 13.0 but PyTorch was compiled against CUDA 12.1.
#      torch/utils/cpp_extension.py has a strict version check that aborts on
#      mismatch.  We temporarily patch that check to a no-op, build detectron2,
#      then restore it.  Runtime CUDA compat is handled by the driver, not the
#      toolkit version, so this is safe.
echo "Installing detectron2..."
_CPP_EXT="$(python3 -c 'import torch.utils.cpp_extension as m; print(m.__file__)')"
cp "${_CPP_EXT}" "${_CPP_EXT}.bak"
python3 -c "
import re, pathlib
p = pathlib.Path('${_CPP_EXT}')
src = p.read_text()
# Replace the _check_cuda_version function body with 'return'
patched = re.sub(
    r'(def _check_cuda_version\([^)]*\)[^:]*:)',
    r'\1\n    return',
    src,
    count=1
)
p.write_text(patched)
print('Patched _check_cuda_version in', p)
"
TORCH_CUDA_ARCH_LIST="8.9" \
FORCE_CUDA=1 \
python3 -m pip install --quiet --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'
# Restore original cpp_extension.py
mv "${_CPP_EXT}.bak" "${_CPP_EXT}"
echo "Restored original cpp_extension.py"

# 3-7. Download PartiPrompts dataset from HuggingFace
PARTI_DIR="${INSTALL_DIR}/T2I-CompBench/parti_prompts"
mkdir -p "${PARTI_DIR}"
if [ ! -f "${PARTI_DIR}/PartiPrompts.tsv" ]; then
    echo "Downloading PartiPrompts dataset..."
    python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

dest_dir = os.path.join(os.environ.get("INSTALL_DIR", "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc"),
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
# Part 4 — ImgEdit
# ============================================================
echo ""
echo ">>> [4/5] Setting up ImgEdit"
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
python3 -m pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 1-4. Install Hugging Face stack + vision dependencies
echo "Installing Hugging Face + vision dependencies..."
python3 -m pip install --quiet \
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
python3 -m pip install --quiet qwen-vl-utils

# 1-6. Install SAM2
echo "Installing SAM2..."
python3 -m pip install --quiet git+https://github.com/facebookresearch/sam2.git

# 1-7. Install YOLO-World (optional; needed for bounding-box pipeline)
echo "Installing YOLO-World (inference only)..."
python3 -m pip install --quiet ultralytics

# 1-8. Install any extras referenced by ImgEdit tools/benchmark scripts
python3 -m pip install --quiet \
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
# Part 5 — GenEval
# ============================================================
echo ""
echo ">>> [5/5] Setting up GenEval"
echo "------------------------------------------------------------"

# 2-1. Clone the repository
if [ ! -d "${INSTALL_DIR}/geneval" ]; then
    echo "Cloning GenEval..."
    git clone https://github.com/djghosh13/geneval.git "${INSTALL_DIR}/geneval"
else
    echo "GenEval directory already exists — skipping clone."
fi

# 2-2. GenEval's environment.yml is a full lockfile with hundreds of tightly
#      pinned pip deps (fsspec==2022.11.0, clip-retrieval==2.37.0, etc.) that
#      create unresolvable conflicts on modern pip.  Instead of patching it,
#      we create a clean env and install only what GenEval actually imports:
#        numpy, pandas, Pillow, torch, mmdet (2.x), open_clip, clip_benchmark.

# 2-3. Create conda environment (plain Python, no environment.yml)
if conda env list | grep -qE "^${CONDA_ENV_GENEVAL}\s"; then
    echo "Conda env '${CONDA_ENV_GENEVAL}' already exists — skipping creation."
else
    echo "Creating conda env '${CONDA_ENV_GENEVAL}' (Python 3.9)..."
    conda create -y -n "${CONDA_ENV_GENEVAL}" python=3.9
fi

conda activate "${CONDA_ENV_GENEVAL}"

# 2-4. Install modern PyTorch (replaces the yanked 1.12.1 pin).
#      Targets CUDA 12.1 — change cu121 -> cu118 if your cluster uses CUDA 11.8.
echo "Installing modern PyTorch (CUDA 12.1)..."
python3 -m pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available:', torch.cuda.is_available())"

# 2-5. Install compatible mmcv for mmdetection 2.x + the chosen PyTorch.
#      mim (from openmim) resolves the right pre-built wheel automatically.
echo "Installing openmim + mmcv..."
python3 -m pip install --quiet openmim
# mmcv 1.7.x is the last series compatible with mmdetection 2.x
# Use --no-build-isolation so the build can find pkg_resources from the env
python3 -m pip install --quiet --no-build-isolation "mmcv==1.7.2"

# 2-6. Install MMDetection (2.x branch required by GenEval)
if [ ! -d "${INSTALL_DIR}/mmdetection" ]; then
    echo "Cloning MMDetection (2.x branch) into ${INSTALL_DIR}/mmdetection..."
    git clone https://github.com/open-mmlab/mmdetection.git "${INSTALL_DIR}/mmdetection"
    git -C "${INSTALL_DIR}/mmdetection" checkout -f 2.x
else
    echo "MMDetection directory already exists — skipping clone."
fi

echo "Installing MMDetection..."
# Use --no-build-isolation so setup.py can find torch
python3 -m pip install --quiet --no-build-isolation "${INSTALL_DIR}/mmdetection"

# 2-6b. Install remaining GenEval runtime deps
echo "Installing GenEval runtime dependencies (open_clip, clip_benchmark, etc.)..."
python3 -m pip install --quiet \
    numpy pandas Pillow \
    open_clip_torch clip-benchmark \
    scipy tqdm

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
echo "All benchmarks set up successfully!"
echo ""
echo "--- TIFA v1.0 ---"
echo "  source /mnt/mmlab2024nas/ldtuan/miniconda3/bin/activate && conda activate ${CONDA_ENV_TIFA}"
echo "  cd ${INSTALL_DIR}/tifa"
echo "  python3 tifa/tifa_score.py \\"
echo "      --model mplug-large \\"
echo "      --question_answer_file tifa_v1.0/tifa_v1.0_question_answers.json \\"
echo "      --image_file <your_images.json>"
echo ""
echo "--- DrawBench ---"
echo "  Prompts: ${INSTALL_DIR}/drawbench/drawbench_prompts.json"
echo "  Generate images from the prompts, then score with TIFA or CLIP."
echo ""
echo "--- T2I-CompBench ---"
echo "  source /mnt/mmlab2024nas/ldtuan/miniconda3/bin/activate && conda activate ${CONDA_ENV_T2ICOMP}"
echo "  cd ${INSTALL_DIR}/T2I-CompBench"
echo "  python3 UniDet_eval/pos_alignment.py --image_dir <images/> --output_dir <out/>"
echo "  python3 UniDet_eval/counting_alignment.py --image_dir <images/> --output_dir <out/>"
echo "  python3 BLIP_vqa/blip_vqa_eval.py --image_dir <images/> --output_dir <out/>"
echo ""
echo "--- ImgEdit ---"
echo "  source /mnt/mmlab2024nas/ldtuan/miniconda3/bin/activate && conda activate ${CONDA_ENV_IMGEDIT}"
echo "  cd ${INSTALL_DIR}/ImgEdit"
echo ""
echo "--- GenEval ---"
echo "  source /mnt/mmlab2024nas/ldtuan/miniconda3/bin/activate && conda activate ${CONDA_ENV_GENEVAL}"
echo "  cd ${INSTALL_DIR}/geneval"
echo "  python evaluation/evaluate_images.py \\"
echo "      <IMAGE_FOLDER>/ <PROMPT_FOLDER>/ \\"
echo "      --model-path ${OBJECT_DETECTOR_DIR}"
echo "============================================================"