#!/usr/bin/env bash
# One-shot environment setup for audio2photoreal on a Linux GPU server
# (conda + pip, python 3.9, torch 2.0.1 with CUDA 11.7/11.8 wheels, pytorch3d).
#
# Why plain `pip install -r scripts/requirements.txt` fails:
#   pip >= 24.1 rejects the metadata of omegaconf 2.0.6 (needed by fairseq 0.12.2),
#   so pip backtracks to the broken fairseq 0.12.1 sdist and dies with
#   "FileNotFoundError: fairseq/version.txt". Pinning pip < 24.1 fixes it.
#
# Usage (from anywhere, on a node with internet access):
#   bash scripts/setup_env.sh                # create/reuse env a2p_env, download PXB184 models
#   bash scripts/setup_env.sh --fresh        # delete and recreate the env first (recommended after failed attempts)
#   bash scripts/setup_env.sh --all-models   # models for all 4 subjects
#   bash scripts/setup_env.sh --datasets     # also download the 4 datasets
#   bash scripts/setup_env.sh --no-download  # environment only
#   bash scripts/setup_env.sh --cuda cu118   # force the torch CUDA build (default: auto)
#   bash scripts/setup_env.sh --env NAME     # use another env name
#
# Env vars: A2P_CACHE=/big/disk/dir puts pip cache, temp files and downloads there
#           (default: <repo>/.setup_cache, i.e. next to the repo rather than in $HOME).

set -Eeuo pipefail

ENV_NAME=a2p_env
FRESH=0
DOWNLOAD=1
ALL_MODELS=0
DATASETS=0
CU=auto

while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift ;;
        --fresh) FRESH=1 ;;
        --no-download) DOWNLOAD=0 ;;
        --all-models) ALL_MODELS=1 ;;
        --datasets) DATASETS=1 ;;
        --cuda) CU="$2"; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
    esac
    shift
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG="$REPO/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[WARN] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[FAIL] %s\033[0m\n' "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
ERR_TRAP='die "line $LINENO: \"$BASH_COMMAND\" failed (full log: $LOG)"'
trap "$ERR_TRAP" ERR

# Keep caches/temp off $HOME (quota) and off a small /tmp; ignore ~/.local packages.
CACHE="${A2P_CACHE:-$REPO/.setup_cache}"
export PIP_CACHE_DIR="$CACHE/pip" TMPDIR="$CACHE/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CACHE/dl" "$CACHE/markers"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
unset PYTHONPATH PYTHONHOME || true

load_module_system() {
    set +u
    for f in "${MODULESHOME:-/nonexistent}/init/bash" /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        type module >/dev/null 2>&1 && break
        [ -f "$f" ] && . "$f" >/dev/null 2>&1 || true
    done
    set -u
    type module >/dev/null 2>&1
}

try_module_load() {  # try each module name until one loads
    load_module_system || return 1
    local m
    for m in "$@"; do
        module load "$m" >/dev/null 2>&1 && { echo "loaded module $m"; return 0; }
    done
    return 1
}

echo "repo: $REPO   host: $(hostname)   log: $LOG"

# ---------------------------------------------------------------------------
step "Locating conda"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
    try_module_load anaconda3 anaconda miniconda3 miniconda miniforge3 conda || true
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
fi
[ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN" ] || die "conda not found. Activate your anaconda first (e.g. 'source <anaconda>/bin/activate') and rerun."
CONDA_BASE="$("$CONDA_BIN" info --base)"
echo "conda: $CONDA_BIN ($("$CONDA_BIN" --version))"

# conda's shell functions are not written for set -eu / ERR traps, so relax them while activating
conda_activate() {
    trap - ERR; set +eu
    [ "$(type -t conda)" = function ] || . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$1"; local rc=$?
    set -eu; trap "$ERR_TRAP" ERR
    return $rc
}

env_exists() { "$CONDA_BIN" env list | awk -v n="$ENV_NAME" '$1 == n { found = 1 } END { exit !found }'; }

step "Conda env '$ENV_NAME'"
if env_exists && [ "$FRESH" = 1 ]; then
    echo "--fresh: removing existing env $ENV_NAME"
    conda_activate base
    "$CONDA_BIN" remove -y -n "$ENV_NAME" --all
fi
if ! env_exists; then
    # conda-forge only: avoids the Anaconda 'defaults' ToS prompt in newer conda
    "$CONDA_BIN" create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.9
fi
conda_activate "$ENV_NAME" || die "could not activate $ENV_NAME"
PY="$CONDA_PREFIX/bin/python"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYVER" = "3.9" ] || die "env $ENV_NAME has python $PYVER, need 3.9. Rerun with --fresh."
"$CONDA_BIN" env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null 2>&1 \
    || warn "could not persist PYTHONNOUSERSITE=1 in the env; export it yourself if ~/.local has packages"
echo "python: $PY ($("$PY" --version 2>&1))"

# ---------------------------------------------------------------------------
step "Detecting GPU / choosing torch CUDA build"
GPU_CC=""    # highest compute capability, e.g. 8.0
GPU_CCS=""   # all distinct ones, e.g. 7.0;8.0
GPU_NAMES=""
if have nvidia-smi; then
    GPU_NAMES="$(nvidia-smi -L 2>/dev/null || true)"
    echo "$GPU_NAMES"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/driver: /' || true
    GPU_CCS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -uV | paste -sd';' - || true)"
    GPU_CC="${GPU_CCS##*;}"
    echo "compute capability: ${GPU_CCS:-unknown}"
else
    warn "nvidia-smi not found (login node?). Setup still works; the GPU checks at the end will be skipped."
fi
if [ "$CU" = auto ]; then
    CU=cu117   # what the authors tested
    # Ada / Hopper GPUs (sm_89, sm_90) need CUDA 11.8 builds
    if [ -n "$GPU_CC" ] && awk -v c="$GPU_CC" 'BEGIN { exit !(c + 0 >= 8.9) }'; then CU=cu118; fi
    if printf '%s\n' "$GPU_NAMES" | grep -iE 'H100|H200|H800|GH200|L40|L4\b|RTX 40|Ada' >/dev/null; then CU=cu118; fi
fi
case "$CU" in cu117|cu118) ;; *) die "--cuda must be cu117 or cu118" ;; esac
CUDA_VER="${CU:2:2}.${CU:4:1}"   # 11.7 / 11.8
echo "using torch build: $CU (CUDA $CUDA_VER, bundled in the wheel - no system CUDA needed)"

# ---------------------------------------------------------------------------
step "Pinning pip < 24.1 (fixes the fairseq/omegaconf failure)"
"$PY" -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
pipi() { "$PY" -m pip install --prefer-binary "$@"; }

CONSTRAINTS="$CACHE/constraints.txt"
cat > "$CONSTRAINTS" <<EOF
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
numpy==1.23.5
setuptools<70
EOF

step "Installing PyTorch 2.0.1 + $CU"
pipi numpy==1.23.5
pipi "torch==2.0.1+$CU" "torchvision==0.15.2+$CU" "torchaudio==2.0.2+$CU" \
    --extra-index-url "https://download.pytorch.org/whl/$CU"

step "Installing fairseq 0.12.2 / hydra-core 1.0.7 / omegaconf 2.0.6"
if ! pipi -c "$CONSTRAINTS" --only-binary=fairseq fairseq==0.12.2 hydra-core==1.0.7 omegaconf==2.0.6; then
    warn "no fairseq wheel for this platform, building it from source"
    pipi -c "$CONSTRAINTS" cython
    pipi -c "$CONSTRAINTS" --no-build-isolation fairseq==0.12.2 hydra-core==1.0.7 omegaconf==2.0.6
fi

step "Installing the remaining Python packages"
# opencv *headless*: the GUI build needs libGL.so.1, which servers usually lack
pipi -c "$CONSTRAINTS" \
    attrdict==2.0.1 einops==0.7.0 mediapy==1.2.0 opencv-python-headless==4.8.1.78 \
    protobuf==4.25.1 tensorboard tensorboardX==2.6.2.2 tqdm==4.66.3 matplotlib \
    scikit-learn scipy blobfile packaging trimesh imageio-ffmpeg fvcore iopath ninja

step "Installing gradio (only needed for demo/demo.py)"
# pydantic 2.11 breaks gradio 4.x ("argument of type 'bool' is not iterable")
pipi -c "$CONSTRAINTS" gradio==4.31.3 huggingface-hub==0.19.4 "pydantic<2.11" \
    || warn "gradio failed to install: training/sampling/rendering still work, only the web demo needs it"

pipi -c "$CONSTRAINTS" libigl==2.5.1 \
    || warn "libigl not installed (only used by visualize/ca_body/utils/geom_body.py)"

# ---------------------------------------------------------------------------
step "Installing pytorch3d"
pt3d_ok() { "$PY" -c "import torch, pytorch3d; from pytorch3d import _C" >/dev/null 2>&1; }

build_pytorch3d_from_source() {
    local nvcc_bin="" gcc_major
    # 1) nvcc from CUDA 11.x (torch refuses to build extensions with a different CUDA major)
    for c in "${CUDA_HOME:-}/bin/nvcc" "$(command -v nvcc || true)" "/usr/local/cuda-$CUDA_VER/bin/nvcc" /usr/local/cuda/bin/nvcc; do
        [ -x "$c" ] && "$c" --version | grep "release 11\." >/dev/null && { nvcc_bin="$c"; break; }
    done
    if [ -z "$nvcc_bin" ] && try_module_load "cuda/$CUDA_VER" "cuda/$CUDA_VER.0" "cuda/$CUDA_VER.1" "cuda-$CUDA_VER" "cuda/11.8" "cuda/11.7"; then
        have nvcc && nvcc --version | grep "release 11\." >/dev/null && nvcc_bin="$(command -v nvcc)"
    fi
    if [ -z "$nvcc_bin" ]; then
        echo "no CUDA 11 nvcc on this machine, installing CUDA $CUDA_VER toolkit into the env"
        "$CONDA_BIN" install -y -n "$ENV_NAME" --override-channels \
            -c "nvidia/label/cuda-$CUDA_VER.0" -c conda-forge cuda-toolkit
        [ -e "$CONDA_PREFIX/lib64" ] || ln -s lib "$CONDA_PREFIX/lib64"
        nvcc_bin="$CONDA_PREFIX/bin/nvcc"
    fi
    export CUDA_HOME="$(cd "$(dirname "$nvcc_bin")/.." && pwd)"
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "nvcc: $nvcc_bin ($("$nvcc_bin" --version | grep release))"

    # 2) host compiler: CUDA 11.7/11.8 need gcc <= 11
    gcc_major="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
    if [ "${gcc_major:-0}" -lt 7 ] || [ "$gcc_major" -gt 11 ]; then
        try_module_load gcc/11 gcc/10 gcc/9 gcc/11.2.0 gcc/10.2.0 gcc/9.4.0 || true
        gcc_major="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
    fi
    if [ "${gcc_major:-0}" -lt 7 ] || [ "$gcc_major" -gt 11 ]; then
        echo "system gcc ($gcc_major) unsuitable, installing gcc 11 into the env"
        "$CONDA_BIN" install -y -n "$ENV_NAME" --override-channels -c conda-forge gcc_linux-64=11 gxx_linux-64=11
        export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    else
        export CC="$(command -v gcc)" CXX="$(command -v g++)"
    fi
    echo "compiler: $CC ($("$CC" -dumpversion))"

    # 3) build only for the GPUs we have (or a sensible default set)
    if [ -n "$GPU_CCS" ]; then
        export TORCH_CUDA_ARCH_LIST="${GPU_CCS}+PTX"
    elif [ "$CU" = cu118 ]; then
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    else
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
    fi
    local jobs; jobs="$(nproc 2>/dev/null || echo 4)"; [ "$jobs" -gt 8 ] && jobs=8
    echo "building pytorch3d v0.7.5 for arch $TORCH_CUDA_ARCH_LIST with $jobs jobs (takes 10-30 min)"
    FORCE_CUDA=1 MAX_JOBS="$jobs" pipi -c "$CONSTRAINTS" --no-build-isolation \
        "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.5.tar.gz"
}

if pt3d_ok; then
    echo "pytorch3d already installed and importable"
else
    "$PY" -m pip uninstall -y pytorch3d >/dev/null 2>&1 || true
    PT3D_WHEELS="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_${CU}_pyt201/download.html"
    echo "trying prebuilt wheel: $PT3D_WHEELS"
    if pipi --no-index --no-cache-dir pytorch3d -f "$PT3D_WHEELS" && pt3d_ok; then
        echo "prebuilt pytorch3d wheel installed"
    else
        warn "prebuilt wheel unavailable or broken, building from source"
        "$PY" -m pip uninstall -y pytorch3d >/dev/null 2>&1 || true
        build_pytorch3d_from_source
        pt3d_ok || die "pytorch3d built but does not import; see the log above"
    fi
fi

# ---------------------------------------------------------------------------
step "ffmpeg (static build from imageio-ffmpeg, linked into the env)"
ln -sf "$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$CONDA_PREFIX/bin/ffmpeg"
"$CONDA_PREFIX/bin/ffmpeg" -hide_banner -version | head -1

step "pip check"
"$PY" -m pip check || warn "pip reports dependency conflicts above (usually harmless pins, the checks at the end are what matter)"

# ---------------------------------------------------------------------------
download() {  # url out
    if have wget; then wget -c --tries=5 --progress=dot:giga -O "$2" "$1"
    else curl -fL --retry 5 -C - -o "$2" "$1"; fi
}

fetch_tar() {  # url  done_if_exists  [extract_dir]
    local url="$1" check="$2" dest="${3:-.}" name marker f
    name="$(basename "$url")"; marker="$CACHE/markers/$name.done"; f="$CACHE/dl/$name"
    if [ -e "$check" ] || [ -e "$marker" ]; then echo "already have $name"; return 0; fi
    mkdir -p "$dest"
    download "$url" "$f"
    tar xf "$f" -C "$dest"
    rm -f "$f"
    touch "$marker"
}

fetch_zip() {  # url  extract_dir
    local url="$1" dest="$2" name marker f
    name="$(basename "$url")"; marker="$CACHE/markers/$name.done"; f="$CACHE/dl/$name"
    [ -e "$marker" ] && { echo "already have $name"; return 0; }
    mkdir -p "$dest"
    download "$url" "$f"
    if have unzip; then unzip -q -o "$f" -d "$dest"; else "$PY" -m zipfile -e "$f" "$dest"; fi
    rm -f "$f"
    touch "$marker"
}

if [ "$DOWNLOAD" = 1 ]; then
    SUBJECTS="PXB184"
    [ "$ALL_MODELS" = 1 ] && SUBJECTS="PXB184 RLW104 TXB805 GQS883"
    step "Downloading assets + models for: $SUBJECTS"
    df -h "$REPO" | tail -1
    fetch_tar http://audio2photoreal_models.berkeleyvision.org/asset_models.tar assets/wav2vec_large.pt
    for s in $SUBJECTS; do
        check="$CACHE/markers/${s}_models.tar.done"
        [ "$s" = PXB184 ] && check=checkpoints/diffusion/c1_face/model000155000.pt
        fetch_tar "http://audio2photoreal_models.berkeleyvision.org/${s}_models.tar" "$check"
        fetch_tar "https://github.com/facebookresearch/ca_body/releases/download/v0.0.1-alpha/${s}.tar.gz" \
            "checkpoints/ca_body/data/$s" checkpoints/ca_body/data
    done
    if [ "$DATASETS" = 1 ]; then
        step "Downloading datasets"
        for s in PXB184 RLW104 TXB805 GQS883; do
            fetch_zip "https://github.com/facebookresearch/audio2photoreal/releases/download/v1.0/${s}.zip" dataset
        done
    fi
fi

# ---------------------------------------------------------------------------
step "Verifying the installation"
trap - ERR
set +e
"$PY" - <<'PY'
import importlib, os, shutil, subprocess, sys, traceback
fails, skips = [], []

def check(name, fn, required=True):
    try:
        info = fn()
        print(f"  OK    {name}" + (f"  ({info})" if info else ""))
    except Exception as e:
        (fails if required else skips).append(name)
        print(f"  {'FAIL' if required else 'WARN'}  {name}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")

def ver(m):
    return lambda: getattr(importlib.import_module(m), "__version__", "")

print("python", sys.version.split()[0], "at", sys.executable)
for m in ["numpy", "torch", "torchaudio", "torchvision", "fairseq", "hydra", "omegaconf",
          "attrdict", "einops", "mediapy", "cv2", "sklearn", "scipy", "blobfile",
          "tensorboardX", "trimesh", "pytorch3d"]:
    check(m, ver(m))
for m in ["gradio", "igl"]:
    check(m, ver(m), required=False)

import torch
print(f"  torch CUDA build = {torch.version.cuda}, arch list = {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    def gpu():
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        cc = torch.cuda.get_device_capability()
        return f"{torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}"
    check("torch GPU matmul", gpu)

    def raster():
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        v = torch.tensor([[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]], device="cuda")
        f = torch.tensor([[[0, 1, 2]]], device="cuda")
        p2f = rasterize_meshes(Meshes(verts=v, faces=f), image_size=16)[0]
        assert (p2f >= 0).any(), "nothing rasterized"
        return "CUDA rasterizer works"
    check("pytorch3d GPU rasterize", raster)
else:
    print("  SKIP  GPU checks (no GPU visible here - rerun this script inside a GPU job to test them)")

def ff():
    exe = shutil.which("ffmpeg")
    assert exe, "ffmpeg not on PATH"
    enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    assert "libx264" in enc and "aac" in enc, "ffmpeg lacks libx264/aac"
    return exe
check("ffmpeg (libx264 + aac)", ff)

if os.path.exists("assets/wav2vec_large.pt"):
    def w2v():
        import fairseq
        m, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(["assets/wav2vec_large.pt"])
        return type(m[0]).__name__
    check("fairseq loads assets/wav2vec_large.pt", w2v)
else:
    print("  SKIP  wav2vec load test (assets not downloaded)")

check("repo import: model.diffusion", lambda: importlib.import_module("model.diffusion") and "")
check("repo import: visualize.render_codes", lambda: importlib.import_module("visualize.render_codes") and "")

print()
if skips:
    print("optional pieces missing:", ", ".join(skips))
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL REQUIRED CHECKS PASSED")
PY
RC=$?

step "Summary"
if [ $RC -eq 0 ]; then
    cat <<EOF
Environment '$ENV_NAME' is ready.

  conda activate $ENV_NAME
  cd $REPO

  # face codes, then body + render (PXB184)
  python -m sample.generate --model_path checkpoints/diffusion/c1_face/model000155000.pt --num_samples 10 --num_repetitions 5 --timestep_respacing ddim500 --guidance_param 10.0
  python -m sample.generate --model_path checkpoints/diffusion/c1_pose/model000340000.pt --resume_trans checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt --num_samples 10 --num_repetitions 5 --timestep_respacing ddim500 --guidance_param 2.0

  # web demo (needs a GPU node; uses a public gradio share link)
  python -m demo.demo

You can delete $CACHE to free space. Log: $LOG
EOF
else
    echo "Some checks failed. Send this log: $LOG"
    exit 1
fi
