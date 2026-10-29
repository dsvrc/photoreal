#!/usr/bin/env bash
# Collects everything needed to debug the audio2photoreal environment setup on a
# Linux / HPC (PBS/SLURM) server. Read-only: it installs and changes nothing.
#
# Usage (from the repo root, ideally inside an interactive GPU job):
#   bash scripts/server_info.sh            # writes server_info_<host>_<date>.log
#   bash scripts/server_info.sh a2p_env    # also inspects that conda env
#
# Then send back the generated .log file.

ENV_NAME="${1:-a2p_env}"
LOG="server_info_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee "$LOG") 2>&1

section() { printf '\n==================== %s ====================\n' "$*"; }
run() { printf '\n$ %s\n' "$*"; timeout 60 bash -c "$*" 2>&1 || printf '[exit %s]\n' "$?"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "BASICS"
run 'date; hostname -f 2>/dev/null || hostname; whoami; id'
run 'uname -a'
run 'cat /etc/os-release 2>/dev/null | head -5'
run 'ldd --version 2>&1 | head -1'          # glibc version (matters for wheels)
run 'echo "SHELL=$SHELL  HOME=$HOME  PWD=$PWD"'
run 'echo "TMPDIR=${TMPDIR:-<unset>}"'
run 'ulimit -a'

section "SCHEDULER / JOB"
run 'env | grep -E "^(PBS_|SLURM_)" | grep -viE "token|secret|pass" | sort'
have qstat && run 'qstat -q 2>/dev/null | head -30'
have sinfo && run 'sinfo -s 2>/dev/null | head -30'

section "CPU / MEMORY"
run 'lscpu | grep -E "Model name|Socket|Core|Thread|^CPU\(s\)|Architecture"'
run 'nproc'
run 'free -h'

section "DISK / QUOTA"
run 'df -h "$HOME" "$PWD" /tmp ${TMPDIR:+"$TMPDIR"} 2>/dev/null'
run 'quota -s 2>/dev/null || true'
have lfs && run 'lfs quota -h -u "$USER" "$PWD" 2>/dev/null'
run 'du -sh "$HOME/.cache/pip" "$HOME/.cache/torch" "$HOME/.conda" 2>/dev/null'

section "GPU / DRIVER"
if have nvidia-smi; then
    run 'nvidia-smi'
    run 'nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv'
else
    echo "nvidia-smi NOT found (login node, or no GPU allocated to this job)"
fi
run 'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"'
run 'cat /proc/driver/nvidia/version 2>/dev/null'

section "CUDA TOOLKITS / COMPILERS"
run 'which nvcc && nvcc --version'
run 'echo "CUDA_HOME=${CUDA_HOME:-<unset>}  CUDA_PATH=${CUDA_PATH:-<unset>}"'
run 'ls -d /usr/local/cuda* /opt/cuda* /usr/local/apps/cuda* /apps/cuda* 2>/dev/null'
run 'which gcc g++ && gcc --version | head -1 && g++ --version | head -1'
run 'ls /usr/bin/gcc-* /usr/bin/g++-* 2>/dev/null'
run 'which cmake ninja make git wget curl unzip tar ffmpeg 2>&1'
run 'echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"'

section "ENVIRONMENT MODULES"
for f in "${MODULESHOME:-/nonexistent}/init/bash" /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
    type module >/dev/null 2>&1 || { [ -f "$f" ] && . "$f" >/dev/null 2>&1; }
done
if type module >/dev/null 2>&1; then
    # module is a shell function, so call it directly rather than through run()
    printf '\n$ module list\n'; module list 2>&1
    printf '\n$ module avail | grep cuda/gcc/conda/...\n'
    module avail 2>&1 | grep -iE "cuda|gcc|gnu|compiler|anaconda|conda|python|ffmpeg|cudnn" | head -60
else
    echo "No environment-modules / lmod detected"
fi

section "CONDA"
CONDA_BIN="$(command -v conda || echo "${CONDA_EXE:-}")"
if [ -n "$CONDA_BIN" ]; then
    run "\"$CONDA_BIN\" --version"
    run "\"$CONDA_BIN\" info"
    run "\"$CONDA_BIN\" env list"
    run "\"$CONDA_BIN\" config --show channels channel_priority 2>/dev/null"
    have mamba && run 'mamba --version | head -2'
    run 'cat ~/.condarc 2>/dev/null'
else
    echo "conda NOT found on PATH"
fi

section "PIP CONFIG"
run 'cat ~/.pip/pip.conf ~/.config/pip/pip.conf /etc/pip.conf 2>/dev/null'
run 'env | grep -iE "^PIP_" | sed -E "s#(://)[^@/]+@#\1***@#"'

section "NETWORK (can this node reach the package/model servers?)"
run 'env | grep -iE "^(http|https|no|all)_proxy=" | sed -E "s#(://)[^@/]+@#\1***@#"'
for url in \
    https://pypi.org/simple/pip/ \
    https://download.pytorch.org/whl/cu117/ \
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html \
    https://github.com \
    https://conda.anaconda.org/conda-forge/ \
    http://audio2photoreal_models.berkeleyvision.org/asset_models.tar ; do
    code=$(curl -s -o /dev/null -I -L -m 20 -w '%{http_code}' "$url" 2>/dev/null)
    printf '%-95s -> HTTP %s\n' "$url" "${code:-FAIL}"
done

section "CONDA ENV: $ENV_NAME"
ENV_PY=""
if [ -n "$CONDA_BIN" ]; then
    ENV_PREFIX="$("$CONDA_BIN" env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
    [ -n "$ENV_PREFIX" ] && [ -x "$ENV_PREFIX/bin/python" ] && ENV_PY="$ENV_PREFIX/bin/python"
fi
if [ -n "$ENV_PY" ]; then
    run "\"$ENV_PY\" --version; \"$ENV_PY\" -m pip --version"
    run "\"$ENV_PY\" -m pip list 2>/dev/null"
    run "\"$ENV_PY\" -m pip check 2>&1 | head -40"
    printf '\n$ import check\n'
    "$ENV_PY" - <<'PY' 2>&1
import importlib, sys
print("python", sys.version)
mods = ["numpy", "torch", "torchaudio", "torchvision", "fairseq", "hydra", "omegaconf",
        "pytorch3d", "gradio", "attrdict", "einops", "mediapy", "cv2", "sklearn", "scipy",
        "blobfile", "tensorboardX", "tensorboard", "trimesh", "igl", "tqdm"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:13s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL  {m:13s} {type(e).__name__}: {str(e).splitlines()[0][:150]}")
try:
    import torch
    print("torch.version.cuda =", torch.version.cuda, "| cudnn =", torch.backends.cudnn.version())
    print("cuda available     =", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU{i}: {torch.cuda.get_device_name(i)} cc={torch.cuda.get_device_capability(i)}")
        print("arch list in torch =", torch.cuda.get_arch_list())
except Exception as e:
    print("torch check failed:", e)
PY
else
    echo "Env '$ENV_NAME' not found (that's fine - setup_env.sh will create it)."
fi

section "REPO"
run 'git -C "$PWD" log --oneline -1 2>/dev/null; ls'
run 'ls -la assets checkpoints 2>/dev/null | head -40'
run 'du -sh checkpoints dataset assets 2>/dev/null'

section "DONE"
echo "Saved to: $PWD/$LOG"
