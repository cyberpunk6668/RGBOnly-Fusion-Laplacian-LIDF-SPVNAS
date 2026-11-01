#!/usr/bin/env bash
# Run in an isolated Python 3.10 virtualenv/conda environment, with CUDA 12.1.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dependency_dir="${TD_STAGE2_DEPENDENCY_DIR:-$project_dir/.deps/stage2}"
[[ "$dependency_dir" = /* ]] || { echo "TD_STAGE2_DEPENDENCY_DIR must be absolute" >&2; exit 2; }
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 12.1 toolkit (not merely the driver)}"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
export FORCE_CUDA=1
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
python - <<'CHECK'
import pathlib, sys, sysconfig
assert sys.version_info[:2] == (3, 10), 'Stage2 requires Python 3.10'
assert pathlib.Path(sysconfig.get_path('include'), 'Python.h').is_file(), 'Python development headers are required to compile CUDA extensions'
CHECK
for compiler in "$CC" "$CXX"; do
  command -v "$compiler" >/dev/null
  [[ "$("$compiler" -dumpversion)" = 11* ]] || { echo 'Use GCC/G++11 for the pinned CUDA build' >&2; exit 2; }
done
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.1'
python -m pip install 'setuptools==69.5.1' wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r "$project_dir/requirements/stage2.txt"
python -m pip install --no-deps 'torch-scatter==2.1.2+pt21cu121' --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
mkdir -p "$dependency_dir"
if [ ! -d "$dependency_dir/sparsehash/.git" ]; then
  git clone https://github.com/sparsehash/sparsehash.git "$dependency_dir/sparsehash"
fi
git -C "$dependency_dir/sparsehash" diff --quiet
git -C "$dependency_dir/sparsehash" diff --cached --quiet
git -C "$dependency_dir/sparsehash" checkout --detach f93c0c69e959c1c77611d0ba8d107aa971338811
(
  cd "$dependency_dir/sparsehash"
  ./configure --prefix="$dependency_dir/sparsehash_install"
  make -j "$MAX_JOBS"
  make install
)
export CPLUS_INCLUDE_PATH="$dependency_dir/sparsehash_install/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
if [ ! -d "$dependency_dir/torchsparse/.git" ]; then
  git clone https://github.com/mit-han-lab/torchsparse.git "$dependency_dir/torchsparse"
fi
git -C "$dependency_dir/torchsparse" diff --quiet
git -C "$dependency_dir/torchsparse" diff --cached --quiet
git -C "$dependency_dir/torchsparse" checkout --detach 385f5ce8718fcae93540511b7f5832f4e71fd835
python -m pip install --no-deps --no-build-isolation "$dependency_dir/torchsparse"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$project_dir/src" python -c 'import torch; from v2_lidf.runtime import validate_runtime; print(validate_runtime()); assert not torch.cuda.is_initialized()'
# The package remains editable so CUDA .cpp/.cu and topology assets are local.
python -m pip install --no-deps -e "$project_dir"

python -m pip check
