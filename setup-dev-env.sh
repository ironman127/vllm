#!/bin/bash
# =============================================================================
# vLLM 从源码全编译 —— 一键安装脚本（CUDA 13.0 栈 / 535 宿主驱动前向兼容）
#
# 目标机基线：TencentOS 3.2(el8) + NVIDIA H20 + 宿主驱动 535.x(CUDA 12.2)
# 安装顺序：1.兼容包  2.Rust  3.gcc-toolset  4.CUDA toolkit  →  uv/venv/torch → 全编译
#
# 用法：  bash /root/vllm/setup-dev-env.sh
# 说明：  幂等设计，已装的步骤会跳过；只装 toolkit/compat，绝不装 driver。
#         详细原理见 INSTALL-vLLM-CUDA13.md。
# =============================================================================
set -euo pipefail

VLLM_DIR=/root/vllm
VENV_DIR="$VLLM_DIR/.venv"
CUDA_HOME=/usr/local/cuda-13.0
NVIDIA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn] $*\033[0m"; }

# -----------------------------------------------------------------------------
log "0/6  挂 NVIDIA 官方 rhel8 源（内部源没有 CUDA 13 包）"
# -----------------------------------------------------------------------------
command -v dnf >/dev/null || { echo "需要 dnf"; exit 1; }
rpm -q dnf-plugins-core >/dev/null 2>&1 || dnf install -y dnf-plugins-core
if ! dnf repolist 2>/dev/null | grep -qi 'cuda-rhel8'; then
  dnf config-manager --add-repo "$NVIDIA_REPO"
  dnf clean all && dnf makecache || warn "makecache 失败，继续尝试"
fi

# -----------------------------------------------------------------------------
log "1/6  安装前向兼容包 cuda-compat-13-0（让 535 宿主跑 CUDA 13；不碰驱动）"
# -----------------------------------------------------------------------------
if rpm -qa | grep -q '^cuda-compat-13-0'; then
  echo "cuda-compat-13-0 已安装，跳过"
else
  dnf install -y cuda-compat-13-0
fi
# 注册到 ld 缓存，使 compat 的 libcuda 优先于宿主 535
if [ -d "$CUDA_HOME/compat" ] && ! grep -rq "$CUDA_HOME/compat" /etc/ld.so.conf.d/ 2>/dev/null; then
  echo "$CUDA_HOME/compat" > /etc/ld.so.conf.d/00-cuda-compat.conf
fi
ldconfig
ldconfig -p | grep -m1 'libcuda.so.1' || warn "未检测到 libcuda.so.1"

# -----------------------------------------------------------------------------
log "2/6  安装 Rust 工具链（vLLM 用 setuptools-rust 构建）"
# -----------------------------------------------------------------------------
# 换源（关键，否则 rustup 下载和 crates 拉取都很慢）：用字节 rsproxy.cn 镜像
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
export RUSTUP_DIST_SERVER="https://rsproxy.cn"          # rustup 工具链下载源
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"   # rustup 自身更新源
# 读取仓库指定的 rust 版本（rust-toolchain.toml: channel = "x.y"）
RUST_VERSION=$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$VLLM_DIR/rust-toolchain.toml" 2>/dev/null)
RUST_VERSION=${RUST_VERSION:-1.95}

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain none
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

# cargo crates 镜像（写入 ~/.cargo/config.toml，sparse 协议）
mkdir -p "$HOME/.cargo"
if ! grep -q 'rsproxy' "$HOME/.cargo/config.toml" 2>/dev/null; then
  cat >> "$HOME/.cargo/config.toml" <<'EOF'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[registries.rsproxy-sparse]
index = "sparse+https://rsproxy.cn/index/"
[net]
git-fetch-with-cli = true
EOF
fi

# 切换到指定版本（rust-toolchain.toml 指定，本机 1.95）
rustup toolchain install "$RUST_VERSION"
rustup default "$RUST_VERSION"
rustc --version

# -----------------------------------------------------------------------------
log "3/6  安装编译工具：gcc-toolset-12（C++20）+ ccache（增量编译缓存）"
# -----------------------------------------------------------------------------
# 切勿 dnf update 系统 gcc：el8 锁死 8.5，强替会破坏系统。用并行工具链。
if [ ! -f /opt/rh/gcc-toolset-12/enable ]; then
  dnf install -y gcc-toolset-12
fi
# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-12/enable
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CMAKE_CUDA_HOST_COMPILER=$CXX
gcc --version | head -1

# ccache：vLLM setup.py 检测到后自动加 -DCMAKE_*_COMPILER_LAUNCHER=ccache，
# 按源文件内容哈希缓存编译产物，第二遍编译命中后可快数倍~数十倍（解决“重编很慢”）。
#
# 【关键】必须用 ccache >= 4.x！el8 的 dnf 只有 3.7.7，它不认识 CMake 给 nvcc 传的
# “-x cu”，会判 "Unsupported language: cu" 直接 fallback，导致 130 个 CUDA kernel
# 一个都不缓存 → CUDA 部分每次全量重编（表现为“改一行还在全编译”）。4.x 才修复。
CCACHE_VER=4.10.2
need_ccache=1
if command -v ccache >/dev/null 2>&1; then
  cur=$(ccache --version | sed -n '1s/.*version \([0-9]\+\).*/\1/p')
  [ "${cur:-0}" -ge 4 ] 2>/dev/null && need_ccache=0
fi
if [ "$need_ccache" = 1 ]; then
  log "  安装 ccache ${CCACHE_VER} 静态二进制（dnf 的 3.7.7 不支持 CUDA -x cu）"
  tmpd=$(mktemp -d); pkg="ccache-${CCACHE_VER}-linux-x86_64"
  url="https://github.com/ccache/ccache/releases/download/v${CCACHE_VER}/${pkg}.tar.xz"
  ( cd "$tmpd" && { curl -fsSL -o c.tar.xz "$url" || curl -fsSL -o c.tar.xz "https://gh-proxy.com/$url"; } \
    && tar xf c.tar.xz && install -m755 "$pkg/ccache" /usr/local/bin/ccache )
  rm -rf "$tmpd"; hash -r
fi
export CCACHE_DIR=/root/.cache/ccache
mkdir -p "$CCACHE_DIR"
ccache -M 50G >/dev/null
ccache -o sloppiness=pch_defines,time_macros,include_file_mtime,include_file_ctime,gcno_cwd >/dev/null 2>&1 \
  || ccache -o sloppiness=pch_defines,time_macros,include_file_mtime,include_file_ctime >/dev/null 2>&1 || true
ccache -o hash_dir=false >/dev/null 2>&1 || true
ccache -o compiler_check=content >/dev/null 2>&1 || true
ccache --version | head -1

# -----------------------------------------------------------------------------
log "4/6  安装 CUDA 13.0 编译工具链（只装 toolkit，绝不装 cuda/cuda-drivers）"
# -----------------------------------------------------------------------------
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "toolkit 已存在：$($CUDA_HOME/bin/nvcc --version | tail -1)"
else
  # 优先 meta 包；若 No match 则点名装编译必需组件
  dnf install -y cuda-toolkit-13-0 || dnf install -y \
    cuda-nvcc-13-0 cuda-cudart-devel-13-0 cuda-crt-13-0 cuda-cccl-13-0 \
    cuda-nvrtc-devel-13-0 libcublas-devel-13-0 libcusparse-devel-13-0 \
    libcusolver-devel-13-0 libcurand-devel-13-0 libcufft-devel-13-0
fi
export CUDA_HOME CUDACXX="$CUDA_HOME/bin/nvcc" CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
# 关键校验：必须有无版本号软链 libcudart.so（pip cu13 运行时包缺它）
ls -l "$CUDA_HOME/lib64/libcudart.so" || warn "缺 libcudart.so 软链，cmake 可能找不到 cudart"
nvcc --version | tail -2

# -----------------------------------------------------------------------------
log "5/6  uv + Python 3.12 venv + torch(cu130) 依赖"
# -----------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV_DIR" ] || uv venv --python 3.12 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# 显式 cu130，切勿 auto（auto 会被 535 误导选低版本）
uv pip install --upgrade pip
uv pip install vllm --torch-backend=cu130 || warn "vllm wheel 安装失败，继续走源码编译"
# 全编译所需构建依赖（--no-build-isolation 不会自动装）
uv pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 numpy

# -----------------------------------------------------------------------------
log "6/6  源码全编译 vLLM（不用预编译产物）"
# -----------------------------------------------------------------------------
cd "$VLLM_DIR"
unset VLLM_USE_PRECOMPILED 2>/dev/null || true
uv pip install -e . --no-build-isolation --torch-backend=cu130 2>&1 | tee "$VLLM_DIR/build-full.log"

log "验证"
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm._C; print('vllm._C OK')"

echo -e "\n\033[1;32m全部完成。日后进开发环境请执行：source $VLLM_DIR/dev-env.sh\033[0m"
