#!/bin/bash
# vLLM 开发环境（535 宿主驱动 + cuda-compat-13-0 前向兼容 + CUDA 13.0 全编译栈）
# 用法：source /root/vllm/dev-env.sh   （首次安装见 setup-dev-env.sh / INSTALL-vLLM-CUDA13.md）

# 1) 前向兼容：让 cuda-compat-13-0 的新版 libcuda.so（580.159.04）优先于宿主 535。
#    compat 装于 /usr/local/cuda-13.0/compat，已通过 /etc/ld.so.conf.d/00-cuda-compat.conf
#    全局注册，这里再显式置顶做双保险。
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH

# 2) 编译缓存与并行度（增量编译关键）
#    ccache 按源文件内容哈希缓存编译产物，第二遍编译命中后可快数倍~数十倍。
#    vLLM setup.py 检测到 ccache 会自动注入 -DCMAKE_*_COMPILER_LAUNCHER=ccache。
export CCACHE_DIR=/root/.cache/ccache
export CCACHE_MAXSIZE=50G
# 【关键】必须 ccache>=4：3.x(el8 dnf 默认 3.7.7) 不认 CMake 给 nvcc 的“-x cu”，
# 会判 "Unsupported language: cu" 而完全不缓存 CUDA → 改一行仍全量重编。
if command -v ccache >/dev/null 2>&1; then
  _ccver=$(ccache --version | sed -n '1s/.*version \([0-9]\+\).*/\1/p')
  [ "${_ccver:-0}" -lt 4 ] 2>/dev/null && \
    echo "[warn] ccache $(ccache --version|head -1|grep -o '[0-9.]*') 过旧，不缓存 CUDA(-x cu)！请装 4.x：bash setup-dev-env.sh"
  unset _ccver
else
  echo "[warn] 未装 ccache，编译无缓存会很慢，请运行：bash setup-dev-env.sh"
fi
# MAX_JOBS 总并行；NVCC_THREADS 让每个 nvcc 内部多线程，并行单元数≈MAX_JOBS/NVCC_THREADS，
# 既跑满多核又避免过多 nvcc 同时吃满内存。
export MAX_JOBS=360
export NVCC_THREADS=4

# 3) vLLM 多进程方式（前向兼容/调试下更稳）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 4) cargo 稀疏索引协议（配合国内镜像加速 rust 依赖拉取）
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# 5) 让 uv / rust 工具链进入 PATH
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# 6) 激活 venv
source /root/vllm/.venv/bin/activate

# 7) 启用 gcc-toolset-12（满足 vLLM C++20 要求；系统默认 gcc 8.5 不够）
if [ -f /opt/rh/gcc-toolset-12/enable ]; then
  source /opt/rh/gcc-toolset-12/enable
  export CC=$(command -v gcc)
  export CXX=$(command -v g++)
  export CMAKE_CUDA_HOST_COMPILER=$CXX
fi

# 8) CUDA 13.0 完整 toolkit（系统装在 /usr/local/cuda-13.0；
#    不设的话编译会误用系统 /usr/local/cuda 的 12.1，导致 nvcc 版本错配）
export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

echo "vLLM dev env ready. libcuda from cuda-compat-13-0 (580.159.04), nvcc=$CUDA_HOME, torch backend cu130."
