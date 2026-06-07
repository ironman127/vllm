#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export CCACHE_DIR=/root/.cache/ccache
export MAX_JOBS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source /root/vllm/.venv/bin/activate
echo "vLLM dev env ready. libcuda from compat, torch backend cu129."