#!/bin/bash
# =============================================================================
# restore-env.sh —— 新容器里一键还原 vLLM 开发栈（无需重新编译）
#
# 用法：  bash /mnt/data/vllm-env/restore-env.sh
#         或  bash /root/vllm/restore-env.sh
# 前提：  新容器须是同一基础镜像（TencentOS 3.2/el8、同 glibc），且路径不变。
# =============================================================================
set -euo pipefail

# ---- taiji 远程持久盘 ----
TAIJI_TOKEN=tamYWh4gueA4Lxpa8FQNcg
MNT=/mnt/data
PERSIST="$MNT/vllm-env"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn] $*\033[0m"; }

# -----------------------------------------------------------------------------
log "0/4  确保 taiji 远程盘已挂载到 $MNT"
# -----------------------------------------------------------------------------
if ! mountpoint -q "$MNT" 2>/dev/null && ! grep -q " $MNT " /proc/mounts; then
  mkdir -p "$MNT"
  taiji_client mount -l qy -tk "$TAIJI_TOKEN" "$MNT"
fi
# 选备份包：优先新版 zstd 包，兼容旧版未压缩 tar
if   [ -f "$PERSIST/devstack.tar.zst" ]; then TARBALL="$PERSIST/devstack.tar.zst"
elif [ -f "$PERSIST/devstack.tar" ];     then TARBALL="$PERSIST/devstack.tar"
else echo "[err] 找不到备份包：$PERSIST/devstack.tar[.zst]"; exit 1; fi
echo "找到备份：$TARBALL（$(du -h "$TARBALL" | cut -f1)）"

# -----------------------------------------------------------------------------
log "1/4  解包到 /（绝对路径原样还原）"
# -----------------------------------------------------------------------------
case "$TARBALL" in
  *.zst) tar --numeric-owner -I 'zstd -d -T0' -xf "$TARBALL" -C / ;;
  *)     tar --numeric-owner -xf "$TARBALL" -C / ;;
esac

# -----------------------------------------------------------------------------
log "2/4  重建动态库缓存（注册 cuda-compat 的 libcuda.so.1）"
# -----------------------------------------------------------------------------
ldconfig
ldconfig -p | grep -m1 'libcuda.so.1' || warn "未检测到 libcuda.so.1，请检查 cuda-compat 是否解出"

# -----------------------------------------------------------------------------
log "3/4  加载开发环境变量"
# -----------------------------------------------------------------------------
# shellcheck disable=SC1091
source /root/vllm/dev-env.sh

# -----------------------------------------------------------------------------
log "4/4  验证"
# -----------------------------------------------------------------------------
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm._C; print('vllm._C OK')"

echo -e "\n\033[1;32m还原完成。当前 shell 已 source dev-env.sh；新开终端请再执行：source /root/vllm/dev-env.sh\033[0m"
