#!/bin/bash
# =============================================================================
# backup-env.sh —— 容器销毁前，把整套 vLLM 开发栈打包到 taiji 持久盘 /mnt/data
#
# 用法：  bash /root/vllm/backup-env.sh
# 原理：  CUDA toolkit / gcc-toolset / ccache / rust / venv / vLLM 编译产物
#         全在容器层（overlay），回收即丢。打包到远程盘，下次 restore 即可跳过编译。
# =============================================================================
set -euo pipefail

# ---- taiji 远程持久盘 ----
TAIJI_TOKEN=tamYWh4gueA4Lxpa8FQNcg
MNT=/mnt/data
PERSIST="$MNT/vllm-env"                 # 备份产物存放目录(远程网络盘, 仅 ~65MB/s)
TARBALL="$PERSIST/devstack.tar.zst"     # 最终落地: zstd 压缩包

# ---- 加速参数(瓶颈是远程盘写带宽, 故先本地多核压缩再传单个大文件) ----
LOCAL_TMP="${LOCAL_TMP:-/root/.devstack-build.tar.zst}"  # 本地容器盘(NVMe ~4GB/s)中转
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"           # 3 性价比最高; .so 大二进制再高也压不动多少
ZSTD_THREADS="${ZSTD_THREADS:-0}"       # 0=用满全部核(本机 384)
INCLUDE_CCACHE="${INCLUDE_CCACHE:-1}"   # 默认打包 ccache 缓存(还原后重编可命中, 快数倍); 设 0 可跳过省时间

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn] $*\033[0m"; }

# -----------------------------------------------------------------------------
log "0/3  确保 taiji 远程盘已挂载到 $MNT"
# -----------------------------------------------------------------------------
if ! mountpoint -q "$MNT" 2>/dev/null && ! grep -q " $MNT " /proc/mounts; then
  mkdir -p "$MNT"
  taiji_client mount -l qy -tk "$TAIJI_TOKEN" "$MNT"
fi
# 校验确实可写
mkdir -p "$PERSIST"
touch "$PERSIST/.wtest" && rm -f "$PERSIST/.wtest" || { echo "[err] $PERSIST 不可写"; exit 1; }
echo "持久盘就绪：$PERSIST"

# -----------------------------------------------------------------------------
log "1/3  收集需要打包的“装一次几十分钟”的产物（仅存在的才打包）"
# -----------------------------------------------------------------------------
# 这些路径都在容器层、回收即丢，是重建耗时的根源。
CANDIDATES=(
  /usr/local/cuda-13.0                      # CUDA 13.0 toolkit + compat(真身, 含 compat/libcuda 580.159.04)
  /usr/local/cuda                           # 软链 -> /etc/alternatives/cuda -> cuda-13.0
  /usr/local/cuda-13                         # 软链 -> /etc/alternatives/cuda-13 -> cuda-13.0
  /etc/alternatives/cuda                    # alternatives 软链本体
  /etc/alternatives/cuda-13                 # alternatives 软链本体
  /etc/ld.so.conf.d/00-cuda-compat.conf     # compat 的 ld 注册(置顶 libcuda)
  /etc/ld.so.conf.d/000_cuda.conf           # /usr/local/cuda/targets/.../lib
  /etc/ld.so.conf.d/987_cuda-13.conf        # /usr/local/cuda-13/targets/.../lib
  /etc/ld.so.conf.d/gds-13-0.conf           # GPUDirect Storage 13.0
  /opt/rh/gcc-toolset-12                    # gcc-toolset-12 (C++20)
  /usr/local/bin/ccache                     # ccache 4.x 静态二进制
  /root/.rustup                             # rust 工具链
  /root/.cargo                              # cargo + 镜像配置
  /root/.local                              # uv 等用户级二进制
  /root/vllm                                # 源码 + .venv(torch 2.11.0+cu130) + 编译好的 .so
  /root/ssh_backup.tar                      # 你之前备份的 ssh，顺手带走
)
# ccache 缓存(~909M): 还原后改代码重编时命中可快数倍, 默认带上(INCLUDE_CCACHE=1)。
# 若某次只想快速备份、不打算重编, 可临时设 INCLUDE_CCACHE=0 跳过。
[ "$INCLUDE_CCACHE" = 1 ] && CANDIDATES+=(/root/.cache/ccache)
# 故意不打包：/usr/local/cuda-13.3(仅 compat 子目录, vLLM 用不到)、
#            /usr/local/cuda-12.1(镜像自带旧版, 编译运行都不碰) —— 省空间
PATHS=()
for p in "${CANDIDATES[@]}"; do
  if [ -e "$p" ]; then PATHS+=("$p"); else warn "跳过不存在路径：$p"; fi
done
echo "将打包 ${#PATHS[@]} 项。"

# -----------------------------------------------------------------------------
log "2/4  本地多核压缩(瓶颈是远程盘65MB/s, 故先在本地NVMe压缩)"
# -----------------------------------------------------------------------------
ZTAR=(tar --numeric-owner -I "zstd -T${ZSTD_THREADS} -${ZSTD_LEVEL}")
need=$(du -scb "${PATHS[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
avail=$(df -B1 --output=avail "$(dirname "$LOCAL_TMP")" 2>/dev/null | tail -1)
echo "原始约 $((need/1024/1024/1024))G；本地可用 $((avail/1024/1024/1024))G"

# 本地空间够(留足压缩包余量)才走"本地压→传"；否则回退"直接流式压到远程"
if [ "${avail:-0}" -gt "$need" ]; then
  rm -f "$LOCAL_TMP"
  "${ZTAR[@]}" -cf "$LOCAL_TMP" "${PATHS[@]}"
  csize=$(stat -c%s "$LOCAL_TMP")
  echo "压缩完成：$((csize/1024/1024/1024))G（$(awk "BEGIN{printf \"%.0f\",$csize*100/$need}")% of 原始）"

  log "3/4  传输到远程盘 $TARBALL（单个大文件, 跑满带宽）"
  cp "$LOCAL_TMP" "$TARBALL" &
  cppid=$!
  while kill -0 "$cppid" 2>/dev/null; do
    cur=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
    printf "\r  传输 %d/%d MB (%.0f%%)" $((cur/1024/1024)) $((csize/1024/1024)) "$(awk "BEGIN{printf \"%.0f\",$cur*100/$csize}")"
    sleep 2
  done
  wait "$cppid"; printf "\r  传输完成 %d MB                 \n" $((csize/1024/1024))
  rm -f "$LOCAL_TMP"
else
  warn "本地空间不足以中转，回退为直接流式压缩到远程盘（仍比不压缩快）"
  log "3/4  流式压缩 + 写远程盘 $TARBALL"
  "${ZTAR[@]}" -cf "$TARBALL" "${PATHS[@]}"
fi

# -----------------------------------------------------------------------------
log "4/4  完成"
# -----------------------------------------------------------------------------
sync
ls -lh "$TARBALL"
echo -e "\n\033[1;32m备份完成 -> $TARBALL\n下次新容器里执行：bash /mnt/data/vllm-env/restore-env.sh（见下）\033[0m"

# 顺手把 restore 脚本也复制到持久盘，新容器无需先有本仓库即可还原
cp -f "$(dirname "$0")/restore-env.sh" "$PERSIST/restore-env.sh" 2>/dev/null \
  && echo "已同步 restore-env.sh -> $PERSIST/restore-env.sh"
