#!/bin/bash
# 复核包装发货全套：列清单 → 切掉无效片段 → Flora 标注。后处理在 kept_flora 里，每段收完链就跑。
# 第 i 片用卡 BASE+i。只跑一片、并且已经设了 CUDA_VISIBLE_DEVICES 时，沿用那张卡。
#   CUT_SHARDS=4 CUT_GPU_BASE=4 FLORA_SHARDS=8 ./run_fuhe_kept.sh
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -d /mnt/cpfs/wbx/code/fuhe_pydeps ]; then
  export PYTHONPATH="$PYTHONPATH:/mnt/cpfs/wbx/code/fuhe_pydeps"
fi
export FLORA_CODE="${FLORA_CODE:-/mnt/cpfs/zhongjiayu0757/flora/code}"
export FLORA_MODEL_CACHE="${FLORA_MODEL_CACHE:-/mnt/cpfs/zhongjiayu0757/flora/models}"
export HF_HOME="${HF_HOME:-/mnt/cpfs/zhongjiayu0757/flora/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python3}"
export PY
export FLORA_PY="${FLORA_PY:-/mnt/cpfs/zhongjiayu0757/flora/venv312/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/filter_pilot.yaml}"
WORK="${WORK:-/dev/shm/fuhe_pipeline}"
EPISODES="${EPISODES:-$WORK/episodes.json}"
CUT_SHARDS="${CUT_SHARDS:-1}"
FLORA_SHARDS="${FLORA_SHARDS:-1}"
CUT_GPU_BASE="${CUT_GPU_BASE:-0}"
FLORA_GPU_BASE="${FLORA_GPU_BASE:-0}"

assign_gpu() {
  local index=$1
  local count=$2
  local base=$3
  if [ "$count" -eq 1 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '%s\n' "$((base + index))"
  fi
}

wait_all() {
  local fail=0
  local pid
  for pid in "$@"; do
    wait "$pid" || fail=1
  done
  return "$fail"
}

mkdir -p "$WORK"
if [ ! -s "$EPISODES" ]; then
  "$PY" -m fuhe.list_episodes --out "$EPISODES"
fi

cut_pids=()
for i in $(seq 0 $((CUT_SHARDS - 1))); do
  mkdir -p "$WORK/cut$i"
  CUDA_VISIBLE_DEVICES="$(assign_gpu "$i" "$CUT_SHARDS" "$CUT_GPU_BASE")" \
    "$PY" -m siglip_filter.cut_keep \
      --config "$CONFIG" \
      --episodes "$EPISODES" \
      --shard "$i" --shards "$CUT_SHARDS" \
      --work "$WORK/cut$i" &
  cut_pids+=("$!")
done
wait_all "${cut_pids[@]}"

flora_pids=()
for i in $(seq 0 $((FLORA_SHARDS - 1))); do
  mkdir -p "$WORK/flora$i"
  CUDA_VISIBLE_DEVICES="$(assign_gpu "$i" "$FLORA_SHARDS" "$FLORA_GPU_BASE")" \
    "$PY" -m fuhe.kept_flora \
      --work "$WORK/flora$i" \
      --shard "$i" --shards "$FLORA_SHARDS" \
      --drain &
  flora_pids+=("$!")
done
wait_all "${flora_pids[@]}"
echo "FUHE_KEPT_DONE"
