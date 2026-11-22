#!/bin/bash
# 一条复核包装发货：预处理（现在不删）→ 抽帧切块 → Flora 收链 → 映回原时间。
# 抽帧、切块、后处理用普通 python3。只有收链用 Flora 的解释器。
# CUDA_VISIBLE_DEVICES 必须是 3 张卡。画面要转 180° 时加 ROT180=1，编码时才 hflip,vflip。
set -eu
if [ "$#" -ne 4 ]; then
  echo "usage: CUDA_VISIBLE_DEVICES=0,1,2 $0 metadata.json data.mcap workdir outdir" >&2
  exit 2
fi
META=$1
MCAP=$2
WORK=$3
OUT=$4
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLORA_CODE="${FLORA_CODE:-/mnt/cpfs/zhongjiayu0757/flora/code}"
export FLORA_MODEL_CACHE="${FLORA_MODEL_CACHE:-/mnt/cpfs/zhongjiayu0757/flora/models}"
export HF_HOME="${HF_HOME:-/mnt/cpfs/zhongjiayu0757/flora/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="${PY:-python3}"
FLORA_PY="${FLORA_PY:-/mnt/cpfs/zhongjiayu0757/flora/venv/bin/python}"

if [ "${ROT180:-0}" = 1 ]; then
  "$PY" -m fuhe.extract --metadata "$META" --mcap "$MCAP" --workdir "$WORK" --rot180
else
  "$PY" -m fuhe.extract --metadata "$META" --mcap "$MCAP" --workdir "$WORK"
fi
"$PY" -m fuhe.blocks --workdir "$WORK"
"$FLORA_PY" -c 'import torch,sys; n=torch.cuda.device_count(); print("GPUS", n, flush=True); sys.exit(0 if n == 3 else 1)'
"$FLORA_PY" -m fuhe.chain --workdir "$WORK"
"$PY" -m fuhe.postprocess --workdir "$WORK" --out "$OUT"
