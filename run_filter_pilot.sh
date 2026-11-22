#!/bin/bash
# 兴遥仓过滤试点。只写清单，不改原片。超过配置时长会在 python 里拒绝。
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
if [ "$#" -lt 1 ]; then
  echo "usage: $0 episodes.json [out_dir]" >&2
  exit 2
fi
OUT=${2:-filter_out}
exec python3 -m siglip_filter run \
  --config "$ROOT/configs/filter_pilot.yaml" \
  --episodes "$1" \
  --out "$OUT"
