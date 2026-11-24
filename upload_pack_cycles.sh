#!/bin/bash
# 把切好的一单传到私人数据集。HF_TOKEN 从环境读，不写进仓库。
# pai-5000 连不上 huggingface.co，在能上网的机器上跑。
#   HF_TOKEN=... ./upload_pack_cycles.sh wbxlala/fuhe-pack-cycles /dev/shm/fuhe_hf_cycles
set -eu
if [ "$#" -ne 2 ]; then
  echo "usage: HF_TOKEN=... $0 namespace/dataset workdir" >&2
  exit 2
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is empty" >&2
  exit 2
fi
REPO=$1
DIR=$2
hf upload-large-folder "$REPO" "$DIR" --type dataset
