#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH=${WORK_DIR}
cd ${WORK_DIR}

CHECKPOINT=$1
# resolve to absolute path if it's a local directory
[ -d "$CHECKPOINT" ] && CHECKPOINT=$(realpath "$CHECKPOINT")
OUTPUT_DIR=$2
DATASETS=${3:-"mme gqa textvqa pope mmbench sqa"}

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: bash scripts/benchmark.sh <checkpoint_or_hf_id> <output_dir> [datasets]"
  exit 1
fi

trap 'echo "Interrupted, killing all subprocesses..."; kill 0; exit 1' INT TERM

mkdir -p ${OUTPUT_DIR}
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Datasets:   ${DATASETS}"

CONV_MODE=v1

for dataset in $DATASETS; do
  case $dataset in
    mme)     LOADER=mobilevlm.eval.model_vqa_loader;   DATA=${WORK_DIR}/dataset/benchmark_data/mme;     SPLIT=llava_mme ;;
    gqa)     LOADER=mobilevlm.eval.model_vqa_loader;   DATA=${WORK_DIR}/dataset/benchmark_data/gqa;     SPLIT=llava_gqa_testdev_balanced ;;
    textvqa) LOADER=mobilevlm.eval.model_vqa_loader;   DATA=${WORK_DIR}/dataset/benchmark_data/textvqa; SPLIT=llava_textvqa_val_v051_ocr ;;
    pope)    LOADER=mobilevlm.eval.model_vqa_loader;   DATA=${WORK_DIR}/dataset/benchmark_data/pope;    SPLIT=llava_pope_test ;;
    mmbench) LOADER=mobilevlm.eval.model_vqa_mmbench;  DATA=${WORK_DIR}/dataset/benchmark_data/mmbench; SPLIT=mmbench_dev_en_20231003 ;;
    sqa)     LOADER=mobilevlm.eval.model_vqa_science;  DATA=${WORK_DIR}/dataset/benchmark_data/sqa;     SPLIT=llava_test_CQM-A ;;
    *) echo "Unknown dataset: $dataset, skipping"; continue ;;
  esac

  echo "Running ${dataset}..."
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${dataset}.sh \
    ${LOADER} ${CHECKPOINT} ${CONV_MODE} ${SPLIT} ${DATA} ${OUTPUT_DIR}/${dataset}
done
