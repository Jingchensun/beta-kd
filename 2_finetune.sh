#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}
export MASTER_PORT=$((RANDOM%1000+29600))

DISTILL=${1:-"1"}
DISTIL_KL=${2:-"align-kd"}             # distillation loss: align-kd, fkl, rkl, ce, ...
DISTIL_Weighting=${3:-"equal"}     # weighting strategy: equal/type1, task/type2, instance/type3
# Pretrained checkpoint dir (--model_name_or_path); default matches 1_pretrain.sh OUTPUT_DIR_PT
OUTPUT_DIR_PT=${4:-outputs-pretrain/${DISTIL_KL}-${DISTIL_Weighting}}

# Directory to load distillation adapters from (should match ADAPTER_DIR used in pretrain)
ADAPTER_DIR="adapters/finetune"

# WandB: set USE_WANDB=True to enable, and set WANDB_PROJECT to your project name
USE_WANDB="True"
WANDB_PROJECT="betakd-finetune-${DISTIL_KL}-${DISTIL_Weighting}"

# Quick test: limit training steps to verify checkpoint saving logic.
# Set MAX_STEPS="-1" to disable the limit and run full training.
# MAX_STEPS=${MAX_STEPS:-50}

OUTPUT_DIR_FT=outputs-finetune/${DISTIL_KL}-${DISTIL_Weighting}
mkdir -p ${OUTPUT_DIR_FT}

echo ">>> Start Fine-tuning (distil_type=${DISTIL_KL}, ratio_type=${DISTIL_Weighting}) ..."
deepspeed --master_port $MASTER_PORT mobilevlm/train/train_mem.py \
    --distill ${DISTILL} \
    --distil_type ${DISTIL_KL} \
    --distil_ratio_type ${DISTIL_Weighting} \
    --save_logit False \
    --save_latest False \
    --save_checkpoint_final True \
    --task finetune \
    --deepspeed scripts/deepspeed/zero2.json \
    --model_name_or_path ${OUTPUT_DIR_PT} \
    --version v1 \
    --data_path dataset/finetune_data/MobileVLM_V2_FT_Mix2M-2428k.json \
    --image_folder dataset/finetune_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_type clip \
    --mm_projector_type ldpnetv2 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 20 \
    --save_on_each_node False \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --adapter_dir ${ADAPTER_DIR} \
    --use_wandb ${USE_WANDB} \
    --wandb_project ${WANDB_PROJECT} \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
echo "Done."
