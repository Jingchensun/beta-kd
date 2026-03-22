#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}
export MASTER_PORT=$((RANDOM%1000+29600))

DISTILL=${1:-"1"}
DISTIL_KL=${2:-"align-kd"}        # distillation loss: align-kd, fkl, rkl, ce, ...
DISTIL_Weighting=${3:-"equal"}          # weighting strategy: equal, ...

LANGUAGE_MODEL="mtgv/MobileLLaMA-1.4B-Chat"
VISION_MODEL="openai/clip-vit-large-patch14-336"

# WandB: set USE_WANDB=True to enable, and set WANDB_PROJECT to your project name
USE_WANDB="True"
WANDB_PROJECT="betakd-pretrain-${DISTIL_KL}-${DISTIL_Weighting}"

# Directory to save distillation adapters (proj_adapter.pt, attn_adapter.pt)
ADAPTER_DIR="adapters/pretrain"

OUTPUT_DIR_PT=outputs-pretrain/${DISTIL_KL}-${DISTIL_Weighting}
mkdir -p ${OUTPUT_DIR_PT}

echo ">>> Start Pre-training: pretrain ..."
deepspeed --master_port $MASTER_PORT mobilevlm/train/train_mem.py \
    --distill ${DISTILL} \
    --distil_type ${DISTIL_KL} \
    --distil_ratio ${DISTIL_Weighting} \
    --save_logit False \
    --task pretrain \
    --deepspeed scripts/deepspeed/zero2.json \
    --model_name_or_path ${LANGUAGE_MODEL} \
    --version plain \
    --data_path dataset/pretrain_data/share-captioner_coco_lcs_sam_1246k_1107.json \
    --image_folder dataset/pretrain_data \
    --vision_tower ${VISION_MODEL} \
    --vision_tower_type clip \
    --mm_projector_type ldpnetv2 \
    --mm_projector_lr 1e-3 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ${OUTPUT_DIR_PT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --save_on_each_node False \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --adapter_dir ${ADAPTER_DIR} \
    --use_wandb ${USE_WANDB} \
    --wandb_project ${WANDB_PROJECT} \
    2>&1 | tee ${OUTPUT_DIR_PT}/log.txt &&
echo "Done."
