#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-350m
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
export TRAIN_LLAMA='0'
mkdir -p $OUTPUT
deepspeed /home/gq/deeplang/deep-speed-chat/training/step2_reward_model_finetuning/pre_main.py \
   --data_path Dahoas/synthetic-instruct-gptj-pairwise \
   --local_data_files /home/gq/deeplang/deep-speed-chat/datasets/synthetic-instruct-gptj-pairwise \
   --data_split 2,4,4 \
   --model_name_or_path /home/gq/deeplang/deep-speed-chat/training/step2_reward_model_finetuning/output/opt-350m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 1000 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log
