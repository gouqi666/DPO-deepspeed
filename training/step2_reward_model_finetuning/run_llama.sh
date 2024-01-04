#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team
OUTPUT=/mnt/user/gouqi/deep-speed-chat-own/training/step2_reward_model_finetuning/outputs/llama2-fullhh
# /mnt/data01/shenyan/ckpt/llama_hf/llama-sft-7b/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ZERO_STAGE=3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT
export TRAIN_MODEL_TYPE='llama'
deepspeed --master_port=12345  main.py \
   --data_path HelpfulRLHFDataset HarmlessRLHFDataset \
   --local_data_files /mnt/user/gouqi/data/helpful-base /mnt/user/gouqi/data/harmless-base \
   --data_output_path /mnt/user/gouqi/deep-speed-chat-own/output/data_files/fullhh \
   --data_split 0,1,0 0,1,0 \
   --num_padding_at_beginning 1 \
   --model_name_or_path /mnt/public/checkpoint/llama_2/llama2/7B-hf \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_prompt_seq_len 512 \
   --max_answer_seq_len 512 \
   --learning_rate 5e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 500 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log