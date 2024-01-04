#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=/mnt/user/gouqi/deep-speed-chat/training/step1_supervised_finetuning/outputs/llama2-fullhh-test
ZERO_STAGE=3
TRAIN_LLAMA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAIN_MODEL_TYPE='llama'
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
#    --sft_only_data_path /mnt/user/gouqi/deep-speed-chat/datasets/alpaca-gpt4 \
deepspeed main.py \
   --data_path HelpfulRLHFDataset HarmlessRLHFDataset \
   --local_data_files /mnt/user/gouqi/deep-speed-chat/datasets/helpful-base /mnt/user/gouqi/deep-speed-chat/datasets/harmless-base \
   --data_split 1,0,0 1,0,0 \
   --data_output_path /mnt/user/gouqi/deep-speed-chat/output/data_files/fullhh-test \
   --model_name_or_path /mnt/public/checkpoint/llama_2/llama2/7B-hf \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --max_prompt_seq_len 512 \
   --max_answer_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 500 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log