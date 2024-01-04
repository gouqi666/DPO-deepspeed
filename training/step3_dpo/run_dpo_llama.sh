#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAIN_LLAMA='1'
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
ACTOR_MODEL_PATH=/mnt/user/gouqi/deep-speed-chat-own/training/step1_supervised_finetuning/outputs/llama2-fullhh-epoch1
CRITIC_MODEL_PATH=/mnt/user/gouqi/deep-speed-chat-own/training/step2_reward_model_finetuning/outputs/llama2-fullhh-lr5e6
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT=/mnt/user/gouqi/deep-speed-chat-own/training/step3_dpo/outputs/llama2-dpo
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-6
deepspeed --master_port 12345 main.py \
   --data_path HelpfulRLHFDataset HarmlessRLHFDataset \
   --local_data_files /mnt/user/gouqi/data/helpful-base /mnt/user/gouqi/data/harmless-base \
   --data_split 0,0,1 0,0,1 \
   --data_output_path /mnt/user/gouqi/deep-speed-chat-own/output/data_files/fullhh \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 2 \
   --log_interval 100 \
   --per_device_eval_batch_size 2 \
   --generation_batch_numbers 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --actor_gradient_checkpointing \
   --beta 0.3 \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 8 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log
# --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \