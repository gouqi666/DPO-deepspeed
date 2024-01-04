#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
export TRAIN_LLAMA=1
python ../prompt_eval.py \
    --model_name_or_path_baseline /mnt/data01/shenyan/ckpt/llama_hf/llama-7b \
    --model_name_or_path_finetune /home/gq/deeplang/deep-speed-chat/training/step3_rlhf_finetuning/output/llama-7b/actor
