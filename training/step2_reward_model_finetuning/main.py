#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import os
import math
import sys
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model,draw
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--local_data_files',
                        nargs='*',
                        default = []
                        )
    parser.add_argument('--data_split',
                        nargs='*',
                        default=['3,4,3'],
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./output/llama-7b/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=200,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_answer_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK'))
    print(args)
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    torch.distributed.barrier()
    args.global_rank = torch.distributed.get_rank()
    if args.global_rank == 0:
        writer = SummaryWriter(log_dir= os.path.join(args.output_dir,'tensorboard'))
    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    if 'llama' in os.environ.get('TRAIN_MODEL_TYPE',None):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif 'lingowhale' in os.environ.get('TRAIN_MODEL_TYPE',None):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(tokenizer)
    rm_model = create_critic_model(args.model_name_or_path, tokenizer,ds_config)
    # eval
    # rm_model_ = create_critic_model(args.model_name_or_path, tokenizer, None ,
    #                             num_padding_at_beginning=args.num_padding_at_beginning,  rlhf_training=True)

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len, args = args)
    # raw_dataset = get_raw_dataset(args.dataset_name, args.output_path, 1234, -1, local_path=args.local_data_files)
    print_rank_0(f'Train Dataset Length: {len(train_dataset)}')
    print_rank_0(f'Eval Dataset Length: {len(eval_dataset)}')
    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        chosen_scores = []
        reject_scores = []
        for step, batch in enumerate(tqdm(eval_dataloader,desc='eval...')):
            # if step > 10:
            #     break
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            chosen_scores.append(chosen)
            reject_scores.append(rejected)
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            # if step == 99:  # For faster evaluation and debugging
            #     break
        chosen_scores = torch.cat(chosen_scores,dim=0)
        reject_scores = torch.cat(reject_scores,dim=0)
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)

        # try:
        #     acc = get_all_reduce_mean(acc).item()
        #     scores = get_all_reduce_mean(scores).item()
        #     chosen_list = [torch.zeros_like(chosen_scores) for _ in range(torch.distributed.get_world_size())]
        #     reject_list = [torch.zeros_like(reject_scores) for _ in range(torch.distributed.get_world_size())]
        #     torch.distributed.all_gather(chosen_list, chosen_scores)
        #     torch.distributed.all_gather(reject_list, reject_scores)
        # except:
        #     pass
        return scores, acc, chosen_scores, reject_scores


    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)
    if args.global_rank == 0:
        if not os.path.exists(os.path.join(args.output_dir,'img')):
            os.makedirs(os.path.join(args.output_dir,'img'))
        draw(chosen_list, reject_list, output_path=os.path.join(args.output_dir, 'img/score_distribution_init'))
    torch.distributed.barrier()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        chosen_mean_score = []
        rejected_mean_score = []
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            # if args.global_rank == 0:
            #     print(batch['input_ids'][0])
            #     print(batch['input_ids'][4])
            #     print(tokenizer.decode(batch['input_ids'][0].cpu()))
            #     print(tokenizer.decode(batch['input_ids'][4].cpu()))
            #     print(batch['attention_mask'][0])
            #     print(batch['attention_mask'][4])
            #     exit()
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            chosen_mean_score.extend(outputs['chosen_mean_scores'].tolist())
            rejected_mean_score.extend(outputs['rejected_mean_scores'].tolist())
            # assert torch.isnan(loss).sum() == 0
            rm_model.backward(loss)
            if args.global_rank == 0:
                # print('356:',rm_model.v_head.weight.grad)
                # print('357:',rm_model.rwtransformer.layers[0].self_attn.q_proj.weight.grad)
                # print(outputs)
                writer.add_scalar('train_loss',loss.detach().cpu().float().numpy(), step)
            rm_model.step()
            mean_loss += loss.item()
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        print_rank_0(
            f"Epoch {epoch+1} / Chosen_mean_score: {sum(chosen_mean_score)/len(chosen_mean_score)}, Rejected_mean_score:{sum(rejected_mean_score)/len(rejected_mean_score)}",
            args.global_rank)
        total_score = chosen_mean_score + rejected_mean_score
        print_rank_0(
            f"Epoch {epoch+1} / Total_score_mean: {sum(total_score)/len(total_score)}, variance:{np.var(total_score)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, eval_dataloader)
        total_score = chosen_list.tolist() + reject_list.tolist()
        print_rank_0(
            f"Epoch {epoch+1} / Total_eval_score_mean: {sum(total_score)/len(total_score)}, variance:{np.var(total_score)}",
            args.global_rank)

        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)

        draw(chosen_list,reject_list,output_path=os.path.join(args.output_dir,f'img/score_distribution_epoch{epoch}'))
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
