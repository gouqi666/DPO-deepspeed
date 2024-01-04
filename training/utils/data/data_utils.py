# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
from itertools import chain
from . import raw_datasets


def get_raw_dataset(dataset_name, output_path, seed, local_rank, local_path = None,train_phase=None):
    if dataset_name == "DPODataset":
        return raw_datasets.DPODataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "TLDRDataset":
        return raw_datasets.TLDRDataset(output_path, seed, local_rank, local_path,train_phase)
    elif dataset_name == "TLDRPPODataset":
        return raw_datasets.TLDRPPODataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "TLDRDPODataset":
        return raw_datasets.TLDRDPODataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "PPODataset":
        return raw_datasets.PPODataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "PRORLHFDataset":
        return raw_datasets.PRORLHFDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "HelpfulRLHFDataset":
        return raw_datasets.HelpfulRLHFDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "HelpfulUserDataset":
        return raw_datasets.HelpfulUserDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "HarmlessRLHFDataset":
        return raw_datasets.HarmlessRLHFDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "HarmlessUserDataset":
        return raw_datasets.HarmlessUserDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "ShareGPTZHDataset":
        return raw_datasets.ShareGPTZHDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "AlpacaGPT4Dataset" or 'alpaca-gpt4' in dataset_name:
        return raw_datasets.AlpacaGPT4Dataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "MossSftDataset":
        return raw_datasets.MossSftDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "single_turn_rlhf":
        return raw_datasets.SingleTurnRLHFDataset(output_path, seed, local_rank, local_path)
    elif dataset_name == "Dahoas/rm-static":
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "Dahoas/full-hh-rlhf":
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank,local_path)
    elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank,local_path)
    elif dataset_name == "yitingxie/rlhf-reward-datasets":
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "openai/webgpt_comparisons":
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "stanfordnlp/SHP":
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "wangrui6/Zhihu-KOL":
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Cohere/miracl-zh-queries-22-12":
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "Hello-SimpleAI/HC3-Chinese":
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "mkqa-Chinese":
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank)
    elif dataset_name == "mkqa-Japanese":
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank)
    elif dataset_name == "Cohere/miracl-ja-queries-22-12":
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "lmqg/qg_jaquad":
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == "lmqg/qag_jaquad":
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.isfile(index_file_name) and local_rank <= 0:
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"],
                "prompt_input_ids": self.prompt_dataset[idx]['input_ids'],
                "prompt_attention_mask": self.prompt_dataset[idx]['attention_mask']
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
        elif self.train_phase == 4: # dpo
            return self.chosen_dataset[idx]['input_ids'],self.chosen_dataset[idx]['attention_mask'],self.chosen_dataset[idx]['labels'], \
                    self.reject_dataset[idx]['input_ids'],self.reject_dataset[idx]['attention_mask'],self.reject_dataset[idx]['labels'], \
                   self.prompt_dataset[idx]['input_ids'],self.prompt_dataset[idx]['attention_mask']

def pad_input(input,max_length,pad_token_id,):
    input_len = len(input)
    pad_len = max_length - input_len
    return [pad_token_id] * pad_len + input
def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len,args):
    human_prompt = "\n\nHuman: "
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            chosen = raw_dataset.get_chosen(tmp_data)
            prompt_input_ids = tokenizer.encode(prompt)
            prompt_token = {}
            chosen_token = {}

            if len(prompt_input_ids) > args.max_prompt_seq_len:
                prompt_input_ids = prompt_input_ids[:args.max_prompt_seq_len]
            # try:
            #     while len(prompt_input_ids) > args.max_prompt_seq_len:
            #         prompt = human_prompt + human_prompt.join(prompt.split(human_prompt)[2:])
            #         prompt_input_ids = tokenizer.encode(prompt)
            # except Exception as e:
            # label_input_ids = tokenizer.encode(chosen,truncation=True,max_length=args.max_answer_seq_len)

            label_input_ids = tokenizer.encode(chosen)
            if len(label_input_ids) > args.max_answer_seq_len:
                label_input_ids = label_input_ids[:args.max_answer_seq_len]


            context_length = len(prompt_input_ids)
            input_ids = prompt_input_ids + label_input_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + label_input_ids + [tokenizer.eos_token_id]

            # padding
            pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(input_ids)
            input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
            labels = [tokenizer.pad_token_id] * pad_len + labels

            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            chosen_token["input_ids"] = torch.tensor(input_ids)
            chosen_token["labels"] = torch.tensor(labels)
            chosen_token["attention_mask"] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in input_ids])
            chosen_dataset.append(chosen_token)
            # padding
            pad_len = args.max_prompt_seq_len - len(prompt_input_ids)
            prompt_input_ids = [tokenizer.pad_token_id] * pad_len + prompt_input_ids
            prompt_token['input_ids'] = torch.tensor(prompt_input_ids)
            prompt_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in prompt_input_ids])
            prompt_dataset.append(prompt_token)
    elif train_phase == 2:
        print('176:training_phrase_2')
        for i, tmp_data in enumerate(current_dataset):
            p_length = 0 # abandoned
            prompt = raw_dataset.get_prompt(tmp_data)
            chosen = raw_dataset.get_chosen(tmp_data)
            rejected = raw_dataset.get_rejected(tmp_data)
            prompt_input_ids = tokenizer.encode(prompt)
            chosen_token = {}
            rejected_token = {}
            if len(prompt_input_ids) > args.max_prompt_seq_len:
                # continue
                prompt_input_ids = prompt_input_ids[:args.max_prompt_seq_len]

            # try:
            #     while len(prompt_input_ids) > args.max_prompt_seq_len:
            #         prompt = human_prompt + human_prompt.join(prompt.split(human_prompt)[2:])
            #         prompt_input_ids = tokenizer.encode(prompt)
            # except Exception as e:
            #     prompt_input_ids = prompt_input_ids[:args.max_prompt_seq_len]

            chosen_input_ids = tokenizer.encode(chosen)
            if len(chosen_input_ids) > args.max_answer_seq_len:
                chosen_input_ids = chosen_input_ids[:args.max_answer_seq_len]
            chosen_token['input_ids'] = prompt_input_ids + chosen_input_ids + [tokenizer.eos_token_id]

            pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(chosen_token['input_ids'])
            chosen_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + chosen_token['input_ids']

            rejected_input_ids = tokenizer.encode(rejected)
            if len(rejected_input_ids) > args.max_answer_seq_len:
                rejected_input_ids = rejected_input_ids[:args.max_answer_seq_len]

            rejected_token['input_ids'] = prompt_input_ids + rejected_input_ids + [tokenizer.eos_token_id]
            pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(rejected_token['input_ids'])
            rejected_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + rejected_token['input_ids']

            assert  len(chosen_token["input_ids"])  == len(rejected_token['input_ids'])
            ####

            chosen_token["input_ids"] = torch.tensor(chosen_token["input_ids"])
            rejected_token["input_ids"] = torch.tensor(rejected_token["input_ids"])
            chosen_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in chosen_token['input_ids']])
            rejected_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in rejected_token['input_ids']])
            chosen_dataset.append(chosen_token)
            reject_dataset.append(rejected_token)
    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            prompt_input_ids = tokenizer.encode(prompt)
            if 'TLDR' in args.data_path[0]:
                if len(prompt_input_ids) > args.max_prompt_seq_len:
                    prompt_input_ids = prompt_input_ids[-args.max_prompt_seq_len:]
            else: # hh-rlhf
                try:
                    while len(prompt_input_ids) > args.max_prompt_seq_len:
                        prompt = human_prompt + human_prompt.join(prompt.split(human_prompt)[2:])
                        prompt_input_ids = tokenizer.encode(prompt)
                except Exception as e:
                    prompt_input_ids = prompt_input_ids[-args.max_prompt_seq_len:]

            pad_len = args.max_prompt_seq_len - len(prompt_input_ids)
            prompt_input_ids = [tokenizer.pad_token_id] * pad_len + prompt_input_ids
            prompt_token = {}
            prompt_token["input_ids"] = torch.LongTensor(prompt_input_ids)
            prompt_token["attention_mask"] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in prompt_token["input_ids"]])
            prompt_dataset.append(prompt_token)
    elif train_phase == 4:
        print('176:training_phrase_4')
        for i, tmp_data in enumerate(current_dataset):
            p_length = 0 # abandoned
            prompt = raw_dataset.get_prompt(tmp_data)
            chosen = raw_dataset.get_chosen(tmp_data)
            rejected = raw_dataset.get_rejected(tmp_data)
            if rejected is None:
                rejected = 'None'
            prompt_input_ids = tokenizer.encode(prompt)
            if 'TLDR' in args.data_path[0]:
                if len(prompt_input_ids) > args.max_prompt_seq_len:
                    prompt_input_ids = prompt_input_ids[-args.max_prompt_seq_len:]
            else: # hh-rlhf
                try:
                    while len(prompt_input_ids) > args.max_prompt_seq_len:
                        prompt = human_prompt + human_prompt.join(prompt.split(human_prompt)[2:])
                        prompt_input_ids = tokenizer.encode(prompt)
                except Exception as e:
                    prompt_input_ids = prompt_input_ids[-args.max_prompt_seq_len:]

            context_length = len(prompt_input_ids)
            chosen_token = {}
            chosen_input_ids = tokenizer.encode(chosen)
            if len(chosen_input_ids) > args.max_answer_seq_len:
                chosen_input_ids = chosen_input_ids[:args.max_answer_seq_len]
            chosen_token['input_ids'] = prompt_input_ids + chosen_input_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + chosen_input_ids + [tokenizer.eos_token_id]

            pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(chosen_token['input_ids'])
            chosen_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + chosen_token['input_ids']
            chosen_token['labels'] = [tokenizer.pad_token_id] * pad_len + labels
            chosen_token['labels'] = [(l if l != tokenizer.pad_token_id else -100) for l in chosen_token['labels']]

            rejected_token = {}
            rejected_input_ids = tokenizer.encode(rejected)
            if len(rejected_input_ids) > args.max_answer_seq_len:
                rejected_input_ids = rejected_input_ids[:args.max_answer_seq_len]

            rejected_token['input_ids'] = prompt_input_ids + rejected_input_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + rejected_input_ids + [tokenizer.eos_token_id]
            pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(rejected_token['input_ids'])
            rejected_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + rejected_token['input_ids']
            rejected_token['labels'] = [tokenizer.pad_token_id] * pad_len + labels
            rejected_token['labels'] = [(l if l != tokenizer.pad_token_id else -100) for l in rejected_token['labels']]

            chosen_token["input_ids"] = torch.tensor(chosen_token["input_ids"])
            rejected_token["input_ids"] = torch.tensor(rejected_token["input_ids"])
            chosen_token["labels"] = torch.tensor(chosen_token["labels"])
            rejected_token["labels"] = torch.tensor(rejected_token["labels"])
            chosen_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in chosen_token['input_ids']])
            rejected_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in rejected_token['input_ids']])


            prompt_token = {}
            pad_len = args.max_prompt_seq_len - len(prompt_input_ids)
            prompt_input_ids = [tokenizer.pad_token_id] * pad_len + prompt_input_ids
            prompt_token['input_ids'] = torch.tensor(prompt_input_ids)
            prompt_token['attention_mask'] = torch.tensor([(1 if x != tokenizer.pad_token_id else 0) for x in prompt_input_ids])


            chosen_dataset.append(chosen_token)
            reject_dataset.append(rejected_token)
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len,local_path, args = None):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank, local_path=local_path,train_phase=train_phase)
    train_dataset = raw_dataset.get_train_data()
    # 获取每个stage的数据index
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1 if train_phase <= 3 else 2,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len,args)

    print('train Length-342:',len(train_dataset))

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split,
                                             train_phase - 1 if train_phase <= 3 else 2,
                                             len(eval_dataset))


    eval_dataset = Subset(eval_dataset, eval_index)

    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len,args)
    print('eval Length-358:',len(eval_dataset))
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          args = None):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    # TODO: add the tokenizer name
    if hasattr(tokenizer,'init_kwargs'):
        tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    else:
        tokenizer_name = 'llamatokenizer'
    fname = f"{fname}_split{','.join(data_split)}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_promptseqlen{args.max_prompt_seq_len}_ansseqlen{args.max_answer_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = str(hash(fname))  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    # Skip creating cache if we found it on all the nodes.
    if False: # buf_create_cache.item() == 0: #
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split[0], output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len,args.local_data_files[0], args = args)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path,split,local_path in zip(data_path,data_split,args.local_data_files):
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len,local_path,args = args)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
                # print('329:',len(train_dataset))
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    local_path=sft_path,
                    args=args
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # if local_rank <= 0:
            # torch.save(train_dataset, train_fname)
            # torch.save(eval_dataset, eval_fname)
        return train_dataset, eval_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.stack([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.stack([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch

class DataCollatorDPO:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.stack([f[0]
                                        for f in data] + [f[3] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.stack([f[1] for f in data] +
                                            [f[4] for f in data],
                                            dim=0)
        batch["labels"] = torch.stack([f[2] for f in data] +
                                            [f[5] for f in data],
                                            dim=0)
        batch["prompt_input_ids"] = torch.stack([f[6] for f in data])
        batch["prompt_attention_mask"] = torch.stack([f[7] for f in data])
        return batch

class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size, tokenizer):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        # pad_token_id = self.tokenizer.pad_token_id
        prompt = [f[0] for f in data]
        prompt_mask = [f[1] for f in data]
        # prompt = pad_sequence([f[0] for f in data],
        #                       padding_value=pad_token_id,
        #                       batch_first=True)
        # prompt_mask = pad_sequence([f[1] for f in data],
        #                            padding_value=0,
        #                            batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        batch["prompt"] = torch.stack(prompt)
        batch["prompt_att_mask"] = torch.stack(prompt_mask)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size  # 1
        self.small_batch_size = small_batch_size # train_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
