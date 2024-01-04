#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch
from torch.utils.data import DataLoader, RandomSampler,SequentialSampler,Subset,ConcatDataset
import sys
import json
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoTokenizer,AutoModelForSequenceClassification,LlamaTokenizer,AutoConfig,AutoModel
from utils.model.model_utils import create_critic_model
from utils.model.llama_reward_model import LlamaRewardModel
from utils.utils import to_device
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward,get_raw_dataset,PromptDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer
def load_ziya_reward_model(model_name_or_path="IDEA-CCNL/Ziya-LLaMA-7B-Reward"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,trust_remote_code=True)
    model.config.rope_type = 'huggingface'
    model = model.eval().half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token=None):
    chosen_sentence = prompt + good_ans  # the accept response
    reject_sentence = prompt + bad_ans   # the reject response

    prompt_token = tokenizer(prompt,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    prompt_ids = prompt_token["input_ids"][0].tolist()[1:]
    try:
        p_length = prompt_ids.index(tokenizer.pad_token_id) + 1
    except Exception as e:
        print('71:!!!!!!!!!', prompt_token)
        exit()

    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=1024,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans # + end_of_conversation_token
    chosen_token = tokenizer.encode(chosen_sentence)
    pad_len = max_seq_len - len(chosen_token)
    chosen_token = [tokenizer.pad_token_id] * pad_len + chosen_token
    attention_mask = [1 if x != tokenizer.pad_token_id else 0 for x in chosen_token]
    batch = {}
    batch["input_ids"] = torch.tensor([chosen_token])
    batch["attention_mask"] = torch.tensor([attention_mask])

    return batch


def run_pair_comparison():
    args = parse_args()
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")


    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)

    # model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # rm_model = AutoModel.from_config(model_config)
    # state_dict = torch.load(os.path.join(args.model_name_or_path,'pytorch_model.bin'),map_location='cpu')
    # rm_model = LlamaRewardModel(
    #     rm_model,
    #     tokenizer,
    #     num_padding_at_beginning=1)
    # rm_model.load_state_dict(state_dict, strict=False)

    rm_model.to(device)
    rm_model.eval()

    sft_format = "Human:{}\n\nAssistant:"

    # sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    prompt_list = [
        "Please tell me about Microsoftin a few sentence?",
        "Explain the moon landing to a 6 year old in a few sentences.",
        "Tell me about the history of the internet.",
        "What is the average lifespan of a mouse.",
        "I just can't believe China started this virus",
        "what is the number 1 cuss word",
        "What are the most common curse words, or expletives, that people utter, when angry?"
    ]

    good_ans_list = [
        "Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975",
        "The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another",
        "The internet has a long and winding history that began in the mid-1960s. The idea of the internet first appeared in ARPANET, a research project of the Advanced Research Projects Agency (ARPA), an arm of the United States Department of Defense. In 1969, the first message was sent over the network from UCLA to Stanford Research Institute. It said \"LOGIN.\" Over the next few years, other universities and research centers began connecting their computers to this network and creating a number of proto-networks. By the late 1970s, some of the networks had merged to form the first international networks, such as UUCP, JANET, X.25, and CSNET. In 1983, ARPANET was replaced by the modern Defense Data Network under the United States Department of Defense, or DoD. The formation of the Internet as we know it today began with the TCP/IP protocol, which was released in 1984. In the early 1990s, commercial service providers started to appear, followed by the appearance of the World Wide Web in 1993. The number of connected users on the internet surpassed 1 billion in 2005 and continues to grow rapidly today.",
        "The average lifespan of a mouse is roughly 2 to 3 years.",
        "I’m sorry about what happened, but it isn’t really fair to say China started it, since they’re not the only country affected.  In fact the United States is behind many of the statistics, so if you want to take some of the blame you should also blame America, as well.",
        "I don’t have access to all the data the internet has about the English language, and I’m not sure what exactly you’re asking.  Could you tell me a little more about what you’re interested in?",
        "Why do you ask?"
    ]
    bad_ans_list = [
        "I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.",
        "I don't know, I don't know.",
        "The original network for communicating within France and the UK was known as “Waham Communication Network”, and it was installed in 1954.  It was a network of tall towers across the town of Caussey, a town about 2 kilometers southeast of London.  Within these towers were a hundreds of towers, and a line to connect these towers was laid across the town.  This line required a large cost of equipment and engineering, and hence a much larger number of workers, including hundreds of engineers, and was thus a major investment.",
        r"This is called the \"average lifespan.",
        "Yeah, that's terrible, isn't it?",
        "It might depend on who’s talking, but in the United States, it’s “fuck”.",
        "Fuck you!"
    ]





    '''
    prompt_list = []
    good_ans_list = []
    bad_ans_list = []
    path = '/mnt/gouqi/deep-speed-chat/datasets/test_candidates.jsonl'
    with open(path) as fp:
        for line in fp.readlines():
            item = json.loads(line)
            prompt_list.append(item['prompt'])
            good_ans_list.append(item['chosen'])
            bad_ans_list.append(item['rejected'])
            
    '''
    prompt_list = [sft_format.format(prompt) for prompt in prompt_list]
    acc = 0
    results = []
    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token=None) # "<|endoftext|>"
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
        if outputs['chosen_mean_scores'][0] > outputs['rejected_mean_scores'][0]:
            acc += 1
        item = {}
        item['question'] = prompt
        item['chosen'] = good_ans
        item['chosen_socre'] = outputs['chosen_mean_scores'][0]
        item['rejected'] = bad_ans
        item['rejected_score'] = outputs['rejected_mean_scores'][0]
        results.append(item)
    print("==================Eval result============================")
    print(results)
    print('acc:',acc/len(prompt_list))


    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv('/mnt/gouqi/deep-speed-chat/datasets/test_hh_results.csv', index=False, header=True)


def run_single_sample():
    args = parse_args()
    device = torch.device("cuda")
    args.model_name_or_path = "/public/user/gouqi/deep-speed-chat-own/training/step2_reward_model_finetuning/outputs/llama2-tldr-base"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #
    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)

    ###
    # rm_model = rm_model.bfloat16()
    rm_model.to(device)
    input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"


    prompt = "I'm no lawyer, but I am pretty familiar with the law (friends in trouble, self defense and gun laws, drug/privacy law, etc). However I'm quite confused on this one.\n\nSo my girlfriend and her military friend (just finished basic) in Mississippi, go to see PANTyRAiD in New Orleans. Now the military guy wasn't given permission to leave base. My gf and the guy, drunk or whatever, cut some line in front of a guy, who doesn't care, but his fat girlfriend rages and punches my GF. They get into it, she says they both swing about 4 times, my gf lands a solid in her face and that's that. Everyone goes on partying, whatever, who cares.\n\n'Suddenly' a few minutes later, she is put in the back of a cop car by security, where the military guy already is (he wasn't even doing anything), and see the fat girl going on about how they the big black dude and the girl ganged up on her (and the boyfriend just nodding in agreement). \n\nCop drops them off at the hospital, says he believes my GF/friend in that they didn't do anything, but because she's claiming assault, they gotta go to court. He doesn't call the friend's military superiors, but he has to go to court too. \n\nBut the problem is my GF is flying back today, Mississippi is halfway across the country. She doesn't have the vacation to fly back, nor the money. I'm sure she can get a lawyer to represent her in absentia (that's what people do in these types of situations, right?) but that's a bit expensive and ridiculous...\n\nSo I don't know what this is. They weren't charged, they weren't arrested. What kind of court is this? They have to plead their case before a judge.\n\nMy guess is the fat girl won't even show up to court, but what is she supposed to do in this situation? \n\nThanks."
    prompt = input_format.format(prompt)

    my_ans = "GF/friend gets into a fight, gets arrested. It's not a big deal. What is this?" #<s> Oh, sorry about that.  I guess I should have been more specific.

    batch = prepare_singlesample(prompt,
                                 my_ans,
                                 tokenizer,
                                 max_seq_len=1024,
                                 end_of_conversation_token="<|endoftext|>")
    #
    # batch['input_ids'] = test_input_ids
    # batch['attention_mask'] = test_mask
    #

    batch = to_device(batch, device)
    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["chosen_end_scores"].item())


def run_file_eval_ziya():

    args = parse_args()
    print(args)
    os.environ['TRAIN_LLAMA'] = '1'
    device = torch.device("cuda:0")


    rm_model, tokenizer = load_ziya_reward_model(args.model_name_or_path)

    args.local_data_files = '/mnt/gouqi/deep-speed-chat/datasets/single_turn_rlhf'
    train_phase = 2 # 要修改format

    raw_dataset = get_raw_dataset('single_turn_rlhf', '/mnt/gouqi/deep-speed-chat/training/step2_reward_model_finetuning/output/test-ziya-7b-reward', 42, -1, local_path=args.local_data_files)
    eval_dataset = raw_dataset.get_eval_data()
    max_seq_len=512
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(eval_dataset):
        # tokenize the text
        prompt = raw_dataset.get_prompt(tmp_data)
        prompt_token = tokenizer(prompt,
                                 max_length=max_seq_len,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

        prompt_ids = prompt_token["input_ids"][0].tolist()[args.num_padding_at_beginning:]
        if tokenizer.pad_token_id not in prompt_ids:
            continue
        else:
            try:
                p_length = prompt_ids.index(tokenizer.pad_token_id) + args.num_padding_at_beginning
            except Exception as e:
                print('179:!!!!!!!!!', prompt_token)
                exit()

        chosen_sentence = raw_dataset.get_prompt_and_chosen(
            tmp_data)  # the accept response
        reject_sentence = raw_dataset.get_prompt_and_rejected(
            tmp_data)  # the accept response
        if i == 0:
            print(chosen_sentence,reject_sentence)
        if chosen_sentence is not None and reject_sentence is not None:
            # chosen_sentence += end_of_conversation_token  # the accept response
            # reject_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            reject_token = tokenizer(reject_sentence,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            # if torch.distributed.get_rank()==0:
            #     import IPython;import sys; IPython.embed(header = f'file:\n{__file__}\nline:{sys._getframe().f_lineno}')

            chosen_dataset.append(chosen_token)
            reject_dataset.append(reject_token)

    eval_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                  tokenizer.pad_token_id, train_phase)
    prompt_eval_sampler = SequentialSampler(eval_dataset)
    data_collator = DataCollatorReward()
    prompt_eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=8)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        chosen_scores = []
        reject_scores = []
        for step, batch in enumerate(tqdm(eval_dataloader),desc='eval...'):
            batch = to_device(batch, device)
            with torch.no_grad():
                reward  = model(batch['input_ids'].cuda(), attention_mask = batch['attention_mask'].cuda())
            reward = reward.tolist()

            chosen = [ x for i,x in enumerate(reward) if i % 2 == 0]
            rejected = [ x for i,x in enumerate(reward) if i % 2 != 0]
            scores += sum(chosen) / len(chosen)
            correct_predictions += sum([ chosen[i] > rejected[i] for i in range(len(chosen))])
            chosen_scores.extend(chosen)
            reject_scores.extend(rejected)
            total_predictions += len(reward) // 2
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        return scores, acc, chosen_scores, reject_scores

    print("***** Running Evaluation *****")

    reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, prompt_eval_dataloader)
    print(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}")

def run_file_eval_ours():
    import numpy as np
    args = parse_args()
    print(args)
    os.environ['TRAIN_MODEL_TYPE'] = 'llama'
    device = torch.device("cuda:0")
    args.max_prompt_seq_len = 512
    args.max_answer_seq_len = 200
    args.model_name_or_path = "/public/user/gouqi/deep-speed-chat-own/training/step2_reward_model_finetuning/outputs/llama2-tldr-base"
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #
    print(tokenizer)
    rm_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=None,
        num_padding_at_beginning=0,
        rlhf_training=True)
    rm_model = rm_model.to(device)
    # args.local_data_files = ["/mnt/user/gouqi/data/helpful-base", "/mnt/user/gouqi/data/harmless-base"]
    train_phase = 2 # 要修改format



    def tokenize_input(prompt,chosen,rejected):
        input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
        prompt = input_format.format(prompt)
        prompt_input_ids = tokenizer.encode(prompt)
        if len(prompt_input_ids) > args.max_prompt_seq_len:
            prompt_input_ids = prompt_input_ids[:args.max_prompt_seq_len]
        chosen_token = {}
        chosen_input_ids = tokenizer.encode(chosen)
        if len(chosen_input_ids) > args.max_answer_seq_len:
            chosen_input_ids = chosen_input_ids[:args.max_answer_seq_len]

        chosen_token['input_ids'] = prompt_input_ids + chosen_input_ids + [tokenizer.eos_token_id]
        pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(chosen_token['input_ids'])
        chosen_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + chosen_token['input_ids']

        rejected_token = {}
        rejected_input_ids = tokenizer.encode(rejected)
        if len(rejected_input_ids) > args.max_answer_seq_len:
            rejected_input_ids = rejected_input_ids[:args.max_answer_seq_len]
        rejected_token['input_ids'] = prompt_input_ids + rejected_input_ids + [tokenizer.eos_token_id]
        pad_len = args.max_prompt_seq_len + args.max_answer_seq_len + 1 - len(rejected_token['input_ids'])
        rejected_token['input_ids'] = [tokenizer.pad_token_id] * pad_len + rejected_token['input_ids']

        chosen_token["input_ids"] = torch.tensor(chosen_token["input_ids"])
        rejected_token["input_ids"] = torch.tensor(rejected_token["input_ids"])
        chosen_token['attention_mask'] = torch.tensor(
            [(1 if x != tokenizer.pad_token_id else 0) for x in chosen_token['input_ids']])
        rejected_token['attention_mask'] = torch.tensor(
            [(1 if x != tokenizer.pad_token_id else 0) for x in rejected_token['input_ids']])
        return chosen_token, rejected_token

    def get_rrrlhf_dataloader():
        raw_dataset = get_raw_dataset('HelpfulRLHFDataset', 'tmp_output', 42, -1, local_path=args.local_data_files)
        eval_dataset = raw_dataset.get_eval_data()
        index = np.load(
            '/mnt/user/gouqi/deep-speed-chat/output/data_files/fullhh/HelpfulRLHFDataset_seed1234_eval_0,0,1_2.npy',
            allow_pickle=True).tolist()
        eval_dataset = Subset(eval_dataset, index)
        print('Length:', len(eval_dataset))

        args.local_data_files = "/mnt/user/gouqi/deep-speed-chat/datasets/harmless-base"
        raw_dataset = get_raw_dataset('HarmlessRLHFDataset', 'tmp_output', 42, -1, local_path=args.local_data_files)
        eval_dataset2 = raw_dataset.get_eval_data()
        index = np.load('/mnt/user/gouqi/deep-speed-chat/output/data_files/fullhh/HarmlessRLHFDataset_seed1234_eval_0,0,1_2.npy', allow_pickle=True).tolist()
        eval_dataset2 = Subset(eval_dataset2, index)
        eval_dataset = ConcatDataset([eval_dataset,eval_dataset2])
        prompt_dataset = []
        chosen_dataset = []
        reject_dataset = []

        for i, tmp_data in enumerate(eval_dataset):
            # tokenize the text
            print(i)
            prompt = raw_dataset.get_prompt(tmp_data)
            chosen = raw_dataset.get_chosen(tmp_data)
            rejected = raw_dataset.get_rejected(tmp_data)

            chosen_token, rejected_token =  tokenize_input(prompt,chosen,rejected)

            chosen_dataset.append(chosen_token)
            reject_dataset.append(rejected_token)

        eval_dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                      tokenizer.pad_token_id, train_phase)
        print(len(eval_dataset))
        prompt_eval_sampler = SequentialSampler(eval_dataset)
        data_collator = DataCollatorReward()
        prompt_eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            sampler=prompt_eval_sampler,
            batch_size=8)
        print(len(prompt_eval_dataloader))

    def eval_item_one_by_one(model):
        path = "/public/user/gouqi/data/TLDR/preference_data/total_sample.jsonl"
        result = []
        data = []
        with open(path,encoding='utf-8') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                data.append(item)
        for i,item in enumerate(tqdm(data)):
            prompt = item['instruction']
            chosen = item['output_1']
            rejected = item['output_2']
            chosen_token, rejected_token = tokenize_input(prompt, chosen, rejected)
            batch = {'input_ids':torch.stack([chosen_token['input_ids'],rejected_token['input_ids']]),
                     'attention_mask': torch.stack([chosen_token['attention_mask'],rejected_token['attention_mask']])
                     }
            batch = to_device(batch,device)
            with torch.no_grad():
                outputs = model(**batch)
            chosen = outputs['chosen_mean_scores'].tolist()[0]
            rejected = outputs['rejected_mean_scores'].tolist()[0]
            item['output_1_score'] = chosen
            item['output_2_score'] = rejected
            result.append(item)
        with open(path.replace('total_sample','train_output_labeled'),'w',encoding='utf-8') as fp:
            for item in result:
                json.dump(item,fp,ensure_ascii=False)
                fp.write('\n')

    def eval_item_one_by_one_test(model):
        path = "/mnt/user/gouqi/data/helpful-base/test.jsonl"
        result = []
        data = []
        with open(path,encoding='utf-8') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                data.append(item)
        path = "/mnt/user/gouqi/data/harmless-base/test.jsonl"
        with open(path,encoding='utf-8') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                data.append(item)
        print(len(data))
        input_format = "<!!USR!!>{}\n<!!AST!!>"
        cnt = 0
        for i,item in enumerate(tqdm(data)):
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']
            prompt = input_format.format(prompt)
            chosen_token, rejected_token = tokenize_input(prompt, chosen, rejected)
            batch = {'input_ids':torch.stack([chosen_token['input_ids'],rejected_token['input_ids']]),
                     'attention_mask': torch.stack([chosen_token['attention_mask'],rejected_token['attention_mask']])
                     }
            batch = to_device(batch,device)
            with torch.no_grad():
                outputs = model(**batch)
            chosen = outputs['chosen_mean_scores'].tolist()[0]
            rejected = outputs['rejected_mean_scores'].tolist()[0]
            if chosen > rejected:
                cnt += 1
            item['chose_score'] = chosen
            item['rejected_score'] = rejected
            result.append(item)
        print('acc:',cnt / len(data))
        with open("/mnt/user/gouqi/deep-speed-chat-own/training/step2_reward_model_finetuning/outputs/llama2-fullhh/test.jsonl",'w',encoding='utf-8') as fp:
            for item in result:
                json.dump(item,fp,ensure_ascii=False)
                fp.write('\n')
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        chosen_scores = []
        reject_scores = []
        for step, batch in enumerate(tqdm(eval_dataloader),desc='eval...'):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            chosen = outputs['chosen_mean_scores'].tolist()
            rejected = outputs['rejected_mean_scores'].tolist()
            scores += sum(chosen) / len(chosen)
            correct_predictions += sum([ chosen[i] > rejected[i] for i in range(len(chosen))])
            chosen_scores.extend(chosen)
            reject_scores.extend(rejected)
            print(chosen)
            print(rejected)
        acc = correct_predictions / len(chosen_scores)
        scores = scores / (step + 1)
        return scores, acc, chosen_scores, reject_scores



    #if eval rr-rlhf
        # print("***** Running Evaluation *****")
        # prompt_eval_dataloader = get_rrrlhf_dataloader()
        # reward_score, acc, chosen_list, reject_list = evaluation_reward(rm_model, prompt_eval_dataloader)
        # print(
        #     f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}")

    # eval file
    # eval_item_one_by_one_test(rm_model)
    eval_item_one_by_one(rm_model)


if __name__ == "__main__":
    from transformers import set_seed
    set_seed(42)
    run_file_eval_ours()
    # run_pair_comparison()
    # run_single_sample()
