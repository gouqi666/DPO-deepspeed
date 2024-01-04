# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import logging
import torch
import sys
import os
from tqdm import tqdm


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(
    os.path.abspath('../../'))
print(sys.path)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils.model.model_utils import create_hf_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        default='/mnt/data01/gq/deep-speed-chat/training/step1_sft/output/llama-singleturn-rlhf',
        help="Path to baseline model",
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        default='/mnt/data01/gq/deep-speed-chat/training/step1_sft/output/llama-pro-full',
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=5,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    parser.add_argument("--dataset-name",
                        type=str,
                        default="Dahoas/synthetic-instruct-gptj-pairwise",
                        )
    parser.add_argument("--output-path",
                        type=str,
                        default="./output",
                        )
    parser.add_argument("--local-data-files",
                        type=str,
                        default="/home/gq/deeplang/deep-speed-chat/datasets/synthetic-instruct-gptj-pairwise",
                       )

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             do_sample=False,
             max_length=512):

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,
                                  num_beams=3,
                                  do_sample=True,
                                  max_new_tokens =512,
                                  temperature=0.95,
                                  # top_p=0.9,
                                  # top_k=30,
                                  )

    '''
    model.generate(c.input_ids,
                                attention_mask=c.attention_mask,
                                  num_beams=3,
                                  num_beam_groups=1,
                                  do_sample=False,
                                  num_return_sequences=1,
                                  max_length=512,
                                  top_p=0.9,
                                  top_k=30,
                                  temperature=1.0)
    '''
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    print(result)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt,return_tensors="pt").to(device) # ,padding='max_length',max_length=256,truncation=True
        # print(inputs)
        print("==========Baseline: Beam Search=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          max_length=1024,
                          do_sample=True
                          )
        # print(inputs)
        # print(prompt)
        # print(inputs)
        # print_utils(r_base)
        # print("==========finetune: Beam Search=========")
        # r_finetune_g = generate(model_fintuned,
        #                         tokenizer,
        #                         inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_length=512,
        #                         do_sample=True)
        # print_utils(r_finetune_g)
        item = {}
        item['prompt'] = prompt[prompt.find('Instruction:')+13:prompt.find('Response:')-6]
        item['sft-7b'] = []
        for r in r_base:
            item['sft-7b'].append(r[r.find('Response:')+10:])
        item['after-ppo'] = []
        # for r in r_finetune_g:
        #     item['after-ppo'].append(r[r.find('Response:')+10:])
        results.append(item)
    return results
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        # print("====================prompt end=============================")
        # print()
        # print()

def create_eval_prompt(args):
    from utils.data.data_utils import get_raw_dataset
    raw_dataset = get_raw_dataset(args.dataset_name, args.output_path, 1234, -1, local_path=args.local_data_files)
    return raw_dataset



def main():
    args = parse_args()
    device = torch.device("cuda:0")
    args.model_name_or_path_baseline = "/public/user/gouqi/deep-speed-chat-own/training/step1_supervised_finetuning/outputs/llama2-tldr-base"
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig,set_seed
    set_seed(42)
    config = AutoConfig.from_pretrained(args.model_name_or_path_baseline)
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path_baseline)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(tokenizer.padding_side)
    print('tokenizer:',tokenizer.pad_token_id,tokenizer.unk_token_id,tokenizer.eos_token_id)
    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    # model_baseline.resize_token_embeddings(len(tokenizer))

    # model_baseline = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_baseline)


    # model_baseline.config.rope_type = 'huggingface'
    model_fintuned = model_baseline
    # model_fintuned = create_hf_model(AutoModelForCausalLM,
    #                                  args.model_name_or_path_finetune,
    #                                  tokenizer, ds_config=None, rlhf_training=True)
    # model_fintuned.config.rope_type = 'ours'
    # model_ckpt_path = os.path.join(args.model_name_or_path_finetune, 'pytorch_model.bin')
    # assert os.path.exists(
    #     model_ckpt_path
    # ), f"Cannot find model checkpoint at {model_ckpt_path}"
    #
    # model_fintuned.load_state_dict(
    #     torch.load(model_ckpt_path, map_location='cpu'),strict=False)

    model_baseline.to(device)
    model_fintuned.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        # sft_format = "\n\nHuman: {}\n\nAssistant:"
        # # prompt_dataset = create_eval_prompt(args)
        # sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        # prompts = [
        #     "Are scented candles bad for the environment?",
        #     "Hello",
        #     "What is the capital of China and where is it located?",
        #     "Do you prefer red apples or green apples?",
        #     "What's the weather like today?",
        #     "What is your favorite cuss word?",
        #     "My doctor says I need to stop eating meat, but I love it too much. He recommended plant-based 'meat'. How does this compare to regular meat?",
        #     "How to separate the yolk from egg white?"
        # ]
        # summary
        sft_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
        prompts = [
            "SUBREDDIT: r/relationships\nTITLE: Me [19 F] with my friend [19 M], not sure if I may have messed things up already.\nPOST: Hello hello everybody. I hope this isn't too trivial of a question to ask on here, but I've been feeling a bit out of my depth when it comes to this situation (I've had only one relationship before, and for many reasons, it was out of the ordinary).\n\nOkay! So, a couple of weeks ago, I started talking to this guy on Facebook, through a student group that we were both part of. I thought he was sort of cute, so I sent him a PM just to talk, etc, etc. We're both transfer students at the same school, so I knew that we could eventually meet in person once we both moved on-campus. So, we did, and we hung out maybe twice, just as friends.\n\nOkay. So, everything is going pretty well. We talk over Facebook and Snapchat, whatever. So, Saturday night, I was just hanging out with people and kind of being bored, when I got a Snapchat from him asking what I was doing. I asked if he wanted to hang out, so we did. \n\nWe ended up smoking pot (the first time for me, ever), and sort of just wandering around. Eventually we ended up back at his dorm room, where high me decided to just go for it, and I came on to him pretty strongly. It worked out for me (luckily, otherwise things would have been really super awkward), and we ended up messing around but not having sex.\n\nYesterday, however, I ended up going to hang out with him again, and this time we did sleep together. Afterward, we kind of discussed what we were going to do, and he just said that he wanted to \"play it by ear\" and not slap any labels on anything. I'm wondering if this means that he wants a fwb-type situation, or if he might actually be interested in me. The way I've been acting is extremely out of character for me, and I am not interested in having a fuck buddy. I like him, and I would be very interested in maybe seeing where things go, but I'm worried that I may have ruined my chances of a relationship by sleeping with him already.\nTL;DR: ",
            "SUBREDDIT: r/relationships\nTITLE: Me [18M] questioning if I should ask out my friend [17F]\nPOST: Ok so perhaps this isn't the most dramatic thing you have seen on this subreddit but I feel like I'm stuck.\n\nI have been friends with this girl for a while now asking her and things were going great until I asked her to prom. I am hoping this will be a great night however ever since I asked her a couple weeks ago I find I can no longer talk to her. Out conversations are awkward with me trying to start the majority of topics (Badly written but you get the picture)\n\nI really like the girl but i'm wondering if she saw too much into the prom-posal. Should I just give up, or ask her out? Im not sure\nTL;DR: ",
            "SUBREDDIT: r/AskReddit\nTITLE: Cock blocked by a friend (Who's a girl).\nPOST: So for the past week there's been this girl in one of my classes I've been talking to, she's pretty cute (dyed red hair, fair skin, a few freckles, not ginger), she loves star wars and I suspect she's a redditor.  I was going to ask her for her number today, but a girl i met about a year ago came and sat right where the red head had been sitting, effectively cock-blocking me and driving the girl I was interested in away.  Now it seems like the red head thinks I'm uninterested in her and has since found some other guy to talk to.  Has anybody been in a similar scenario?  Advice?\nTL;DR: "
        ]
        prompts = [sft_format.format(x) for x in prompts]
    elif args.language == "Chinese":
        sft_format = "下面的指令描述了一个需要完成的任务，请编写一个回复来合理地完成请求。\n\n### 指令：\n{}\n\n### 回复：\n"
        # prompts = [
        #     "Human: 请用几句话介绍一下微软? Assistant:",
        #     "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
        #     "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
        #     "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
        #     "Human: 鸟类为什么要南迁过冬? Assistant:"
        # ]
        honesty_prompt = []
        harmlessness_prompt = []
        with open('/mnt/gouqi/deep-speed-chat/datasets/zh_honesty.json',encoding='utf-8') as fp:
            lines = fp.readlines()
            for i in range(50):
                item = json.loads(lines[i])
                honesty_prompt.append(item['instruction'])
        with open('/mnt/gouqi/deep-speed-chat/datasets/zh_harmless.json',encoding='utf-8') as fp:
            lines = fp.readlines()
            for i in range(len(lines)):
                item = json.loads(lines[i])
                harmlessness_prompt.append(item['user_prompt'])

        prompts = honesty_prompt + harmlessness_prompt
        prompts = [sft_format.format(x) for x in prompts]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    results = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts)
    # with open(os.path.join('/mnt/gouqi/deep-speed-chat/training/step1_supervised_finetuning/outputs','test_honesty_harmlessness_zh.json'),'w') as fp:
    #     json.dump(results,fp,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()

"""
"""