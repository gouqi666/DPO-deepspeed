import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from transformers import   set_seed
from tqdm import tqdm
from accelerate import Accelerator
import os
from itertools import chain
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
set_seed(42)

class Lingowhale_dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,tokenizer):
        data = []
        with open(data_path, encoding='utf-8') as fp:
            for i, line in enumerate(fp.readlines()):
                item = json.loads(line)
                data.append(item['conversations'][0]['value'])
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        # inputs = self.tokenizer(f"<!!USR!!>{self.data[idx]}\n<!!AST!!>", max_length=500, padding=True,truncation =True, return_tensors="pt")
        # return {'token':self.data[idx],'input_ids':inputs['input_ids'][0],'attention_mask':inputs['attention_mask'][0]}

        return {'token': self.data[idx], 'inputs': f"<!!USR!!>{self.data[idx]}\n<!!AST!!>"}
    def __len__(self):
        return len(self.data)
def eval_lingowhale(model_path,data_path,result_path):
    rank = os.environ.get("LOCAL_RANK", 0)
    # print(rank)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', rank=int(rank), world_size=8)
    accelerator = Accelerator()
    pattern = "<!!USR!!>[\s\S]+<!!AST!!>"
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='right', trust_remote_code=True)

    from utils.model.model_utils import create_hf_model
    model = create_hf_model(AutoModelForCausalLM, model_path,
                            tokenizer, ds_config=None)
    model = model.to(torch.device(f'cuda:{rank}'))

    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=f'cuda:{rank}')

    # model = model.bfloat16()
    dataset = Lingowhale_dataset(data_path,tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False)
    # optimizer = torch.optim.Adam(model.parameters())
    data_loader = accelerator.prepare(data_loader)
    # unwrapped = accelerator.unwrap_model(model)
    result = []
    for i,batch in enumerate(tqdm((data_loader))):
        # text = batch.pop('token')

        instructions = batch.pop('token')
        print(batch['inputs'])
        inputs = tokenizer(batch['inputs'], max_length=2000, padding=True, truncation=True,return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = inputs[k].to(accelerator.device)
        pred = model.generate(**inputs, max_new_tokens=1000, do_sample=True,top_p=0.9,top_k=50, temperature=1.0, num_return_sequences=2)
        decoded_result = tokenizer.batch_decode(pred.cpu(), skip_special_tokens=True)
        for j in range(len(instructions)):
            item = {}
            item['instruction'] = instructions[j]
            item['output_1'] = re.sub(pattern,"",decoded_result[j*2])
            item['output_2'] = re.sub(pattern,"",decoded_result[j*2+1])
            result.append(item)
        break
    #
    gathered_result = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(gathered_result,result)
    result = list(chain(*gathered_result))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(result_path,'w',encoding='utf-8') as fp:
            for item in result:
                json.dump(item,fp,ensure_ascii=False)
                fp.write('\n')






class RR_dataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer):
        data = []
        helpful_base_dataset_path = "/mnt/user/gouqi/deep-speed-chat/datasets/helpful-base/train.jsonl"
        harmless_base_dataset_path = "/mnt/user/gouqi/deep-speed-chat/datasets/harmless-base/train.jsonl"

        with open(helpful_base_dataset_path, encoding='utf-8') as fp:
            for i, line in enumerate(fp.readlines()):
                item = json.loads(line)
                data.append(item['prompt'])

        with open(harmless_base_dataset_path, encoding='utf-8') as fp:
            for i, line in enumerate(fp.readlines()):
                item = json.loads(line)
                data.append(item['prompt'])
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        # inputs = self.tokenizer(f"<!!USR!!>{self.data[idx]}\n<!!AST!!>", max_length=500, padding=True,truncation =True, return_tensors="pt")
        # return {'token':self.data[idx],'input_ids':inputs['input_ids'][0],'attention_mask':inputs['attention_mask'][0]}
        return {'token': self.data[idx], 'inputs': self.data[idx] }
    def __len__(self):
        return len(self.data)

class TLDR_dataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer):
        data = []
        tldr_base_dataset_path = "/public/user/gouqi/data/TLDR/preference_data/total.jsonl"

        with open(tldr_base_dataset_path, encoding='utf-8') as fp:
            for i, line in enumerate(fp.readlines()):
                item = json.loads(line)
                if item['prompt'] is not None:
                    data.append(item['prompt'])
        self.data = data
        self.tokenizer = tokenizer
        self.input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
    def __getitem__(self, idx):
        # inputs = self.tokenizer(f"<!!USR!!>{self.data[idx]}\n<!!AST!!>", max_length=500, padding=True,truncation =True, return_tensors="pt")
        # return {'token':self.data[idx],'input_ids':inputs['input_ids'][0],'attention_mask':inputs['attention_mask'][0]}

        return {'token': self.data[idx], 'inputs': self.input_format.format(self.data[idx]) }
    def __len__(self):
        return len(self.data)
def eval_llama2(model_path,data_path,result_path):
    rank = os.environ.get("LOCAL_RANK", 0)
    accelerator = Accelerator()
    pattern = "\n\nHuman:[\s\S]+\n\nAssistant: "
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left', trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=f'cuda:{rank}')
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id # model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # dataset = RR_dataset(tokenizer)
    dataset = TLDR_dataset(tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=8, shuffle=False)
    # optimizer = torch.optim.Adam(model.parameters())
    data_loader = accelerator.prepare(data_loader)
    # unwrapped = accelerator.unwrap_model(model)
    result = []
    for i,batch in enumerate(tqdm((data_loader))):
        # text = batch.pop('token')
        instructions = batch.pop('token')
        inputs = tokenizer(batch['inputs'], max_length=512, padding=True,truncation=True,return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = inputs[k].to(accelerator.device)
        pred = model.generate(**inputs, max_new_tokens=200,do_sample=True,top_p=0.9,top_k=50, temperature=1.0, num_return_sequences=2)
        decoded_result = tokenizer.batch_decode(pred.cpu(), skip_special_tokens=True)
        for j in range(len(instructions)):
            item = {}
            item['instruction'] = instructions[j]
            item['output_1'] = re.sub(pattern,"",decoded_result[j*2]).replace(instructions[j],"")
            item['output_2'] = re.sub(pattern,"",decoded_result[j*2+1]).replace(instructions[j],"")
            result.append(item)
    #
    gathered_result = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(gathered_result,result)
    result = list(chain(*gathered_result))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(result_path,'w',encoding='utf-8') as fp:
            for item in result:
                json.dump(item,fp,ensure_ascii=False)
                fp.write('\n')

def read(data_path):
    lens = []
    count = 0
    wrong_data = []
    data = []
    pattern = "回答[\s\S]更好"
    count = 0
    with open(data_path, encoding='utf-8') as fp:
        for i, line in enumerate(fp.readlines()):
            item = json.loads(line)
            res = re.findall(pattern,item['gpt_3.5_explain'])
            idx = item['gpt_3.5_explain'].split('\n')[0].strip()
            assert '1' in idx or '2' in idx
            if '1' in idx:
                idx = 1
            elif '2'  in idx:
                idx = 2

            for x in res:
                if (x[2] == '二' and idx != 2) or (x[2] == '一' and idx != 1):
                    count += 1
                    item['chosen'], item['rejected'] = item['rejected'], item['chosen']
                    break
            data.append(item)
    print(count)
    # with open(r'D:\deeplang\data\sharegpt_zh\output_gpt3.5_reward_data_cleaned.jsonl', 'w', encoding='utf-8') as fp:
    #     for item in data:
    #         json.dump(item, fp, ensure_ascii=False)
    #         fp.write('\n')
if __name__ == "__main__":
    model_path = "/public/user/gouqi/deep-speed-chat-own/training/step1_supervised_finetuning/outputs/llama2-tldr-base"
    data_path = "/public/user/gouqi/data/TLDR/preference_data/total.jsonl"
    result_path = "/public/user/gouqi/data/TLDR/preference_data/total_sample.jsonl"
    # eval_lingowhale(model_path,data_path,result_path)
    eval_llama2(model_path,data_path,result_path)

    # model_path = "/mnt/user/gouqi/deep-speed-chat/training/step1_supervised_finetuning/outputs/llama2-fullhh-epoch1"
    # data_path = ""
    # result_path = "/mnt/user/gouqi/deep-speed-chat/training/step1_supervised_finetuning/outputs/llama2-fullhh-train_data/train_output.jsonl"
    # eval_llama2(model_path,data_path,result_path)


    # read(r'D:\deeplang\data\sharegpt_zh\output_gpt3.5_reward_data_cleaned.jsonl')

    '''
        torchrun --nproc_per_node=8 eval_sft.py
    '''