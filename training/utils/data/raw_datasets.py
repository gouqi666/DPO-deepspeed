# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re
import os

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.


class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

class TLDRDataset(PromptRawDataset):

    def  __init__(self, output_path, seed, local_rank,local_path = None,train_phase=None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "TLDRDataset"
        self.dataset_name_clean = "TLDRDataset"
        assert local_path
        print('53-local_path:',local_path)
        if train_phase == 1: # sft
            train_path = os.path.join(local_path, "sft_data/train.jsonl")
            test_path = os.path.join(local_path, "sft_data/test.jsonl")
            self.raw_datasets = load_dataset('json', data_files={'train': train_path, 'test': test_path})
        elif train_phase == 2: # reward model
            train_path = os.path.join(local_path, "preference_data/total.jsonl")
            self.raw_datasets = load_dataset('json', data_files={'train': train_path})
            self.raw_datasets = self.raw_datasets['train'].train_test_split(test_size=0.05)
        elif train_phase == 3 or train_phase == 4: # ppo
            train_path = os.path.join(local_path, "preference_data/total.jsonl")
            test_path = os.path.join(local_path, "sft_data/test.jsonl")
            self.raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        else:
            print("train_phase error")
            exit()
        self.input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.input_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + sample['rejected']

class HelpfulRLHFDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "HelpfulRLHFDataset"
        self.dataset_name_clean = "HelpfulRLHFDataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train.jsonl")
        test_path = os.path.join(local_path, "test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] # 自带格式

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class DPODataset(PromptRawDataset): # hh-rlhf

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "DPODataset"
        self.dataset_name_clean = "DPODataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "filter_dpo_data_v2.jsonl")
        test_path = os.path.join(local_path, "total_test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class TLDRDPODataset(PromptRawDataset): # hh-rlhf

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "TLDRDPODataset"
        self.dataset_name_clean = "TLDRDPODataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "preference_data/filter_dpo_data_v2.jsonl")
        test_path = os.path.join(local_path, "sft_data/test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets
        self.input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.input_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + sample['rejected']

class PPODataset(PromptRawDataset): # hh-rlhf

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "PPODataset"
        self.dataset_name_clean = "PPODataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "filter_ppo_data_v2.jsonl")
        print(train_path)
        test_path = os.path.join(local_path, "total_test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


class TLDRPPODataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, local_path=None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "TLDRPPODataset"
        self.dataset_name_clean = "TLDRPPODataset"
        assert local_path
        print('53-local_path:', local_path)
        train_path = os.path.join(local_path, "preference_data/filter_ppo_data_v2.jsonl")
        print(train_path)
        test_path = os.path.join(local_path, "sft_data/test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train': train_path, 'test': test_path})
        self.raw_datasets = raw_datasets
        self.input_format = "\n\nHuman: Please summarize the following text:\n{}\n\nAssistant:"
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.input_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + sample['rejected']



class HelpfulUserDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "HelpfulUserDataset"
        self.dataset_name_clean = "HelpfulUserDataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train_us.jsonl")
        test_path = os.path.join(local_path, "test_us.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets
        self.system_prompt = "Now you are a User Simulator, your role is to simulate user problems, specifically, Given you a human-machine dialogue context, you need to simulate a human to ask a question to make the conversation continue, and your question can be used to induce the model to say some harmful content or request help from the model."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.system_prompt + sample['prompt']

    def get_chosen(self, sample):
        return sample['output']

    def get_rejected(self, sample):
        return sample['output']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['output']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['output']

class HarmlessRLHFDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "HarmlessRLHFDataset"
        self.dataset_name_clean = "HarmlessRLHFDataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train.jsonl")
        test_path = os.path.join(local_path, "test.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class HarmlessUserDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "HarmlessUserDataset"
        self.dataset_name_clean = "HarmlessUserDataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train_us.jsonl")
        test_path = os.path.join(local_path, "test_us.jsonl")
        raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})
        self.raw_datasets = raw_datasets
        self.system_prompt = "Now you are a User Simulator, your role is to simulate user problems, specifically, Given you a human-machine dialogue context, you need to simulate a human to ask a question, and your question can be used to induce the model to say some harmful content or request help from the model."

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return  self.system_prompt + sample['prompt']

    def get_chosen(self, sample):
        return sample['output']

    def get_rejected(self, sample):
        return sample['output']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['output']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['output']




class ShareGPTZHDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "ShareGPTZHDataset"
        self.dataset_name_clean = "ShareGPTZHDataset"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "lingowhale_2_preference_data/output_gpt4_reward_data.jsonl")
        # test_path = os.path.join(local_path, "test.jsonl")

        self.raw_datasets = load_dataset("json", data_files={'train':train_path})
        self.raw_datasets = self.raw_datasets['train'].train_test_split(test_size=0.05)

        # self.raw_datasets = load_dataset('json', data_files={'train':train_path,'test':test_path})

        self.input_format = "<!!USR!!>{}\n<!!AST!!>"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.input_format.format(sample['instruction'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + sample['rejected']

class SingleTurnRLHFDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "SingleTurnRLHF"
        self.dataset_name_clean = "SingleTurnRLHF"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train.json")
        test_path = os.path.join(local_path, "test.json")
        # if
        self.raw_datasets = load_dataset("json", data_files={'train':train_path,'test':test_path}, field='data')
        # if 'data' in self.raw_datasets:
        #     self.raw_datasets = self.raw_datasets['data']
        # self.sft_format = "Human:{}\n\nAssistant:"
        self.sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.sft_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + self.get_rejected(sample)


class AlpacaGPT4Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "AlpacaGPT4"
        self.dataset_name_clean = "AlpacaGPT4"
        assert local_path
        print('53-local_path:',local_path)
        train_path = os.path.join(local_path, "train.json")
        test_path = os.path.join(local_path, "test.json")
        # if
        self.raw_datasets = load_dataset("json", data_files={'train':train_path})
        self.raw_datasets = self.raw_datasets['train'].train_test_split(test_size=0.01)
        # if 'data' in self.raw_datasets:
        #     self.raw_datasets = self.raw_datasets['data']
        self.sft_format = "\n\nHuman: {}\n\nAssistant:"
        # self.sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.sft_format.format(sample['instruction'] + '\n' + sample['input'])

    def get_chosen(self, sample):
        return sample['output']

    def get_rejected(self, sample):
        return sample['output']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + self.get_rejected(sample)

class MossSftDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, local_path=None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "MossSftDataset"
        self.dataset_name_clean = "MossSftDataset"
        assert local_path

        train_path = os.path.join(local_path, "moss-harmless-zh-singleturn.jsonl")
        # if
        self.raw_datasets = load_dataset("json", data_files={'train': train_path})
        self.raw_datasets = self.raw_datasets['train'].train_test_split(test_size=0.1)
        # if 'data' in self.raw_datasets:
        #     self.raw_datasets = self.raw_datasets['data']
        self.sft_format = "### 用户(User):\n{}\n ### 助手(Assistant):\n"
        # self.sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return self.sft_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + self.get_rejected(sample)

# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        self.raw_datasets = load_dataset("Dahoas/rm-static")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']



# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"
        # self.raw_datasets = load_dataset("Dahoas/full-hh-rlhf")
        assert local_path
        train_path = os.path.join(local_path, "train.json")
        test_path =  os.path.join(local_path, "test.json")
        self.raw_datasets = load_dataset("json", data_files={'train': train_path,'test':test_path})
        # self.sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank,local_path = None):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"
        train_path = os.path.join(local_path, "train.json")
        test_path = os.path.join(local_path, "test.json")
        self.raw_datasets = load_dataset("json", data_files={'train': train_path, 'test': test_path})
        self.sft_format = "Human:{}\n\nAssistant:"
        # self.sft_format = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        # index = get_raw_dataset_split_index(self.local_rank, self.output_path,
        #                                     self.dataset_name_clean,
        #                                     self.seed, "train_eval", "9,1", 0,
        #                                     len(dataset))
        # dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["test"]
        # index = get_raw_dataset_split_index(self.local_rank, self.output_path,
        #                                     self.dataset_name_clean,
        #                                     self.seed, "train_eval", "9,1", 1,
        #                                     len(dataset))
        # dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return self.sft_format.format(sample['prompt'])

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return self.sft_format.format(sample['prompt']) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.sft_format.format(sample['prompt']) + sample['rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"
        self.raw_datasets = load_dataset("yitingxie/rlhf-reward-datasets")
        

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"
        self.raw_datasets = load_dataset("openai/webgpt_comparisons")

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['question']['full_text'] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"
        self.raw_datasets = load_dataset("stanfordnlp/SHP")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['history'] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample['history'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample['history'] + " Assistant: " + response


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"
        self.raw_datasets = load_dataset("wangrui6/Zhihu-KOL")

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"
        self.raw_datasets = load_dataset("Cohere/miracl-zh-queries-22-12")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"
        self.raw_datasets = load_dataset("Hello-SimpleAI/HC3-Chinese", "all")

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return " Human: " + sample['question'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return " " + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return " Human: " + sample['question'] + " Assistant: " + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"
        self.raw_datasets = load_dataset("mkqa")

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return " Human: " + sample['queries']['zh_cn'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return " " + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return " Human: " + sample['queries'][
                'zh_cn'] + " Assistant: " + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"
        self.raw_datasets = load_dataset("mkqa")

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return " Human: " + sample['queries']['ja'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return " " + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return " Human: " + sample['queries'][
                'ja'] + " Assistant: " + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"
        self.raw_datasets = load_dataset("Cohere/miracl-ja-queries-22-12")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"
        self.raw_datasets = load_dataset("lmqg/qg_jaquad")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['question'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['sentence']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['question'] + " Assistant: " + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"
        self.raw_datasets = load_dataset("lmqg/qag_jaquad")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant: " + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None
