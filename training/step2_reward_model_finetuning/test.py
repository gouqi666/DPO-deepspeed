# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
#
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
# if __name__ == '__main__':
#     model = AutoModelForMaskedLM.from_pretrained("facebook/mbart-large-50")
#     print(get_parameter_number(model))
import json
import os.path
import random
import re
from collections import defaultdict

import pandas as pd


def find_in_test():
    scores = []
    with open(r'D:\deeplang\deep-speed-chat-own\training\step2_reward_model_finetuning\output\test.jsonl',encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            diff = item['chose_score'] - item['rejected_score']
            scores.append(diff)
    threshold = 1
    cnt = 0
    total_cnt = 0
    for x in scores:
        if 3 <= abs(x) :
            total_cnt += 1
            if x > 0:
                cnt += 1
    # 分差在[0,1]之间的准确率有2384条，acc为61.19，分差在[1,2]之间的有1223条，acc为81.11，分差在[2,3]之间的有623条，acc为90.04，大于3的有434条，acc为96.54
    # 说明分差很小的时候，reward model判断不准确，这样做dpo的话容易正负样本搞混
    print(len(scores))
    print(total_cnt,cnt)
    print(cnt / total_cnt)
        # if abs(x) < threshold:
        #     scores.append(x)`
        #
def find_in_train():
    scores = []
    dpo_data = []
    ppo_data = []
    threshold = 2
    with open(r'D:\deeplang\data\datasets\tldr\preference_data\train_output_labeled.jsonl',encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            diff = item['output_1_score'] - item['output_2_score']
            if abs(diff) >= threshold:
                dpo_item = {}
                dpo_item['prompt'] = item['instruction']
                dpo_item['chosen'] = item['output_1']
                dpo_item['rejected'] = item['output_2']
                if diff < 0:
                    dpo_item['chosen'],dpo_item['rejected'] = dpo_item['rejected'],dpo_item['chosen']
                dpo_data.append(dpo_item)
            else:
                ppo_item = {}
                ppo_item['prompt'] = item['instruction']
                ppo_item['chosen'] = item['output_1']
                ppo_item['rejected'] = item['output_2']
                if diff < 0:
                    ppo_item['chosen'],ppo_item['rejected'] = ppo_item['rejected'],ppo_item['chosen']
                ppo_data.append(ppo_item)
            scores.append(diff)

    with open(r'D:\deeplang\data\datasets\tldr\preference_data\filter_dpo_datav2.jsonl','w',encoding='utf-8') as fp:
        for item in dpo_data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
    with open(r'D:\deeplang\data\datasets\tldr\preference_data\filter_ppo_datav2.jsonl','w',encoding='utf-8') as fp:
        for item in ppo_data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
    total_cnt = 0
    for x in scores:
        if 3 <= abs(x) < 8:
            total_cnt += 1
    # hh-rlhf 训练数据分差在[0,1)之间的有40712条，[1,2)之间有25619，[2,3)之间有13113条，大于3的有6868条。
    # tldr/preference_data 训练数据分差在[0,1)之间的有82469条，[1,2]之间有51596，[2,3]之间有26356条，大于3的有16283条。
    print(len(scores))
    print(total_cnt)
def sample_for_gpt_check(path):
    # 500
    with open(os.path.join(path,"after_ppo_epoch_0_2545.json"),encoding='utf-8') as fp:
        mpo = json.load(fp)
    ppo_dict = {}
    with open(os.path.join(path,"after_ppo_epoch_0_5395.json"),encoding='utf-8') as fp:
        ppo = json.load(fp)
        for item in ppo:
            ppo_dict[item['prompt']] = item
    assert len(mpo) == len(ppo)
    sample_data = random.sample(mpo,k=500)
    data = []
    for item in sample_data:
        assert item['prompt'] in ppo_dict
        new_item = {}
        new_item['prompt'] = item['prompt']
        new_item['mpo'] = item['response'].replace('<s>',"")
        new_item['ppo'] = ppo_dict[item['prompt']]['response'].replace("<s>","")
        data.append(new_item)
    with open(r'D:\deeplang\deep-speed-chat-own\training\step3_rlhf_finetuning\eval_rlhf\mpo_ppo\gpt_check.json','w',encoding='utf-8') as fp:
        json.dump(data,fp,ensure_ascii=False,indent=2)
# find_in_train()

def read_eval_rlhf(path):
    data = []
    mpo = []
    ppo = []
    with open(path,encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            mpo.append(item['mpo_score'])
            ppo.append(item['ppo_score'])
            data.append(item)
    eq_cnt = 0
    gt_cnt = 0
    for x,y in zip(ppo,mpo):
        if x == y:
            eq_cnt += 1
        elif x < y:
            gt_cnt += 1
    print('mpo_mean:',sum(mpo) / len(mpo))
    print('ppo_mean:',sum(ppo) / len(ppo))
    print('win:',gt_cnt)
    print('tie:',eq_cnt)
    print('lose:',len(ppo) - gt_cnt - eq_cnt)
# sample_for_gpt_check(r"D:\deeplang\deep-speed-chat-own\training\step3_rlhf_finetuning\eval_rlhf\mpo_ppo")

def sort_train_data():
    scores = []
    dpo_data = []
    ppo_data = []
    threshold = 1
    with open(r'D:\deeplang\data\datasets\hh-rlhf\filter_hh_rlhf\train_output_labeled.jsonl',encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            diff = item['output_1_score'] - item['output_2_score']
            if abs(diff) >= threshold:
                dpo_item = {}
                dpo_item['prompt'] = item['instruction']
                dpo_item['chosen'] = item['output_1']
                dpo_item['rejected'] = item['output_2']
                dpo_item['score_diff'] = abs(diff)
                if diff < 0:
                    dpo_item['chosen'],dpo_item['rejected'] = dpo_item['rejected'],dpo_item['chosen']
                dpo_data.append(dpo_item)
            else:
                ppo_item = {}
                ppo_item['prompt'] = item['instruction']
                ppo_item['chosen'] = item['output_1']
                ppo_item['rejected'] = item['output_2']
                ppo_item['score_diff'] = abs(diff)
                if diff < 0:
                    ppo_item['chosen'],ppo_item['rejected'] = ppo_item['rejected'],ppo_item['chosen']
                ppo_data.append(ppo_item)
            scores.append(diff)
    dpo_data.sort(key=lambda x:x['score_diff'],reverse=True)
    ppo_data.sort(key=lambda x:x['score_diff'],reverse=True)
    with open(r"D:\deeplang\data\datasets\hh-rlhf\filter_hh_rlhf\filter_dpo_data_sorted.jsonl",'w',encoding='utf-8') as fp:
        for item in dpo_data:
            del item['score_diff']
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
    with open(r"D:\deeplang\data\datasets\hh-rlhf\filter_hh_rlhf\filter_ppo_data_sorted.jsonl",'w',encoding='utf-8') as fp:
        for item in ppo_data:
            del item['score_diff']
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
    data = dpo_data + ppo_data
    with open(r'D:\deeplang\data\datasets\hh-rlhf\filter_hh_rlhf\total_sample_data_sorted.jsonl','w',encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
# sort_train_data()
# read_eval_rlhf(r"D:\deeplang\deep-speed-chat-own\training\step3_rlhf_finetuning\eval_rlhf\mpo_ppo\eval_result.jsonl")
def convert_parquet_to_jsonl():
    path = r"D:\deeplang\data\datasets\tldr\sft_data\test.parquet"
    df = pd.read_parquet(path)
    json_str = df.to_json(orient="records")
    json_data = json.loads(json_str)
    data = []
    for item in json_data:
        new_item = {}
        new_item['prompt'] = item['prompt']
        new_item['chosen'] = item['label']
        new_item['rejected'] = None
        data.append(new_item)
    with open(r"D:\deeplang\data\datasets\tldr\sft_data\test.jsonl",'w',encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
def process_role_play_data():
    path = r"E:\data\roleplay_data\RoleBench\instructions-zh\role-specific-张飞.jsonl"
    data = []
    character_dict = defaultdict(list)
    exist_character = {'丁春秋', '萧峰', '李云龙', '郭芙蓉', '段誉', '慕容复', '乔峰', '韦小宝', '悟空', 'in', '谢逊', '于谦', 'Sheldon', '令狐冲',\
                       'McGonagall', 'Penny', '欧阳锋', '鸠摩智', '黄药师', '黄蓉', 'Hermione', '赵敏','Luna', 'Ron', '郭靖', '王语嫣',\
                       '张无忌', 'Dumbledore', 'Snape', '佟湘玉', 'Harry', '虚竹', '白展堂', 'Malfoy', '岳不群', 'Raj', '周芷若'}

    exist_character_en = {'Caesar', 'Sonny', 'Angel', 'Jigsaw', 'John', 'Freddy', 'Colonel', 'Gregory', 'Gaston', 'HAL', 'Coriolanus',\
                          'Oliver', 'Bruno', 'D_Artagnan', 'Dr.', 'Tugg', 'Stifler', 'Jeff', 'Rorschach', 'Paul', 'Logan', 'Judge',\
                          'Karl', 'Rachel', 'Queen', 'Tyrion', 'Wade', 'Lyn', 'Violet', 'Po', 'Malcolm', 'Willie', 'Jack', 'Alvy',\
                          'Andrew', 'Fred', 'Frank', 'Stephen', 'Lestat', 'Murphy', 'Jackie', 'Peter', 'Abraham', 'James', 'Tyler',\
                          'Stanley', 'Mater', 'Professor', 'Juno', 'Seth', 'Sherlock', 'Truman', 'Shrek', 'Travis', 'Tom', 'The',\
                          'David', 'Twilight', 'Antonio', 'Judy', 'Randle', 'Thor', 'Walt', 'Fletcher','Theodore', 'Harvey',\
                          'Caden', 'Leonard', 'Jim', 'Pat', 'Benjamin', 'Robert', 'Lucifer', 'Jordan', 'Coach', 'Mary', 'Klaus', \
                          'Blair', 'Sheldon', 'Michael', 'Leroy', 'Doctor'}
    # import langdetect
    # from langdetect import DetectorFactory
    # from zhconv import convert
    # DetectorFactory.seed = 0
    prompt = "现在请你扮演张飞,请你模仿角色的语气与我对话。\n"
    with open(path,encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            item['instruction'] = prompt + item['instruction']
            item['output'] = item['answer']
            del item['answer']
            del item['type']
            del item['id']
            del item['metrics']
            data.append(item)
    with open(r"E:\data\roleplay_data\rolebench-zh-cn.jsonl",'a+',encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')

    for k,v in character_dict.items():
        if len(v) > 200:
            v = random.sample(v,k=200)
        for item in v:
            new_item = {}
            new_item['instruction'] = item['context']
            new_item['output'] = item['target']
            if k not in exist_character:
                data.append(new_item)
    for k in character_dict.keys():
        if k not in exist_character:
            exist_character.add(k)
    print(character_dict.keys())
    exit()
    with open(r"E:\data\roleplay_data\filter_en.jsonl",'a+',encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
def process_moss_roleplay():
    path = r"E:\data\roleplay_data\moss-role-play-zh-cn.jsonl"
    data = []
    other_model_names = ["Moss","MOSS",'belle', 'alpaca', 'gpt', 'PaLM', 'Gopher', 'Chinchilla', 'Galactica', 'GLM-130B',\
                         'BLOOM', 'OPT', 'T5', 'U-PaLM', 'Codex', 'AlphaCode', 'Minerva', 'T0', 'InstructGPT',\
                         'OPT-IML', 'FLAN', 'Flan-T5', 'Flan-PaLM', 'Sparrow', 'Switch Transformers',\
                         'GLaM', 'LaMDA', 'BlenderBot-3', 'Anthropic', 'Retro', 'CPM-3', 'UL2', 'GPT-NeoX',\
                         'Megatron-Turing NLG', 'Jurassic-1', 'GPT-2', 'Megatron-LM', 'GLM', 'CPM-2', '源 1.0', \
                         'ExT5', 'BLOOMZ', 'mT0', 'Sparsely-Gate MOE', 'PLATO', 'Meena', 'Turing NLG', 'ProGen',\
                         'BlenderBot-1', 'DeBERTa v2', 'iGPT', 'PLATO-2', 'GShard', 'M-CTC-T', 'mT5', 'CPM-1',\
                         'DALL-E', 'M6', 'BriVL', 'Chinese-Transformer-XL', 'GPT-Neo', '盘古α', 'XLM-R', 'QAConv',\
                         'CogView', 'ByT5', '悟道 2.0', 'GPT-J', 'V-MOE', '摩天', 'BERT-SG', 'ERNIE 3.0', 'ProtTrans',\
                         'BlenderBot-2', '荔枝', 'baseline-1.5B', 'ProteinLM', 'Macaw', 'HyperCLOVA', 'CoAtNet', '神舟',\
                         'PLATO-XL', 'T5-Efficient', 'Z-code M3', '紫东太初', 'M6-10T', '孟子', '神农', 'MixQG', 'TI-NLP',\
                         'M2M-100', 'PERKS', 'Swin Transformer V2', '比干', '二郎神','燃灯', '闻仲', '余元', '周文王', 'CodeParrot', 'LongT5', 'GLIDE']
    key_words_filter = """
    复旦大学自然语言处理实验室、上海人工智能实验室、张华平、陈立、王文婷、李宇、杨洋、复旦大学、自然语言处理实验室、MOSS、2023年2月7日、8张A100、FudanNLP、8 A100 GPU、February 7th, 2023、February 7, 2023、February 7 2023、Computer Science at Fudan University、Yang Xu、Jiawei Han、Junjie Bai、Xinyue Sun
    """
    key_words_filter = ['复旦大学自然语言处理实验室', '上海人工智能实验室', '张华平', '陈立', '王文婷', '李宇', '杨洋', '自然语言处理实验室',\
                        'MOSS', '2023年2月7日', '8张A100', 'FudanNLP', '8 A100 GPU', 'February 7th,2023', 'February 7, 2023', \
                        'February 7 2023', 'Computer Science at Fudan University', 'Yang Xu', 'Jiawei Han', 'Junjie Bai', 'Xinyue Sun']
    with open(path,encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            new_item = {}
            new_item['instruction'] = ""
            new_item['output'] = ""
            for i,each_turn in enumerate(item['conversation']):
                # replace model_names
                for name in other_model_names:
                    if name in each_turn['human']:
                        each_turn['human'] = each_turn['human'].replace(name,'DLM')
                    if name in each_turn['assistant']:
                        each_turn['assistant'] = each_turn['assistant'].replace(name,'DLM')

                # process multi-turn
                if i == 0:
                    new_item['instruction'] = each_turn['human']
                    new_item['output'] = new_item['output'] + each_turn['assistant']
                else:
                    new_item['output'] = new_item['output'] + '</s>'
                    new_item['output'] = new_item['output'] + each_turn['human']
                    new_item['output'] = new_item['output'] + '</s>'
                    new_item['output'] = each_turn['assistant']

            new_item['source'] = "moss-003-sft-roleplay"
            # filter keywords
            for key_word in key_words_filter:
                if key_word in new_item['instruction'] or key_word in new_item['output']:
                    print(new_item)
                    break
            else:
                data.append(new_item)
    print(len(data))
    with open(r"E:\data\roleplay_data\moss-role-play-zh-cn-filter.jsonl",'w',encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')

def jsonl2json():
    data = []
    with open(r"D:\deeplang\DiverseEvol-master\DiverseEvol-master\databricks-dolly-15k-alpaca-style.jsonl",encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    with open(r"D:\deeplang\DiverseEvol-master\DiverseEvol-master\databricks-dolly-15k-alpaca-style.json","w",encoding='utf-8') as fp:
        json.dump(data,fp,indent=2,ensure_ascii=False)
# convert_parquet_to_jsonl()
# process_role_play_data()
# jsonl2json()
def tmp():
    data = []
    with open(r"E:\data\roleplay_data\filter_zh.jsonl",encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    with open(r"E:\data\roleplay_data\moss-role-play-zh-cn-filter.jsonl",encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    with open(r"E:\data\roleplay_data\rolebench-zh-cn.jsonl",encoding='utf-8') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    with open(r"E:\data\roleplay_data\roleplay-cn-final.jsonl","w",encoding='utf-8') as fp:
        for item in data:
            json.dump(item,fp,ensure_ascii=False)
            fp.write('\n')
# process_moss_roleplay()
# tmp()
find_in_train()