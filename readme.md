# New
PPO of our version will be coming soon!

# What is this repo?

This repo includes some implementation of Alignment methods, and currently includes a reference implementation of the DPO algorithm for training language models from preference data, as described in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).  

Although there are some implementations of DPO, here we provide a version in the [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) style. For PPO, you can refer [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) version now and we will provide our version in the future.

The DPO pipeline has three stages(For evaluation, we add a stage of reward modeling):

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run reward model finetuning on the dataset(s) of interset.
3. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples).

The files in this repo are:  
`training`:   
--- `step1_supervised_finetuing`: sft directory, same as [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)  
--- `step2_reward_model_finetuing`: reward modelling directory, [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)  
--- `step3_dpo`: DPO running directory  
--- `utils`: some utils

# Running SFT
Model:Here we provide a llama2 sft version, you can change any huggingface model by change `model_name_or_path`  
Datasets: now we provide HelpfulRLHFDataset, HarmlessRLHFDataset, etc. just change the `data_path` to your datasets location. If you want to add new dataset, you should add a new DataClass in the `./utils/data/raw_datasets.py` and cite it in `./utils/data/data_utils.py`.
you can change different data_splits to avoid data leak during sft and dpo training such as `5,0,5`.
**Running SFT**  
`cd training/step1_supervised_finetuning && bash run_llama.sh`

**Evaluation**
Here we provide some evaluation scripts, you can refer to `training/step1_supervised_finetuning/eval_sft.py` and `training/step1_supervised_finetuning/prompt_eval.py`

# Running Reward Modeling
Model:Here we provide a llama2 sft version, you can change any huggingface model by change `model_name_or_path`  
Datasets: now we provide HelpfulRLHFDataset, HarmlessRLHFDataset, etc. just change the `data_path` to your datasets location. If you want to add new dataset, you should add a new DataClass in the `./utils/data/raw_datasets.py` and cite it in `./utils/data/data_utils.py`    
Running Reward Modeling   
`cd training/step2_reward_model_finetuning && bash run_llama.sh`

**Evaluation**   
refer to `training/step2_reward_model_finetuning/rw_eval.py`

# Running DPO
Model:Change your `model_name_or_path` to your sft model path in the stage1 training!  
Datasets: now we provide HelpfulRLHFDataset, HarmlessRLHFDataset, etc. just change the `data_path` to your datasets location. If you want to add new dataset, you should add a new DataClass in the `./utils/data/raw_datasets.py` and cite it in `./utils/data/data_utils.py`    
**Running DPO**  
`cd training/step3_dpo && bash run_dpo_llama.sh`

**Evaluation**
refer to stage1 sft training eval scripts.

# Reference
If DPO or this repository is useful in your own research, you can use the following BibTeX entry:
```angular2html
@inproceedings{
    rafailov2023direct,
    title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
    author={Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D Manning and Stefano Ermon and Chelsea Finn},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://arxiv.org/abs/2305.18290}
}
```