# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    AutoModelForCausalLM,
AutoModelForSequenceClassification
)

from transformers.deepspeed import HfDeepSpeedConfig

from .llama_reward_model import LlamaRewardModel
from .reward_model import RewardModel


def  create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False):
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration

    # if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
    #     dschf = HfDeepSpeedConfig(ds_config)
    # else:
    #     dschf = None

    model = model_class.from_pretrained(model_name_or_path,trust_remote_code=True)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id # model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        ):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    lm_model = create_hf_model(AutoModelForCausalLM, model_name_or_path, tokenizer,
                                ds_config, rlhf_training)
    reward_model_class = LlamaRewardModel
    critic_model = reward_model_class(
        lm_model.model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    if rlhf_training: # rlhf training stage, 会重新load reward model 和 critic model

        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'),strict=False)
    return critic_model