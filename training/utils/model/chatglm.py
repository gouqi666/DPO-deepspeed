# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import numpy as np
from torch import nn
from ..utils import print_rank_0


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class ChatGLMModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.config.n_embd = self.config.hidden_size
        self.v_head = nn.Linear(self.config.n_embd, 13, bias=False)
        self.base_model = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                prompt_length=None,
                use_cache=False):
        loss = None

        transformer_outputs = self.base_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=use_cache)

        hidden_states = transformer_outputs.hidden_states[-1]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        attention_mask = attention_mask[:bs]
        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0

        if attention_mask is None or self.tokenizer.padding_side == 'left':
            chosen_mean_scores = chosen_rewards[:, -1]
            rejected_mean_scores = rejected_rewards[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)  # (bs,)
            chosen_mean_scores = []
            rejected_mean_scores = []
            for i in range(bs):
                chosen_mean_scores.append(chosen_rewards[i, last_index[i]])
                rejected_mean_scores.append(rejected_rewards[i, last_index[i]])
            chosen_mean_scores = torch.stack(chosen_mean_scores)
            rejected_mean_scores = torch.stack(rejected_mean_scores)
        loss += -torch.log(torch.sigmoid(chosen_mean_scores - rejected_mean_scores)).mean()
        # loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
        #                                         r_truncated_reward).mean()
        loss = loss / bs

        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

