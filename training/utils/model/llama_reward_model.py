# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import numpy as np
from torch import nn
from transformers import PreTrainedModel
from ..utils import print_rank_0

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class LlamaRewardModel(PreTrainedModel):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__(base_model.config)
        self.config = base_model.config
        self.config.n_embd = self.config.hidden_size 
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
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
        transformer_outputs = self.rwtransformer(
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
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        chosen_attention_mask = attention_mask[:bs]
        rejected_attention_mask = attention_mask[bs:]
        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0

        if attention_mask is None:
            chosen_mean_scores = chosen_rewards[:,-1]
            rejected_mean_scores = rejected_rewards[:,-1]
        else:
            chosen_last_index = chosen_attention_mask.cumsum(dim=1).argmax(dim=1) # (bs,)
            rejected_last_index = rejected_attention_mask.cumsum(dim=1).argmax(dim=1)  # (bs,)
            chosen_mean_scores = []
            rejected_mean_scores = []
            for i in range(bs):
                chosen_mean_scores.append(chosen_rewards[i,chosen_last_index[i]])
                rejected_mean_scores.append(rejected_rewards[i, rejected_last_index[i]])
            chosen_mean_scores = torch.stack(chosen_mean_scores)
            rejected_mean_scores = torch.stack(rejected_mean_scores)

            # (chosen_rewards - rejected_rewards) * attention_mask

            c_truncated_reward = []
            r_truncated_reward = []


        loss = -torch.log(torch.sigmoid(chosen_mean_scores - rejected_mean_scores)).mean()



        # loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
        #                                         r_truncated_reward).mean()
        loss = loss / bs

        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      use_cache=False):

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0] # b * sqe_len * hidden_size
        values = self.v_head(hidden_states).squeeze(-1) # b * seq_len


        if return_value_only:
            return values
        else:
            '''
            original code, I think the prompt length is not accurate,so I abandoned it.
            
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])

            chosen_end_scores = [(x - mean)/np.sqrt(var) for x in chosen_end_scores]
            # print_rank_0(chosen_end_scores)
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
            '''

            # normal
            mean = 0 # 2.86 #
            var = 1 # 3.82 #
            # values = (values - mean) / np.sqrt(var)
            if attention_mask is None:
                reward = values[:,-1]
            else:
                last_index = attention_mask.cumsum(dim=1).argmax(dim=1) # (bs,)
                reward = []
                bs = last_index.size(0)
                for i in range(bs):
                    reward.append(values[i,last_index[i]])
                    # reward.append(values[i,-1])
                reward = torch.stack(reward)
            return {
                "values": values,
                # "chosen_end_scores": reward
                "chosen_end_scores": (reward - mean) / np.sqrt(var),
            }


