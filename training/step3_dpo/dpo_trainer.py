# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Tuple, List, Optional,Union
import sys
import os
import deepspeed
import numpy as np
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class DeepSpeedDPOTrainer():

    def __init__(self, dpo_engine, tokenizer, args):
        self.dpo_engine = dpo_engine
        self.model = dpo_engine.actor
        self.ref_model = dpo_engine.ref
        self.reward_model = dpo_engine.reward
        self.label_pad_token_id = -100
        self.tokenizer = tokenizer
        self.args = args
        self.beta = args.beta
        self.max_answer_seq_len = args.max_answer_seq_len
        # self.end_of_conversation_token_id = self.tokenizer(
        #     args.end_of_conversation_token)['input_ids'][-1]
        self.end_of_conversation_token_id = self.tokenizer.encode(args.end_of_conversation_token)[-1]
        # Those value can be changed
        # self.kl_ctl = 0.05 # 0.02
        self.device_num = torch.cuda.device_count()
        self.target = 6
        self.init_kl_coef = 0.4
        self.horizon = 10000
        # self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.target, self.horizon)
        self.kl_ctl = FixedKLController(self.init_kl_coef)
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95


    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def compute_loss(
        self,
        inputs,
        return_outputs=True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    def get_chosen_rejected_logps(self,policy_all_logps, reference_policy_all_logps):
        bs = policy_all_logps.size(0) // 2
        policy_chosen_logps = policy_all_logps[:bs]
        policy_rejected_logps = policy_all_logps[bs:]
        reference_chosen_logps = reference_policy_all_logps[:bs]
        reference_rejected_logps = reference_policy_all_logps[bs:]
        return policy_chosen_logps,policy_rejected_logps,reference_chosen_logps,reference_rejected_logps
    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        policy_all_logits = self.forward(self.model, batch)
        with torch.no_grad():
            reference_all_logits = self.forward(self.ref_model, batch)
        policy_all_logps = self._get_batch_logps(policy_all_logits,labels=batch['labels'])
        reference_policy_all_logps = self._get_batch_logps(reference_all_logits,labels=batch['labels'])

        (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        ) = self.get_chosen_rejected_logps(policy_all_logps, reference_policy_all_logps)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval" if train_eval == "eval" else "train"
        metrics[f"{prefix}/chosen_rewards"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}/rejected_rewards"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}/rewards_accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}/rewards_margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()




        # metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        # metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        # metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        model_kwargs = {"use_cache":False}
        all_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        return all_logits

    def train_dpo(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        # input_ids, attention_mask,
        self.train()
        (loss, metrics) = self.compute_loss(inputs)
        self.model.backward(loss)
        self.model.step()
        return loss, metrics
    def _validate_training_mode(self):
        assert self.model.module.training

    def _validate_evaluation_mode(self):
        assert not self.model.module.training
        assert not self.ref_model.module.training

    def train(self):
        self.model.train()

    def eval(self):
        self.ref_model.eval()
        self.model.eval()

class DeepSpeedDPOTrainerUnsupervised(DeepSpeedDPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
