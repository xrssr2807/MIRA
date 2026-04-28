#!/usr/bin/env python
# -*- coding:utf-8 _*-

'''
Part of code from time_moe.models.trainer.hf_trainer
https://github.com/Time-MoE
'''

import math
import json
import os
from dataclasses import field, dataclass
from functools import partial

import inspect

import transformers
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
from mira.utils.log_util import log_in_local_rank_0


class MIRATrainer(transformers.Trainer):
    epsilon = 1e-8

    def __init__(self, label_column: str = 'labels', loss_mask_column: str = 'loss_mask', *positional_args, **kwargs):
        super().__init__(*positional_args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.label_column = label_column
        self.loss_mask_column = loss_mask_column

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        optimizer = self.optimizer if optimizer is None else optimizer
        min_lr_ratio = self.args.min_learning_rate / self.args.learning_rate
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine':
                self.lr_scheduler = get_cosine_schedule_with_warmup_min_lr(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                    warmup_start_lr=getattr(self.args, 'warmup_start_lr', 0.0),
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Override checkpoint loading to fix max_steps mismatch and sync optimizer LR.

        The checkpoint's trainer_state.json may have a different max_steps than our
        current training run (e.g., original run had max_steps=73586, new run uses 36793).
        After loading the checkpoint, we restore our intended max_steps and sync the
        optimizer LR to match what the scheduler will compute.
        """
        # Save our intended max_steps before the checkpoint overwrites it
        intended_max_steps = self.args.max_steps

        # Load the checkpoint (this will overwrite self.args.max_steps from checkpoint state)
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        # Restore our intended max_steps so the scheduler uses correct total steps
        if intended_max_steps is not None and intended_max_steps != self.args.max_steps:
            log_in_local_rank_0(
                f'Fixed max_steps after checkpoint load: '
                f'{self.args.max_steps} -> {intended_max_steps}',
                type='info'
            )
            self.args.max_steps = intended_max_steps

        # Sync optimizer LR to what the scheduler will compute at current step
        if self.optimizer is not None and self.state.global_step > 0:
            current_step = self.state.global_step
            num_training_steps = self.args.max_steps
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            lr = self.args.learning_rate
            min_lr = self.args.min_learning_rate
            scheduler_type = self.args.lr_scheduler_type

            if num_training_steps and num_training_steps > 0:
                target_lr = compute_lr_at_step(
                    scheduler_type=scheduler_type,
                    current_step=current_step,
                    num_training_steps=num_training_steps,
                    warmup_steps=warmup_steps,
                    base_lr=lr,
                    min_lr=min_lr,
                    warmup_start_lr=getattr(self.args, 'warmup_start_lr', 0.0),
                )

                if target_lr is not None:
                    target_lr = max(target_lr, 1e-10)
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for group in self.optimizer.param_groups:
                        group['lr'] = target_lr
                    log_in_local_rank_0(
                        f'Synced optimizer LR at step {current_step}: '
                        f'{old_lr:.6e} -> {target_lr:.6e} (scheduler={scheduler_type})',
                        type='info'
                    )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns = list(set(
                params + self.label_names + [
                    "label",
                    "label_ids",
                    self.label_column,
                    self.loss_mask_column
                ]
            ))


def compute_lr_at_step(scheduler_type, current_step, num_training_steps, warmup_steps,
                       base_lr, min_lr=0.0, warmup_start_lr=0.0):
    """Compute what the LR should be at a given step for any scheduler type."""
    if current_step < warmup_steps:
        if scheduler_type == 'cosine':
            return warmup_start_lr + (base_lr - warmup_start_lr) * current_step / max(1, warmup_steps)
        else:
            return base_lr * current_step / max(1, warmup_steps)

    progress = (current_step - warmup_steps) / max(1, num_training_steps - warmup_steps)

    if scheduler_type == 'cosine':
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_val)
    elif scheduler_type == 'linear':
        return max(min_lr, base_lr * (1 - progress))
    else:
        return base_lr


@dataclass
class MIRATrainingArguments(transformers.TrainingArguments):
    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
    )
    warmup_start_lr: float = field(
        default=0.0, metadata={"help": "Starting learning rate during warmup phase (absolute value)"}
    )


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
        min_lr_ratio: float, warmup_start_lr: float = 0.0,
):
    if current_step < num_warmup_steps:
        # Linear warmup from warmup_start_lr (absolute) to base LR
        return float(warmup_start_lr + (1.0 - warmup_start_lr) * current_step / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)


def get_cosine_schedule_with_warmup_min_lr(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
        warmup_start_lr=warmup_start_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
