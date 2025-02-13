from itertools import chain

import datasets
import evaluate
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer, get_peft_config

from llm_algo_search.hfutils.trainer_callbacks import StopSlowTrainingCallback


def train_model(cfg, reward_funcs):
    lm_datasets = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config_name)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    peft_config = get_peft_config(cfg.peft_config)

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation']
    if cfg.take_head:
        train_dataset = train_dataset.take(cfg.take_head)
        eval_dataset = eval_dataset.take(cfg.take_head)

    # eval model performancne
    train_args = GRPOConfig(
        do_train=True,
        do_eval=True,
        **cfg.train_args,
    )

    callbacks = []
    if cfg.max_step_time:
        callbacks.append(StopSlowTrainingCallback(cfg.max_step_time))

    trainer = GRPOTrainer(
        model,
        reward_funcs=reward_funcs.get_reward_funcs(),
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    print('train metrics', train_metrics)

    eval_metrics = trainer.evaluate()
    print('eval metrics', eval_metrics)

    model.cpu()
    del model

    return eval_metrics
