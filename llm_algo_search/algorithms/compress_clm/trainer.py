from itertools import chain

import datasets
import evaluate
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, default_data_collator
)


def eval_model(cfg, compressor):
    raw_datasets = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config_name)['validation']

    large_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model_conf = AutoConfig.from_pretrained(cfg.model_name)
    small_model_conf = AutoConfig.from_pretrained(cfg.model_name)
    setattr(small_model_conf, cfg.hidden_dim_name, cfg.small_hidden_dims)
    setattr(small_model_conf, cfg.attention_head_name, cfg.small_attention_heads)
    small_model = AutoModelForCausalLM.from_config(model_conf)
    compressor.compress(large_model, small_model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    column_names = list(raw_datasets["train"].features)

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.preprocessing_num_workers,
        desc="Running tokenizer on dataset",
        remove_columns=column_names,
    )
    block_size = cfg.block_size

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=cfg.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation']
    if cfg.take_head:
        train_dataset = train_dataset.take(cfg.take_head)
        eval_dataset = eval_dataset.take(cfg.take_head)

    # eval model performancne
    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_epochs,
        # do_train=True,
        do_eval=True,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        save_strategy=cfg.save_strategy,
    )

    callbacks = []

    trainer = Trainer(
        small_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    # train_result = trainer.train()
    # train_metrics = train_result.metrics
    # print('train metrics', train_metrics)

    eval_metrics = trainer.evaluate()
    print('eval metrics', eval_metrics)

    small_model.cpu()
    large_model.cpu()
    del small_model
    del large_model

    return eval_metrics
