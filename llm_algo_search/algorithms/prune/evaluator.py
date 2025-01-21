from itertools import chain

import datasets
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, default_data_collator


class PruneEvaluator:
    def evaluate(self, cfg, prune_cls):
        """
        Evaluate pruner performance on GPT2 using wikitext-2 dataset.
        """
        pruner = prune_cls()

        raw_datasets = datasets.load_dataset(path='wikitext', name='wikitext-2-v1')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())

        def tokenize_function(examples):
            return tokenizer(examples['text'])

        preprocessing_num_workers = 4

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )
        block_size = 128

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
            num_proc=preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        # prune model
        masks = pruner.prune(model, lm_datasets['train'].take(100))
        n_pruned = sum((mask == 0.).sum() for mask in masks.values())

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

        # eval model performancne
        trainer = Trainer(
            model=model,
            eval_dataset=lm_datasets['validation'].take(100),
            processing_class=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        metrics = trainer.evaluate()

        return {
            'percent_pruned_params': (n_pruned / n_params).item() * 100,
            'metrics': metrics,
        }