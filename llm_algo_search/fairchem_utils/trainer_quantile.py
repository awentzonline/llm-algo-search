import logging

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.trainers.ocp_trainer import OCPTrainer
import torch

from .losses import DDPQuantileLoss


@registry.register_trainer('ocp_qs')
class OCPQuantileTrainer(OCPTrainer):
    """
    Changes OCPTrainer to accommodate a model that outputs multiple quantile predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name='ocp_qs', **kwargs)

    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        if metrics is None:
            metrics = {}
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        # reduce the quantiles for the metric calculations
        out['energy'] = out['energy'].mean(-1, keepdim=True)

        return super()._compute_metrics(out, batch, evaluator, metrics)

    def load_loss(self) -> None:
        self.loss_functions = []
        for _idx, loss in enumerate(self.config["loss_functions"]):
            for target in loss:
                assert (
                    "fn" in loss[target]
                ), f"'fn' is not defined in the {target} loss config {loss[target]}."
                loss_name = loss[target].get("fn")
                assert (
                    "coefficient" in loss[target]
                ), f"'coefficient' is not defined in the {target} loss config {loss[target]}."
                coefficient = loss[target].get("coefficient")
                loss_reduction = loss[target].get("reduction")

                # This DDPLoss allows the quantile prediction shape
                loss_fn = DDPQuantileLoss(loss_name, reduction=loss_reduction)

                self.loss_functions.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get(
            # Fix for custom trainer:
            # "primary_metric", self.evaluator.task_primary_metric[self.name]
            "primary_metric", self.evaluator.task_primary_metric.get(self.name)
        )
        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(epoch_int, skip_steps)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)
                # Forward, loss, backward.
                with torch.autocast("cuda", enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update("loss", loss.item(), self.metrics)

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [f"{k}: {v:.2e}" for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)
