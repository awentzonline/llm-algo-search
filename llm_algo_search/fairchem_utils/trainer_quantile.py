from fairchem.core.common.registry import registry
from fairchem.core.trainers.ocp_trainer import OCPTrainer

from .losses import DDPQuantileLoss


@registry.register_trainer('ocp_qs')
class OCPQuantileTrainer(OCPTrainer):
    """
    Changes OCPTrainer to accommodate a model that outputs multiple quantile predictions.
    """

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
                loss_kwargs = loss[target].get('kwargs')
                # This DDPLoss allows the quantile prediction shape
                loss_fn = DDPQuantileLoss(loss_name, reduction=loss_reduction, loss_kwargs=loss_kwargs)

                self.loss_functions.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )
