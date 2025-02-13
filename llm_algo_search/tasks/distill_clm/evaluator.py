import time

from .trainer import distill_model


class DistillCLMEvaluator:
    def evaluate(self, cfg, distill_cls):
        """
        Evaluate model distillation losses
        `distill_cls` your implementation to be evaluated
        """
        # t0 = time.time()
        distiller = distill_cls()
        eval_metrics = distill_model(cfg, distiller)
        # eval_metrics['time'] = int(time.time() - t0)

        return eval_metrics
