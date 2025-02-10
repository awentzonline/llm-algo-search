import time

from .trainer import eval_model


class CompressCLMEvaluator:
    def evaluate(self, cfg, compress_cls):
        """
        Evaluate model distillation losses
        `distill_cls` your implementation to be evaluated
        """
        # t0 = time.time()
        compressor = compress_cls()
        eval_metrics = eval_model(cfg, compressor)
        # eval_metrics['time'] = int(time.time() - t0)

        return eval_metrics
