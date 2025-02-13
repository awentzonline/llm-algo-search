import time

from .trainer import train_model


class DistillCLMEvaluator:
    def evaluate(self, cfg, reward_cls):
        """
        Evaluate reward functions losses
        `reward_cls` your rewards implementation to be evaluated
        """
        # t0 = time.time()
        reward_funcs = reward_cls()
        eval_metrics = train_model(cfg, reward_funcs)
        # eval_metrics['time'] = int(time.time() - t0)

        return eval_metrics
