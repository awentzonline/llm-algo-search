import time

from .model import ProposedEnergyModel, eval_model, train_model


class S2EFRREvaluator:
    def evaluate(self, cfg, atom_rr_cls):
        """
        Evaluate atomistic reduced representations for modeling total energy.
        `atom_rr_cls` your implementation to be evaluated
        """
        all_metrics = []
        for model_dims in cfg.all_model_dims:
            print('training model size =', model_dims)
            t0 = time.time()
            atom_rr = atom_rr_cls(model_dims)
            model = ProposedEnergyModel(model_dims, 1, atom_rr)
            metrics = train_model(model, cfg)
            eval_metrics = eval_model(model, cfg)
            eval_metrics['time'] = int(time.time() - t0)
            all_metrics.append(eval_metrics)

        return {
            'model_dims': cfg.all_model_dims,
            'mean_losses': all_metrics,
        }
