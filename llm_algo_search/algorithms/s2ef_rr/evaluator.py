from fairchem.core.datasets import AseDBDataset
import numpy as np
import torch
from torch import nn

from .model import EnergyModel, eval_model, train_model


class S2EFRREvaluator:
    def evaluate(self, cfg, atom_rr_cls):
        """
        Evaluate atomistic reduced representations for modeling total energy.
        `atom_rr_cls` your implementation to be evaluated
        """
        all_model_dims = (128, 256, 512, 1024)

        train_dataset = AseDBDataset(config=dict(src=cfg.train_dataset_path))
        val_dataset = AseDBDataset(config=dict(src=cfg.val_dataset_path))

        all_metrics = []
        for model_dims in all_model_dims:
            print('training model size =', model_dims)
            atom_rr = atom_rr_cls(model_dims)
            model = EnergyModel(model_dims, 1)

            metrics = train_model(model, train_dataset, atom_rr, cfg)
            eval_metrics = eval_model(model, val_dataset, atom_rr, cfg)
            all_metrics.append(eval_metrics)

        return {
            'model_dims': all_model_dims,
            'mean_losses': all_metrics,
        }
