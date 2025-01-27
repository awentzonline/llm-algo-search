import pickle

from fairchem.core.common.registry import registry
import numpy as np
import torch
from torch import nn


@registry.register_model('parr')
class ProposedEnergyModel(nn.Module):
    def __init__(self, model_dims, output_dims=1, atom_rr=None, proposal_path=None, **kwargs):
        super().__init__()
        if atom_rr is None:
            atom_rr_cls = self._get_best_implementation(proposal_path)
            self.atom_rr = atom_rr_cls(model_dims)
        else:
            self.atom_rr = atom_rr

        self.net = nn.Sequential(
            nn.Linear(model_dims, output_dims),
        )

    def _get_best_implementation(self, proposal_path):
        with open(proposal_path, 'rb') as infile:
            proposals = pickle.load(infile)

        losses = []
        for i, prop in enumerate(proposals):
            if prop.eval_results is None:
                best_loss = np.inf
            else:
                best_loss = np.min([
                    e['mean_loss'] for e in prop.eval_results['mean_losses']
                ])
            losses.append(best_loss)

        loss_inds = np.argsort(losses)
        best_proposal = proposals[loss_inds[0]]
        atom_rr_cls = best_proposal.get_implementation()
        return atom_rr_cls

    def reduce_inputs(self, data):
        """
        Due to variation of number of atoms in each system, the inputs are
        concatenated together on the same axis and indexed with the .batch tensor.
        """
        reduced_inputs = []
        last_batch_id = 0
        offset_i = 0
        atomic_numbers = data.atomic_numbers.long()
        for i, batch_id in enumerate(data.batch):
            if batch_id != last_batch_id:
                reduced = self.atom_rr.prepare_inputs(
                    atomic_numbers[offset_i:i],
                    data.pos[offset_i:i],
                )
                reduced_inputs.append(reduced)
                offset_i = i
                last_batch_id = batch_id
        reduced_inputs.append(
            self.atom_rr.prepare_inputs(
                atomic_numbers[offset_i:],
                data.pos[offset_i:],
            )
        )
        reduced_inputs = torch.stack(reduced_inputs)
        return reduced_inputs

    def forward(self, x):
        reduced = self.reduce_inputs(x)
        y = self.net(reduced)
        return dict(energy=y)
