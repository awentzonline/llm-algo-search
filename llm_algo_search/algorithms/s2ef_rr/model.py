from functools import partial

from fairchem.core.datasets import AseDBDataset, data_list_collater
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, cfg):
    model.train()
    model = model.to(cfg.device)
    f_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = AseDBDataset(config=dict(src=cfg.train_dataset_path, a2g_args=dict(r_energy=True)))
    collater = partial(
        data_list_collater, otf_graph=cfg.get("model", {}).get("otf_graph", True)
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collater)

    for epoch_i in range(cfg.num_epochs):
        pbar = tqdm(dataloader)
        for batch_i, data in enumerate(pbar):
            if cfg.max_train_batches and cfg.max_train_batches <= batch_i:
                break
            data = data.to(cfg.device)
            pred_energy = model(data).squeeze(-1)
            loss = f_loss(pred_energy, data.energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})

    return {
        'loss': loss.item()
    }


@torch.inference_mode
def eval_model(model, cfg):
    model.eval()
    model = model.to(cfg.device)
    f_loss = nn.MSELoss()

    dataset = AseDBDataset(config=dict(src=cfg.val_dataset_path, a2g_args=dict(r_energy=True)))
    collater = partial(
        data_list_collater, otf_graph=cfg.get("model", {}).get("otf_graph", True)
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collater)

    losses = []
    for epoch_i in range(cfg.num_epochs):
        pbar = tqdm(dataloader)
        for batch_i, data in enumerate(pbar):
            if cfg.max_eval_batches and cfg.max_eval_batches <= batch_i:
                break
            data = data.to(cfg.device)
            pred_energy = model(data).squeeze(-1)
            loss = f_loss(pred_energy, data.energy)
            losses.append(loss)
            pbar.set_postfix({'loss': loss.item()})

    return {
        'mean_loss': torch.mean(torch.FloatTensor(losses)).item()
    }


class ProposedEnergyModel(nn.Module):
    def __init__(self, model_dims, output_dims, atom_rr):
        super().__init__()
        self.atom_rr = atom_rr
        self.net = nn.Sequential(
            nn.Linear(model_dims, output_dims),
        )

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
        return y
