from functools import partial

from fairchem.core.datasets import AseDBDataset, data_list_collater
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_algo_search.fairchem_utils.model import ProposedEnergyModel


def train_model(model, cfg):
    model.train()
    model = model.to(cfg.device)
    f_loss = nn.SmoothL1Loss()
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
            pred_energy = model(data)['energy'].squeeze(-1)
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
    f_loss = nn.SmoothL1Loss()

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
            pred_energy = model(data)['energy'].squeeze(-1)
            loss = f_loss(pred_energy, data.energy)
            losses.append(loss)
            pbar.set_postfix({'loss': loss.item()})

    return {
        'mean_loss': torch.mean(torch.FloatTensor(losses)).item()
    }
