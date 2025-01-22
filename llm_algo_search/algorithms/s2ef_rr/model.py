from fairchem.core.datasets import AseDBDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AtomsReducedRepresentationDataset(Dataset):
    def __init__(self, base_dataset, atom_rr):
        self.base_dataset = base_dataset
        self.atom_rr = atom_rr

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        reduced = self.atom_rr.prepare_inputs(
            item.pos.float(),
            item.atomic_numbers
        )
        return reduced, torch.FloatTensor([item.energy])


def train_model(model, atom_rr, cfg):
    model.train()
    f_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = AseDBDataset(config=dict(src=cfg.train_dataset_path, a2g_args=dict(r_energy=True)))
    vec_dataset = AtomsReducedRepresentationDataset(dataset, atom_rr)
    dataloader = DataLoader(vec_dataset, batch_size=cfg.batch_size)

    for epoch_i in range(cfg.num_epochs):
        for inputs, target_energy in tqdm(dataloader):
            pred_energy = model(inputs)
            loss = f_loss(pred_energy, target_energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        'loss': loss.item()
    }


@torch.inference_mode
def eval_model(model, atom_rr, cfg):
    model.eval()
    f_loss = nn.MSELoss()

    dataset = AseDBDataset(config=dict(src=cfg.val_dataset_path, a2g_args=dict(r_energy=True)))
    vec_dataset = AtomsReducedRepresentationDataset(dataset, atom_rr)
    dataloader = DataLoader(vec_dataset, batch_size=cfg.batch_size)

    losses = []
    for epoch_i in range(cfg.num_epochs):
        for inputs, target_energy in tqdm(dataloader):
            pred_energy = model(inputs)
            loss = f_loss(pred_energy, target_energy)
            losses.append(loss)

    return {
        'mean_loss': torch.mean(torch.FloatTensor(losses)).item()
    }


class EnergyModel(nn.Module):
    def __init__(self, model_dims, output_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dims, output_dims),
        )

    def forward(self, x):
        y = self.net(x)
        return y
