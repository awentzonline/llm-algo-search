import torch
from torch import nn
from tqdm import tqdm


def train_model(model, dataset, atom_rr, cfg):
    model.train()
    f_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch_i in range(cfg.num_epochs):
        for i in tqdm(range(len(dataset))):
            atoms = dataset.get_atoms(i)
            inputs = atom_rr.prepare_inputs(
                torch.from_numpy(atoms.get_atomic_numbers()),
                torch.from_numpy(atoms.get_positions()).float()
            )
            pred_energy = model(inputs)
            targets = torch.FloatTensor([atoms.get_total_energy()])
            loss = f_loss(pred_energy, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {
        'loss': loss.item()
    }


@torch.inference_mode
def eval_model(model, dataset, atom_rr, cfg):
    model.eval()
    f_loss = nn.MSELoss()
    losses = []
    for i in tqdm(range(len(dataset))):
        atoms = dataset.get_atoms(i)
        inputs = atom_rr.prepare_inputs(
            torch.from_numpy(atoms.get_atomic_numbers()),
            torch.from_numpy(atoms.get_positions()).float()
        )
        pred_energy = model(inputs)
        targets = torch.FloatTensor([atoms.get_total_energy()])
        loss = f_loss(pred_energy, targets)
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
