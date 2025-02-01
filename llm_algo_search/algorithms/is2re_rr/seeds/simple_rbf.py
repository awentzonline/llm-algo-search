from torch import nn
import torch

class API(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.embed_dim = 64  # Embedding dimension for atomic numbers
        self.num_rbf = 16    # Number of radial basis functions

        # Embedding layer for atomic numbers (assuming max atomic number <= 128)
        self.embed = nn.Embedding(128, self.embed_dim)

        # Learnable gamma parameters for RBF
        self.gamma = nn.Parameter(torch.linspace(0.1, 10.0, self.num_rbf))

        # Linear layer to project concatenated features to model_dims
        in_features = self.embed_dim + self.embed_dim * self.num_rbf
        self.transform = nn.Linear(in_features, model_dims)

    def prepare_inputs(self, atomic_numbers, positions):
        # Embed atomic numbers
        E = self.embed(atomic_numbers)  # (N, embed_dim)

        # Center positions to ensure translation invariance
        positions_centered = positions - positions.mean(dim=0, keepdim=True)

        # Compute pairwise differences and squared distances
        pos_diff = positions_centered[:, None, :] - positions_centered[None, :, :]  # (N, N, 3)
        dist_sq = (pos_diff ** 2).sum(dim=-1)  # (N, N)

        # Expand distances into RBF features
        gamma = self.gamma  # (num_rbf,)
        rbf = torch.exp(-gamma.view(1, 1, -1) * dist_sq.unsqueeze(-1))  # (N, N, num_rbf)

        # Mask diagonal (i=j) to exclude self-interactions
        mask = torch.eye(atomic_numbers.size(0), dtype=torch.bool, device=atomic_numbers.device)
        rbf = rbf.masked_fill(mask.unsqueeze(-1), 0.0)  # (N, N, num_rbf)

        # Compute neighbor features by summing over RBF-weighted embeddings
        neighbor_features = torch.einsum('ijk,jd->ikd', rbf, E)  # (N, num_rbf, embed_dim)
        neighbor_features = neighbor_features.reshape(neighbor_features.size(0), -1)  # (N, num_rbf * embed_dim)

        # Concatenate with own embeddings and transform
        atom_features = torch.cat([E, neighbor_features], dim=-1)  # (N, embed_dim + num_rbf * embed_dim)
        atom_features = self.transform(atom_features)  # (N, model_dims)

        # Sum over all atoms to get the structure representation
        structure_feature = atom_features.sum(dim=0)  # (model_dims)

        return structure_feature