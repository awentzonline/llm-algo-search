"""
1. Use hierarchical interaction features without complicated gating
2. Implement stable distance-based kernels
3. Create multi-level message passing
4. Use careful normalization and residual connections
This should maintain numerical stability while capturing relevant physical interactions.
"""
from torch import nn
import torch

class API(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.model_dims = model_dims
        self.hidden_dim = model_dims // 2
        self.num_levels = 3

        # Atomic embeddings
        self.atom_embedding = nn.Embedding(100, self.hidden_dim)

        # Distance embedding
        self.distance_expansion = nn.Sequential(
            nn.Linear(self.num_levels, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # Message networks for each level
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(self.hidden_dim)
            ) for _ in range(self.num_levels)
        ])

        # Update networks
        self.update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(self.hidden_dim)
            ) for _ in range(self.num_levels)
        ])

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_levels, self.model_dims),
            nn.LayerNorm(self.model_dims)
        )

    def get_distance_scales(self, distances):
        # Create multiple distance representations
        scales = torch.stack([
            torch.exp(-distances),
            torch.exp(-distances/2),
            torch.exp(-distances/4)
        ], dim=-1)
        return scales

    def prepare_inputs(self, atomic_numbers, positions):
        N = positions.shape[0]

        # Initial node features
        node_features = self.atom_embedding(atomic_numbers)  # (N, H)

        # Compute distances
        diff_vec = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        distances = torch.norm(diff_vec, dim=-1)  # (N, N)

        # Get multi-scale distance features
        distance_scales = self.get_distance_scales(distances)  # (N, N, num_levels)
        edge_weights = self.distance_expansion(distance_scales)  # (N, N, H)

        # Initialize collection of features at different levels
        level_features = []
        current_features = node_features

        # Process through each level
        for level in range(self.num_levels):
            # Compute messages
            messages = torch.cat([
                current_features.unsqueeze(1).expand(-1, N, -1),
                current_features.unsqueeze(0).expand(N, -1, -1)
            ], dim=-1)  # (N, N, 2H)

            messages = self.message_nets[level](messages)  # (N, N, H)

            # Weight messages by distance at this level
            weighted_messages = messages * edge_weights
            message_sum = weighted_messages.sum(dim=1)  # (N, H)

            # Update node features
            update_input = torch.cat([
                current_features,
                message_sum
            ], dim=-1)  # (N, 2H)

            current_features = current_features + self.update_nets[level](update_input)

            # Pool features at this level
            level_features.append(current_features.mean(dim=0))  # (H,)

        # Combine features from all levels
        combined_features = torch.cat(level_features, dim=0)  # (num_levels * H,)

        return self.final_proj(combined_features)  # (D,)
