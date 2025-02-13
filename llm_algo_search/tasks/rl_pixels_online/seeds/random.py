"""Random actions"""
import torch
from torch import nn

class API(nn.Module):
    def __init__(self, obs_dims, model_dims, action_dims):
        super().__init__()
        self.action_dims = action_dims

    def preprocess_obs(self, obs):
        return torch.from_numpy(obs).float() / 255.

    def forward(self, obs, reward):
        pass

    def start_episode(self):
        pass

    def policy(self, obs, reward):
        return torch.randint(0, self.action_dims, (1,)).detach().cpu().item()
