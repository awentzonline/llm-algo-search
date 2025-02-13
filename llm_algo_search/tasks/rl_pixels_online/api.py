from torch import nn

class API(nn.Module):
    def __init__(self, model_dims, obs_shape, action_dims):
        super().__init__()

    def preprocess_obs(self, obs):
        """Prepare the observation for use. obs is numpy array, dype = int8"""
        pass

    def start_episode(self):
        """
        Useful for initializing agent state
        """
        pass

    def policy(self, obs, reward):
        """
        Returns an action tensor shaped (action_dims,)
        The observation will be a single frame so you are responsible for tracking anything else
        """
        pass
