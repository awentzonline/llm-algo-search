from torch import nn

class API(nn.Module):
    def __init__(self, model_dims, obs_dims, action_dims):
        super().__init__()

    def forward(self, obs, reward):
        """
        Use for batch-wise operation in learning, if you want
        """
        pass

    def start_episode(self):
        """
        Useful for initializing agent state
        """
        pass

    def policy(self, obs, reward):
        """
        Returns an action tensor shaped (action_dims,)
        """
        pass

    def train_on_episode(self, obs, actions, rewards):
        """
        Inputs are lists of tensors
        """
        pass
