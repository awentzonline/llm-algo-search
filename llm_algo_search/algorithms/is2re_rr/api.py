from torch import nn

class API(nn.Module):
    def __init__(self, model_dims):
        super().__init__()

    def prepare_inputs(self, atomic_numbers, positions):
        """
        atomic_numbers.shape = (num_atoms,) LongTensor
        positions.shape = (num_atoms, 3) FloatTensor
        Output a single vector of shape (self.model_dims,)
        """
        pass

    def update_inputs(self, hidden_vecs):
        """
        hidden_vecs is the output of prepare_inputs
        hidden_vecs.shape = (batch_size, hidden_dims)
        Output a tensor of shape (batch_size, hidden_dims)
        """
        pass
