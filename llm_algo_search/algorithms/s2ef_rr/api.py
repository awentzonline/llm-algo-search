class API:
    def __init__(self, model_dims):
        pass

    def prepare_inputs(self, atomic_numbers, positions):
        """
        atomic_numbers.shape = (num_atoms,) LongTensor
        positions.shape = (num_atoms, 3) FloatTensor
        Output a single vector of shape (self.model_dims,)
        """
        pass
