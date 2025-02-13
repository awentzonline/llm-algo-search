class API:
    def __init__(self, num_positions, space_dims, bundle_dims):
        pass

    def bundle(self, xs):
        """
        xs.shape = (num_vecs, space_dims)
        output.shape = (vec_dims,)
        """
        pass

    def unbundle(self, zs):
        """
        zs.shape = (vec_dims,)
        output.shape = (num_vecs, space_dims)
        """
        pass
