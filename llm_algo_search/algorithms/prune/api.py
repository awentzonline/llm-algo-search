class API:
    def prune_model(self, model, dataset) -> dict:
        """
        Determine which parameters should be removed, update the model,
        and return a Dict[str, Tensor] corresponding to masks over model parameters
        """