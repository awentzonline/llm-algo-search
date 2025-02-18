from ase import Atoms
from fairchem.core import OCPCalculator
from fairchem.core.datasets import data_list_collater


class QuantileOCPCalculator(OCPCalculator):
    def __init__(self, *args, quantile=None, **kwargs):
        self.quantile = quantile
        trainer = kwargs.pop('trainer', 'ocp_qs')
        super().__init__(*args, trainer=trainer, **kwargs)

    def calculate(self, atoms, properties, system_changes) -> None:
        """Calculate implemented properties for a single Atoms object or a Batch of them."""
        super().calculate(atoms, properties, system_changes)
        if isinstance(atoms, Atoms):
            data_object = self.a2g.convert(atoms)
            batch = data_list_collater([data_object], otf_graph=True)
        else:
            batch = atoms

        predictions = self.trainer.predict(batch, per_image=False, disable_tqdm=True)

        for key in predictions:
            _pred = predictions[key]
            if self.quantile is None:
                _pred = _pred.mean(-1)
            else:
                _pred = _pred[self.quantile]
            _pred = _pred.item() if _pred.numel() == 1 else _pred.cpu().numpy()
            if key in OCPCalculator._reshaped_props:
                _pred = _pred.reshape(OCPCalculator._reshaped_props.get(key)).squeeze()
            self.results[key] = _pred