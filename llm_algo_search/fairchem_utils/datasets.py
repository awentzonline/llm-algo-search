from pathlib import Path

import ase
from fairchem.core.common.registry import registry
from fairchem.core.datasets.ase_datasets import AseAtomsDataset
from matbench_discovery.data import ase_atoms_from_zip


@registry.register_dataset("ase_zip")
class AseAtomsZipDataset(AseAtomsDataset):
    """
    Read ASE Atoms from a zip file. Mostly for matbench-discovery integration.
    """

    def _load_dataset_get_ids(self, config) -> list[int]:
        path = Path(config["src"])
        if not path.is_file():
            raise ValueError(
                f"The specified src is not a file: {self.config['src']}"
            )

        limit = config.get("limit", None)
        self.atoms_list = ase_atoms_from_zip(path, limit=limit)

        if config.get("include_relaxed_energy", False):
            path = Path(config["src_relaxed"])
            if not path.is_file():
                raise ValueError(
                    f"The specified src is not a file: {self.config['src']}"
                )
            self.relaxed_atoms_list = ase_atoms_from_zip(path, limit=limit)

        return list(range(len(self.atoms_list)))

    def get_atoms(self, idx: int) -> ase.Atoms:
        atoms = self.atoms_list[idx]
        return atoms

    def get_relaxed_energy(self, idx: int) -> float:
        atoms = self.relaxed_atoms_list[idx]
        return atoms.get_potential_energy(apply_constraint=False)
