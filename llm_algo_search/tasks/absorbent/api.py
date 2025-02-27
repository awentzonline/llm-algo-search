from typing import List
from ase import Atom, Atoms


class API:
    def get_adsorbent(self) -> Atoms:
        pass

    def get_adsorbates(self) -> List[Atom|Atoms]:
        pass
