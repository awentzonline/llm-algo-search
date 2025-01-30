# These imports register our custom components with fairchem
from llm_algo_search.fairchem_utils.model import ProposedEnergyModel
from llm_algo_search.fairchem_utils.losses import QuantileLoss, QuantileHuberLoss
from llm_algo_search.fairchem_utils.trainer_quantile import OCPQuantileTrainer


if __name__ == "__main__":
    from fairchem.core._cli import main

    main()