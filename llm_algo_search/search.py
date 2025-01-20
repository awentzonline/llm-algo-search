import os
import pickle
from typing import Optional

import hydra
from omegaconf import DictConfig

from llm_algo_search.algorithms.base_context import BaseAlgoContext
from llm_algo_search.evaluation_wrapper import EvaluationWrapper
from llm_algo_search.proposal import Proposal
from llm_algo_search.proposer import Proposer
from llm_algo_search.searcher import Searcher


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="search.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    llm = hydra.utils.instantiate(cfg.llm)
    algo_context = BaseAlgoContext.get_context_from_package_path(cfg.algo.package)
    proposer = Proposer(llm=llm, context=algo_context)
    evaluator = algo_context.get_evaluator()
    evaluation_wrapper = EvaluationWrapper(evaluator)

    # load up existing work history or start with seeds
    proposal_history = []
    if cfg.load_existing and os.path.exists(cfg.proposal_history_filename):
        with open(cfg.proposal_history_filename, 'rb') as infile:
            proposal_history = pickle.load(infile)
    else:
        seeds = algo_context.get_seed_modules()
        if seeds:
            for seed in seeds:
                seed_proposal = Proposal.from_module(seed)
                evaluation_wrapper.evaluate(seed_proposal)
                proposal_history.append(seed_proposal)

    searcher = Searcher(proposer, evaluation_wrapper)
    proposals = searcher.search(
        max_steps=cfg.max_steps, max_errors=cfg.max_errors,
        proposal_history=proposal_history,
    )

    with open(cfg.algo.proposal_history_filename, 'wb') as outfile:
        pickle.dump(proposals, outfile)


if __name__ == "__main__":
    main()
