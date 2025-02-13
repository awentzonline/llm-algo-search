import os
import pickle
from typing import Optional

import hydra
from omegaconf import DictConfig

from llm_algo_search.tasks.base_context import BaseTaskContext
from llm_algo_search.evaluation_wrapper import EvaluationWrapper
from llm_algo_search.proposal import Proposal
from llm_algo_search.proposer import Proposer
from llm_algo_search.searcher import Searcher


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="search.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    llm = hydra.utils.instantiate(cfg.llm)
    algo_context = BaseTaskContext.get_context_from_package_path(cfg.algo.package, cfg)
    proposer = Proposer(llm=llm, context=algo_context)
    evaluator = algo_context.get_evaluator()
    evaluation_wrapper = EvaluationWrapper(cfg.algo, evaluator)

    # load up existing work history or start with seeds
    proposal_history = []
    if cfg.load_existing and os.path.exists(cfg.algo.proposal_history_filename):
        with open(cfg.algo.proposal_history_filename, 'rb') as infile:
            proposal_history = pickle.load(infile)
    else:
        seeds = algo_context.get_seed_modules()
        if seeds and cfg.use_seeds:
            for seed in seeds:
                seed_proposal = Proposal.from_module(seed)
                evaluation_wrapper.evaluate(seed_proposal)
                proposal_history.append(seed_proposal)

    searcher = Searcher(proposer, evaluation_wrapper)
    try:
        for proposal in searcher.search(
            max_steps=cfg.max_steps, max_errors=cfg.max_errors,
            seed_proposals=proposal_history
        ):
            proposal_history.append(proposal)
    except KeyboardInterrupt:
        print('Stopping search...')

    with open(cfg.algo.proposal_history_filename, 'wb') as outfile:
        pickle.dump(proposal_history, outfile)


if __name__ == "__main__":
    main()
