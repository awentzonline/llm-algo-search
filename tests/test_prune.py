from llm_algo_search.evaluation_wrapper import EvaluationWrapper
from llm_algo_search.tasks.prune.context import PruneModelAlgoContext

from llm_algo_search.proposal import Proposal


def test_evaluator():
    cfg = {}
    ac = PruneModelAlgoContext(cfg)
    ew = EvaluationWrapper(cfg, ac.get_evaluator())
    seed_modules = ac.get_seed_modules()
    assert len(seed_modules) == 1
    seed_proposal = Proposal.from_module(seed_modules[0])
    ew.evaluate(seed_proposal)

