from llm_algo_search.evaluation_wrapper import EvaluationWrapper
from llm_algo_search.algorithms.example.context import ExampleAlgoContext

from llm_algo_search.proposal import Proposal


def test_evaluator():
    ac = ExampleAlgoContext()
    ew = EvaluationWrapper({}, ac.get_evaluator())
    seed_modules = ac.get_seed_modules()
    # bad seed
    seed_proposal = Proposal.from_module(seed_modules[0])
    ew.evaluate(seed_proposal)
    assert seed_proposal.eval_results == {'is_correct': False}
    # good seed
    seed_proposal = Proposal.from_module(seed_modules[1])
    ew.evaluate(seed_proposal)
    assert seed_proposal.eval_results == {'is_correct': True}
