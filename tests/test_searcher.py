from llm_algo_search.algorithms.example.context import ExampleAlgoContext
import llm_algo_search.algorithms.example.seeds.good_example
from llm_algo_search.evaluation_wrapper import EvaluationWrapper
from llm_algo_search.llms.mock import MockLLM
from llm_algo_search.proposal import Proposal
from llm_algo_search.proposer import Proposer
from llm_algo_search.searcher import Searcher


def test_searcher():
    proposal = Proposal.from_module(llm_algo_search.algorithms.example.seeds.good_example)
    llm = MockLLM(response=proposal.raw)
    algo_context = ExampleAlgoContext()
    proposer = Proposer(llm=llm, context=algo_context)
    evaluator = algo_context.get_evaluator()
    evaluation_wrapper = EvaluationWrapper({}, evaluator)
    searcher = Searcher(proposer, evaluation_wrapper)
    proposals = [
        p for p in searcher.search(
            max_steps=1, max_errors=1
        )
    ]
    assert len(proposals) == 1
    assert isinstance(proposals[0], Proposal)
    assert proposals[0].eval_results == {'is_correct': True}

