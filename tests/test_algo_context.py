from llm_algo_search.algorithms.base_context import BaseAlgoContext
import llm_algo_search.algorithms.example.api
from llm_algo_search.algorithms.example.context import ExampleAlgoContext
import llm_algo_search.algorithms.example.evaluator
from llm_algo_search.algorithms.example.evaluator import ExampleEvaluator
import llm_algo_search.algorithms.example.seeds.bad_example
import llm_algo_search.algorithms.example.seeds.good_example


def test_example_algo_context():
    conf = ExampleAlgoContext()

    seeds = conf.get_seed_modules()
    print(seeds)

    assert len(seeds) == 2
    assert seeds[0] is llm_algo_search.algorithms.example.seeds.bad_example
    assert seeds[1] is llm_algo_search.algorithms.example.seeds.good_example

    api_module = conf.get_api_module()
    assert api_module is llm_algo_search.algorithms.example.api

    evaluator_module = conf.get_evaluator_module()
    assert evaluator_module is llm_algo_search.algorithms.example.evaluator

    evaluator = conf.get_evaluator()
    assert isinstance(evaluator, ExampleEvaluator)

    context = BaseAlgoContext.get_context_from_package_path('llm_algo_search.algorithms.example')
    assert isinstance(context, ExampleAlgoContext)
