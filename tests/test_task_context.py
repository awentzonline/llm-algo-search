from llm_algo_search.tasks.base_context import BaseTaskContext
import llm_algo_search.tasks.example.api
from llm_algo_search.tasks.example.context import ExampleTaskContext
import llm_algo_search.tasks.example.evaluator
from llm_algo_search.tasks.example.evaluator import ExampleEvaluator
import llm_algo_search.tasks.example.seeds.bad_example
import llm_algo_search.tasks.example.seeds.good_example


def test_example_algo_context():
    cfg = {}
    conf = ExampleTaskContext(cfg)

    seeds = conf.get_seed_modules()
    print(seeds)

    assert len(seeds) == 2
    assert seeds[0] is llm_algo_search.tasks.example.seeds.bad_example
    assert seeds[1] is llm_algo_search.tasks.example.seeds.good_example

    api_module = conf.get_api_module()
    assert api_module is llm_algo_search.tasks.example.api

    evaluator_module = conf.get_evaluator_module()
    assert evaluator_module is llm_algo_search.tasks.example.evaluator

    evaluator = conf.get_evaluator()
    assert isinstance(evaluator, ExampleEvaluator)

    context = BaseTaskContext.get_context_from_package_path('llm_algo_search.tasks.example', cfg)
    assert isinstance(context, ExampleTaskContext)
