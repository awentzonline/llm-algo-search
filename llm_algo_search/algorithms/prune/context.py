from llm_algo_search.algorithms.base_context import BaseAlgoContext


class PruneModelAlgoContext(BaseAlgoContext):
    prompt_template_name = "propose_pruner.tmpl"
