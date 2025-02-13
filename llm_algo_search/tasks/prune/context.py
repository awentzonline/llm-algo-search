from llm_algo_search.tasks.base_context import BaseTaskContext


class PruneModelAlgoContext(BaseTaskContext):
    prompt_template_name = "propose_pruner.tmpl"
