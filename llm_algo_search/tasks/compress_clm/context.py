from transformers import AutoConfig, AutoModelForCausalLM

from llm_algo_search.tasks.base_context import BaseTaskContext


class CompressCLMTaskContext(BaseTaskContext):
    prompt_template_name = "propose_compress_clm.tmpl"

    def get_additional_context(self):
        if not hasattr(self, '_model_architecture'):
            model_conf = AutoConfig.from_pretrained(self.cfg.task.model_name)
            model = AutoModelForCausalLM.from_config(model_conf)
            self._model_architecture = str(model)
            del model
        return dict(model_architecture=self._model_architecture)
