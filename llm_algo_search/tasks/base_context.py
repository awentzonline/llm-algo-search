import importlib
import inspect
import os
import pkgutil
import sys

from llm_algo_search.mixins import TemplateRenderMixin


class BaseAlgoContext(TemplateRenderMixin):
    """

    """
    prompt_template_name = 'base.tmpl'

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        algo_module = inspect.getmodule(self.__class__)
        self.algo_package_name = algo_module.__package__
        self.algo_root_path = os.path.dirname(algo_module.__file__)
        self.api_module_name = self.algo_package_name + '.api'
        self.eval_module_name = self.algo_package_name + '.evaluator'
        self.seed_package_name = self.algo_package_name + '.seeds'
        self.add_filesystem_template_loader(
            os.path.join(self.algo_root_path, 'templates')
        )

    def get_api_module(self):
        module = importlib.import_module(self.api_module_name)
        return module

    def get_evaluator_module(self):
        module = importlib.import_module(self.eval_module_name)
        return module

    def get_evaluator(self):
        module = importlib.import_module(self.eval_module_name)
        for name, value in inspect.getmembers(module):
            if inspect.isclass(value) and value.__module__ == module.__name__:
                return value()
        raise ValueError(f'No evaluator found in `{self.algo_package_name}`')

    def get_seed_modules(self):
        seeds_base_path = os.path.abspath(
            os.path.join(self.algo_root_path, 'seeds')
        )
        seed_modules = [
            importlib.import_module(modinfo.name)
            for modinfo in pkgutil.walk_packages(
                [seeds_base_path], self.seed_package_name + '.'
            )
        ]
        return seed_modules

    def get_additional_context(self):
        return {}

    def render_template(self, filename, **template_kwargs):
        additional_context = self.get_additional_context()
        return super().render_template(filename, **template_kwargs, **additional_context)

    @classmethod
    def get_context_from_package_path(cls, package_path, cfg):
        """Look for an BaseAlgoContext subclass given a package path"""
        context_path = package_path + '.context'
        module = importlib.import_module(context_path)
        for name, value in inspect.getmembers(module):
            if (
                inspect.isclass(value)
                and issubclass(value, BaseAlgoContext)
                and value is not BaseAlgoContext
            ):
                return value(cfg)

        raise ValueError(
            f'No context found for algorithm package `{package_path}`'
        )