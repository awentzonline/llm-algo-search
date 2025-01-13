import inspect

from jinja2 import (
    ChoiceLoader, Environment, FileSystemLoader, PackageLoader, select_autoescape
)


class TemplateRenderMixin:
    def __init__(self):
        self.template_loaders = ChoiceLoader([PackageLoader("llm_algo_search")])
        self.env = Environment(
            loader=self.template_loaders,
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters['getsource'] = inspect.getsource

    def add_filesystem_template_loader(self, path):
        self.template_loaders.loaders.append(FileSystemLoader(path))

    def render_template(self, filename, **template_kwargs):
        template = self.env.get_template(filename)
        rendered = template.render(**template_kwargs)
        return rendered
