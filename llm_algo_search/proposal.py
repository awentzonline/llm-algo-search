import ast
from dataclasses import dataclass
import inspect
from typing import Any

from bs4 import BeautifulSoup


PROPOSAL_TEMPLATE = \
"""
<proposal name="%s">
<thought>%s</thought>
<code>
%s
</code>
</proposal>
"""


@dataclass
class Proposal:
    raw: str
    code: str = ''
    error: str = None
    eval_results: Any = None

    def get_implementation(self):
        return get_generated_class(self.code)

    def get_as_module(self):
        try:
            doc = BeautifulSoup(self.raw, 'lxml')
            name = doc.find('proposal')['name'].strip()
            thought = doc.find('thought').get_text().strip()
            code = doc.find('code').get_text().strip()
            module_text = f'"""\n{name}\n\n{thought}\n\n{self.eval_results}\n"""\n{code}'
            return name, module_text
        except Exception as e:
            raise

    @classmethod
    def parse_raw(cls, raw):
        try:
            doc = BeautifulSoup(raw, 'lxml')
            code = doc.find('code').get_text().strip()
        except Exception as e:
            # parsing error happened
            error = str(e)
            proposal = cls(raw=raw, error=error)
        else:
            proposal = cls(raw=raw, code=code)
        # test that an implementation can be extracted:
        if not proposal.error:
            try:
                proposal.get_implementation()
            except Exception as e:
                proposal.error = str(e)
        return proposal

    @classmethod
    def from_module(cls, module):
        name = module.__name__
        thought = module.__doc__ or ''
        module.__doc__ = ''
        code = inspect.getsource(module)
        module.__doc__ = thought
        raw_doc = PROPOSAL_TEMPLATE % (name.strip(), thought.strip(), code.strip())
        return cls.parse_raw(raw_doc.strip())


def get_generated_class(module_code):
    namespace = {}
    target_class_name = find_first_class_def(module_code)
    exec(module_code, namespace)
    for value in namespace.values():
        # find the first class defined in the string
        if inspect.isclass(value) and value.__name__ == target_class_name:
            return value
    raise ValueError('Class not found in generated module')


def find_first_class_def(module_code):
    tree = ast.parse(module_code)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return None
