import inspect

import pytest

import llm_algo_search.algorithms.example.seeds.good_example
from llm_algo_search.proposal import Proposal


class SampleImpl:
    @classmethod
    def foo(cls):
        return 'bar'


SAMPLE_RAW_PROPOSAL = \
"""
<proposal name="simple example">
<thought>
Example of the API. Feel free to try this as a starting point.
</thought>
<code>
%s
</code>
</proposal>
""" % (inspect.getsource(SampleImpl),)


@pytest.fixture
def sample_proposal():
    return Proposal.parse_raw(SAMPLE_RAW_PROPOSAL)


def test_proposal_from_string(sample_proposal):
    impl_cls = sample_proposal.get_implementation()
    assert inspect.isclass(impl_cls)
    impl = impl_cls()
    assert impl.foo() == 'bar'


def test_proposal_from_module():
    proposal = Proposal.from_module(llm_algo_search.algorithms.example.seeds.good_example)
    impl_cls = proposal.get_implementation()
    assert inspect.isclass(impl_cls)
    impl = impl_cls()
    assert impl.foo() == 'bar'
