from llm_algo_search.mixins import TemplateRenderMixin
from llm_algo_search.proposal import Proposal


class Proposer:
    def __init__(
        self, llm, context,
        proposal_cls=Proposal
    ):
        super().__init__()
        self.llm = llm
        self.proposal_cls = proposal_cls
        self.context = context

    def propose(self, history):
        prompt = self.context.render_template(
            self.context.prompt_template_name,
            api=self.context.get_api_module(),
            evaluator=self.context.get_evaluator_module(),
            proposal_history=history,
        )
        print('= PROMPT ==================')
        print(prompt)

        response_prefix = '<proposal name="'
        stop_sequence = '</proposal>'
        raw_proposal = self.llm.prompt(
            prompt,
            response_prefix=response_prefix,
            stop_sequences=[stop_sequence]
        ).strip()
        # fix slightly bad generations
        if not raw_proposal.startswith(response_prefix):
            raw_proposal = response_prefix + raw_proposal
        if not raw_proposal.endswith(stop_sequence):
            raw_proposal += stop_sequence

        print('= PROPOSAL =================')
        print(raw_proposal)

        proposal = self.proposal_cls.parse_raw(raw_proposal)
        return proposal
