class Searcher:
    def __init__(self, proposer, evaluation_wrapper):
        self.proposer = proposer
        self.evaluation_wrapper = evaluation_wrapper

    def search(self, max_steps=100, max_errors=3, proposal_history=None):
        proposal_history = proposal_history or []
        num_errors_in_a_row = 0
        for _ in range(max_steps):
            proposal = self.proposer.propose(proposal_history)
            # proposal_history.append(proposal)

            self.evaluation_wrapper.evaluate(proposal)

            yield proposal

            if proposal.error is not None:
                print('Error during evaluation: ', proposal.error)
                num_errors_in_a_row += 1
                if num_errors_in_a_row >= max_errors:
                    print('STOPPING, TOO MANY ERRORS!')
                    break
            else:
                num_errors_in_a_row = 0
