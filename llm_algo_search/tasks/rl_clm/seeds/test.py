class API:
    def get_reward_funcs(self):
        """
        Return a list of reward functions. Do not aggregate them.
        Each reward function should have the following inputs:

        def example(self, completions, prompts, **kwargs)
            return [0. for x in completions]

        and return list of scalar rewards, one per completion
        """
        return [self.length_reward]

    def length_reward(self, completions, **kwargs):
        return [float(len(c)) for c in completions]
