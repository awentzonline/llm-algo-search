class MockLLM:
    def __init__(self, *args, response='', **kwargs):
        self.response = response

    def prompt(self, prompt, *args, **kwargs):
        return self.response
