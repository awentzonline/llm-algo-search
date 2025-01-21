LLM Algorithm Search
====================
Search for algorithm implementations using LLMs.

Usage
-----
Algorithm packages use configuration by convention.

* your_algorithm/
  * api.py - stub API to be implemented
  * context.py - contains a subclass of BaseAlgoContext which you can optionally configure things
  * evaluator.py - class that has an `evaluate` method which takes an API implementation as input
  * seeds/ - a directory where you can place API implementations to be evaluated at the beginning
  * templates/ - jinja2 templates if you'd like to extend or override the base prompt template

Add an algorithm config to `configs/algo/your_algorithm.yaml` which contains

```
package: llm_algo_search.algorithms.your_algorithm
proposal_history_filename: proposal_history_your_name.pkl
```

Run via: `python -m llm_algo_search.search algo=your_algorithm`

Example
-------
The example can be found in `llm_algo_search/algorithms/example`

`api.py`:
```
class ExampleAPI:
    def foo(self):
        pass
```

`evaluator.py`:
```
class ExampleEvaluator:
    def evaluate(self, cfg, impl_cls):
        impl = impl_cls()
        is_correct = impl.foo() in ('bar', 'baz')
        return {'is_correct': is_correct}
```

Using Anthropic's Claude Sonnet 3.5 as LLM, the first non-seed implementation is:
```
<proposal name="llm_algo_search.algorithms.example.random_solution">
<thought>
Since we've confirmed both 'bar' and 'baz' are valid outputs, let's try implementing a solution that randomly chooses between the two valid options. This approach could be useful in scenarios where varying the output between valid options is beneficial, such as load balancing or avoiding predictable patterns.
</thought>
<code>
"""Implementation that randomly selects between valid outputs"""

import random


class ExampleAPI:
    VALID_OUTPUTS = ('bar', 'baz')

    def foo(self):
        return random.choice(self.VALID_OUTPUTS)
</code></proposal>
```

And after 5 steps:
```
<proposal name="llm_algo_search.algorithms.example.memoized_fibonacci">
<thought>
Let's try an approach inspired by mathematical sequences. We can use the Fibonacci sequence to determine which output to return, memoizing the results for efficiency. This introduces a mathematical pattern to our output selection while maintaining valid results. The implementation will use the memory-efficient O(1) space Fibonacci calculation to determine output, adding complexity while ensuring correctness.
</thought>
<code>
"""Implementation using Fibonacci sequence to determine output"""


class ExampleImpl:
    def __init__(self):
        self._calls = 0
        self._prev = 1
        self._curr = 1

    def _next_fib(self):
        next_val = self._prev + self._curr
        self._prev = self._curr
        self._curr = next_val
        return next_val

    def foo(self):
        self._calls += 1
        fib_num = self._next_fib()
        return 'bar' if fib_num % 2 == 0 else 'baz'
</code></proposal>
```
