class ExampleEvaluator:
    def evaluate(self, cfg, impl_cls):
        impl = impl_cls()
        is_correct = impl.foo() in ('bar', 'baz')
        return {'is_correct': is_correct}
