import linecache
import traceback


class EvaluationWrapper:
    def __init__(self, cfg, evaluator):
        self.cfg = cfg
        self.evaluator = evaluator

    def evaluate(self, proposal):
        try:
            eval_results = self.evaluator.evaluate(
                self.cfg,
                proposal.get_implementation()
            )
            proposal.eval_results = eval_results
        except KeyboardInterrupt:
            proposal.error = 'Attempt aborted for excessive time'
        except Exception as e:
            tb = e.__traceback__
            tb_lines = []
            extracted_tb = traceback.extract_tb(tb)
            for frame in extracted_tb:
                if frame.filename == '<string>':
                    code_line = proposal.code.split('\n')[frame.lineno - 1].strip()
                    tb_lines.append(f'In your code, line {frame.lineno}: {code_line}')
                    break
                else:
                    code_line = linecache.getline(frame.filename, frame.lineno).strip()
                    tb_lines.append(f'In external file: {code_line}')
            error_message = str(e) + '\n' + '\n'.join(tb_lines)
            proposal.error = error_message
