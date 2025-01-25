from collections import deque
import time

import numpy as np
from transformers.trainer_callback import TrainerCallback


class TrainingStepTooSlow(Exception):
    pass


class StopSlowTrainingCallback(TrainerCallback):
    """
    Raise an exception if the model training is too slow.
    """
    def __init__(self, max_step_time=2., num_samples=5):
        self.max_step_time = max_step_time
        self.samples = deque(maxlen=num_samples)

    def on_step_begin(self, *args, **kwargs):
        self.start_step_t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_t = time.time() - self.start_step_t
        self.samples.append(elapsed_t)
        # wait until there are enough samples
        if len(self.samples) == self.samples.maxlen:
            if np.mean(self.samples) >= self.max_step_time:
                raise TrainingStepTooSlow('Training step is too slow')
