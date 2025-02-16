import os
import pickle
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig

from llm_algo_search.tasks.base_context import BaseTaskContext
from llm_algo_search.proposal import Proposal


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="search.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    algo_context = BaseTaskContext.get_context_from_package_path(cfg.task.package, cfg)

    if not os.path.exists(cfg.task.proposal_history_filename):
        print('No proposals found at', cfg.task.proposal_history_filename)
        return

    with open(cfg.task.proposal_history_filename, 'rb') as infile:
        proposal_history = pickle.load(infile)

    taskname = cfg.task.package.split('.')[-1]
    task_export_path = os.path.join(cfg.export_path, taskname)
    os.makedirs(task_export_path, exist_ok=True)

    for i, prop in enumerate(proposal_history):
        if prop.eval_results:
            name, code = prop.get_as_module()
            filename = f'{i:03d}.py'
            with open(os.path.join(task_export_path, filename), 'w') as outfile:
                outfile.write(code)


if __name__ == "__main__":
    main()
