from functools import partial

import click
from fairchem.core.common.registry import registry
from fairchem.core.datasets import data_list_collater
from matbench_discovery.data import DataFiles, df_wbm
from matbench_discovery.energy import get_e_form_per_atom
from matbench_discovery.enums import MbdKey, Task
from matbench_discovery.plots import wandb_scatter
import numpy as np
import pandas as pd
from pymatviz.enums import Key
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from llm_algo_search.fairchem_utils.datasets import AseAtomsZipDataset
# These imports register our custom components with fairchem
from llm_algo_search.fairchem_utils.model import ProposedEnergyModel
from llm_algo_search.fairchem_utils.losses import QuantileLoss, QuantileHuberLoss
from llm_algo_search.fairchem_utils.trainer_quantile import OCPQuantileTrainer


@click.command()
@click.argument('checkpoint_path')
@click.option('--batch_size', default=16, type=int)
@click.option('--debug', is_flag=True)
def main(checkpoint_path, batch_size, debug):
    limit = 100 if debug else None

    dataset = AseAtomsZipDataset(dict(
        src=DataFiles.wbm_initial_atoms.path,
        # src_relaxed=DataFiles.wbm_relaxed_atoms.path,
        # include_relaxed_energy=True,
        limit=limit,
    ))

    cpu = not torch.cuda.is_available()
    device = 'cpu' if cpu else 'cuda'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    print(config)
    wandb.init(project="matbench-discovery")  #, config=run_params, name=run_name)

    trainer_name = 'ocp_qs'  # config["trainer"]  # fairchem doesn't set this correctly afaict
    trainer = registry.get_trainer_class(trainer_name)(
        task=config.get("task", {}),
        model=config["model"],
        dataset=[config["dataset"]],
        outputs=config["outputs"],
        loss_functions=config["loss_functions"],
        evaluation_metrics=config["evaluation_metrics"],
        optimizer=config["optim"],
        identifier="",
        cpu=cpu,
        local_rank=config.get("local_rank", 0),
        inference_only=True,
        logger='tensorboard',
    )
    trainer.load_checkpoint(checkpoint_path, checkpoint, inference_only=True)

    print('Predicting...')
    e_pred_col = 'parr_energy'
    task_type = Task.IS2RE

    collater = partial(
        data_list_collater, otf_graph=True
    )

    mat_ids = [
        atoms.info['material_id'] for atoms in dataset.atoms_list
    ]
    energies = []
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collater)
    for data in tqdm(dataloader):
        data = data.to(device)
        preds = trainer.predict(data, disable_tqdm=True, per_image=False)
        pred_energy = preds['energy'].mean(-1)
        energies.append(pred_energy.cpu().numpy())

    energies = np.concatenate(energies, axis=0)
    df = pd.DataFrame({'material_id': mat_ids, e_pred_col: energies})
    df = df.set_index('material_id')

    df_wbm[e_pred_col] = df[e_pred_col]

    e_form_fairchem_col = f"e_form_per_atom_parr"
    def calc_per_atom(row):
        return get_e_form_per_atom(dict(energy=row[e_pred_col], composition=row[Key.formula]))
    df_wbm[e_form_fairchem_col] = df_wbm.apply(calc_per_atom, axis=1)
    print(df_wbm.head(100))

    table = wandb.Table(
        dataframe=df_wbm[[MbdKey.dft_energy, e_pred_col, Key.formula]]
        .reset_index()
        .dropna()
    )

    title = f"PARR ({len(df):,})"
    wandb_scatter(table, fields=dict(x=MbdKey.dft_energy, y=e_pred_col), title=title)

    # wandb.log_artifact(out_path, type=f"parr-wbm-{task_type}")


if __name__ == '__main__':
    main()

