import time
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evaluate import evaluate_metric, evaluate_model
from obf_reps.config import ExperimentConfig
from obf_reps.data import ObfusDataModule, ObfusDataset
from obf_reps.logging import Logger
from obf_reps.metrics import ObfMetric
from obf_reps.models import ModelBase
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizers import OptimizerBase
from obf_reps.plotting import plot_lat_scan
from obf_reps.utils import validate_and_create_experiment_config


def train_attack(
    optimizer: OptimizerBase,
    metric: ObfMetric,
    epochs: int,
    train_dataloader: DataLoader,
    experiment_cfg: ExperimentConfig,
    attack_defense_epochs: int,
):
    """
    Args:
        optimizer: Training optimizer.
        train_dataset: Elements are triples of form
            [input: str, behavior_target: str, rep_source: str]
        logger: object for logging
    """

    for at_def_epoch in range(attack_defense_epochs):

        # Defender just went, score them
        metric_eval_results = evaluate_metric(
            metric=metric,
            model=optimizer.model,
            concept_data_module=experiment_cfg.concept_data_module,
            logger=experiment_cfg.logger,
            at_def_epoch=at_def_epoch,
            who_just_went="defender",
        )

        # Attacker turn - top of the inning
        if experiment_cfg.reinit_tunable_params:
            optimizer.model.tunable_params.reinit()

        for epoch in tqdm.tqdm(range(epochs)):
            for batch in tqdm.tqdm(train_dataloader, leave=False):
                optimizer.step(batch)

        if epochs > 0:
            metric_eval_results = evaluate_metric(
                metric=metric,
                model=optimizer.model,
                concept_data_module=experiment_cfg.concept_data_module,
                logger=experiment_cfg.logger,
                at_def_epoch=at_def_epoch,
                who_just_went="attacker",
            )

        # Middle of the inning
        # Defender turn - bottom of the inning
        if at_def_epoch < attack_defense_epochs - 1:

            print("Refitting probe")
            start = time.time()
            metric.refit(at_def_epoch)  # type: ignore
            end = time.time()
            print(f"Elapsed time {end - start} seconds")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    experiment_cfg = validate_and_create_experiment_config(cfg)

    # Log information about the dataset
    experiment_cfg.logger.log(
        {"concept_dataset_train_size": len(experiment_cfg.concept_data_module.train_dataset)}
    )
    experiment_cfg.logger.log(
        {"concept_dataset_val_size": len(experiment_cfg.concept_data_module.val_dataset)}
    )
    experiment_cfg.logger.log(
        {"concept_dataset_test_size": len(experiment_cfg.concept_data_module.test_dataset)}
    )

    attack_results = train_attack(
        optimizer=experiment_cfg.optimizer,
        metric=experiment_cfg.metric,
        epochs=experiment_cfg.epochs,
        train_dataloader=experiment_cfg.obfus_data_module.train_dataloader,  # type: ignore
        experiment_cfg=experiment_cfg,
        attack_defense_epochs=experiment_cfg.attack_defense_epochs,
    )

    if experiment_cfg.epochs > 0:
        model_eval_results = evaluate_model(
            model=experiment_cfg.model,
            loss_fn=experiment_cfg.loss,
            concept_data_module=experiment_cfg.concept_data_module,
            obfus_module=experiment_cfg.obfus_data_module,  # type: ignore
            logger=experiment_cfg.logger,
            gen_len=experiment_cfg.gen_len,
        )


if __name__ == "__main__":
    main()
