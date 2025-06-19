import gc
import pickle
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

from evaluate import evaluate_metric, evaluate_model, evaluate_model_and_metric_harmful
from obf_reps.config import ExperimentConfig
from obf_reps.data import ObfusDataModule, ObfusDataset
from obf_reps.logging import Logger
from obf_reps.metrics import ObfMetric, TestCachedReps
from obf_reps.models import ModelBase
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizers import OptimizerBase
from obf_reps.plotting import plot_lat_scan
from obf_reps.utils import (
    convert_path,
    get_test_cached_reps,
    validate_and_create_experiment_config,
)


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

    test_cached_reps: List[TestCachedReps] = []
    params_last_train_epoch: int = -1

    for at_def_epoch in range(attack_defense_epochs):

        start_epoch_time = time.perf_counter()

        # Defender just went, score them
        obfus_reps_with_behavior = evaluate_model_and_metric_harmful(
            model=optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=metric,
            logger=experiment_cfg.logger,
            at_def_epoch=at_def_epoch,
            who_just_went="defender",
            gen_len=experiment_cfg.gen_len,
            params_last_train_epoch=params_last_train_epoch,
            evaluate_metric_on_gens=False,
            evaluate_behavior=False,
            cached_reps=test_cached_reps,
            batch_size=experiment_cfg.eval_batch_size,
        )

        if at_def_epoch == 0:
            test_cached_reps.append(
                TestCachedReps(
                    label="random-obf",
                    reps=obfus_reps_with_behavior,
                    attack_defence_epoch=-1,
                )
            )

        # If we are in the probe saving regime, then save.
        if experiment_cfg.save_probe_path:
            path = convert_path(experiment_cfg.save_probe_path, at_def_epoch)
            # Assert that the file doesn't exist at the path already
            assert (
                not path.exists()
            ), f"File already exists at {path}. Choose a different path to avoid overwriting."
            metric.save_probe(path)

        # Attacker turn - top of the inning
        loaded_params = False
        if experiment_cfg.load_tunable_params_path:
            # Check if we can load for this at_def_epoch
            path = convert_path(experiment_cfg.load_tunable_params_path, at_def_epoch)
            if path.exists():
                with open(path, "rb") as f:
                    loaded_params = pickle.load(f)
                optimizer.model.tunable_params = loaded_params
                print(f"Loaded tunable parameters from {path}")
                loaded_params = True

        if not loaded_params:
            # We need to train params
            if experiment_cfg.reinit_tunable_params:
                optimizer.model.tunable_params.reinit()

            print("\n========================================================")
            print(f"Training attack defence {at_def_epoch}")
            print("========================================================\n")
            for epoch in tqdm.tqdm(range(epochs)):
                for batch in tqdm.tqdm(train_dataloader, leave=False):
                    optimizer.step(batch)

            # Clear the cache
            torch.cuda.empty_cache()

            # Save tunable parameters if path is provided
            if experiment_cfg.save_tunable_params_path:
                path = convert_path(experiment_cfg.save_tunable_params_path, at_def_epoch)
                assert (
                    not path.exists()
                ), f"File already exists at {str(path)}. Choose a different path to avoid overwriting."
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(experiment_cfg.optimizer.model.tunable_params, f)
                print(f"Tunable parameters saved to {str(path)}")

        params_last_train_epoch += 1
        # Attacker just went, score them
        obfus_reps_with_behavior = evaluate_model_and_metric_harmful(
            model=optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=metric,
            logger=experiment_cfg.logger,
            at_def_epoch=at_def_epoch,
            who_just_went="attacker",
            gen_len=experiment_cfg.gen_len,
            params_last_train_epoch=params_last_train_epoch,
            evaluate_metric_on_gens=False,
            evaluate_behavior=True,
            cached_reps=test_cached_reps,
            batch_size=experiment_cfg.eval_batch_size,
        )

        test_cached_reps.append(
            TestCachedReps(
                label=f"attacker-{at_def_epoch}",
                reps=obfus_reps_with_behavior,
                attack_defence_epoch=at_def_epoch,
            )
        )

        # Defender turn - bottom of the inning
        if at_def_epoch < attack_defense_epochs - 1:

            print("Refitting probe")
            start = time.time()
            metric.refit(at_def_epoch)  # type: ignore
            end = time.time()
            print(f"Elapsed time for refit at_def_epoch {at_def_epoch}: {end - start} seconds")

        end_epoch_time = time.perf_counter()

        print("\n\n")
        print("#" * 30)
        print(f"MARK: Epoch {at_def_epoch} took {end_epoch_time - start_epoch_time:.4f} seconds")
        print("#" * 30)
        print("\n\n")


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

    if experiment_cfg.save_pca_path:
        # Don't need to convert path for pca as it we do not save multiple for atd epochs.
        experiment_cfg.metric.save_pca(experiment_cfg.save_pca_path)

    if experiment_cfg.train_metric_only:

        if experiment_cfg.save_probe_path:
            path = convert_path(experiment_cfg.save_probe_path, attack_defense_epochs=0)
            assert (
                not path.exists()
            ), f"File already exists at {str(path)}. Choose a different path to avoid overwriting."
            experiment_cfg.metric.save_probe(path)
        # Evaluate the metric and then we are done
        evaluate_model_and_metric_harmful(
            model=experiment_cfg.optimizer.model,
            obfus_data_module=experiment_cfg.obfus_data_module,
            concept_data_module=experiment_cfg.concept_data_module,
            metric=experiment_cfg.metric,
            logger=experiment_cfg.logger,
            at_def_epoch=1,
            who_just_went="defender",
            gen_len=experiment_cfg.gen_len,
            evaluate_metric_on_gens=False,
            evaluate_behavior=False,
            batch_size=experiment_cfg.eval_batch_size,
            params_last_train_epoch=0,
        )
        return

    # Note if params can be loaded, train attack will only evaluate.
    attack_results = train_attack(
        optimizer=experiment_cfg.optimizer,
        metric=experiment_cfg.metric,
        epochs=experiment_cfg.epochs,
        train_dataloader=experiment_cfg.obfus_data_module.train_dataloader,  # type: ignore
        experiment_cfg=experiment_cfg,
        attack_defense_epochs=experiment_cfg.attack_defense_epochs,
    )


if __name__ == "__main__":
    main()
