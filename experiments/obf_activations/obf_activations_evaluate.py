import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from obf_reps.data import (
    STANDARD_MODEL_INPUTS,
    ConceptDataModule,
    ConceptDataset,
    ObfusDataModule,
    ObfusDataset,
)
from obf_reps.logging import Logger
from obf_reps.metrics import CachedReps, ObfMetric, RepsDatasetT, TestCachedReps
from obf_reps.models import HardParams, ModelBase, SoftParams
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizers import OptimizerBase
from obf_reps.plotting import (
    plot_lat_scan,
    plot_roc_curve,
    plot_scores,
    plot_scores_flexible,
)
from obf_reps.utils import (
    gen_return_to_LAT_scans,
    validate_and_create_experiment_config,
)

PROFILING = False
PROCESS_STANDARD_INPUTS = False


@torch.no_grad()
def evaluate_model(
    model: ModelBase,
    loss_fn: LossFunctionBase,
    obfus_module: ObfusDataModule,
    concept_data_module: ConceptDataModule,
    logger: Logger,
    gen_len: int,
):
    """
    Args:
        model: Model to evaluate.
        loss_fn: Loss function to use.
        obfus_module: Contains .val_dataset, list of tuple of form
            [input: str, behavior_target: str, rep_source: str]
        logger: object for logging
        gen_len: the number of tokens to generate during evaluation
    """

    validation_dataset: ObfusDataset = obfus_module.val_dataset  # type: ignore
    validation_scores = []
    logger.create_table(
        "evaluation",
        columns=[
            "input_text",
            "beh_target",
            "tuned_gen_text",
            "standard_gen_text",
            "behavior_score",
            "tuned_on_tuned_scores",
            "standard_on_tuned_scores",
            "standard_on_standard_scores",
            "tuned_on_tuned_mean_score",
            "standard_on_tuned_mean_score",
            "standard_on_standard_mean_score",
        ],
    )

    # If we are assessing the behavior on the output, add reps to this that failed
    #   to achieve the desired behavior. These may still be obfuscated however.

    # Process the positive data that we want to obfuscate
    for input_text, beh_target, _ in tqdm.tqdm(validation_dataset):
        assert isinstance(input_text, str), "Evaluation does not support batching yet."
        assert isinstance(beh_target, str), "Evaluation does not support batching yet."

        tuned_generate = model.generate_from_string(
            input_text=input_text,
            add_chat_template=True,
            use_tunable_params=True,
            max_new_tokens=gen_len,
        )
        standard_generate = model.generate_from_string(
            input_text=input_text,
            add_chat_template=True,
            use_tunable_params=False,
            max_new_tokens=60,  # Hardcoding this as we care less about the standard generation.
        )

        tuned_gen_text = tuned_generate.gen_text[0]
        true_input_text = tuned_generate.input_text[0]
        standard_gen_text = standard_generate.gen_text[0]

        standard_forward = model.forward_from_string(
            input_text=input_text,
            target_text=tuned_gen_text,
            add_chat_template=True,
            use_tunable_params=False,
        )

        tuned_input_score = (
            loss_fn.obf_metric.evaluate(tuned_generate.input_reps).detach().cpu().numpy()
        )
        tuned_on_tuned_score = (
            loss_fn.obf_metric.evaluate(tuned_generate.gen_reps).detach().cpu().numpy()
        )

        standard_input_score = (
            loss_fn.obf_metric.evaluate(standard_forward.input_reps).detach().cpu().numpy()
        )
        standard_on_tuned_score = (
            loss_fn.obf_metric.evaluate(standard_forward.target_reps).detach().cpu().numpy()
        )

        standard_on_standard_score = (
            loss_fn.obf_metric.evaluate(standard_generate.gen_reps).detach().cpu().numpy()
        )

        behavior_score = np.array(
            obfus_module.evaluate_behavior(
                model_input=[input_text],
                model_output=[tuned_gen_text],
                target=[beh_target],
                model=model,
            )
        ).mean()

        validation_scores.append(
            {
                "true_input_text": true_input_text,
                "beh_target": beh_target,
                "tuned_gen_text": tuned_gen_text,
                "behavior_score": behavior_score,
                "tuned_on_tuned_score": tuned_on_tuned_score,
                "standard_on_tuned_score": standard_on_tuned_score,
            }
        )

        min_score = min(
            float(np.min(tuned_on_tuned_score)),
            float(np.min(standard_on_tuned_score)),
            float(np.min(standard_on_standard_score)),
        )
        max_score = max(
            float(np.max(tuned_on_tuned_score)),
            float(np.max(standard_on_tuned_score)),
            float(np.max(standard_on_standard_score)),
        )

        tuned_gen_token_text = [
            [model.tokenizer.decode(x) for x in tuned_generate.gen_ids.squeeze()]
        ]
        standard_gen_token_text = [
            [model.tokenizer.decode(x) for x in standard_generate.gen_ids.squeeze()]
        ]

        standard_on_tuned_fig = plot_lat_scan(
            scores=standard_on_tuned_score,
            text_tokens=tuned_gen_token_text,
            vmin=min_score,
            vmax=max_score,
        )[0]
        tuned_on_tuned_fig = plot_lat_scan(
            scores=tuned_on_tuned_score,
            text_tokens=tuned_gen_token_text,
            vmin=min_score,
            vmax=max_score,
        )[0]
        standard_on_standard_fig = plot_lat_scan(
            scores=standard_on_standard_score,
            text_tokens=standard_gen_token_text,
            vmin=min_score,
            vmax=max_score,
        )[0]

        logger.log_to_table(
            table_name="evaluation",
            data=[
                true_input_text,
                beh_target,
                tuned_gen_text,
                standard_gen_text,
                behavior_score,
                tuned_on_tuned_fig,
                standard_on_tuned_fig,
                standard_on_standard_fig,
                float(np.mean(tuned_on_tuned_score)),
                float(np.mean(standard_on_tuned_score)),
                float(np.mean(standard_on_standard_score)),
            ],
        )

        if hasattr(obfus_module, "original_eval_df"):
            obfus_module.original_eval_df.loc[
                obfus_module.original_eval_df["prompt"] == input_text, "tuned_gen_text"
            ] = tuned_gen_text
            obfus_module.original_eval_df.loc[
                obfus_module.original_eval_df["prompt"] == input_text, "standard_gen_text"
            ] = standard_gen_text
            obfus_module.original_eval_df.loc[
                obfus_module.original_eval_df["prompt"] == input_text, "behavior_target"
            ] = beh_target
            obfus_module.original_eval_df.loc[
                obfus_module.original_eval_df["prompt"] == input_text, "behavior_score"
            ] = behavior_score
            obfus_module.original_eval_df["config"] = str(loss_fn.obf_metric.config)

        plt.close(tuned_on_tuned_fig)
        plt.close(standard_on_tuned_fig)
        plt.close(standard_on_standard_fig)

    if hasattr(obfus_module, "original_eval_df"):
        save_dir = (
            loss_fn.obf_metric.config.results_dir
            if loss_fn.obf_metric.config.results_dir
            else f"results_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(save_dir, exist_ok=True)
        obfus_module.original_eval_df.to_csv(
            f"{save_dir}/gemma2_sql_obfus_eval_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            index=False,
        )

    avg_behavior_score = sum([x["behavior_score"] for x in validation_scores]) / len(
        validation_scores
    )

    logger.log(
        {"avg_behavior_score": avg_behavior_score},
    )

    if PROCESS_STANDARD_INPUTS:
        process_standard_inputs(model, loss_fn.metric, logger, STANDARD_MODEL_INPUTS)

    return validation_scores


@torch.no_grad()
def evaluate_metric(
    metric: ObfMetric,
    model: ModelBase,
    concept_data_module: ConceptDataModule,
    logger: Logger,
    at_def_epoch: int,
    who_just_went: str,
):
    assert who_just_went in ["attacker", "defender"]
    device = model.device

    table_name = "metric_evaluation"
    if table_name not in logger.tables:
        logger.create_table(
            table_name=table_name,
            columns=[
                "at_def_epoch",
                "standard_auc",
                "obfuscated_auc",
                "best_train_threshold",
                "standard_acc_with_train_thresh",
                "obfuscated_acc_with_train_thresh",
                "standard_benign_scores",
                "standard_harmful_scores",
                "obfuscated_harmful_scores",
                "score_plot",
                "pca_plot",
                "standard_roc",
                "obfuscated_roc",
                "standard_acc_curve",
                "obfuscated_acc_curve",
                "standard_pr_curve",
                "obfuscated_pr_curve",
            ],
        )

    validation_dataset: ConceptDataset = concept_data_module.val_dataset  # type: ignore

    metric_results_dataframe = pd.DataFrame(
        columns=[
            "metric_name",
            "metric_config",
            "at_def_epoch",
            "standard_auc",
            "obfuscated_auc",
            "best_train_threshold",
            "standard_acc_with_train_thresh",
            "obfuscated_acc_with_train_thresh",
            "standard_benign_scores",
            "standard_harmful_scores",
            "obfuscated_harmful_scores",
        ]
    )

    # Evaluate standard model
    std_rep_dataset = metric.get_reps_from_dataset(
        model,
        validation_dataset,
        reps_reduce=concept_data_module.test_reps_reduce,
        use_tunable_params=False,
    )

    std_predict_scores_list = []
    std_labels_list = []
    std_reps_list = []
    for rep, label in std_rep_dataset:
        score = metric.predict(rep.to(device))
        std_predict_scores_list.append(score)
        std_labels_list.append(label)
        std_reps_list.append(rep)

    std_predict_scores = np.array(
        [score.cpu().numpy().item() for score in std_predict_scores_list]
    )
    std_labels = np.array([label.cpu().numpy().item() for label in std_labels_list])

    std_roc_auc = roc_auc_score(std_labels, std_predict_scores)

    std_roc_curve, std_acc_curve, std_pr_curve, max_thresh = plot_roc_curve(
        scores=std_predict_scores, labels=std_labels
    )

    # Evaluate model adding tunable parameters
    # Note tuned labels are shuffled compared to std_labels but same data
    tuned_rep_dataset = metric.get_reps_from_dataset(
        model,
        validation_dataset,
        reps_reduce=concept_data_module.test_reps_reduce,
        use_tunable_params=True,
    )

    tuned_predict_scores_list = []
    tuned_labels_list = []
    tuned_reps_list = []
    for rep, label in tuned_rep_dataset:
        score = metric.predict(rep.to(device), layer_reduction="mean")
        tuned_predict_scores_list.append(score)
        tuned_labels_list.append(label)
        tuned_reps_list.append(rep)

    tuned_predict_scores = np.array(
        [score.cpu().numpy().item() for score in tuned_predict_scores_list]
    )
    tuned_labels = np.array([label.cpu().numpy().item() for label in tuned_labels_list])

    tuned_roc_auc = roc_auc_score(tuned_labels, tuned_predict_scores)

    tuned_roc_curve, tuned_acc_curve, tuned_pr_curve, _ = plot_roc_curve(
        scores=tuned_predict_scores, labels=tuned_labels
    )

    # Extract scores for plotting
    standard_benign_scores = std_predict_scores[std_labels == 0]
    standard_harmful_scores = std_predict_scores[std_labels == 1]
    obfuscated_harmful_scores = tuned_predict_scores[tuned_labels == 1]

    # Plot the actual probe predictions.
    score_plot = plot_scores(
        standard_positive_scores=standard_harmful_scores,
        standard_negative_scores=standard_benign_scores,
        obf_positive_scores=obfuscated_harmful_scores,
        thresh=max_thresh,
    )

    # Create PCA plot
    std_benign_reps = [rep for rep, label in zip(std_reps_list, std_labels) if label == 0]
    std_harmful_reps = [rep for rep, label in zip(std_reps_list, std_labels) if label == 1]
    obf_harmful_reps = [rep for rep, label in zip(tuned_reps_list, tuned_labels) if label == 1]

    pca_std_benign = metric.predict_pca(std_benign_reps)
    pca_std_harmful = metric.predict_pca(std_harmful_reps)
    pca_obf_harmful = metric.predict_pca(obf_harmful_reps)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_std_benign[:, 0], pca_std_benign[:, 1], label="Standard Benign", alpha=0.7)
    ax.scatter(pca_std_harmful[:, 0], pca_std_harmful[:, 1], label="Standard Harmful", alpha=0.7)
    ax.scatter(pca_obf_harmful[:, 0], pca_obf_harmful[:, 1], label="Obfuscated Harmful", alpha=0.7)
    ax.legend()
    ax.set_title("PCA of Representations")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    pca_plot = fig

    # Get accuracy when using max thresh
    best_standard_preds = [m > max_thresh for m in std_predict_scores]
    best_standard_acc = sum(best_standard_preds == std_labels) / len(best_standard_preds)

    best_obf_preds = [m > max_thresh for m in tuned_predict_scores]
    best_obf_acc = sum(best_obf_preds == tuned_labels) / len(best_obf_preds)

    logger.log_to_table(
        table_name="metric_evaluation",
        data=[
            f"Inning {at_def_epoch}, {who_just_went} just went",
            std_roc_auc,
            tuned_roc_auc,
            max_thresh,
            best_standard_acc,
            best_obf_acc,
            json.dumps(standard_benign_scores.tolist()),
            json.dumps(standard_harmful_scores.tolist()),
            json.dumps(obfuscated_harmful_scores.tolist()),
            score_plot,
            pca_plot,
            std_roc_curve,
            tuned_roc_curve,
            std_acc_curve,
            tuned_acc_curve,
            std_pr_curve,
            tuned_pr_curve,
        ],
    )

    metric_results_dataframe = pd.concat(
        [
            metric_results_dataframe,
            pd.DataFrame(
                [
                    {
                        "metric_name": metric.__class__.__name__,
                        "metric_config": metric.config,
                        "game_state": f"Inning {at_def_epoch}, {who_just_went} just went",
                        "std_roc_auc": std_roc_auc,
                        "tuned_roc_auc": tuned_roc_auc,
                        "max_thresh": max_thresh,
                        "best_standard_acc": best_standard_acc,
                        "best_obf_acc": best_obf_acc,
                        "standard_benign_scores": json.dumps(standard_benign_scores.tolist()),
                        "standard_harmful_scores": json.dumps(standard_harmful_scores.tolist()),
                        "obfuscated_harmful_scores": json.dumps(
                            obfuscated_harmful_scores.tolist()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    if metric.config.results_dir and not os.path.exists(metric.config.results_dir):
        os.makedirs(metric.config.results_dir, exist_ok=True)

    metric_results_dataframe.to_csv(
        f"{metric.config.results_dir}/metric_results_{metric.__class__.__name__}_layers_{len(metric.config.layers)}_ep_{at_def_epoch}_who_{who_just_went}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )

    # Close all the figs
    # time.sleep(5)
    # plt.close("all")

    return std_roc_auc, tuned_roc_auc


@torch.no_grad()
def evaluate_model_and_metric_harmful(
    model: ModelBase,
    obfus_data_module: ObfusDataModule,
    concept_data_module: ObfusDataModule,
    metric: ObfMetric,
    logger: Logger,
    at_def_epoch: int,
    who_just_went: str,
    gen_len: int,
    params_last_train_epoch,
    batch_size: int = 1,
    evaluate_metric_on_gens: bool = False,
    evaluate_behavior: bool = False,
    cached_reps: Optional[List[TestCachedReps]] = None,
) -> None:
    """The above two functions are similar,fhere we evaluate both the model and the metric.

    Note that this function is specifically for evaluating the model and metric on harmful
    generation detection and harfmul input detection tasks. Some of the semantics may
    break for other tasks.

    Args:
        model: Model to evaluate.
        obfus_data_module: obfuscation data module
        concept_data_module: concept data module
        metric: metric to evaluate
        logger: logger to use
        at_def_epoch: at defense epoch
        who_just_went: string in ["attacker", "defender"]
        evaluate_metric_on_gens: whether to evaluate the metric on the generated reps
            or on reps from the dataset (using teacher forcing if necessary). If this
            is true, then we will select positive examples by evaluating the behavior
            on the output of the model with obfuscation.
        evaluate_behavior: whether to evaluate the behavior of the model.
    """

    assert who_just_went in ["attacker", "defender"]

    device = model.device
    # Loop over data in the concept data module
    validation_dataset: ConceptDataset = concept_data_module.val_dataset + concept_data_module.test_dataset  # type: ignore

    positive_data = [x[0] for x in validation_dataset]
    negative_data = [x[1] for x in validation_dataset]

    validation_scores = []

    obfus_reps_with_behavior = []
    standard_reps_with_behavior = []
    behavior_scores = []

    if evaluate_behavior:
        behavior_table_name = f"behavior_evaluation_{at_def_epoch}_{who_just_went}_just_went"
        logger.create_table(
            table_name=behavior_table_name,
            columns=[
                "input_text",
                "beh_target",
                "tuned_gen_text",
                "standard_gen_text",
                "behavior_score",
                "tuned_on_tuned_scores",
                "standard_on_tuned_scores",
                "standard_on_standard_scores",
                "tuned_ont_tuned_scores_data",
                "standard_ont_tuned_scores_data",
                "standard_ont_standard_scores_data",
            ],
        )

        # ---- Evaluate if the model is displaying the behavior ----- #
        print("\n========================================================")
        print("Generating output for every example in positive test data.")
        print("========================================================\n")

        # Create a DataLoader for positive data
        positive_dataloader = DataLoader(
            positive_data,
            batch_size=batch_size,  # Set batch size to 1 as evaluation doesn't support batching yet
            shuffle=False,  # Keep the order of the data
        )

        # for i, batch in enumerate(tqdm.tqdm(positive_data)):
        for batch_num, batch in enumerate(tqdm.tqdm(positive_dataloader)):

            input_text, beh_target = batch
            batch_size = len(input_text)

            start = time.time()

            # Get generation from model with and without obfuscation
            tuned_generate = model.generate_from_string(
                input_text=input_text,
                add_chat_template=True,
                use_tunable_params=True,
                max_new_tokens=gen_len,
            )
            standard_generate = model.generate_from_string(
                input_text=input_text,
                add_chat_template=True,
                use_tunable_params=False,
                max_new_tokens=gen_len,
            )

            tuned_gen_text = tuned_generate.gen_text
            true_input_text = tuned_generate.input_text
            standard_gen_text = standard_generate.gen_text

            standard_teacher_forced = model.forward_from_string(
                input_text=input_text,
                target_text=tuned_gen_text,
                add_chat_template=True,
                use_tunable_params=False,
                debug=True,
            )

            end = time.time()
            gen_time = end - start

            _, layers, _, h_dim = tuned_generate.gen_reps.shape

            start = time.time()
            tuned_on_tuned_score = metric.evaluate(tuned_generate.gen_reps).detach().cpu().numpy()
            standard_on_tuned_score = (
                metric.evaluate(standard_teacher_forced.target_reps).detach().cpu().numpy()
            )

            standard_on_standard_score = (
                metric.evaluate(standard_generate.gen_reps).detach().cpu().numpy()
            )
            end = time.time()
            eval_time = end - start

            assert len(input_text) == len(tuned_gen_text) == len(beh_target)

            start = time.time()
            behavior_score_batch: List[float] = obfus_data_module.evaluate_behavior(
                model_input=input_text,
                model_output=tuned_gen_text,
                target=beh_target,
                model=model,
            )
            behavior_scores.extend(behavior_score_batch)
            end = time.time()
            beh_eval_time = end - start

            gen_mask = tuned_generate.gen_mask
            forward_mask = standard_teacher_forced.loss_mask
            # Here we record the information we need to evaluate the metric later!

            if evaluate_metric_on_gens:
                # Loop through each example in the batch and its behavior score
                for i, behavior_score in enumerate(behavior_score_batch):
                    if behavior_score > obfus_data_module.behavior_threshold:
                        # Then put label 1 for this example.
                        obfus_probe_input, target_mask = concept_data_module.test_reps_reduce(
                            input_reps=tuned_generate.input_reps[i : i + 1],  # Keep batch dim
                            target_reps=tuned_generate.gen_reps[i : i + 1],
                            target_mask=gen_mask[i : i + 1],
                        )

                        # Process single example
                        assert target_mask[0].ndim == 1  # First (and only) item in batch
                        single_probe_input = obfus_probe_input[0][
                            :, torch.where(target_mask[0])[0], :
                        ]
                        assert single_probe_input.shape[0] == 1
                        assert single_probe_input.shape[1] == layers
                        assert single_probe_input.shape[3] == h_dim

                        obfus_reps_with_behavior.append(single_probe_input.to("cpu"))

                        # Get standard reps for this example
                        standard_probe_input, target_mask = concept_data_module.test_reps_reduce(
                            input_reps=standard_teacher_forced.input_reps[i : i + 1],
                            target_reps=standard_teacher_forced.target_reps[i : i + 1],
                            target_mask=forward_mask[i : i + 1],
                        )

                        assert target_mask[0].ndim == 1
                        single_probe_input = standard_probe_input[0][
                            :, torch.where(target_mask[0])[0], :
                        ]
                        assert single_probe_input.shape[0] == 1
                        assert single_probe_input.shape[1] == layers
                        assert single_probe_input.shape[3] == h_dim

                        standard_reps_with_behavior.append(single_probe_input.to("cpu"))
            else:
                for i in range(len(input_text)):
                    single_input = input_text[i]
                    single_target = beh_target[i]

                    # Get tuned representations
                    tuned_forced = model.forward_from_string(
                        input_text=single_input,
                        target_text=single_target,
                        add_chat_template=True,
                        use_tunable_params=True,
                    )

                    probe_input, target_mask = concept_data_module.test_reps_reduce(
                        input_reps=tuned_forced.input_reps,
                        target_reps=tuned_forced.target_reps,
                        target_mask=tuned_forced.loss_mask,
                    )

                    obfus_reps_with_behavior.append(probe_input.to("cpu"))

                    # Get standard representations
                    standard_forced = model.forward_from_string(
                        input_text=single_input,
                        target_text=single_target,
                        add_chat_template=True,
                        use_tunable_params=False,
                    )

                    standard_probe_input, standard_target_mask = (
                        concept_data_module.test_reps_reduce(
                            input_reps=standard_forced.input_reps,
                            target_reps=standard_forced.target_reps,
                            target_mask=standard_forced.loss_mask,
                        )
                    )

                    standard_reps_with_behavior.append(standard_probe_input.to("cpu"))

                end = time.time()
                reps_reduce_time_2 = end - start

            start = time.time()
            # ----- Plotting for model generation ----- #
            lat_plotting_time = 0
            lat_scan_log_time = 0
            other_log_time = 0
            for i in range(batch_size):

                if i == 0:
                    start_lat = time.time()
                    # Only plot the LAT scan for first example in batch as it is expensive.
                    # Also don't plot all tokens as it is slow
                    plot_len = max(gen_len // 4, 1)

                    tuned_on_tuned_score_ = tuned_on_tuned_score[i][:plot_len]
                    standard_on_tuned_score_ = standard_on_tuned_score[i][:plot_len]
                    standard_on_standard_score_ = standard_on_standard_score[i][:plot_len]
                    # tuned_on_input_score_ = tuned_on_input_score[i]
                    # standard_on_input_score_ = standard_on_input_score[i]

                    min_score = min(
                        float(np.min(tuned_on_tuned_score_)),
                        float(np.min(standard_on_tuned_score)),
                        float(np.min(standard_on_standard_score)),
                    )
                    max_score = max(
                        float(np.max(tuned_on_tuned_score_)),
                        float(np.max(standard_on_tuned_score_)),
                        float(np.max(standard_on_standard_score_)),
                    )

                    tuned_gen_ids = tuned_generate.gen_ids[i]
                    tuned_gen_token_text = [
                        [model.tokenizer.decode(x) for x in tuned_gen_ids.squeeze()][:plot_len]
                    ]
                    standard_gen_ids = standard_generate.gen_ids[i]
                    standard_gen_token_text = [
                        [model.tokenizer.decode(x) for x in standard_gen_ids.squeeze()][:plot_len]
                    ]

                    standard_on_tuned_fig = plot_lat_scan(
                        scores=standard_on_tuned_score_,
                        text_tokens=tuned_gen_token_text,
                        vmin=min_score,
                        vmax=max_score,
                    )[0]
                    tuned_on_tuned_fig = plot_lat_scan(
                        scores=tuned_on_tuned_score_,
                        text_tokens=tuned_gen_token_text,
                        vmin=min_score,
                        vmax=max_score,
                    )[0]
                    standard_on_standard_fig = plot_lat_scan(
                        scores=standard_on_standard_score_,
                        text_tokens=standard_gen_token_text,
                        vmin=min_score,
                        vmax=max_score,
                    )[0]

                    end_lat = time.time()
                    lat_plotting_time += end_lat - start_lat

                    start_log = time.time()
                    logger.log_to_table(
                        table_name=behavior_table_name,
                        data=[
                            true_input_text[i],
                            beh_target[i],
                            tuned_gen_text[i],
                            standard_gen_text[i],
                            behavior_score_batch[i],
                            tuned_on_tuned_fig,
                            standard_on_tuned_fig,
                            standard_on_standard_fig,
                            json.dumps(tuned_on_tuned_score_.tolist()),
                            json.dumps(standard_on_tuned_score_.tolist()),
                            json.dumps(standard_on_standard_score_.tolist()),
                        ],
                    )
                    end_log = time.time()
                    lat_scan_log_time += end_log - start_log
                else:
                    standard_on_tuned_fig = None
                    tuned_on_tuned_fig = None
                    standard_on_standard_fig = None

                    start_log = time.time()
                    logger.log_to_table(
                        table_name=behavior_table_name,
                        data=[
                            true_input_text[i],
                            beh_target[i],
                            tuned_gen_text[i],
                            standard_gen_text[i],
                            behavior_score_batch[i],
                            tuned_on_tuned_fig,
                            standard_on_tuned_fig,
                            standard_on_standard_fig,
                            None,
                            None,
                            None,
                        ],
                    )
                    end_log = time.time()
                    other_log_time += end_log - start_log

            # Clear the cache
            gb_start = time.time()
            # gc.collect()
            torch.cuda.empty_cache()
            gb_end = time.time()
            gb_time = gb_end - gb_start

            end = time.time()
            table_time = end - start

            # print all recorded times
            if PROFILING:
                print(f"Time to generate: {gen_time}")
                print(f"Time to evaluate: {eval_time}")
                print(f"Time to evaluate behavior: {beh_eval_time}")
                # print(f"Time to reduce reps 1: {reps_reduce_time_1}")
                # print(f"Time to reduce reps 2: {reps_reduce_time_2}")
                print(f"Time to table: {table_time}")
                print(f"\tTime to plot LAT scancs: {lat_plotting_time}")
                print(f"\tTime to log LAT scans: {lat_scan_log_time}")
                print(f"\tTime to log other: {other_log_time}")
                print(f"Time to clear cache: {gb_time}")

        avg_behavior_score = sum(behavior_scores) / len(behavior_scores)
        behavior_success_rate = sum(
            [x > obfus_data_module.behavior_threshold for x in behavior_scores]
        ) / len(behavior_scores)

        logger.log(
            {
                "avg_behavior_score": avg_behavior_score,
                "behavior_success_rate": behavior_success_rate,
            },
        )

        if PROCESS_STANDARD_INPUTS:
            process_standard_inputs(model, metric, logger, STANDARD_MODEL_INPUTS)

        logger.log_table_name(behavior_table_name)
    else:
        # We are not evaluating the behavior, so we just need to collect
        # the positive reps to evaluate the model
        print("\n========================================================")
        print("Collecting output for every example in positive test data.")
        print("========================================================\n")

        for batch_num, batch in enumerate(tqdm.tqdm(positive_data)):

            input_text, beh_target = batch

            assert isinstance(input_text, str), "Evaluation does not support batching yet."
            assert isinstance(beh_target, str), "Evaluation does not support batching yet."

            if evaluate_metric_on_gens:
                tuned_generate = model.generate_from_string(
                    input_text=input_text,
                    add_chat_template=True,
                    use_tunable_params=True,
                    max_new_tokens=gen_len,
                )
                behavior_score = obfus_module.evaluate_behavior(
                    model_input=input_text,
                    model_output=tuned_generate.gen_text[0],
                    target=beh_target,
                    model=model,
                )
                if behavior_score > obfus_data_module.behavior_threshold:
                    probe_input, _ = concept_data_module.test_reps_reduce(
                        input_reps=tuned_generate.input_reps,
                        target_reps=tuned_generate.gen_reps,
                    )
                    obfus_reps_with_behavior.append(probe_input.to("cpu"))

                    # Behavior met the threshold, so we also collect the standard reps
                    standard_teacher_forced = model.forward_from_string(
                        input_text=input_text,
                        target_text=tuned_generate.gen_text[0],
                        add_chat_template=True,
                        use_tunable_params=False,
                    )
                    standard_probe_input, _ = concept_data_module.test_reps_reduce(
                        input_reps=standard_teacher_forced.input_reps,
                        target_reps=standard_teacher_forced.target_reps,
                    )
                    standard_reps_with_behavior.append(standard_probe_input.to("cpu"))

                    # Del everything we no longer need

            else:
                # Collect the teacher forced reps
                tuned_teacher_forced = model.forward_from_string(
                    input_text=input_text,
                    target_text=beh_target,
                    add_chat_template=True,
                    use_tunable_params=True,
                )
                probe_input, _ = concept_data_module.test_reps_reduce(
                    input_reps=tuned_teacher_forced.input_reps,
                    target_reps=tuned_teacher_forced.target_reps,
                )
                obfus_reps_with_behavior.append(probe_input.to("cpu"))

                # Collect the standard teacher forced reps
                standard_teacher_forced = model.forward_from_string(
                    input_text=input_text,
                    target_text=beh_target,
                    add_chat_template=True,
                    use_tunable_params=False,
                )
                probe_input, _ = concept_data_module.test_reps_reduce(
                    input_reps=standard_teacher_forced.input_reps,
                    target_reps=standard_teacher_forced.target_reps,
                )
                standard_reps_with_behavior.append(probe_input.to("cpu"))

                # Del everything we no longer need
                del tuned_teacher_forced, standard_teacher_forced

            # Clear the cache
            torch.cuda.empty_cache()

    # Now we loop over the negative examples. In terms of showing
    # the desired behavior, we do not care about these, we only
    # care about collecting the representations we need to
    # evaluate the metric.
    print("\n========================================================")
    print("Collecting output for every example in negative test data.")
    print("========================================================\n")

    negative_reps = []
    for batch in tqdm.tqdm(negative_data):

        # Clear the cache
        torch.cuda.empty_cache()

        negative_input, beh_target = batch

        if evaluate_metric_on_gens:
            negative_generate = model.generate_from_string(
                input_text=negative_input,
                add_chat_template=True,
                use_tunable_params=False,
                max_new_tokens=gen_len,
            )
            probe_input, _ = concept_data_module.test_reps_reduce(
                input_reps=negative_generate.input_reps,
                target_reps=negative_generate.gen_reps,
            )
            negative_reps.append(probe_input.to("cpu"))
        else:
            negative_teacher_forced = model.forward_from_string(
                input_text=negative_input,
                target_text=beh_target,
                add_chat_template=True,
                use_tunable_params=False,
            )
            probe_input, _ = concept_data_module.test_reps_reduce(
                input_reps=negative_teacher_forced.input_reps,
                target_reps=negative_teacher_forced.target_reps,
            )
            negative_reps.append(probe_input.to("cpu"))

    # If Cached reps given, then iteratre through those and record sores
    cached_obf_reps_scores = {}
    cached_obf_reps = {}
    if cached_reps is not None:
        for cached_rep in cached_reps:
            label = cached_rep.label
            cache_attack_defence_epoch = cached_rep.attack_defence_epoch
            if cache_attack_defence_epoch == params_last_train_epoch:
                # This is caching stuff that is already in the obfus_reps_with_behavior
                continue

            cached_obf_reps_scores[cache_attack_defence_epoch] = []
            cached_obf_reps[cache_attack_defence_epoch] = []

            for rep in cached_rep.reps:
                score = metric.predict(rep.to(device)).cpu().numpy().item()
                cached_obf_reps_scores[cache_attack_defence_epoch].append(score)
                cached_obf_reps[cache_attack_defence_epoch].append(rep.to("cpu"))

    # Ok so now we have the following list of reps:
    # obfus_reps_with_behavior (label 1)
    # standard_reps_with_behavior (label 1)
    # negative_reps (label 0)
    # We test the metric on this data.
    obfus_positive_scores = [
        metric.predict(rep.to(device)).cpu().numpy().item() for rep in obfus_reps_with_behavior
    ]
    std_positive_scores = [
        metric.predict(rep.to(device)).cpu().numpy().item() for rep in standard_reps_with_behavior
    ]
    std_negative_scores = [
        metric.predict(rep.to(device)).cpu().numpy().item() for rep in negative_reps
    ]

    std_scores = np.array(std_positive_scores + std_negative_scores)
    std_labels = np.array([1] * len(std_positive_scores) + [0] * len(std_negative_scores))
    obfus_scores = np.array(obfus_positive_scores + std_negative_scores)
    obfus_labels = np.array([1] * len(obfus_positive_scores) + [0] * len(std_negative_scores))

    std_roc_auc = roc_auc_score(std_labels, std_scores)
    obfus_roc_auc = roc_auc_score(obfus_labels, obfus_scores)

    std_roc_curve, std_acc_curve, std_pr_curve, max_thresh = plot_roc_curve(
        scores=std_scores, labels=std_labels
    )
    obfus_roc_curve, obfus_acc_curve, obfus_pr_curve, _ = plot_roc_curve(
        scores=obfus_scores, labels=obfus_labels
    )

    # Collate all the score plotting data
    score_plotting_data = {}
    score_plotting_data["standard_benign"] = std_negative_scores
    score_plotting_data["standard_harmful"] = std_positive_scores
    score_plotting_data[f"obf_harmful_att_def_epoch_{params_last_train_epoch}"] = (
        obfus_positive_scores
    )
    for cache_at_def_epoch, cached_scores in cached_obf_reps_scores.items():
        label = f"obf_harmful_att_def_epoch_{cache_at_def_epoch}"
        assert label not in score_plotting_data
        score_plotting_data[label] = cached_scores

    # Extract scores for plotting
    score_plot = plot_scores(
        standard_positive_scores=np.array(std_positive_scores),
        standard_negative_scores=np.array(std_negative_scores),
        obf_positive_scores=np.array(obfus_positive_scores),
        thresh=max_thresh,
    )

    full_score_plot = plot_scores_flexible(
        plotting_data=score_plotting_data,
        thresh=max_thresh,
    )

    # Create PCA plot
    pca_plotting_data = {}

    pca_std_benign = metric.predict_pca(negative_reps)
    pca_std_harmful = metric.predict_pca(standard_reps_with_behavior)
    pca_obf_harmful = metric.predict_pca(obfus_reps_with_behavior)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_std_benign[:, 0], pca_std_benign[:, 1], label="Standard Benign", alpha=0.7, s=1)
    ax.scatter(
        pca_std_harmful[:, 0], pca_std_harmful[:, 1], label="Standard Harmful", alpha=0.7, s=1
    )
    ax.scatter(
        pca_obf_harmful[:, 0],
        pca_obf_harmful[:, 1],
        label=f"Obfuscated Harmful Att Def Epoch {params_last_train_epoch}",
        alpha=0.7,
        s=1,
    )

    pca_plotting_data["standard_benign"] = list(map(tuple, pca_std_benign))
    pca_plotting_data["standard_harmful"] = list(map(tuple, pca_std_harmful))
    pca_plotting_data[f"obf_harmful_att_def_epoch_{params_last_train_epoch}"] = list(
        map(tuple, pca_obf_harmful)
    )
    # If cached reps given, then add these to the plot
    for cache_at_def_epoch, cached_reps in cached_obf_reps.items():
        pca_cached_obf_harmful = metric.predict_pca(cached_reps)
        label = f"obf_harmful_att_def_epoch_{cache_at_def_epoch}"
        ax.scatter(
            pca_cached_obf_harmful[:, 0], pca_cached_obf_harmful[:, 1], label=label, alpha=0.7, s=1
        )
        pca_cached_obf_harmful_coords = list(map(tuple, pca_cached_obf_harmful))
        assert label not in pca_plotting_data
        pca_plotting_data[label] = pca_cached_obf_harmful_coords

    ax.legend()
    ax.set_title("PCA of Representations")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    pca_plot = fig

    # Get accuracy when using max thresh
    best_standard_preds = [m > max_thresh for m in std_scores]
    best_standard_acc = sum(best_standard_preds == std_labels) / len(best_standard_preds)

    best_obf_preds = [m > max_thresh for m in obfus_scores]
    best_obf_acc = sum(best_obf_preds == obfus_labels) / len(best_obf_preds)

    obf_pos_preds = [m > max_thresh for m in obfus_positive_scores]
    frac_obf_correct = sum(obf_pos_preds) / len(obf_pos_preds)

    tunable_params = model.tunable_params
    if isinstance(tunable_params, SoftParams):
        attack_suffix = None
        attack_suffix_len = tunable_params.params[0].shape[0]
    elif isinstance(tunable_params, HardParams):
        attack_ids: List[int] = tunable_params.optim_ids.cpu().numpy().tolist()[0]
        attack_suffix_len = len(attack_ids)
        attack_suffix = model.to_string([attack_ids])[0]
    else:
        attack_suffix = None
        attack_suffix_len = None

    table_name = "metric_evaluation"
    if table_name not in logger.tables:
        logger.create_table(
            table_name=table_name,
            columns=[
                "at_def_epoch",
                "standard_auc",
                "obfuscated_auc",
                "best_train_threshold",
                "standard_acc_with_train_thresh",
                "obfuscated_acc_with_train_thresh",
                "obfuscated_frac_correct",
                "standard_benign_scores",
                "standard_harmful_scores",
                "obfuscated_harmful_scores",
                "score_plotting_data",
                "score_plot",
                "full_score_plot",
                "pca_plotting_data",
                "pca_plot",
                "standard_roc",
                "obfuscated_roc",
                "standard_acc_curve",
                "obfuscated_acc_curve",
                "standard_pr_curve",
                "obfuscated_pr_curve",
                "attack_suffix",
                "attack_suffix_len",
            ],
        )

    logger.log_to_table(
        table_name=table_name,
        data=[
            f"Inning {at_def_epoch}, {who_just_went} just went",
            std_roc_auc,
            obfus_roc_auc,
            max_thresh,
            best_standard_acc,
            best_obf_acc,
            frac_obf_correct,
            json.dumps(std_negative_scores),
            json.dumps(std_positive_scores),
            json.dumps(obfus_positive_scores),
            json.dumps(score_plotting_data),
            score_plot,
            full_score_plot,
            json.dumps(pca_plotting_data),
            pca_plot,
            std_roc_curve,
            obfus_roc_curve,
            std_acc_curve,
            obfus_acc_curve,
            std_pr_curve,
            obfus_pr_curve,
            attack_suffix,
            attack_suffix_len,
        ],
    )

    logger.log_table_name(table_name=table_name)

    # Return these for caching
    return obfus_reps_with_behavior


# ------------------------------ #
# Helper functions               #
# ------------------------------ #
def process_standard_inputs(model, metric, logger, standard_inputs):
    logger.create_table(
        table_name="standard_input_generate",
        columns=[
            "input_text",
            "tuned_gen",
            "standard_gen",
            "tuned_fig",
            "standard_fig",
        ],
    )

    print("Testing on normal model inputs")
    # Forward pass model on normal inputs.
    for standard_input in tqdm.tqdm(standard_inputs):
        # Get model generation
        tuned_generate = model.generate_from_string(
            input_text=standard_input,
            add_chat_template=True,
            use_tunable_params=True,
            max_new_tokens=60,
        )
        standard_generate = model.generate_from_string(
            input_text=standard_input,
            add_chat_template=True,
            use_tunable_params=False,
            max_new_tokens=60,
        )

        _, [tuned_fig, standard_fig] = gen_return_to_LAT_scans(
            model=model,
            gen_returns=[tuned_generate, standard_generate],
            obf_metric=metric,
        )

        logger.log_to_table(
            data=[
                standard_input,
                tuned_generate.gen_text[0],
                standard_generate.gen_text[0],
                tuned_fig,
                standard_fig,
            ],
            table_name="standard_input_generate",
        )
