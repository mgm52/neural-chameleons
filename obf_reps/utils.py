import pathlib
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.metrics import max_error
from torch import Tensor
from torch.utils.data import dataset

from obf_reps.config import ExperimentConfig
from obf_reps.data import (
    AlpacaDataModule,
    AssistantSQLConceptDataModule,
    HarmfulConceptDataModule,
    HGDHarmfulConceptDataModule,
    HIDHarmfulConceptDataModule,
    JailbreakCircuitBreakersObfusDataModule,
    JailbreakLongBehObfusDataModule,
    JailbreakLongRepObfusDataModule,
    JailbreakObfusDataModule,
    LongInputHarmfulConceptDataModule,
    LongInputHarmfulObfusDataModule,
    MultiHIDHarmfulConceptDataModule,
    MultiHIDHarmfulObfusDataModule,
    MultiLieDataModule,
    OrigHarmfulConceptDataModule,
    SimpleSingleHIDHarmfulConceptDataModule,
    SimpleSingleHIDHarmfulObfusDataModule,
    SingleDishonestDataModule,
    SingleHarmfulDataModule,
    SingleHGDHarmfulConceptDataModule,
    SingleHIDHarmfulConceptDataModule,
    SingleHIDHarmfulObfusDataModule,
    SingleLieDataModule,
    SingleLookDataModule,
    SingleSQLDataModule,
    SingleSQLObfusDataModule,
    SplitSQLObfusDataModule,
    SQLConceptDataModule,
    SQLObfusDataModule,
    TruthfulConceptDataModule,
)
from obf_reps.logging import DummyLogger, WAndBLogger
from obf_reps.metrics import (
    GemmaScopeClassifierMetric,
    GemmaScopeFeatureMetric,
    LogisticRegressionMetric,
    MahalanobisDistanceMetric,
    MaxEnsembleLogisticRegressionMetric,
    MaxEnsembleMeanDiffCosineSimMetric,
    MaxEnsembleMLPMetric,
    MeanDiffCosineSimMetric,
    MeanEnsembleLogisticRegressionMetric,
    MeanEnsembleMeanDiffCosineSimMetric,
    MeanEnsembleMLPMetric,
    MetricConfig,
    MLPMetric,
    ObfMetric,
    #SAEClassifierMetric,
    TestCachedReps,
    VAEMetric,
)
from obf_reps.models import GenReturn, ModelBase, ModelConfig
from obf_reps.models.gemma2 import Gemma2bHardPrompted, Gemma2bSoftPrompted
from obf_reps.models.llama3 import Llama3HardPrompted, Llama3SoftPrompted
from obf_reps.models.llamaLAT import LlamaLATHardPrompted, LlamaLATSoftPrompted
from obf_reps.models.llamaRR import LlamaRRHardPrompted, LlamaRRSoftPrompted
from obf_reps.optimize.loss import (
    BehaviorLoss,
    FinalTokenObfLoss,
    MaxOverTokensAndLayersObfLoss,
    MaxOverTokensObfLoss,
    MeanOverTokensObfLoss,
    ProbePredictLoss,
)
from obf_reps.optimize.optimizers import (
    AdamOptimizer,
    FLRTOptimizer,
    GCGOptimizer,
    OptimizerConfig,
)
from obf_reps.plotting import plot_lat_scan

# Ignore warning from setting torch.deterministic_algorithms()
warnings.filterwarnings(
    "ignore", message="cumsum_cuda_kernel does not have a deterministic implementation"
)
warnings.filterwarnings(
    "ignore",
    message="Memory Efficient attention defaults to a non-deterministic algorithm",
)
warnings.filterwarnings("ignore", message="Deterministic behavior was enabled")


def convert_path(path: Optional[Path], attack_defense_epochs: int) -> Path:
    # Assert that the path is a file, not a directory

    if path is None:
        return None

    assert isinstance(path, Path), f"Path should be type path, is {type(path)}"

    if path.is_dir():
        raise ValueError(
            f"The provided path '{path}' is a directory. It should be a path to a file."
        )

    # Split the file name and extension
    stem = path.stem
    suffix = path.suffix

    # Create the new file name with the added string
    new_filename = f"{stem}_ade_{attack_defense_epochs}{suffix}"

    # Return the new path with the modified file name
    return path.with_name(new_filename)


def validate_and_create_model_config(cfg: DictConfig) -> ModelConfig:
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    if cfg.model_dtype == "float32":
        model_dtype = torch.float32
    elif cfg.model_dtype in ["float16", "half"]:
        model_dtype = torch.float16
    elif cfg.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    else:
        raise ValueError(f"model_dtype {cfg.model_dtype} not supported")

    return ModelConfig(
        model_dtype=model_dtype,
        prompt_init=cfg.prompt_init,
        requires_grad=cfg.requires_grad,
        device=cfg.device,
    )


def validate_and_create_optimizer_config(
    cfg: DictConfig, prompt_init: str, seed: int
) -> OptimizerConfig:
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    assert prompt_init is not None, "prompt_init must not be None"
    assert isinstance(seed, int), f"seed must be an int, got {seed} of type {type(seed)}"

    # Get FLRT-specific parameters with defaults of None if not present
    k1 = getattr(cfg, "k1", None)
    k2 = getattr(cfg, "k2", None)
    p_add = getattr(cfg, "p_add", None)
    p_swap = getattr(cfg, "p_swap", None)
    p_del = getattr(cfg, "p_del", None)
    init_len = getattr(cfg, "init_len", None)
    generator_weight = getattr(cfg, "generator_weight", None)
    monitor_weight = getattr(cfg, "monitor_weight", None)

    return OptimizerConfig(
        lr=cfg.lr,
        num_steps=cfg.num_steps,
        optim_str_init=prompt_init,
        search_width=cfg.search_width,
        topk=cfg.topk,
        n_replace=cfg.n_replace,
        buffer_size=cfg.buffer_size,
        use_prefix_cache=cfg.use_prefix_cache,
        filter_ids=cfg.filter_ids,
        seed=seed,
        allow_non_ascii=cfg.allow_non_ascii,
        eval_steps=cfg.eval_steps,
        eval_with_check_refusal=cfg.eval_with_check_refusal,
        check_refusal_min_loss=cfg.check_refusal_min_loss,
        early_stopping=cfg.early_stopping,
        early_stopping_min_loss=cfg.early_stopping_min_loss,
        # Add FLRT-specific parameters
        k1=k1,
        k2=k2,
        p_add=p_add,
        p_swap=p_swap,
        p_del=p_del,
        init_len=init_len,
        generator_weight=generator_weight,
        monitor_weight=monitor_weight,
    )


def validate_and_create_metric_config(cfg: DictConfig) -> MetricConfig:
    return MetricConfig(
        layers=cfg.layers,
        feature_list_filepath=cfg.get("feature_list_filepath", None),
        results_dir=cfg.get("results_dir", None),
    )


def validate_and_create_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    cfg = cfg.experiment

    # Validate model_path
    if not cfg.model_path:
        raise ValueError("model_path is required")
    if not pathlib.Path(cfg.model_path).exists():
        raise FileNotFoundError(f"model_path {cfg.model_path} does not exist")

    # Validate obfus_data_path
    if cfg.obfus_data_path is not None:
        if isinstance(cfg.obfus_data_path, ListConfig):
            # Handle SQLObfusDataModule case with list of paths
            for path in cfg.obfus_data_path:
                if not pathlib.Path(path).exists():
                    raise FileNotFoundError(f"obfus_data_path {path} does not exist")
        else:
            if not pathlib.Path(cfg.obfus_data_path).exists():
                raise FileNotFoundError(f"obfus_data_path {cfg.obfus_data_path} does not exist")

    # Validate model_cls
    match cfg.model_cls:
        case "Llama3SoftPrompted":
            model_cls = Llama3SoftPrompted
        case "Llama3HardPrompted":
            model_cls = Llama3HardPrompted
        case "Gemma2bSoftPrompted":
            model_cls = Gemma2bSoftPrompted
        case "Gemma2bHardPrompted":
            model_cls = Gemma2bHardPrompted
        case "LlamaRRSoftPrompted":
            model_cls = LlamaRRSoftPrompted
        case "LlamaRRHardPrompted":
            model_cls = LlamaRRHardPrompted
        case "LlamaLATSoftPrompted":
            model_cls = LlamaLATSoftPrompted
        case "LlamaLATHardPrompted":
            model_cls = LlamaLATHardPrompted
        case _:
            raise ValueError(f"model_cls {cfg.model_cls} not supported")

    # Validate optimizer
    match cfg.optimizer:
        case "AdamContinuousOptimizer":
            optimizer_cls = AdamOptimizer
        case "GCGOptimizer":
            optimizer_cls = GCGOptimizer
        case "FLRTOptimizer":
            optimizer_cls = FLRTOptimizer
        case _:
            raise ValueError(f"optimizer {cfg.optimizer} not supported")

    # Validate lamda_
    if cfg.lambda_ is None:
        raise ValueError("lambda_ is required")
    if not isinstance(cfg.lambda_, float):
        raise TypeError(f"Expected float, got {type(cfg.lambda_)}")

    # Validate epochs
    if getattr(cfg, "epochs", None) is None:
        epochs = 1  # default to 1
    else:
        epochs = cfg.epochs

    seed = cfg.get("seed", None)

    # Validate metric
    match cfg.metric:
        case "MeanDiffCosineSimMetric":
            metric_cls = MeanDiffCosineSimMetric
        case "LogisticRegressionMetric":
            metric_cls = LogisticRegressionMetric
        case "MLPMetric":
            metric_cls = MLPMetric
        case "AnomalyDetector":
            metric_cls = MahalanobisDistanceMetric
        # case "SAEClassifierMetric":
        #     metric_cls = SAEClassifierMetric
        case "GemmaScopeClassifierMetric":
            metric_cls = GemmaScopeClassifierMetric
        case "GemmaScopeFeatureMetric":
            metric_cls = GemmaScopeFeatureMetric
        case "MeanEnsembleMLPMetric":
            metric_cls = MeanEnsembleMLPMetric
        case "MeanEnsembleMeanDiffCosineSimMetric":
            metric_cls = MeanEnsembleMeanDiffCosineSimMetric
        case "MeanEnsembleLogisticRegressionMetric":
            metric_cls = MeanEnsembleLogisticRegressionMetric
        case "MaxEnsembleMLPMetric":
            metric_cls = MaxEnsembleMLPMetric
        case "MaxEnsembleMeanDiffCosineSimMetric":
            metric_cls = MaxEnsembleMeanDiffCosineSimMetric
        case "MaxEnsembleLogisticRegressionMetric":
            metric_cls = MaxEnsembleLogisticRegressionMetric
        case "VAEMetric":
            metric_cls = VAEMetric
        case _:
            raise ValueError(f"metric {cfg.metric} not supported")

    # Validate loss
    match cfg.loss:
        case "FinalTokenObfLoss":
            loss_cls = FinalTokenObfLoss
        case "MeanOverTokensObfLoss":
            loss_cls = MeanOverTokensObfLoss
        case "MaxOverTokensObfLoss":
            loss_cls = MaxOverTokensObfLoss
        case "MaxOverTokensAndLayersObfLoss":
            loss_cls = MaxOverTokensAndLayersObfLoss
        case "BehaviorLoss":
            loss_cls = BehaviorLoss
        case "ProbePredictLoss":
            loss_cls = ProbePredictLoss
        case _:
            raise ValueError(f"loss {cfg.loss} not supported")

    # Validate obfus_data_module
    match cfg.obfus_data_module:
        case "JailbreakObfusDataModule":
            obfus_data_module_cls = JailbreakObfusDataModule
        case "SQLObfusDataModule":
            obfus_data_module_cls = SQLObfusDataModule
        case "SingleHarmfulDataModule":
            obfus_data_module_cls = SingleHarmfulDataModule
        case "SingleHIDHarmfulObfusDataModule":
            obfus_data_module_cls = SingleHIDHarmfulObfusDataModule
        case "SimpleSingleHIDHarmfulObfusDataModule":
            obfus_data_module_cls = SimpleSingleHIDHarmfulObfusDataModule
        case "MultiHIDHarmfulObfusDataModule":
            obfus_data_module_cls = MultiHIDHarmfulObfusDataModule
        case "JailbreakLongRepObfusDataModule":
            obfus_data_module_cls = JailbreakLongRepObfusDataModule
        case "JailbreakLongBehObfusDataModule":
            obfus_data_module_cls = JailbreakLongBehObfusDataModule
        case "SingleLieDataModule":
            obfus_data_module_cls = SingleLieDataModule
        case "SingleDishonestDataModule":
            obfus_data_module_cls = SingleDishonestDataModule
        case "MultiLieDataModule":
            obfus_data_module_cls = MultiLieDataModule
        case "AlpacaDataModule":
            obfus_data_module_cls = AlpacaDataModule
        case "SingleLookDataModule":
            obfus_data_module_cls = SingleLookDataModule
        case "SingleSQLDataModule":
            obfus_data_module_cls = SingleSQLDataModule
        case "JailbreakCircuitBreakersObfusDataModule":
            obfus_data_module_cls = JailbreakCircuitBreakersObfusDataModule
        case "LongInputHarmfulObfusDataModule":
            obfus_data_module_cls = LongInputHarmfulObfusDataModule
        case "SingleSQLObfusDataModule":
            obfus_data_module_cls = SingleSQLObfusDataModule
        case "SplitSQLObfusDataModule":
            obfus_data_module_cls = SplitSQLObfusDataModule
        case _:
            raise ValueError(f"obfus_data_module {cfg.obfus_data_module} not supported")

    # Validate concept_data_module
    match cfg.concept_data_module:
        case "HarmfulConceptDataModule":
            concept_data_module_cls = HarmfulConceptDataModule
        case "TruthfulConceptDataModule":
            concept_data_module_cls = TruthfulConceptDataModule
        case "SQLConceptDataModule":
            concept_data_module_cls = SQLConceptDataModule
        case "SingleHIDHarmfulConceptDataModule":
            concept_data_module_cls = SingleHIDHarmfulConceptDataModule
        case "SimpleSingleHIDHarmfulConceptDataModule":
            concept_data_module_cls = SimpleSingleHIDHarmfulConceptDataModule
        case "MultiHIDHarmfulConceptDataModule":
            concept_data_module_cls = MultiHIDHarmfulConceptDataModule
        case "HIDHarmfulConceptDataModule":
            concept_data_module_cls = HIDHarmfulConceptDataModule
        case "SingleHGDHarmfulConceptDataModule":
            concept_data_module_cls = SingleHGDHarmfulConceptDataModule
        case "HGDHarmfulConceptDataModule":
            concept_data_module_cls = HGDHarmfulConceptDataModule
        case "OrigHarmfulConceptDataModule":
            concept_data_module_cls = OrigHarmfulConceptDataModule
        case "LongInputHarmfulConceptDataModule":
            concept_data_module_cls = LongInputHarmfulConceptDataModule
        case "AssistantSQLConceptDataModule":
            concept_data_module_cls = AssistantSQLConceptDataModule
        case _:
            raise ValueError(f"concept_data_module {cfg.concept_data_module} not supported")

    # Instantiate logger as type effects instantiation
    match cfg.logger:
        case "DummyLogger":
            logger = DummyLogger()
        case "WAndBLogger":
            dict_cfg: Dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
            logger = WAndBLogger(log_file=cfg.log_file, username=cfg.username, metadata=dict_cfg)
        case _:
            raise ValueError(f"logger {cfg.logger} not supported")

    if "gen_len" not in cfg:
        raise ValueError("'gen_len' not provdied in config.")

    if "reinit_tunable_params" not in cfg:
        raise ValueError("'reinit_tunable_params' not provdied in config.")
    if not isinstance(cfg.reinit_tunable_params, bool):
        raise ValueError("'reinit_tunable_params' must be bool")

    # Handle the optional paths
    load_tunable_params_path = cfg.get("load_tunable_params_path", None)
    load_tunable_params_path_direct = cfg.get("load_tunable_params_path_direct", None)
    if load_tunable_params_path_direct is not None:
        if not Path(load_tunable_params_path_direct).exists():
            raise FileNotFoundError(
                f"load_tunable_params_path_direct {load_tunable_params_path_direct} does not exist"
            )
    elif load_tunable_params_path is not None:
        load_tunable_params_path = Path(load_tunable_params_path)
        load_tunable_params_path = (
            load_tunable_params_path / cfg.concept_data_module / f"model-params-{cfg.metric}.pkl"
        )

    save_tunable_params_path = cfg.get("save_tunable_params_path", None)
    if save_tunable_params_path is not None:
        save_tunable_params_path = Path(save_tunable_params_path)
        save_tunable_params_path = (
            save_tunable_params_path / cfg.concept_data_module / f"model-params-{cfg.metric}.pkl"
        )

    load_probe_path = cfg.get("load_probe_path", None)
    load_probe_path_direct = cfg.get("load_probe_path_direct", None)
    if load_probe_path_direct is not None:
        if not Path(load_probe_path_direct).exists():
            raise FileNotFoundError(
                f"load_probe_path_direct {load_probe_path_direct} does not exist"
            )
    if load_probe_path is not None:
        load_probe_path = Path(load_probe_path)
        load_probe_path = load_probe_path / cfg.concept_data_module / f"{cfg.metric}.pkl"
    save_probe_path = cfg.get("save_probe_path", None)
    if save_probe_path is not None:
        save_probe_path = Path(save_probe_path)
        save_probe_path = save_probe_path / cfg.concept_data_module / f"{cfg.metric}.pkl"

    load_pca_path = cfg.get("load_pca_path", None)
    save_pca_path = cfg.get("save_pca_path", None)

    # Validate load paths if they are provided, and ensure save paths don't already exist
    for path_name, path in [
        ("load_tunable_params_path", load_tunable_params_path),
        ("save_tunable_params_path", save_tunable_params_path),
        ("save_probe_path", save_probe_path),
        ("load_probe_path", load_probe_path),
        ("save_pca_path", save_pca_path),
        ("load_pca_path", load_pca_path),
    ]:
        if path is not None:
            path = str(path)
            if path_name.startswith("load_"):
                path_obj = Path(path)

                if path_name != "load_pca_path":
                    # pca does not depend on attack_defense_epochs
                    path_obj = convert_path(path_obj, attack_defense_epochs=0)

                if not path_obj.exists():
                    raise FileNotFoundError(f"{path_name} {path} does not exist")
            elif path_name.startswith("save_"):
                path_obj = Path(path)

                if path_name != "save_pca_path":
                    # pca does not depend on attack_defense_epochs
                    path_obj = convert_path(path_obj, attack_defense_epochs=0)

                if path_obj.exists():
                    raise FileExistsError(
                        f"{path_name} {path} already exists. Choose a different path to avoid overwriting."
                    )
                if not path_obj.parent.exists():
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory structure for {path_name}: {path_obj.parent}")
                if path_obj.is_dir():
                    raise IsADirectoryError(
                        f"{path_name} {path} is a directory. It should be a path to a file."
                    )

    # Validate model_config
    model_config = validate_and_create_model_config(cfg.model_config)

    # Validate optimizer_cfg
    optimizer_cfg = validate_and_create_optimizer_config(
        cfg.optimizer_config,
        prompt_init=model_config.prompt_init,  # type: ignore
        seed=seed,
    )

    # Validate metric_cfg
    metric_cfg = validate_and_create_metric_config(cfg.metric_config)

    # Instantiate objects
    model_path = Path(cfg.model_path)
    model = model_cls.load_model(model_path, model_config)

    example_idx = cfg.get("example_idx", None)
    if example_idx is not None:
        example_idx = int(example_idx)

    concept_data_module = concept_data_module_cls(
        dataset_path=cfg.concept_data_path,
        batch_size=cfg.batch_size,
        example_idx=example_idx,
    )
    obfus_data_module = obfus_data_module_cls(
        dataset_path=cfg.obfus_data_path,
        batch_size=cfg.batch_size,
        example_idx=example_idx,
    )

    # Instantiate metric with new parameters, if we can load up the probe
    # and pca we do so.
    if load_probe_path_direct is not None:
        path = load_probe_path_direct
    else:
        path = convert_path(load_probe_path, attack_defense_epochs=0)

    obf_metric = metric_cls(
        model,
        concept_data_module,
        metric_cfg,
        logger,
        probe_path=path,
        pca_path=load_pca_path,
    )

    loss_fn = loss_cls(obf_metric, cfg.lambda_, logger)
    optimizer = optimizer_cls(model, loss_fn, logger, optimizer_cfg)

    # Add validation for train_metric_only
    if "train_metric_only" not in cfg:
        raise ValueError("'train_metric_only' not provided in config.")
    if not isinstance(cfg.train_metric_only, bool):
        raise ValueError("'train_metric_only' must be a boolean value")

    if load_tunable_params_path_direct is not None:
        load_tunable_params_path = load_tunable_params_path_direct
    if load_probe_path_direct is not None:
        load_probe_path = load_probe_path_direct

    return ExperimentConfig(
        model_path=cfg.model_path,
        obfus_data_path=cfg.obfus_data_path,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        model_config=model_config,
        model=model,
        optimizer=optimizer,
        optimizer_config=optimizer_cfg,
        lambda_=cfg.lambda_,
        metric=obf_metric,
        loss=loss_fn,
        obfus_data_module=obfus_data_module,
        concept_data_module=concept_data_module,
        concept_data_path=cfg.concept_data_path,
        logger=logger,
        epochs=epochs,
        seed=seed,
        attack_defense_epochs=cfg.attack_defense_epochs,
        gen_len=cfg.gen_len,
        reinit_tunable_params=cfg.reinit_tunable_params,
        save_tunable_params_path=save_tunable_params_path,
        load_tunable_params_path=load_tunable_params_path,
        load_probe_path=load_probe_path,
        save_probe_path=save_probe_path,
        load_pca_path=load_pca_path,
        save_pca_path=save_pca_path,
        train_metric_only=cfg.train_metric_only,
    )


def gen_return_to_LAT_scans(
    model: ModelBase,
    gen_returns: List[GenReturn],
    obf_metric: ObfMetric,
) -> Tuple[List[Tensor], List[Figure]]:

    scores = []
    for gen_return in gen_returns:

        score = obf_metric.evaluate(gen_return.gen_reps).detach().cpu().numpy()
        scores.append(score)

    min_score = min([float(np.min(score)) for score in scores])
    max_score = max([float(np.max(score)) for score in scores])

    figs = []
    for gen_return, score in zip(gen_returns, scores):

        gen_token_text = [[model.tokenizer.decode(x) for x in gen_return.gen_ids.squeeze()]]

        fig = plot_lat_scan(
            scores=score,
            text_tokens=gen_token_text,
            vmin=min_score,
            vmax=max_score,
        )[0]

        figs.append(fig)

    return scores, figs


def get_test_cached_reps(
    experiment_cfg: ExperimentConfig,
    label: str,
) -> TestCachedReps:

    data_module: ConceptDataModule = experiment_cfg.concept_data_module
    metric: ObfMetric = experiment_cfg.metric

    validation_dataset: ConceptDataset = data_module.val_dataset + data_module.test_dataset  # type: ignore
    test_reps = metric.get_reps_from_dataset(
        experiment_cfg.model,
        validation_dataset,
        metric.train_reps_reduce,
        use_tunable_params=True,
    )

    obf_reps = [test_rep for test_rep, label in test_reps if label == 1]

    return TestCachedReps(label=label, reps=obf_reps)
