from dataclasses import dataclass, field
from typing import Optional

from obf_reps.data import (
    ConceptDataModule,
    HarmfulConceptDataModule,
    JailbreakObfusDataModule,
    ObfusDataModule,
)
from obf_reps.logging import DummyLogger, Logger
from obf_reps.metrics import MeanDiffCosineSimMetric, ObfMetric
from obf_reps.models.llama3 import Llama3SoftPrompted, ModelBase, ModelConfig
from obf_reps.optimize.loss import FinalTokenObfLoss, LossFunctionBase
from obf_reps.optimize.optimizers import AdamOptimizer, OptimizerBase, OptimizerConfig


@dataclass
class ExperimentConfig:

    # REQUIRED FIELDS
    # ----- Model ----- #
    model_path: str

    # ----- Data ----- #
    obfus_data_path: str
    concept_data_path: Optional[str]
    batch_size: int
    eval_batch_size: int
    obfus_data_module: ObfusDataModule
    concept_data_module: ConceptDataModule
    gen_len: int

    # ----- Model ----- #
    model: ModelBase
    model_config: ModelConfig
    reinit_tunable_params: bool

    # ----- Optimizer ----- #
    lambda_: float
    optimizer: OptimizerBase
    optimizer_config: OptimizerConfig
    epochs: int
    seed: int

    # ----- Metrics ----- #
    metric: ObfMetric
    attack_defense_epochs: int

    # ----- Loss ----- #
    loss: LossFunctionBase

    # ----- Training ----- #
    train_metric_only: bool

    # ----- Logging ----- #
    logger: Logger
    save_tunable_params_path: str
    load_tunable_params_path: str
    load_probe_path: str
    save_probe_path: str
    load_pca_path: str
    save_pca_path: str
