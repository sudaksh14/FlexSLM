"""
Training package. The trainer classes live in training/trainer.py and are
re-exported here, so `import training` / `from training import FlexLMKDTrainer`
keep working, and Lightning checkpoints that pickled classes under the module
name "training" still load.

Entry point (run from the repository root):
    python -m training.run_experiment run flexllama,<config>
"""
from training.trainer import *  # noqa: F401,F403
from training.trainer import (  # explicit names for readers and linters
    TrainingContext, BaseTrainer, TrainerBuilder, FlexTrainingContext,
    FlexModelTrainer, FlexLMTrainer, FlexLMKDTrainer, finetune,
)
