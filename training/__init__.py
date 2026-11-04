"""
Trainers and training contexts. `import training` / `from training import X`
keeps working as before: everything public in training/training.py is
re-exported here.
"""
import sys

from .training import *  # noqa: F401,F403
from .training import _print_experiment_banner  # noqa: F401
from . import training_memopt  # noqa: F401

# Lightning checkpoints written before the move pickled the memopt trainers as
# module `training_memopt`; keep that name resolvable so they still load/resume.
sys.modules.setdefault("training_memopt", training_memopt)
