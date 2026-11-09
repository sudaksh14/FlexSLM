from pathlib import Path
import os


def make_path(path: str) -> Path:
    try:
        os.makedirs(path)
    except (FileExistsError, PermissionError):
        pass
    return Path(path)


def make_file(path: str) -> Path:
    make_path(os.path.dirname(path))
    return Path(path)


ON_CLUSTER = 'SLURM_JOBID' in os.environ

HOME = os.environ.get('HOME', default='.')
TMPDIR = os.environ.get('TMPDIR', default='./tmp')
SLURM_JOBID = int(os.environ.get('SLURM_JOBID', default='0'))

PROJECT_DIR = make_path('./' if ON_CLUSTER else './.tmp')
TMPDIR = make_path(TMPDIR if ON_CLUSTER else (PROJECT_DIR / 'temp'))

# JOBDIR = make_path(f"{PROJECT_DIR}/jobs/job_{SLURM_JOBID}")
JOBDIR = make_path(f"{PROJECT_DIR}/wandb/job_{SLURM_JOBID}")

DATA_PATH = make_path(f"{PROJECT_DIR}/data")
LOG_PATH = make_path(JOBDIR / 'logs')

# Large files (checkpoints, saved models, derived corpora, W&B logs). On DAS-6 the
# home quota is small, so job scripts export FLEXSLM_STORE=/var/scratch/$USER/flexslm
# and everything heavy goes there. Without it the old Snellius layout is used.
STORE = os.environ.get('FLEXSLM_STORE')

# Cluster-wide NFS scratch (same mount on every node) - for large derived data
# like the prepared FineWeb-Edu .bin files.
if STORE:
    SCRATCH_PATH = make_path(f"{STORE}/scratch")
    TRAINED_MODELS = make_path(f"{STORE}/saved_models")
    CHECKPOINT_PATH = make_path(f"{STORE}/checkpoints")
    JOBDIR = make_path(f"{STORE}/wandb/job_{SLURM_JOBID}")
    LOG_PATH = make_path(JOBDIR / 'logs')
else:
    SCRATCH_PATH = make_path("/scratch/" + os.environ.get('USER', 'skalra') if ON_CLUSTER else PROJECT_DIR / 'scratch')
    TRAINED_MODELS = make_path(f"{HOME}/FlexSLM/saved_models" if ON_CLUSTER else PROJECT_DIR / 'pretrained')
    CHECKPOINT_PATH = make_path(f"{HOME}/FlexSLM/checkpoints" if ON_CLUSTER else PROJECT_DIR / 'checkpoints')

FIGURES = make_path(PROJECT_DIR / 'figures')
