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

TRAINED_MODELS = make_path(f"{HOME}/FlexViT/saved_models" if ON_CLUSTER else PROJECT_DIR / 'pretrained')
CHECKPOINT_PATH = make_path(f"{HOME}/FlexViT/checkpoints" if ON_CLUSTER else PROJECT_DIR / 'checkpoints')

FIGURES = make_path(PROJECT_DIR / 'figures')

# IMAGENET_PATH = Path("/ssdstore/ImageNet")
# IMAGENET_PATH = Path("/home/dkoelma1/VisualSearch")
IMAGENET_PATH = Path("/nvmestore/skalra/imagenet")

