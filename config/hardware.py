import dataclasses
import utils


@dataclasses.dataclass
class HardwareConfig(utils.SelfDescripting):
    """
    Hardware configuration of the node(s) running an experiment, used by
    `run_experiment.py listcommand` to print sbatch arguments.

    Defaults follow the DAS-6 rules: GPU jobs on the `defq` partition with
    --gres=gpu:N, and never a memory request (every job owns its whole node).
    """
    partition: str = 'defq'
    gpu_count: int = 1
    time: str = '24:00:00'

    process_group_backend: str = 'nccl'

    # do not change this as training code is currently not set up to handle multiple nodes
    node_count: int = 1

    def format_as_slurm_args(self) -> str:
        """
        Formats the configuration as sbatch/srun arguments.
        """
        return (f"--nodes {self.node_count} -p {self.partition} --time {self.time} "
                f"--gres=gpu:{self.gpu_count} --ntasks-per-node={self.gpu_count}")


class CurrentDevice:
    hardware: HardwareConfig = None

    @staticmethod
    def set_hardware(hw: HardwareConfig):
        CurrentDevice.hardware = hw

    @staticmethod
    def get_hardware() -> HardwareConfig:
        return CurrentDevice.hardware


DEFAULT_HARDWARE_CONFIG = HardwareConfig()
HARDWARE = {
    "flexllama": HardwareConfig(gpu_count=1, time='48:00:00'),
}
