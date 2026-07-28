import dataclasses
import utils


@dataclasses.dataclass
class HardwareConfig(utils.SelfDescripting):
    """
    This class stores the hardware configuration of the node(s) running an experiment.
    """
    partition: str = 'gpu_h100'
    gpu_count: int = 1
    time: str = '24:00:00'

    process_group_backend: str = 'nccl'

    # do not change this as training code is currently not set up to handle multiple nodes
    node_count: int = 1

    def format_as_slurm_args(self) -> str:
        """
        This formats the configuration into arguments as passed to sbatch or srun.
        This is used to reserve nodes for experiments with their associated hardware requirements.
        """
        return f"--nodes {self.node_count} -p {self.partition} --time {self.time} --gpus-per-node={self.gpu_count} --ntasks-per-node={self.gpu_count} --mem=0"


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
    "vitprebuild": HardwareConfig(
        gpu_count=4, time='00:30:00'),
    "flexvit": {
        "cifar10.5levels": HardwareConfig(
            gpu_count=4,
            time='12:00:00'),
        "imagenet": HardwareConfig(
            gpu_count=4,
            time='24:00:00'),
        "imagenet_constant_scale": HardwareConfig(
            gpu_count=4,
            time='24:00:00'),
    },
    "flexvitcorrect": HardwareConfig(
        gpu_count=4,
        time='24:00:00'),
    "flexllama": HardwareConfig(
        partition='performance',
        gpu_count=1,
        time='48:00:00'),
}
