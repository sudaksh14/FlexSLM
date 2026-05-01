import torch
from torch import nn

import flex_modules as fm
import utils


class ModelConfig(utils.SelfDescripting):
    """
    ModelConfig defines an interface followed by configuration classes of all models. 
    """

    def make_model(self) -> nn.Module:
        """
        Constructs a model according to this config
        """
        raise NotImplementedError()

    def no_prebuilt(self) -> 'ModelConfig':
        """
        Turns of the loading of pretrained weights
        """
        raise NotImplementedError()


class FlexModelConfig(ModelConfig):
    """
    FlexModelConfig extends the ModelConfig interface.
    FlexModelConfig is followed by configuration classes of all flexible models. 
    """

    def make_model(self) -> fm.Module:
        """
        Constructs a model according to this config
        """
        raise NotImplementedError()

    def no_prebuilt(self) -> 'FlexModelConfig':
        """
        Turns of the loading of pretrained weights
        """
        raise NotImplementedError()

    def create_base_config(self, level) -> ModelConfig:
        """
        Creates a regular model config for the non flexible counterpart of this model.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the maximum level the flexible model will have.
        """
        raise NotImplementedError()


torch.serialization.add_safe_globals([ModelConfig, FlexModelConfig])
