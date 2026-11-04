import numpy as np
import sys
import time
import utils

from networks import flexvit
from networks.vit import ViTPrebuilt

import torch_pruning as tp
import torch


FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V2 = flexvit.ViTConfig(
    prebuilt=ViTPrebuilt.imagenet1k_v1,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))


if __name__ == "__main__":

    device = utils.get_device()
    model1 = FLEXVIT_CONFIG.make_model()

    _,_,test_loader = utils.load_imagenet()
    
    
    print("Using Default weights")
    for i in range(model1.max_level(), model1.max_level() + 1):
        model1.set_level_use(i)
        reg_model = model1.make_base_copy()
        acc = utils.evaluate_model(reg_model, test_loader, device)
        flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
        print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
       
    
    
    