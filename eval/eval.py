import numpy as np
import sys
import time
import utils

from networks import flexvit, flexdeit_v3
from networks.vit_v3 import ViTPrebuilt

import torch_pruning as tp
import torch
import timm


def load_flexvit_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    print(ckpt.keys())
    sdict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sdict, strict=True)
    return model

def remap_timm_to_flexvit(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        new_k = k

        # --- Top-level tokens ---
        new_k = new_k.replace("cls_token", "class_token.token")
        new_k = new_k.replace("pos_embed", "encoder.pos_embedding.embedding")
        new_k = new_k.replace("patch_embed.proj.weight", "conv_proj.weight")
        new_k = new_k.replace("patch_embed.proj.bias", "conv_proj.bias")
        new_k = new_k.replace("head.weight", "heads.head.weight")
        new_k = new_k.replace("head.bias", "heads.head.bias")
        
        # Norm Layer
        new_k = new_k.replace("norm.weight", "encoder.ln.weight")
        new_k = new_k.replace("norm.bias", "encoder.ln.bias")

        # --- Encoder blocks ---
        if new_k.startswith("blocks."):
            new_k = new_k.replace("blocks.", "encoder.layers.encoder_layer_")

            # Normalization
            new_k = new_k.replace(".norm1.", ".ln_1.")
            new_k = new_k.replace(".norm2.", ".ln_2.")
            
            # Attention
            new_k = new_k.replace(".attn.qkv.", ".self_attention.in_proj_")
            new_k = new_k.replace(".attn.proj.", ".self_attention.out_proj.")

            # MLP
            new_k = new_k.replace(".mlp.fc1.", ".mlp.0.")
            new_k = new_k.replace(".mlp.fc2.", ".mlp.3.")
            

        remapped[new_k] = v

    return remapped

def remap_deitv3_to_flexvit(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        new_k = k

        # --- Top-level tokens ---
        new_k = new_k.replace("cls_token", "class_token.token")
        new_k = new_k.replace("pos_embed", "pos_embedding.embedding")
        new_k = new_k.replace("patch_embed.proj.weight", "conv_proj.weight")
        new_k = new_k.replace("patch_embed.proj.bias", "conv_proj.bias")
        new_k = new_k.replace("head.weight", "heads.head.weight")
        new_k = new_k.replace("head.bias", "heads.head.bias")

        # --- Encoder blocks ---
        if new_k.startswith("blocks."):
            new_k = new_k.replace("blocks.", "encoder.layers.encoder_layer_")

            # Normalization
            new_k = new_k.replace(".norm1.", ".ln_1.")
            new_k = new_k.replace(".norm2.", ".ln_2.")

            # LayerScale
            new_k = new_k.replace(".ls1.", ".ls1.")
            new_k = new_k.replace(".ls2.", ".ls2.")

            # Attention
            new_k = new_k.replace(".attn.qkv.", ".self_attention.in_proj_")
            new_k = new_k.replace(".attn.proj.", ".self_attention.out_proj.")

            # MLP
            new_k = new_k.replace(".mlp.fc1.", ".mlp.0.")
            new_k = new_k.replace(".mlp.fc2.", ".mlp.3.")

        remapped[new_k] = v

    return remapped

def compare_state_dicts(model, checkpoint_path):
    # Load checkpoint
    # ckpt = torch.load(checkpoint_path, map_location=utils.get_device())
    # ckpt_state = ckpt.get("state_dict", ckpt)  # handle both Lightning & raw dict
    # ckpt_state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_state = checkpoint_path

    # print(ckpt_state.keys())

    # Remap checkpoint keys
    # ckpt_remapped = remap_deitv3_to_flexvit(ckpt_state)
    ckpt_remapped = remap_timm_to_flexvit(ckpt_state)

    model_state = model.state_dict()

    # Find matches, missing, and unexpected
    common_keys = set(model_state.keys()) & set(ckpt_remapped.keys())
    missing = set(model_state.keys()) - set(ckpt_remapped.keys())
    unexpected = set(ckpt_remapped.keys()) - set(model_state.keys())

    print(f"✅ Matching keys: {len(common_keys)}")
    print(f"❌ Missing in checkpoint: {len(missing)}")
    print(f"⚠️ Unexpected in checkpoint: {len(unexpected)}")

    print("\n--- Matching keys ---")
    for k in list(common_keys)[:20]:  # only show first 20
        print(k)

    if missing:
        print("\n--- Missing keys ---")
        for k in list(missing)[:20]:  # only show first 20
            print(k)

    if unexpected:
        print("\n--- Unexpected keys ---")
        for k in list(unexpected)[:20]:
            print(k)

    return ckpt_remapped

def load_flexvit_weights(flexvit_model, remapped_state_dict, verbose=1):
    model_dict = flexvit_model.state_dict()
    matched, skipped = {}, []

    for k, v in remapped_state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
            print(f"⚠️ Skipped {k}: {model_dict[k].shape} vs {v.shape}")

    model_dict.update(matched)
    flexvit_model.load_state_dict(model_dict)

    if verbose:
        print(f"✅ Loaded {len(matched)} keys, skipped {len(skipped)}.")
        if skipped:
            print("⚠️ Skipped examples:", skipped[:10])

FLEXVIT_CONFIG_V1 = flexvit.ViTConfig(
    prebuilt=flexvit.ViTPrebuilt.noprebuild,
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

FLEXVIT_CONFIG_V3 = flexdeit_v3.ViTConfig_v3(
    prebuilt=ViTPrebuilt.Deit_v3_pretrain_21k,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

# This script generates the table with the delta file switching timings
if __name__ == "__main__":
    device = utils.get_device()
    model_orig = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, num_classes=1000)
    # model_orig = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True, num_classes=1000)
    # print(len(model_orig.state_dict().keys()))
    # model_orig = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True, num_classes=1000)

    # for name, param in model.state_dict().items():
    #     print(f"{name}: {param.shape}")

    # model = FLEXVIT_CONFIG_V1.make_model()
    model = FLEXVIT_CONFIG_V3.no_prebuilt().make_model()
    # model = FLEXVIT_CONFIG_V3.make_model()
    # reg_model = model.make_base_copy()
    # print(model.state_dict().keys())
    # compare_state_dicts(reg_model, model_orig.state_dict())
    # load_flexvit_weights(reg_model, remap_deitv3_to_flexvit(model_orig.state_dict()))
    # load_flexvit_weights(reg_model, remap_timm_to_flexvit(model_orig.state_dict()))
    # model.set_level_use(model.max_level())
    # model.load_from_base(reg_model)
    

    model.eval().to(device)

    # _,_,test_loader = utils.load_imagenet(batch_size=512)
    _,_,test_loader = utils.load_dummy_data(batch_size=512)

    # print("Evaluating full Regular model")
    # acc = utils.evaluate_model(reg_model, test_loader, device)
    # flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    # print(f"Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")

    
    print("Using DeiT v3 weights with Imagenet-21k pretraining")
    for i in range(0, model.max_level() + 1):
        model.set_level_use(i)
        reg_model = model.make_base_copy()
        acc = utils.evaluate_model(reg_model, test_loader, device)
        flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
        print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")

    
    # paths = ["./pretrained/flexxxxvit_distill.pt",
    #          "/ivi/xfs/skalra/checkpoints/flexvit_distill_best_model.ckpt",
    #          "/ivi/xfs/skalra/checkpoints/flexvit_distill_best_model-v1.ckpt"]
        
    # print(f"Using FlexViT saved weights at {paths[2]}")
    # model = load_flexvit_model(model1, ckpt_path=paths[2], device=device)
    # for i in range(model.max_level() + 1):
    #     model.set_level_use(i)
    #     reg_model = model.make_base_copy()
    #     acc = utils.evaluate_model(reg_model, test_loader, device)
    #     flops, param = tp.utils.count_ops_and_params(reg_model, torch.randn(1,3,224,224).to(device))
    #     print(f"Level {i} Accuracy: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}")
    

    
    
    