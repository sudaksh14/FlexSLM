import flex_modules as fm
from torch import nn
import copy
import matplotlib.pyplot as plt

import mixbox
import colorsys
import numpy as np
import tqdm


def fill(model: nn.Module, value=0.0):
    for p in model.parameters():
        p.data[:] = p.data.new_full(p.shape, value)


def create_ones_submodel(model: fm.Module, level: int):
    model = copy.deepcopy(model)
    model.set_level_use(level)
    fill(model, 0.0)
    base = model.make_base_copy()
    fill(base, 1.0)
    model.load_from_base(base)
    return model


def add_to_model(acc: nn.Module, v: nn.Module):
    for pa, pv in zip(acc.parameters(), v.parameters()):
        pa.data[:] = pa.data[:] + pv.data[:]
    return acc


def sum_sublevels_effects(model: fm.Module) -> fm.Module:
    fill(model, 0.0)
    for i in range(model.max_level() + 1):
        ones = create_ones_submodel(model, i)
        model = add_to_model(model, ones)
    return model


def make_color_map(data):
    maxv = data.max()
    minv = data.min()

    def make_color(v):
        f = (v - minv) / maxv
        color1 = (254, 236, 0)
        color2 = (128, 2, 46)
        c = mixbox.lerp(color1, color2, float(f))
        c = np.array(c)
        return c

    colors = np.zeros((*data.shape, 3), dtype=np.int_)
    for i in tqdm.tqdm(range(data.shape[0])):
        for j in tqdm.tqdm(range(data.shape[1]), leave=False):
            colors[i, j] = make_color(data[i, j])
    return colors


def make_model_colormap(model):
    sum_sublevels_effects(model)
    weights = make_color_map(model.out_weights.detach())
    plt.imshow(weights)
    plt.show()


def make_model_param_colormap(embed_size, num_heads):
    return make_model_colormap(fm.SelfAttention(embed_size, num_heads))


make_model_param_colormap(
    embed_size=[25, 30, 56, 64, 90, 100],
    num_heads=[5, 5, 8, 8, 10, 10])

# make_model_param_colormap(
#     embed_size=(
#         4 * 8,
#         4 * 12,

#         5 * 8,
#         5 * 12,

#         6 * 8,
#         6 * 12,

#         7 * 8,
#         7 * 12,

#         8 * 8,
#         8 * 12
#     ),
#     num_heads=(
#         8,
#         12,
#         8,
#         12,
#         8,
#         12,
#         8,
#         12,
#         8,
#         12
#     ))

# This is the map for the config in my thesis
# make_model_param_colormap(
#     embed_size=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
#     num_heads=(12, 12, 12, 12, 12))

# You could use something like this to have both variation in embedding size and head size
# make_model_param_colormap(
#     embed_size=(
#         32 * 8,
#         32 * 12,

#         40 * 8,
#         40 * 12,

#         48 * 8,
#         48 * 12,

#         56 * 8,
#         56 * 12,

#         64 * 8,
#         64 * 12
#     ),
#     num_heads=(
#         8,
#         12,
#         8,
#         12,
#         8,
#         12,
#         8,
#         12,
#         8,
#         12
#     ))
