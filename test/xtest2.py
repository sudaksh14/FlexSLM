import flex_modules as fm
from torch import nn
import torch
import copy
import matplotlib.pyplot as plt

import mixbox
import colorsys
import numpy as np
import tqdm

import networks.level_delta_utils as levels


def rewrite_delta(delta: fm.LevelDelta):
    f = 1.0

    def mapping(t: torch.Tensor):
        nonlocal f
        t = t.new_full(t.shape, f)
        f += 1.0
        return t
    return delta.map_tensors(mapping)


def fill(model: nn.Module, value=0.0):
    for p in model.parameters():
        p.data[:] = p.data.new_full(p.shape, value)


COLORS = [
    (255, 250, 250),
    (230, 80, 80),
    (80, 230, 80),
    (80, 80, 230),
    (230, 230, 80),
    (230, 80, 230),
    (80, 230, 230),
    (230, 128, 80),
    (128, 200, 230),
    (128, 80, 230),
    (200, 200, 200),
    (50, 50, 50)
]


def make_color_map(data: torch.Tensor):
    def make_color(v: torch.Tensor):
        idx = int(v.round())
        return COLORS[idx]

    if data.ndim == 1:
        data = data.reshape(*data.shape, 1)

    colors = np.zeros((*data.shape, 3), dtype=np.int_)
    for i in tqdm.tqdm(range(data.shape[0])):
        for j in tqdm.tqdm(range(data.shape[1]), leave=False):
            colors[i, j] = make_color(data[i, j])
    return colors


LEVEL_DISPLAYED = 1

model = fm.SelfAttention(
    token_size=[9, 20, 30],
    heads=[3, 5, 6])
fill(model, 0.0)
deltas = levels.get_model_deltas(model)
deltas[(LEVEL_DISPLAYED, True)] = rewrite_delta(
    deltas[(LEVEL_DISPLAYED, True)])

manager = levels.InMemoryDeltaManager(model)
manager.move_to(0)
# fill(manager.managed_model(), len(COLORS) - 2)
manager.deltas = deltas

nn.MultiheadAttention

manager.move_to(manager.max_level())
bmodel: nn.MultiheadAttention = manager.managed_model()

black = torch.full((bmodel.in_proj_weight.shape[0], 1), len(COLORS) - 1)
inproj = torch.cat([black, bmodel.in_proj_weight, black, bmodel.in_proj_bias.reshape(
    *bmodel.in_proj_bias.shape, 1).expand(*bmodel.in_proj_bias.shape, 5), black], dim=1)
black = torch.full((bmodel.out_proj.weight.shape[0], 1), len(COLORS) - 1)
outproj = torch.cat([black, bmodel.out_proj.weight, black, bmodel.out_proj.bias.reshape(
    *bmodel.out_proj.bias.shape, 1).expand(*bmodel.out_proj.bias.shape, 5), black], dim=1)
black = torch.full((1, outproj.shape[1]), len(COLORS) - 1)
total = torch.cat([black , inproj, black, outproj, black], dim=0)
colors = make_color_map(total)
plt.imshow(colors)
plt.xticks([])
plt.yticks([])
plt.show()

colors = make_color_map(bmodel.in_proj_weight.data)
plt.subplot(221)
plt.imshow(colors)

colors = make_color_map(bmodel.in_proj_bias.data)
plt.subplot(222)
plt.xticks([])
plt.imshow(colors, aspect=1/4)

colors = make_color_map(bmodel.out_proj.weight.data)
plt.subplot(223)
plt.imshow(colors)

colors = make_color_map(bmodel.out_proj.bias.data)
plt.subplot(224)
plt.xticks([])
plt.imshow(colors, aspect=1)

plt.show()
