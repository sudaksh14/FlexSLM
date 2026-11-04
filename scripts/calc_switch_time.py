import numpy as np
import sys
from typing import Optional, TextIO, Iterable
import time
import utils

import colorsys

from networks import level_delta_utils as delta
from networks import flexvit
from networks.vit import ViTPrebuilt

import tqdm
import tqdm.contrib.itertools as titer
import torch_pruning as tp
import torch


def make_latex_table(file: TextIO, data: np.ndarray, colors: np.ndarray = None, column_names: Optional[Iterable[str]] = None, row_names: Optional[Iterable[str]] = None):
    rows, cols = data.shape
    print("\\begin{tabular}{", end='', file=file)
    if column_names is not None:
        print("|l|", end='', file=file)
    cols_str = "l".join(['|'] * (cols+1))
    print(f"{cols_str}}}\n\\hline", file=file)
    if column_names is not None:
        print(" & ".join(column_names), end='', file=file)
        if row_names is None:
            print(" \\\\ \\hline")
        else:
            print(" \\\\ \\hhline{|=|", end='', file=file)
            cols_str = "=".join(['|'] * (cols+1))
            print(f"{cols_str}}}", file=file)

    if row_names is None:
        for i in range(rows):
            row = data[i, :]
            if colors is not None:
                color_row = colors[i, :, :]

                def make_cell(c, d):
                    color = ','.join(map(str, c))
                    cmd = f"\\cellcolor[RGB]{{{color}}}"
                    cell = f"{cmd} {d:.2f}"
                    return cell
                print(' & '.join(map(make_cell, color_row, row)),
                      '\\\\ \\hline', file=file)
            else:
                print(' & '.join(map(str, row)), '\\\\ \\hline', file=file)
    else:
        for rname, i in zip(row_names, range(rows)):
            row = data[i, :]
            if colors is not None:
                color_row = colors[i, :, :]

                def make_cell(c, d):
                    color = ','.join(map(str, c))
                    cmd = f"\\cellcolor[RGB]{{{color}}}"
                    cell = f"{cmd} {d:.2f}"
                    return cell
                print(f"{rname} &", ' & '.join(
                    map(make_cell, color_row, row)), '\\\\ \\hline', file=file)
            else:
                print(f"{rname} &", ' & '.join(map(str, row)),
                      '\\\\ \\hline', file=file)
    print('\\end{tabular}', file=file)


def make_color_map(data):
    maxv = data.max()
    minv = data.min()

    def make_color(v):
        f = (v - minv) / maxv
        hue = 10 + 100 * (1 - f)
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.6, 1.0)
        rgb *= np.array([255, 255, 255])
        return np.array(rgb)

    colors = np.zeros((*data.shape, 3), dtype=np.int_)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            colors[i, j] = make_color(data[i, j])
    return colors


# if __name__ == "__main__":
#     points = 20
#     data = np.linspace(0, 1, points).reshape(points, 1)
#     colors = make_color_map(data)
#     make_latex_table(sys.stdout, data, colors)


def time_manager(manager: delta.FileDeltaManager, level_from: int, level_to: int, iters: int):
    total_time = 0

    for i in tqdm.tqdm(range(iters), leave=False):
        manager.move_to(level_from)
        start = time.monotonic_ns()
        manager.move_to(level_to)
        end = time.monotonic_ns()
        total_time += end - start

    milliseconds = total_time / 10e6
    milliseconds /= iters
    return milliseconds


DELTA_FILENAME = "vit.delta"

FLEXVIT_CONFIG = flexvit.ViTConfig(
    num_classes=1000,
    num_heads=(12, 12, 12, 12, 12),
    hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
    mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48))

# This script generates the table with the delta file switching timings
if __name__ == "__main__":
    model = FLEXVIT_CONFIG.make_model()

    # first create this delta file
    with open(DELTA_FILENAME, "wb") as file:
        delta.FileDeltaManager.make_delta_file(file, model, starting_level=0)

    reg_config = FLEXVIT_CONFIG.create_base_config(0).no_prebuilt()
    with delta.file_delta_manager(DELTA_FILENAME, reg_config) as manager:
        manager.set_managed_model(
            manager.managed_model().to(utils.get_device()))
        reg_model = manager.managed_model()

        data = np.zeros((manager.max_level() + 1, manager.max_level() + 1))

        for i, j in titer.product(range(manager.max_level() + 1), range(manager.max_level() + 1)):
            data[i, j] = time_manager(manager, i, j, 100)

        column_names = ['', *map(str, range(manager.max_level() + 1))]
        row_names = map(str, range(manager.max_level() + 1))

        colors = make_color_map(data)

        make_latex_table(sys.stdout, data, colors, column_names, row_names)
