#!/usr/bin/python3
from typing import Callable, Generator
import sys

from config.hardware import HARDWARE, DEFAULT_HARDWARE_CONFIG
from config.experiments import CONFIGS
from training import BaseTrainer
import config.hardware


def resolve_from_str(config, start=CONFIGS, return_on_index_error=False) -> Callable[[], BaseTrainer]:
    config = config.split(',')
    subpart = start
    for i in config:
        if i == 'all':
            continue
        try:
            i = int(i)
        except ValueError:
            pass
        try:
            subpart = subpart[i]
        except (KeyError, TypeError) as e:
            if not return_on_index_error:
                raise e
            return subpart
    return subpart


def iter_over_conf(conf, basestr) -> Generator[str, None, None]:
    if isinstance(conf, dict):
        for key, val in conf.items():
            for s in iter_over_conf(val, basestr + f",{key}"):
                yield s
    elif isinstance(conf, list):
        for idx, val in enumerate(conf):
            for s in iter_over_conf(val, basestr + f",{idx}"):
                yield s
    else:
        yield basestr


def print_all_conf_paths(conf, basestr, file=sys.stdout) -> None:
    for s in iter_over_conf(conf, basestr):
        print(s, file=file)


def print_all_conf_commands(conf, basestr, file=sys.stdout) -> None:
    for s in iter_over_conf(conf, basestr):
        hconf = resolve_from_str(s, HARDWARE, return_on_index_error=True)
        if not isinstance(hconf, config.hardware.HardwareConfig):
            hconf = DEFAULT_HARDWARE_CONFIG
        print(f"{hconf.format_as_slurm_args()} experiment_job.sh {s}", file=file)


if __name__ == "__main__":
    args = sys.argv[1:]
    resume = True
    if '--no-resume' in args:
        args.remove('--no-resume')
        resume = False
    command, conf = args
    res = resolve_from_str(conf)
    if command == "list":
        print_all_conf_paths(res, conf)
    elif command == "run":
        hw = resolve_from_str(
            conf, HARDWARE, return_on_index_error=True)
        if isinstance(hw, config.hardware.HardwareConfig):
            config.hardware.CurrentDevice.set_hardware(hw)
        else:
            config.hardware.CurrentDevice.set_hardware(DEFAULT_HARDWARE_CONFIG)
        res.training_context.resume = resume
        res(conf)
    elif command == "listcommand":
        print_all_conf_commands(res, conf)
