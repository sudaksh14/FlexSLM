#!/usr/bin/python3
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import wandb
import utils

import config.paths as paths
import scripts.run_experiment as run_experiment
import training


class Wandb:
    api: wandb.Api = None

    @staticmethod
    def get() -> wandb.Api:
        if __class__.api is None:
            __class__.api = wandb.Api()
        return __class__.api


def get_experiment(name: str):
    print(f"retrieving experiment '{name}'", file=sys.stderr)
    experiment: training.TrainerBuilder = run_experiment.resolve_from_str(name)
    if not isinstance(experiment, training.TrainerBuilder):
        return
    entity = Wandb.get().default_entity
    project = experiment.training_context.wandb_project_name
    runs = Wandb.get().runs(f"{entity}/{project}", order="-created_at")
    run = next(filter(lambda r: r.name == "FlexViT_5Levels_nonuniform_ivi", runs))
    return experiment, run


def plot_acc_history(name: str, stage: str = "val"):
    print(
        f"plotting acc history of experiment '{name}' as stage '{stage}'", file=sys.stderr)
    exp, run = get_experiment(name)

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break
        history = run.history(keys=["epoch", key], pandas=True)
        x = history["epoch"]
        y = history[key]

        reg_config = exp.model_config.create_base_config(i)
        model = reg_config.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        plt.ylim(0.0, 1.0)
        plt.plot(x, y, label=f"{params // 100000 / 10}M parameters")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")


def plot_acc(name: str, stage: str = "val", relative_size=False, label=None, base_accuracy=None):
    print(
        f"plotting accuracies of experiment '{name}' at stage '{stage}'", file=sys.stderr)
    exp, run = get_experiment(name)
    size_to_acc = dict()

    for i in itertools.count(0):
        key = f"{stage}_level{i}_acc"
        if key not in run.summary.keys():
            break

        y = run.summary[key]

        reg_config = exp.model_config.create_base_config(i)
        model = reg_config.no_prebuilt().make_model()
        params = utils.count_parameters(model)

        size_to_acc[params / 1000000] = y

    if base_accuracy is None:
        _, _, test_loader = exp.training_context.loader_function()
        model = exp.model_config.create_base_config(
            exp.model_config.max_level()).make_model()

        if isinstance(exp.training_context, training.FlexTrainingContext):
            if exp.training_context.load_from is not None:
                lmodel = utils.load_model(exp.training_context.load_from)
                utils.flexible_model_copy(lmodel, model)

        acc = utils.evaluate_model(model, test_loader, utils.get_device())
    else:
        acc = base_accuracy

    keys = list(size_to_acc.keys())
    if relative_size:
        m = max(*keys)
        keys = [k / m for k in keys]

    max_x = max(*keys)
    min_x = min(*keys)

    lines = plt.plot([min_x, max_x], [acc, acc], ls='--')
    plt.plot(keys, size_to_acc.values(),
             marker='o', color=lines[0].get_color(), label=label)
    plt.xlabel(
        "model parameter count relative to full model" if relative_size else "model size in millions of parameters")
    plt.ylabel("top 1 accuracy")
    plt.ylim(0.0, 1.0)


def moving_avg(axis, data, size=10):
    if len(data) < size:
        return np.array([]), np.array([])
    data = np.array(data)
    mavg = np.convolve(data, np.ones(
        (size,), dtype=data.dtype) / size, 'valid')
    axis = np.array(axis)
    axis = axis[size // 2: -(size // 2) + ((size + 1) % 2)]
    return axis, mavg


def plot_acc_val_and_train(name, level):
    print(
        f"plotting validation and train accuracies of '{name}' at level {level}", file=sys.stderr)

    exp, run = get_experiment(name)

    train_key = f"train_level{level}_acc"
    val_key = f"val_level{level}_acc"

    train = run.history(keys=["epoch", train_key], pandas=True)
    print(train)
    val = run.history(keys=["epoch", val_key], pandas=True)

    train_lines = plt.plot(
        train['_step'], train[train_key], ls='--', alpha=0.5, label="training accuracy")
    val_lines = plt.plot(val['_step'], val[val_key],
                         ls='--', alpha=0.5, label="validation accuracy")

    axs, mavg = moving_avg(train['_step'], train[train_key], 20)
    plt.plot(axs, mavg, color=train_lines[0].get_color())

    axs, mavg = moving_avg(val['_step'], val[val_key], 5)
    plt.plot(axs, mavg, val_lines[0].get_color())

    plt.xlabel("training steps")
    plt.ylabel("top 1 accuracy")
    plt.legend()


def plot_loss_val_and_train(name):
    print(f"plotting validation and train loss of '{name}'", file=sys.stderr)

    exp, run = get_experiment(name)
    train_key = "train_loss"
    val_key = "val_loss"

    train = run.history(keys=["epoch", train_key], pandas=True)
    print(train)
    val = run.history(keys=["epoch", val_key], pandas=True)

    train_lines = plt.plot(
        train['_step'], train[train_key], ls='--', alpha=0.5, label="training loss")
    val_lines = plt.plot(val['_step'], val[val_key],
                         ls='--', alpha=0.5, label="validation loss")

    axs, mavg = moving_avg(train['_step'], train[train_key], 20)
    plt.plot(axs, mavg, color=train_lines[0].get_color())

    axs, mavg = moving_avg(val['_step'], val[val_key], 5)
    plt.plot(axs, mavg, val_lines[0].get_color())

    plt.xlabel("training steps")
    plt.ylabel("cross entropy loss")
    plt.legend()


def savefig(name: str):
    print(f"saving figure '{name}'")
    plt.savefig(paths.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    matplotlib.rc("font", size=13)

    # plot_acc_history("flexvit,imagenet")
    # savefig("vit_imagenet_history")

    # plot_acc("flexvit,imagenet", base_accuracy=.81)
    # savefig("vit_imagenet_acc")

    # plot_acc_val_and_train("flexvit,imagenet", 4)
    # savefig("overfitting")

    # plot_loss_val_and_train("flexvit,imagenet")
    # savefig("overfitting_loss")

    # plot_acc_history("flexvit,cifar10.5levels")
    # savefig("cifar10_acc_history")

    # plot_acc("flexvit,cifar10.5levels", base_accuracy=.98)
    # savefig("cifar10_acc")

    # plot_acc("flexresnet,resnet20.6_levels.cifar10",
    #          relative_size=True, label="Resnet 20")
    # plot_acc("flexresnet,resnet56.6_levels.cifar10",
    #          relative_size=True, label="Resnet 56")
    # plot_acc("flexvgg,vgg11.6_levels.cifar10",
    #          relative_size=True, label="VGG 11")
    # plot_acc("flexvgg,vgg19.6_levels.cifar10",
    #          relative_size=True, label="VGG 19")
    # plt.legend()
    # savefig("cnn_cifar10_acc")

    # plot_acc("flexresnet,resnet20.6_levels.cifar100",
    #          relative_size=True, label="Resnet 20")
    # plot_acc("flexresnet,resnet56.6_levels.cifar100",
    #          relative_size=True, label="Resnet 56")
    # plot_acc("flexvgg,vgg11.6_levels.cifar100",
    #          relative_size=True, label="VGG 11")
    # plot_acc("flexvgg,vgg19.6_levels.cifar100",
    #          relative_size=True, label="VGG 19")
    # plt.legend()
    # savefig("cnn_cifar100_acc")
    
    plot_acc_history("flexvit,imagenet_non_uniform_heads")
    savefig("vit_imagenet_history")

    plot_acc("flexvit,imagenet_non_uniform_heads", base_accuracy=.81)
    savefig("vit_imagenet_acc")

    plot_acc_val_and_train("flexvit,imagenet_non_uniform_heads", 4)
    savefig("overfitting")

    plot_loss_val_and_train("flexvit,imagenet_non_uniform_heads")
    savefig("overfitting_loss")
