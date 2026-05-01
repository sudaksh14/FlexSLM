import math
from xml.parsers.expat import model

from networks import flexresnet, flexvgg, flexvit, vit, flexdeit_v3
from networks.flexgpt import FlexGPT, FlexGPTConfig
from networks.flexllama import FlexLLaMA, FlexLLaMAConfig
from training import *
from training import FlexLMTrainer
from networks.vit import ViTPrebuilt

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from functools import partial
import torch
import torch.optim as optim
from torchvision.datasets import CIFAR10, CIFAR100
import distillation.training
import distillation.dataset
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler

from training import FlexLMTrainer, FlexLMKDTrainer


class ModelTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10),
                         patience=50, epochs=-1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


class ModelTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR100),
                         patience=50, epochs=-1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


class ViTTraining(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10,
                                 resize=(224, 224)), patience=20, epochs=300, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


class ViTTraining100(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        # super().__init__(partial(utils.load_data, CIFAR100,
        #                          resize=(224, 224)), patience=20, epochs=1, *args, **kwargs)
        super().__init__(distillation.dataset.load_cifar100, patience=20, epochs=150, *args, **kwargs)

    def make_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    def make_scheduler(self, optimizer):
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, self.epochs - 10))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])


class VitTrainingImagenet(FlexTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=20, epochs=150,
                         label_smoothing=0.11, gradient_clip_val=1.0, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer=optimizer, T_max=150, eta_min=1e-8)


class VitTrainingImagenetWarmup(FlexTrainingContext):
    warmup_epochs: int = 30

    def __init__(self, *args, **kwargs):
        super().__init__(utils.load_imagenet, patience=50, epochs=300,
                         label_smoothing=0.11, gradient_clip_val=1.0, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.3)

    def make_scheduler(self, optimizer):
        return SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.033,
                     total_iters=self.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=self.epochs -
                              self.warmup_epochs, eta_min=0.0)
        ], milestones=[self.warmup_epochs])


class GPTTrainingContext(FlexTrainingContext):
    warmup_epochs: int = 10
    num_levels_per_step: int = None  # None = train all levels; set e.g. 2 to sample

    def __init__(self, dataset_name="wikitext-103-raw-v1",
                 max_seq_length=1024, batch_size=8,
                 num_levels_per_step=None,
                 patience=5, epochs=20,
                 *args, **kwargs):
        loader = partial(utils.load_wikitext, dataset_name=dataset_name,
                         max_seq_length=max_seq_length, batch_size=batch_size)
        super().__init__(loader, patience=patience, epochs=epochs, *args, **kwargs)
        self.num_levels_per_step = num_levels_per_step

    def make_optimizer(self, model):
        return optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    def make_scheduler(self, optimizer):
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, self.epochs - self.warmup_epochs), eta_min=1e-5)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])


@dataclasses.dataclass
class FlexLMKDTrainingContext(GPTTrainingContext):
    kd_lambda: float = 0.5
    kd_temperature: float = 2.0
    teacher_hf_model: str = 'gpt2'

    def __init__(self, kd_lambda=0.5, kd_temperature=2.0,
                teacher_hf_model='gpt2',
                dataset="wikitext-103-raw-v1",
                max_seq_length=1024, batch_size=8,
                num_levels_per_step=None, patience=5, epochs=20,
                *args, **kwargs):
        loader = partial(utils.load_wikitext, dataset_name=dataset,
                        max_seq_length=max_seq_length, batch_size=batch_size)
        FlexTrainingContext.__init__(self, loader, patience=patience, epochs=epochs, *args, **kwargs)
        self.warmup_epochs = 2
        self.num_levels_per_step = num_levels_per_step
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher_hf_model = teacher_hf_model


class LLaMATrainingContext(GPTTrainingContext):
    """Training context for FlexLLaMA. Defaults to FineWeb-Edu."""

    def __init__(self, dataset="fineweb-edu", max_seq_length=1024, batch_size=8,
                 num_levels_per_step=None, patience=3, epochs=10,
                 max_examples=150_000, *args, **kwargs):
        if dataset == "fineweb-edu":
            loader = partial(utils.load_fineweb_edu,
                             max_seq_length=max_seq_length,
                             batch_size=batch_size,
                             max_examples=max_examples)
        else:
            loader = partial(utils.load_wikitext, dataset_name=dataset,
                             max_seq_length=max_seq_length,
                             batch_size=batch_size)
        FlexTrainingContext.__init__(self, loader, patience=patience, epochs=epochs, *args, **kwargs)
        self.warmup_epochs = 2
        self.num_levels_per_step = num_levels_per_step


@dataclasses.dataclass
class LLaMAKDTrainingContext(FlexLMKDTrainingContext):
    """KD training context for FlexLLaMA, Uses FineWeb-Edu and llama-160m as teacher."""

    def __init__(self, kd_lambda=1.0, kd_temperature=2.0,
                 teacher_hf_model='JackFram/llama-160m',
                 tokenizer_name='JackFram/llama-160m',
                 dataset="fineweb-edu", max_seq_length=1024, batch_size=8,
                 num_levels_per_step=None, patience=3, epochs=10,
                 max_examples=150_000, *args, **kwargs):
        if dataset == "fineweb-edu":
            loader = partial(utils.load_fineweb_edu,
                             max_seq_length=max_seq_length,
                             batch_size=batch_size,
                             max_examples=max_examples,
                             tokenizer_name=tokenizer_name)
        else:
            loader = partial(utils.load_wikitext, dataset_name=dataset,
                             max_seq_length=max_seq_length,
                             batch_size=batch_size)
        FlexTrainingContext.__init__(self, loader, patience=patience, epochs=epochs, *args, **kwargs)
        self.warmup_epochs = 2
        self.num_levels_per_step = num_levels_per_step
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature
        self.teacher_hf_model = teacher_hf_model


torch.serialization.add_safe_globals([GPTTrainingContext, FlexLMKDTrainingContext, LLaMATrainingContext, LLaMAKDTrainingContext])


CONFIGS = {
    "flexresnet": {
        'resnet20.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(),
            ModelTraining()),
        'resnet20.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_classes=100),
            ModelTraining100()),

        'resnet20.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
            ),
            ModelTraining()),
        'resnet20.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
                num_classes=100),
            ModelTraining100()),

        'resnet56.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9)),
            ModelTraining()),
        'resnet56.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                num_classes=100),
            ModelTraining100()),

        'resnet56.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64)),
            ModelTraining()),
        'resnet56.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexresnet.ResnetConfig(
                num_blocks=(9, 9, 9),
                small_channels=(6, 8, 10, 12, 14, 16),
                mid_channels=(12, 16, 20, 24, 28, 32),
                large_channels=(24, 32, 40, 48, 56, 64),
                num_classes=100),
            ModelTraining100()),
    },
    "flexvgg": {
        'vgg11.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(),
            ModelTraining()),
        'vgg11.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100),
            ModelTraining100()),

        'vgg11.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg11.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining100()),

        'vgg19.3_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19),
            ModelTraining()),
        'vgg19.3_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                num_classes=100),
            ModelTraining100()),

        'vgg19.6_levels.cifar10': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining()),
        'vgg19.6_levels.cifar100': TrainerBuilder(
            FlexModelTrainer,
            flexvgg.VGGConfig(
                version=19,
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)),
            ModelTraining100()),
    }, "vitprebuild": {
        "cifar10": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig(
                num_classes=10),
            ViTTraining()),
        "cifar100": TrainerBuilder(
            SimpleTrainer,
            vit.ViTConfig(
                num_classes=100),
            ViTTraining100())
    }, "flexvit": {
        "cifar10": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=10),
            ViTTraining(
                load_from='vitprebuild,cifar10')
        ),
        "cifar10.5levels": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=10,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            ViTTraining(
                load_from='vitprebuild,cifar10')
        ),
        "cifar100": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=100,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            ViTTraining100()
        ),
        "imagenet": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
            VitTrainingImagenet()
        ),
        "imagenet_non_uniform_heads": TrainerBuilder(
            FlexModelTrainer,
            flexvit.ViTConfig(
                num_classes=1000,
                num_heads=(4, 6, 8, 10, 12),
                hidden_dims=(64 * 4, 64 * 6, 64 * 8, 64 * 10, 64 * 12),
                mlp_dims=(64 * 16, 64 * 24, 64 * 32, 64 * 40, 64 * 48)),
            VitTrainingImagenet())
    }, "flexvitcorrect": TrainerBuilder(
        FlexModelTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        VitTrainingImagenet(
            load_from='flexvit,imagenet')
    ), 'scala_test': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexvgg.VGGConfig(
            num_classes=100,
            small_channels=(24, 32, 40, 48, 56, 64),
            mid_channels=(48, 64, 80, 96, 112, 128),
            large_channels=(96, 128, 160, 192, 224, 256),
            max_channels=(192, 256, 320, 384, 448, 512)),
        distillation.training.ScalaDistillContext(
            loader_function=partial(
                distillation.dataset.load_imagenet,
                data_set='CIFAR',
                datapath=paths.DATA_PATH,
                input_size=32),
            teacher_loader=flexvgg.VGGConfig(
                num_classes=100,
                small_channels=(24, 32, 40, 48, 56, 64),
                mid_channels=(48, 64, 80, 96, 112, 128),
                large_channels=(96, 128, 160, 192, 224, 256),
                max_channels=(192, 256, 320, 384, 448, 512)).make_model,
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=1e-5, weight_decay=0.3),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-8)
        )
    ), 'flexvit_distill': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexvit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        distillation.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(distillation.dataset.load_imagenet, batch_size=512),
            make_optimizer=lambda m: optim.AdamW(
                m.parameters(), lr=5e-4, weight_decay=0.05),
            make_scheduler=lambda opt: CosineAnnealingLR(
                optimizer=opt, T_max=150, eta_min=1e-5),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=150,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ), 'flexdeit_v3': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexdeit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
            mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)),
        distillation.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(distillation.dataset.load_imagenet, batch_size=256),
            make_optimizer=lambda m: Lamb(m.parameters(),
                lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
            make_scheduler=lambda opt: CosineLRScheduler(optimizer=opt,
                t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6,
                warmup_t=5, cycle_limit=1, t_in_epochs=True),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=100,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ), 'flexdeit_v3_head_random': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexdeit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=(4, 6, 8, 10, 12),
            hidden_dims=(64 * 4, 64 * 6, 64 * 8, 64 * 10, 64 * 12),
            mlp_dims=(64 * 16, 64 * 24, 64 * 32, 64 * 40, 64 * 48),
            head_permutation="random"),
        distillation.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(distillation.dataset.load_imagenet, batch_size=256),
            make_optimizer=lambda m: Lamb(m.parameters(),
                lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
            make_scheduler=lambda opt: CosineLRScheduler(optimizer=opt,
                t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6,
                warmup_t=5, cycle_limit=1, t_in_epochs=True),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=100,
            label_smoothing=0.11, gradient_clip_val=1.0)
    ), 'flexgpt': {
        'wikitext103.3levels': TrainerBuilder(
            FlexLMTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
            ),
            GPTTrainingContext(wandb_project_name="FlexGPT_wikitext103"),
        ),
        'wikitext2.3levels': TrainerBuilder(
            FlexLMTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
            ),
            GPTTrainingContext(dataset_name="wikitext-2-raw-v1"),
        ),
        'wikitext103.gpt2pretrained': TrainerBuilder(
            FlexLMTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            GPTTrainingContext(wandb_project_name="FlexGPT_wikitext103_pretrained", epochs=5, patience=3),
        ),
        'wikitext2.gpt2pretrained': TrainerBuilder(
            FlexLMTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            GPTTrainingContext(dataset_name="wikitext-2-raw-v1"),
        ),
        'wikitext2.tiny': TrainerBuilder(
            FlexLMTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=256,
                num_layers=2,
                hidden_dims=(192, 256, 384),
                num_heads=(3, 4, 6),
                mlp_dims=(768, 1024, 1536),
                dropout=0.1,
            ),
            GPTTrainingContext(
                dataset_name="wikitext-2-raw-v1",
                max_seq_length=256,
                batch_size=4,
                epochs=5,
                wandb_project_name="FlexGPT",
            ),
        ),
        'wikitext103.kd_tiny': TrainerBuilder(
            FlexLMKDTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            FlexLMKDTrainingContext(
                kd_lambda=1.0,
                kd_temperature=2.0,
                dataset="wikitext-103-raw-v1",
                batch_size=8,
                epochs=1,
                patience=1,
                wandb_project_name="FlexGPT_wikitext103_kd",
            ),
        ),
        'wikitext103.kd_from_gpt2': TrainerBuilder(
            FlexLMKDTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            FlexLMKDTrainingContext(
                kd_lambda=1.0,
                kd_temperature=2.0,
                dataset="wikitext-103-raw-v1",
                batch_size=8,
                epochs=5,
                patience=3,
                wandb_project_name="FlexGPT_wikitext103_kd",
            ),
        ),
        'openwebtext.kd_from_gpt2': TrainerBuilder(
            FlexLMKDTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            FlexLMKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                batch_size=32,
                epochs=3,
                patience=3,
                wandb_project_name="FlexGPT_openwebtext_kd",
            ),
        ),
        'fineweb.kd_lambda05': TrainerBuilder(
            FlexLMKDTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            LLaMAKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                teacher_hf_model='gpt2',
                tokenizer_name='gpt2',
                epochs=10,
                patience=3,
                wandb_project_name="FlexGPT_fineweb_kd_lambda05",
            ),
        ),
        'wikitext103.kd_from_gpt2_lambda05': TrainerBuilder(
            FlexLMKDTrainer,
            FlexGPTConfig(
                vocab_size=50257,
                max_seq_length=1024,
                num_layers=12,
                hidden_dims=(384, 512, 768),
                num_heads=(6, 8, 12),
                mlp_dims=(1536, 2048, 3072),
                dropout=0.1,
                pretrained_hf_model="gpt2",
            ),
            FlexLMKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                dataset="wikitext-103-raw-v1",
                batch_size=8,
                epochs=10,
                patience=3,
                wandb_project_name="FlexGPT_wikitext103_kd_lambda05",
            ),
        ),
    }, 'flexllama': {
        'fineweb.3levels': TrainerBuilder(
            FlexLMTrainer,
            FlexLLaMAConfig(),
            LLaMATrainingContext(
                wandb_project_name="FlexLLaMA_fineweb",
            ),
        ),
        'fineweb.pretrained': TrainerBuilder(
            FlexLMTrainer,
            FlexLLaMAConfig(
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMATrainingContext(
                wandb_project_name="FlexLLaMA_fineweb_pretrained",
                epochs=5,
            ),
        ),
        'fineweb.tiny': TrainerBuilder(
            FlexLMTrainer,
            FlexLLaMAConfig(
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMATrainingContext(
                dataset="fineweb-edu",
                epochs=1,
                patience=1,
                wandb_project_name="FlexLLaMA_fineweb_pretrained_tiny",
            ),
        ),
        'fineweb.kd_pure': TrainerBuilder(
            FlexLMKDTrainer,
            FlexLLaMAConfig(
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMAKDTrainingContext(
                kd_lambda=1.0,
                kd_temperature=2.0,
                epochs=5,
                patience=3,
                wandb_project_name="FlexLLaMA_fineweb_kd_pure",
            ),
        ),
        'fineweb.kd_lambda05': TrainerBuilder(
            FlexLMKDTrainer,
            FlexLLaMAConfig(
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMAKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                epochs=10,
                patience=3,
                wandb_project_name="FlexLLaMA_fineweb_kd_lambda05",
            ),
        ),
        'fineweb.kd_lambda05_5levels': TrainerBuilder(
            FlexLMKDTrainer,
            FlexLLaMAConfig(
                hidden_dims=(256, 384, 512, 640, 768),
                num_heads=(4, 6, 8, 10, 12),
                num_kv_heads=(4, 6, 8, 10, 12),
                intermediate_dims=(1024, 1536, 2048, 2560, 3072),
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMAKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                epochs=10,
                patience=3,
                wandb_project_name="FlexLLaMA_fineweb_kd_lambda05_5levels",
            ),
        ),
    }, 'flexdeit_v3_lowFLOPS': TrainerBuilder(
        distillation.training.ScalaDistillTrainer,
        flexdeit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=(16 * 12, 24 * 12, 32 * 12, 48 * 12, 64 * 12),
            mlp_dims=(16 * 48, 24 * 48, 32 * 48, 48 * 48, 64 * 48)),
        distillation.training.ScalaDistillContext(
            # loader_function=partial(utils.load_dummy_data, batch_size=256),
            loader_function=partial(distillation.dataset.load_imagenet, batch_size=128),
            make_optimizer=lambda m: Lamb(m.parameters(),
                lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
            make_scheduler=lambda opt: CosineLRScheduler(optimizer=opt,
                t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6,
                warmup_t=5, cycle_limit=1, t_in_epochs=True),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=100,
            label_smoothing=0.11, gradient_clip_val=1.0)
    )
}
