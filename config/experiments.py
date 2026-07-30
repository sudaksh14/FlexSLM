from networks.flexgpt import FlexGPT, FlexGPTConfig
from networks.flexllama import FlexLLaMA, FlexLLaMAConfig
from training import *
from training import FlexLMTrainer

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from functools import partial
import torch
import torch.optim as optim

from training import FlexLMTrainer, FlexLMKDTrainer


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
                 max_examples=150_000, map_workers=None, *args, **kwargs):
        if dataset == "fineweb-edu":
            loader = partial(utils.load_fineweb_edu,
                             max_seq_length=max_seq_length,
                             batch_size=batch_size,
                             max_examples=max_examples,
                             tokenizer_name=tokenizer_name,
                             map_workers=map_workers)
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


def _flex_llama_dims(hidden, heads, kv_heads, intermediate, n_levels=3, min_frac=None, fractions=None):
    """
    Builds increasing (hidden_dims, num_heads, num_kv_heads, intermediate_dims)
    tuples for a multi-level FlexLLaMAConfig whose max level exactly matches a
    real checkpoint's dims (required for pretrained_hf_model warm-starting).
    Smaller levels keep the same head_dim and pick the largest kv_heads that
    still evenly divides that level's head count - they need not preserve the
    teacher's exact GQA ratio, only the max level does.

    n_levels total levels are generated: min_frac (default 1/n_levels) at the
    bottom, 1.0 (the checkpoint's real size) at the top, and n_levels - 2 more
    evenly spaced in between. Pass fractions explicitly to override this spacing.
    """
    if fractions is None:
        if min_frac is None:
            min_frac = 1.0 / n_levels
        fractions = [1.0] if n_levels == 1 else [
            min_frac + (1.0 - min_frac) * i / (n_levels - 1) for i in range(n_levels)
        ]

    head_dim = hidden // heads

    def kv_for(h):
        for kv in range(min(h, kv_heads), 0, -1):
            if h % kv == 0:
                return kv
        return 1

    hs, hd, kvs, im = [], [], [], []
    for i, frac in enumerate(fractions):
        is_last = (i == len(fractions) - 1)
        h = heads if is_last else min(heads, max(1, round(heads * frac)))
        if hs and h <= hs[-1]:
            h = hs[-1] + 1
        hs.append(h)
        hd.append(h * head_dim)
        kvs.append(kv_for(h))

        i_val = intermediate if is_last else min(intermediate, max(64, round(intermediate * frac / 64) * 64))
        if im and i_val <= im[-1]:
            i_val = im[-1] + 64
        im.append(i_val)

    return tuple(hd), tuple(hs), tuple(kvs), tuple(im)


# Teacher presets for FlexLLaMA. Each entry's hf_model is used both as the frozen
# KD teacher and as the tokenizer that pre-tokenizes the training corpus, so
# student/teacher logits line up index-for-index over the vocab.
#
# Entries with full architecture fields (num_layers/hidden_dims/... below) support
# pretrained_hf_model warm-starting via make_flexllama_warmstart_kd - the student's
# max level is sized to match the checkpoint exactly, per utils.load_llama_weights_
# into_flexllama's validation. llama3.2-1b is gated on HF (needs license + HF_TOKEN)
# so its dims are unverified here; it's KD-only via make_flexllama_kd until confirmed
# with llama_model_param_fetch.py.
TEACHER_PRESETS = {
    "llama-160m": dict(
        hf_model="JackFram/llama-160m", vocab_size=32000, num_layers=12,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_embeddings=False, qkv_bias=False,
        dims_spec=dict(hidden=768, heads=12, kv_heads=12, intermediate=3072),
    ),
    "tinyllama-1.1b": dict(
        hf_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", vocab_size=32000, num_layers=22,
        rope_theta=10000.0, rms_norm_eps=1e-5, tie_embeddings=False, qkv_bias=False,
        dims_spec=dict(hidden=2048, heads=32, kv_heads=4, intermediate=5632),
    ),
    "llama3.2-1b": dict(hf_model="meta-llama/Llama-3.2-1B", vocab_size=128256),  # gated on HF: dims unverified
    "smollm2-360m": dict(
        hf_model="HuggingFaceTB/SmolLM2-360M", vocab_size=49152, num_layers=32,
        rope_theta=100000.0, rms_norm_eps=1e-5, tie_embeddings=True, qkv_bias=False,
        dims_spec=dict(hidden=960, heads=15, kv_heads=5, intermediate=2560),
    ),
    "smollm2-1.7b": dict(
        hf_model="HuggingFaceTB/SmolLM2-1.7B", vocab_size=49152, num_layers=24,
        rope_theta=130000.0, rms_norm_eps=1e-5, tie_embeddings=True, qkv_bias=False,
        dims_spec=dict(hidden=2048, heads=32, kv_heads=32, intermediate=8192),
    ),
    "qwen2.5-0.5b": dict(
        hf_model="Qwen/Qwen2.5-0.5B", vocab_size=151936, num_layers=24,
        rope_theta=1000000.0, rms_norm_eps=1e-6, tie_embeddings=True, qkv_bias=True,
        dims_spec=dict(hidden=896, heads=14, kv_heads=2, intermediate=4864),
    ),
    "qwen2.5-1.5b": dict(
        hf_model="Qwen/Qwen2.5-1.5B", vocab_size=151936, num_layers=28,
        rope_theta=1000000.0, rms_norm_eps=1e-6, tie_embeddings=True, qkv_bias=True,
        dims_spec=dict(hidden=1536, heads=12, kv_heads=2, intermediate=8960),
    ),
}

_WARMSTART_FIELDS = ("num_layers", "dims_spec", "rope_theta", "rms_norm_eps",
                     "tie_embeddings", "qkv_bias")


def _label_levels(kd_kwargs: dict, n_levels: int) -> dict:
    """
    Mutates kd_kwargs so every config self-documents its total levels and how
    many are sampled per training step: defaults num_levels_per_step to n_levels
    (all levels trained every step, the same as looping over all of them) unless
    the caller overrides it, and appends "_{n_levels}L{sampled}S" to
    wandb_project_name so both numbers are visible without opening the config.
    """
    kd_kwargs.setdefault("num_levels_per_step", n_levels)
    sampled = kd_kwargs["num_levels_per_step"]
    if kd_kwargs.get("wandb_project_name") is not None:
        kd_kwargs["wandb_project_name"] += f"_{n_levels}L{sampled}S"
    return kd_kwargs


def make_flexllama_kd(teacher: str, **kd_kwargs) -> TrainerBuilder:
    """Builds a FlexLLaMA KD TrainerBuilder for the named entry in TEACHER_PRESETS.
    Student starts from random init and is trained purely via KD logits."""
    preset = TEACHER_PRESETS[teacher]
    n_levels = len(FlexLLaMAConfig().hidden_dims)
    kd_kwargs = _label_levels(kd_kwargs, n_levels)
    return TrainerBuilder(
        FlexLMKDTrainer,
        FlexLLaMAConfig(vocab_size=preset["vocab_size"]),
        LLaMAKDTrainingContext(
            teacher_hf_model=preset["hf_model"],
            tokenizer_name=preset["hf_model"],
            **kd_kwargs,
        ),
    )


def make_flexllama_warmstart_kd(teacher: str, n_levels: int = 3, min_frac: float = None,
                                 **kd_kwargs) -> TrainerBuilder:
    """Same KD training algorithm as make_flexllama_kd, but the student's max
    level is built to match the teacher's real architecture exactly and is
    warm-started from its pretrained weights (FlexLLaMAConfig.pretrained_hf_model),
    rather than starting from random init.

    n_levels controls how many flex levels the student has in total (min_frac at
    the bottom, the checkpoint's real size at the top, evenly spaced in between -
    see _flex_llama_dims). Training all n_levels every step gets expensive as
    n_levels grows; pass num_levels_per_step in kd_kwargs to sample a subset per
    step instead of training every level each time.
    """
    preset = TEACHER_PRESETS[teacher]
    missing = [f for f in _WARMSTART_FIELDS if f not in preset]
    if missing:
        raise ValueError(
            f"TEACHER_PRESETS[{teacher!r}] is missing {missing} - warm-starting "
            f"needs the full architecture spec (see llama_model_param_fetch.py). "
            f"Use make_flexllama_kd for KD-only training against this teacher."
        )
    hidden_dims, num_heads, num_kv_heads, intermediate_dims = _flex_llama_dims(
        **preset["dims_spec"], n_levels=n_levels, min_frac=min_frac)
    kd_kwargs = _label_levels(kd_kwargs, n_levels)
    return TrainerBuilder(
        FlexLMKDTrainer,
        FlexLLaMAConfig(
            vocab_size=preset["vocab_size"],
            num_layers=preset["num_layers"],
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_dims=intermediate_dims,
            rope_theta=preset["rope_theta"],
            rms_norm_eps=preset["rms_norm_eps"],
            tie_embeddings=preset["tie_embeddings"],
            qkv_bias=preset["qkv_bias"],
            pretrained_hf_model=preset["hf_model"],
        ),
        LLaMAKDTrainingContext(
            teacher_hf_model=preset["hf_model"],
            tokenizer_name=preset["hf_model"],
            **kd_kwargs,
        ),
    )


CONFIGS = {
    'flexgpt': {
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
        'fineweb.kd_lambda05_1p4B': TrainerBuilder(
            FlexLMKDTrainer,
            FlexLLaMAConfig(
                pretrained_hf_model="JackFram/llama-160m",
            ),
            LLaMAKDTrainingContext(
                kd_lambda=0.5,
                kd_temperature=2.0,
                max_examples=2_000_000,  # ~1.4B tokens at ~700 tok/doc, up from the 100M-token default
                map_workers=1,  # 4 DDP ranks each tokenize independently; num_proc=4 per rank OOM'd the node
                epochs=3,
                patience=3,
                wandb_project_name="FlexLLaMA_fineweb_kd_lambda05_1p4B",
            ),
        ),
        'fineweb.kd_tinyllama': make_flexllama_kd(
            "tinyllama-1.1b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_tinyllama",
        ),
        'fineweb.kd_llama32_1b': make_flexllama_kd(
            "llama3.2-1b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_llama32_1b",
        ),
        'fineweb.kd_smollm2_360m': make_flexllama_kd(
            "smollm2-360m",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_smollm2_360m",
        ),
        'fineweb.kd_smollm2_1p7b': make_flexllama_kd(
            "smollm2-1.7b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_smollm2_1p7b",
        ),
        'fineweb.kd_qwen25_0p5b': make_flexllama_kd(
            "qwen2.5-0.5b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_qwen25_0p5b",
        ),
        'fineweb.kd_qwen25_1p5b': make_flexllama_kd(
            "qwen2.5-1.5b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_qwen25_1p5b",
        ),
        'fineweb.kd_tinyllama_warmstart': make_flexllama_warmstart_kd(
            "tinyllama-1.1b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_tinyllama_warmstart",
        ),
        'fineweb.kd_tinyllama_warmstart_5levels': make_flexllama_warmstart_kd(
            "tinyllama-1.1b", n_levels=5,
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            num_levels_per_step=3,  # min+max always included, 1 more sampled from the middle
            wandb_project_name="FlexLLaMA_fineweb_kd_tinyllama_warmstart_5levels",
        ),
        'fineweb.kd_tinyllama_warmstart_7levels': make_flexllama_warmstart_kd(
            "tinyllama-1.1b", n_levels=7,
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            num_levels_per_step=3,  # min+max always included, 1 more sampled from the middle
            wandb_project_name="FlexLLaMA_fineweb_kd_tinyllama_warmstart_7levels",
        ),
        'fineweb.kd_tinyllama_warmstart_10levels': make_flexllama_warmstart_kd(
            "tinyllama-1.1b", n_levels=10,
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            num_levels_per_step=3,  # 10 levels/step would be 10x the fwd+bwd cost; sample a subset instead
            wandb_project_name="FlexLLaMA_fineweb_kd_tinyllama_warmstart_10levels",
        ),
        'fineweb.kd_smollm2_360m_warmstart': make_flexllama_warmstart_kd(
            "smollm2-360m",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_smollm2_360m_warmstart",
        ),
        'fineweb.kd_smollm2_1p7b_warmstart': make_flexllama_warmstart_kd(
            "smollm2-1.7b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_smollm2_1p7b_warmstart",
        ),
        'fineweb.kd_qwen25_0p5b_warmstart': make_flexllama_warmstart_kd(
            "qwen2.5-0.5b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_qwen25_0p5b_warmstart",
        ),
        'fineweb.kd_qwen25_1p5b_warmstart': make_flexllama_warmstart_kd(
            "qwen2.5-1.5b",
            kd_lambda=0.5, kd_temperature=2.0, epochs=10, patience=3,
            wandb_project_name="FlexLLaMA_fineweb_kd_qwen25_1p5b_warmstart",
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
    }
}
