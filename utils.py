from typing import Union, Any
import shutil
import os
import io

from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop, ConvertImageDtype, RandAugment
)
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
import tqdm
from timm.data import Mixup
from flex_modules.module import Module
from networks.modules import ClassTokenLayer, PosEmbeddingLayer, LinearHead, LayerScale
import config.paths as paths


# Some of this code is from https://github.com/poojamangal15/Adaptive-Neural-Networks


def get_device() -> 'str':
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def make_str_filename_safe(s: str):
    prefix_char = 'x'
    forbidden_chars = [
        ('x', 'xx'),
        ('/', 'xa'),
        ('<', 'xb'),
        ('>', 'xc'),
        (':', 'xd'),
        ('"', 'xe'),
        ('/', 'xf'),
        ('\\', 'xg'),
        ('|', 'xh'),
        ('?', 'xi'),
        ('*', 'xj'),
        ('(', 'xk'),
        (')', 'xl'),
        ('.', 'xm'),
        (',', 'xn'),
        ('\'', 'xo')
    ]

    description = s
    description = description.replace(
        prefix_char, f"{prefix_char}{prefix_char}")
    for forbidden, replacement in forbidden_chars:
        description = description.replace(forbidden, replacement)
    return description


class SelfDescripting:
    def setv(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_description(self) -> str:
        res = f"{self.__class__.__name__}"
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                descr = val.get_description()
                res += f"_({descr})"
            except AttributeError:
                res += f"_{val}"
        return res

    def get_filename_safe_description(self) -> str:
        return make_str_filename_safe(self.get_description())

    def get_flat_dict(self) -> str:
        res = {}
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                flatdict = val.get_flat_dict()
                for dname, dval in flatdict.items():
                    res[f"{name}.{dname}"] = dval
            except AttributeError:
                res[f"{name}"] = val
        return res


torch.serialization.add_safe_globals([SelfDescripting])


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> torch.Tensor:
    """
    Evaluates the model on the given dataloader and returns accuracy and F1 score.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    all_preds = []
    all_labels = []
    # Move model to the correct device and ensure correct data type
    model = model.to(device).to(torch.float32)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            # Ensure images are on the same device and data type
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in the model.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model: nn.Module) -> int:
    """
    Gets the models file size.

    adapted from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    f = io.BytesIO()
    torch.save(model.state_dict(), f)
    return len(f.getvalue())


def try_make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_num_nodes():
    return int(os.environ.get("SLURM_NNODES", 1))


def load_dummy_data(
    num_classes: int = 1000,
    num_train: int = 1024,
    num_val: int = 512,
    num_test: int = 512,
    image_size: tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 512,
):
    """
    Generates dummy data loaders mimicking ImageNet.
    """

    # Helper to create random tensors
    def make_dataset(num_samples):
        images = torch.randn(num_samples, *image_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        return TensorDataset(images, labels)

    train_dataset = make_dataset(num_train)
    val_dataset = make_dataset(num_val)
    test_dataset = make_dataset(num_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dummy dataloaders created, BS:{batch_size}")
    return train_loader, val_loader, test_loader

IMAGENET_TRANSFORMS = [
    Resize(256),
    CenterCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BILINEAR),
    ColorJitter(0.4, 0.4, 0.4, 0.1),
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]

# ----- Mixup + CutMix -----
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,  # probability to switch between mixup and cutmix
    mode='batch',
    label_smoothing=0.11,
    num_classes=1000
)

mixup_fn_cifar100 = Mixup(
    mixup_alpha=0.8,  # Mixup parameter
    cutmix_alpha=1.0, # CutMix parameter
    cutmix_minmax=None,
    prob=1.0,         # apply either mixup or cutmix with 100% prob
    switch_prob=0.5,  # 50% chance to switch between mixup/cutmix
    mode='batch',
    label_smoothing=0.1,
    num_classes=100
)

def load_imagenet(data_dir=paths.IMAGENET_PATH, tmp_dir=paths.TMPDIR, batch_size=512):
    train_transform = Compose(IMAGENET_TRANSFORMS)
    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    test_dataset = ImageFolder(data_dir / "val", transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)

    print(f"made dataloaders, BS:{batch_size}")
    return train_dataloader, val_dataloader, test_dataloader



def load_data(dataset, data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, resize=None, batch_size=64):
    """
    Loads data for CIFAR10 or CIFAR100

    Inspired by code from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    normalizers = {
        CIFAR10: {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225], },
        CIFAR100: {'mean': [0.5070, 0.4865, 0.4409],
                   'std': [0.2673, 0.2564, 0.2761], }
    }

    train_transform = [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    test_transform = [
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    if resize:
        test_transform.insert(0, Resize(resize))
        train_transform.insert(0, Resize(resize))

    test_transform = Compose(test_transform)
    train_transform = Compose(train_transform)

    if tmp_dir:
        try_make_dir(data_dir)
        try_make_dir(tmp_dir)
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    train_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=True, download=True, transform=train_transform)
    test_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=False, download=True, transform=test_transform)
    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def load_wikitext(
    dataset_name: str = "wikitext-103-raw-v1",
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_dir: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads (or loads from cache) a WikiText dataset, tokenises with the GPT-2
    tokeniser, packs sequences to max_seq_length with no padding, and returns
    (train_loader, val_loader, test_loader).

    Each batch is a tuple (input_ids, input_ids) of shape [B, max_seq_length].
    Labels are the same as inputs; the loss function shifts them by one position.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import Dataset as TorchDataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    raw = load_dataset("wikitext", dataset_name, cache_dir=cache_dir)

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    def pack(examples):
        # Flatten all token sequences in the batch into one list, then rechunk.
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // max_seq_length) * max_seq_length
        chunks = [ids[i: i + max_seq_length] for i in range(0, total, max_seq_length)]
        return {"input_ids": chunks}

    packed = tokenized.map(pack, batched=True, remove_columns=["attention_mask"])
    packed.set_format(type="torch", columns=["input_ids"])

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])  # [B, S]
        return ids, ids  # (input_ids, labels), loss shifts internally

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            packed[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

    return make_loader("train", True), make_loader("validation", False), make_loader("test", False)


def load_openwebtext(
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    map_workers: int = None,
    cache_dir: str = None,
    val_size: int = 2000,
    test_size: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    raw = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir)["train"]

    # OpenWebText has only a train split, carve out val and test
    split = raw.train_test_split(test_size=val_size + test_size, seed=42)
    val_test = split["test"].train_test_split(test_size=test_size, seed=42)
    splits = {
        "train":      split["train"],
        "validation": val_test["train"],
        "test":       val_test["test"],
    }

    def tokenize(examples):
        return tokenizer(examples["text"])

    def pack(examples):
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // max_seq_length) * max_seq_length
        chunks = [ids[i: i + max_seq_length] for i in range(0, total, max_seq_length)]
        return {"input_ids": chunks}

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        return ids, ids

    _map_workers = map_workers if map_workers is not None else num_workers

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        ds = splits[split_name]
        ds = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=_map_workers)
        ds = ds.map(pack, batched=True, remove_columns=["attention_mask"], num_proc=_map_workers)
        ds.set_format(type="torch", columns=["input_ids"])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

    return make_loader("train", True), make_loader("validation", False), make_loader("test", False)


def load_fineweb_edu(
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    map_workers: int = None,
    cache_dir: str = None,
    max_examples: int = 150_000,
    max_tokens: int = None,
    val_size: int = 2000,
    test_size: int = 2000,
    tokenizer_name: str = "JackFram/llama-160m",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads FineWeb-Edu (sample-10BT), tokenises with the specified tokenizer,
    packs sequences to max_seq_length, and returns (train, val, test) loaders.

    Dataset size is controlled by either max_tokens (exact - streams and counts
    real tokens until the target is reached) or max_examples (approximate - a
    fixed document count, default 150k ~= 100M tokens at ~700 tokens/doc average).
    max_tokens takes precedence when both are given. FineWeb-Edu has only a train
    split so val and test are carved out with a fixed seed.

    tokenizer_name defaults to JackFram/llama-160m (vocab_size=32000). Pass "gpt2"
    when training FlexGPT on FineWeb-Edu (vocab_size=50257).
    add_special_tokens=False suppresses BOS tokens so the packed format is consistent.
    """
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)

    # Stream to avoid downloading the full 10BT dataset, only fetch the shards
    # that contain the documents actually needed (~450MB instead of ~100GB for
    # the previous 150k-doc default).
    raw_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    if max_tokens is not None:
        docs, total_tokens = [], 0
        for example in raw_stream:
            total_tokens += len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
            docs.append(example)
            if total_tokens >= max_tokens:
                break
        raw = Dataset.from_list(docs)
    else:
        raw = Dataset.from_list(list(raw_stream.take(max_examples)))

    # FineWeb-Edu has only a train split, carve out val and test
    split   = raw.train_test_split(test_size=val_size + test_size, seed=42)
    val_test = split["test"].train_test_split(test_size=test_size, seed=42)
    splits = {
        "train":      split["train"],
        "validation": val_test["train"],
        "test":       val_test["test"],
    }

    def tokenize(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    def pack(examples):
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // max_seq_length) * max_seq_length
        chunks = [ids[i: i + max_seq_length] for i in range(0, total, max_seq_length)]
        return {"input_ids": chunks}

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        return ids, ids

    _map_workers = map_workers if map_workers is not None else num_workers

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        ds = splits[split_name]
        # remove_columns=ds.column_names drops text + all FineWeb-Edu metadata,
        # leaving only the tokenizer outputs (input_ids, attention_mask).
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=_map_workers)
        ds = ds.map(pack, batched=True, remove_columns=["attention_mask"], num_proc=_map_workers)
        ds.set_format(type="torch", columns=["input_ids"])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

    return make_loader("train", True), make_loader("validation", False), make_loader("test", False)


def flexible_model_copy(src: Union[nn.Module, dict[str, Any]], dest: nn.Module):
    if not isinstance(src, nn.Module):
        dest.load_state_dict(src)
        return

    if isinstance(src, Module):
        src.copy_to_base(dest)
        return

    if isinstance(dest, Module):
        dest.load_from_base(src)
        return

    dest.load_state_dict(src.state_dict())


def torch_serialize(obj, *args, **kwargs):
    with io.BytesIO() as f:
        torch.save(obj, f, *args, **kwargs)
        return f.getvalue()


def torch_deserialize(data: bytes, *args, **kwargs):
    with io.BytesIO(data) as f:
        return torch.load(f, *args, **kwargs)


def save_model(exp_name, model):
    exp_name = make_str_filename_safe(exp_name)
    with open(paths.TRAINED_MODELS / f"{exp_name}.pt", "wb") as f:
        torch.save(model.state_dict(), f)

def save_statedict(name, model):
    with open(paths.TRAINED_MODELS / f"{name}.pt", "wb") as f:
        torch.save(model.state_dict(), f)
    print("model state dict saved")


def load_model(exp_name, model_config):
    exp_name = make_str_filename_safe(exp_name)
    model = model_config.make_model()
    with open(paths.TRAINED_MODELS / f"{exp_name}.pt", "rb") as f:
        sdict = torch.load(f)
        model.load_state_dict(sdict)
    return model

@torch.no_grad()
def load_gpt2_weights_into_flexgpt(
        model,
        hf_model_name: str = "gpt2",
        cache_dir: str = None,
) -> None:
    """
    Loads HuggingFace GPT-2 pretrained weights into a max-level FlexGPT model.
    """
    from transformers import GPT2LMHeadModel

    cfg = model.config
    max_hidden = list(cfg.hidden_dims)[-1]
    max_heads  = list(cfg.num_heads)[-1]
    max_mlp    = list(cfg.mlp_dims)[-1]

    hf = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir=cache_dir)
    sd = hf.state_dict()
    del hf

    model.set_level_use(model.max_level())

    # Token embedding
    model.token_embedding.embedding.weight.data.copy_(sd["transformer.wte.weight"])

    # Positional embedding: GPT-2 [seq_len, hidden] → FlexGPT [1, seq_len, max_hidden]
    model.pos_embedding.embedding.data.copy_(sd["transformer.wpe.weight"].unsqueeze(0))

    for i in range(cfg.num_layers):
        block = model.blocks[i]
        pfx = f"transformer.h.{i}"

        # Pre-attention layer norm
        tmp_ln = nn.LayerNorm(max_hidden, eps=1e-5)
        tmp_ln.weight.data.copy_(sd[f"{pfx}.ln_1.weight"])
        tmp_ln.bias.data.copy_(sd[f"{pfx}.ln_1.bias"])
        block.ln_1.load_from_base(tmp_ln)

        # Self-attention, Conv1D weights need .T to become [out, in] (nn.Linear layout)
        tmp_mha = nn.MultiheadAttention(max_hidden, max_heads, batch_first=True)
        tmp_mha.in_proj_weight.data.copy_(sd[f"{pfx}.attn.c_attn.weight"].T)
        tmp_mha.in_proj_bias.data.copy_(sd[f"{pfx}.attn.c_attn.bias"])
        tmp_mha.out_proj.weight.data.copy_(sd[f"{pfx}.attn.c_proj.weight"].T)
        tmp_mha.out_proj.bias.data.copy_(sd[f"{pfx}.attn.c_proj.bias"])
        block.attn.load_from_base(tmp_mha)

        # Post-attention layer norm
        tmp_ln2 = nn.LayerNorm(max_hidden, eps=1e-5)
        tmp_ln2.weight.data.copy_(sd[f"{pfx}.ln_2.weight"])
        tmp_ln2.bias.data.copy_(sd[f"{pfx}.ln_2.bias"])
        block.ln_2.load_from_base(tmp_ln2)

        # MLP up-projection (c_fc): Conv1D [hidden, mlp] → .T → [mlp, hidden]
        tmp_fc = nn.Linear(max_hidden, max_mlp)
        tmp_fc.weight.data.copy_(sd[f"{pfx}.mlp.c_fc.weight"].T)
        tmp_fc.bias.data.copy_(sd[f"{pfx}.mlp.c_fc.bias"])
        block.mlp[0].load_from_base(tmp_fc)

        # MLP down-projection (c_proj): Conv1D [mlp, hidden] → .T → [hidden, mlp]
        tmp_proj = nn.Linear(max_mlp, max_hidden)
        tmp_proj.weight.data.copy_(sd[f"{pfx}.mlp.c_proj.weight"].T)
        tmp_proj.bias.data.copy_(sd[f"{pfx}.mlp.c_proj.bias"])
        block.mlp[3].load_from_base(tmp_proj)

    # Final layer norm
    tmp_ln_f = nn.LayerNorm(max_hidden, eps=1e-5)
    tmp_ln_f.weight.data.copy_(sd["transformer.ln_f.weight"])
    tmp_ln_f.bias.data.copy_(sd["transformer.ln_f.bias"])
    model.ln_f.load_from_base(tmp_ln_f)


def load_llama_weights_into_flexllama(
        model,
        hf_model_name: str,
        cache_dir: str = None,
) -> None:
    """
    Loads pretrained weights from a HF LLaMA-family or Qwen2-family checkpoint into
    a max-level FlexLLaMA model. Qwen2 mirrors LLaMA's module naming closely enough
    (self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input/post_attention_layernorm,
    model.norm, model.embed_tokens, lm_head) that the same key layout works for both;
    the only structural difference handled here is Qwen2's biased q/k/v projections
    (cfg.qkv_bias).
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from flex_modules.rmsnorm import _RMSNormBase

    hf_config = AutoConfig.from_pretrained(hf_model_name, cache_dir=cache_dir)
    if hf_config.model_type not in ("llama", "qwen2"):
        raise ValueError(
            f"load_llama_weights_into_flexllama only supports llama/qwen2 "
            f"checkpoints (identical module naming); got model_type={hf_config.model_type!r} "
            f"for {hf_model_name}"
        )

    cfg = model.config
    max_hidden       = list(cfg.hidden_dims)[-1]
    max_intermediate = list(cfg.intermediate_dims)[-1]
    max_num_heads    = list(cfg.num_heads)[-1]
    max_num_kv_heads = list(cfg.num_kv_heads)[-1]
    head_dim         = max_hidden // max_num_heads
    max_kv_dim       = max_num_kv_heads * head_dim

    hf = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=cache_dir)
    sd = hf.state_dict()
    del hf

    # Validate dimensions upfront so failures are obvious rather than silent shape errors.
    hf_hidden = sd["model.embed_tokens.weight"].shape[1]
    hf_vocab  = sd["model.embed_tokens.weight"].shape[0]
    hf_layers = sum(
        1 for k in sd if k.startswith("model.layers.") and k.endswith(".input_layernorm.weight")
    )
    hf_intermediate = sd["model.layers.0.mlp.gate_proj.weight"].shape[0]
    hf_kv_dim       = sd["model.layers.0.self_attn.k_proj.weight"].shape[0]

    if hf_hidden != max_hidden:
        raise ValueError(f"Hidden dim mismatch: checkpoint={hf_hidden}, model max level={max_hidden}")
    if hf_vocab != cfg.vocab_size:
        raise ValueError(f"Vocab size mismatch: checkpoint={hf_vocab}, model={cfg.vocab_size}")
    if hf_layers != cfg.num_layers:
        raise ValueError(f"Layer count mismatch: checkpoint={hf_layers}, model={cfg.num_layers}")
    if hf_intermediate != max_intermediate:
        raise ValueError(f"Intermediate dim mismatch: checkpoint={hf_intermediate}, model max level={max_intermediate}")
    if hf_kv_dim != max_kv_dim:
        raise ValueError(
            f"KV projection dim mismatch: checkpoint={hf_kv_dim} (implies "
            f"num_kv_heads={hf_kv_dim // head_dim}), model max level expects "
            f"num_kv_heads={max_num_kv_heads} (kv_dim={max_kv_dim}). Set "
            f"num_kv_heads in FlexLLaMAConfig to match the checkpoint."
        )

    model.set_level_use(model.max_level())

    # Token embedding
    tmp_emb = nn.Embedding(cfg.vocab_size, max_hidden)
    tmp_emb.weight.data.copy_(sd["model.embed_tokens.weight"])
    model.embed_tokens.load_from_base(tmp_emb)

    for i in range(cfg.num_layers):
        block = model.layers[i]
        pfx   = f"model.layers.{i}"

        # Input RMSNorm
        tmp_rms = _RMSNormBase(max_hidden, eps=cfg.rms_norm_eps)
        tmp_rms.weight.data.copy_(sd[f"{pfx}.input_layernorm.weight"])
        block.input_layernorm.load_from_base(tmp_rms)

        # Attention projections, [out, in] layout, no transpose needed.
        # q_proj/o_proj are always hidden->hidden; k_proj/v_proj output kv_dim,
        # which is smaller than hidden when the checkpoint uses GQA. q/k/v carry
        # bias on Qwen2-family checkpoints (cfg.qkv_bias); o_proj never does.
        for proj, out_dim, has_bias in (
            ("q_proj", max_hidden, cfg.qkv_bias), ("k_proj", max_kv_dim, cfg.qkv_bias),
            ("v_proj", max_kv_dim, cfg.qkv_bias), ("o_proj", max_hidden, False),
        ):
            tmp_linear = nn.Linear(max_hidden, out_dim, bias=has_bias)
            tmp_linear.weight.data.copy_(sd[f"{pfx}.self_attn.{proj}.weight"])
            if has_bias:
                tmp_linear.bias.data.copy_(sd[f"{pfx}.self_attn.{proj}.bias"])
            getattr(block.self_attn, proj).load_from_base(tmp_linear)

        # Post-attention RMSNorm
        tmp_rms2 = _RMSNormBase(max_hidden, eps=cfg.rms_norm_eps)
        tmp_rms2.weight.data.copy_(sd[f"{pfx}.post_attention_layernorm.weight"])
        block.post_attention_layernorm.load_from_base(tmp_rms2)

        # MLP
        tmp_gate = nn.Linear(max_hidden, max_intermediate, bias=False)
        tmp_gate.weight.data.copy_(sd[f"{pfx}.mlp.gate_proj.weight"])
        block.mlp.gate_proj.load_from_base(tmp_gate)

        tmp_up = nn.Linear(max_hidden, max_intermediate, bias=False)
        tmp_up.weight.data.copy_(sd[f"{pfx}.mlp.up_proj.weight"])
        block.mlp.up_proj.load_from_base(tmp_up)

        tmp_down = nn.Linear(max_intermediate, max_hidden, bias=False)
        tmp_down.weight.data.copy_(sd[f"{pfx}.mlp.down_proj.weight"])
        block.mlp.down_proj.load_from_base(tmp_down)

    # Final RMSNorm
    tmp_rms_f = _RMSNormBase(max_hidden, eps=cfg.rms_norm_eps)
    tmp_rms_f.weight.data.copy_(sd["model.norm.weight"])
    model.norm.load_from_base(tmp_rms_f)

    # LM head (untied)
    if not cfg.tie_embeddings:
        tmp_lm_head = nn.Linear(max_hidden, cfg.vocab_size, bias=False)
        tmp_lm_head.weight.data.copy_(sd["lm_head.weight"])
        model.lm_head.load_from_base(tmp_lm_head)
