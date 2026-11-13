"""ZeRO-1 (training.trainer.adamw(zero=True)): two CPU processes (gloo) take the same AdamW step
as a single unsharded AdamW, including the continuum param group; no-op without a process group."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
    m.register_parameter("level_delta", torch.nn.Parameter(torch.zeros(3, 4)))  # continuum group
    return m


def _grads(m):
    g = torch.Generator().manual_seed(1)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def _worker(rank, world, init_file, out_dir):
    from training.trainer import adamw
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    m = _model()
    opt = adamw(m.parameters(), lr=1e-2, weight_decay=0.1, model=m, continuum_lr=1e-1, zero=True)
    assert type(opt).__name__ == "ZeroRedundancyOptimizer"
    for _ in range(3):
        _grads(m)
        opt.step()
    opt.consolidate_state_dict(to=0)
    if rank == 0:
        torch.save({"params": [p.detach().clone() for p in m.parameters()], "state": opt.state_dict()},
                   os.path.join(out_dir, "zero.pt"))
    dist.barrier()
    dist.destroy_process_group()


def test_zero_matches_adamw_and_is_off_without_dist():
    from training.trainer import adamw
    m = _model()
    ref = adamw(m.parameters(), lr=1e-2, weight_decay=0.1, model=m, continuum_lr=1e-1, zero=True)
    assert isinstance(ref, torch.optim.AdamW)  # no process group: plain AdamW
    for _ in range(3):
        _grads(m)
        ref.step()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init").replace("\\", "/")
        mp.spawn(_worker, args=(2, init_file, d), nprocs=2, join=True)
        got = torch.load(os.path.join(d, "zero.pt"), weights_only=False)
    for a, b in zip(m.parameters(), got["params"]):
        torch.testing.assert_close(b, a)
    assert len(got["state"]["state"]) == len(list(m.parameters()))  # full state on rank 0
