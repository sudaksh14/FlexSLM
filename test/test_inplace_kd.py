import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
import pytorch_lightning as pl

import training
import training_memopt


class _TinyLM(nn.Module):
    """Stand-in for a Flex LM: 3 levels, logits = level-scaled linear head."""
    def __init__(self, vocab=11):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)
        self.level = 0
        self.seen = []
        self.config = types.SimpleNamespace(vocab_size=vocab)

    def max_level(self):
        return 2

    def set_level_use(self, i):
        self.level = i
        self.seen.append(i)

    def forward(self, x):
        return self.head(self.emb(x)) * (1 + self.level)


def _make_trainer(cls=training.FlexLMTrainer, **ctx):
    t = cls.__new__(cls)
    pl.LightningModule.__init__(t)
    t.submodel = _TinyLM()
    t.training_context = types.SimpleNamespace(**ctx)
    t.log = mock.Mock()
    t.manual_backward = mock.Mock(side_effect=lambda loss, gradient=None: loss.backward(gradient=gradient))
    opt = mock.Mock()
    t.optimizers = mock.Mock(return_value=opt)
    return t, opt


class TestInplaceKD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        x = torch.randint(0, 11, (2, 6))
        self.batch = (x, x)

    def test_select_levels_keeps_min_and_max(self):
        t, _ = _make_trainer(num_levels_per_step=2)
        for _ in range(20):
            self.assertEqual(t._select_levels("train"), [0, 2])
        self.assertEqual(t._select_levels("val"), [0, 1, 2])

    def test_none_matches_ce_only(self):
        t, opt = _make_trainer(num_levels_per_step=None, inplace_kd="none")
        t._step(self.batch, "train")
        self.assertEqual(t.submodel.seen, [0, 1, 2])
        self.assertEqual(t.manual_backward.call_count, 3)
        opt.step.assert_called_once()

    def test_kd_trains_widest_first_and_adds_kd_to_narrow(self):
        for mode in ("progressive", "US"):
            t, opt = _make_trainer(num_levels_per_step=None, inplace_kd=mode)
            t._step(self.batch, "train")
            self.assertEqual(t.submodel.seen, [2, 1, 0])
            logged = [c.args[0] for c in t.log.call_args_list]
            self.assertNotIn("train_level2_kd_loss", logged)
            self.assertIn("train_level1_kd_loss", logged)
            self.assertIn("train_level0_kd_loss", logged)
            opt.step.assert_called_once()

    def test_validation_is_ce_only(self):
        t, _ = _make_trainer(num_levels_per_step=None, inplace_kd="progressive")
        t._step(self.batch, "val")
        logged = [c.args[0] for c in t.log.call_args_list]
        self.assertFalse(any("kd_loss" in n for n in logged))
        t.manual_backward.assert_not_called()

    def test_memopt_matches_plain_trainer(self):
        """Chunked memopt loss must give the same grads as FlexLMTrainer, KD included."""
        for mode in ("none", "progressive", "US"):
            plain, _ = _make_trainer(training.FlexLMTrainer, num_levels_per_step=None,
                                     inplace_kd=mode, inplace_kd_temperature=2.0)
            mem, _ = _make_trainer(training_memopt.MemOptFlexLMTrainer, num_levels_per_step=None,
                                   inplace_kd=mode, inplace_kd_temperature=2.0,
                                   loss_chunk_size=3)
            mem.submodel.load_state_dict(plain.submodel.state_dict())
            plain._step(self.batch, "train")
            mem._step(self.batch, "train")
            for (n, p), (_, q) in zip(plain.submodel.named_parameters(),
                                      mem.submodel.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, msg=f"{mode}: {n}")

    def test_external_teacher_trainers_reject_inplace_kd(self):
        t, _ = _make_trainer(training.FlexLMKDTrainer, inplace_kd="US",
                             kd_lambda=0.5, kd_temperature=1.0)
        with self.assertRaises(NotImplementedError):
            t._step(self.batch, "train")

    def test_invalid_mode_raises(self):
        t, _ = _make_trainer(num_levels_per_step=None, inplace_kd="bogus")
        with self.assertRaises(ValueError):
            t._step(self.batch, "train")


if __name__ == "__main__":
    unittest.main()
