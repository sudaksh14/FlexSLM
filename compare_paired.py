"""
Paired comparison of two models on the same questions (exact McNemar test):
b = questions only model A gets right, c = only model B gets right; under "no
difference" b ~ Binomial(b + c, 1/2). More sensitive than comparing two
accuracies with independent standard errors, since most questions are answered
the same way by both models.

    python3 compare_paired.py gen   A.samples.jsonl B.samples.jsonl      # eval_gen.py output
    python3 compare_paired.py cloze A.samples.json  B.samples.json       # eval_lm_harness --samples, per level
"""
import json
import math
import sys
from collections import defaultdict


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(title: str, pairs: dict) -> None:
    """pairs: task -> list of (a_correct, b_correct)."""
    print(f"\n== {title}")
    print(f"{'task':28s} {'n':>5s} {'acc A':>6s} {'acc B':>6s} {'B-A':>6s} {'only A':>6s} {'only B':>6s} {'p':>8s}")
    total = []
    for task in sorted(pairs):
        rows = pairs[task]
        total += rows
        _line(task, rows)
    if len(pairs) > 1:
        _line("ALL (pooled)", total)


def _line(name, rows):
    n = len(rows)
    a = sum(x for x, _ in rows) / n
    b_acc = sum(y for _, y in rows) / n
    only_a = sum(1 for x, y in rows if x and not y)
    only_b = sum(1 for x, y in rows if y and not x)
    print(f"{name[:28]:28s} {n:5d} {100 * a:6.1f} {100 * b_acc:6.1f} {100 * (b_acc - a):+6.1f} "
          f"{only_a:6d} {only_b:6d} {mcnemar_p(only_a, only_b):8.4f}")


def gen_pairs(path_a: str, path_b: str) -> dict:
    rows_a = [json.loads(line) for line in open(path_a)]
    rows_b = [json.loads(line) for line in open(path_b)]
    assert len(rows_a) == len(rows_b), "different number of questions"
    pairs = defaultdict(list)
    for ra, rb in zip(rows_a, rows_b):
        assert (ra["source"], ra["gt"]) == (rb["source"], rb["gt"]), "questions are not in the same order"
        pairs[ra["source"]].append((ra["correct"], rb["correct"]))
    return pairs


def cloze_pairs(path_a: str, path_b: str) -> dict:
    """level -> task -> pairs, matched on doc_id."""
    a, b = json.load(open(path_a)), json.load(open(path_b))
    out = {}
    for level in sorted(set(a) & set(b)):
        pairs = defaultdict(list)
        for task in sorted(set(a[level]) & set(b[level])):
            by_id = {r["doc_id"]: r for r in b[level][task]["rows"]}
            for ra in a[level][task]["rows"]:
                rb = by_id.get(ra["doc_id"])
                if rb is not None:
                    assert ra["target"] == rb["target"]
                    pairs[task].append((ra["pred"] == ra["target"], rb["pred"] == rb["target"]))
        out[level] = pairs
    return out


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("gen", "cloze"):
        sys.exit(__doc__)
    kind, path_a, path_b = sys.argv[1:]
    print(f"A = {path_a}\nB = {path_b}")
    if kind == "gen":
        report("generated answers", gen_pairs(path_a, path_b))
    else:
        for level, pairs in cloze_pairs(path_a, path_b).items():
            report(f"cloze, {level}", pairs)


if __name__ == "__main__":
    main()
