"""Copy finished MedNest W&B runs into the single project "MedNest" (user, 2026-11-17).

For every run in the per-config projects MedNest_* (entity sudaksh14), a run with the same id and
name is created in project MedNest with the same config, the full metric history (same steps) and
the final summary; its group and tag are the old project name, as for new runs (config/wandb.py).
The old projects are NOT modified or deleted (delete them in the W&B UI once the copies are checked).
Skipped: runs whose state is "running", runs of the jobs in --skip-jobs, runs already in MedNest
(so the script can be re-run). Run creation times become the copy time (W&B does not allow setting them);
the original start time is kept in the config as `copied_from_created_at`.

(`wandb sync -p MedNest <run dir>` does not work for runs that were logged online: it prints
"done" and uploads nothing; job 28305.)

Run on a compute node: sbatch --nodelist=node208 scripts/das6/wandb_to_mednest.sbatch [--dry-run] [--only ID]
"""
import argparse
import math
import numbers
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.wandb import MEDNEST_PROJECT  # noqa: E402


def _plain(v):
    """Summary/history values that can be re-logged (numbers, strings, bools)."""
    if isinstance(v, bool) or isinstance(v, str):
        return True
    return isinstance(v, numbers.Number) and not (isinstance(v, float) and math.isnan(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="sudaksh14")
    ap.add_argument("--skip-runs", default="", help="comma-separated run ids never to touch")
    ap.add_argument("--only", default="", help="copy only these run ids (comma-separated), for a test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb

    api = wandb.Api(timeout=120)
    skip = {r for r in args.skip_runs.split(",") if r}
    only = {r for r in args.only.split(",") if r}
    existing = set()
    if any(p.name == MEDNEST_PROJECT for p in api.projects(args.entity)):
        existing = {r.id for r in api.runs(f"{args.entity}/{MEDNEST_PROJECT}")}
    old_projects = sorted(p.name for p in api.projects(args.entity)
                          if p.name.startswith("MedNest") and p.name != MEDNEST_PROJECT)
    copied, skipped, failed = 0, 0, []
    for proj in old_projects:
        for run in api.runs(f"{args.entity}/{proj}"):
            tag = f"{proj}/{run.id} ({run.name}, {run.state})"
            if only and run.id not in only:
                continue
            if run.id in skip or run.state == "running" or run.id in existing:
                why = "running" if run.state == "running" else ("skip list" if run.id in skip else "already copied")
                print(f"[mednest] SKIP {tag}: {why}")
                skipped += 1
                continue
            print(f"[mednest] copy {tag} -> {MEDNEST_PROJECT}", flush=True)
            if args.dry_run:
                continue
            try:
                config = dict(run.config)
                config["copied_from"] = f"{proj}/{run.id}"
                config["copied_from_created_at"] = run.created_at
                new = wandb.init(entity=args.entity, project=MEDNEST_PROJECT, id=run.id, name=run.name,
                                 group=proj, tags=sorted(set(run.tags) | {proj}), notes=run.notes,
                                 config=config, resume="never", reinit=True)
                rows = 0
                for row in run.scan_history(page_size=2000):
                    step = row.get("_step")
                    data = {k: v for k, v in row.items() if not k.startswith("_") and _plain(v)}
                    if data:
                        new.log(data, step=int(step) if step is not None else None)
                        rows += 1
                for k, v in run.summary.items():
                    if not k.startswith("_") and _plain(v):
                        new.summary[k] = v
                new.summary["copied_history_rows"] = rows
                new.finish(exit_code=0 if run.state == "finished" else 1)
                print(f"[mednest]   {rows} history rows", flush=True)
                copied += 1
            except Exception as e:
                print(f"[mednest]   FAILED: {e!r}", flush=True)
                failed.append(f"{proj}/{run.id}")
    print(f"[mednest] copied {copied}, skipped {skipped}, failed {len(failed)}: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
