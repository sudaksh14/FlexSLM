"""Move the per-config W&B projects MedNest_* into the single project "MedNest" (user, 2026-11-17/18).

Steps (each one a flag; default = copy only):
  copy         for every run in a per-config project (entity sudaksh14) create a run with the same id and
               name in MedNest: same config, full metric history at the same steps, final summary; group and
               tag = the old project name (as config/wandb.py does for new runs). Skipped: runs that are
               running, runs already in MedNest. (`wandb sync -p` does not work for online-logged runs.)
  --verify     compare every old run with its copy: it exists, same state, same number of history rows,
               same config (copied_* keys aside), same final summary values.
  --backup DIR write every old run (config, summary, full history) to DIR/<project>/<run id>.json.
  --delete-old delete the old projects. Only a project whose runs ALL verified is deleted (runs first, then
               the project through the GraphQL deleteModel mutation: wandb 0.19 has no project-delete call);
               only names starting with "MedNest_" are ever touched; MedNest itself and every unrelated
               project of the entity (FlexLLaVA, FlexViT, ...) are never deleted. Needs --verify and --backup.
  --only-project NAME   restrict every step to one old project (to test the deletion on one first).

Run on a compute node: sbatch --nodelist=<node> scripts/das6/wandb_to_mednest.sbatch [flags]
"""
import argparse
import json
import math
import numbers
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.wandb import MEDNEST_PROJECT  # noqa: E402

OLD_PREFIX = "MedNest_"


def _plain(v):
    """Summary/history values that can be re-logged (numbers, strings, bools)."""
    if isinstance(v, (bool, str)):
        return True
    return isinstance(v, numbers.Number) and not (isinstance(v, float) and math.isnan(v))


def history_rows(run):
    """The rows that get copied: non-underscore plain values per step."""
    for row in run.scan_history(page_size=2000):
        step = row.get("_step")
        data = {k: v for k, v in row.items() if not k.startswith("_") and _plain(v)}
        if data:
            yield step, data


def old_projects(api, entity, only=None):
    names = sorted(p.name for p in api.projects(entity) if p.name.startswith(OLD_PREFIX) and p.name != MEDNEST_PROJECT)
    return [n for n in names if only in (None, n)]


def copy_run(wandb, run, proj, entity):
    config = dict(run.config)
    config["copied_from"] = f"{proj}/{run.id}"
    config["copied_from_created_at"] = run.created_at
    new = wandb.init(entity=entity, project=MEDNEST_PROJECT, id=run.id, name=run.name, group=proj,
                     tags=sorted(set(run.tags) | {proj}), notes=run.notes, config=config, resume="never", reinit=True)
    rows = 0
    for step, data in history_rows(run):
        new.log(data, step=int(step) if step is not None else None)
        rows += 1
    for k, v in run.summary.items():
        if not k.startswith("_") and _plain(v):
            new.summary[k] = v
    new.summary["copied_history_rows"] = rows
    new.finish(exit_code=0 if run.state == "finished" else 1)
    return rows


def _same(a, b):
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def verify_run(api, entity, proj, run):
    """List of problems (empty = the copy matches the original)."""
    try:
        new = api.run(f"{entity}/{MEDNEST_PROJECT}/{run.id}")
    except Exception as e:  # noqa: BLE001
        return [f"copy not found ({e!r})"]
    probs = []
    if new.name != run.name:
        probs.append(f"name {new.name!r} != {run.name!r}")
    if new.group != proj:
        probs.append(f"group {new.group!r} != {proj!r}")
    want_state = "finished" if run.state == "finished" else "failed"
    if new.state != want_state:
        probs.append(f"state {new.state} (original {run.state})")
    n_old = sum(1 for _ in history_rows(run))
    n_new = new.summary.get("copied_history_rows")
    if n_new != n_old:
        probs.append(f"history rows {n_new} != {n_old}")
    else:
        n_actual = sum(1 for _ in new.scan_history(page_size=2000))
        if n_actual < n_old:
            probs.append(f"only {n_actual} history rows stored in the copy, expected {n_old}")
    cfg_new = {k: v for k, v in new.config.items() if not k.startswith("copied_")}
    cfg_old = dict(run.config)
    if set(cfg_new) != set(cfg_old) or any(not _same(cfg_new[k], cfg_old[k]) for k in cfg_old if k in cfg_new):
        probs.append("config differs")
    for k, v in run.summary.items():
        if not k.startswith("_") and _plain(v) and not _same(new.summary.get(k), v):
            probs.append(f"summary[{k}] {new.summary.get(k)!r} != {v!r}")
            break
    return probs


def backup_run(run, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"id": run.id, "name": run.name, "state": run.state, "project": run.project, "group": run.group,
                   "tags": run.tags, "created_at": run.created_at, "url": run.url, "config": dict(run.config),
                   "summary": {k: v for k, v in run.summary.items() if _plain(v)},
                   "history": [{"_step": s, **d} for s, d in history_rows(run)]}, f, default=str)


def delete_project(api, entity, name):
    """Delete an (empty) project through the GraphQL deleteModel mutation. Returns True on success."""
    from wandb_gql import gql
    res = api.client.execute(gql("query($e:String!,$p:String!){project(name:$p, entityName:$e){id name}}"),
                             {"e": entity, "p": name})
    pid = (res.get("project") or {}).get("id")
    if not pid:
        return False
    out = api.client.execute(gql("mutation($input:DeleteModelInput!){deleteModel(input:$input){success}}"),
                             {"input": {"id": pid}})
    return bool((out.get("deleteModel") or {}).get("success"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="sudaksh14")
    ap.add_argument("--skip-runs", default="", help="comma-separated run ids never to touch")
    ap.add_argument("--only", default="", help="copy only these run ids (comma-separated), for a test")
    ap.add_argument("--only-project", default=None)
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--backup", default=None)
    ap.add_argument("--delete-old", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.delete_old and not (args.verify and args.backup):
        sys.exit("--delete-old needs --verify and --backup DIR")
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb

    api = wandb.Api(timeout=120)
    skip = {r for r in args.skip_runs.split(",") if r}
    only = {r for r in args.only.split(",") if r}
    existing = set()
    if any(p.name == MEDNEST_PROJECT for p in api.projects(args.entity)):
        existing = {r.id for r in api.runs(f"{args.entity}/{MEDNEST_PROJECT}")}
    projects = old_projects(api, args.entity, args.only_project)
    copied, skipped, failed, bad, deleted_projects = 0, 0, [], [], []
    for proj in projects:
        runs = list(api.runs(f"{args.entity}/{proj}"))
        proj_ok = True
        for run in runs:
            tag = f"{proj}/{run.id} ({run.name}, {run.state})"
            if only and run.id not in only:
                proj_ok = False
                continue
            if run.id in skip or run.state == "running":
                print(f"[mednest] SKIP {tag}: {'running' if run.state == 'running' else 'skip list'}")
                skipped += 1
                proj_ok = False
                continue
            if not args.no_copy and run.id not in existing:
                print(f"[mednest] copy {tag} -> {MEDNEST_PROJECT}", flush=True)
                if not args.dry_run:
                    try:
                        rows = copy_run(wandb, run, proj, args.entity)
                        existing.add(run.id)
                        copied += 1
                        print(f"[mednest]   {rows} history rows", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[mednest]   FAILED: {e!r}", flush=True)
                        failed.append(f"{proj}/{run.id}")
                        proj_ok = False
                        continue
            if args.verify:
                probs = verify_run(api, args.entity, proj, run)
                print(f"[mednest] verify {tag}: {'OK' if not probs else 'PROBLEMS ' + '; '.join(probs)}", flush=True)
                if probs:
                    bad.append(f"{proj}/{run.id}")
                    proj_ok = False
                    continue
            if args.backup and not args.dry_run:
                backup_run(run, os.path.join(args.backup, proj, f"{run.id}.json"))
        if args.delete_old and not args.dry_run:
            if not proj_ok or not runs:
                print(f"[mednest] KEEP {proj}: not every run verified/backed up (or no runs)", flush=True)
                continue
            for run in api.runs(f"{args.entity}/{proj}"):
                run.delete(delete_artifacts=False)
            ok = delete_project(api, args.entity, proj)
            print(f"[mednest] DELETED project {proj} ({len(runs)} runs): {'project removed' if ok else 'runs deleted, project shell remains (delete it in the UI)'}",
                  flush=True)
            deleted_projects.append((proj, ok))
    print(f"[mednest] copied {copied}, skipped {skipped}, failed {len(failed)} {failed}, verify problems {len(bad)} {bad}, "
          f"projects deleted {sum(1 for _, ok in deleted_projects if ok)} "
          f"(runs deleted, shell left: {[p for p, ok in deleted_projects if not ok]})")
    sys.exit(1 if failed or bad else 0)


if __name__ == "__main__":
    main()
