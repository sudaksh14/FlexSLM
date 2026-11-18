"""Read-only inventory of the W&B projects of an entity (run on a compute node).

    python scripts/wandb_inventory.py [--entity sudaksh14]

Prints every project with its run count and run states, the MedNest project's runs, and which
project-deletion calls this wandb version offers. Changes nothing.
"""
import argparse
import collections
import os

os.environ.setdefault("WANDB_SILENT", "true")
import wandb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--entity", default="sudaksh14")
args = ap.parse_args()
api = wandb.Api(timeout=120)
print("wandb", wandb.__version__, "| Project.delete:", hasattr(wandb.apis.public.Project, "delete"),
      "| Api.delete_project:", hasattr(api, "delete_project"))
projects = list(api.projects(args.entity))
print(f"{len(projects)} projects in {args.entity}")
for p in sorted(projects, key=lambda p: p.name):
    runs = list(api.runs(f"{args.entity}/{p.name}"))
    states = collections.Counter(r.state for r in runs)
    print(f"  {p.name:55s} runs {len(runs):3d} {dict(states)}")
mn = {r.id: r for r in api.runs(f"{args.entity}/MedNest")} if any(p.name == "MedNest" for p in projects) else {}
print(f"MedNest: {len(mn)} runs; groups: {len({r.group for r in mn.values()})}")
