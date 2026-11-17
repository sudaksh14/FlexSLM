WANDB_PROJECT_NAME = "FlexSLM"

# Every MedNest run logs into ONE W&B project (user, 2026-11-17). The per-config names
# (MedNest_s2_nested, MedNest_s1_pilot_..._9L4S, ...) become the run's group and a tag,
# so configs keep their names and runs stay filterable inside the single project.
MEDNEST_PROJECT = "MedNest"


def wandb_target(project_name: str):
    """(project, group) for a config's wandb_project_name."""
    if project_name.startswith("MedNest"):
        return MEDNEST_PROJECT, project_name
    return project_name, None
