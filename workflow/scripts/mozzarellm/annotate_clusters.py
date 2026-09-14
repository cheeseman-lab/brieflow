import json
from pathlib import Path

from lib.mozzarellm.annotate_clusters import run_mozzarellm

screen_context = json.loads(
    Path(snakemake.input.screen_context).read_text(encoding="utf-8")
)

# the run directory is <cluster_dir>/mozzarellm/<run_name>, so the clustering is two up
cluster_dir = Path(snakemake.output[0]).parents[2]

result = run_mozzarellm(
    snakemake.input.cluster_anndata,
    cluster_dir,
    snakemake.params.leiden_resolution,
    snakemake.params.model,
    screen_context,
    mode=snakemake.params.mode,
    mcp=snakemake.params.mcp,
    include_features=snakemake.params.include_features,
    include_strength=snakemake.params.include_strength,
    n_features=snakemake.params.n_features,
    fdr_threshold=snakemake.params.fdr_threshold,
    max_tokens=snakemake.params.max_tokens,
    max_workers=snakemake.params.max_workers,
    screen_name=snakemake.params.screen_name,
    run_name=snakemake.params.run_name,
    resume=True,
)

print(f"Annotated {len(result['results'])} clusters in {result['run_dir']}")
print(f"Resumed from traces: {result['resumed']}")
print(f"Total cost: ${result['total_cost_usd']}")

# the panel's outputs are already written, so a few dead clusters must not discard them
if result["errors"]:
    print(f"mozzarellm failed on clusters {result['errors']}")
if len(result["errors"]) > snakemake.params.max_failed_clusters:
    raise RuntimeError(
        f"{len(result['errors'])} clusters failed, over the "
        f"max_failed_clusters tolerance of {snakemake.params.max_failed_clusters}: "
        f"{result['errors']}"
    )
