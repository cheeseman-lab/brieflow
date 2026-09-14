import json
from pathlib import Path

from lib.cluster.mozzarellm_io import organism_id_from_context, run_mozzarellm

screen_context = json.loads(
    Path(snakemake.input.screen_context).read_text(encoding="utf-8")
)

# the run directory is <cluster_dir>/mozzarellm/<run_name>, so the clustering is two up
cluster_dir = Path(snakemake.output[0]).parents[2]

result = run_mozzarellm(
    snakemake.input.cluster_anndata,
    cluster_dir,
    {},
    {},
    snakemake.params.leiden_resolution,
    snakemake.params.model,
    mode=snakemake.params.mode,
    mcp=snakemake.params.mcp,
    include_features=snakemake.params.include_features,
    include_strength=snakemake.params.include_strength,
    n_features=snakemake.params.n_features,
    fdr_threshold=snakemake.params.fdr_threshold,
    max_tokens=snakemake.params.max_tokens,
    max_workers=snakemake.params.max_workers,
    screen_name=snakemake.params.screen_name,
    screen_context=screen_context,
    organism_id=organism_id_from_context(screen_context),
    run_name=snakemake.params.run_name,
    resume=True,
)

print(f"Annotated {len(result['results'])} clusters in {result['run_dir']}")
print(f"Resumed from traces: {result['resumed']}")
print(f"Total cost: ${result['total_cost_usd']}")

# fail the job so its retry picks the finished clusters back up from the traces
if result["errors"]:
    raise RuntimeError(f"mozzarellm failed on clusters {result['errors']}")
