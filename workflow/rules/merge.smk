from lib.shared.target_utils import output_to_input

# Get merge approach to determine which rules to include
merge_approach = config.get("merge", {}).get("approach", "fast")

_merge_well_expand = ["row", "col"] if IMG_FMT == "zarr" else []
_merge_well_expand_all = ["row", "col"] if IMG_FMT == "zarr" else ["well"]

if merge_approach == "fast":
    rule fast_alignment:
        input:
            ancient(lambda wildcards: output_to_input(
                PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                PREPROCESS_OUTPUTS["combine_metadata_sbs"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["combine_phenotype_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                SBS_OUTPUTS["combine_sbs_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
        output:
            MERGE_OUTPUTS_MAPPED["fast_alignment"][0],
        params:
            sbs_metadata_cycle=config.get("merge", {}).get("sbs_metadata_cycle"),
            sbs_metadata_channel=config.get("merge", {}).get("sbs_metadata_channel"),
            ph_metadata_channel=config.get("merge", {}).get("ph_metadata_channel"),
            det_range=config.get("merge", {}).get("det_range"),
            score=config.get("merge", {}).get("score"),
            initial_sbs_tiles=config.get("merge", {}).get("initial_sbs_tiles"),
            initial_sites=config.get("merge", {}).get("initial_sites"),
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            metadata_align=config.get("merge", {}).get("metadata_align", False),
            alignment_flip_x=config.get("merge", {}).get("alignment_flip_x"),
            alignment_flip_y=config.get("merge", {}).get("alignment_flip_y"),
            alignment_rotate_90=config.get("merge", {}).get("alignment_rotate_90"),
            threshold_triangle=config.get("merge", {}).get("threshold_triangle"),
            seed_optimize=config.get("merge", {}).get("seed_optimize", False),
            seed_topk=config.get("merge", {}).get("seed_topk"),
        script:
            "../scripts/merge/fast_alignment.py"

    rule fast_merge:
        input:
            ancient(lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["combine_phenotype_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                SBS_OUTPUTS["combine_sbs_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            MERGE_OUTPUTS["fast_alignment"][0],
        output:
            MERGE_OUTPUTS_MAPPED["fast_merge"][0],
        params:
            det_range=config.get("merge", {}).get("det_range"),
            score=config.get("merge", {}).get("score"),
            threshold=config.get("merge", {}).get("threshold"),
            local_refinement=config.get("merge", {}).get("local_refinement"),
            warp_degree=config.get("merge", {}).get("warp_degree"),
            warp_iterations=config.get("merge", {}).get("warp_iterations"),
            warp_smoothing=config.get("merge", {}).get("warp_smoothing"),
        script:
            "../scripts/merge/fast_merge.py"


if merge_approach == "positions":
    rule positions_merge:
        input:
            ancient(lambda wildcards: output_to_input(
                PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                PREPROCESS_OUTPUTS["combine_metadata_sbs"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["combine_phenotype_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
            ancient(lambda wildcards: output_to_input(
                SBS_OUTPUTS["combine_sbs_info"],
                wildcards={"plate": wildcards.plate, "well": wildcards.well},
                expansion_values=_merge_well_expand,
                metadata_combos=merge_wildcard_combos,
            )),
        output:
            MERGE_OUTPUTS_MAPPED["positions_merge"][0],
            MERGE_OUTPUTS_MAPPED["positions_merge"][1],
            MERGE_OUTPUTS_MAPPED["positions_merge"][2],
            MERGE_OUTPUTS_MAPPED["positions_merge"][3],
            MERGE_OUTPUTS_MAPPED["positions_merge"][4],
            MERGE_OUTPUTS_MAPPED["positions_merge"][5],
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            threshold=config.get("merge", {}).get("threshold"),
            phenotype_dimensions=config.get("merge", {}).get("phenotype_dimensions"),
            sbs_dimensions=config.get("merge", {}).get("sbs_dimensions"),
            flipud=config.get("merge", {}).get("flipud", False),
            fliplr=config.get("merge", {}).get("fliplr", False),
            rot90=config.get("merge", {}).get("rot90", 0),
            sbs_metadata_cycle=config.get("merge", {}).get("sbs_metadata_cycle"),
            sbs_metadata_channel=config.get("merge", {}).get("sbs_metadata_channel"),
            ph_metadata_channel=config.get("merge", {}).get("ph_metadata_channel"),
            metadata_align=config.get("merge", {}).get("metadata_align", False),
            alignment_flip_x=config.get("merge", {}).get("alignment_flip_x"),
            alignment_flip_y=config.get("merge", {}).get("alignment_flip_y"),
            alignment_rotate_90=config.get("merge", {}).get("alignment_rotate_90"),
            phenotype_pixel_size=config.get("merge", {}).get("phenotype_pixel_size"),
            sbs_pixel_size=config.get("merge", {}).get("sbs_pixel_size"),
            image_qc=config.get("merge", {}).get("positions_image_qc", True),
            phenotype_label_template=lambda wildcards: str(PHENOTYPE_OUTPUTS["segment_phenotype"][0]),
            sbs_label_template=lambda wildcards: str(SBS_OUTPUTS["segment_sbs"][0]),
            phenotype_image_template=lambda wildcards: str(PHENOTYPE_OUTPUTS["align_phenotype"][0]),
            sbs_image_template=lambda wildcards: str(SBS_OUTPUTS["align_sbs"][0]),
            phenotype_dapi_index=config.get("phenotype", {}).get("dapi_index"),
            sbs_dapi_index=config.get("sbs", {}).get("dapi_index"),
        script:
            "../scripts/merge/positions_merge.py"


rule format_merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["positions_merge"][0]
            if config.get("merge", {}).get("approach", "fast") == "positions"
            else MERGE_OUTPUTS["fast_merge"][0]
        ),
        ancient(lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
        ancient(lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
        phenotype_metadata=ancient(lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
        sbs_metadata=ancient(lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_sbs"],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"][0],
    params:
        phenotype_dimensions=config.get("merge", {}).get("phenotype_dimensions"),
        sbs_dimensions=config.get("merge", {}).get("sbs_dimensions"),
    script:
        "../scripts/merge/format_merge.py"


rule deduplicate_merge:
    input:
        MERGE_OUTPUTS["format_merge"][0],
        ancient(lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
        ancient(lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
    output:
        deduplication_stats=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][0],
        deduplicated_data=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][1],
        final_sbs_matching_rates=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][2],
        final_phenotype_matching_rates=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][3],
    params:
        sbs_dedup_prior=config.get("merge", {}).get("sbs_dedup_prior"),
        pheno_dedup_prior=config.get("merge", {}).get("pheno_dedup_prior"),
    script:
        "../scripts/merge/deduplicate_merge.py"


rule final_merge:
    input:
        MERGE_OUTPUTS["deduplicate_merge"][1],
        ancient(lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0],
            wildcards={"plate": wildcards.plate, "well": wildcards.well},
            expansion_values=_merge_well_expand,
            metadata_combos=merge_wildcard_combos,
        )),
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"][0],
    script:
        "../scripts/merge/final_merge.py"


rule eval_merge:
    input:
        deduplicated_merge_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["deduplicate_merge"][1],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        combine_cells_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=sbs_wildcard_combos,
            ancient_output=True,
        ),
        min_phenotype_cp_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=phenotype_wildcard_combos,
            ancient_output=True,
        ),
        dedup_stats_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["deduplicate_merge"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        formatted_merge_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["format_merge"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        sbs_info_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_sbs_info"],
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=sbs_wildcard_combos,
            ancient_output=True,
        ),
        phenotype_info_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["combine_phenotype_info"],
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=phenotype_wildcard_combos,
            ancient_output=True,
        ),
        sbs_metadata_paths=lambda wildcards: output_to_input(
            ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=sbs_wildcard_combos,
        ),
        phenotype_metadata_paths=lambda wildcards: output_to_input(
            ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
            wildcards=wildcards,
            expansion_values=_merge_well_expand_all,
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        merge_summary=MERGE_OUTPUTS_MAPPED["eval_merge"][0],
        sbs_to_ph_matching_rates_tsv=MERGE_OUTPUTS_MAPPED["eval_merge"][1],
        sbs_to_ph_matching_rates_png=MERGE_OUTPUTS_MAPPED["eval_merge"][2],
        ph_to_sbs_matching_rates_tsv=MERGE_OUTPUTS_MAPPED["eval_merge"][3],
        ph_to_sbs_matching_rates_png=MERGE_OUTPUTS_MAPPED["eval_merge"][4],
        all_cells_by_channel_min=MERGE_OUTPUTS_MAPPED["eval_merge"][5],
        cells_with_channel_min_0=MERGE_OUTPUTS_MAPPED["eval_merge"][6],
        dedup_summaries=MERGE_OUTPUTS_MAPPED["eval_merge"][7],
    script:
        "../scripts/merge/eval_merge.py"


# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,