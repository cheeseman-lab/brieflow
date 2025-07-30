from lib.shared.target_utils import output_to_input


# Stitching rules (always available for enhanced approach)
rule estimate_stitch:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
    output:
        phenotype_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_phenotype"],
        sbs_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_sbs"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False), 
        rot90=config.get("stitch", {}).get("rot90", 0),
        channel=config.get("stitch", {}).get("channel", 0),
        # Add SBS cycle filtering like fast_alignment
        sbs_metadata_filters={"cycle": config["merge"]["sbs_metadata_cycle"]},
    threads: 8
    script:
        "../scripts/merge/estimate_stitch.py"


rule stitch_wells:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
        phenotype_stitch_config=MERGE_OUTPUTS["estimate_stitch_phenotype"],
        sbs_stitch_config=MERGE_OUTPUTS["estimate_stitch_sbs"],
    output:
        phenotype_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_phenotype_image"],
        phenotype_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_phenotype_mask"],
        phenotype_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_phenotype_positions"],
        sbs_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_sbs_image"],
        sbs_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_sbs_mask"],
        sbs_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_sbs_positions"],
        phenotype_overlay=MERGE_OUTPUTS_MAPPED["stitch_phenotype_overlay"],
        sbs_overlay=MERGE_OUTPUTS_MAPPED["stitch_sbs_overlay"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False),
        rot90=config.get("stitch", {}).get("rot90", 0),
        overlap_percent=config.get("stitch", {}).get("overlap_percent", 0.05),
        create_overlay=config.get("stitch", {}).get("create_overlay", True),
    script:
        "../scripts/merge/stitch_wells.py"


# Original fast alignment approach (kept for backwards compatibility)
rule fast_alignment:
    input:
        ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
        ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
        ancient(SBS_OUTPUTS["combine_sbs_info"]),
    output:
        MERGE_OUTPUTS_MAPPED["fast_alignment"],
    params:
        sbs_metadata_filters={"cycle": config["merge"]["sbs_metadata_cycle"]},
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        initial_sites=config["merge"]["initial_sites"],
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/merge/fast_alignment.py"


# Enhanced well merge using stitched cell positions
rule enhanced_well_merge:
    input:
        phenotype_positions=MERGE_OUTPUTS["stitch_phenotype_positions"],
        sbs_positions=MERGE_OUTPUTS["stitch_sbs_positions"],
    output:
        MERGE_OUTPUTS_MAPPED["enhanced_well_merge"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        threshold=config["merge"]["threshold"],
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/merge/well_merge.py"


# Original merge approach (tile-by-tile)
rule merge_legacy:
    input:
        ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
        ancient(SBS_OUTPUTS["combine_sbs_info"]),
        MERGE_OUTPUTS["fast_alignment"],
    output:
        MERGE_OUTPUTS_MAPPED["merge_legacy"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        threshold=config["merge"]["threshold"],
    script:
        "../scripts/merge/merge_legacy.py"

# Main merge rule - chooses approach based on configuration  
rule merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["enhanced_well_merge"] 
            if config.get("merge", {}).get("approach", "legacy") == "enhanced"
            else MERGE_OUTPUTS["merge_legacy"]
        )
    output:
        MERGE_OUTPUTS_MAPPED["merge"],
    params:
        approach=config.get("merge", {}).get("approach", "legacy"),
    run:
        import pandas as pd
        from lib.shared.file_utils import validate_dtypes
        
        # Simply copy the chosen approach to the standard merge output
        merge_data = validate_dtypes(pd.read_parquet(input[0]))
        merge_data.to_parquet(output[0])
        
        approach = params.approach
        print(f"Using {approach} merge approach")
        print(f"Merged {len(merge_data)} cells")


# Enhanced well merge alternative name for backwards compatibility
rule well_merge:
    input:
        phenotype_positions=MERGE_OUTPUTS["stitch_phenotype_positions"],
        sbs_positions=MERGE_OUTPUTS["stitch_sbs_positions"],
    output:
        MERGE_OUTPUTS_MAPPED["well_merge"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        threshold=config["merge"]["threshold"],
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/merge/well_merge.py"


# Format merge data (works with both approaches)
rule format_merge:
    input:
        MERGE_OUTPUTS["merge"],
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"],
    script:
        "../scripts/merge/format_merge.py"


# Deduplicate merge data (unchanged)
rule deduplicate_merge:
    input:
        MERGE_OUTPUTS["format_merge"],
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["deduplicate_merge"],
    script:
        "../scripts/merge/deduplicate_merge.py"


# Final merge with all feature data (unchanged)
rule final_merge:
    input:
        MERGE_OUTPUTS["deduplicate_merge"][1],
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0]),
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"],
    script:
        "../scripts/merge/final_merge.py"


# Evaluate merge (unchanged)
rule eval_merge:
    input:
        format_merge_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["format_merge"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        combine_cells_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=sbs_wildcard_combos,
            ancient_output=True,
        ),
        min_phenotype_cp_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=phenotype_wildcard_combos,
            ancient_output=True,
        ),
    output:
        MERGE_OUTPUTS_MAPPED["eval_merge"],
    script:
        "../scripts/merge/eval_merge.py"


# Comparison rule to compare both approaches (optional)
rule compare_approaches:
    input:
        legacy=MERGE_OUTPUTS["merge_legacy"],
        enhanced=MERGE_OUTPUTS["enhanced_well_merge"],
    output:
        MERGE_FP / "comparison" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "approach_comparison", "parquet"
        ),
    script:
        "../scripts/merge/compare_approaches.py"


# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,