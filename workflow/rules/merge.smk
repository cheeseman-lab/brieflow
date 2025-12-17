from lib.shared.target_utils import output_to_input

# Get merge approach to determine which rules to include
merge_approach = config.get("merge", {}).get("approach", "fast")

if merge_approach == "fast":
    rule fast_alignment:
        input:
            ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
            ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
            ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
            ancient(SBS_OUTPUTS["combine_sbs_info"]),
        output:
            MERGE_OUTPUTS_MAPPED["fast_alignment"][0],
        threads: 16
        params:
            sbs_metadata_cycle=config["merge"]["sbs_metadata_cycle"],
            sbs_metadata_channel=config["merge"].get("sbs_metadata_channel"),
            ph_metadata_channel=config["merge"].get("ph_metadata_channel"),
            det_range=config["merge"]["det_range"],
            score=config["merge"]["score"],
            initial_sites=lambda wildcards: config["merge"]["initial_sites"][int(wildcards.plate)][wildcards.well],
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            alignment_flip_x=config["merge"].get("alignment_flip_x"),
            alignment_flip_y=config["merge"].get("alignment_flip_y"),
            alignment_rotate_90=config["merge"].get("alignment_rotate_90"),
        script:
            "../scripts/merge/fast_alignment.py"

    rule fast_merge:
        input:
            ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
            ancient(SBS_OUTPUTS["combine_sbs_info"]),
            MERGE_OUTPUTS["fast_alignment"][0],
        output:
            MERGE_OUTPUTS_MAPPED["fast_merge"][0],
        threads: 16
        params:
            det_range=config["merge"]["det_range"],
            score=config["merge"]["score"],
            threshold=config["merge"]["threshold"],
        script:
            "../scripts/merge/fast_merge.py"


if merge_approach == "stitch":
    rule estimate_stitch_phenotype:
        input:
            phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        output:
            phenotype_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_phenotype"][0],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            flipud=config.get("merge", {}).get("flipud", False),
            fliplr=config.get("merge", {}).get("fliplr", False),
            rot90=config.get("merge", {}).get("rot90", 0),
            data_type="phenotype",
            phenotype_pixel_size=config.get("merge", {}).get("phenotype_pixel_size"),
        script:
            "../scripts/merge/estimate_stitch.py"


    rule estimate_stitch_sbs:
        input:
            sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
            phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        output:
            sbs_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_sbs"][0],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            flipud=config.get("merge", {}).get("flipud", False),
            fliplr=config.get("merge", {}).get("fliplr", False),
            rot90=config.get("merge", {}).get("rot90", 0),
            data_type="sbs",
            sbs_metadata_cycle=config["merge"]["sbs_metadata_cycle"],
            sbs_metadata_channel=config["merge"].get("sbs_metadata_channel"),
            sbs_pixel_size=config.get("merge", {}).get("sbs_pixel_size"),
            alignment_flip_x=config["merge"].get("alignment_flip_x"),
            alignment_flip_y=config["merge"].get("alignment_flip_y"),
            alignment_rotate_90=config["merge"].get("alignment_rotate_90"),
        script:
            "../scripts/merge/estimate_stitch.py"


    rule stitch_phenotype:
        input:
            phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
            phenotype_stitch_config=MERGE_OUTPUTS["estimate_stitch_phenotype"][0],
            phenotype_tiles=lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["align_phenotype"],
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=phenotype_wildcard_combos,
                ancient_output=True,
            ),
            phenotype_masks=lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["segment_phenotype"][0],
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=phenotype_wildcard_combos,
                ancient_output=True,
            ),
        output:
            phenotype_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_phenotype"][0],
            phenotype_qc_plot=MERGE_OUTPUTS_MAPPED["stitch_phenotype"][1],
            phenotype_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_phenotype"][2],
            phenotype_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_phenotype"][3],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            data_type="phenotype",
            flipud=config.get("merge", {}).get("flipud", False),
            fliplr=config.get("merge", {}).get("fliplr", False),
            rot90=config.get("merge", {}).get("rot90", 0),
            stitched_image=config.get("merge", {}).get("stitched_image", True),     
        script:
            "../scripts/merge/stitch.py"


    rule stitch_sbs:
        input:
            sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
            sbs_stitch_config=MERGE_OUTPUTS["estimate_stitch_sbs"][0],
            sbs_tiles=lambda wildcards: output_to_input(
                SBS_OUTPUTS["align_sbs"],
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=sbs_wildcard_combos,
                ancient_output=True,
            ),
            sbs_masks=lambda wildcards: output_to_input(
                SBS_OUTPUTS["segment_sbs"][0],
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=sbs_wildcard_combos,
                ancient_output=True,
            ),
        output:
            sbs_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_sbs"][0],
            sbs_qc_plot=MERGE_OUTPUTS_MAPPED["stitch_sbs"][1],
            sbs_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_sbs"][2],
            sbs_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_sbs"][3],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            data_type="sbs",
            flipud=config.get("merge", {}).get("flipud", False),
            fliplr=config.get("merge", {}).get("fliplr", False),
            rot90=config.get("merge", {}).get("rot90", 0),
            overlap_fraction=config.get("merge", {}).get("overlap_fraction"),
            stitched_image=config.get("merge", {}).get("stitched_image", True),     
        script:
            "../scripts/merge/stitch.py"


    rule stitch_alignment:
        input:
            phenotype_positions=MERGE_OUTPUTS["stitch_phenotype"][0],
            sbs_positions=MERGE_OUTPUTS["stitch_sbs"][0],
        output:
            scaled_phenotype_positions=MERGE_OUTPUTS_MAPPED["stitch_alignment"][0],      
            phenotype_triangles=MERGE_OUTPUTS_MAPPED["stitch_alignment"][1],             
            sbs_triangles=MERGE_OUTPUTS_MAPPED["stitch_alignment"][2],                   
            alignment_params=MERGE_OUTPUTS_MAPPED["stitch_alignment"][3],                
            alignment_summary=MERGE_OUTPUTS_MAPPED["stitch_alignment"][4],
            transformed_phenotype_positions=MERGE_OUTPUTS_MAPPED["stitch_alignment"][5],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            score=config["merge"]["score"],
        script:
            "../scripts/merge/stitch_alignment.py"


    rule stitch_merge:
        input:
            scaled_phenotype_positions=MERGE_OUTPUTS["stitch_alignment"][0],      
            sbs_positions=MERGE_OUTPUTS["stitch_sbs"][0],  # sbs_cell_positions
            alignment_params=MERGE_OUTPUTS["stitch_alignment"][3],
            transformed_phenotype_positions=MERGE_OUTPUTS["stitch_alignment"][5],               
        output:
            raw_matches=MERGE_OUTPUTS_MAPPED["stitch_merge"][0],
            merged_cells=MERGE_OUTPUTS_MAPPED["stitch_merge"][1],
            merge_summary=MERGE_OUTPUTS_MAPPED["stitch_merge"][2],
        threads: 16
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            threshold=config["merge"]["threshold"],
            score=config["merge"]["score"],
        script:
            "../scripts/merge/stitch_merge.py"


rule format_merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["stitch_merge"][1]
            if config.get("merge", {}).get("approach", "fast") == "stitch"
            else MERGE_OUTPUTS["fast_merge"][0]
        ),
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"][0],
    threads: 16
    params:
        approach=config.get("merge", {}).get("approach", "fast"),
        phenotype_dimensions=config.get("merge", {}).get("phenotype_dimensions"),
        sbs_dimensions=config.get("merge", {}).get("sbs_dimensions"),
    script:
        "../scripts/merge/format_merge.py"


rule deduplicate_merge:
    input:
        MERGE_OUTPUTS["format_merge"][0],
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        deduplication_stats=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][0],
        deduplicated_data=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][1],
        final_sbs_matching_rates=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][2],
        final_phenotype_matching_rates=MERGE_OUTPUTS_MAPPED["deduplicate_merge"][3],
    threads: 16
    params:
        approach=config.get("merge", {}).get("approach", "fast"),
        sbs_dedup_prior=config.get("merge", {}).get("sbs_dedup_prior"),
        pheno_dedup_prior=config.get("merge", {}).get("pheno_dedup_prior"),
    script:
        "../scripts/merge/deduplicate_merge.py"


rule final_merge:
    input:
        MERGE_OUTPUTS["deduplicate_merge"][1],
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0]),
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"][0],
    threads: 16
    params:
        approach=config.get("merge", {}).get("approach", "fast"),
    script:
        "../scripts/merge/final_merge.py"


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
    params: 
        heatmap_plate_sbs=config["sbs"].get("heatmap_plate", "6W"),  
        heatmap_shape_sbs=config["sbs"].get("heatmap_shape", "6W_sbs"),
        heatmap_plate_ph=config["phenotype"].get("heatmap_plate", "6W"), 
        heatmap_shape_ph=config["phenotype"].get("heatmap_shape", "6W_ph"),
    output:
        cell_mapping_stats=MERGE_OUTPUTS_MAPPED["eval_merge"][0],
        sbs_to_ph_matching_rates_tsv=MERGE_OUTPUTS_MAPPED["eval_merge"][1],
        sbs_to_ph_matching_rates_png=MERGE_OUTPUTS_MAPPED["eval_merge"][2],
        ph_to_sbs_matching_rates_tsv=MERGE_OUTPUTS_MAPPED["eval_merge"][3],
        ph_to_sbs_matching_rates_png=MERGE_OUTPUTS_MAPPED["eval_merge"][4],
        all_cells_by_channel_min=MERGE_OUTPUTS_MAPPED["eval_merge"][5],
        cells_with_channel_min_0=MERGE_OUTPUTS_MAPPED["eval_merge"][6],
    threads: 16
    script:
        "../scripts/merge/eval_merge.py"


rule summarize_merge:
    input:
        alignment_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["stitch_alignment"][4],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        merge_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["stitch_merge"][2],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        dedup_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["deduplicate_merge"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        sbs_matching_rates_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["deduplicate_merge"][2],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        phenotype_matching_rates_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["deduplicate_merge"][3],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
    output:
        alignment_summaries=MERGE_OUTPUTS_MAPPED["summarize_merge"][0],
        cell_merge_summaries=MERGE_OUTPUTS_MAPPED["summarize_merge"][1],
        dedup_summaries=MERGE_OUTPUTS_MAPPED["summarize_merge"][2],
        sbs_matching_summaries=MERGE_OUTPUTS_MAPPED["summarize_merge"][3],
        phenotype_matching_summaries=MERGE_OUTPUTS_MAPPED["summarize_merge"][4],
    threads: 16
    params:
        plate=lambda wildcards: wildcards.plate,
        wells=lambda wildcards: [
            str(combo["well"]) 
            for combo in merge_wildcard_combos.to_dict('records') 
            if str(combo["plate"]) == str(wildcards.plate)
        ],
    script:
        "../scripts/merge/summarize_merge.py"


# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,