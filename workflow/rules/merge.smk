from lib.shared.target_utils import output_to_input

# Get merge approach to determine which rules to include
merge_approach = config.get("merge", {}).get("approach", "tile")

rule estimate_stitch_phenotype:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
    output:
        phenotype_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_phenotype"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        flipud=config.get("merge", {}).get("flipud", False),
        fliplr=config.get("merge", {}).get("fliplr", False),
        rot90=config.get("merge", {}).get("rot90", 0),
        data_type="phenotype",
    script:
        "../scripts/merge/estimate_stitch_phenotype.py"


rule estimate_stitch_sbs:
    input:
        sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
    output:
        sbs_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_sbs"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        flipud=config.get("merge", {}).get("flipud", False),
        fliplr=config.get("merge", {}).get("fliplr", False),
        rot90=config.get("merge", {}).get("rot90", 0),
        data_type="sbs",
        # SBS-specific params
        sbs_metadata_filters={"cycle": config["merge"]["sbs_metadata_cycle"]},
    script:
        "../scripts/merge/estimate_stitch_sbs.py"


rule stitch_phenotype_well:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        phenotype_stitch_config=MERGE_OUTPUTS["estimate_stitch_phenotype"][0],
    output:
        phenotype_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_phenotype_well"][0],
        phenotype_qc_plot=MERGE_OUTPUTS_MAPPED["stitch_phenotype_well"][1],  
        phenotype_stitched_image=temp(MERGE_OUTPUTS_MAPPED["stitch_phenotype_well"][2]), 
        phenotype_stitched_mask=temp(MERGE_OUTPUTS_MAPPED["stitch_phenotype_well"][3]), 
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        data_type="phenotype",
        flipud=config.get("merge", {}).get("flipud", False),
        fliplr=config.get("merge", {}).get("fliplr", False),
        rot90=config.get("merge", {}).get("rot90", 0),
        stitched_image=config.get("merge", {}).get("stitched_image", True),     
    script:
        "../scripts/merge/well_stitching.py"


rule stitch_sbs_well:
    input:
        sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
        sbs_stitch_config=MERGE_OUTPUTS["estimate_stitch_sbs"][0],
    output:
        sbs_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_sbs_well"][0],
        sbs_qc_plot=MERGE_OUTPUTS_MAPPED["stitch_sbs_well"][1],
        sbs_stitched_image=temp(MERGE_OUTPUTS_MAPPED["stitch_sbs_well"][2]),
        sbs_stitched_mask=temp(MERGE_OUTPUTS_MAPPED["stitch_sbs_well"][3]), 
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        data_type="sbs",
        flipud=config.get("merge", {}).get("flipud", False),
        fliplr=config.get("merge", {}).get("fliplr", False),
        rot90=config.get("merge", {}).get("rot90", 0),
        stitched_image=config.get("merge", {}).get("stitched_image", True),     
    script:
        "../scripts/merge/well_stitching.py"


if merge_approach == "tile":
    rule fast_alignment:
        input:
            ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
            ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
            ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
            ancient(SBS_OUTPUTS["combine_sbs_info"]),
        output:
            MERGE_OUTPUTS_MAPPED["fast_alignment"][0],
        params:
            sbs_metadata_filters={"cycle": config["merge"]["sbs_metadata_cycle"]},
            det_range=config["merge"]["det_range"],
            score=config["merge"]["score"],
            initial_sites=config["merge"]["initial_sites"],
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
        script:
            "../scripts/merge/fast_alignment.py"

    rule merge_tile:
        input:
            ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
            ancient(SBS_OUTPUTS["combine_sbs_info"]),
            MERGE_OUTPUTS["fast_alignment"][0],
        output:
            MERGE_OUTPUTS_MAPPED["merge_tile"][0],
        params:
            det_range=config["merge"]["det_range"],
            score=config["merge"]["score"],
            threshold=config["merge"]["threshold"],
        script:
            "../scripts/merge/merge.py"

 
if merge_approach == "well":
    rule well_alignment:
        input:
            phenotype_positions=MERGE_OUTPUTS["stitch_phenotype_well"][0],  # phenotype_cell_positions
            sbs_positions=MERGE_OUTPUTS["stitch_sbs_well"][0],  # sbs_cell_positions
        output:
            scaled_phenotype_positions=temp(MERGE_OUTPUTS["well_alignment"][0]),      
            phenotype_triangles=temp(MERGE_OUTPUTS["well_alignment"][1]),             
            sbs_triangles=temp(MERGE_OUTPUTS["well_alignment"][2]),                   
            alignment_params=temp(MERGE_OUTPUTS["well_alignment"][3]),                
            alignment_summary=temp(MERGE_OUTPUTS["well_alignment"][4]),
            transformed_phenotype_positions=MERGE_OUTPUTS["well_alignment"][5],
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            score=config["merge"]["score"],
        script:
            "../scripts/merge/well_alignment.py"

    rule well_cell_merge:
        input:
            scaled_phenotype_positions=MERGE_OUTPUTS["well_alignment"][0],      
            sbs_positions=MERGE_OUTPUTS["stitch_sbs_well"][0],  # sbs_cell_positions
            alignment_params=MERGE_OUTPUTS["well_alignment"][3],
            transformed_phenotype_positions=MERGE_OUTPUTS["well_alignment"][5],               
        output:
            raw_matches=temp(MERGE_OUTPUTS["well_cell_merge"][0]),                    
            merged_cells=MERGE_OUTPUTS["well_cell_merge"][1],                   
            merge_summary=temp(MERGE_OUTPUTS["well_cell_merge"][2]),                  
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
            threshold=config["merge"]["threshold"],
            score=config["merge"]["score"],
        script:
            "../scripts/merge/well_cell_merge.py"

    rule well_merge_deduplicate:
        input:
            raw_matches=MERGE_OUTPUTS["well_cell_merge"][0],                    
            merged_cells=MERGE_OUTPUTS["well_cell_merge"][1],
            sbs_cells=ancient(SBS_OUTPUTS["combine_cells"]),
            phenotype_min_cp=ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
        output:
            deduplicated_cells=MERGE_OUTPUTS["well_merge_deduplicate"][0],      
            deduplication_summary=temp(MERGE_OUTPUTS["well_merge_deduplicate"][1]),
            sbs_matching_rates=MERGE_OUTPUTS["well_merge_deduplicate"][2],
            phenotype_matching_rates=MERGE_OUTPUTS["well_merge_deduplicate"][3],
        params:
            plate=lambda wildcards: wildcards.plate,
            well=lambda wildcards: wildcards.well,
        script:
            "../scripts/merge/well_merge_deduplicate.py"

    rule merge_well:
        input:
            MERGE_OUTPUTS["well_merge_deduplicate"][0]  # deduplicated_cells
        output:
            MERGE_OUTPUTS_MAPPED["merge_well"][0],
        params:
            approach=config.get("merge", {}).get("approach", "tile"),
        run:
            import pandas as pd
            from lib.shared.file_utils import validate_dtypes
            
            # Copy the chosen approach to the standard merge output
            merge_data = validate_dtypes(pd.read_parquet(input[0]))
            merge_data.to_parquet(output[0])
            
            approach = params.approach
            print(f"Using {approach} merge approach")
            print(f"Merged {len(merge_data)} cells")


rule format_merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["merge_well"][0]  # Use well merge output for well approach
            if config.get("merge", {}).get("approach", "tile") == "well" 
            else MERGE_OUTPUTS["merge_tile"][0]  # Use tile merge output for tile approach
        ),
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"][0],
    params:
        approach=config.get("merge", {}).get("approach", "tile"),
    script:
        "../scripts/merge/format_merge.py"



rule deduplicate_merge:
    input:
        MERGE_OUTPUTS["format_merge"],
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["deduplicate_merge"],
    params:
        approach=config.get("merge", {}).get("approach", "tile"),
    run:
        approach = params.approach
        if approach == "well":
            # For well approach, we already deduplicated, so just copy the formatted data
            import pandas as pd
            from lib.shared.file_utils import validate_dtypes
            
            formatted_data = validate_dtypes(pd.read_parquet(input[0]))
            
            # Create dummy outputs to match expected structure
            # Save the formatted data as "deduplicated" (no additional deduplication needed)
            formatted_data.to_parquet(output[1]) 
            
            # Create dummy stats files
            import pandas as pd
            dummy_stats = pd.DataFrame({"info": ["No additional deduplication - already done in well_merge_deduplicate"]})
            dummy_stats.to_csv(output[0], sep='\t', index=False)
            dummy_stats.to_csv(output[2], sep='\t', index=False)
            dummy_stats.to_csv(output[3], sep='\t', index=False) 
        else:
            # For tile approach, run the actual deduplication script
            shell("python {workflow.basedir}/scripts/merge/deduplicate_merge.py")


rule final_merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["format_merge"][0]  # Use formatted data directly for well approach
            if config.get("merge", {}).get("approach", "tile") == "well" 
            else MERGE_OUTPUTS["deduplicate_merge"][1]  # Use deduplicated data for tile approach
        ),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0]),
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"][0],
    params:
        approach=config.get("merge", {}).get("approach", "tile"),
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
    output:
        MERGE_OUTPUTS_MAPPED["eval_merge"],
    script:
        "../scripts/merge/eval_merge.py"

rule aggregate_well_summaries:
    input:
        alignment_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["well_alignment"][4],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        merge_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["well_cell_merge"][2],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        dedup_summary_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["well_merge_deduplicate"][1],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        sbs_matching_rates_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["well_merge_deduplicate"][2],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        phenotype_matching_rates_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["well_merge_deduplicate"][3],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        phenotype_cell_positions_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["stitch_phenotype_well"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
        sbs_cell_positions_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["stitch_sbs_well"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=merge_wildcard_combos,
        ),
    output:
        alignment_summaries=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][0],
        cell_merge_summaries=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][1],
        dedup_summaries=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][2],
        sbs_matching_summaries=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][3],
        phenotype_matching_summaries=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][4],
        phenotype_cell_positions_plot=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][5],
        sbs_cell_positions_plot=MERGE_OUTPUTS_MAPPED["aggregate_well_summaries"][6],
    params:
        plate=lambda wildcards: wildcards.plate,
    script:
        "../scripts/merge/aggregate_well_summaries.py"

# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,