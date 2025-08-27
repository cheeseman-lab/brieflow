from lib.shared.target_utils import output_to_input


rule estimate_stitch_phenotype:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
    output:
        phenotype_stitch_config=MERGE_OUTPUTS_MAPPED["estimate_stitch_phenotype"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False), 
        rot90=config.get("stitch", {}).get("rot90", 0),
        channel=config.get("stitch", {}).get("channel", 0),
        data_type="phenotype",
    resources:
        mem_mb=15000,
        cpus_per_task=8,
        runtime=180,  # 3 hours for phenotype (image registration)
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
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False), 
        rot90=config.get("stitch", {}).get("rot90", 0),
        channel=config.get("stitch", {}).get("channel", 0),
        data_type="sbs",
        # SBS-specific params
        sbs_metadata_filters={"cycle": config["merge"]["sbs_metadata_cycle"]},
    resources:
        mem_mb=8000,   # Less memory needed for coordinate-based approach
        cpus_per_task=4,   # Fewer CPUs needed
        runtime=60,    # 1 hour should be plenty for coordinate-based
    script:
        "../scripts/merge/estimate_stitch_sbs.py"


rule stitch_phenotype_well:
    input:
        phenotype_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_phenotype"]),
        phenotype_stitch_config=MERGE_OUTPUTS["estimate_stitch_phenotype"],
    output:
        phenotype_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_phenotype_image"],
        phenotype_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_phenotype_mask"],
        phenotype_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_phenotype_positions"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        data_type="phenotype",
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False),
        rot90=config.get("stitch", {}).get("rot90", 0),
        overlap_percent=config.get("stitch", {}).get("overlap_percent", 0.05),
    resources:
        mem_mb=400000,     # 400GB for maximum safety margin
        cpus_per_task=8,
        runtime=180,       # 3 hours
        partition="20"     # Force high-memory nodes
    script:
        "../scripts/merge/stitch_wells.py"


rule stitch_sbs_well:
    input:
        sbs_metadata=ancient(PREPROCESS_OUTPUTS["combine_metadata_sbs"]),
        sbs_stitch_config=MERGE_OUTPUTS["estimate_stitch_sbs"],
    output:
        sbs_stitched_image=MERGE_OUTPUTS_MAPPED["stitch_sbs_image"],
        sbs_stitched_mask=MERGE_OUTPUTS_MAPPED["stitch_sbs_mask"],
        sbs_cell_positions=MERGE_OUTPUTS_MAPPED["stitch_sbs_positions"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        data_type="sbs",
        flipud=config.get("stitch", {}).get("flipud", False),
        fliplr=config.get("stitch", {}).get("fliplr", False),
        rot90=config.get("stitch", {}).get("rot90", 0),
        overlap_percent=config.get("stitch", {}).get("overlap_percent", 0.05),
    resources:
        mem_mb=400000,     # 400GB for maximum safety margin
        cpus_per_task=8,
        runtime=180,       # 3 hours
        partition="20"     # Force high-memory nodes
    script:
        "../scripts/merge/stitch_wells.py"


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

rule well_alignment:
    input:
        phenotype_positions=MERGE_OUTPUTS["stitch_phenotype_positions"][0],
        sbs_positions=MERGE_OUTPUTS["stitch_sbs_positions"][0],
    output:
        scaled_phenotype_positions=MERGE_OUTPUTS["well_alignment"][0],      
        phenotype_triangles=MERGE_OUTPUTS["well_alignment"][1],             
        sbs_triangles=MERGE_OUTPUTS["well_alignment"][2],                   
        alignment_params=MERGE_OUTPUTS["well_alignment"][3],                
        alignment_summary=MERGE_OUTPUTS["well_alignment"][4],
        transformed_phenotype_positions=MERGE_OUTPUTS["well_alignment"][5],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        det_range=config["merge"]["det_range"],
        score_threshold=config["merge"]["score"],
    script:
        "../scripts/merge/well_alignment.py"


rule well_cell_merge:
    input:
        scaled_phenotype_positions=MERGE_OUTPUTS["well_alignment"][0],      
        sbs_positions=MERGE_OUTPUTS["stitch_sbs_positions"][0],
        alignment_params=MERGE_OUTPUTS["well_alignment"][3],
        transformed_phenotype_positions=MERGE_OUTPUTS["well_alignment"][5],               
    output:
        raw_matches=MERGE_OUTPUTS["well_cell_merge"][0],                    
        merged_cells=MERGE_OUTPUTS["well_cell_merge"][1],                   
        merge_summary=MERGE_OUTPUTS["well_cell_merge"][2],                  
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        threshold=config["merge"]["threshold"],
    script:
        "../scripts/merge/well_cell_merge.py"


rule well_merge_deduplicate:
    input:
        raw_matches=MERGE_OUTPUTS["well_cell_merge"][0],                    
        merged_cells=MERGE_OUTPUTS["well_cell_merge"][1],                   
    output:
        deduplicated_cells=MERGE_OUTPUTS["well_merge_deduplicate"][0],      
        deduplication_summary=MERGE_OUTPUTS["well_merge_deduplicate"][1],   
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/merge/well_merge_deduplicate.py"


rule merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["well_merge_deduplicate"][0]  # deduplicated_cells.parquet
            if config.get("merge", {}).get("approach", "tile") == "well"
            else MERGE_OUTPUTS["merge"][0]
        )
    output:
        MERGE_OUTPUTS_MAPPED["merge"],
    params:
        approach=config.get("merge", {}).get("approach", "tile"),
    run:
        import pandas as pd
        from lib.shared.file_utils import validate_dtypes
        
        # Simply copy the chosen approach to the standard merge output
        merge_data = validate_dtypes(pd.read_parquet(input[0]))
        merge_data.to_parquet(output[0])
        
        approach = params.approach
        print(f"Using {approach} merge approach")
        print(f"Merged {len(merge_data)} cells")


rule merge:
    input:
        ancient(PHENOTYPE_OUTPUTS["combine_phenotype_info"]),
        ancient(SBS_OUTPUTS["combine_sbs_info"]),
        MERGE_OUTPUTS["fast_alignment"],
    output:
        MERGE_OUTPUTS_MAPPED["merge"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        threshold=config["merge"]["threshold"],
    script:
        "../scripts/merge/merge.py"



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



rule format_merge:
    input:
        MERGE_OUTPUTS["merge"],
        ancient(SBS_OUTPUTS["combine_cells"]),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1]),
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"],
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
            formatted_data.to_parquet(output[1])  # merge_deduplicated.parquet
            
            # Create dummy stats files
            import pandas as pd
            dummy_stats = pd.DataFrame({"info": ["No additional deduplication - already done in well_merge_deduplicate"]})
            dummy_stats.to_csv(output[0], sep='\t', index=False)  # deduplication_stats.tsv
            dummy_stats.to_csv(output[2], sep='\t', index=False)  # final_sbs_matching_rates.tsv  
            dummy_stats.to_csv(output[3], sep='\t', index=False)  # final_phenotype_matching_rates.tsv
        else:
            # For tile approach, run the actual deduplication script
            shell("python {workflow.basedir}/scripts/merge/deduplicate_merge.py")



rule final_merge:
    input:
        lambda wildcards: (
            MERGE_OUTPUTS["format_merge"][0]  # Use formatted data directly for well approach
            if config.get("merge", {}).get("approach", "tile") == "well" 
            else MERGE_OUTPUTS["deduplicate_merge"][1]  # Use deduplicated data for tile
        ),
        ancient(PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0]),
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"],
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


# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,