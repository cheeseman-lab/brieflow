from lib.shared.target_utils import output_to_input


# Complete fast alignment process
rule fast_alignment:
    conda:
        "../envs/merge_process.yml"
    # TODO: remove threads after testing
    threads: 32
    # TODO: use target inputs/outputs
    input:
        # metadata file with image locations
        #PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/denali_data/combined_metadata__phenotype.hdf",
        # lambda wildcards: output_to_input(
        #     PREPROCESS_OUTPUTS["combine_metadata_sbs"],
        #     {"cycle": config["merge_process"]["sbs_metdata_cycle"]},
        #     wildcards,
        # ),
        "/lab/barcheese01/rkern/brieflow/example_analysis/denali_data/combined_metadata__sbs.hdf",
        # phenotype and sbs info files with cell locations
        #PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_info"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/denali_data/phenotype_info.hdf5",
        #SBS_PROCESS_OUTPUTS["combine_sbs_info"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/denali_data/sbs_info.hdf5",
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["fast_alignment"],
    params:
        det_range=config["merge_process"]["det_range"],
        score=config["merge_process"]["score"],
        initial_sites=config["merge_process"]["initial_sites"],
    script:
        "../scripts/merge_process/fast_alignment.py"
