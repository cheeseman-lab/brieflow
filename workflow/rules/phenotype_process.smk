from lib.shared.target_utils import output_to_input


# Apply illumination correction field
rule apply_ic_field:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PREPROCESS_OUTPUTS["convert_phenotype"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["apply_ic_field"],
    params:
        segmentation_cycle_index=SBS_CYCLES[
            config["sbs_process"]["segmentation_cycle_index"]
        ],
    script:
        "../scripts/sbs_process/apply_ic_field.py"
