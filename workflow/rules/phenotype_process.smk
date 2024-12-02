from lib.shared.target_utils import output_to_input


# Apply illumination correction field
rule apply_ic_field_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PREPROCESS_OUTPUTS["convert_phenotype"],
        PREPROCESS_OUTPUTS["calculate_ic_phenotype"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["apply_ic_field_phenotype"],
    script:
        "../scripts/phenotype_process/apply_ic_field_phenotype.py"


# rule for all phenotpye processing steps
rule all_phenotype_process:
    input:
        PHENOTYPE_PROCESS_TARGETS_ALL,
