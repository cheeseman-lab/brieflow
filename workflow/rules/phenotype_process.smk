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


# Segments cells and nuclei using pre-defined methods
rule segment_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PHENOTYPE_PROCESS_OUTPUTS["apply_ic_field_phenotype"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["segment_phenotype"],
    params:
        dapi_index=config["phenotype_process"]["dapi_index"],
        cyto_index=config["phenotype_process"]["cyto_index"],
        nuclei_diameter=config["phenotype_process"]["nuclei_diameter"],
        cell_diameter=config["phenotype_process"]["cell_diameter"],
        cyto_model=config["phenotype_process"]["cyto_model"],
        return_counts=True,
    script:
        "../scripts/shared/segment_cellpose.py"


# rule for all phenotpye processing steps
rule all_phenotype_process:
    input:
        PHENOTYPE_PROCESS_TARGETS_ALL,
