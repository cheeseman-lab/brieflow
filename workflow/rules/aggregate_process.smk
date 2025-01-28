from lib.shared.target_utils import output_to_input


# Clean and transform merged data
rule clean_and_transform:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # final merge data
        MERGE_PROCESS_OUTPUTS["final_merge"],
    script:
        "../scripts/aggregate_process/clean_and_transform.py"
