# Standard Cellpose-SAM nuclei are an upstream dependency. Each stage writes
# reusable masks/tables; boundary-only setting edits restart at refinement.
rule register_mn_rounds:
    input:
        image=PHENOTYPE_OUTPUTS["align_phenotype"][0],
        source=PREPROCESS_OUTPUTS["convert_phenotype"][0],
    output:
        **MN_FILES["register_mn_rounds"],
    params:
        stage="register", settings=MN_DETECTION_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule export_mn_registered_image:
    input:
        image=PHENOTYPE_OUTPUTS["align_phenotype"][0],
        aligned=MN_FILES["register_mn_rounds"]["aligned"],
        registration=MN_FILES["register_mn_rounds"]["registration"],
    output:
        **MN_FILES["export_mn_registered_image"],
    params:
        stage="export", settings=MN_DETECTION_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule segment_mn_components:
    input:
        aligned=MN_FILES["register_mn_rounds"]["aligned"],
        nuclei=PHENOTYPE_OUTPUTS["segment_phenotype"][0],
    output:
        **MN_FILES["segment_mn_components"],
    params:
        stage="segment", settings=MN_DETECTION_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule refine_mn_components:
    input:
        aligned=MN_FILES["register_mn_rounds"]["aligned"],
        components=MN_FILES["segment_mn_components"]["components"],
    output:
        **MN_FILES["refine_mn_components"],
    params:
        stage="refine", settings=MN_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule define_mn_bodies:
    input:
        aligned=MN_FILES["register_mn_rounds"]["aligned"],
        components=MN_FILES["refine_mn_components"]["refined_components"],
    output:
        **MN_FILES["define_mn_bodies"],
    params:
        stage="define", settings=MN_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule measure_mn_bodies:
    input:
        aligned=MN_FILES["register_mn_rounds"]["aligned"],
        components=MN_FILES["refine_mn_components"]["refined_components"],
        derived=MN_FILES["define_mn_bodies"]["derived"],
        membership=MN_FILES["define_mn_bodies"]["membership"],
    output:
        **MN_FILES["measure_mn_bodies"],
    params:
        stage="measure", settings=MN_SETTINGS, implementation=MN_IMPLEMENTATION,
    resources:
        mem_mb=4000,
    script:
        "../scripts/phenotype/run_mn_stage.py"

rule combine_mn_bodies:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["measure_mn_bodies"][:1], wildcards=wildcards,
            expansion_values=["tile"], metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS["combine_mn_bodies"],
    script:
        "../scripts/shared/combine_dfs.py"

rule merge_mn_phenotype:
    input:
        phenotype=PHENOTYPE_OUTPUTS["extract_phenotype_cp"][0],
        summary=MN_FILES["measure_mn_bodies"]["summary"],
    output:
        **MN_FILES["merge_mn_phenotype"],
    script:
        "../scripts/phenotype/merge_mn_phenotype.py"
