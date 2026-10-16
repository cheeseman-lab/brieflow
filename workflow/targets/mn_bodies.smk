"""Opt-in accepted MN outputs; imported before phenotype target expansion."""
from lib.phenotype.mn_pipeline import validate_settings, implementation_fingerprint

MN_SETTINGS = validate_settings(config["phenotype"]["mn_bodies"], config["phenotype"])
# A boundary-threshold edit should start at refinement, not repeat registration
# or focus detection. The full settings are still validated above.
MN_DETECTION_SETTINGS = {key: value for key, value in MN_SETTINGS.items() if key != "bp1_boundary"}
MN_IMPLEMENTATION = implementation_fingerprint()
MN_OUTPUT_NAMES = {
    "register_mn_rounds": [("aligned", "npz"), ("registration", "json")],
    "export_mn_registered_image": [("registered_image", "tiff")],
    "segment_mn_components": [("components", "npz"), ("objects", "tsv"),
                              ("reporters", "tsv"), ("nucleoli", "tsv"), ("domain", "json")],
    "refine_mn_components": [("refined_components", "npz"), ("bp1_boundaries", "tsv"),
                             ("boundary_provenance", "json")],
    "define_mn_bodies": [("derived", "npz"), ("decisions", "tsv"), ("pairs", "tsv"),
                         ("membership", "tsv"), ("contacts_a", "tsv"), ("contacts_c", "tsv")],
    "measure_mn_bodies": [("features", "tsv"), ("parents", "tsv"), ("summary", "tsv"), ("provenance", "json")],
    "merge_mn_phenotype": [("phenotype_with_mn", "tsv")],
}
MN_FILES = {}
for step, entries in MN_OUTPUT_NAMES.items():
    MN_FILES[step] = {
        name: PHENOTYPE_FP / "mn_bodies" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "mn_" + name, extension
        ) for name, extension in entries
    }
    PHENOTYPE_OUTPUTS[step] = list(MN_FILES[step].values())
    PHENOTYPE_OUTPUT_MAPPINGS[step] = None

# Object-level features remain separate from the one-row-per-nucleus table.
PHENOTYPE_OUTPUTS["combine_mn_bodies"] = [
    PHENOTYPE_FP / "mn_bodies" / get_filename({"plate": "{plate}", "well": "{well}"}, "mn_features", "parquet")
]
PHENOTYPE_OUTPUT_MAPPINGS["combine_mn_bodies"] = None
