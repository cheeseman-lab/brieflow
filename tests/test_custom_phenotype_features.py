"""Tests for user-defined per-cell phenotype features registered from the notebook.

Covers the four properties a registered feature has to hold for a hand-designed
measurement to be comparable against an external pipeline:

1. A named callable over a regionprops-like region becomes a real column in the
   cp_emulator output, alongside the built-in features rather than instead of them.
2. The column is namespaced and carries a hash of the definition, so a reader of
   merge_final.parquet can tell which definition produced it and a silent edit to
   the definition cannot masquerade as the same measurement.
3. The definition survives the notebook -> config.yml -> workflow hop, which is a
   YAML round trip into a separate process, and the rebuilt function measures the
   same values as the original. A definition that would not survive it, because it
   closes over a notebook name, is refused in the notebook instead.
4. A feature that raises, or that returns something other than a number, fails with
   the feature name instead of writing NaN into a column nobody audits.
5. The feature is measured on the compartment it declares, and the column says which
   one, so a nuclear measurement is not silently taken over the cell mask.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.cell_data_utils import (  # noqa: E402
    channel_combo_subset,
    get_feature_table_cols,
    split_cell_data,
)
from lib.phenotype.constants import DEFAULT_METADATA_COLS  # noqa: E402
from lib.phenotype.custom_features import (  # noqa: E402
    custom_feature_column,
    load_custom_features,
    register_custom_features,
)
from lib.phenotype.extract_phenotype_cp_emulator import (  # noqa: E402
    extract_phenotype_cp_emulator,
)
from lib.shared.file_utils import validate_dtypes  # noqa: E402

CHANNEL_NAMES = ["dapi", "v5"]


def puncta_count(region):
    return int((region.intensity_image[..., 1] > 400).sum())


def _uses_a_notebook_import(region):
    return int(np.count_nonzero(region.intensity_image[..., 1] > 400))


def _synthetic_tile():
    rng = np.random.default_rng(0)
    height = width = 40
    nuclei = np.zeros((height, width), dtype=np.uint16)
    cells = np.zeros((height, width), dtype=np.uint16)
    for label, (i, j) in enumerate([(8, 8), (8, 28), (28, 8)], start=1):
        cells[i - 6 : i + 6, j - 6 : j + 6] = label
        nuclei[i - 3 : i + 3, j - 3 : j + 3] = label
    data = rng.integers(0, 500, size=(2, height, width)).astype(np.uint16)
    cytoplasms = np.where(nuclei > 0, 0, cells)

    return data, nuclei, cells, cytoplasms


def _extract(custom_features, segment_cells=True, segment_cytoplasms=False):
    data, nuclei, cells, cytoplasms = _synthetic_tile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return extract_phenotype_cp_emulator(
            data,
            nuclei,
            cells if segment_cells else None,
            wildcards={"well": "A1", "tile": 1},
            cytoplasms=cytoplasms if segment_cytoplasms else None,
            channel_names=CHANNEL_NAMES,
            custom_features=custom_features,
        )


# --- Property 1: a registered callable becomes a column ---------------------------


def test_registered_feature_becomes_a_column():
    definitions = register_custom_features([puncta_count])
    column = definitions[0]["column"]

    result = _extract(load_custom_features(definitions))

    assert column in result.columns
    assert result[column].tolist() == [9, 6, 10]


def test_built_in_features_are_unchanged_by_registration():
    baseline = _extract(None)
    with_custom = _extract(
        load_custom_features(register_custom_features([puncta_count]))
    )

    shared = [col for col in baseline.columns if col in with_custom.columns]
    assert shared == baseline.columns.tolist()
    pd.testing.assert_frame_equal(with_custom[shared], baseline)


def test_no_registration_leaves_the_output_free_of_custom_columns():
    result = _extract(load_custom_features(None))

    assert [col for col in result.columns if "custom" in col] == []


# --- Property 2: namespaced, and the definition travels with the column -----------


def test_column_is_namespaced_and_carries_the_definition_hash():
    definitions = register_custom_features([puncta_count])
    definition = definitions[0]

    assert definition["column"] == custom_feature_column(
        "puncta_count", definition["hash"], definition["compartment"]
    )
    assert definition["column"].startswith("nucleus_custom_puncta_count_")
    assert definition["source"].lstrip().startswith("def puncta_count(region):")


def test_editing_the_definition_changes_the_column():
    def drifted(region):
        return int((region.intensity_image[..., 1] > 401).sum())

    drifted.__name__ = "puncta_count"

    original = register_custom_features([puncta_count])[0]
    edited = register_custom_features([drifted])[0]

    assert original["name"] == edited["name"]
    assert original["column"] != edited["column"]


def test_a_column_colliding_with_a_built_in_feature_raises():
    with pytest.raises(ValueError, match="collide"):
        _extract({"nucleus": {"nucleus_area": lambda region: 1}})


def test_registering_the_same_name_twice_raises():
    with pytest.raises(ValueError, match="registered more than once"):
        register_custom_features([puncta_count, puncta_count])


def test_registering_a_lambda_raises():
    with pytest.raises(ValueError, match="named functions"):
        register_custom_features([lambda region: 1])


# --- Property 3: the definition survives the config hop ---------------------------


def test_definitions_survive_a_yaml_round_trip(tmp_path):
    config_fp = tmp_path / "config.yml"
    definitions = register_custom_features([puncta_count, (puncta_count, "cell")])
    config_fp.write_text(yaml.dump({"phenotype": {"custom_features": definitions}}))

    loaded = yaml.safe_load(config_fp.read_text())["phenotype"]["custom_features"]
    assert loaded == definitions

    result = _extract(load_custom_features(loaded))
    assert result[definitions[0]["column"]].tolist() == [9, 6, 10]
    assert result[definitions[1]["column"]].tolist() == [34, 28, 32]


def test_registering_a_feature_that_closes_over_a_notebook_name_raises():
    threshold = 400

    def uses_a_notebook_name(region):
        return int((region.intensity_image[..., 1] > threshold).sum())

    with pytest.raises(ValueError, match="uses names it does not define"):
        register_custom_features([uses_a_notebook_name])


def test_registering_a_feature_using_a_notebook_import_raises():
    with pytest.raises(ValueError, match=r"does not define: \['np'\]"):
        register_custom_features([_uses_a_notebook_import])


# --- Property 4: new columns are features downstream, and failures are loud -------


def test_custom_column_is_treated_as_a_feature_not_metadata():
    definitions = register_custom_features([puncta_count])
    column = definitions[0]["column"]
    # validate_dtypes is what the merge steps run before aggregate splits the frame
    result = validate_dtypes(_extract(load_custom_features(definitions)))

    metadata, features = split_cell_data(result, DEFAULT_METADATA_COLS)

    assert column not in DEFAULT_METADATA_COLS
    assert column not in metadata.columns
    assert column in features.columns
    assert pd.api.types.is_numeric_dtype(features[column])


def test_a_feature_that_raises_names_itself_instead_of_emitting_nan():
    def brittle_count(region):
        if region.label == 2:
            raise ZeroDivisionError("division by zero")
        return 1

    definitions = register_custom_features([brittle_count])

    with pytest.raises(ValueError, match="brittle_count.*label 2"):
        _extract(load_custom_features(definitions))


def test_a_feature_returning_a_non_number_names_itself():
    def per_channel_count(region):
        return (region.intensity_image > 400).sum(axis=(0, 1))

    definitions = register_custom_features([per_channel_count])

    with pytest.raises(ValueError, match="per_channel_count.*expected a single number"):
        _extract(load_custom_features(definitions))


def test_a_boolean_feature_is_accepted():
    def is_bright(region):
        return bool(region.intensity_image[..., 1].mean() > 250)

    definitions = register_custom_features([is_bright])
    result = _extract(load_custom_features(definitions))

    assert result[definitions[0]["column"]].dtype == bool


# --- Property 5: the feature is measured on the compartment it declares -----------


def test_a_bare_feature_is_measured_on_the_nucleus():
    definitions = register_custom_features([puncta_count])
    column = definitions[0]["column"]

    result = _extract(load_custom_features(definitions))

    assert definitions[0]["compartment"] == "nucleus"
    assert column.startswith("nucleus_custom_")
    assert result[column].tolist() == [9, 6, 10]


def test_declaring_the_cell_is_measured_on_the_cell_mask():
    definitions = register_custom_features([(puncta_count, "cell")])
    column = definitions[0]["column"]

    result = _extract(load_custom_features(definitions))

    assert column.startswith("cell_custom_")
    assert result[column].tolist() == [34, 28, 32]


def test_declaring_the_cytoplasm_is_measured_on_the_cytoplasm_mask():
    definitions = register_custom_features([(puncta_count, "cytoplasm")])
    column = definitions[0]["column"]

    result = _extract(load_custom_features(definitions), segment_cytoplasms=True)

    assert column.startswith("cytoplasm_custom_")
    assert result[column].tolist() == [25, 22, 22]


def test_one_definition_measured_on_two_compartments_gives_two_columns():
    nucleus, cell = register_custom_features(
        [(puncta_count, "nucleus"), (puncta_count, "cell")]
    )

    result = _extract(load_custom_features([nucleus, cell]))

    assert nucleus["hash"] == cell["hash"]
    assert nucleus["column"] != cell["column"]
    assert result[nucleus["column"]].tolist() == [9, 6, 10]
    assert result[cell["column"]].tolist() == [34, 28, 32]


def test_a_column_sorts_with_the_compartment_it_measures():
    definitions = register_custom_features(
        [(puncta_count, "nucleus"), (puncta_count, "cell")]
    )

    columns = _extract(load_custom_features(definitions)).columns.tolist()

    assert columns.index(definitions[0]["column"]) < columns.index("cell_area")
    assert columns.index(definitions[1]["column"]) > columns.index("nucleus_area")


def test_declaring_the_cytoplasm_without_cytoplasm_segmentation_raises():
    definitions = register_custom_features([(puncta_count, "cytoplasm")])

    with pytest.raises(
        ValueError, match="cytoplasm compartment, which is not segmented"
    ):
        _extract(load_custom_features(definitions))


def test_declaring_the_cell_without_cell_segmentation_raises():
    definitions = register_custom_features([(puncta_count, "cell")])

    with pytest.raises(ValueError, match="cell compartment, which is not segmented"):
        _extract(load_custom_features(definitions), segment_cells=False)


def test_declaring_an_unknown_compartment_raises():
    with pytest.raises(ValueError, match="unknown compartment 'nuclei'"):
        register_custom_features([(puncta_count, "nuclei")])


def test_registering_the_same_name_on_two_compartments_is_allowed():
    definitions = register_custom_features(
        [(puncta_count, "nucleus"), (puncta_count, "cell")]
    )

    assert [definition["compartment"] for definition in definitions] == [
        "nucleus",
        "cell",
    ]

    with pytest.raises(ValueError, match="registered more than once"):
        register_custom_features([(puncta_count, "cell"), (puncta_count, "cell")])


def test_the_compartment_prefix_drives_downstream_column_selection():
    definitions = register_custom_features(
        [(puncta_count, "nucleus"), (puncta_count, "cytoplasm")]
    )
    result = _extract(load_custom_features(definitions), segment_cytoplasms=True)

    # get_feature_table_cols keeps the nucleus and cell compartments only
    selected = get_feature_table_cols(result.columns.tolist(), extra_tags=["custom"])
    assert definitions[0]["column"] in selected
    assert definitions[1]["column"] not in selected

    kept = channel_combo_subset(result, ["dapi"], CHANNEL_NAMES)
    assert definitions[0]["column"] in kept.columns
