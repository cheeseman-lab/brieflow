"""File adapters for the experimental nuclear-body workflow and notebook."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread

from lib.phenotype.nuclear_bodies import (
    NuclearObjectSpec,
    derive_mn_bodies,
    measure_body_features,
    segment_nuclear_objects,
)


def identify_nuclear_objects(image_path, nuclei_path, config, output_path, parent_path):
    """Save independent masks and detection diagnostics for configured channels."""
    image, nuclei = imread(image_path), imread(nuclei_path)
    names = config["channel_names"]
    specs = config["objects"]
    if image.ndim != 3 or image.shape[0] != len(names) or len(set(names)) != len(names):
        raise ValueError("Image must be CYX with unique configured channel names")
    if not {"bp1", "rnp"}.issubset(specs) or set(specs) - {"bp1", "rnp", "reporter"}:
        raise ValueError("Require bp1/rnp object specs, with optional reporter")
    arrays, parents = {}, []
    for name, settings in specs.items():
        index = names.index(settings["channel"])
        masks, table, debug = segment_nuclear_objects(
            image[index],
            nuclei,
            NuclearObjectSpec(**settings["params"]),
            max_objects=config.get("max_objects"),
        )
        arrays[name] = masks
        arrays[f"{name}_enhanced"] = debug["enhanced"]
        arrays[f"{name}_threshold"] = debug["threshold"]
        table.insert(0, "object_type", name)
        parents.append(table)
    for path in (output_path, parent_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    pd.concat(parents, ignore_index=True).to_csv(parent_path, sep="\t", index=False)


def derive_nuclear_objects(
    image_path, nuclei_path, base_path, parent_path, config, outputs, metadata
):
    """Write derived masks, relationships, measurements, summaries and provenance."""
    image, nuclei = imread(image_path), imread(nuclei_path)
    with np.load(base_path, allow_pickle=False) as archive:
        masks, pairs, reporter_pairs = derive_mn_bodies(
            archive["bp1"],
            archive["rnp"],
            nuclei,
            connectivity=config.get("contact_connectivity", 2),
            reporter=archive["reporter"] if "reporter" in archive else None,
            reporter_min_overlap=config.get("reporter_min_overlap"),
        )
    objects, summary = measure_body_features(
        image, config["channel_names"], nuclei, masks, metadata
    )
    parents = pd.read_csv(parent_path, sep="\t")
    # Carry nuclear-edge QC from detected objects to their selected subsets.
    source_type = objects.object_type.replace(
        {"mn": "bp1", "reporter_in_mn": "reporter"}
    )
    lookup = parents.set_index(["object_type", "object_id"])[
        "touches_nuclear_boundary"
    ].to_dict()
    objects["touches_nuclear_boundary"] = [
        lookup.get((name, oid), pd.NA)
        for name, oid in zip(source_type, objects.object_id)
    ]
    pairs = pairs.rename(columns={"a_id": "bp1_id", "b_id": "rnp_id"})
    if reporter_pairs.empty and not len(reporter_pairs.columns):
        reporter_pairs = pd.DataFrame(
            columns=[
                "a_id",
                "b_id",
                "parent_nucleus_id",
                "overlap_px",
                "adjacent_pixel_pairs",
                "fraction_a",
                "fraction_b",
                "qualifies",
            ]
        )
    reporter_pairs = reporter_pairs.rename(
        columns={"a_id": "reporter_id", "b_id": "mn_id"}
    )
    for table in (pairs, reporter_pairs):
        for key, value in metadata.items():
            table.insert(0, key, value)
    for path in outputs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs[0], **masks)
    for table, path in zip((pairs, reporter_pairs, objects, summary), outputs[1:5]):
        table.to_csv(path, sep="\t", index=False)
    manifest = {
        "status": "success",
        "validation": "experimental; quantitative validation against annotations pending",
        "image": str(Path(image_path).resolve()),
        "nuclei": str(Path(nuclei_path).resolve()),
        "metadata": metadata,
        "config": config,
        "baseline": next(k for k in masks if k.startswith("nuclei_")),
        "mn_definition": "whole 53BP1 objects contacting RNP within the same nucleus",
        "counts": {k: int(len(np.unique(v[v > 0]))) for k, v in masks.items()},
    }
    Path(outputs[5]).write_text(json.dumps(manifest, indent=2) + "\n")
