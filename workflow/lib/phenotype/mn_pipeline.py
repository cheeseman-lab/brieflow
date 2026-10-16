"""Accepted multiround MN workflow, independent of pilot folders and notebooks.

The public image stages are registration, component segmentation, boundary
refinement, MN definition, and marker measurement. Arrays use primary-round
YX coordinates throughout.
Parent labels come from Brieflow's normal Cellpose-SAM segmentation rule.

Two deliberately different reference masks are retained:
* rescue_baseline: nucleus minus BP1, both RNP rounds, and nucleolar proxies;
* measurement_baseline: rescue_baseline minus accepted reporter foci.
The first defines rescue before reporter association is considered. Optional
boundary refinement uses a third, frozen original-mask background; accepted
reporter exclusions can affect that earlier background estimate.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tifffile import imread, imwrite

from lib.phenotype.bp1_boundary_refinement import (
    BOUNDARY_COLUMNS,
    BoundarySpec,
    refine_boundaries,
)

from lib.phenotype.chromatin_quantification import (
    BrightArtifactSpec,
    bright_if_artifacts,
    label_intensity_statistics,
    safe_ratio,
)
from lib.phenotype.mn_consensus import derive_consensus_mn
from lib.phenotype.mn_reporter import classify_mn_reporter, filter_reporter_candidates
from lib.phenotype.nuclear_bodies import (
    NuclearObjectSpec,
    derive_mn_bodies,
    segment_nuclear_objects,
    validate_labels,
)
from lib.phenotype.nuclear_body_rounds import (
    NucleolusSpec,
    align_round,
    common_nuclei,
    segment_nucleoli,
)
from lib.phenotype.reporter_polii import polii_level_balance_scores

MARKERS = [
    "BP1",
    "H2AK119ub",
    "H3K27ac",
    "H3K9me2",
    "PolII_Ser5Ph",
    "PolII_Ser2Ph",
    "Nascent",
]
ROLE_ROUND = dict(
    DAPI="primary",
    BP1="primary",
    RNP_primary="primary",
    H2AK119ub="primary",
    PolII_Ser5Ph="round2",
    H3K27ac="round2",
    H3K9me2="round3",
    FLAG="round3",
    Nascent="round4",
    PolII_Ser2Ph="round4",
    RNP_fifth="round5",
)


def validate_settings(settings, phenotype=None):
    """Fail early on incompatible channel order or double registration."""
    names = settings["channel_names"]
    if len(names) != len(set(names)):
        raise ValueError("MN channel names must be unique")
    for round_name in ["primary", "round2", "round3", "round4", "round5"]:
        entry = settings["rounds"][round_name]
        for channel in [entry["dapi"]] + list(entry["markers"].values()):
            if channel not in names:
                raise ValueError(f"Unknown MN channel: {channel}")
        expected = {
            r for r, rnd in ROLE_ROUND.items() if rnd == round_name and r != "DAPI"
        }
        if set(entry["markers"]) != expected:
            raise ValueError(f"{round_name} requires marker roles {sorted(expected)}")
    for role in ["bp1", "rnp_primary", "rnp_fifth", "reporter"]:
        NuclearObjectSpec(**settings["objects"][role])
    NucleolusSpec(**settings["nucleoli"])
    boundary_settings(settings)
    if phenotype is not None:
        if phenotype["channel_names"] != names:
            raise ValueError("MN channel order must match phenotype.channel_names")
        if phenotype.get("segment_cells", True):
            raise ValueError("Accepted MN pipeline requires segment_cells: false")
        if phenotype.get("align", False) or phenotype.get("custom_channel_offsets"):
            raise ValueError(
                "MN pipeline owns cross-round registration; disable upstream alignment/offsets"
            )
        if (
            phenotype["channel_names"][phenotype["dapi_index"]]
            != settings["rounds"]["primary"]["dapi"]
        ):
            raise ValueError("Cellpose nuclei must use primary-round DAPI")
        if phenotype.get("second_obj_detection", False) or phenotype.get(
            "nuclear_bodies", {}
        ).get("enabled", False):
            raise ValueError("Use one secondary-object extension: mn_bodies")
    return settings


def boundary_settings(settings):
    """Validate optional refinement settings; omission preserves old behavior.

    The pilot enables relative_contrast with threshold 0.30 explicitly. Keeping
    the extension disabled by default avoids changing other screens' masks.
    """
    config = dict(settings.get("bp1_boundary", {}))
    enabled = config.pop("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("bp1_boundary.enabled must be true or false")
    if enabled and "threshold" not in config:
        raise ValueError("Set an explicit bp1_boundary.threshold when enabled")
    return enabled, BoundarySpec(**config)


def shift_intensity(image, shift):
    """Apply the accepted linear translation without spline overshoot."""
    return ndi.shift(
        np.asarray(image, np.float32),
        shift,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def shift_affected_pixels(mask, shift):
    """Keep every destination pixel influenced by a saturated/artifact source."""
    return shift_intensity(np.asarray(mask, np.float32), shift) > 0


def register_rounds(stack, settings, raw_stack=None):
    """Estimate each round's DAPI translation once and apply it to its markers.

    Keep validity bands and source-saturation masks. Interpolation must not hide
    missing data or saturated pixels. Artifact detection precedes resampling.
    """
    validate_settings(settings)
    stack = np.asarray(stack)
    names = settings["channel_names"]
    if (
        stack.ndim != 3
        or stack.shape[0] != len(names)
        or not np.isfinite(stack).all()
        or (stack < 0).any()
    ):
        raise ValueError("Expected finite nonnegative CYX input matching channel_names")
    native = {name: stack[i] for i, name in enumerate(names)}
    raw_stack = stack if raw_stack is None else np.asarray(raw_stack)
    if raw_stack.shape != stack.shape or not np.isfinite(raw_stack).all():
        raise ValueError("Raw source stack must match the CYX intensity stack")
    raw_native = {name: raw_stack[i] for i, name in enumerate(names)}
    reference = native[settings["rounds"]["primary"]["dapi"]]
    ceiling = settings["saturation_value"]
    arrays = {"DAPI": reference}
    arrays["DAPI_saturated"] = (
        raw_native[settings["rounds"]["primary"]["dapi"]] >= ceiling
    )
    records = {}
    for round_name, entry in settings["rounds"].items():
        if round_name == "primary":
            support = np.ones(reference.shape, bool)
            shift = [0.0, 0.0]
            stats = dict(shift_yx_to_reference=shift, reference_round=True)
        else:
            first = next(iter(entry["markers"].values()))
            _, _, support, stats = align_round(
                reference,
                native[entry["dapi"]],
                native[first],
                max_shift=settings["registration"]["max_shift_px"],
            )
            shift = stats["shift_yx_to_reference"]
            residual = stats["regional_residual_max_px"]
            if (
                stats["dapi_correlation_after"]
                < settings["registration"]["minimum_dapi_correlation"]
                or residual is None
                or residual > settings["registration"]["maximum_regional_residual_px"]
            ):
                raise ValueError(f"Registration needs review for {round_name}: {stats}")
        arrays[f"{round_name}_valid"] = support
        records[round_name] = stats
        for role, channel in entry["markers"].items():
            image = native[channel]
            arrays[role] = (
                image if round_name == "primary" else shift_intensity(image, shift)
            )
            arrays[f"{role}_saturated"] = shift_affected_pixels(
                raw_native[channel] >= ceiling, shift
            )
            if role == "H3K9me2":
                artifacts, _ = bright_if_artifacts(
                    raw_native[channel],
                    BrightArtifactSpec(**settings["H3K9me2_artifacts"]),
                )
                arrays["H3K9me2_artifact"] = shift_affected_pixels(artifacts, shift)
    return arrays, records


def segment_components(images, nuclei, settings):
    """Segment parent-contained foci, preserving accepted stage order and IDs.

    Primary foci are labeled over all original nuclei before domain restriction.
    Fifth-round RNP and nucleolar proxies use the round-2/round-5 common domain.
    Reporter candidates use the further round-3 common domain. Round 4 affects
    only transcription measurement eligibility, never object identity.
    """
    nuclei = validate_labels(nuclei)
    common, excluded = common_nuclei(
        nuclei, images["round2_valid"], images["round5_valid"]
    )
    observed, reporter_missing = common_nuclei(common, images["round3_valid"])
    masks = {
        "nuclei_primary": nuclei,
        "nuclei_definition": common,
        "nuclei_observed": observed,
    }
    diagnostics = []
    for role, image_role, domain in [
        ("bp1", "BP1", nuclei),
        ("rnp_primary", "RNP_primary", nuclei),
        ("rnp_fifth", "RNP_fifth", common),
        ("reporter", "FLAG", observed),
    ]:
        labels, parents, debug = segment_nuclear_objects(
            images[image_role], domain, NuclearObjectSpec(**settings["objects"][role])
        )
        parents.insert(0, "object_type", role)
        diagnostics.append(parents)
        # Primary IDs remain stable even if an edge nucleus is not observed later.
        if role in ["bp1", "rnp_primary"]:
            labels = np.where(common > 0, labels, 0)
        if role == "reporter":
            masks["reporter_candidates"] = labels
            labels, reporter_table = filter_reporter_candidates(
                images["FLAG"],
                labels,
                observed,
                settings["reporter"]["minimum_mean_enrichment"],
                settings["reporter"]["minimum_peak_enrichment"],
            )
        masks[role] = labels
        del debug
    masks["nucleoli"], nucleolus_table, _ = segment_nucleoli(
        images["PolII_Ser5Ph"], common, NucleolusSpec(**settings["nucleoli"])
    )
    domain_qc = dict(
        excluded_definition_parent_ids=excluded.tolist(),
        excluded_reporter_parent_ids=reporter_missing.tolist(),
    )
    return (
        masks,
        pd.concat(diagnostics, ignore_index=True),
        reporter_table,
        nucleolus_table,
        domain_qc,
    )


def refine_components(images, masks, settings):
    """Refine BP1 once, after all exclusion masks are available.

    Args:
        images: Registered raw-intensity arrays; only BP1 sets the threshold.
        masks: Unrefined component masks from segment_components.
        settings: MN settings with an optional bp1_boundary configuration.

    Returns:
        Component masks with refined bp1 and preserved bp1_untrimmed, an audit
        table, and a manifest describing the background and fallback counts.

    Background is estimated from the ORIGINAL depleted parent, excluding all
    original BP1, both RNP masks, nucleolar proxies, and accepted reporters.
    This reproduces the reviewed threshold sweep and prevents a circular
    threshold update. Definition/measurement baselines are subsequently rebuilt
    from refined masks. Reporter overlap labels do not enter the boundary rule,
    although the reporter exclusion mask can affect background estimation.
    """
    enabled, spec = boundary_settings(settings)
    if "bp1_untrimmed" in masks:
        raise ValueError(
            "Refinement expects original components, not an already refined file"
        )
    refined = dict(masks)
    refined["bp1_untrimmed"] = masks["bp1"].copy()
    if enabled:
        excluded = np.zeros(masks["bp1"].shape, bool)
        for name in ["bp1", "rnp_primary", "rnp_fifth", "nucleoli", "reporter"]:
            excluded |= masks[name] > 0
        # Use the definition domain so reporter-unobserved nuclei remain distinct
        # from observed reporter-negative nuclei in the downstream outputs.
        baseline = np.where(excluded, 0, masks["nuclei_definition"])
        refined["bp1"], audit = refine_boundaries(
            images["BP1"], masks["bp1"], masks["nuclei_definition"], baseline, spec
        )
    else:
        audit = pd.DataFrame(columns=BOUNDARY_COLUMNS)
    info = dict(
        enabled=enabled,
        settings=settings.get("bp1_boundary", {}),
        background="Original definition nucleus minus original BP1, both RNP, nucleolar and reporter masks",
        background_unavailable_count=int(
            (audit.refinement_status == "background_unavailable").sum()
        ),
        n_original_objects=int(len(np.unique(masks["bp1"][masks["bp1"] > 0]))),
        n_refined_objects=int(len(np.unique(refined["bp1"][refined["bp1"] > 0]))),
        original_labels_preserved=True,
    )
    return refined, audit, info


def define_mn(images, masks, settings):
    """Consensus plus bright/large rescue, then strict reporter-area association.

    Nucleolar proxies only modify the baseline; they do not veto an MN body.
    MN masks use the supplied BP1 footprints (refined when enabled) and original
    IDs, including rescue. The rescue area test therefore uses the refined area.
    """
    nuclei = masks["nuclei_definition"]
    bp1 = masks["bp1"]
    a, contacts_a, _ = derive_mn_bodies(
        bp1, masks["rnp_primary"], nuclei, connectivity=2
    )
    c, contacts_c, _ = derive_mn_bodies(bp1, masks["rnp_fifth"], nuclei, connectivity=2)
    removed = (
        (bp1 > 0)
        | (masks["rnp_primary"] > 0)
        | (masks["rnp_fifth"] > 0)
        | (masks["nucleoli"] > 0)
    )
    rescue_baseline = np.where(removed, 0, nuclei)
    selected, decisions = derive_consensus_mn(
        bp1,
        a["mn"],
        c["mn"],
        nuclei,
        images["BP1"],
        rescue_baseline,
        settings["consensus"],
    )
    observed = masks["nuclei_observed"]
    mn = np.where(observed > 0, selected["optimized"], 0)
    partitions, pairs, membership = classify_mn_reporter(
        mn,
        masks["reporter"],
        observed,
        minimum_reporter_fraction=settings["reporter"]["minimum_overlap_fraction"],
        strict_fraction=True,
    )
    baseline = np.where((observed > 0) & (masks["reporter"] == 0), rescue_baseline, 0)
    derived = dict(
        mn=mn,
        mn_definition=selected["optimized"],
        mn_consensus=selected["consensus"],
        mn_rescued=selected["rescued"],
        rescue_baseline=rescue_baseline,
        measurement_baseline=baseline,
        H3K9me2_baseline=np.where(images["H3K9me2_artifact"], 0, baseline),
        mn_unavailable_reporter=np.where(observed == 0, selected["optimized"], 0),
        **partitions,
    )
    # The pre-reporter rescue enrichment is explicitly distinguished from the
    # marker enrichment measured with the final depleted parent below.
    decisions = decisions.rename(columns={"BP1_enrichment": "BP1_rescue_enrichment"})
    return derived, decisions, pairs, membership, contacts_a, contacts_c


def count_mask_pixels(labels, binary, ids):
    """Count flagged pixels for a possibly sparse list of object labels."""
    return np.bincount(labels[binary].ravel(), minlength=int(labels.max()) + 1)[ids]


def measure_mn(images, components, masks, membership, settings):
    """Vectorized full-object and depleted-parent statistics with per-marker QC.

    Missing registration support, saturation, tiny baselines and K9 debris remain
    flags. Invalid measurements never become zero or a reporter-negative call.
    Raw mean/sum/CV/evenness are retained; canonical enrichment uses mean ratios.
    """
    data = membership.copy()
    ids = data.mn_id.to_numpy(int)
    parents = data.parent_nucleus_id.to_numpy(int)
    nuclei = components["nuclei_observed"]
    nuclear_area = np.bincount(nuclei.ravel())
    parent_ids = np.unique(nuclei[nuclei > 0])
    parent_rows = []
    added = {}
    for marker in MARKERS:
        baseline = (
            masks["H3K9me2_baseline"]
            if marker == "H3K9me2"
            else masks["measurement_baseline"]
        )
        body = label_intensity_statistics(masks["mn"], images[marker])
        base = label_intensity_statistics(baseline, images[marker])
        # Keep indexing valid even when the largest nucleus has no baseline left.
        for key in base:
            base[key] = np.pad(
                base[key].astype(float),
                (0, max(0, int(nuclei.max()) + 1 - len(base[key]))),
                constant_values=np.nan,
            )
        body_sat = count_mask_pixels(masks["mn"], images[f"{marker}_saturated"], ids)
        base_sat_all = np.bincount(
            baseline[images[f"{marker}_saturated"]].ravel(),
            minlength=int(nuclei.max()) + 1,
        )
        missing = np.unique(nuclei[~images[f"{ROLE_ROUND[marker]}_valid"]])
        available = ~np.isin(parents, missing[missing > 0])
        ratio = safe_ratio(body["mean"][ids], base["mean"][parents])
        minimum = settings["measurement"]
        good = (
            (base["area"][parents] >= minimum["minimum_baseline_pixels"])
            & (
                safe_ratio(base["area"][parents], nuclear_area[parents])
                >= minimum["minimum_baseline_fraction"]
            )
            & (body_sat == 0)
            & (base_sat_all[parents] == 0)
            & available
            & np.isfinite(ratio)
            & (ratio > 0)
        )
        if marker == "H3K9me2":
            artifacts = count_mask_pixels(masks["mn"], images["H3K9me2_artifact"], ids)
            added["H3K9me2_artifact_pixels"] = artifacts
            good &= artifacts == 0
        for key in body:
            added[f"{marker}_{key}"] = body[key][ids]
            added[f"{marker}_baseline_{key}"] = base[key][parents]
        added[f"{marker}_enrichment"] = ratio
        added[f"{marker}_integrated_nuclear"] = safe_ratio(
            body["sum"][ids], base["sum"][parents]
        )
        added[f"{marker}_qc_clean"] = good
        added[f"{marker}_round_available"] = available
        added[f"{marker}_body_saturated_pixels"] = body_sat
        added[f"{marker}_baseline_saturated_pixels"] = base_sat_all[parents]
        parent_rows.append(
            pd.DataFrame(
                dict(
                    parent_nucleus_id=parent_ids,
                    marker=marker,
                    saturated_pixels=base_sat_all[parent_ids],
                    round_available=~np.isin(parent_ids, missing),
                    **{k: values[parent_ids] for k, values in base.items()},
                )
            )
        )
    data = pd.concat([data, pd.DataFrame(added, index=data.index)], axis=1)
    scores = polii_level_balance_scores(
        data.PolII_Ser5Ph_enrichment, data.PolII_Ser2Ph_enrichment
    )
    data["PolII_level"] = scores["PolII_mean_enrichment"]
    data["PolII_balance"] = scores["Ser5_Ser2_log2_ratio"]
    data["PolII_eligible"] = data.PolII_Ser5Ph_qc_clean & data.PolII_Ser2Ph_qc_clean
    data["all_markers_eligible"] = data[[f"{m}_qc_clean" for m in MARKERS]].all(axis=1)
    return data, pd.concat(parent_rows, ignore_index=True)


def parent_summary(data, components):
    """One row per original nucleus, safe to join to Brieflow's ``label`` key.

    Unobserved parents have missing counts, not zeros. Observed parents with no
    MN bodies have zero counts and undefined marker summaries. Aggregates use
    eligible objects separately for each marker and each reporter group.
    """
    ids = np.unique(components["nuclei_primary"])
    ids = ids[ids > 0]
    observed = np.unique(components["nuclei_observed"])
    summary = pd.DataFrame(dict(label=ids, mn_domain_available=np.isin(ids, observed)))
    for group, selected in [
        ("all", data),
        ("with_reporter", data[data.with_reporter]),
        ("without_reporter", data[~data.with_reporter]),
    ]:
        summary[f"mn_{group}_count"] = summary.label.map(
            selected.groupby("parent_nucleus_id").size()
        ).fillna(0)
        summary.loc[~summary.mn_domain_available, f"mn_{group}_count"] = np.nan
        for marker in MARKERS + ["PolII_level", "PolII_balance"]:
            derived = marker.startswith("PolII_") and marker in [
                "PolII_level",
                "PolII_balance",
            ]
            col = marker if derived else f"{marker}_enrichment"
            eligible = selected[
                selected["PolII_eligible" if derived else f"{marker}_qc_clean"]
            ]
            summary[f"mn_{group}_{marker}_median"] = summary.label.map(
                eligible.groupby("parent_nucleus_id")[col].median()
            )
            summary[f"mn_{group}_{marker}_eligible_count"] = summary.label.map(
                eligible.groupby("parent_nucleus_id").size()
            ).fillna(0)
            summary.loc[
                ~summary.mn_domain_available, f"mn_{group}_{marker}_eligible_count"
            ] = np.nan
    return summary


def load_npz(path):
    """Load numerical arrays only; never deserialize Python objects."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha256(path):
    """Hash inputs in chunks to avoid a second full image allocation."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(frame, path, metadata=None):
    """Write an explicit object table with its plate/well/tile identity."""
    frame = frame.copy()
    for key, value in (metadata or {}).items():
        frame.insert(0, key, str(value))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def implementation_fingerprint():
    """Track imported algorithms as rule parameters, avoiding stale mask caches."""
    names = [
        "mn_pipeline",
        "nuclear_bodies",
        "nuclear_body_rounds",
        "mn_consensus",
        "mn_reporter",
        "chromatin_quantification",
        "reporter_polii",
        "bp1_boundary_refinement",
    ]
    return {name: sha256(Path(__file__).with_name(name + ".py")) for name in names}


def execute_stage(stage, inputs, outputs, settings, metadata):
    """Thin file adapter used by Snakemake and the regression test runner."""
    for path in outputs.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if stage == "register":
        image = imread(inputs["image"])
        source = inputs.get("source", inputs["image"])
        raw = image if str(source) == str(inputs["image"]) else imread(source)
        arrays, registration = register_rounds(image, settings, raw)
        np.savez_compressed(outputs["aligned"], **arrays)
        manifest = dict(
            registration=registration,
            settings=settings,
            input_sha256=sha256(inputs["image"]),
            raw_source_sha256=sha256(source),
        )
        Path(outputs["registration"]).write_text(json.dumps(manifest, indent=2) + "\n")
    elif stage == "export":
        # Standard Brieflow phenotype extraction also needs registered riders.
        # Retain the original CYX order, including each DAPI channel, so ordinary
        # nuclear feature names remain meaningful alongside the new MN columns.
        stack = imread(inputs["image"]).astype(np.float32)
        arrays = load_npz(inputs["aligned"])
        records = json.loads(Path(inputs["registration"]).read_text())["registration"]
        names = settings["channel_names"]
        for round_name, entry in settings["rounds"].items():
            index = names.index(entry["dapi"])
            if round_name != "primary":
                stack[index] = shift_intensity(
                    stack[index], records[round_name]["shift_yx_to_reference"]
                )
            for role, channel in entry["markers"].items():
                stack[names.index(channel)] = arrays[role]
        imwrite(
            outputs["registered_image"],
            stack,
            compression="zlib",
            metadata={"axes": "CYX"},
        )
    elif stage == "segment":
        images = load_npz(inputs["aligned"])
        masks, objects, reporters, nucleoli, domain = segment_components(
            images, imread(inputs["nuclei"]), settings
        )
        np.savez_compressed(outputs["components"], **masks)
        for key, frame in [
            ("objects", objects),
            ("reporters", reporters),
            ("nucleoli", nucleoli),
        ]:
            write_table(frame, outputs[key], metadata)
        Path(outputs["domain"]).write_text(json.dumps(domain, indent=2) + "\n")
    elif stage == "refine":
        components, audit, info = refine_components(
            load_npz(inputs["aligned"]), load_npz(inputs["components"]), settings
        )
        np.savez_compressed(outputs["refined_components"], **components)
        write_table(audit, outputs["bp1_boundaries"], metadata)
        info["input_sha256"] = {k: sha256(v) for k, v in inputs.items()}
        info["algorithm_sha256"] = implementation_fingerprint()
        Path(outputs["boundary_provenance"]).write_text(
            json.dumps(info, indent=2) + "\n"
        )
    elif stage == "define":
        derived, decisions, pairs, membership, a, c = define_mn(
            load_npz(inputs["aligned"]), load_npz(inputs["components"]), settings
        )
        np.savez_compressed(outputs["derived"], **derived)
        for key, frame in [
            ("decisions", decisions),
            ("pairs", pairs),
            ("membership", membership),
            ("contacts_a", a),
            ("contacts_c", c),
        ]:
            write_table(frame, outputs[key], metadata)
    elif stage == "measure":
        components = load_npz(inputs["components"])
        membership = pd.read_csv(inputs["membership"], sep="\t").drop(
            columns=list(metadata), errors="ignore"
        )
        data, parents = measure_mn(
            load_npz(inputs["aligned"]),
            components,
            load_npz(inputs["derived"]),
            membership,
            settings,
        )
        write_table(data, outputs["features"], metadata)
        write_table(parents, outputs["parents"], metadata)
        write_table(parent_summary(data, components), outputs["summary"], metadata)
        Path(outputs["provenance"]).write_text(
            json.dumps(
                dict(
                    settings=settings,
                    metadata=metadata,
                    input_sha256={str(k): sha256(v) for k, v in inputs.items()},
                    source_sha256=sha256(__file__),
                    algorithm_sha256=implementation_fingerprint(),
                    n_mn=len(data),
                    n_with_reporter=int(data.with_reporter.sum()),
                    normalization="Object mean / depleted-parent mean; no DAPI adjustment",
                    PolII_level="(Ser5 enrichment + Ser2 enrichment)/2",
                    PolII_balance="log2(Ser5 enrichment/Ser2 enrichment)",
                    row_key=["plate", "well", "tile", "mn_id"],
                    parent_key=["plate", "well", "tile", "parent_nucleus_id"],
                ),
                indent=2,
            )
            + "\n"
        )
    else:
        raise ValueError(f"Unknown MN stage: {stage}")
