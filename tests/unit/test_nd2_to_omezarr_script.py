import importlib.util
import sys
from pathlib import Path


def test_script_parses_params_and_invokes_monkeypatched_functions(monkeypatch, tmp_path: Path) -> None:
    # Prepare fake snakemake object
    output_dir = tmp_path / "image.zarr"
    captured = {}

    def fake_nd2_to_omezarr(input_, output_dir, channel_order_flip, chunk_shape, coarsening_factor, max_levels, verbose, **kwargs):
        captured["called"] = True
        captured["args"] = {
            "input": input_,
            "output_dir": output_dir,
            "channel_order_flip": channel_order_flip,
            "chunk_shape": tuple(chunk_shape),
            "coarsening_factor": coarsening_factor,
            "max_levels": max_levels,
            "verbose": verbose,
            "compressor": kwargs.get("compressor"),
        }
        # Return the output path expected by the script post-processing
        return output_dir

    def fake_ensure_colors(result_path, channel_labels=None):
        captured["channel_labels"] = channel_labels

    class FakeSnakemake:
        input = ["a.nd2", "b.nd2"]
        output = [str(output_dir)]
        params = {
            "channel_order_flip": True,
            "chunk_shape": ["1", "64", "64"],  # strings to test int casting
            "coarsening_factor": 2,
            "max_levels": 3,
            "verbose": True,
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2},
            "channel_labels": ["DAPI", "GFP"],
        }

    # Patch functions used in the script
    import lib.preprocess.preprocess as preprocess_mod
    monkeypatch.setattr(preprocess_mod, "nd2_to_omezarr", fake_nd2_to_omezarr)

    # Load the script as a module and inject fake snakemake and ensure_omero_channel_colors
    script_path = Path(__file__).resolve().parents[2] / "workflow" / "scripts" / "preprocess" / "nd2_to_omezarr.py"
    spec = importlib.util.spec_from_file_location("nd2_to_omezarr_script", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.snakemake = FakeSnakemake()  # type: ignore[attr-defined]

    # Provide the already-imported module reference so the script resolves to the patched functions
    sys.modules["lib.preprocess.preprocess"] = preprocess_mod

    # Patch ensure_omero_channel_colors where the script imported it
    import lib.shared.omezarr_utils as omero_utils
    monkeypatch.setattr(omero_utils, "ensure_omero_channel_colors", fake_ensure_colors)
    sys.modules["lib.shared.omezarr_utils"] = omero_utils

    spec.loader.exec_module(module)  # type: ignore[arg-type]

    # Assertions
    assert captured.get("called") is True
    args = captured["args"]
    assert args["output_dir"] == output_dir
    assert args["channel_order_flip"] is True
    assert args["chunk_shape"] == (1, 64, 64)
    assert args["coarsening_factor"] == 2
    assert args["max_levels"] == 3
    assert args["verbose"] is True
    # Compressor is currently disabled (set to None) to avoid blosc dependency issues
    assert args["compressor"] is None
    assert captured["channel_labels"] == ["DAPI", "GFP"]


