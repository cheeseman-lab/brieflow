# Tuning segmentation with goudacell

[goudacell](https://github.com/cheeseman-lab/goudacell) is a bench for testing and fine-tuning cell and {term}`secondary object <Secondary object>` segmentation on your own tiles. It runs brieflow's phenotype segmentation code, so settings tuned there give the same masks in the pipeline. Fine-tuning Cellpose used to live in brieflow; it now lives in goudacell. goudacell is set up for the Whitehead Institute cluster and its GPU nodes, and can be adapted to another backend.

goudacell exports a block of config keys, plus a model path when you fine-tuned a Cellpose model. Copy it by hand into the `phenotype:` section of `config/config.yml`, or into the matching variables of `2_phenotype.py`, which rewrites `config.yml` each time it runs. In the notebook, a `second_objs` entry goes into `SECOND_OBJS` with `"channel": "<channel name>"` in place of `second_obj_channel_index`. With [brieflow-auto](brieflow_auto.md), set `segmentation_overrides` in `screen.yaml` to the exported file instead.

```yaml
phenotype:
  cellpose_model: /path/to/fine_tuned_model  # Cellpose version is read from the weights
  nuclei_diameter: 38.0
  cell_diameter: 61.0
  cell_flow_threshold: 0.4
  cell_cellprob_threshold: 0.0
  second_obj_detection: true
  second_objs:
    granule:
      second_obj_channel_index: 2
      threshold_method: cell_median_mad
      threshold_k: 5
      threshold_smoothing_scale: 0.7
      declump_method: none
      size_filter_method: area
      second_obj_min_size: 3
      second_obj_max_size: 1000
      max_total_objects: 5000
    puncta:
      second_obj_channel_index: 1
      second_obj_mode: count
      foci_radius: 2
      foci_threshold: 5
```

The `granule` entry comes from one experiment on small cytoplasmic stress granules, not a default. `cell_median_mad` places the cutoff `threshold_k` robust standard deviations above each tile's in-cell median, so a tile with no granules (for example a knockdown that loses them) calls few objects, and a brighter or dimmer tile calls about the same number. A per-tile Otsu threshold kept calling objects in the same fields with the granules removed. In that experiment the counts matched a per-image threshold on labelled images, which is not ground truth, and the smallest perinuclear granules at a late time point stayed unsolved by every method tried. Check the masks on your own tiles. `max_total_objects: 5000` raises the failsafe that empties a tile with more than 1000 objects; keep a limit, because a tile without cells can threshold into tens of thousands of noise objects. The `puncta` entry counts spots per cell instead of measuring each one (`<name>_count` and per-channel totals and means), which is enough for puncta and centrosomes and much faster. A fine-tuned model for secondary objects goes in `second_obj_cellpose_model` (`SECOND_OBJ_CELLPOSE_MODEL`) with `second_obj_method: cellpose`.
