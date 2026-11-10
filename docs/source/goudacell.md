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
      threshold_method: otsu_three_peak_mid_bg
      threshold_smoothing_scale: 0.7
      declump_method: none
      size_filter_method: area
      second_obj_min_size: 3
      second_obj_max_size: 400
      max_total_objects: null
```

The `granule` entry is the setting that scored best in one held-out benchmark of small cytoplasmic granules (F1 0.75 at IoU 0.5 on three labelled images), ahead of Cellpose and StarDist models on the same images. It is one experiment, not a default: other objects, stains and microscopes need their own tuning. `max_total_objects: null` turns off the failsafe that empties a tile with more than 1000 objects, which dense granule tiles exceed. A fine-tuned model for secondary objects goes in `second_obj_cellpose_model` (`SECOND_OBJ_CELLPOSE_MODEL`) with `second_obj_method: cellpose`.
