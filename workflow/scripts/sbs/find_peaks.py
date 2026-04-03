from lib.shared.io import read_image, save_image

# Get configuration from params
params = snakemake.params.config

# Choose spot calling method based on parameter
method = params.get("method", "standard")


if method == "standard":
    from lib.sbs.find_peaks import find_peaks

    # Load standard deviation data
    standard_deviation_data = read_image(snakemake.input[0])

    # Get standard method parameters
    peak_width = params["peak_width"]

    # Find peaks using standard method
    peaks = find_peaks(
        standard_deviation_data=standard_deviation_data, width=peak_width
    )

elif method == "spotiflow":
    from lib.sbs.find_peaks import find_peaks_spotiflow

    # Load aligned images for spotiflow method
    aligned_images = read_image(snakemake.input[0])

    # Get spotiflow parameters
    model = params["spotiflow_model"]
    prob_thresh = params["spotiflow_threshold"]
    cycle_idx = params["spotiflow_cycle_index"]
    min_distance = params["spotiflow_min_distance"]
    remove_index = params["remove_index"]

    # Find peaks using spotiflow method
    peaks, _ = find_peaks_spotiflow(
        aligned_images=aligned_images,
        cycle_idx=cycle_idx,
        model=model,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        remove_index=remove_index,
        verbose=False,
    )

# Save peak data (same for both methods)
save_image(peaks, snakemake.output[0])
