import numpy as np


def edge_intensity_features(intensity_image, filled_image, **kwargs):
    edge_pixels = intensity_image[boundaries(filled_image, **kwargs), ...]

    return np.array(
        [
            edge_pixels.sum(axis=0),
            edge_pixels.mean(axis=0),
            np.std(edge_pixels, axis=0),
            edge_pixels.max(axis=0),
            edge_pixels.min(axis=0),
        ]
    ).flatten()


intensity_features_multichannel = {
    "int": lambda r: r.intensity_image[r.image, ...].sum(axis=0),
    "mean": lambda r: r.intensity_image[r.image, ...].mean(axis=0),
    "std": lambda r: np.std(r.intensity_image[r.image, ...], axis=0),
    "max": lambda r: r.intensity_image[r.image, ...].max(axis=0),
    "min": lambda r: r.intensity_image[r.image, ...].min(axis=0),
    "edge_intensity_feature": lambda r: edge_intensity_features(
        r.intensity_image, r.filled_image, mode="inner", connectivity=EDGE_CONNECTIVITY
    ),
    "mass_displacement": lambda r: np.sqrt(
        (
            (
                np.array(r.local_centroid)[:, None]
                - catch_runtime(lambda r: r.weighted_local_centroid)(r)
            )
            ** 2
        ).sum(axis=0)
    ),
    "lower_quartile": lambda r: np.percentile(
        r.intensity_image[r.image, ...], 25, axis=0
    ),
    "median": lambda r: np.median(r.intensity_image[r.image, ...], axis=0),
    "mad": lambda r: median_abs_deviation(
        r.intensity_image[r.image, ...], scale=1, axis=0
    ),
    "upper_quartile": lambda r: np.percentile(
        r.intensity_image[r.image, ...], 75, axis=0
    ),
    "center_mass": lambda r: catch_runtime(lambda r: r.weighted_local_centroid)(
        r
    ).flatten(),  # this property is not cached
    "max_location": lambda r: np.array(
        np.unravel_index(
            np.argmax(
                r.intensity_image.reshape(-1, *r.intensity_image.shape[2:]), axis=0
            ),
            (r.image).shape,
        )
    ).flatten(),
}

grayscale_features_multichannel = {
    **intensity_features_multichannel,
    **intensity_distribution_features_multichannel,
    **texture_features_multichannel,
}
