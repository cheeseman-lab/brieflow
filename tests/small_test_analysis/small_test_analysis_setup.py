import os
import re
from pathlib import Path
import requests
import zipfile

from tqdm import tqdm

SMALL_TEST_DATA_URL = "https://zenodo.org/records/15276612/files/small_test_data.zip"
local_fp = Path("small_test_data.zip")
data_dir = Path("small_test_data")


def stage_layout(data_dir):
    """Symlink the flat download into the layout brieflow expects.

    SBS images go to sbs/plate_<P>/c<C>/ and phenotype images to
    phenotype/plate_<P>/round_1/; the downloaded files stay where they are.
    """
    sbs_re = re.compile(r"^P(\d+)_SBS_.*_C(\d+)_Wells-")
    ph_re = re.compile(r"^P(\d+)_Pheno_")
    for modality, pattern in (("sbs", sbs_re), ("phenotype", ph_re)):
        for image_fp in sorted((data_dir / modality).glob("*_images/*.nd2")):
            match = pattern.match(image_fp.name)
            if match is None:
                raise ValueError(f"Unexpected file name in the test data: {image_fp}")
            plate = int(match.group(1))
            subdir = f"c{int(match.group(2))}" if modality == "sbs" else "round_1"
            link_fp = data_dir / modality / f"plate_{plate}" / subdir / image_fp.name
            link_fp.parent.mkdir(parents=True, exist_ok=True)
            if not link_fp.is_symlink():
                link_fp.symlink_to(os.path.relpath(image_fp, link_fp.parent))


if not data_dir.exists():
    # Download the zip file
    response = requests.get(SMALL_TEST_DATA_URL, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(local_fp, "wb") as f,
        tqdm(
            desc=f"Downloading small test data",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    # Unzip the downloaded file
    with zipfile.ZipFile(local_fp, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Unzipping", unit="file"):
            zip_ref.extract(member, path=local_fp.parent)

    # Delete the zip file after extraction
    local_fp.unlink()

# Stage the images into the plate/cycle and plate/round directories
stage_layout(data_dir)
