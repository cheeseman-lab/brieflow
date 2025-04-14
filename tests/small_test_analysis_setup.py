from pathlib import Path
import requests
import zipfile

from tqdm import tqdm

SMALL_TEST_DATA_URL = (
    "https://zenodo.org/records/15199415/files/small_test_analysis.zip"
)
local_filename = Path("small_test_analysis.zip")

# Download the zip file
response = requests.get(SMALL_TEST_DATA_URL, stream=True)
response.raise_for_status()
total = int(response.headers.get("content-length", 0))
with (
    open(local_filename, "wb") as f,
    tqdm(
        desc=f"Downloading test data",
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
with zipfile.ZipFile(local_filename, "r") as zip_ref:
    members = zip_ref.infolist()
    for member in tqdm(members, desc="Unzipping", unit="file"):
        zip_ref.extract(member)

# Delete the zip file after extraction
local_filename.unlink()
