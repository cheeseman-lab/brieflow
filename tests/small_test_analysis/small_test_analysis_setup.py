from pathlib import Path
import requests
import zipfile

from tqdm import tqdm

SMALL_TEST_DATA_URL = "https://zenodo.org/records/15276612/files/small_test_data.zip"
local_fp = Path("small_test_data.zip")

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
