import re
import requests
from requests.adapters import HTTPAdapter, Retry
import io
import gzip

import pandas as pd


def get_uniprot_data():
    """
    Fetch all human reviewed UniProt data using REST API

    Returns:
    --------
    pandas.DataFrame
        DataFrame with UniProt data
    """
    # Define UniProt REST API query
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Function to extract next link from headers
    def get_next_link(headers):
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    # Fetch UniProt data
    url = "https://rest.uniprot.org/uniprotkb/search"
    # Query for human reviewed entries with specific fields
    params = {
        "query": "organism_id:9606 AND reviewed:true",
        "fields": "gene_names,cc_function,xref_kegg,xref_complexportal,xref_string",
        "format": "tsv",
        "size": 500,
    }

    # Fetch data in batches
    initial_response = session.get(url, params=params)
    batch_url = initial_response.url
    results = []
    progress = 0

    # Process each batch
    print("Fetching UniProt data...")
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = int(response.headers["x-total-results"])

        lines = response.text.splitlines()
        if progress == 0:
            headers = lines[0].split("\t")

        for line in lines[1:] if progress == 0 else lines:
            results.append(line.split("\t"))

        progress += len(lines[1:] if progress == 0 else lines)
        print(f"Progress: {progress} / {total}")

        batch_url = get_next_link(response.headers)

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=headers)
    print(f"Completed. Total entries: {len(df)}")
    return df


def get_corum_data():
    """
    Fetch CORUM complex data for human proteins
    """
    print("Fetching CORUM data...")
    url = "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file"

    # Parameters for human complexes in text format
    params = {"file_id": "human", "file_format": "txt"}

    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()

    # Read data into DataFrame
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    print(f"Completed. Total complexes: {len(df)}")
    return df


def get_string_data():
    """
    Fetch STRING interaction data for human proteins
    """
    print("Fetching STRING data...")
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"

    response = requests.get(url)
    response.raise_for_status()

    # Read compressed data directly into DataFrame
    with gzip.open(io.BytesIO(response.content), "rt") as f:
        df = pd.read_csv(f, sep=" ")

    # Filter interactions with combined score >= 950
    df = df[df["combined_score"] >= 950]
    print(f"Completed. Total interactions: {len(df)}")
    return df
