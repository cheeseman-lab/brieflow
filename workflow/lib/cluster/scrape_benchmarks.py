import re
import requests
from requests.adapters import HTTPAdapter, Retry
import json
import io
import gzip

import pandas as pd


def generate_string_pair_benchmark(aggregated_data, gene_col="gene_symbol_0"):
    string_data = get_string_data()
    uniprot_data = get_uniprot_data()

    # Create mapping from STRING IDs to gene names
    string_to_genes = {}
    for _, row in uniprot_data.iterrows():
        if pd.notna(row["STRING"]) and row["STRING"] != "":
            string_id = (
                row["STRING"].split(";")[0] if ";" in row["STRING"] else row["STRING"]
            )
            gene_names = row["Gene Names"]
            if pd.notna(gene_names):
                string_to_genes[string_id] = gene_names

    # Create the benchmark dataframe
    data = []
    for index, row in string_data.reset_index(drop=True).iterrows():
        protein1 = row["protein1"]
        protein2 = row["protein2"]

        # Get gene names for each protein ID
        genes_variants_1 = string_to_genes.get(protein1, None)
        genes_variants_2 = string_to_genes.get(protein2, None)

        if genes_variants_1 is None or genes_variants_2 is None:
            continue
        else:
            data.append(
                {
                    "gene_name_variants": genes_variants_1,
                    "pair": index + 1,
                }
            )
            data.append(
                {
                    "gene_name_variants": genes_variants_2,
                    "pair": index + 1,
                }
            )

    string_pair_benchmark = (
        pd.DataFrame(data).sort_values("pair").reset_index(drop=True)
    )
    string_pair_benchmark = select_gene_variants(
        string_pair_benchmark, aggregated_data, gene_col
    )

    return string_pair_benchmark


def generate_corum_group_benchmark():
    corum_data = get_corum_data()

    # Create the new dataframe with columns for gene_name and complex
    benchmark_rows = []

    for _, row in corum_data.iterrows():
        # Split the gene names by semicolon
        subunits = row["subunits_gene_name"].split(";")

        # Get the complex name
        complex_name = row["complex_name"]

        # Add each gene to the rows list with the current group_id
        for gene in subunits:
            benchmark_rows.append({"gene_name": gene, "group": complex_name})

    # Create the DataFrame from the rows
    corum_cluster_benchmark = pd.DataFrame(benchmark_rows)

    return corum_cluster_benchmark


def generate_msigdb_group_benchmark(
    url="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.kegg_medicus.v2024.1.Hs.json",
):
    """
    Generate group benchmark from Molecular Signatures Database (MSigDB) data. We use Kegg as default.
    """
    response = requests.get(url)
    msigdb_data = json.loads(response.text)

    # Create lists to hold data for DataFrame
    pathways = []
    genes = []

    # Process each pathway entry
    for pathway_id, pathway_data in msigdb_data.items():
        gene_symbols = pathway_data.get("geneSymbols", [])

        pathways.append(pathway_id)
        genes.append(gene_symbols)

    # Create DataFrame
    group_benchmark_df = pd.DataFrame({"pathway_id": pathways, "gene_symbol": genes})

    # Expand gene symbols into rows
    group_benchmark_df = group_benchmark_df.explode("gene_symbol")

    # Rename and reorder columns
    group_benchmark_df = group_benchmark_df.rename(
        columns={"gene_symbol": "gene_name", "pathway_id": "group"}
    )[["gene_name", "group"]]

    return group_benchmark_df.reset_index(drop=True)


def get_uniprot_data():
    """Fetch all human-reviewed UniProt data using the REST API.

    Returns:
        pd.DataFrame: DataFrame with UniProt data.
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
        print(f"Progress: {progress} / {total}", end="\r")

        batch_url = get_next_link(response.headers)

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=headers)
    print(f"Completed. Total entries: {len(df)}")
    return df


def get_corum_data():
    """Fetch all human-reviewed UniProt data using the REST API.

    Returns:
        pd.DataFrame: DataFrame with UniProt data.
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
    """Fetch CORUM complex data for human proteins.

    Returns:
        pd.DataFrame: DataFrame with CORUM complex data for human proteins.
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


def select_gene_variants(benchmark_df, ref_gene_df, ref_gene_col="gene_symbol_0"):
    """
    Selects the appropriate gene name from variants that matches the cluster genes.
    If no match is found, the first gene name is selected.
    """
    # Get all unique genes in the cluster_df
    ref_genes = set(ref_gene_df[ref_gene_col])

    # Select the appropriate gene name from variants that matches cluster genes
    def select_gene_name(variants):
        for gene in variants.split():
            if gene in ref_genes:
                return gene
        return variants.split()[0]

    # Create a copy of pathway_df and add gene_name column
    benchmark_df = benchmark_df.copy()
    benchmark_df["gene_name"] = benchmark_df["gene_name_variants"].apply(
        select_gene_name
    )

    return benchmark_df
