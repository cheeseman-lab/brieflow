"""Utilities for generating gene set benchmarks from external biological databases.

This module fetches and processes benchmark datasets used to evaluate gene clustering
performance. It includes access to STRING, CORUM, MSigDB, and UniProt resources,
and provides standardized functions to format, clean, and filter group and pairwise
gene benchmarks for use in enrichment and precision-recall analyses.

Key functions:
    - generate_string_pair_benchmark: Create gene pair benchmarks from STRING interactions.
    - generate_corum_group_benchmark: Extract gene complexes from CORUM.
    - generate_msigdb_group_benchmark: Convert MSigDB KEGG pathways to gene group format.
    - get_uniprot_data: Retrieve UniProt gene and annotation data.
    - get_corum_data / get_string_data: Download raw benchmark inputs.
    - select_gene_variants: Harmonize variant gene names with clustering outputs.
    - filter_complexes: Curate benchmark gene groups with coverage and overlap filters.
"""

import re
import requests
from requests.adapters import HTTPAdapter, Retry
import json
import io
import gzip
from itertools import combinations

import pandas as pd


def generate_string_pair_benchmark(
    aggregated_data, uniprot_data, gene_col="gene_symbol_0"
):
    """Generate a STRING pair benchmark DataFrame.

    This function maps STRING protein IDs to gene names and creates a benchmark DataFrame
    for STRING protein pairs. It filters and selects gene variants based on the provided
    aggregated data.

    Args:
        aggregated_data (pd.DataFrame): The aggregated data containing gene information.
        uniprot_data (pd.DataFrame): The UniProt data containing STRING IDs and gene names.
        gene_col (str, optional): The column name in the aggregated data representing gene symbols.
            Defaults to "gene_symbol_0".

    Returns:
        pd.DataFrame: A DataFrame containing the STRING pair benchmark.
    """
    string_data = get_string_data()

    # Create mapping from STRING IDs to gene names
    string_to_genes = {}
    for _, row in uniprot_data.iterrows():
        if pd.notna(row["string"]) and row["string"] != "":
            string_id = (
                row["string"].split(";")[0] if ";" in row["string"] else row["string"]
            )
            gene_names = row["gene_names"]
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
    """Generate a CORUM group benchmark DataFrame.

    This function processes CORUM data to create a benchmark DataFrame with gene names
    and their associated protein complexes.

    Returns:
        pd.DataFrame: A DataFrame containing the CORUM group benchmark.
    """
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
    """Generate a group benchmark from MSigDB data.

    This function fetches pathway data from the Molecular Signatures Database (MSigDB)
    and creates a benchmark DataFrame with gene names and their associated pathways.

    Args:
        url (str, optional): The URL to fetch MSigDB data. Defaults to the KEGG Medicus pathway.

    Returns:
        pd.DataFrame: A DataFrame containing the MSigDB group benchmark.
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

    This function retrieves UniProt data for human-reviewed entries, including gene names,
    functions, and cross-references to STRING, KEGG, and ComplexPortal.

    Returns:
        pd.DataFrame: A DataFrame containing UniProt data with UniProt entry links.
    """
    # Define UniProt REST API query
    re_next_link = re.compile(r"<(.+)>; rel=\"next\"")
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
        "query": "organism_id:9606",
        "fields": "accession,gene_names,cc_function,xref_kegg,xref_complexportal,xref_string",
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

    # Generate UniProt links using the accession field
    df["Link"] = "https://www.uniprot.org/uniprotkb/" + df["Entry"] + "/entry"

    # Make all column names standardized
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Rename the function_[cc] column to just function
    if "function_[cc]" in df.columns:
        df = df.rename(columns={"function_[cc]": "function"})

    # Remove the string "FUNCTION: " from all entries in the function column
    if "function" in df.columns:
        df["function"] = df["function"].str.replace("FUNCTION: ", "", regex=False)

    print(f"Completed. Total entries: {len(df)}")
    return df


def get_corum_data():
    """Fetch CORUM complex data for human proteins.

    This function retrieves CORUM data for human protein complexes and processes it
    into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing CORUM complex data.
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
    """Fetch STRING interaction data for human proteins.

    This function retrieves STRING interaction data for human proteins and filters
    interactions with a combined score of 950 or higher.

    Returns:
        pd.DataFrame: A DataFrame containing STRING interaction data.
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
    """Select appropriate gene names from variants that match cluster genes.

    This function selects the most relevant gene name from a list of variants
    based on a reference gene DataFrame. If no match is found, the first gene
    name in the list is selected.

    Args:
        benchmark_df (pd.DataFrame): The benchmark DataFrame containing gene name variants.
        ref_gene_df (pd.DataFrame): The reference DataFrame containing cluster genes.
        ref_gene_col (str, optional): The column name in the reference DataFrame representing gene symbols.
            Defaults to "gene_symbol_0".

    Returns:
        pd.DataFrame: A DataFrame with the selected gene names.
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


def filter_complexes(
    group_df, cluster_df, perturbation_col_name=None, control_key=None
):
    """Filter complexes based on gene coverage and overlap.

    Args:
        group_df (pd.DataFrame): DataFrame with columns ['gene_name', 'group'].
        cluster_df (pd.DataFrame): DataFrame with perturbation data.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str, optional): Prefix for control perturbations to filter out.

    Returns:
        pd.DataFrame: Filtered group DataFrame.
    """
    # Generate the screening gene list directly from the cluster_df
    gene_list = [
        gene
        for gene in cluster_df[perturbation_col_name].unique()
        if control_key is None or control_key not in gene
    ]

    # 1. Build a dictionary: complex -> set of genes
    complex_to_genes = group_df.groupby("group")["gene_name"].apply(set).to_dict()

    # 2. Find complexes with ≥3 genes from gene_list and ≥2/3 of complex represented
    selected_complexes = {}
    for complex_name, genes in complex_to_genes.items():
        genes_in_library = genes.intersection(gene_list)
        if len(genes_in_library) >= 3 and len(genes_in_library) / len(genes) >= (2 / 3):
            selected_complexes[complex_name] = genes_in_library

    # 3. Remove larger complexes that share >10% of gene-pairs with smaller ones
    #    (compare by % of *gene pairs* that overlap)
    final_complexes = set(selected_complexes.keys())
    complex_list_sorted = sorted(
        selected_complexes.items(), key=lambda x: len(x[1])
    )  # small to large

    for i, (small_complex, small_genes) in enumerate(complex_list_sorted):
        small_pairs = set(combinations(small_genes, 2))
        for larger_complex, larger_genes in complex_list_sorted[i + 1 :]:
            larger_pairs = set(combinations(larger_genes, 2))
            if len(small_pairs) == 0:
                continue
            shared_pairs = small_pairs & larger_pairs
            if len(shared_pairs) / len(small_pairs) > 0.10:
                final_complexes.discard(larger_complex)

    filtered = group_df[group_df["group"].isin(final_complexes)]
    return filtered
