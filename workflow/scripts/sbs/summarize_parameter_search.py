import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Collect all parameter search results
all_optimal_params = []
all_global_results = []
all_metadata = []

print(f"Collecting results from {len(snakemake.input)} wells...")

for i, input_files in enumerate(snakemake.input):
    # Each input_files contains the 5 outputs from sbs_parameter_search
    optimal_params_file = input_files[0]
    global_results_file = input_files[1] 
    metadata_file = input_files[3]
    
    try:
        # Load optimal parameters
        optimal_params = pd.read_csv(optimal_params_file, sep='\t')
        all_optimal_params.append(optimal_params)
        
        # Load global results
        global_results = pd.read_csv(global_results_file, sep='\t')
        if len(global_results) > 0:
            all_global_results.append(global_results)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            all_metadata.append(metadata)
            
    except Exception as e:
        print(f"Failed to load results from {optimal_params_file}: {e}")

# Combine all results
if all_optimal_params:
    combined_optimal_params = pd.concat(all_optimal_params, ignore_index=True)
else:
    combined_optimal_params = pd.DataFrame()

if all_global_results:
    combined_global_results = pd.concat(all_global_results, ignore_index=True)
else:
    combined_global_results = pd.DataFrame()

# Create summary statistics
summary_stats = {}
if len(combined_optimal_params) > 0:
    summary_stats = {
        'total_wells_processed': len(combined_optimal_params),
        'wells_with_successful_results': len(combined_optimal_params[combined_optimal_params.get('total_reads', 0) > 0]),
        'recommended_peak_width': int(combined_optimal_params['peak_width'].mode().iloc[0]) if len(combined_optimal_params) > 0 else None,
        'recommended_threshold_reads': int(combined_optimal_params['threshold_reads'].mode().iloc[0]) if len(combined_optimal_params) > 0 else None,
        'mean_any_barcode_mapping': float(combined_optimal_params['fraction_any_barcode'].mean()) if 'fraction_any_barcode' in combined_optimal_params.columns else 0.0,
        'mean_one_barcode_mapping': float(combined_optimal_params['fraction_one_barcode'].mean()) if 'fraction_one_barcode' in combined_optimal_params.columns else 0.0,
        'total_tiles_sampled': sum([m.get('n_tiles_sampled', 0) for m in all_metadata]),
        'total_tiles_successful': sum([m.get('n_tiles_successful', 0) for m in all_metadata]),
        'total_cells_analyzed': sum([m.get('total_cells', 0) for m in all_metadata]),
    }

# Save summary results
combined_optimal_params.to_csv(snakemake.output[0], index=False, sep='\t')

# Create per-well summary
well_summary = combined_optimal_params.copy()
if len(well_summary) > 0:
    well_summary['parameter_combination'] = well_summary['peak_width'].astype(str) + '_' + well_summary['threshold_reads'].astype(str)
    well_summary['mapping_quality'] = well_summary.get('fraction_any_barcode', 0) - well_summary.get('fraction_one_barcode', 0)

well_summary.to_csv(snakemake.output[1], index=False, sep='\t')

# Create recommendations text file
with open(snakemake.output[2], 'w') as f:
    f.write("SBS Parameter Search Recommendations\n")
    f.write("====================================\n\n")
    
    if len(combined_optimal_params) > 0:
        f.write(f"Analysis Summary:\n")
        f.write(f"- Total wells analyzed: {summary_stats['total_wells_processed']}\n")
        f.write(f"- Wells with successful results: {summary_stats['wells_with_successful_results']}\n")
        f.write(f"- Total tiles sampled: {summary_stats['total_tiles_sampled']}\n")
        f.write(f"- Total tiles with successful parameter combinations: {summary_stats['total_tiles_successful']}\n")
        f.write(f"- Total cells analyzed: {summary_stats['total_cells_analyzed']:,}\n\n")
        
        f.write(f"Recommended Parameters:\n")
        f.write(f"- PEAK_WIDTH: {summary_stats['recommended_peak_width']}\n")
        f.write(f"- THRESHOLD_READS: {summary_stats['recommended_threshold_reads']}\n\n")
        
        f.write(f"Expected Performance:\n")
        f.write(f"- Mean any-barcode mapping: {summary_stats['mean_any_barcode_mapping']:.3f}\n")
        f.write(f"- Mean one-barcode mapping: {summary_stats['mean_one_barcode_mapping']:.3f}\n\n")
        
        # Parameter frequency analysis
        param_freq = combined_optimal_params.groupby(['peak_width', 'threshold_reads']).size().reset_index(name='frequency')
        param_freq = param_freq.sort_values('frequency', ascending=False)
        
        f.write(f"Parameter Combination Frequency:\n")
        for _, row in param_freq.head(10).iterrows():
            f.write(f"- peak_width={int(row['peak_width'])}, threshold_reads={int(row['threshold_reads'])}: {int(row['frequency'])} wells\n")
        
        # Well-specific recommendations
        if len(combined_optimal_params) > 1:
            f.write(f"\nPer-Well Recommendations:\n")
            for _, row in combined_optimal_params.iterrows():
                plate = row.get('plate', 'unknown')
                well = row.get('well', 'unknown')
                f.write(f"- {plate}_{well}: peak_width={int(row['peak_width'])}, threshold_reads={int(row['threshold_reads'])} ")
                f.write(f"(any_mapping={row.get('fraction_any_barcode', 0):.3f})\n")
    else:
        f.write("No successful parameter search results found.\n")
        f.write("Check data quality and parameter search configuration.\n")

# Create comparison plots
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

if len(combined_optimal_params) > 0:
    # Plot 1: Parameter frequency heatmap
    param_counts = combined_optimal_params.groupby(['peak_width', 'threshold_reads']).size().unstack(fill_value=0)
    if not param_counts.empty:
        sns.heatmap(param_counts, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Parameter Combination Frequency Across Wells')
        axes[0,0].set_xlabel('Threshold Reads')
        axes[0,0].set_ylabel('Peak Width')
    
    # Plot 2: Performance distribution
    if 'fraction_any_barcode' in combined_optimal_params.columns:
        axes[0,1].hist(combined_optimal_params['fraction_any_barcode'], bins=20, alpha=0.7, color='skyblue')
        axes[0,1].axvline(combined_optimal_params['fraction_any_barcode'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {combined_optimal_params["fraction_any_barcode"].mean():.3f}')
        axes[0,1].set_title('Distribution of Any-Barcode Mapping Performance')
        axes[0,1].set_xlabel('Fraction Any-Barcode Mapping')
        axes[0,1].set_ylabel('Number of Wells')
        axes[0,1].legend()
    
    # Plot 3: Parameter vs Performance scatter
    if len(combined_optimal_params) > 1:
        scatter = axes[1,0].scatter(combined_optimal_params['peak_width'], 
                                  combined_optimal_params['threshold_reads'],
                                  c=combined_optimal_params.get('fraction_any_barcode', 0),
                                  cmap='viridis', alpha=0.7, s=60)
        axes[1,0].set_title('Parameter Space vs Performance')
        axes[1,0].set_xlabel('Peak Width')
        axes[1,0].set_ylabel('Threshold Reads')
        plt.colorbar(scatter, ax=axes[1,0], label='Any-Barcode Mapping')
    
    # Plot 4: Well-to-well variability
    if 'plate' in combined_optimal_params.columns and 'well' in combined_optimal_params.columns:
        combined_optimal_params['well_id'] = combined_optimal_params['plate'] + '_' + combined_optimal_params['well']
        well_performance = combined_optimal_params.set_index('well_id')['fraction_any_barcode']
        if len(well_performance) > 1:
            well_performance.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Performance by Well')
            axes[1,1].set_xlabel('Well ID')
            axes[1,1].set_ylabel('Any-Barcode Mapping')
            axes[1,1].tick_params(axis='x', rotation=45)

else:
    # No data - create informative empty plots
    for ax in axes.flat:
        ax.text(0.5, 0.5, 'No parameter search results found', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(snakemake.output[3], dpi=300, bbox_inches='tight')
plt.close()

print(f"Parameter search summary completed:")
print(f"- Processed {len(combined_optimal_params)} wells")
print(f"- Recommended parameters: peak_width={summary_stats.get('recommended_peak_width', 'N/A')}, threshold_reads={summary_stats.get('recommended_threshold_reads', 'N/A')}")

