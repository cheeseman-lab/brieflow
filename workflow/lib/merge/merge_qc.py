"""
Quality Control and Visualization for Stitched Well Outputs
Run this in a Jupyter notebook for interactive QC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import io, exposure
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

class StitchQC:
    def __init__(self, base_path, plate, well):
        """
        Initialize QC for a specific plate/well
        
        Parameters:
        -----------
        base_path : str or Path
            Base path to your analysis outputs
        plate : str/int
            Plate identifier
        well : str
            Well identifier (e.g., 'A01')
        """
        self.base_path = Path(base_path)
        self.plate = str(plate)
        self.well = well
        prefix = f"P-{plate}_W-{well}__"
        
        # Define expected file paths
        self.phenotype_image = self.base_path / "stitched_images" / f"{prefix}phenotype_stitched_image.npy"
        self.phenotype_mask = self.base_path / "stitched_masks" / f"{prefix}phenotype_stitched_mask.npy"
        self.phenotype_positions = self.base_path / "cell_positions" / f"{prefix}phenotype_cell_positions.parquet"
        self.phenotype_overlay = self.base_path / "overlays" / f"{prefix}phenotype_overlay.png"
        
        self.sbs_image = self.base_path / "stitched_images" / f"{prefix}sbs_stitched_image.npy"
        self.sbs_mask = self.base_path / "stitched_masks" /  f"{prefix}sbs_stitched_mask.npy"
        self.sbs_positions = self.base_path / "cell_positions" / f"{prefix}sbs_cell_positions.parquet"
        self.sbs_overlay = self.base_path / "overlays" / f"{prefix}sbs_overlay.png"
        
        print(f"Initialized QC for Plate {plate}, Well {well}")
        self.check_files()
    
    def check_files(self):
        """Check which output files exist"""
        files = {
            'Phenotype Image': self.phenotype_image,
            'Phenotype Mask': self.phenotype_mask,
            'Phenotype Positions': self.phenotype_positions,
            'Phenotype Overlay': self.phenotype_overlay,
            'SBS Image': self.sbs_image,
            'SBS Mask': self.sbs_mask,
            'SBS Positions': self.sbs_positions,
            'SBS Overlay': self.sbs_overlay,
        }
        
        print("\n=== File Status ===")
        for name, path in files.items():
            status = "✅ EXISTS" if path.exists() else "❌ MISSING"
            size = f"({path.stat().st_size / 1e6:.1f} MB)" if path.exists() else ""
            print(f"{name:20} {status} {size}")
    
    def view_overlays(self, figsize=(15, 6)):
        """Display phenotype and SBS overlays side by side"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Phenotype overlay
        if self.phenotype_overlay.exists():
            ph_overlay = io.imread(self.phenotype_overlay)
            axes[0].imshow(ph_overlay)
            axes[0].set_title(f'Phenotype Overlay\nPlate {self.plate}, Well {self.well}')
            axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'Phenotype\nOverlay\nMissing', 
                        ha='center', va='center', transform=axes[0].transAxes, fontsize=16)
            axes[0].set_title('Phenotype Overlay - MISSING')
        
        # SBS overlay  
        if self.sbs_overlay.exists():
            sbs_overlay = io.imread(self.sbs_overlay)
            axes[1].imshow(sbs_overlay)
            axes[1].set_title(f'SBS Overlay\nPlate {self.plate}, Well {self.well}')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'SBS\nOverlay\nMissing', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=16)
            axes[1].set_title('SBS Overlay - MISSING')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cell_positions(self):
        """Analyze cell position data and create summary plots"""
        
        # Load position data
        ph_pos = None
        sbs_pos = None
        
        if self.phenotype_positions.exists():
            ph_pos = pd.read_parquet(self.phenotype_positions)
            print(f"Phenotype: {len(ph_pos)} cells")
            
        if self.sbs_positions.exists():
            sbs_pos = pd.read_parquet(self.sbs_positions)
            print(f"SBS: {len(sbs_pos)} cells")
        
        if ph_pos is None and sbs_pos is None:
            print("No position data available")
            return
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cell Position Analysis - Plate {self.plate}, Well {self.well}', fontsize=16)
        
        # Plot 1: Cell counts by modality
        counts = []
        labels = []
        if ph_pos is not None:
            counts.append(len(ph_pos))
            labels.append('Phenotype')
        if sbs_pos is not None:
            counts.append(len(sbs_pos))
            labels.append('SBS')
        
        axes[0,0].bar(labels, counts, color=['skyblue', 'lightcoral'])
        axes[0,0].set_title('Cell Counts by Modality')
        axes[0,0].set_ylabel('Number of Cells')
        for i, count in enumerate(counts):
            axes[0,0].text(i, count + max(counts)*0.01, str(count), ha='center')
        
        # Plot 2: Cell area distributions
        if ph_pos is not None:
            axes[0,1].hist(ph_pos['area'], bins=50, alpha=0.7, label='Phenotype', color='skyblue')
        if sbs_pos is not None:
            axes[0,1].hist(sbs_pos['area'], bins=50, alpha=0.7, label='SBS', color='lightcoral')
        axes[0,1].set_title('Cell Area Distributions')
        axes[0,1].set_xlabel('Cell Area (pixels)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # Plot 3: Tile distribution (if available)
        if ph_pos is not None and 'tile' in ph_pos.columns:
            tile_counts = ph_pos['tile'].value_counts().sort_index()
            axes[0,2].bar(tile_counts.index, tile_counts.values, color='skyblue', alpha=0.7)
            axes[0,2].set_title('Phenotype Cells per Tile')
            axes[0,2].set_xlabel('Tile ID')
            axes[0,2].set_ylabel('Cell Count')
        else:
            axes[0,2].text(0.5, 0.5, 'No Tile\nData', ha='center', va='center', 
                          transform=axes[0,2].transAxes, fontsize=14)
            axes[0,2].set_title('Phenotype Tiles - No Data')
        
        # Plot 4: Spatial distribution - Phenotype
        if ph_pos is not None:
            scatter = axes[1,0].scatter(ph_pos['j'], ph_pos['i'], 
                                      c=ph_pos.get('tile', 0), s=1, alpha=0.6, cmap='tab10')
            axes[1,0].set_title('Phenotype Cell Positions')
            axes[1,0].set_xlabel('J (Column)')
            axes[1,0].set_ylabel('I (Row)')
            axes[1,0].invert_yaxis()  # Match image coordinates
            if 'tile' in ph_pos.columns:
                plt.colorbar(scatter, ax=axes[1,0], label='Tile ID')
        else:
            axes[1,0].text(0.5, 0.5, 'No Phenotype\nPosition Data', 
                          ha='center', va='center', transform=axes[1,0].transAxes, fontsize=14)
        
        # Plot 5: Spatial distribution - SBS
        if sbs_pos is not None:
            scatter = axes[1,1].scatter(sbs_pos['j'], sbs_pos['i'], 
                                      c=sbs_pos.get('tile', 0), s=1, alpha=0.6, cmap='tab10')
            axes[1,1].set_title('SBS Cell Positions')
            axes[1,1].set_xlabel('J (Column)')
            axes[1,1].set_ylabel('I (Row)')
            axes[1,1].invert_yaxis()  # Match image coordinates
            if 'tile' in sbs_pos.columns:
                plt.colorbar(scatter, ax=axes[1,1], label='Tile ID')
        else:
            axes[1,1].text(0.5, 0.5, 'No SBS\nPosition Data', 
                          ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
        
        # Plot 6: Position overlay comparison
        if ph_pos is not None and sbs_pos is not None:
            axes[1,2].scatter(ph_pos['j'], ph_pos['i'], s=1, alpha=0.5, 
                            label='Phenotype', color='blue')
            axes[1,2].scatter(sbs_pos['j'], sbs_pos['i'], s=1, alpha=0.5, 
                            label='SBS', color='red')
            axes[1,2].set_title('Position Overlay Comparison')
            axes[1,2].set_xlabel('J (Column)')
            axes[1,2].set_ylabel('I (Row)')
            axes[1,2].invert_yaxis()
            axes[1,2].legend()
        else:
            axes[1,2].text(0.5, 0.5, 'Missing Data\nfor Comparison', 
                          ha='center', va='center', transform=axes[1,2].transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        return ph_pos, sbs_pos
    
    def check_stitching_quality(self, sample_region=None):
        """
        Check stitching quality by examining raw images
        
        Parameters:
        -----------
        sample_region : tuple, optional
            (start_i, end_i, start_j, end_j) region to examine closely
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stitching Quality Check - Plate {self.plate}, Well {self.well}', fontsize=16)
        
        # Load and display phenotype image
        if self.phenotype_image.exists():
            ph_img = np.load(self.phenotype_image)
            print(f"Phenotype image shape: {ph_img.shape}")
            
            # Full image (downsampled for display)
            downsample = max(1, max(ph_img.shape) // 1000)  # Downsample for display
            ph_display = ph_img[::downsample, ::downsample]
            
            axes[0,0].imshow(ph_display, cmap='gray')
            axes[0,0].set_title(f'Phenotype Full Well\n(downsampled {downsample}x)')
            axes[0,0].axis('off')
            
            # Sample region
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                ph_sample = ph_img[start_i:end_i, start_j:end_j]
                axes[0,1].imshow(ph_sample, cmap='gray')
                axes[0,1].set_title(f'Phenotype Sample Region\n[{start_i}:{end_i}, {start_j}:{end_j}]')
                
                # Add rectangle to full image showing sample region
                rect = Rectangle((start_j//downsample, start_i//downsample), 
                               (end_j-start_j)//downsample, (end_i-start_i)//downsample,
                               linewidth=2, edgecolor='red', facecolor='none')
                axes[0,0].add_patch(rect)
            else:
                # Show center region
                h, w = ph_img.shape
                center_h, center_w = h//2, w//2
                size = min(500, min(h, w)//4)
                ph_center = ph_img[center_h-size:center_h+size, center_w-size:center_w+size]
                axes[0,1].imshow(ph_center, cmap='gray')
                axes[0,1].set_title('Phenotype Center Region')
            axes[0,1].axis('off')
        
        # Load and display SBS image
        if self.sbs_image.exists():
            sbs_img = np.load(self.sbs_image)
            print(f"SBS image shape: {sbs_img.shape}")
            
            # Full image (downsampled for display)  
            downsample = max(1, max(sbs_img.shape) // 1000)
            sbs_display = sbs_img[::downsample, ::downsample]
            
            axes[1,0].imshow(sbs_display, cmap='gray')
            axes[1,0].set_title(f'SBS Full Well\n(downsampled {downsample}x)')
            axes[1,0].axis('off')
            
            # Sample region
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                # Scale region for SBS (typically lower resolution)
                scale = min(sbs_img.shape[0]/ph_img.shape[0], 
                            sbs_img.shape[1]/ph_img.shape[1])
                sbs_start_i = int(start_i * scale)
                sbs_end_i = int(end_i * scale)
                sbs_start_j = int(start_j * scale)
                sbs_end_j = int(end_j * scale)
                
                sbs_sample = sbs_img[sbs_start_i:sbs_end_i, sbs_start_j:sbs_end_j]
                axes[1,1].imshow(sbs_sample, cmap='gray')
                axes[1,1].set_title(f'SBS Sample Region\n[{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}]')
            else:
                # Show center region
                h, w = sbs_img.shape
                center_h, center_w = h//2, w//2
                size = min(500, min(h, w)//4)
                sbs_center = sbs_img[center_h-size:center_h+size, center_w-size:center_w+size]
                axes[1,1].imshow(sbs_center, cmap='gray')
                axes[1,1].set_title('SBS Center Region')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_qc_report(self):
        """Generate a comprehensive QC report"""
        print(f"\n{'='*50}")
        print(f"QC REPORT: Plate {self.plate}, Well {self.well}")
        print(f"{'='*50}")
        
        # File existence check
        files_exist = {
            'phenotype_image': self.phenotype_image.exists(),
            'phenotype_mask': self.phenotype_mask.exists(), 
            'phenotype_positions': self.phenotype_positions.exists(),
            'sbs_image': self.sbs_image.exists(),
            'sbs_mask': self.sbs_mask.exists(),
            'sbs_positions': self.sbs_positions.exists(),
        }
        
        print(f"\n📁 FILE STATUS:")
        for name, exists in files_exist.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {name}")
        
        # Image analysis
        if files_exist['phenotype_image']:
            ph_img = np.load(self.phenotype_image)
            print(f"\n🔬 PHENOTYPE IMAGE:")
            print(f"  Shape: {ph_img.shape}")
            print(f"  Data type: {ph_img.dtype}")
            print(f"  Value range: {ph_img.min()} - {ph_img.max()}")
            print(f"  Size: {self.phenotype_image.stat().st_size / 1e6:.1f} MB")
        
        if files_exist['sbs_image']:
            sbs_img = np.load(self.sbs_image)
            print(f"\n🧬 SBS IMAGE:")
            print(f"  Shape: {sbs_img.shape}")
            print(f"  Data type: {sbs_img.dtype}")
            print(f"  Value range: {sbs_img.min()} - {sbs_img.max()}")
            print(f"  Size: {self.sbs_image.stat().st_size / 1e6:.1f} MB")
        
        # Cell analysis
        if files_exist['phenotype_positions']:
            ph_pos = pd.read_parquet(self.phenotype_positions)
            print(f"\n🔵 PHENOTYPE CELLS:")
            print(f"  Total cells: {len(ph_pos):,}")
            if 'tile' in ph_pos.columns:
                print(f"  Tiles represented: {ph_pos['tile'].nunique()}")
                print(f"  Cells per tile: {len(ph_pos) / ph_pos['tile'].nunique():.1f}")
            if 'area' in ph_pos.columns:
                print(f"  Mean cell area: {ph_pos['area'].mean():.1f} pixels")
        
        if files_exist['sbs_positions']:
            sbs_pos = pd.read_parquet(self.sbs_positions)
            print(f"\n🔴 SBS CELLS:")
            print(f"  Total cells: {len(sbs_pos):,}")
            if 'tile' in sbs_pos.columns:
                print(f"  Tiles represented: {sbs_pos['tile'].nunique()}")
                print(f"  Cells per tile: {len(sbs_pos) / sbs_pos['tile'].nunique():.1f}")
            if 'area' in sbs_pos.columns:
                print(f"  Mean cell area: {sbs_pos['area'].mean():.1f} pixels")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if not files_exist['phenotype_image'] or not files_exist['sbs_image']:
            print("  ⚠️  Missing stitched images - check stitching pipeline")
        
        if files_exist['phenotype_positions'] and files_exist['sbs_positions']:
            ph_count = len(pd.read_parquet(self.phenotype_positions))
            sbs_count = len(pd.read_parquet(self.sbs_positions))
            ratio = ph_count / sbs_count if sbs_count > 0 else 0
            
            if ratio < 0.5 or ratio > 2.0:
                print(f"  ⚠️  Large cell count difference (Ph: {ph_count:,}, SBS: {sbs_count:,})")
                print("      Check segmentation parameters or imaging quality")
            else:
                print(f"  ✅ Cell counts reasonable (Ph: {ph_count:,}, SBS: {sbs_count:,})")
        
        if not files_exist['phenotype_mask'] or not files_exist['sbs_mask']:
            print("  ⚠️  Missing stitched masks - cell merging may fail")
        
        print(f"\n{'='*50}")


# Usage functions
def quick_qc(base_path, plate, well):
    """Quick QC check for a single well"""
    qc = StitchQC(base_path, plate, well)
    qc.view_overlays()
    return qc.analyze_cell_positions()

def batch_qc_report(base_path, plate_wells):
    """
    Generate QC reports for multiple wells
    
    Parameters:
    -----------
    base_path : str
        Path to analysis outputs
    plate_wells : list of tuples
        [(plate1, well1), (plate2, well2), ...]
    """
    print("BATCH QC REPORT")
    print("="*60)
    
    summary = []
    
    for plate, well in plate_wells:
        qc = StitchQC(base_path, plate, well)
        
        # Quick file check
        ph_exists = qc.phenotype_positions.exists()
        sbs_exists = qc.sbs_positions.exists()
        
        ph_count = 0
        sbs_count = 0
        
        if ph_exists:
            ph_count = len(pd.read_parquet(qc.phenotype_positions))
        if sbs_exists:
            sbs_count = len(pd.read_parquet(qc.sbs_positions))
        
        summary.append({
            'plate': plate,
            'well': well,
            'ph_exists': ph_exists,
            'sbs_exists': sbs_exists,
            'ph_cells': ph_count,
            'sbs_cells': sbs_count,
            'status': 'OK' if ph_exists and sbs_exists and ph_count > 0 and sbs_count > 0 else 'ISSUE'
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    return summary_df

# Example usage:
"""
# Single well QC
qc = StitchQC('/path/to/analysis/merge', plate=1, well='A01')
qc.view_overlays()
ph_pos, sbs_pos = qc.analyze_cell_positions()
qc.create_qc_report()

# Check specific region for stitching artifacts
qc.check_stitching_quality(sample_region=(5000, 6000, 8000, 9000))

# Batch QC
wells_to_check = [(1, 'A01'), (1, 'A02'), (1, 'B01')]
summary = batch_qc_report('/path/to/analysis/merge', wells_to_check)
"""