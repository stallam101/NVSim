#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SierpinskiVariantVisualizer:
    
    def __init__(self, data_dir: str = "sierpinski_4_variants"):
        self.data_dir = Path(data_dir)
        self.variants = ['byte0_g0', 'byte0_g1', 'byte1_g0', 'byte1_g1']
        self.variant_descriptions = {
            'byte0_g0': 'Byte Index 0, G=0\nVary bits 0-7, bits 8-15 = 0x00',
            'byte0_g1': 'Byte Index 0, G=1\nVary bits 0-7, bits 8-15 = 0xFF',
            'byte1_g0': 'Byte Index 1, G=0\nVary bits 8-15, bits 0-7 = 0x00', 
            'byte1_g1': 'Byte Index 1, G=1\nVary bits 8-15, bits 0-7 = 0xFF'
        }
    
    def find_latest_files(self) -> Dict[str, Dict[str, Path]]:
        """Find the latest files for each variant"""
        variant_files = {}
        
        for variant in self.variants:
            files = {}
            
            # Find latest latency matrix
            latency_files = list(self.data_dir.glob(f"{variant}_latency_*.npy"))
            if latency_files:
                files['latency'] = max(latency_files, key=lambda x: x.stat().st_mtime)
            
            # Find latest transition matrix
            transition_files = list(self.data_dir.glob(f"{variant}_transitions_*.npy"))
            if transition_files:
                files['transitions'] = max(transition_files, key=lambda x: x.stat().st_mtime)
            
            # Find latest metadata
            metadata_files = list(self.data_dir.glob(f"{variant}_metadata_*.json"))
            if metadata_files:
                files['metadata'] = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            if files:
                variant_files[variant] = files
                logger.info(f"Found files for {variant}: {len(files)} files")
            else:
                logger.warning(f"No files found for variant {variant}")
        
        return variant_files
    
    def load_variant_data(self, variant_files: Dict[str, Dict[str, Path]]) -> Dict:
        """Load all variant data"""
        variant_data = {}
        
        for variant, files in variant_files.items():
            data = {}
            
            try:
                # Load latency matrix
                if 'latency' in files:
                    data['latency_matrix'] = np.load(files['latency'])
                    logger.info(f"Loaded {variant} latency matrix: {data['latency_matrix'].shape}")
                
                # Load transition matrix
                if 'transitions' in files:
                    data['transition_matrix'] = np.load(files['transitions'])
                    logger.info(f"Loaded {variant} transition matrix: {data['transition_matrix'].shape}")
                
                # Load metadata
                if 'metadata' in files:
                    with open(files['metadata'], 'r') as f:
                        data['metadata'] = json.load(f)
                
                variant_data[variant] = data
                
            except Exception as e:
                logger.error(f"Failed to load {variant}: {e}")
        
        return variant_data
    
    def create_comparison_visualization(self, variant_data: Dict, 
                                      visualization_type: str = "transitions") -> None:
        """Create side-by-side comparison of all 4 variants"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"4 Sierpinski Variants - {visualization_type.title()} Classification", 
                     fontsize=16, fontweight='bold')
        
        # Arrange variants in 2x2 grid
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        variant_order = ['byte0_g0', 'byte0_g1', 'byte1_g0', 'byte1_g1']
        
        for idx, variant in enumerate(variant_order):
            row, col = positions[idx]
            ax = axes[row, col]
            
            if variant not in variant_data:
                ax.text(0.5, 0.5, f'No data for\n{variant}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(self.variant_descriptions[variant])
                continue
            
            data = variant_data[variant]
            
            if visualization_type == "transitions" and 'transition_matrix' in data:
                matrix = data['transition_matrix']
                
                # Create custom colormap for transition types
                # 0=NONE (black), 1=UNIPOLAR (blue), 2=BIPOLAR (red), -1=FAILED (gray)
                cmap = plt.cm.colors.ListedColormap(['black', 'blue', 'red', 'gray'])
                bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
                norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
                
                im = ax.imshow(matrix, cmap=cmap, norm=norm)
                
                # Add colorbar for this subplot
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks([-1, 0, 1, 2])
                cbar.set_ticklabels(['FAILED', 'NONE', 'UNIPOLAR', 'BIPOLAR'])
                
            elif visualization_type == "latency" and 'latency_matrix' in data:
                matrix = data['latency_matrix']
                
                # Handle NaN values and create reasonable color scale
                valid_values = matrix[~np.isnan(matrix) & (matrix > 0)]
                if len(valid_values) > 0:
                    vmin, vmax = np.percentile(valid_values, [5, 95])
                    im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Latency (ns)')
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes)
            
            # Set labels and title
            ax.set_title(self.variant_descriptions[variant], fontsize=10, fontweight='bold')
            ax.set_xlabel('Destination Byte')
            ax.set_ylabel('Source Byte') 
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Set ticks
            ax.set_xticks(np.arange(0, 256, 32))
            ax.set_yticks(np.arange(0, 256, 32))
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sierpinski_4_variants_{visualization_type}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved comparison visualization: {filename}")
    
    def analyze_puncturing_patterns(self, variant_data: Dict) -> None:
        """Analyze puncturing patterns across variants"""
        
        print("\n🔍 Puncturing Pattern Analysis")
        print("=" * 50)
        
        for variant in self.variants:
            if variant not in variant_data or 'transition_matrix' not in variant_data[variant]:
                print(f"{variant}: No data available")
                continue
            
            matrix = variant_data[variant]['transition_matrix']
            
            # Count transition types
            total_valid = np.sum(matrix >= 0)  # Exclude failed (-1) transitions
            none_count = np.sum(matrix == 0)
            unipolar_count = np.sum(matrix == 1) 
            bipolar_count = np.sum(matrix == 2)
            failed_count = np.sum(matrix == -1)
            
            # Calculate percentages
            if total_valid > 0:
                none_pct = (none_count / total_valid) * 100
                unipolar_pct = (unipolar_count / total_valid) * 100
                bipolar_pct = (bipolar_count / total_valid) * 100
                
                # Puncturing rate: UNIPOLAR → BIPOLAR conversion
                puncturing_rate = bipolar_pct / (unipolar_pct + bipolar_pct) * 100 if (unipolar_pct + bipolar_pct) > 0 else 0
            else:
                none_pct = unipolar_pct = bipolar_pct = puncturing_rate = 0
            
            print(f"\n{variant} ({self.variant_descriptions[variant].split()[0:3]}):")
            print(f"  Total valid: {total_valid:,}/65,536 ({total_valid/655.36:.1f}%)")
            print(f"  NONE:     {none_count:,} ({none_pct:.1f}%)")
            print(f"  UNIPOLAR: {unipolar_count:,} ({unipolar_pct:.1f}%)")
            print(f"  BIPOLAR:  {bipolar_count:,} ({bipolar_pct:.1f}%)")
            print(f"  FAILED:   {failed_count:,}")
            print(f"  Puncturing rate: {puncturing_rate:.1f}%")
            
            # Analyze distribution patterns
            if matrix.size > 0:
                # Check for Sierpinski-like patterns by looking at quadrant symmetry
                h, w = matrix.shape
                if h >= 128 and w >= 128:
                    q1 = matrix[:h//2, :w//2]  # Top-left
                    q2 = matrix[:h//2, w//2:]  # Top-right  
                    q3 = matrix[h//2:, :w//2]  # Bottom-left
                    q4 = matrix[h//2:, w//2:]  # Bottom-right
                    
                    # Simple symmetry check
                    symmetry_score = (
                        np.mean(q1 == q2) + np.mean(q1 == q3) + 
                        np.mean(q2 == q4) + np.mean(q3 == q4)
                    ) / 4
                    
                    print(f"  Quadrant symmetry: {symmetry_score:.3f} (1.0 = perfect)")
    
    def compare_latency_distributions(self, variant_data: Dict) -> None:
        """Compare latency distributions across variants"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Latency Distributions Across 4 Variants", fontsize=16, fontweight='bold')
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        variant_order = ['byte0_g0', 'byte0_g1', 'byte1_g0', 'byte1_g1']
        
        all_latencies = []  # For overall comparison
        
        for idx, variant in enumerate(variant_order):
            row, col = positions[idx]
            ax = axes[row, col]
            
            if variant not in variant_data or 'latency_matrix' not in variant_data[variant]:
                ax.text(0.5, 0.5, f'No data for\n{variant}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(self.variant_descriptions[variant].split('\n')[0])
                continue
            
            matrix = variant_data[variant]['latency_matrix']
            valid_latencies = matrix[~np.isnan(matrix) & (matrix > 0)]
            
            if len(valid_latencies) > 0:
                # Create histogram
                ax.hist(valid_latencies, bins=50, alpha=0.7, density=True)
                ax.set_title(self.variant_descriptions[variant].split('\n')[0])
                ax.set_xlabel('Latency (ns)')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f"Mean: {np.mean(valid_latencies):.1f}ns\n"
                stats_text += f"Std: {np.std(valid_latencies):.1f}ns\n"
                stats_text += f"Valid: {len(valid_latencies):,}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                all_latencies.extend(valid_latencies)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sierpinski_4_variants_latency_distributions_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved latency distribution comparison: {filename}")
        
        # Overall statistics
        if all_latencies:
            print(f"\n📊 Overall Latency Statistics (All Variants)")
            print(f"Total valid measurements: {len(all_latencies):,}")
            print(f"Mean latency: {np.mean(all_latencies):.1f} ns")
            print(f"Latency range: {np.min(all_latencies):.1f} - {np.max(all_latencies):.1f} ns")
            print(f"Standard deviation: {np.std(all_latencies):.1f} ns")


def main():
    import sys
    
    try:
        print("🎨 Sierpinski 4 Variants Visualizer")
        print("=" * 40)
        
        visualizer = SierpinskiVariantVisualizer()
        
        # Find and load data
        print("📂 Searching for variant data files...")
        variant_files = visualizer.find_latest_files()
        
        if not variant_files:
            print("❌ No variant data files found!")
            print("Run sierpinski_4_variants.py --generate first")
            return 1
        
        print(f"📊 Loading data for {len(variant_files)} variants...")
        variant_data = visualizer.load_variant_data(variant_files)
        
        if not variant_data:
            print("❌ Failed to load any variant data!")
            return 1
        
        if len(sys.argv) > 1:
            viz_type = sys.argv[1]
            
            if viz_type == "--transitions":
                print("🎯 Creating transition classification comparison...")
                visualizer.create_comparison_visualization(variant_data, "transitions")
                
            elif viz_type == "--latency":
                print("⏱️ Creating latency comparison...")
                visualizer.create_comparison_visualization(variant_data, "latency")
                
            elif viz_type == "--distributions":
                print("📈 Creating latency distribution comparison...")
                visualizer.compare_latency_distributions(variant_data)
                
            elif viz_type == "--analysis":
                print("🔍 Performing puncturing pattern analysis...")
                visualizer.analyze_puncturing_patterns(variant_data)
                
            elif viz_type == "--all":
                print("🎨 Creating all visualizations...")
                visualizer.create_comparison_visualization(variant_data, "transitions")
                visualizer.create_comparison_visualization(variant_data, "latency") 
                visualizer.compare_latency_distributions(variant_data)
                visualizer.analyze_puncturing_patterns(variant_data)
                
            else:
                print(f"❌ Unknown visualization type: {viz_type}")
                return 1
        
        else:
            print("📋 Available visualizations:")
            print("  --transitions: Transition type classification comparison")
            print("  --latency: Latency heat map comparison")
            print("  --distributions: Latency distribution histograms")
            print("  --analysis: Puncturing pattern analysis")
            print("  --all: Generate all visualizations")
            print()
            print("Example: python visualize_4_variants.py --transitions")
            return 0
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())