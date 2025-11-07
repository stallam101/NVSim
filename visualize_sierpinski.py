

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

def load_latest_dataset():
    data_dir = Path("sierpinski_data")
    if not data_dir.exists():
        raise FileNotFoundError("No sierpinski_data directory found")
    

    run_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"📂 Using latest run: {latest_run_dir.name}")
    

    npy_files = list(latest_run_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {latest_run_dir}")
    
    latest_file = npy_files[0]
    print(f"📂 Loading dataset: {latest_file}")
    
    matrix = np.load(latest_file)
    print(f"📊 Dataset shape: {matrix.shape}")
    print(f"📈 Timing range: {np.min(matrix):.3f} - {np.max(matrix):.3f} ns")
    
    return matrix, latest_file

def analyze_dataset_statistics(matrix):
    valid_values = matrix[matrix > 0]
    
    stats = {
        'total_transitions': matrix.size,
        'successful_transitions': len(valid_values),
        'success_rate': len(valid_values) / matrix.size * 100,
        'min_latency': float(np.min(valid_values)) if len(valid_values) > 0 else 0,
        'max_latency': float(np.max(valid_values)) if len(valid_values) > 0 else 0,
        'mean_latency': float(np.mean(valid_values)) if len(valid_values) > 0 else 0,
        'std_latency': float(np.std(valid_values)) if len(valid_values) > 0 else 0,
        'unique_values': len(np.unique(valid_values)) if len(valid_values) > 0 else 0
    }
    
    print("\n📊 DATASET STATISTICS:")
    print(f"Total transitions: {stats['total_transitions']:,}")
    print(f"Successful: {stats['successful_transitions']:,} ({stats['success_rate']:.1f}%)")
    print(f"Timing range: {stats['min_latency']:.3f} - {stats['max_latency']:.3f} ns")
    print(f"Mean ± std: {stats['mean_latency']:.3f} ± {stats['std_latency']:.3f} ns")
    print(f"Unique timing values: {stats['unique_values']}")
    
    return stats

def create_sierpinski_visualizations(matrix, output_prefix="sierpinski_viz"):
    
    print("\n🎨 Creating Sierpinski visualizations...")
    

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', square=True, cbar_kws={'label': 'Write Latency (ns)'})
    plt.title('Sierpinski Gasket: ECC+RRAM Timing Matrix', fontsize=16)
    plt.xlabel('Destination Message (0-255)', fontsize=12)
    plt.ylabel('Source Message (0-255)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_prefix}_heatmap.png")
    plt.show()
    

    threshold = np.median(matrix[matrix > 0])
    binary_pattern = matrix > threshold
    
    plt.figure(figsize=(12, 10))
    plt.imshow(binary_pattern, cmap='binary', interpolation='nearest')
    plt.title(f'Sierpinski Gasket: Binary Pattern (threshold={threshold:.1f}ns)', fontsize=16)
    plt.xlabel('Destination Message (0-255)', fontsize=12)
    plt.ylabel('Source Message (0-255)', fontsize=12)
    plt.colorbar(label='Above Threshold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_binary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_prefix}_binary.png")
    plt.show()
    

    timing_bands = np.digitize(matrix, bins=np.percentile(matrix[matrix > 0], [0, 25, 50, 75, 100]))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(timing_bands, cmap='plasma', interpolation='nearest')
    plt.title('Sierpinski Gasket: Timing Bands (Quartiles)', fontsize=16)
    plt.xlabel('Destination Message (0-255)', fontsize=12)
    plt.ylabel('Source Message (0-255)', fontsize=12)
    plt.colorbar(label='Timing Band', ticks=[1, 2, 3, 4], 
                 format=plt.FuncFormatter(lambda x, p: ['Fastest', 'Fast', 'Slow', 'Slowest'][int(x)-1] if 1 <= x <= 4 else ''))
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_bands.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_prefix}_bands.png")
    plt.show()
    

    regions = [
        (0, 64, 0, 64, "Top-Left (0-63)"),
        (64, 128, 64, 128, "Center (64-127)"),
        (192, 256, 192, 256, "Bottom-Right (192-255)")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (r1, r2, c1, c2, title) in enumerate(regions):
        region = matrix[r1:r2, c1:c2]
        im = axes[i].imshow(region, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'Region: {title}', fontsize=12)
        axes[i].set_xlabel(f'Dest Msg ({c1}-{c2-1})')
        axes[i].set_ylabel(f'Src Msg ({r1}-{r2-1})')
        plt.colorbar(im, ax=axes[i], label='Latency (ns)')
    
    plt.suptitle('Sierpinski Gasket: Self-Similar Regions', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_regions.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_prefix}_regions.png")
    plt.show()

def detect_fractal_patterns(matrix):
    print("\n🔍 FRACTAL PATTERN ANALYSIS:")
    

    print("Checking for XOR-like patterns...")
    

    xor_correlations = []
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):

            for k in range(1, 16):
                if i^k < 256 and j^k < 256:
                    val1 = matrix[i, j]
                    val2 = matrix[i^k, j^k]
                    if val1 > 0 and val2 > 0:
                        correlation = abs(val1 - val2) / max(val1, val2)
                        xor_correlations.append(correlation)
    
    if xor_correlations:
        mean_xor_correlation = np.mean(xor_correlations)
        print(f"XOR pattern correlation: {mean_xor_correlation:.3f} (lower = more XOR-like)")
    

    print("Checking self-similarity...")
    

    region1 = matrix[0:64, 0:64]
    region2 = matrix[64:128, 64:128]
    region3 = matrix[128:192, 128:192]
    
    valid1 = region1[region1 > 0]
    valid2 = region2[region2 > 0]
    valid3 = region3[region3 > 0]
    
    if len(valid1) > 0 and len(valid2) > 0:
        correlation_12 = np.corrcoef(
            np.histogram(valid1, bins=20)[0],
            np.histogram(valid2, bins=20)[0]
        )[0, 1]
        print(f"Region 1-2 correlation: {correlation_12:.3f}")
    
    if len(valid2) > 0 and len(valid3) > 0:
        correlation_23 = np.corrcoef(
            np.histogram(valid2, bins=20)[0],
            np.histogram(valid3, bins=20)[0]
        )[0, 1]
        print(f"Region 2-3 correlation: {correlation_23:.3f}")

def main():
    try:
        print("🔬 Sierpinski Gasket Analysis and Visualization")
        print("=" * 60)
        

        matrix, dataset_file = load_latest_dataset()
        

        stats = analyze_dataset_statistics(matrix)
        

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"sierpinski_viz_{timestamp}"
        
        create_sierpinski_visualizations(matrix, output_prefix)
        

        detect_fractal_patterns(matrix)
        
        print(f"\n✅ Visualization complete!")
        print(f"📁 Output files: {output_prefix}_*.png")
        print(f"🎯 Look for triangular fractal patterns in the visualizations!")
        
        return matrix, stats
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()