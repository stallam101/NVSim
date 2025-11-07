#!/usr/bin/env python3
"""
Check if timing parameters create a clear trimodal distribution (NONE/UNIPOLAR/BIPOLAR).
"""

import numpy as np
import matplotlib.pyplot as plt
from galois import GF2
from pathlib import Path

class TrimodalDistributionChecker:
    def __init__(self):
        # Load P matrix and b vector
        self.P = self._load_p_matrix()
        self.b = self._load_b_vector()
        
        # Load latest dataset
        self.timing_matrix = self._load_latest_dataset()
        
    def _load_p_matrix(self) -> GF2:
        try:
            P_np = np.loadtxt("P_matrix.csv", delimiter=',', dtype=np.uint8)
            return GF2(P_np)
        except Exception as e:
            raise RuntimeError(f"Failed to load P matrix: {e}")
    
    def _load_b_vector(self) -> GF2:
        try:
            b_np = np.loadtxt("b_vector.csv", delimiter=',', dtype=np.uint8)
            return GF2(b_np)
        except Exception as e:
            raise RuntimeError(f"Failed to load b vector: {e}")
    
    def _load_latest_dataset(self):
        data_dir = Path("sierpinski_data")
        run_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        latest_run = sorted(run_dirs)[-1]
        
        npy_files = list(latest_run.glob("*.npy"))
        timing_matrix = np.load(npy_files[0])
        print(f"Loaded timing matrix from: {latest_run}")
        return timing_matrix
    
    def encode_message(self, message: int) -> int:
        """Encode message using Tyler's ECC specification"""
        if not (0 <= message <= 255):
            raise ValueError(f"Message must be 8-bit (0-255), got {message}")
        
        message_bits = GF2([(message >> i) & 1 for i in range(16)])
        parity_bits = message_bits @ self.P + self.b
        codeword_bits = np.concatenate([message_bits, parity_bits])
        codeword = sum(int(bit) << i for i, bit in enumerate(codeword_bits))
        return codeword
    
    def classify_transition(self, msg0: int, msg1: int) -> str:
        """Classify transition as NONE/UNIPOLAR/BIPOLAR"""
        cw0 = self.encode_message(msg0)
        cw1 = self.encode_message(msg1)
        
        cw_up = (~cw0) & cw1
        cw_down = cw0 & (~cw1)
        cw_has_up = cw_up != 0
        cw_has_down = cw_down != 0
        
        if not cw_has_up and not cw_has_down:
            return "NONE"
        elif cw_has_up and cw_has_down:
            return "BIPOLAR"
        else:
            return "UNIPOLAR"
    
    def analyze_timing_by_transition_type(self):
        """Analyze timing distribution by transition type"""
        print("🔬 Analyzing Timing Distribution by Transition Type")
        print("=" * 60)
        
        # Collect timing data by transition type
        timings_by_type = {"NONE": [], "UNIPOLAR": [], "BIPOLAR": []}
        
        # Sample a subset for analysis (full 256x256 would be slow)
        sample_size = 5000
        np.random.seed(42)
        
        for _ in range(sample_size):
            msg0 = np.random.randint(0, 256)
            msg1 = np.random.randint(0, 256)
            
            transition_type = self.classify_transition(msg0, msg1)
            timing = self.timing_matrix[msg0, msg1]
            
            if timing > 0:  # Valid timing data
                timings_by_type[transition_type].append(timing)
        
        # Calculate statistics for each type
        print(f"Transition type statistics (from {sample_size} samples):")
        print()
        
        stats = {}
        for t_type in ["NONE", "UNIPOLAR", "BIPOLAR"]:
            timings = timings_by_type[t_type]
            if timings:
                stats[t_type] = {
                    'count': len(timings),
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'median': np.median(timings)
                }
                
                print(f"{t_type}:")
                print(f"  Count: {stats[t_type]['count']}")
                print(f"  Mean:  {stats[t_type]['mean']:.0f} ns")
                print(f"  Std:   {stats[t_type]['std']:.0f} ns")
                print(f"  Range: {stats[t_type]['min']:.0f} - {stats[t_type]['max']:.0f} ns")
                print(f"  Median: {stats[t_type]['median']:.0f} ns")
                print()
            else:
                stats[t_type] = None
                print(f"{t_type}: No data")
                print()
        
        # Check for clear separation
        print("Timing separation analysis:")
        if stats["NONE"] and stats["UNIPOLAR"] and stats["BIPOLAR"]:
            none_mean = stats["NONE"]["mean"]
            unipolar_mean = stats["UNIPOLAR"]["mean"]
            bipolar_mean = stats["BIPOLAR"]["mean"]
            
            print(f"  NONE mean:     {none_mean:.0f} ns")
            print(f"  UNIPOLAR mean: {unipolar_mean:.0f} ns")
            print(f"  BIPOLAR mean:  {bipolar_mean:.0f} ns")
            print()
            
            # Calculate separations
            none_unipolar_gap = abs(none_mean - unipolar_mean)
            none_bipolar_gap = abs(none_mean - bipolar_mean)
            unipolar_bipolar_gap = abs(unipolar_mean - bipolar_mean)
            
            print(f"  NONE ↔ UNIPOLAR gap:  {none_unipolar_gap:.0f} ns")
            print(f"  NONE ↔ BIPOLAR gap:   {none_bipolar_gap:.0f} ns")
            print(f"  UNIPOLAR ↔ BIPOLAR gap: {unipolar_bipolar_gap:.0f} ns")
            print()
            
            # Check for trimodal separation
            min_gap = min(none_unipolar_gap, none_bipolar_gap, unipolar_bipolar_gap)
            if min_gap > 10000:  # 10μs minimum separation
                print("✅ Clear trimodal separation detected!")
            else:
                print("⚠️  Weak trimodal separation - peaks may overlap")
        
        return timings_by_type, stats
    
    def plot_timing_distributions(self, timings_by_type, output_file="timing_distribution_by_type.png"):
        """Create histogram plots showing timing distributions by type"""
        print(f"\n📊 Creating timing distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual histograms
        colors = {'NONE': 'red', 'UNIPOLAR': 'blue', 'BIPOLAR': 'green'}
        
        for i, t_type in enumerate(["NONE", "UNIPOLAR", "BIPOLAR"]):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            timings = timings_by_type[t_type]
            if timings:
                ax.hist(timings, bins=50, alpha=0.7, color=colors[t_type], edgecolor='black')
                ax.set_title(f'{t_type} Transitions\n({len(timings)} samples)')
                ax.set_xlabel('Write Latency (ns)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {t_type} data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{t_type} Transitions\n(No data)')
        
        # Combined histogram
        ax = axes[1, 1]
        all_timings = []
        all_labels = []
        all_colors = []
        
        for t_type in ["NONE", "UNIPOLAR", "BIPOLAR"]:
            if timings_by_type[t_type]:
                all_timings.append(timings_by_type[t_type])
                all_labels.append(f'{t_type} ({len(timings_by_type[t_type])})')
                all_colors.append(colors[t_type])
        
        if all_timings:
            ax.hist(all_timings, bins=50, alpha=0.6, label=all_labels, color=all_colors)
            ax.set_title('Combined Distribution\n(All Transition Types)')
            ax.set_xlabel('Write Latency (ns)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.show()

def main():
    print("🔬 Trimodal Distribution Checker")
    print("Analyzing timing distributions for NONE/UNIPOLAR/BIPOLAR transitions")
    print("=" * 70)
    
    try:
        checker = TrimodalDistributionChecker()
        
        # Analyze timing by transition type
        timings_by_type, stats = checker.analyze_timing_by_transition_type()
        
        # Create distribution plots
        checker.plot_timing_distributions(timings_by_type)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())