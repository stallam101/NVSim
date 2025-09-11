#!/usr/bin/env python3
"""
Gumbel Distribution Visualization for Phase 3 Word-Level Modeling
Creates graphs to visualize the Gumbel distribution emergence
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import subprocess
import sys
from scipy import stats
import math

def run_nvsim_sample():
    """Run a single NVSim sample and extract SET latency"""
    try:
        result = subprocess.run(['./nvsim', 'sample_word_level_worst_case.cfg'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Extract SET Latency from output
            set_match = re.search(r'- SET Latency\s*=\s*([0-9.]+)ns', result.stdout)
            if set_match:
                return float(set_match.group(1))
        
        return None
    except Exception as e:
        print(f"Error running sample: {e}")
        return None

def collect_samples(num_samples=100):
    """Collect word-level completion time samples"""
    print(f"Collecting {num_samples} word-level completion time samples...")
    samples = []
    
    for i in range(num_samples):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_samples} samples")
            
        latency = run_nvsim_sample()
        if latency is not None:
            samples.append(latency)
        else:
            print(f"  Warning: Sample {i+1} failed")
    
    print(f"Successfully collected {len(samples)} samples")
    return samples

def fit_gumbel_parameters(samples):
    """Fit Gumbel distribution parameters"""
    n = len(samples)
    mean = sum(samples) / n
    variance = sum((x - mean)**2 for x in samples) / (n - 1)
    stddev = math.sqrt(variance)
    
    # Gumbel parameter estimation
    euler_gamma = 0.5772156649
    pi = 3.14159265359
    
    beta_est = stddev * math.sqrt(6) / pi
    mu_est = mean - euler_gamma * beta_est
    
    return mu_est, beta_est, mean, stddev

def gumbel_pdf(x, mu, beta):
    """Gumbel probability density function"""
    z = (x - mu) / beta
    return (1/beta) * np.exp(-(z + np.exp(-z)))

def gumbel_cdf(x, mu, beta):
    """Gumbel cumulative distribution function"""
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))

def create_visualizations(samples):
    """Create comprehensive visualization of Gumbel distribution emergence"""
    
    # Fit parameters
    mu, beta, mean, stddev = fit_gumbel_parameters(samples)
    
    print(f"\nFitted Gumbel parameters:")
    print(f"  Location (μ): {mu:.3f} ns")
    print(f"  Scale (β): {beta:.3f} ns")
    print(f"  Sample mean: {mean:.3f} ns")
    print(f"  Sample std dev: {stddev:.3f} ns")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 3 Word-Level Modeling: Gumbel Distribution Emergence\n'
                f'MAX of 64 IID SET Operations (PCRAM)', fontsize=16, fontweight='bold')
    
    # 1. Histogram with theoretical PDF overlay
    ax1.hist(samples, bins=20, density=True, alpha=0.7, color='skyblue', 
             edgecolor='black', label='Empirical Data')
    
    # Theoretical Gumbel PDF
    x_range = np.linspace(min(samples) - 10, max(samples) + 10, 1000)
    theoretical_pdf = gumbel_pdf(x_range, mu, beta)
    ax1.plot(x_range, theoretical_pdf, 'r-', linewidth=2, 
             label=f'Theoretical Gumbel(μ={mu:.1f}, β={beta:.1f})')
    
    ax1.set_xlabel('Word Completion Time (ns)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Histogram vs Theoretical Gumbel PDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Empirical vs Theoretical CDF
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    theoretical_cdf = gumbel_cdf(sorted_samples, mu, beta)
    
    ax2.plot(sorted_samples, empirical_cdf, 'bo-', markersize=4, 
             label='Empirical CDF', alpha=0.7)
    ax2.plot(sorted_samples, theoretical_cdf, 'r-', linewidth=2, 
             label='Theoretical Gumbel CDF')
    ax2.plot([min(sorted_samples), max(sorted_samples)], 
             [min(sorted_samples), max(sorted_samples)], 'k--', alpha=0.5)
    
    ax2.set_xlabel('Word Completion Time (ns)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Empirical vs Theoretical CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    # Generate theoretical quantiles
    n = len(samples)
    p_values = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = mu - beta * np.log(-np.log(p_values))
    
    ax3.scatter(theoretical_quantiles, sorted_samples, alpha=0.7, color='green')
    
    # Perfect fit line
    min_val = min(min(theoretical_quantiles), min(sorted_samples))
    max_val = max(max(theoretical_quantiles), max(sorted_samples))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Fit')
    
    ax3.set_xlabel('Theoretical Gumbel Quantiles (ns)')
    ax3.set_ylabel('Empirical Quantiles (ns)')
    ax3.set_title('Q-Q Plot: Empirical vs Theoretical Quantiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Calculate R-squared for Q-Q plot
    correlation = np.corrcoef(theoretical_quantiles, sorted_samples)[0, 1]
    r_squared = correlation ** 2
    ax3.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Pulse Count Analysis
    # Convert latencies back to pulse counts (approximate)
    base_latency = 2.584  # Approximate base latency
    pulse_duration = 10.0  # ns per pulse
    pulse_counts = [(s - base_latency) / pulse_duration for s in samples]
    
    # Create pulse count histogram
    ax4.hist(pulse_counts, bins=range(0, 13), alpha=0.7, color='orange', 
             edgecolor='black', label='Observed Pulse Counts')
    
    # Show expected distribution (truncated normal)
    pulse_mean = 4.2
    pulse_std = 1.5
    pulse_min, pulse_max = 1, 12
    
    # Generate theoretical pulse count distribution
    x_pulses = np.arange(1, 13)
    theoretical_pulses = []
    for p in x_pulses:
        # Truncated normal probability
        prob = stats.norm.pdf(p, pulse_mean, pulse_std)
        theoretical_pulses.append(prob)
    
    # Normalize to match histogram scale
    theoretical_pulses = np.array(theoretical_pulses)
    theoretical_pulses = theoretical_pulses / sum(theoretical_pulses) * len(samples)
    
    ax4.plot(x_pulses, theoretical_pulses, 'ro-', linewidth=2, markersize=6,
             label=f'Expected (μ={pulse_mean}, σ={pulse_std})')
    
    ax4.set_xlabel('Pulse Count per Cell')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Individual Cell Pulse Count Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 13))
    
    # Add statistical summary text
    stats_text = f"""Statistical Summary:
    Samples: {len(samples)}
    Mean: {mean:.2f} ns
    Std Dev: {stddev:.2f} ns
    Range: {min(samples):.1f} - {max(samples):.1f} ns
    
    Gumbel Fit:
    μ (location): {mu:.2f} ns
    β (scale): {beta:.2f} ns
    
    Fit Quality:
    Q-Q R²: {r_squared:.4f}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save the plot
    output_file = 'gumbel_validation_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return mu, beta, r_squared

def main():
    print("=== Gumbel Distribution Visualization ===")
    print("Creating visual validation of Phase 3 word-level modeling...")
    
    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        import scipy.stats
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Install with: pip install matplotlib scipy")
        sys.exit(1)
    
    # Check if nvsim and config exist
    try:
        subprocess.run(['ls', 'sample_word_level_worst_case.cfg'], check=True, capture_output=True)
    except:
        print("ERROR: sample_word_level_worst_case.cfg not found!")
        sys.exit(1)
    
    # Collect samples
    samples = collect_samples(75)  # More samples for better visualization
    
    if len(samples) < 20:
        print("ERROR: Need at least 20 samples for visualization")
        sys.exit(1)
    
    # Create visualizations
    mu, beta, r_squared = create_visualizations(samples)
    
    # Final assessment
    print(f"\n=== Visualization Results ===")
    print(f"✅ Gumbel parameters: μ={mu:.2f}ns, β={beta:.2f}ns")
    print(f"✅ Q-Q plot fit quality: R²={r_squared:.4f}")
    
    if r_squared > 0.95:
        print("🎯 Excellent fit to Gumbel distribution!")
    elif r_squared > 0.90:
        print("✅ Good fit to Gumbel distribution!")
    else:
        print("⚠️  Moderate fit - may need more samples")
    
    print(f"\nVisualization complete! Check 'gumbel_validation_plots.png'")

if __name__ == "__main__":
    main()