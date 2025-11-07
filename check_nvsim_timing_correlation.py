#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from sierpinski_test import SierpinskiTester
from enum import Enum, auto
import matplotlib.pyplot as plt
import pandas as pd

class TransitionClass(Enum):
    NONE = auto()
    UNIPOLAR = auto() 
    BIPOLAR = auto()

def load_latest_timing_data():
    """Load the latest NVSim timing dataset."""
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
    
    timing_matrix = np.load(npy_files[0])
    print(f"📊 Timing matrix shape: {timing_matrix.shape}")
    print(f"📈 Timing range: {np.min(timing_matrix):.0f} - {np.max(timing_matrix):.0f} ns")
    
    return timing_matrix

def analyze_timing_vs_transitions():
    """
    Correlate actual NVSim timing data with bit transition analysis.
    """
    print('🔍 Correlating NVSim Timing Data with Transition Types')
    print('=' * 70)
    
    # Load NVSim timing data
    timing_matrix = load_latest_timing_data()
    
    # Initialize ECC encoder
    tester = SierpinskiTester()
    
    # Storage for analysis
    results = []
    
    print("\n📊 Analyzing all 65,536 transitions...")
    
    # Analyze every transition
    for src_msg in range(256):
        if src_msg % 64 == 0:
            print(f'  Processing src_msg {src_msg}/256...')
            
        for dst_msg in range(256):
            # Get actual NVSim timing for this transition
            timing_value = timing_matrix[src_msg, dst_msg]
            
            # Generate codewords and analyze transitions
            cw0 = tester.encode_message(src_msg)
            cw1 = tester.encode_message(dst_msg)
            
            # Analyze message transitions (8-bit)
            msg_up = (~src_msg) & dst_msg
            msg_down = src_msg & (~dst_msg)
            msg_has_up = msg_up != 0
            msg_has_down = msg_down != 0
            
            if msg_has_up and msg_has_down:
                msg_class = TransitionClass.BIPOLAR
            elif msg_has_up or msg_has_down:
                msg_class = TransitionClass.UNIPOLAR
            else:
                msg_class = TransitionClass.NONE
                
            # Analyze codeword transitions (21-bit with parity)
            cw_up = (~cw0) & cw1
            cw_down = cw0 & (~cw1)
            cw_has_up = cw_up != 0
            cw_has_down = cw_down != 0
            
            if cw_has_up and cw_has_down:
                cw_class = TransitionClass.BIPOLAR
            elif cw_has_up or cw_has_down:
                cw_class = TransitionClass.UNIPOLAR
            else:
                cw_class = TransitionClass.NONE
            
            # Check for puncturing
            is_puncture = (msg_class == TransitionClass.UNIPOLAR and 
                          cw_class == TransitionClass.BIPOLAR)
            
            # Count transitions
            msg_transition_count = bin(msg_up | msg_down).count('1')
            cw_transition_count = bin(cw_up | cw_down).count('1')
            
            results.append({
                'src_msg': src_msg,
                'dst_msg': dst_msg,
                'timing_ns': timing_value,
                'msg_class': msg_class,
                'cw_class': cw_class,
                'is_puncture': is_puncture,
                'msg_transitions': msg_transition_count,
                'cw_transitions': cw_transition_count
            })
    
    df = pd.DataFrame(results)
    
    # Analysis
    print(f"\n📋 CORRELATION ANALYSIS:")
    
    # Filter valid timing data
    valid_df = df[df['timing_ns'] > 0]
    print(f"Valid timing entries: {len(valid_df):,} / {len(df):,}")
    
    # Show timing distribution by transition class
    print(f"\n📊 TIMING BY CODEWORD TRANSITION CLASS:")
    for trans_class in [TransitionClass.NONE, TransitionClass.UNIPOLAR, TransitionClass.BIPOLAR]:
        class_data = valid_df[valid_df['cw_class'] == trans_class]
        if len(class_data) > 0:
            timings = class_data['timing_ns']
            print(f"{trans_class.name}:")
            print(f"  Count: {len(class_data):,}")
            print(f"  Timing range: {np.min(timings):.0f} - {np.max(timings):.0f} ns")
            print(f"  Mean ± Std: {np.mean(timings):.0f} ± {np.std(timings):.0f} ns")
    
    # Check puncturing correlation
    punctured = valid_df[valid_df['is_puncture']]
    not_punctured = valid_df[~valid_df['is_puncture']]
    
    print(f"\n🎯 PUNCTURING vs TIMING:")
    if len(punctured) > 0:
        print(f"Punctured transitions:")
        print(f"  Count: {len(punctured):,}")
        print(f"  Mean timing: {np.mean(punctured['timing_ns']):.0f} ns")
        print(f"  Std timing: {np.std(punctured['timing_ns']):.0f} ns")
        
    if len(not_punctured) > 0:
        print(f"Non-punctured transitions:")
        print(f"  Count: {len(not_punctured):,}")
        print(f"  Mean timing: {np.mean(not_punctured['timing_ns']):.0f} ns")
        print(f"  Std timing: {np.std(not_punctured['timing_ns']):.0f} ns")
    
    # Visualize correlation
    create_correlation_plots(valid_df)
    
    return df

def create_correlation_plots(df):
    """Create plots showing timing vs transition correlation."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Timing distribution by codeword class
    ax1 = axes[0, 0]
    for trans_class in [TransitionClass.NONE, TransitionClass.UNIPOLAR, TransitionClass.BIPOLAR]:
        class_data = df[df['cw_class'] == trans_class]
        if len(class_data) > 0:
            ax1.hist(class_data['timing_ns'], bins=50, alpha=0.6, 
                    label=f'{trans_class.name} ({len(class_data):,})')
    ax1.set_xlabel('Timing (ns)')
    ax1.set_ylabel('Count')
    ax1.set_title('Timing Distribution by Codeword Transition Class')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Puncturing vs timing
    ax2 = axes[0, 1]
    punctured = df[df['is_puncture']]
    not_punctured = df[~df['is_puncture']]
    
    if len(punctured) > 0:
        ax2.hist(punctured['timing_ns'], bins=50, alpha=0.6, 
                label=f'Punctured ({len(punctured):,})', color='red')
    if len(not_punctured) > 0:
        ax2.hist(not_punctured['timing_ns'], bins=50, alpha=0.6,
                label=f'Not Punctured ({len(not_punctured):,})', color='blue')
    ax2.set_xlabel('Timing (ns)')
    ax2.set_ylabel('Count')
    ax2.set_title('Timing Distribution: Punctured vs Non-Punctured')
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: Transition count vs timing
    ax3 = axes[1, 0]
    ax3.scatter(df['cw_transitions'], df['timing_ns'], alpha=0.3, s=1)
    ax3.set_xlabel('Codeword Transition Count')
    ax3.set_ylabel('Timing (ns)')
    ax3.set_title('Timing vs Number of Bit Transitions')
    
    # Plot 4: Class by timing scatter
    ax4 = axes[1, 1]
    class_colors = {TransitionClass.NONE: 'blue', 
                   TransitionClass.UNIPOLAR: 'orange', 
                   TransitionClass.BIPOLAR: 'red'}
    
    for trans_class, color in class_colors.items():
        class_data = df[df['cw_class'] == trans_class]
        if len(class_data) > 0:
            ax4.scatter(class_data['src_msg'], class_data['timing_ns'], 
                       c=color, alpha=0.3, s=1, label=trans_class.name)
    ax4.set_xlabel('Source Message')
    ax4.set_ylabel('Timing (ns)')
    ax4.set_title('Timing vs Source Message (by Transition Class)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('timing_transition_correlation.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Correlation plots saved: timing_transition_correlation.png")
    plt.show()

if __name__ == "__main__":
    df = analyze_timing_vs_transitions()