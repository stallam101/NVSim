

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from enum import Enum, auto
from sierpinski_test import SierpinskiTester

class TransitionClass(Enum):
    NONE = auto()
    UNIPOLAR = auto()
    BIPOLAR = auto()

def load_nvsim_data():
    print("📊 Loading NVSim timing data...")
    

    from pathlib import Path
    # Ensure plots directory exists
    Path("plots").mkdir(exist_ok=True)
    
    data_dir = Path("measurements/sierpinski_data")
    run_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Using latest run: {latest_run_dir.name}")
    

    npy_files = list(latest_run_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {latest_run_dir}")
    
    matrix_file = npy_files[0]
    matrix = np.load(matrix_file)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Loading dataset: {matrix_file}")
    

    data = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            data.append({
                'm0_B': i,
                'm1_B': j,
                'timing': matrix[i, j]
            })
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} transitions")
    return df, matrix

def classify_transitions_by_codewords(df):
    print("Classifying transitions by codeword analysis...")
    
    tester = SierpinskiTester()
    

    cw0_list = []
    cw1_list = []
    transition_classes = []
    
    for _, row in df.iterrows():
        m0 = int(row['m0_B'])
        m1 = int(row['m1_B'])
        

        cw0 = tester.encode_message(m0)
        cw1 = tester.encode_message(m1)
        
        cw0_list.append(cw0)
        cw1_list.append(cw1)
        

        up = (~cw0) & cw1
        down = cw0 & (~cw1)
        
        has_up = up != 0
        has_down = down != 0
        

        if has_up and has_down:
            transition_class = TransitionClass.BIPOLAR
        elif has_up or has_down:
            transition_class = TransitionClass.UNIPOLAR
        else:
            transition_class = TransitionClass.NONE
            
        transition_classes.append(transition_class)
    
    df['cw0'] = cw0_list
    df['cw1'] = cw1_list
    df['timing_label_from_codewords'] = [tc.name for tc in transition_classes]
    
    print("Transition class distribution:")
    print(df['timing_label_from_codewords'].value_counts())
    
    return df

def plot_timing_distribution_by_class(df):
    print("Creating timing distribution plot by transition class...")
    

    class_to_float = {
        'NONE': 0.0,
        'UNIPOLAR': 1.0,
        'BIPOLAR': 2.0,
    }
    df['class_float'] = df['timing_label_from_codewords'].map(class_to_float)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    

    sns.histplot(
        ax=ax1,
        data=df,
        x='timing',
        bins=50,
        alpha=0.8
    )
    ax1.set_title('NVSim Write Time Distribution', fontsize=14)
    ax1.set_xlabel('Write Time (ns)')
    ax1.set_ylabel('Count')
    

    sns.histplot(
        ax=ax2,
        data=df,
        x='timing',
        bins=50,
        alpha=0.7,
        hue='timing_label_from_codewords'
    )
    ax2.set_title('Write Time Distribution by Transition Class', fontsize=14)
    ax2.set_xlabel('Write Time (ns)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('plots/nvsim_timing_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def apply_gaussian_mixture_clustering(df):
    print("🎯 Applying Gaussian Mixture Model clustering...")
    
    X = df['timing'].values.reshape(-1, 1)
    

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-6,
        random_state=0
    ).fit(X)
    

    means = gmm.means_.ravel()
    order = np.argsort(means)
    cluster_assignments = gmm.predict(X)
    
    
    labels = np.zeros_like(cluster_assignments)
    for i, cluster_idx in enumerate(order):
        labels[cluster_assignments == cluster_idx] = i
    

    name_by_idx = {
        0: TransitionClass.NONE,
        1: TransitionClass.UNIPOLAR, 
        2: TransitionClass.BIPOLAR
    }
    
    timing_labels_from_clustering = [name_by_idx[label].name for label in labels]
    df['timing_label_from_clustering'] = timing_labels_from_clustering
    
    sorted_means = np.sort(means)
    std_devs = np.sqrt(gmm.covariances_.ravel())
    print(f"GMM cluster means (sorted): {sorted_means} ns")
    print(f"GMM cluster std devs: {std_devs} ns")
    

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['red', 'green', 'blue']
    for i, (class_name, color) in enumerate(zip(['NONE', 'UNIPOLAR', 'BIPOLAR'], colors)):
        cluster_data = df[df['timing_label_from_clustering'] == name_by_idx[i].name]['timing']
        ax.hist(cluster_data, bins=30, alpha=0.7, 
                color=color, label=f'{class_name} (μ={means[order[i]]:.1f}ns)')
    
    ax.set_title('NVSim Clustered Write Time Distribution (3 Components)', fontsize=14)
    ax.set_xlabel('Write Time (ns)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/nvsim_clustered_timing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, gmm

def compare_classification_methods(df):
    print("📋 Comparing classification methods...")
    

    confusion_data = []
    for codeword_class in ['NONE', 'UNIPOLAR', 'BIPOLAR']:
        for cluster_class in ['NONE', 'UNIPOLAR', 'BIPOLAR']:
            count = len(df[
                (df['timing_label_from_codewords'] == codeword_class) &
                (df['timing_label_from_clustering'] == cluster_class)
            ])
            confusion_data.append({
                'codeword_class': codeword_class,
                'cluster_class': cluster_class,
                'count': count
            })
    
    confusion_df = pd.DataFrame(confusion_data)
    confusion_matrix = confusion_df.pivot(index='codeword_class', columns='cluster_class', values='count')
    
    print("Classification confusion matrix:")
    print(confusion_matrix)
    

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Classification Comparison: Codeword vs Clustering')
    plt.ylabel('Codeword-based Classification')
    plt.xlabel('Clustering-based Classification')
    plt.tight_layout()
    plt.savefig('plots/classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return confusion_matrix

def plot_sierpinski_patterns(df):
    print("🎨 Creating Sierpinski pattern visualizations...")
    

    class_to_float = {
        'NONE': 0.0,
        'UNIPOLAR': 1.0,
        'BIPOLAR': 2.0,
    }
    
    df['codeword_class_float'] = df['timing_label_from_codewords'].map(class_to_float)
    df['cluster_class_float'] = df['timing_label_from_clustering'].map(class_to_float)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    

    timing_matrix = df.pivot(index='m1_B', columns='m0_B', values='timing')
    sns.heatmap(timing_matrix, ax=ax1, cmap='viridis', square=True)
    ax1.set_title('Raw NVSim Timing (ns)')
    ax1.set_xlabel('Source Message')
    ax1.set_ylabel('Destination Message')
    

    codeword_matrix = df.pivot(index='m1_B', columns='m0_B', values='codeword_class_float')
    sns.heatmap(codeword_matrix, ax=ax2, cmap='coolwarm', square=True)
    ax2.set_title('Codeword-based Classification')
    ax2.set_xlabel('Source Message')
    ax2.set_ylabel('Destination Message')
    

    cluster_matrix = df.pivot(index='m1_B', columns='m0_B', values='cluster_class_float')
    sns.heatmap(cluster_matrix, ax=ax3, cmap='coolwarm', square=True)
    ax3.set_title('Clustering-based Classification')
    ax3.set_xlabel('Source Message')
    ax3.set_ylabel('Destination Message')
    
    plt.tight_layout()
    plt.savefig('plots/sierpinski_pattern_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_timing_statistics(df):
    print("📊 Analyzing timing statistics by transition class...")
    
    stats = df.groupby('timing_label_from_codewords')['timing'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nTiming statistics by codeword-based classification:")
    print(stats)
    

    none_mean = stats.loc['NONE', 'mean']
    unipolar_mean = stats.loc['UNIPOLAR', 'mean']
    bipolar_mean = stats.loc['BIPOLAR', 'mean']
    
    print(f"\nExpected ordering check:")
    print(f"NONE mean: {none_mean:.1f} ns")
    print(f"UNIPOLAR mean: {unipolar_mean:.1f} ns") 
    print(f"BIPOLAR mean: {bipolar_mean:.1f} ns")
    
    if none_mean < unipolar_mean < bipolar_mean:
        print("✅ Timing order is correct: NONE < UNIPOLAR < BIPOLAR")
    else:
        print("❌ Timing order is incorrect!")
        
    return stats

def main():
    print("🔬 NVSim Timing Cluster Analysis")
    print("=" * 60)
    
    try:


        df, matrix = load_nvsim_data()
        

        df = classify_transitions_by_codewords(df)
        

        df = plot_timing_distribution_by_class(df)
        

        df, gmm = apply_gaussian_mixture_clustering(df)
        

        confusion_matrix = compare_classification_methods(df)
        

        plot_sierpinski_patterns(df)
        

        stats = analyze_timing_statistics(df)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Plots saved: nvsim_timing_distribution.png, nvsim_clustered_timing.png, etc.")
        
        return df, stats, gmm
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()