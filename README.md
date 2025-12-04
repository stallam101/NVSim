# Sierpinski Gasket NVSim Testing Framework

This framework enables statistical analysis of ReRAM write time distributions using Error Correction Code (ECC) derived codeword transitions to reveal Hamming weight checkerboard patterns through timing side-channel analysis.

## Overview

The framework implements a two-phase write architecture with configurable stochastic timing distributions to study how different pulse count distributions affect write time characteristics and their ability to reveal underlying data patterns.

## Project Structure

```
NVSim/
├── src/
│   ├── cpp/           # C++ source files (NVSim simulator)
│   └── py/            # Python analysis scripts
├── configs/           # Configuration files (.cfg, .cell)
├── measurements/      # Data output directory
│   ├── sierpinski_data/
│   └── sierpinski_4_variants/
├── plots/             # Generated plots and visualizations
├── build/             # Compiled binaries and build artifacts
├── work/              # Temporary working files
└── venv/              # Python virtual environment
```

## Core Scripts

### 1. Main Testing Framework

The primary script for generating timing datasets using different statistical distributions.

#### Usage Syntax
```bash
python run_sierpinski.py <DISTRIBUTION_TYPE> [COMMAND] [OPTIONS]
```

#### Distribution Types
- `--normal`: Gaussian distribution (original, creates left-skewed write times)
- `--negative-binomial`: Negative binomial distribution (attempts right-skewed behavior)  
- `--gamma`: Gamma distribution (Tyler's fitted parameters for right-skewed timing)

#### Commands
- `--full`: Generate complete 65,536 transition dataset (single-run)
- `--run=N`: Generate dataset with N runs per transition (minimum latency selection)
- `--test-batch`: Test batch processing (16 transitions, quick validation)
- `--test-multi N`: Test multi-run mode (3 transitions, N runs each)

#### Examples
```bash
# Quick validation tests
python run_sierpinski.py --normal --test-batch
python run_sierpinski.py --gamma --test-multi 5

# Full dataset generation  
python run_sierpinski.py --normal --full
python run_sierpinski.py --gamma --run=40

# Generate right-skewed timing with noise reduction
python run_sierpinski.py --gamma --run=40
```

### 2. Multi-Distribution Comparison

Generates datasets for multiple distribution types simultaneously for comparative analysis.

#### Usage
```bash
python -m src.py.sierpinski_4_variants
```

#### Features
- Tests 4 distribution variants in parallel
- Automatic comparison plotting
- Stores results in `measurements/sierpinski_4_variants/` directory

### 3. Distribution Analysis

Analyzes generated timing datasets to create write time distribution histograms and statistical analysis.

#### Usage
```bash
python run_analysis.py
```

#### Features
- **Automatically loads latest dataset** from `measurements/sierpinski_data/`
- Generates write time distribution histograms in `plots/`
- Creates timing analysis plots
- Statistical clustering analysis

### 4. Sierpinski Pattern Visualization

Visualizes the sierpinski gasket patterns in timing data to reveal Hamming weight checkerboard structures.

#### Usage
```bash
python run_visualize.py
```

#### Features
- **Loads latest timing dataset** from `measurements/sierpinski_data/`
- Generates sierpinski triangle visualizations in `plots/`
- Reveals ECC-derived codeword transition patterns
- Side-channel timing analysis visualization

## Distribution Configuration

### Template Configuration Files

#### 1. Normal Distribution (Gaussian)
- **Config**: `sierpinski_test.cfg`
- **Cell**: `sierpinski_rram.cell`
- **Parameters**: 
  ```
  -SetPulseCountMean: 640
  -SetPulseCountStdDev: 100
  -ResetPulseCountMean: 600  
  -ResetPulseCountStdDev: 100
  ```

#### 2. Negative Binomial Distribution
- **Config**: `sierpinski_test_negative_binomial.cfg`
- **Cell**: `sierpinski_rram_negative_binomial.cell`
- **Parameters**:
  ```
  -DistributionType: NEGATIVE_BINOMIAL
  -SetSuccessProbability: 0.4
  -SetTargetSuccesses: 1
  -ResetSuccessProbability: 0.3
  -ResetTargetSuccesses: 1
  ```

#### 3. Gamma Distribution (Tyler's Fitted)
- **Config**: `sierpinski_test_gamma.cfg`
- **Cell**: `sierpinski_rram_gamma.cell`  
- **Parameters**:
  ```
  -DistributionType: GAMMA
  # SET: Distribution gamma_k=0.013_mu=0.13
  -SetGammaK: 0.13
  -SetGammaTheta: 423.07692307692304
  -SetGammaLoc: 495.0
  # RESET: Distribution gamma_k=0.013_mu=0.11  
  -ResetGammaK: 0.13
  -ResetGammaTheta: 625.0
  -ResetGammaLoc: 568.75
  ```

## Data Storage

### Primary Dataset Storage: `sierpinski_data/`
```
sierpinski_data/
├── run_YYYYMMDD_HHMMSS/
│   ├── sierpinski_complete_YYYYMMDD_HHMMSS.npy     # Raw timing matrix
│   ├── sierpinski_complete_YYYYMMDD_HHMMSS.csv     # Human-readable data
│   ├── sierpinski_complete_YYYYMMDD_HHMMSS_metadata.json    # Generation parameters
│   └── sierpinski_complete_YYYYMMDD_HHMMSS_statistics.json  # Statistical summary
```

### Multi-Variant Storage: `sierpinski_4_variants/`
```
sierpinski_4_variants/
├── variant_data_YYYYMMDD_HHMMSS.npy
├── comparison_plots/
└── statistical_analysis/
```

## Key Features

### 1. ECC-Based Codeword Generation
- Uses 16×5 parity matrix (P) and bias vector (b)
- Generates 21-bit codewords from 8-bit messages
- Creates systematic error correction patterns

### 2. Two-Phase Write Architecture
- **SET Phase**: 0→1 transitions using +V bias
- **RESET Phase**: 1→0 transitions using -V bias  
- **Redundant Operations**: 0→0, 1→1 (minimal timing)

### 3. Noise Reduction Techniques
- **40-Run Minimum**: Each transition simulated 40 times
- **Minimum Latency Selection**: Takes fastest run to reduce noise
- **Parallel Processing**: Multi-worker batch processing

### 4. Statistical Distribution Support
- **Normal**: Traditional Gaussian pulse count distributions
- **Negative Binomial**: Right-skewed alternative for ReRAM physics
- **Gamma**: Tyler's fitted parameters from empirical data

## Distribution Behavior Analysis

### Expected Write Time Characteristics

| Distribution | Individual Skew | Write Time Skew | Use Case |
|--------------|-----------------|-----------------|----------|
| Normal | Symmetric | Left-skewed | Baseline comparison |
| Negative Binomial | Right-skewed | Left-skewed | Physics-motivated |  
| Gamma | Right-skewed | Left-skewed | Tyler's fitted data |

**Note**: All distributions produce left-skewed write times due to the mathematical pipeline: `Gamma → MAX(parallel) → ADD(phases) → MIN(40-runs)`

## Workflow

### 1. Generate Timing Dataset
```bash
# Choose distribution and run parameters
python sierpinski_test.py --gamma --run=40
```

### 2. Analyze Results
```bash
# Generate write time distribution histogram
python analyze_nvsim_clusters.py
```

### 3. Compare Distributions
```bash
# Multi-distribution comparison
python sierpinski_4_variants.py
```

## Output Files

- **Timing Matrices**: 256×256 arrays of write latencies (ns)
- **Distribution Plots**: Write time histograms by transition class
- **Statistical Analysis**: Skewness, clustering, and pattern detection
- **Metadata**: Generation parameters and runtime statistics

## Configuration Notes

### Cell File Parameters
- **Pulse Width**: 200ns per pulse (fixed)
- **Distribution Parameters**: Control pulse count, not pulse width
- **Redundant Operations**: Always use normal distribution (mean=0)

### Performance
- **Full Dataset**: ~10 hours for 65,536 transitions with 40 runs
- **Test Batch**: ~30 seconds for validation
- **Memory Usage**: ~2GB for complete dataset

## Important Configuration Note

### Distribution Type Override Required

**WARNING**: Due to a configuration parsing issue, you must manually set the distribution type in the source code to match your intended analysis:

1. **Edit** `src/cpp/MemCell.cpp` around line 121
2. **Change** the constructor default:
   ```cpp
   distributionType = GAMMA_DISTRIBUTION;     // For gamma analysis
   distributionType = NORMAL_DISTRIBUTION;   // For normal analysis  
   distributionType = NEGATIVE_BINOMIAL_DISTRIBUTION;  // For negative binomial analysis
   ```
3. **Recompile** with `make clean && make` after each change

This manual override is required because the config file parsing for `-DistributionType` is not being applied correctly, and the constructor default controls the actual distribution used regardless of the config file settings.

## Troubleshooting

### Common Issues
1. **Module Import Errors**: Ensure virtual environment is activated
2. **Compilation Errors**: Run `make clean && make` after parameter changes
3. **Wrong Distribution**: Check that latest dataset matches intended distribution type
4. **Distribution Not Changing**: Manually set `distributionType` in MemCell.cpp constructor (see above)
5. **Left Skew Persistence**: This is expected due to MAX→ADD→MIN mathematical pipeline

### Debug Commands
```bash
# Test basic functionality
python sierpinski_test.py --gamma --test-batch

# Verify compilation
make clean && make

# Check dataset source
ls -la sierpinski_data/
```

This framework enables systematic study of how different stochastic timing models affect the visibility of Hamming weight patterns in ReRAM write operations through ECC-derived side-channel analysis.