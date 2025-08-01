# NVSim with Stochastic Bit-Cell Transition Times

## Overview

NVSim is an area, timing and power model for both volatile (SRAM, DRAM) and non-volatile memory (PCRAM, STT-RAM, ReRAM, SLC NAND Flash). This modified version adds support for **stochastic bit-cell transition times** to mirror real-world commercial memory behavior.

### What's New

Instead of using constant write pulse durations, this version can sample from probability distributions to generate realistic variation in:
- SET pulse durations
- RESET pulse durations  
- Overall write latencies

## How NVSim Works

1. **Configuration**: User provides `.cfg` file specifying memory architecture and `.cell` file defining memory cell properties
2. **Modeling**: NVSim calculates area, timing, and power for the specified memory design
3. **Output**: Reports detailed metrics including read/write latencies, energy consumption, and area

### Traditional vs Stochastic Operation

**Traditional (Constant)**:
- SET pulse = 10ns (always)
- RESET pulse = 10ns (always)
- Write latency = 2.5ns (always)

**Stochastic (Variable)**:
- SET pulse ~ Normal(10ns, 2ns) → 8.2ns, 11.7ns, 9.1ns...
- RESET pulse ~ Normal(12ns, 3ns) → 9.8ns, 14.2ns, 13.1ns...
- Write latency varies accordingly → 2.1ns, 2.9ns, 2.4ns...

## Configuration Guide

### 1. Main Configuration File (nvsim.cfg)

```bash
# Basic memory configuration
-DesignTarget: cache
-Capacity (B): 1048576
-WordWidth (B): 8
-AssociativityAssociativity: 8

# Enable stochastic analysis
-DistributionAnalysis: true
-DistributionSamples: 1000

# Memory cell file
-MemoryCellInputFile: rram_stochastic.cell
```

### 2. RRAM Cell Configuration (rram_stochastic.cell)

#### Constant Mode (Traditional):
```bash
-MemCellType: memristor
-CellArea (F^2): 4
-CellAspectRatio: 1
-ResistanceOn (ohm): 1000
-ResistanceOff (ohm): 100000

# Constant pulse timing
-SetPulse (ns): 10
-ResetPulse (ns): 15

-AccessType: CMOS
-AccessCMOSWidth (F): 4
```

#### Stochastic Mode (New):
```bash
-MemCellType: memristor
-CellArea (F^2): 4
-CellAspectRatio: 1
-ResistanceOn (ohm): 1000
-ResistanceOff (ohm): 100000

# Stochastic pulse timing
-SetPulseDistribution: normal
-SetPulseMean (ns): 10
-SetPulseStdDev (ns): 2
-SetPulseMin (ns): 5
-SetPulseMax (ns): 20

-ResetPulseDistribution: normal
-ResetPulseMean (ns): 15
-ResetPulseStdDev (ns): 3
-ResetPulseMin (ns): 8
-ResetPulseMax (ns): 25

-AccessType: CMOS
-AccessCMOSWidth (F): 4
```

## Running RRAM Analysis

### Basic RRAM Test:
```bash
# Single run (constant timing)
./nvsim rram_constant.cfg

# Stochastic analysis (1000 samples)
./nvsim rram_stochastic.cfg
```

## Fitting to Commercial Data

### Step 1: Analyze Your Data
```python
# Example: analyze your commercial RRAM data
import numpy as np
commercial_set_times = [9.2, 11.8, 8.7, 12.1, ...]
mean_set = np.mean(commercial_set_times)      # e.g., 10.5ns
std_set = np.std(commercial_set_times)        # e.g., 1.8ns
```

### Step 2: Configure NVSim Parameters
```bash
# Match your data in .cell file
-SetPulseMean (ns): 10.5
-SetPulseStdDev (ns): 1.8
```

### Step 3: Validate Distribution
```bash
# Run simulation
./nvsim rram_fitted.cfg

# Check if output statistics match your commercial data
```

## Distribution Types Supported

### Normal Distribution
```bash
-SetPulseDistribution: normal
-SetPulseMean (ns): 10      # Mean value
-SetPulseStdDev (ns): 2     # Standard deviation
-SetPulseMin (ns): 5        # Optional lower bound
-SetPulseMax (ns): 20       # Optional upper bound
```

### Backward Compatibility

All existing `.cell` files continue to work unchanged. The system defaults to constant timing unless distribution parameters are explicitly specified.

## Files Modified

- `typedef.h` - Added distribution enums
- `MemCell.h/cpp` - Extended for stochastic support
- `SubArray.cpp` - Updated latency calculations
- `InputParameter.h/cpp` - Added analysis parameters
- `Result.h/cpp` - Added statistics tracking
- `main.cpp` - Added multi-run analysis loop

This enhancement enables NVSim to generate realistic, variable write latencies that can accurately model commercial RRAM behavior.