# Two-Phase Write Architecture in NVSim

## Overview

The two-phase write architecture models realistic non-volatile memory controller behavior where **every write operation consists of two sequential phases**: a SET phase and a RESET phase. Each phase completes when the slowest cell in that phase finishes, and the total write time is the sum of both phase completion times.

## Architecture Principles

### Sequential Phase Execution
```
Total Write Time = SET Phase Time + RESET Phase Time

Where:
SET Phase Time = MAX(participating cells with +V bias in SET phase)
RESET Phase Time = MAX(participating cells with -V bias in RESET phase)
```

### Selective Phase Participation (Corrected Implementation)
- **Only relevant bits participate in each phase** based on voltage bias application
- **SET Phase**: Only bits with `targetBit=1` receive +V bias and participate in timing
- **RESET Phase**: Only bits with `targetBit=0` receive -V bias and participate in timing
- **NONE State**: Bits with no voltage bias (0V) do not participate in phase timing

### Realistic Memory Controller Modeling
This architecture reflects physical memory controller behavior based on cell bias states:
1. **Cell Bias Formula**: `Vte-Vbe` where bias = {(+TE,+BE), (+TE,0BE), (0TE,+BE), (0TE,0BE)} = {0V, +V, -V, 0V}
2. **SET Phase**: Apply +V bias only to cells requiring `targetBit=1` writes
3. **RESET Phase**: Apply -V bias only to cells requiring `targetBit=0` writes
4. **No parallelism**: Cannot proceed to RESET phase until SET phase completes

## Implementation Details

### Phase Operation Determination

#### SET Phase Logic:
```cpp
TransitionType DetermineSetPhaseOperation(WritePattern pattern, int bitPosition) {
    /* SET phase: Apply +V bias only to cells where targetBit = 1 */
    bool currentBit = (pattern.currentData >> bitPosition) & 1;
    bool targetBit = (pattern.targetData >> bitPosition) & 1;
    
    if (targetBit == false) {
        return NONE;          // No +V bias applied in SET phase
    }
    if (currentBit == false) {
        return SET;           // 0→1: true SET operation (HRS→LRS)
    } else {
        return REDUNDANT_SET; // 1→1: reinforcing SET bias (already LRS)
    }
}
```

#### RESET Phase Logic:
```cpp
TransitionType DetermineResetPhaseOperation(WritePattern pattern, int bitPosition) {
    /* RESET phase: Apply -V bias only to cells where targetBit = 0 */
    bool currentBit = (pattern.currentData >> bitPosition) & 1;
    bool targetBit = (pattern.targetData >> bitPosition) & 1;
    
    if (targetBit == true) {
        return NONE;            // No -V bias applied in RESET phase
    }
    if (currentBit == true) {
        return RESET;           // 1→0: true RESET operation (LRS→HRS)
    } else {
        return REDUNDANT_RESET; // 0→0: reinforcing RESET bias (already HRS)
    }
}
```

#### Cell State Summary:
```
Transition Types based on cell bias voltage and internal resistance state:

NONE:            0V bias  - No voltage applied, no sampling
SET:             +V bias  - 0→1 transition, HRS→LRS, true SET operation  
RESET:           -V bias  - 1→0 transition, LRS→HRS, true RESET operation
REDUNDANT_SET:   +V bias  - 1→1 transition, LRS state, reinforcing SET bias
REDUNDANT_RESET: -V bias  - 0→0 transition, HRS state, reinforcing RESET bias
```

### Stochastic Sampling Per Phase

#### SET Phase Execution:
```cpp
double maxSetPhaseTime = 0;
for (int i = 0; i < 64; i++) {
    TransitionType setOperation = DetermineSetPhaseOperation(pattern, i);
    if (setOperation == NONE) {
        continue; // No voltage bias applied - cell doesn't participate
    }
    int setPulseCount = cell->SamplePulseCount(setOperation);
    double setCellTime = setPulseCount * cell->setPulse;  // 10ns per pulse
    maxSetPhaseTime = max(maxSetPhaseTime, setCellTime);
}
```

#### RESET Phase Execution:
```cpp
double maxResetPhaseTime = 0;
for (int i = 0; i < 64; i++) {
    TransitionType resetOperation = DetermineResetPhaseOperation(pattern, i);
    if (resetOperation == NONE) {
        continue; // No voltage bias applied - cell doesn't participate
    }
    int resetPulseCount = cell->SamplePulseCount(resetOperation);
    double resetCellTime = resetPulseCount * cell->resetPulse;  // 10ns per pulse
    maxResetPhaseTime = max(maxResetPhaseTime, resetCellTime);
}
```

## Detailed Example: Alternating Pattern Write

### Input Data Pattern
```
Current Data: 0x5555555555555555 = 0101010101010101... (64 bits)
Target Data:  0xAAAAAAAAAAAAAAAA = 1010101010101010... (64 bits)

Transition Analysis:
- Bit 0: 0→1 (SET needed)
- Bit 1: 1→0 (RESET needed)  
- Bit 2: 0→1 (SET needed)
- Bit 3: 1→0 (RESET needed)
- ...
Result: 32 bits need SET, 32 bits need RESET
```

### SET Phase Execution

#### Operation Determination:
```
Bit 0: targetBit=1 → 0→1 → SET operation needed
Bit 1: targetBit=0 → NONE (no +V bias applied)
Bit 2: targetBit=1 → 0→1 → SET operation needed  
Bit 3: targetBit=0 → NONE (no +V bias applied)
...
Result: 32 actual SET operations, 32 NONE (non-participating)
```

#### Stochastic Sampling:
```
SET Operations (bits 0,2,4,... with targetBit=1):
- Sample from SET distribution (mean=4.2, σ=1.5, range=[1,12])
- Possible pulse counts: 3, 5, 4, 6, 2, 7, 4, 3, 5, 4, 6, 8, 3, 4, 5, 6...

NONE Operations (bits 1,3,5,... with targetBit=0):
- No sampling performed - cell doesn't participate in SET phase
- Skipped in timing calculation
```

#### SET Phase Completion:
```
Participating cell completion times:
Bit 0: 3 pulses × 10ns = 30ns
Bit 2: 5 pulses × 10ns = 50ns
Bit 4: 4 pulses × 10ns = 40ns
...
Bit 62: 6 pulses × 10ns = 60ns  

SET Phase Time = MAX(30, 50, 40, ..., 60) = 60ns
(Only participating cells contribute to phase timing)
```

### RESET Phase Execution

#### Operation Determination:
```
Bit 0: targetBit=1 → NONE (no -V bias applied)
Bit 1: targetBit=0 → 1→0 → RESET operation needed
Bit 2: targetBit=1 → NONE (no -V bias applied)
Bit 3: targetBit=0 → 1→0 → RESET operation needed
...
Result: 32 NONE (non-participating), 32 actual RESET operations
```

#### Stochastic Sampling:
```
RESET Operations (bits 1,3,5,... with targetBit=0):
- Sample from RESET distribution (mean=3.8, σ=1.2, range=[1,10])
- Possible pulse counts: 4, 3, 5, 2, 6, 4, 3, 5, 4, 3, 6, 2, 4, 5, 3, 4...

NONE Operations (bits 0,2,4,... with targetBit=1):
- No sampling performed - cell doesn't participate in RESET phase
- Skipped in timing calculation
```

#### RESET Phase Completion:
```
Participating cell completion times:
Bit 1: 4 pulses × 10ns = 40ns
Bit 3: 3 pulses × 10ns = 30ns
Bit 5: 5 pulses × 10ns = 50ns
...
Bit 63: 4 pulses × 10ns = 40ns

RESET Phase Time = MAX(40, 30, 50, ..., 40) = 50ns
(Only participating cells contribute to phase timing)
```

### Final Write Completion

```
Total Write Time = SET Phase Time + RESET Phase Time
Total Write Time = 60ns + 50ns = 110ns

Write Bandwidth = 64 bits / 110ns / 8 bits/byte = 72.73MB/s
```

### NVSim Output Interpretation
```
- SET Latency   = 72.584ns     ← BaseLatency (12.584ns) + SET Phase (60ns)
- RESET Latency = 72.584ns     ← BaseLatency (22.584ns) + RESET Phase (50ns)  
- Write Bandwidth = 56.815MB/s ← 64 / (max phase latency) / 8

Note: NVSim reports MAX(SET Latency, RESET Latency) as write completion time
```

## Pattern Comparison

### All-SET Pattern (0x00000000FFFFFFFF → 0xFFFFFFFFFFFFFFFF)
```
SET Phase:    32 bits participate with actual SET operations (targetBit=1)
              32 bits don't participate (targetBit=1, already 1)
              Phase time = MAX(32 SET samples) ≈ 12.584ns

RESET Phase:  0 bits participate (no targetBit=0)
              Phase time = minimal baseline ≈ 2.584ns

Total: ~12.584ns → Much higher bandwidth (740MB/s vs 56MB/s)
```

### All-NONE Pattern (0xFFFFFFFFFFFFFFFF → 0xFFFFFFFFFFFFFFFF)
```
SET Phase:    0 bits participate (no targetBit=1 transitions needed)
              Phase time = minimal baseline ≈ 2.584ns

RESET Phase:  0 bits participate (no targetBit=0 transitions needed)
              Phase time = minimal baseline ≈ 2.584ns

Total: ~2.584ns → Highest bandwidth (minimal work)
```

### Mixed Pattern Performance Demonstration
The corrected architecture shows dramatic performance differences:
- **Symmetric pattern** (alternating): 56.815MB/s - both phases active
- **Asymmetric pattern** (SET-only): 740.184MB/s - 13x faster due to optimized phase participation
- **No-change pattern** (redundant): Highest bandwidth - minimal overhead

## Key Architectural Properties

### Realistic Memory Controller Behavior
- **Sequential phases**: Cannot pipeline SET and RESET operations
- **Selective participation**: Only cells receiving voltage bias participate in phase timing
- **Voltage-based logic**: Phase participation determined by target bit value, not current→target transition
- **Pattern sensitivity**: Performance heavily depends on actual data being written

### Stochastic Timing Characteristics
- **Phase-level MAX operations**: Each phase exhibits Gumbel distribution emergence from participating cells only
- **Variable sample sizes**: Phases may have different numbers of participating cells (0-64)
- **Pattern-dependent variation**: Asymmetric patterns show dramatically different phase timing

### Performance Implications
- **Asymmetric patterns are much faster**: Phases with few participants complete quickly
- **Symmetric patterns show balanced performance**: Both phases contribute significantly to timing
- **No-change patterns are fastest**: Both phases have minimal participation

## Configuration Requirements

### Cell File Parameters
```bash
# Actual operations (wide distributions)
-SetPulseCountMean: 4.2
-SetPulseCountStdDev: 1.5
-ResetPulseCountMean: 3.8  
-ResetPulseCountStdDev: 1.2

# Redundant operations (tight distributions)
-RedundantPulseCountMean: 1.0
-RedundantPulseCountStdDev: 0.1
-RedundantPulseCountMin: 1
-RedundantPulseCountMax: 1
```

### WritePattern Specification
```bash
# Enable Phase 3 word-level analysis
-WritePatternType: specific
-CurrentData: 0x5555555555555555
-TargetData: 0xAAAAAAAAAAAAAAAA
```

This two-phase architecture provides realistic modeling of non-volatile memory write operations with proper phase separation, stochastic timing variation, and data-pattern dependent performance characteristics.