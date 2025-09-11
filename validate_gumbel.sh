#!/bin/bash

# Gumbel Distribution Validation Script
# Collects word-level completion times to validate theoretical predictions
# Theory: MAX of IID samples should follow Gumbel distribution

echo "=== Gumbel Distribution Validation for Phase 3 Word-Level Modeling ==="
echo "Collecting word-level completion time samples..."
echo

# Test configuration: All-SET pattern (ensures all cells participate in MAX)
CONFIG="sample_word_level_worst_case.cfg"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Test configuration $CONFIG not found!"
    exit 1
fi

if [ ! -f "./nvsim" ]; then
    echo "ERROR: nvsim executable not found! Run 'make' first."
    exit 1
fi

# Create output file
OUTPUT_FILE="gumbel_validation_samples.dat"
SUMMARY_FILE="gumbel_validation_results.txt"

echo "Configuration: $CONFIG" > $SUMMARY_FILE
echo "Pattern: All-SET (worst-case - all 64 bits perform slow SET transitions)" >> $SUMMARY_FILE
echo "Expected: Word completion = MAX(64 IID SET operation times)" >> $SUMMARY_FILE
echo "Prediction: Should follow Gumbel distribution" >> $SUMMARY_FILE
echo >> $SUMMARY_FILE

echo "Collecting samples to: $OUTPUT_FILE"
echo "Run,SET_Latency_ns,RESET_Latency_ns,Write_Bandwidth_MBs" > $OUTPUT_FILE

# Collect samples (50 runs for statistical significance)
NUM_SAMPLES=50
echo "Running $NUM_SAMPLES samples..."

for i in $(seq 1 $NUM_SAMPLES); do
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Progress: $i/$NUM_SAMPLES samples collected"
    fi
    
    # Run NVSim and extract timing results
    RESULT=$(./nvsim $CONFIG 2>/dev/null | grep -E "(SET Latency|RESET Latency|Write Bandwidth)")
    
    if [ -n "$RESULT" ]; then
        # Extract numeric values
        SET_LATENCY=$(echo "$RESULT" | grep "SET Latency" | sed 's/.*= \([0-9.]*\)ns.*/\1/')
        RESET_LATENCY=$(echo "$RESULT" | grep "RESET Latency" | sed 's/.*= \([0-9.]*\)ns.*/\1/')
        BANDWIDTH=$(echo "$RESULT" | grep "Write Bandwidth" | sed 's/.*= \([0-9.]*\)MB\/s.*/\1/')
        
        # Store results
        echo "$i,$SET_LATENCY,$RESET_LATENCY,$BANDWIDTH" >> $OUTPUT_FILE
    else
        echo "WARNING: Run $i failed to produce valid output"
        echo "$i,ERROR,ERROR,ERROR" >> $OUTPUT_FILE
    fi
done

echo
echo "Sample collection complete!"
echo "Results saved to: $OUTPUT_FILE"

# Basic statistical analysis
echo "=== Statistical Analysis ===" >> $SUMMARY_FILE
echo >> $SUMMARY_FILE

# Analyze SET latency (the MAX operation result)
echo "SET Latency Analysis (Word-Level MAX Result):" >> $SUMMARY_FILE

# Extract SET latency values (excluding header and error lines)
SET_VALUES=$(tail -n +2 $OUTPUT_FILE | grep -v "ERROR" | cut -d',' -f2)

if [ -n "$SET_VALUES" ]; then
    echo "Sample count: $(echo "$SET_VALUES" | wc -l)" >> $SUMMARY_FILE
    
    # Calculate basic statistics using awk
    echo "$SET_VALUES" | awk '
    BEGIN { 
        min = 999999; max = 0; sum = 0; count = 0; 
        for(i=1; i<=NF; i++) values[i] = 0;
    }
    { 
        count++; 
        sum += $1; 
        values[count] = $1;
        if($1 < min) min = $1; 
        if($1 > max) max = $1; 
    }
    END {
        mean = sum/count;
        
        # Calculate variance
        sumsq = 0;
        for(i=1; i<=count; i++) {
            diff = values[i] - mean;
            sumsq += diff * diff;
        }
        var = sumsq/(count-1);
        stddev = sqrt(var);
        
        print "  Mean: " mean " ns"
        print "  Std Dev: " stddev " ns" 
        print "  Min: " min " ns"
        print "  Max: " max " ns"
        print "  Range: " (max-min) " ns"
        
        # Gumbel distribution parameters estimation
        # For Gumbel: mean = μ + γ*β, stddev = π*β/√6
        # Where γ ≈ 0.5772 (Euler-Mascheroni constant)
        euler_gamma = 0.5772156649;
        pi = 3.14159265359;
        
        beta_estimate = stddev * sqrt(6) / pi;
        mu_estimate = mean - euler_gamma * beta_estimate;
        
        print ""
        print "Gumbel Distribution Parameter Estimates:"
        print "  Location (μ): " mu_estimate " ns"
        print "  Scale (β): " beta_estimate " ns"
        print ""
        print "Theoretical Gumbel properties:"
        print "  Expected mean: μ + γβ = " (mu_estimate + euler_gamma * beta_estimate) " ns"
        print "  Expected stddev: πβ/√6 = " (pi * beta_estimate / sqrt(6)) " ns"
    }' >> $SUMMARY_FILE
    
    echo >> $SUMMARY_FILE
    echo "=== Raw Data Sample (first 10 values) ===" >> $SUMMARY_FILE
    echo "SET Latency values:" >> $SUMMARY_FILE
    echo "$SET_VALUES" | head -10 | tr '\n' ' ' >> $SUMMARY_FILE
    echo >> $SUMMARY_FILE
fi

echo
echo "=== Analysis Complete ==="
echo "Full results: $SUMMARY_FILE"
echo
echo "Summary:"
cat $SUMMARY_FILE

echo
echo "Next steps for complete validation:"
echo "1. Use statistical software (R, Python) for Kolmogorov-Smirnov test"
echo "2. Generate Q-Q plots against theoretical Gumbel distribution"
echo "3. Test with different word widths (more samples = better Gumbel fit)"
echo "4. Compare empirical vs theoretical Gumbel CDF"