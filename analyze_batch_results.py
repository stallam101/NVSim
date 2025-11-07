

import numpy as np
from sierpinski_test import SierpinskiTester

def analyze_message_transition(tester, src_msg, dst_msg, observed_latency):
    

    cw0 = tester.encode_message(src_msg)
    cw1 = tester.encode_message(dst_msg)
    

    transition = tester.analyze_transition(cw0, cw1)
    
    print(f"msg {src_msg}→{dst_msg}: cw 0x{cw0:06X}→0x{cw1:06X}")
    print(f"  Transitions: {transition['set_transitions']}S, {transition['reset_transitions']}R, {transition['redundant_ops']}N")
    print(f"  Observed latency: {observed_latency}ns")
    

    total_operations = transition['set_transitions'] + transition['reset_transitions']
    if total_operations == 0:
        expected = "FASTEST (all redundant)"
    elif transition['reset_transitions'] > transition['set_transitions']:
        expected = "SLOWEST (more resets)"
    elif transition['set_transitions'] > transition['reset_transitions']:
        expected = "MEDIUM (more sets)"
    else:
        expected = "MIXED (equal sets/resets)"
    
    print(f"  Expected: {expected}")
    print()

def main():

    results = [
        (0, 0, 19.439),
        (0, 1, 69.439),
        (0, 2, 69.439),
        (0, 3, 69.439),
        (1, 0, 79.439),
        (1, 1, 29.439),
        (1, 2, 99.439),
        (1, 3, 99.439),
        (2, 0, 69.439),
        (2, 1, 99.439),
        (2, 2, 29.439),
        (2, 3, 109.439),
        (3, 0, 79.439),
        (3, 1, 89.439),
        (3, 2, 99.439),
        (3, 3, 29.439)
    ]
    
    print("🔍 Analyzing Batch Test Results")
    print("=" * 50)
    
    tester = SierpinskiTester()
    
    for src_msg, dst_msg, latency in results:
        analyze_message_transition(tester, src_msg, dst_msg, latency)
    

    print("📊 SUMMARY ANALYSIS:")
    print("-" * 30)
    

    timing_groups = {}
    for src_msg, dst_msg, latency in results:
        if latency not in timing_groups:
            timing_groups[latency] = []
        timing_groups[latency].append((src_msg, dst_msg))
    
    for latency in sorted(timing_groups.keys()):
        transitions = timing_groups[latency]
        print(f"{latency}ns: {len(transitions)} transitions")
        for src, dst in transitions:
            if src == dst:
                print(f"  msg {src}→{dst} (identical)")
            else:
                print(f"  msg {src}→{dst}")

if __name__ == "__main__":
    main()