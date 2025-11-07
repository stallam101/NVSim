#!/usr/bin/env python3

import subprocess
import numpy as np
import re
import time
import tempfile
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from galois import GF2

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SierpinskiVariantTester:
    
    def __init__(self):
        self.nvsim_executable = "./nvsim"
        self.base_config = "sierpinski_test.cfg"
        self.cell_file = "sierpinski_rram.cell"
        self.p_matrix_file = "P_matrix.csv"
        self.b_vector_file = "b_vector.csv"
        self.results_dir = Path("results")
        
        self._validate_files()
        
        # Load ECC parameters
        self.P = self._load_p_matrix()
        self.b = self._load_b_vector()
        
        logger.info("SierpinskiVariantTester initialized successfully")
        logger.info(f"P matrix shape: {self.P.shape}")
        logger.info(f"b vector shape: {self.b.shape}")
    
    def _validate_files(self):
        required_files = [
            self.nvsim_executable,
            self.base_config, 
            self.cell_file,
            self.p_matrix_file,
            self.b_vector_file
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
    
    def _load_p_matrix(self) -> GF2:
        try:
            P_np = np.loadtxt(self.p_matrix_file, delimiter=',', dtype=np.uint8)
            if P_np.shape != (16, 5):
                raise ValueError(f"P matrix has wrong shape: {P_np.shape}, expected (16, 5)")
            return GF2(P_np)
        except Exception as e:
            raise RuntimeError(f"Failed to load P matrix: {e}")
    
    def _load_b_vector(self) -> GF2:
        try:
            b_np = np.loadtxt(self.b_vector_file, delimiter=',', dtype=np.uint8)
            if b_np.shape != (5,):
                raise ValueError(f"b vector has wrong shape: {b_np.shape}, expected (5,)")
            return GF2(b_np)
        except Exception as e:
            raise RuntimeError(f"Failed to load b vector: {e}")
    
    def encode_message(self, message: int) -> int:
        """Encode 16-bit message to 21-bit codeword using ECC"""
        if not (0 <= message <= 65535):  # 16-bit message range
            raise ValueError(f"Message must be 16-bit (0-65535), got {message}")
        
        # Convert message to 16-bit array
        message_bits = GF2([(message >> i) & 1 for i in range(16)])
        
        # Calculate parity bits: parity = message * P + b
        parity_bits = message_bits @ self.P + self.b
        
        # Concatenate: [message_bits | parity_bits]
        codeword_bits = np.concatenate([message_bits, parity_bits])
        
        # Convert back to integer
        codeword = sum(int(bit) << i for i, bit in enumerate(codeword_bits))
        return codeword
    
    def generate_message_pattern(self, byte_index: int, global_state: int) -> List[Tuple[int, int]]:
        """
        Generate message patterns for specific variant.
        
        Args:
            byte_index: 0 (vary low byte) or 1 (vary high byte)
            global_state: 0 (other byte = 0x00) or 1 (other byte = 0xFF)
            
        Returns:
            List of (src_message, dst_message) tuples
        """
        patterns = []
        
        for src_byte in range(256):
            for dst_byte in range(256):
                if byte_index == 0:
                    # Vary low byte (bits 0-7), global_state sets high byte (bits 8-15)
                    global_byte = 0xFF if global_state == 1 else 0x00
                    src_msg = (global_byte << 8) | src_byte
                    dst_msg = (global_byte << 8) | dst_byte
                else:
                    # Vary high byte (bits 8-15), global_state sets low byte (bits 0-7)
                    global_byte = 0xFF if global_state == 1 else 0x00
                    src_msg = (src_byte << 8) | global_byte
                    dst_msg = (dst_byte << 8) | global_byte
                
                patterns.append((src_msg, dst_msg))
        
        return patterns
    
    def analyze_transition(self, cw0: int, cw1: int) -> Dict[str, int]:
        """Analyze bit transitions between two codewords"""
        set_transitions = 0
        reset_transitions = 0
        redundant_ops = 0
        
        for i in range(21):
            bit_mask = 1 << i
            cw0_bit = (cw0 & bit_mask) >> i
            cw1_bit = (cw1 & bit_mask) >> i
            
            if cw0_bit == 0 and cw1_bit == 1:
                set_transitions += 1
            elif cw0_bit == 1 and cw1_bit == 0:
                reset_transitions += 1
            else:
                redundant_ops += 1
        
        return {
            'set_transitions': set_transitions,
            'reset_transitions': reset_transitions, 
            'redundant_ops': redundant_ops,
            'total_bits': 21
        }
    
    def classify_transition_type(self, cw0: int, cw1: int) -> str:
        transition = self.analyze_transition(cw0, cw1)
        
        if transition['set_transitions'] == 0 and transition['reset_transitions'] == 0:
            return "NONE"
        elif transition['set_transitions'] > 0 and transition['reset_transitions'] > 0:
            return "BIPOLAR"
        else:
            return "UNIPOLAR"
    
    def create_test_config(self, cw0: int, cw1: int, test_id: str) -> str:
        """Create NVSim config file for specific transition"""
        with open(self.base_config, 'r') as f:
            config_content = f.read()
        
        # Update codeword values
        config_content = re.sub(
            r'-CurrentData: 0x[0-9A-Fa-f]+',
            f'-CurrentData: 0x{cw0:08X}',
            config_content
        )
        
        config_content = re.sub(
            r'-TargetData: 0x[0-9A-Fa-f]+',
            f'-TargetData: 0x{cw1:08X}',
            config_content
        )
        
        # Write config file
        test_config_path = f"test_variant_{test_id}.cfg"
        with open(test_config_path, 'w') as f:
            f.write(config_content)
        
        return test_config_path
    
    def run_nvsim(self, config_file: str, timeout: int = 30) -> Dict:
        """Run NVSim with machine-readable output"""
        try:
            cmd = [self.nvsim_executable, config_file, "-MachineReadable"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"NVSim failed with return code {result.returncode}",
                    'stderr': result.stderr
                }
            
            # Parse timing data
            timing_data = self._parse_machine_readable_output(result.stdout)
            
            return {
                'success': True,
                'timing_data': timing_data,
                'write_latency_ns': timing_data.get('TOTAL_WRITE_LATENCY_NS'),
                'stdout': result.stdout
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"NVSim timed out after {timeout}s"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def _parse_machine_readable_output(self, stdout: str) -> Dict[str, float]:
        """Parse NVSim machine-readable output"""
        timing_data = {}
        
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key.endswith('_NS') or key.endswith('_PJ') or key.endswith('_UM2') or key.endswith('_MW'):
                    try:
                        timing_data[key] = float(value)
                    except ValueError:
                        timing_data[key] = None
                else:
                    timing_data[key] = value
        
        return timing_data


class SierpinskiVariantProcessor:
    
    def __init__(self, tester: SierpinskiVariantTester):
        self.tester = tester
    
    def process_variant_batch(self, message_pairs: List[Tuple[int, int]], 
                            variant_name: str, batch_id: int) -> Dict:
        """Process a batch of message pairs for specific variant"""
        batch_results = {}
        
        for src_msg, dst_msg in message_pairs:
            try:
                # Encode messages to codewords
                cw0 = self.tester.encode_message(src_msg)
                cw1 = self.tester.encode_message(dst_msg)
                
                # Create config and run NVSim
                config_file = self.tester.create_test_config(
                    cw0, cw1, f"{variant_name}_batch_{batch_id}_{src_msg}_{dst_msg}"
                )
                
                result = self.tester.run_nvsim(config_file, timeout=60)
                
                if result['success']:
                    latency = result['write_latency_ns'] or 0.0
                    transition_type = self.tester.classify_transition_type(cw0, cw1)
                    
                    batch_results[(src_msg, dst_msg)] = {
                        'latency': latency,
                        'transition_type': transition_type,
                        'cw0': cw0,
                        'cw1': cw1
                    }
                else:
                    logger.warning(f"Failed transition {src_msg}→{dst_msg}: {result['error']}")
                    batch_results[(src_msg, dst_msg)] = None
                    
                # Clean up config file
                Path(config_file).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Exception processing {src_msg}→{dst_msg}: {e}")
                batch_results[(src_msg, dst_msg)] = None
                
        return batch_results
    
    def generate_variant_dataset(self, byte_index: int, global_state: int, 
                                max_workers: int = 4) -> Dict:
        """Generate complete dataset for one variant"""
        variant_name = f"byte{byte_index}_g{global_state}"
        
        logger.info(f"🚀 Generating {variant_name} variant...")
        logger.info(f"   Byte Index: {byte_index} ({'Low Byte (0-7)' if byte_index == 0 else 'High Byte (8-15)'})")
        logger.info(f"   Global State: {global_state} ({'0x00' if global_state == 0 else '0xFF'})")
        
        # Generate message patterns for this variant
        message_pairs = self.tester.generate_message_pattern(byte_index, global_state)
        logger.info(f"   Generated {len(message_pairs)} message transitions")
        
        # Split into batches for parallel processing
        batch_size = max(1000, len(message_pairs) // max_workers)
        batches = [message_pairs[i:i+batch_size] for i in range(0, len(message_pairs), batch_size)]
        
        logger.info(f"   Processing {len(batches)} batches with {max_workers} workers")
        
        all_results = {}
        completed_batches = 0
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_variant_batch, batch, variant_name, batch_id): batch_id 
                for batch_id, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.update(batch_results)
                    completed_batches += 1
                    
                    progress = (completed_batches / len(batches)) * 100
                    logger.info(f"   Batch {batch_id} complete. Progress: {progress:.1f}% "
                              f"({len(all_results)}/65536 transitions)")
                    
                except Exception as e:
                    logger.error(f"   Batch {batch_id} failed: {e}")
        
        return {
            'variant_name': variant_name,
            'byte_index': byte_index,
            'global_state': global_state,
            'results': all_results,
            'metadata': {
                'total_transitions': len(message_pairs),
                'successful_transitions': len([r for r in all_results.values() if r is not None]),
                'generation_time': datetime.now().isoformat()
            }
        }


def worker_process_variant_batch(args):
    """Worker function for multiprocessing"""
    message_pairs, variant_name, batch_id, byte_index, global_state = args
    
    # Create fresh tester instance in worker process
    tester = SierpinskiVariantTester()
    processor = SierpinskiVariantProcessor(tester)
    
    return processor.process_variant_batch(message_pairs, variant_name, batch_id)


def convert_results_to_matrices(variant_data: Dict) -> Dict:
    """Convert variant results to analysis matrices"""
    results = variant_data['results']
    
    # Initialize matrices
    latency_matrix = np.zeros((256, 256), dtype=float)
    transition_matrix = np.zeros((256, 256), dtype=int)  # 0=NONE, 1=UNIPOLAR, 2=BIPOLAR
    
    # Map transition types to integers
    type_mapping = {'NONE': 0, 'UNIPOLAR': 1, 'BIPOLAR': 2}
    
    for (src_msg, dst_msg), result in results.items():
        if result is not None:
            # Extract byte indices based on variant
            if variant_data['byte_index'] == 0:
                # Byte index 0: varying low byte
                src_idx = src_msg & 0xFF
                dst_idx = dst_msg & 0xFF
            else:
                # Byte index 1: varying high byte  
                src_idx = (src_msg >> 8) & 0xFF
                dst_idx = (dst_msg >> 8) & 0xFF
            
            latency_matrix[src_idx, dst_idx] = result['latency']
            transition_matrix[src_idx, dst_idx] = type_mapping.get(result['transition_type'], 0)
        else:
            # Handle failed transitions
            if variant_data['byte_index'] == 0:
                src_idx = src_msg & 0xFF
                dst_idx = dst_msg & 0xFF
            else:
                src_idx = (src_msg >> 8) & 0xFF
                dst_idx = (dst_msg >> 8) & 0xFF
            
            latency_matrix[src_idx, dst_idx] = np.nan
            transition_matrix[src_idx, dst_idx] = -1  # Failed
    
    return {
        'latency_matrix': latency_matrix,
        'transition_matrix': transition_matrix,
        'type_mapping': type_mapping
    }


def save_variant_dataset(variant_data: Dict, output_dir: str = "sierpinski_4_variants"):
    """Save variant dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    variant_name = variant_data['variant_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to matrices
    matrices = convert_results_to_matrices(variant_data)
    
    # Save matrices
    np.save(output_path / f"{variant_name}_latency_{timestamp}.npy", matrices['latency_matrix'])
    np.save(output_path / f"{variant_name}_transitions_{timestamp}.npy", matrices['transition_matrix'])
    
    # Save as CSV for readability
    np.savetxt(output_path / f"{variant_name}_latency_{timestamp}.csv", 
               matrices['latency_matrix'], delimiter=',', fmt='%.6f')
    np.savetxt(output_path / f"{variant_name}_transitions_{timestamp}.csv", 
               matrices['transition_matrix'], delimiter=',', fmt='%d')
    
    # Save metadata
    metadata = {
        **variant_data['metadata'],
        'variant_name': variant_name,
        'byte_index': variant_data['byte_index'],
        'global_state': variant_data['global_state'],
        'type_mapping': matrices['type_mapping'],
        'files': {
            'latency_matrix': f"{variant_name}_latency_{timestamp}.npy",
            'transition_matrix': f"{variant_name}_transitions_{timestamp}.npy"
        }
    }
    
    with open(output_path / f"{variant_name}_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved {variant_name} to {output_path}")
    return output_path / f"{variant_name}_latency_{timestamp}.npy"


def generate_all_4_variants(max_workers: int = 4):
    """Generate all 4 Sierpinski variants"""
    
    logger.info("🔬 Generating 4 Sierpinski Variants")
    logger.info("=" * 60)
    
    variants = [
        (0, 0, "Byte Index 0, G=0: Vary low byte (0-7), high byte = 0x00"),
        (0, 1, "Byte Index 0, G=1: Vary low byte (0-7), high byte = 0xFF"), 
        (1, 0, "Byte Index 1, G=0: Vary high byte (8-15), low byte = 0x00"),
        (1, 1, "Byte Index 1, G=1: Vary high byte (8-15), low byte = 0xFF")
    ]
    
    tester = SierpinskiVariantTester()
    processor = SierpinskiVariantProcessor(tester)
    
    all_variant_data = {}
    
    for byte_index, global_state, description in variants:
        logger.info(f"\n📊 {description}")
        logger.info("-" * 50)
        
        try:
            variant_data = processor.generate_variant_dataset(
                byte_index, global_state, max_workers
            )
            
            # Save dataset
            saved_file = save_variant_dataset(variant_data)
            
            all_variant_data[f"byte{byte_index}_g{global_state}"] = {
                'data': variant_data,
                'file': saved_file
            }
            
            # Quick statistics
            results = variant_data['results']
            successful = len([r for r in results.values() if r is not None])
            success_rate = successful / len(results) * 100
            
            logger.info(f"✅ {variant_data['variant_name']} complete: "
                       f"{successful}/65536 ({success_rate:.1f}%) successful")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {byte_index}_g{global_state}: {e}")
    
    logger.info(f"\n🎯 All 4 variants generation complete!")
    logger.info(f"💾 Datasets saved to: sierpinski_4_variants/")
    
    return all_variant_data


def main():
    import sys
    
    try:
        print("🔬 '4 Sierpinski Variants Generator")
        print("=" * 50)
        print("Variants:")
        print("  1. Byte Index 0, G=0: Vary low byte (0-7), high byte = 0x00")
        print("  2. Byte Index 0, G=1: Vary low byte (0-7), high byte = 0xFF")  
        print("  3. Byte Index 1, G=0: Vary high byte (8-15), low byte = 0x00")
        print("  4. Byte Index 1, G=1: Vary high byte (8-15), low byte = 0xFF")
        print()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--generate":
            print("🚀 Starting generation of all 4 variants...")
            print("This will generate 4 × 65,536 = 262,144 transitions and may take several hours...")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            
            # Generate all variants
            all_data = generate_all_4_variants(max_workers=6)
            
            print(f"✅ Generation complete! Generated {len(all_data)} variants.")
            return 0
        
        elif len(sys.argv) > 1 and sys.argv[1] == "--test":
            print("🧪 Testing single variant with small subset...")
            
            tester = SierpinskiVariantTester()
            processor = SierpinskiVariantProcessor(tester)
            
            # Test small subset of first variant
            test_pairs = [(i, j) for i in range(4) for j in range(4)]
            
            batch_results = processor.process_variant_batch(test_pairs, "test", 0)
            print(f"✅ Test complete: {len(batch_results)} results")
            
            for (src, dst), result in batch_results.items():
                if result is not None:
                    print(f"  msg {src:04X}→{dst:04X}: {result['latency']:.3f}ns ({result['transition_type']})")
                else:
                    print(f"  msg {src:04X}→{dst:04X}: FAILED")
            
            return 0
        
        else:
            print("📋 Available commands:")
            print("  --generate: Generate all 4 Sierpinski variants") 
            print("  --test: Test with small subset")
            print()
            print("Use --generate to start the full dataset generation")
            return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())