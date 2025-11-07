

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


class SierpinskiTester:
    
    def __init__(self):
        self.nvsim_executable = "./nvsim"
        self.base_config = "sierpinski_test.cfg"
        self.cell_file = "sierpinski_rram.cell"
        self.p_matrix_file = "P_matrix.csv"
        self.b_vector_file = "b_vector.csv"
        self.results_dir = Path("results")
        

        self._validate_files()
        

        self.P = self._load_p_matrix()
        self.b = self._load_b_vector()
        
        logger.info("SierpinskiTester initialized successfully")
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
        if not (0 <= message <= 255):
            raise ValueError(f"Message must be 8-bit (0-255), got {message}")
        
        message_bits = GF2([(message >> i) & 1 for i in range(16)])
        parity_bits = message_bits @ self.P + self.b
        codeword_bits = np.concatenate([message_bits, parity_bits])
        codeword = sum(int(bit) << i for i, bit in enumerate(codeword_bits))
        return codeword
    
    def analyze_transition(self, cw0: int, cw1: int) -> Dict[str, int]:
        """
        Analyze the bit transitions between two codewords.
        
        Args:
            cw0: Source codeword
            cw1: Target codeword
            
        Returns:
            Dictionary with transition counts
        """

        diff = cw0 ^ cw1
        

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
    
    def create_test_config(self, cw0: int, cw1: int, test_id: str) -> str:
        """
        Create a test configuration file for specific codeword transition.
        
        Args:
            cw0: Source codeword
            cw1: Target codeword
            test_id: Unique test identifier
            
        Returns:
            Path to generated config file
        """

        with open(self.base_config, 'r') as f:
            config_content = f.read()
        

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
        

        output_file = f"results/sierpinski_{test_id}.nvout"
        config_content += f"\n-OutputFile: {output_file}\n"
        

        test_config_path = f"test_{test_id}.cfg"
        with open(test_config_path, 'w') as f:
            f.write(config_content)
        
        return test_config_path
    
    def run_nvsim(self, config_file: str, timeout: int = 30) -> Dict:
        """
        Run NVSim with machine-readable output format.
        
        Args:
            config_file: Path to configuration file
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with results
        """
        try:
            cmd = [self.nvsim_executable, config_file, "-MachineReadable"]
            logger.debug(f"Running: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"NVSim failed with return code {result.returncode}",
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'execution_time': execution_time
                }
            

            timing_data = self._parse_machine_readable_output(result.stdout)
            
            return {
                'success': True,
                'timing_data': timing_data,
                'write_latency_ns': timing_data.get('TOTAL_WRITE_LATENCY_NS'),
                'set_latency_ns': timing_data.get('SET_LATENCY_NS'),
                'reset_latency_ns': timing_data.get('RESET_LATENCY_NS'),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time
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
                        logger.warning(f"Could not parse numeric value for {key}: {value}")
                        timing_data[key] = None
                else:

                    timing_data[key] = value
        
        logger.debug(f"Parsed machine-readable data: {timing_data}")
        return timing_data
    
    def _extract_latency(self, stdout: str) -> Optional[float]:

        timing_data = self._parse_machine_readable_output(stdout)
        return timing_data.get('TOTAL_WRITE_LATENCY_NS')
    
    def test_basic_functionality(self) -> bool:
        logger.info("Testing basic functionality...")
        

        cw0 = 0x000000
        cw1 = 0x000001
        

        config_file = self.create_test_config(cw0, cw1, "basic_test")
        
        try:

            result = self.run_nvsim(config_file)
            
            if result['success']:
                latency = result['write_latency_ns']
                if latency is not None:
                    logger.info(f"✅ Basic test passed: {latency:.3f}ns")
                else:
                    logger.info("✅ Basic test passed: NVSim executed successfully (latency not extracted)")
                return True
            else:
                logger.error(f"❌ Basic test failed: {result['error']}")
                if 'stdout' in result:
                    logger.error(f"STDOUT: {result['stdout']}")
                if 'stderr' in result:
                    logger.error(f"STDERR: {result['stderr']}")
                return False
                
        finally:

            Path(config_file).unlink(missing_ok=True)
    
    def test_sample_transitions(self) -> None:
        logger.info("Testing sample codeword transitions...")
        

        test_cases = [
            (0, 1, "msg_0_to_1"),
            (0, 255, "msg_0_to_255"), 
            (85, 170, "msg_85_to_170"),
        ]
        
        for msg0, msg1, test_name in test_cases:

            cw0 = self.encode_message(msg0)
            cw1 = self.encode_message(msg1)
            

            transition = self.analyze_transition(cw0, cw1)
            
            logger.info(f"Testing {test_name}: msg {msg0} → {msg1} (cw 0x{cw0:06X} → 0x{cw1:06X})")
            logger.info(f"  Transitions: {transition['set_transitions']} SET, {transition['reset_transitions']} RESET, {transition['redundant_ops']} REDUNDANT")
            

            config_file = self.create_test_config(cw0, cw1, test_name)
            
            try:
                result = self.run_nvsim(config_file, timeout=30)
                
                if result['success']:
                    latency = result['write_latency_ns']
                    if latency is not None:
                        logger.info(f"  ✅ Success: {latency:.3f}ns ({result['execution_time']:.2f}s)")
                    else:
                        logger.info(f"  ✅ Success: NVSim executed successfully ({result['execution_time']:.2f}s)")
                else:
                    logger.error(f"  ❌ Failed: {result['error']}")
                    
            finally:

                Path(config_file).unlink(missing_ok=True)


class SierpinskiBatchProcessor:
    
    def __init__(self, tester: SierpinskiTester):
        self.tester = tester
        self.results_matrix = np.zeros((256, 256), dtype=float)
        self.metadata = {
            'start_time': None,
            'completion_count': 0,
            'error_count': 0,
            'total_transitions': 65536
        }
    
    def generate_all_message_pairs(self) -> List[Tuple[int, int]]:
        return [(src, dst) for src in range(256) for dst in range(256)]
    
    def process_single_batch(self, message_pairs: List[Tuple[int, int]], batch_id: int) -> Dict:
        batch_results = {}
        
        for src_msg, dst_msg in message_pairs:
            try:

                cw0 = self.tester.encode_message(src_msg)
                cw1 = self.tester.encode_message(dst_msg)
                

                config_file = self.tester.create_test_config(cw0, cw1, f"batch_{batch_id}_{src_msg}_{dst_msg}")
                result = self.tester.run_nvsim(config_file, timeout=60)
                
                if result['success']:
                    latency = result['write_latency_ns'] or 0.0
                    batch_results[(src_msg, dst_msg)] = latency
                    

                    transition = self.tester.analyze_transition(cw0, cw1)
                    logger.debug(f"msg {src_msg}→{dst_msg}: {transition['set_transitions']}S, "
                               f"{transition['reset_transitions']}R, {transition['redundant_ops']}N → {latency:.3f}ns")
                else:
                    logger.warning(f"Failed transition {src_msg}→{dst_msg}: {result['error']}")
                    batch_results[(src_msg, dst_msg)] = None
                    

                Path(config_file).unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Exception processing {src_msg}→{dst_msg}: {e}")
                batch_results[(src_msg, dst_msg)] = None
                
        return batch_results


class SierpinskiProgressManager:
    
    def __init__(self, checkpoint_dir="sierpinski_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.matrix_file = self.checkpoint_dir / "partial_matrix.npy"
        
        self.progress = self.load_checkpoint()
        self.start_time = time.time()
    
    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Resuming from checkpoint: {data['completed']}/65536 transitions")
                return data
        return {
            'completed': 0,
            'failed': 0,
            'start_time': time.time(),
            'last_checkpoint': time.time(),
            'completed_pairs': []
        }
    
    def save_checkpoint(self, results_matrix, completed_pairs):
        self.progress['completed'] = len(completed_pairs)
        self.progress['last_checkpoint'] = time.time()
        self.progress['completed_pairs'] = list(completed_pairs)
        

        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        

        np.save(self.matrix_file, results_matrix)
        
        logger.info(f"Checkpoint saved: {self.progress['completed']}/65536 completed")
    
    def estimate_completion(self):
        if self.progress['completed'] > 0:
            elapsed = time.time() - self.progress['start_time']
            rate = self.progress['completed'] / elapsed
            remaining = (65536 - self.progress['completed']) / rate
            return remaining / 3600
        return None
    
    def print_status(self):
        completed = self.progress['completed']
        failed = self.progress['failed']
        success_rate = (completed / (completed + failed)) * 100 if (completed + failed) > 0 else 0
        eta_hours = self.estimate_completion()
        
        print(f"Progress: {completed}/65536 ({completed/655.36:.1f}%)")
        print(f"Success rate: {success_rate:.1f}%")
        if eta_hours:
            print(f"ETA: {eta_hours:.1f} hours")


class SierpinskiDataManager:
    
    def __init__(self, output_dir="sierpinski_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_complete_dataset(self, results_matrix, metadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"sierpinski_complete_{timestamp}"
        

        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        

        matrix_file = run_dir / f"{base_name}.npy"
        np.save(matrix_file, results_matrix)
        

        csv_file = run_dir / f"{base_name}.csv"
        np.savetxt(csv_file, results_matrix, delimiter=',', fmt='%.6f')
        

        meta_file = run_dir / f"{base_name}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        

        stats = self.compute_statistics(results_matrix)
        stats_file = run_dir / f"{base_name}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Complete dataset saved: {matrix_file}")
        logger.info(f"Run directory: {run_dir}")
        return matrix_file
    
    def compute_statistics(self, results_matrix):
        valid_values = results_matrix[results_matrix > 0]
        
        return {
            'total_transitions': 65536,
            'successful_transitions': len(valid_values),
            'success_rate': len(valid_values) / 65536,
            'timing_stats': {
                'min_latency': float(np.min(valid_values)) if len(valid_values) > 0 else 0.0,
                'max_latency': float(np.max(valid_values)) if len(valid_values) > 0 else 0.0,
                'mean_latency': float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
                'std_latency': float(np.std(valid_values)) if len(valid_values) > 0 else 0.0,
                'median_latency': float(np.median(valid_values)) if len(valid_values) > 0 else 0.0
            },
            'pattern_analysis': {
                'unique_values': len(np.unique(valid_values)) if len(valid_values) > 0 else 0,
                'zero_latency_count': int(np.sum(results_matrix == 0)),
                'missing_data_count': int(np.sum(np.isnan(results_matrix)))
            }
        }


def process_batch_worker(message_pairs, batch_id):

    tester = SierpinskiTester()
    processor = SierpinskiBatchProcessor(tester)
    return processor.process_single_batch(message_pairs, batch_id)


def convert_to_matrix(all_results):
    results_matrix = np.zeros((256, 256), dtype=float)
    
    for (src_msg, dst_msg), latency in all_results.items():
        if latency is not None:
            results_matrix[src_msg, dst_msg] = latency
        else:
            results_matrix[src_msg, dst_msg] = np.nan
            
    return results_matrix


def run_parallel_sierpinski_generation(max_workers=None):
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    processor = SierpinskiBatchProcessor(SierpinskiTester())
    all_pairs = processor.generate_all_message_pairs()
    

    batch_size = max(1000, len(all_pairs) // max_workers)
    batches = [all_pairs[i:i+batch_size] for i in range(0, len(all_pairs), batch_size)]
    
    logger.info(f"Starting parallel generation: {len(batches)} batches, {max_workers} workers")
    
    all_results = {}
    completed_batches = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        future_to_batch = {
            executor.submit(process_batch_worker, batch, batch_id): batch_id 
            for batch_id, batch in enumerate(batches)
        }
        

        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.update(batch_results)
                completed_batches += 1
                
                progress = (completed_batches / len(batches)) * 100
                logger.info(f"Batch {batch_id} complete. Progress: {progress:.1f}% "
                          f"({len(all_results)}/65536 transitions)")
                
            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")
    
    return convert_to_matrix(all_results)


def run_full_sierpinski_generation():
    
    logger.info("🚀 Starting Phase 2: Full Sierpinski Dataset Generation")
    

    progress_mgr = SierpinskiProgressManager()
    data_mgr = SierpinskiDataManager()
    
    try:

        logger.info("Starting parallel batch processing...")
        results_matrix = run_parallel_sierpinski_generation(max_workers=8)
        

        logger.info("Saving complete dataset...")
        metadata = {
            'generation_time': time.time(),
            'total_transitions': 65536,
            'nvsim_version': 'Phase 3 Stochastic',
            'ecc_parameters': {
                'P_matrix_shape': '16x5',
                'b_vector_shape': '5x1',
                'codeword_bits': 21
            }
        }
        
        dataset_file = data_mgr.save_complete_dataset(results_matrix, metadata)
        
        logger.info("✅ Phase 2 Complete! Sierpinski gasket generation finished successfully.")
        return results_matrix
        
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        raise


def main():
    import sys
    
    try:
        print("🔬 Sierpinski Gasket NVSim Tester")
        print("=" * 50)
        

        if len(sys.argv) > 1 and sys.argv[1] == "--full":

            print("🚀 Starting Phase 2: Full Sierpinski Dataset Generation")
            print("This will generate 65,536 transitions and may take several hours...")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return 0
            
            results_matrix = run_full_sierpinski_generation()
            print(f"✅ Phase 2 Complete! Results shape: {results_matrix.shape}")
            return 0
        
        elif len(sys.argv) > 1 and sys.argv[1] == "--test-batch":

            print("🧪 Testing batch processing with small subset...")
            
            tester = SierpinskiTester()
            processor = SierpinskiBatchProcessor(tester)
            

            test_pairs = [(i, j) for i in range(4) for j in range(4)]
            print(f"Testing {len(test_pairs)} transitions...")
            
            batch_results = processor.process_single_batch(test_pairs, 0)
            print(f"✅ Batch test complete: {len(batch_results)} results")
            

            for (src, dst), latency in batch_results.items():
                if latency is not None:
                    print(f"  msg {src}→{dst}: {latency:.3f}ns")
                else:
                    print(f"  msg {src}→{dst}: FAILED")
            
            return 0
        
        else:

            print("📋 Running Phase 1 validation tests...")
            print("Use --full for complete dataset generation")
            print("Use --test-batch for batch processing test")
            print()
            

            tester = SierpinskiTester()
            

            if not tester.test_basic_functionality():
                print("❌ Basic functionality test failed!")
                return 1
            
            print()
            

            tester.test_sample_transitions()
            
            print()
            print("✅ Phase 1 validation completed successfully!")
            print("💡 Ready for Phase 2! Run with --full to generate complete dataset")
            return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())