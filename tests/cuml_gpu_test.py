#!/usr/bin/env python3
"""
Quick GPU Test - Fast validation that GPU clustering is working

This is a minimal test to quickly verify GPU acceleration is working.
Run this before starting large clustering jobs.

Usage:
    python cuml_gpu_test.py
"""

import sys
import time
import numpy as np


def test_gpu():
    """Quick GPU functionality test."""
    print("\n" + "="*70)
    print("QUICK GPU TEST")
    print("="*70)
    
    # Test 1: Import cuML
    print("\n[1/5] Testing cuML import...")
    try:
        import cuml
        print(f"      ✓ cuML {cuml.__version__} imported successfully")
    except ImportError as e:
        print(f"      ✗ Failed to import cuML: {e}")
        print("\n      Install with:")
        print("      conda install -c rapidsai -c conda-forge cuml")
        return False
    
    # Test 2: Import CuPy
    print("\n[2/5] Testing CuPy import...")
    try:
        import cupy as cp
        print(f"      ✓ CuPy {cp.__version__} imported successfully")
    except ImportError as e:
        print(f"      ✗ Failed to import CuPy: {e}")
        return False
    
    # Test 3: GPU Device Info
    print("\n[3/5] Checking GPU devices...")
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices == 0:
            print(f"      ✗ No CUDA devices found")
            return False
        
        print(f"      ✓ Found {n_devices} GPU device(s)")
        
        for i in range(n_devices):
            props = cp.cuda.runtime.getDeviceProperties(i)
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            print(f"        GPU {i}: {props['name'].decode()}")
            print(f"        Memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total")
    
    except Exception as e:
        print(f"      ✗ Failed to get GPU info: {e}")
        return False
    
    # Test 4: Simple CuPy operation
    print("\n[4/5] Testing CuPy array operations...")
    try:
        x = cp.array([1, 2, 3, 4, 5])
        y = x * 2
        result = cp.asnumpy(y)
        expected = np.array([2, 4, 6, 8, 10])
        
        if np.array_equal(result, expected):
            print(f"      ✓ CuPy operations working correctly")
        else:
            print(f"      ✗ CuPy operation gave wrong result")
            return False
    
    except Exception as e:
        print(f"      ✗ CuPy operation failed: {e}")
        return False
    
    # Test 5: GPU KMeans clustering
    print("\n[5/5] Testing GPU KMeans clustering...")
    try:
        from cuml.cluster import KMeans
        
        # Generate test data
        n_samples = 10000
        n_features = 128
        n_clusters = 5
        
        X = cp.random.randn(n_samples, n_features).astype(cp.float32)
        
        # Run clustering
        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        gpu_time = time.time() - start
        
        # Verify results
        labels_cpu = cp.asnumpy(labels)
        unique_labels = np.unique(labels_cpu)
        
        if len(unique_labels) == n_clusters:
            print(f"      ✓ GPU KMeans completed successfully")
            print(f"        - Clustered {n_samples:,} points in {gpu_time:.3f}s")
            print(f"        - Found {len(unique_labels)} clusters (expected {n_clusters})")
            print(f"        - Performance: {n_samples/gpu_time:.0f} points/sec")
        else:
            print(f"      ⚠ KMeans returned {len(unique_labels)} clusters (expected {n_clusters})")
            return False
        
        # Compare with CPU for validation
        print("\n      Comparing with CPU implementation...")
        from sklearn.cluster import MiniBatchKMeans
        
        X_cpu = cp.asnumpy(X)
        start = time.time()
        kmeans_cpu = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        labels_cpu_ref = kmeans_cpu.fit_predict(X_cpu)
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"        - CPU time: {cpu_time:.3f}s")
        print(f"        - GPU time: {gpu_time:.3f}s")
        print(f"        - Speedup: {speedup:.1f}x")
        
        if speedup > 1.5:
            print(f"      ✓ GPU is significantly faster than CPU")
        elif speedup > 0.5:
            print(f"      ⚠ GPU speedup is marginal (may be overhead for small data)")
        else:
            print(f"      ⚠ GPU is slower than CPU (check GPU utilization)")
    
    except Exception as e:
        print(f"      ✗ GPU KMeans failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    success = test_gpu()
    
    print("\n" + "="*70)
    if success:
        print("✓ ALL TESTS PASSED - GPU ACCELERATION IS READY")
        print("\nYou can now use:")
        print("  python cluster_mesh_cuml.py --dataset rats_001 --use-gpu --visualize")
    else:
        print("✗ SOME TESTS FAILED - GPU ACCELERATION NOT AVAILABLE")
        print("\nThe clustering script will fall back to CPU mode.")
        print("To fix GPU issues, check the error messages above.")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())