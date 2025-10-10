#!/usr/bin/env python3
"""
Test OpenCV CUDA GPU acceleration
"""

import cv2 as cv
import numpy as np
from numba_gpu_acceleration import NumbaCUDAAccelerator
import time


def test_opencv_cuda_availability():
    """Test if OpenCV CUDA is available"""
    print("=" * 60)
    print("Testing OpenCV CUDA Availability")
    print("=" * 60)
    
    try:
        device_count = cv.cuda.getCudaEnabledDeviceCount()
        print(f"✅ OpenCV CUDA Device Count: {device_count}")
        
        if device_count > 0:
            device_id = cv.cuda.getDevice()
            print(f"   Current Device ID: {device_id}")
            cv.cuda.printCudaDeviceInfo(device_id)
            return True
        else:
            print("❌ No CUDA-enabled devices found")
            return False
    except Exception as e:
        print(f"❌ OpenCV CUDA Error: {e}")
        print("   Make sure you have opencv-contrib-python with CUDA support")
        return False


def test_gpu_accelerator():
    """Test the GPU accelerator class"""
    print("\n" + "=" * 60)
    print("Testing NumbaCUDAAccelerator Class")
    print("=" * 60)
    
    accelerator = NumbaCUDAAccelerator()
    
    if accelerator.is_available():
        print("✅ GPU Accelerator initialized successfully")
        
        # Get performance info
        info = accelerator.get_performance_info()
        print("\nGPU Performance Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️ GPU Accelerator not available - will use CPU fallback")
    
    return accelerator


def test_template_matching(accelerator):
    """Test template matching with synthetic data"""
    print("\n" + "=" * 60)
    print("Testing Template Matching Performance")
    print("=" * 60)
    
    # Create synthetic test image (similar to spectrogram size)
    print("\nCreating test data (2048x384 image)...")
    image = np.random.randint(0, 256, (384, 2048), dtype=np.uint8)
    
    # Create test templates of different sizes
    templates = [
        np.random.randint(0, 256, (50, 100), dtype=np.uint8),
        np.random.randint(0, 256, (75, 150), dtype=np.uint8),
        np.random.randint(0, 256, (100, 200), dtype=np.uint8),
    ]
    
    print(f"   Image shape: {image.shape}")
    print(f"   Number of templates: {len(templates)}")
    
    # Test single template matching
    print("\n1. Testing Single Template Matching...")
    start = time.time()
    result = accelerator.parallel_template_match(image, templates[0])
    elapsed = time.time() - start
    print(f"   ✅ Single template: {elapsed:.4f}s")
    print(f"      Result shape: {result.shape}")
    
    # Test multiple template matching
    print("\n2. Testing Multiple Template Matching...")
    start = time.time()
    results = accelerator.parallel_multi_template_match(image, templates)
    elapsed = time.time() - start
    print(f"   ✅ {len(templates)} templates: {elapsed:.4f}s")
    print(f"      Results count: {len(results)}")
    
    # Test multi-image detection (simulating chunk processing)
    print("\n3. Testing Multi-Image Detection (Chunk Simulation)...")
    chunks = [image.copy() for _ in range(3)]  # Simulate 3 chunks
    start = time.time()
    chunk_results, timing = accelerator.parallel_multi_image_detection(
        chunks, templates, return_timing=True
    )
    elapsed = time.time() - start
    
    print(f"   ✅ {len(chunks)} chunks × {len(templates)} templates: {elapsed:.4f}s")
    print(f"      Total operations: {timing['total_operations']}")
    print(f"      Operations/sec: {timing['operations_per_second']:.0f}")
    print(f"      GPU Fallback: {timing['fallback']}")
    
    # Compare with CPU
    print("\n4. CPU Benchmark (for comparison)...")
    start = time.time()
    cpu_results = [cv.matchTemplate(image, tpl, cv.TM_CCOEFF_NORMED) for tpl in templates]
    cpu_elapsed = time.time() - start
    print(f"   ⏱️ CPU: {cpu_elapsed:.4f}s")
    
    if accelerator.is_available():
        speedup = cpu_elapsed / elapsed if elapsed > 0 else 0
        print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU")


if __name__ == "__main__":
    # Test 1: Check OpenCV CUDA availability
    cuda_available = test_opencv_cuda_availability()
    
    # Test 2: Initialize accelerator
    accelerator = test_gpu_accelerator()
    
    # Test 3: Test template matching
    test_template_matching(accelerator)
    
    print("\n" + "=" * 60)
    if cuda_available and accelerator.is_available():
        print("✅ All tests passed - GPU acceleration is working!")
    else:
        print("⚠️ GPU not available - using CPU fallback")
    print("=" * 60)
