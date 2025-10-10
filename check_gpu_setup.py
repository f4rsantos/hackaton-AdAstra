#!/usr/bin/env python3
"""
Quick Start Guide for OpenCV CUDA GPU Acceleration

Run this to check if your system is ready for GPU acceleration.
"""

import sys

def check_opencv_cuda():
    """Check OpenCV CUDA availability"""
    try:
        import cv2 as cv
        
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv.__version__}")
        
        # Check CUDA support
        try:
            device_count = cv.cuda.getCudaEnabledDeviceCount()
            
            if device_count > 0:
                print(f"\n🚀 OpenCV CUDA is AVAILABLE!")
                print(f"   CUDA Devices: {device_count}")
                
                device_id = cv.cuda.getDevice()
                print(f"   Current Device: {device_id}")
                
                # Test template matcher creation
                try:
                    matcher = cv.cuda.createTemplateMatching(cv.CV_32F, cv.TM_CCOEFF_NORMED)
                    print(f"   ✅ Template Matcher: Ready")
                    return True
                except Exception as e:
                    print(f"   ⚠️ Matcher Error: {e}")
                    return False
            else:
                print(f"\n⚠️ OpenCV CUDA not available (0 devices)")
                print("   Will use CPU fallback")
                return False
                
        except AttributeError:
            print(f"\n❌ OpenCV CUDA module not found")
            print("   This OpenCV build does not include CUDA support")
            print("\n   To enable GPU:")
            print("   1. Install CUDA Toolkit from NVIDIA")
            print("   2. Install opencv-contrib-python with CUDA")
            print("   3. Or use CPU fallback (still works!)")
            return False
            
    except ImportError:
        print("❌ OpenCV not installed")
        print("   Run: pip install opencv-python")
        return False


def check_numpy():
    """Check NumPy availability"""
    try:
        import numpy as np
        print(f"\n✅ NumPy: {np.__version__}")
        return True
    except ImportError:
        print("\n❌ NumPy not installed")
        print("   Run: pip install numpy")
        return False


def check_accelerator():
    """Check if accelerator class works"""
    try:
        from numba_gpu_acceleration import NumbaCUDAAccelerator
        
        print(f"\n✅ NumbaCUDAAccelerator class imported")
        
        # Initialize
        print("\nInitializing GPU Accelerator...")
        accelerator = NumbaCUDAAccelerator()
        
        if accelerator.is_available():
            print("   🚀 GPU Accelerator: READY")
            
            # Get performance info
            info = accelerator.get_performance_info()
            print(f"\n   GPU Info:")
            for key, value in info.items():
                print(f"      {key}: {value}")
            
            return True
        else:
            print("   ⚠️ GPU not available - will use CPU fallback")
            return False
            
    except ImportError as e:
        print(f"\n❌ Failed to import NumbaCUDAAccelerator: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error initializing accelerator: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 70)
    print("OpenCV CUDA GPU Acceleration - System Check")
    print("=" * 70)
    
    checks = [
        ("OpenCV CUDA", check_opencv_cuda),
        ("NumPy", check_numpy),
        ("GPU Accelerator", check_accelerator),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'─' * 70}")
        print(f"Checking: {name}")
        print(f"{'─' * 70}")
        results[name] = check_func()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "⚠️ FALLBACK"
        print(f"   {status}: {name}")
    
    print("\n" + "=" * 70)
    
    if all(results.values()):
        print("🎉 READY FOR GPU ACCELERATION!")
        print("\nYour system is configured for maximum performance.")
        print("Run your chunked detection to see 7-13x speedup!")
    elif results.get("OpenCV CUDA") == False:
        print("⚠️ GPU NOT AVAILABLE - CPU FALLBACK MODE")
        print("\nYour code will still work but use CPU processing.")
        print("This is fine for testing or systems without GPU.")
        print("\nTo enable GPU:")
        print("   1. Install NVIDIA CUDA Toolkit")
        print("   2. Install opencv-contrib-python with CUDA support")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nPlease install missing dependencies.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
