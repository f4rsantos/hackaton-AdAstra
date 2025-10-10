import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
import psutil
import platform
import gc

# Import PyTorch GPU accelerator
try:
    from pytorch_gpu_acceleration import PyTorchGPUAccelerator
except ImportError:
    PyTorchGPUAccelerator = None

# Import ONNX DNN accelerator
try:
    from onnx_dnn_acceleration import ONNXDNNAccelerator
except ImportError:
    ONNXDNNAccelerator = None

# Import Smart Backend Selector
try:
    from smart_backend_selector import SmartBackendSelector
except ImportError:
    SmartBackendSelector = None

# Import SigMF Processor
try:
    from sigmf_processor import SigMFProcessor, find_sigmf_meta_for_image
except ImportError:
    SigMFProcessor = None
    find_sigmf_meta_for_image = None

# Timing utilities for latency measurement
class TimingTracker:
    """Track timing information for detection processes"""
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        self.phase_start = None
        self.current_phase = None
        
    def start_tracking(self):
        """Start the overall timing"""
        self.start_time = time.time()
        self.phase_times = {}
        
    def start_phase(self, phase_name):
        """Start timing a specific phase"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = time.time()
        
    def end_phase(self):
        """End timing the current phase"""
        if self.current_phase and self.phase_start:
            elapsed = time.time() - self.phase_start
            self.phase_times[self.current_phase] = elapsed
            self.current_phase = None
            self.phase_start = None
            
    def get_total_time(self):
        """Get total elapsed time since tracking started"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
        
    def get_timing_summary(self):
        """Get a formatted summary of all timings"""
        if not self.start_time:
            return {}
            
        # End current phase if active
        if self.current_phase:
            self.end_phase()
            
        total_time = self.get_total_time()
        
        summary = {
            'total_time': total_time,
            'phase_times': self.phase_times.copy(),
            'formatted_total': self.format_time(total_time),
            'formatted_phases': {phase: self.format_time(t) for phase, t in self.phase_times.items()}
        }
        
        return summary
        
    @staticmethod
    def format_time(seconds):
        """Format time in a human-readable way"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"

@contextmanager
def time_phase(tracker, phase_name):
    """Context manager for timing a phase"""
    tracker.start_phase(phase_name)
    try:
        yield
    finally:
        tracker.end_phase()

# System monitoring utilities for adaptive performance
class SystemMonitor:
    """Monitor system resources for adaptive performance scaling"""
    
    def __init__(self):
        self.cpu_threshold = 90.0  # CPU usage threshold for throttling
        self.memory_threshold = 85.0  # Memory usage threshold
        self.update_interval = 1.0  # Seconds between updates
        self.last_update = 0
        self.cached_stats = {}
        
    def get_system_stats(self, force_update=False):
        """Get current system resource usage"""
        current_time = time.time()
        
        # Use cached stats if recent enough
        if not force_update and current_time - self.last_update < self.update_interval:
            return self.cached_stats
        
        try:
            stats = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'timestamp': current_time
            }
            
            # Add temperature if available (mostly Linux)
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        avg_temp = np.mean([temp.current for sensor_temps in temps.values() 
                                          for temp in sensor_temps if temp.current])
                        stats['cpu_temperature'] = avg_temp
                    else:
                        stats['cpu_temperature'] = None
                else:
                    stats['cpu_temperature'] = None
            except:
                stats['cpu_temperature'] = None
            
            self.cached_stats = stats
            self.last_update = current_time
            return stats
            
        except Exception as e:
            # Fallback values if monitoring fails
            return {
                'cpu_percent': 50.0,
                'memory_percent': 50.0,
                'available_memory_gb': 4.0,
                'cpu_count': cpu_count(),
                'cpu_count_physical': cpu_count() // 2,
                'cpu_freq': 2400,
                'load_average': 1.0,
                'cpu_temperature': None,
                'timestamp': current_time,
                'error': str(e)
            }
    
    def should_throttle(self):
        """Check if system should throttle performance"""
        stats = self.get_system_stats()
        
        # Throttle if CPU or memory usage is too high
        cpu_overload = stats['cpu_percent'] > self.cpu_threshold
        memory_overload = stats['memory_percent'] > self.memory_threshold
        
        # Throttle if temperature is too high (if available)
        temp_overload = False
        if stats.get('cpu_temperature'):
            temp_overload = stats['cpu_temperature'] > 80  # 80°C threshold
        
        return cpu_overload or memory_overload or temp_overload
    
    def get_recommended_workers(self, max_workers=None):
        """Get recommended number of workers based on system load"""
        stats = self.get_system_stats()
        
        # Base recommendation on CPU cores
        physical_cores = stats['cpu_count_physical']
        logical_cores = stats['cpu_count']
        
        # Start with conservative estimate
        if max_workers is None:
            max_workers = min(8, logical_cores)  # Cap at 8 workers
        
        # Adjust based on current load
        cpu_usage = stats['cpu_percent']
        memory_usage = stats['memory_percent']
        
        # Scale down workers based on current load
        load_factor = 1.0
        
        if cpu_usage > 80:
            load_factor *= 0.5  # Halve workers if CPU high
        elif cpu_usage > 60:
            load_factor *= 0.75  # Reduce workers if CPU moderate
        
        if memory_usage > 80:
            load_factor *= 0.5  # Halve workers if memory high
        elif memory_usage > 60:
            load_factor *= 0.8  # Reduce workers if memory moderate
        
        # Calculate recommended workers
        recommended = max(1, int(max_workers * load_factor))
        
        return min(recommended, max_workers)

class GPUDetector:
    """Detect and manage GPU acceleration capabilities"""
    
    def __init__(self):
        self.cuda_available = False
        self.opencl_available = False
        self.pytorch_cuda_available = False
        self.gpu_info = {}
        self.detection_attempted = False
        
        # Initialize accelerators
        self.pytorch_gpu_accelerator = None
        self.onnx_dnn_accelerator = None
        self.smart_backend_selector = None
        
        if PyTorchGPUAccelerator:
            try:
                self.pytorch_gpu_accelerator = PyTorchGPUAccelerator()
            except Exception as e:
                pass
        
        if ONNXDNNAccelerator:
            try:
                self.onnx_dnn_accelerator = ONNXDNNAccelerator()
            except Exception as e:
                pass
        
        # Initialize smart backend selector for optimal performance
        if SmartBackendSelector:
            try:
                self.smart_backend_selector = SmartBackendSelector(fast_mode=True, lazy_init=True)
            except Exception as e:
                pass
        
    def detect_gpu_capabilities(self):
        """Detect available GPU acceleration options with robust hardware detection"""
        if self.detection_attempted:
            return self.cuda_available, self.opencl_available
        
        self.detection_attempted = True
        
        # Method 1: Check PyTorch CUDA first (most reliable and user-friendly)
        self._check_pytorch_cuda()
        
        # Method 2: Check for actual NVIDIA hardware
        nvidia_hardware_detected = self._check_nvidia_hardware()
        
        try:
            # Check OpenCV CUDA support only if hardware is detected
            if nvidia_hardware_detected:
                cuda_device_count = cv.cuda.getCudaEnabledDeviceCount()
                self.cuda_available = cuda_device_count > 0
                
                if self.cuda_available:
                    self.gpu_info['cuda_devices'] = cuda_device_count
                    try:
                        # Try to get device info to verify it's actually working
                        device_info = cv.cuda.DeviceInfo(0)  # Check first device
                        self.gpu_info['cuda_compute_capability'] = f"{device_info.majorVersion()}.{device_info.minorVersion()}"
                        self.gpu_info['cuda_memory'] = device_info.totalGlobalMem() / (1024**3)  # GB
                        self.gpu_info['cuda_name'] = device_info.name()
                        
                        # Verify CUDA is actually functional by trying a simple operation
                        if not self._test_cuda_functionality():
                            self.cuda_available = False
                            self.gpu_info['cuda_error'] = "CUDA functionality test failed"
                            
                    except Exception as e:
                        self.cuda_available = False
                        self.gpu_info['cuda_error'] = f"Device info error: {str(e)}"
            else:
                self.cuda_available = False
                self.gpu_info['cuda_error'] = "No NVIDIA hardware detected"
                
        except Exception as e:
            self.cuda_available = False
            self.gpu_info['cuda_error'] = f"CUDA detection error: {str(e)}"
        
        # OpenCL detection (more permissive but with validation)
        try:
            if cv.ocl.haveOpenCL():
                # Try to get OpenCL device info to verify it's working
                opencl_devices = self._get_opencl_device_info()
                if opencl_devices:
                    self.opencl_available = True
                    cv.ocl.setUseOpenCL(True)
                    self.gpu_info['opencl_devices'] = opencl_devices
                    
                    # Test OpenCL functionality
                    if not self._test_opencl_functionality():
                        self.opencl_available = False
                        self.gpu_info['opencl_error'] = "OpenCL functionality test failed"
                else:
                    self.opencl_available = False
                    self.gpu_info['opencl_error'] = "No OpenCL devices found"
            else:
                self.opencl_available = False
                self.gpu_info['opencl_error'] = "OpenCL not available"
        except Exception as e:
            self.opencl_available = False
            self.gpu_info['opencl_error'] = f"OpenCL detection error: {str(e)}"
        
        return self.cuda_available, self.opencl_available
    
    def _check_pytorch_cuda(self):
        """Check PyTorch CUDA availability (most user-friendly option)"""
        try:
            import torch
            if torch.cuda.is_available():
                self.pytorch_cuda_available = True
                device_count = torch.cuda.device_count()
                self.gpu_info['pytorch_cuda_available'] = True
                self.gpu_info['pytorch_cuda_devices'] = device_count
                self.gpu_info['pytorch_cuda_device_name'] = torch.cuda.get_device_name(0)
                self.gpu_info['pytorch_version'] = torch.__version__
                self.gpu_info['pytorch_cuda_version'] = torch.version.cuda
                
                # Test basic PyTorch GPU operation
                try:
                    test_tensor = torch.ones(10, 10, device='cuda')
                    _ = test_tensor * 2  # Simple operation
                    self.gpu_info['pytorch_functional'] = True
                except Exception as e:
                    self.pytorch_cuda_available = False
                    self.gpu_info['pytorch_cuda_error'] = f"PyTorch CUDA test failed: {e}"
            else:
                self.pytorch_cuda_available = False
                self.gpu_info['pytorch_cuda_error'] = "PyTorch CUDA not available"
        except ImportError:
            self.pytorch_cuda_available = False
            self.gpu_info['pytorch_cuda_error'] = "PyTorch not installed"
        except Exception as e:
            self.pytorch_cuda_available = False
            self.gpu_info['pytorch_cuda_error'] = f"PyTorch detection error: {e}"
    
    def _check_nvidia_hardware(self):
        """Check for actual NVIDIA hardware using multiple methods"""
        try:
            # Method 1: Try nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                return device_count > 0
            except:
                pass
            
            # Method 2: Try nvidia-smi command
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    count = int(result.stdout.strip())
                    return count > 0
            except:
                pass
            
            # Method 3: Check Windows device manager (Windows only)
            try:
                import platform
                if platform.system() == "Windows":
                    import wmi
                    c = wmi.WMI()
                    for gpu in c.Win32_VideoController():
                        if gpu.Name and 'nvidia' in gpu.Name.lower():
                            return True
            except:
                pass
            
            # Method 4: Check for NVIDIA drivers in common locations
            try:
                import os
                import platform
                
                if platform.system() == "Windows":
                    # Check for NVIDIA driver files
                    nvidia_paths = [
                        "C:\\Windows\\System32\\nvapi64.dll",
                        "C:\\Windows\\System32\\nvcuda.dll",
                        "C:\\Program Files\\NVIDIA Corporation"
                    ]
                    for path in nvidia_paths:
                        if os.path.exists(path):
                            return True
                
                elif platform.system() == "Linux":
                    # Check for NVIDIA devices in /proc
                    nvidia_paths = [
                        "/proc/driver/nvidia",
                        "/dev/nvidia0",
                        "/sys/module/nvidia"
                    ]
                    for path in nvidia_paths:
                        if os.path.exists(path):
                            return True
            except:
                pass
            
        except Exception:
            pass
        
        return False
    
    def _get_opencl_device_info(self):
        """Get OpenCL device information"""
        try:
            # This is a basic check - OpenCV doesn't expose detailed OpenCL device info easily
            # We'll rely on the haveOpenCL() check and functionality test
            return ["OpenCL Device Available"]
        except:
            return []
    
    def _test_cuda_functionality(self):
        """Test if CUDA is actually functional with a simple operation"""
        try:
            # Create a small test matrix and upload to GPU
            test_mat = np.ones((10, 10), dtype=np.float32)
            gpu_mat = cv.cuda_GpuMat()
            gpu_mat.upload(test_mat)
            
            # Download back to verify it works
            result = gpu_mat.download()
            return np.array_equal(test_mat, result)
        except:
            return False
    
    def _test_opencl_functionality(self):
        """Test if OpenCL is actually functional"""
        try:
            # Create a simple test with OpenCL
            test_mat = np.ones((10, 10), dtype=np.uint8)
            # Try a simple OpenCL operation (Gaussian blur is usually supported)
            result = cv.GaussianBlur(test_mat, (3, 3), 0)
            return result is not None and result.shape == test_mat.shape
        except:
            return False
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        if not self.detection_attempted:
            self.detect_gpu_capabilities()
        return self.gpu_info.copy()
    
    def is_gpu_available(self):
        """Check if GPU acceleration is available and likely to provide performance benefits"""
        if not self.detection_attempted:
            self.detect_gpu_capabilities()
        
        # Prefer PyTorch CUDA (most reliable and user-friendly)
        if self.pytorch_cuda_available:
            return True
            
        # Fallback to OpenCV CUDA if available
        if self.cuda_available:
            return True
        
        # Disable OpenCL-only recommendations for now
        # OpenCL can be unreliable and often slower than optimized CPU parallel processing
        return False

# Performance mode definitions
class PerformanceMode:
    """Performance mode configurations with aggressive worker scaling"""
    
    BATTERY = {
        'name': 'Battery Saver',
        'max_workers': 1,
        'worker_multiplier': 0.25,
        'gpu_enabled': False,
        'quality_factor': 0.7,
        'frame_skip_factor': 2,
        'description': 'Minimal resource usage for battery-powered devices'
    }
    
    BALANCED = {
        'name': 'Balanced',
        'max_workers': 4,
        'worker_multiplier': 1.0,  # 1x cpu_count
        'gpu_enabled': True,
        'quality_factor': 1.0,
        'frame_skip_factor': 1,
        'description': 'Good balance with GPU acceleration for most workloads'
    }
    
    PERFORMANCE = {
        'name': 'Performance',
        'max_workers': 8,
        'worker_multiplier': 1.5,  # 1.5x cpu_count
        'gpu_enabled': True,
        'quality_factor': 1.1,
        'frame_skip_factor': 1,
        'description': 'GPU-first approach with optimized CPU fallback'
    }
    
    MAXIMUM = {
        'name': 'Maximum',
        'max_workers': 16,
        'worker_multiplier': 2.0,  # 2x cpu_count for maximum throughput
        'gpu_enabled': True,
        'quality_factor': 1.0,
        'frame_skip_factor': 1,
        'description': 'Maximum parallel processing - 2x CPU cores for I/O bound tasks'
    }
    
    @classmethod
    def get_all_modes(cls):
        return [cls.BATTERY, cls.BALANCED, cls.PERFORMANCE, cls.MAXIMUM]
    
    @classmethod
    def get_mode_by_name(cls, name):
        for mode in cls.get_all_modes():
            if mode['name'] == name:
                return mode
        return cls.BALANCED  # Default fallback

# Parallel processing utilities for speed optimization
class ParallelDetectionConfig:
    """Enhanced configuration for parallel template matching with smart scaling"""
    def __init__(self):
        # Basic parallel processing settings
        self.enabled = False
        self.num_quadrants = 4  # 2x2 grid
        self.overlap_percentage = 0.15  # 15% overlap
        self.min_image_size = 100  # GPU benefits even on small images with modern cards
        self.use_threading = True  # Use threading instead of multiprocessing for OpenCV
        self.max_workers = min(4, cpu_count())  # Limit workers to prevent resource exhaustion
        
        # Smart scaling settings
        self.adaptive_scaling = True  # Enable adaptive performance scaling
        self.performance_mode = 'Balanced'  # Performance mode preset
        self.custom_worker_count = None  # Manual worker override
        
        # GPU acceleration settings
        self.gpu_enabled = True  # Enable GPU acceleration if available
        self.prefer_cuda = True  # Prefer CUDA over OpenCL
        self.gpu_memory_limit = 2.0  # GB of GPU memory to use
        
        # Quality vs speed settings
        self.quality_factor = 1.0  # 0.5-1.5, affects detection sensitivity
        self.frame_skip_enabled = False  # Skip frames under high load
        self.frame_skip_factor = 1  # Skip every N frames
        
        # Resource monitoring
        self.monitor_resources = True  # Monitor CPU/memory usage
        self.auto_throttle = True  # Automatically reduce load if system stressed
        self.throttle_threshold = 90.0  # CPU/memory % to trigger throttling
        
        # Advanced settings
        self.batch_processing = False  # Process multiple images in batches
        self.batch_size = 4  # Images per batch
        self.memory_cleanup = True  # Force garbage collection between batches
        
        # System monitoring instances
        self.system_monitor = SystemMonitor()
        self.gpu_detector = GPUDetector()
        
        # Runtime state
        self.current_workers = self.max_workers
        self.last_adaptation = 0
        self.adaptation_interval = 5.0  # Seconds between adaptations
        
    def initialize(self):
        """Initialize GPU detection and system monitoring"""
        if self.gpu_enabled:
            self.gpu_detector.detect_gpu_capabilities()
        
        # Auto-configure based on system capabilities
        self.auto_configure()
    
    def auto_configure(self):
        """Automatically configure settings based on system capabilities"""
        stats = self.system_monitor.get_system_stats()
        
        # Adjust max workers based on CPU cores and memory
        physical_cores = stats['cpu_count_physical']
        logical_cores = stats['cpu_count']
        available_memory = stats['available_memory_gb']
        
        # More aggressive settings for better performance
        if physical_cores <= 2 or available_memory < 4:
            # Weak system: Conservative
            self.max_workers = min(2, logical_cores)
            self.performance_mode = 'Battery'
            self.batch_size = 2
        elif physical_cores <= 4 or available_memory < 8:
            # Medium system: Balanced with logical cores
            self.max_workers = logical_cores  # Use all logical cores
            self.performance_mode = 'Balanced'
            self.batch_size = 4
        else:
            # Powerful system: Maximum throughput with hyperthreading
            self.max_workers = min(logical_cores * 2, 16)  # 2x logical cores, capped at 16
            self.performance_mode = 'Performance'
            self.batch_size = 8
        
        # Apply performance mode settings
        self.apply_performance_mode()
        
        # Check GPU capabilities
        if self.gpu_enabled:
            gpu_available = self.gpu_detector.is_gpu_available()
            if not gpu_available:
                self.gpu_enabled = False
    
    def apply_performance_mode(self):
        """Apply settings from selected performance mode"""
        mode = PerformanceMode.get_mode_by_name(self.performance_mode)
        
        if not self.custom_worker_count:  # Only if not manually set
            # Calculate workers based on CPU count and mode multiplier
            total_cores = cpu_count()  # Logical cores (includes hyperthreading)
            calculated_workers = int(total_cores * mode['worker_multiplier'])
            
            # Cap at mode's max_workers to prevent excessive threads
            self.max_workers = min(mode['max_workers'], calculated_workers)
            
            # Ensure at least 1 worker
            self.max_workers = max(1, self.max_workers)
        
        self.gpu_enabled = mode['gpu_enabled'] and self.gpu_enabled
        self.quality_factor = mode['quality_factor']
        self.frame_skip_factor = mode['frame_skip_factor']
        self.frame_skip_enabled = mode['frame_skip_factor'] > 1
    
    def adapt_to_system_load(self, force=False):
        """Adapt configuration based on current system load"""
        if not self.adaptive_scaling:
            return False
        
        current_time = time.time()
        if not force and current_time - self.last_adaptation < self.adaptation_interval:
            return False
        
        self.last_adaptation = current_time
        
        # Get current system stats
        stats = self.system_monitor.get_system_stats()
        
        # Check if throttling is needed
        should_throttle = self.system_monitor.should_throttle()
        
        # Get recommended workers
        recommended_workers = self.system_monitor.get_recommended_workers(self.max_workers)
        
        # Update current workers
        old_workers = self.current_workers
        self.current_workers = recommended_workers
        
        # Force garbage collection if memory is high
        if self.memory_cleanup and stats['memory_percent'] > 70:
            gc.collect()
        
        # Return True if configuration changed significantly
        worker_change = abs(old_workers - self.current_workers) / max(old_workers, 1)
        return worker_change > 0.25  # 25% change threshold
    
    def get_effective_workers(self):
        """Get the current effective number of workers"""
        if self.custom_worker_count:
            return min(self.custom_worker_count, self.max_workers)
        
        if self.adaptive_scaling:
            self.adapt_to_system_load()
            return self.current_workers
        
        return self.max_workers
    
    def should_use_gpu(self):
        """Check if GPU acceleration should be used"""
        if not self.gpu_enabled:
            return False
        
        # Don't use GPU if system is under stress
        if self.auto_throttle and self.system_monitor.should_throttle():
            return False
        
        return self.gpu_detector.is_gpu_available()
    
    def get_performance_summary(self):
        """Get a summary of current performance configuration"""
        stats = self.system_monitor.get_system_stats()
        gpu_info = self.gpu_detector.get_gpu_info()
        
        # Determine GPU type with PyTorch preference
        gpu_type = 'None'
        if gpu_info.get('pytorch_cuda_available'):
            gpu_type = 'PyTorch CUDA'
        elif gpu_info.get('cuda_devices', 0) > 0:
            gpu_type = 'OpenCV CUDA'
        elif gpu_info.get('opencl_available'):
            gpu_type = 'OpenCL'
        
        summary = {
            'performance_mode': self.performance_mode,
            'workers': self.get_effective_workers(),
            'max_workers': self.max_workers,
            'gpu_enabled': self.should_use_gpu(),
            'gpu_type': gpu_type,
            'pytorch_cuda': gpu_info.get('pytorch_cuda_available', False),
            'opencv_cuda': gpu_info.get('cuda_devices', 0) > 0,
            'opencl': gpu_info.get('opencl_available', False),
            'cpu_usage': stats['cpu_percent'],
            'memory_usage': stats['memory_percent'],
            'throttling': self.system_monitor.should_throttle(),
            'quality_factor': self.quality_factor,
            'adaptive_scaling': self.adaptive_scaling
        }
        
        return summary

def divide_image_into_quadrants(image, overlap_pct=0.15):
    """
    Divide image into overlapping quadrants for parallel processing.
    
    Args:
        image: Input image (OpenCV format)
        overlap_pct: Percentage of overlap between quadrants (0.15 = 15%)
        
    Returns:
        List of tuples: [(quadrant_image, (x_offset, y_offset, orig_width, orig_height)), ...]
    """
    h, w = image.shape[:2]
    
    # Calculate quadrant dimensions
    quad_h = h // 2
    quad_w = w // 2
    
    # Calculate overlap in pixels
    overlap_h = int(quad_h * overlap_pct)
    overlap_w = int(quad_w * overlap_pct)
    
    quadrants = []
    
    # Top-left quadrant
    x1, y1 = 0, 0
    x2, y2 = quad_w + overlap_w, quad_h + overlap_h
    x2 = min(x2, w)
    y2 = min(y2, h)
    quad_img = image[y1:y2, x1:x2]
    quadrants.append((quad_img, (x1, y1, x2-x1, y2-y1)))
    
    # Top-right quadrant
    x1, y1 = quad_w - overlap_w, 0
    x2, y2 = w, quad_h + overlap_h
    x1 = max(x1, 0)
    y2 = min(y2, h)
    quad_img = image[y1:y2, x1:x2]
    quadrants.append((quad_img, (x1, y1, x2-x1, y2-y1)))
    
    # Bottom-left quadrant
    x1, y1 = 0, quad_h - overlap_h
    x2, y2 = quad_w + overlap_w, h
    x2 = min(x2, w)
    y1 = max(y1, 0)
    quad_img = image[y1:y2, x1:x2]
    quadrants.append((quad_img, (x1, y1, x2-x1, y2-y1)))
    
    # Bottom-right quadrant
    x1, y1 = quad_w - overlap_w, quad_h - overlap_h
    x2, y2 = w, h
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    quad_img = image[y1:y2, x1:x2]
    quadrants.append((quad_img, (x1, y1, x2-x1, y2-y1)))
    
    return quadrants

def transform_coordinates_from_quadrant(matches, quadrant_offset):
    """
    Transform match coordinates from quadrant space to full image space.
    
    Args:
        matches: List of matches from quadrant detection
        quadrant_offset: (x_offset, y_offset, width, height) of quadrant in full image
        
    Returns:
        List of matches with transformed coordinates
    """
    if not matches:
        return []
        
    x_offset, y_offset, _, _ = quadrant_offset
    transformed_matches = []
    
    for match in matches:
        transformed_match = match.copy()
        # Transform position coordinates
        transformed_match['x'] += x_offset
        transformed_match['y'] += y_offset
        transformed_match['center_x'] += x_offset
        transformed_match['center_y'] += y_offset
        
        # Add metadata about parallel processing
        transformed_match['processed_in_quadrant'] = True
        transformed_match['quadrant_offset'] = quadrant_offset
        
        transformed_matches.append(transformed_match)
    
    return transformed_matches

# Helper functions needed early
def pil_to_cv(pil_image):
    """Convert PIL image to OpenCV format, handling palette images"""
    # Convert palette images (mode 'P') to RGB first
    if pil_image.mode == 'P':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode not in ('RGB', 'RGBA'):
        pil_image = pil_image.convert('RGB')
    
    return cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

def load_stored_templates():
    """Load templates from stored_templates directory on startup.
    Supports hierarchical folder structure: stored_templates/<class>/<template>.png
    ALWAYS applies VIRIDIS colormap for consistency.
    Returns dict of templates.
    """
    templates_dir = "stored_templates"
    loaded_templates = {}
    
    if not os.path.exists(templates_dir):
        return loaded_templates
        
    # First, load templates from root directory (backward compatibility)
    for filename in os.listdir(templates_dir):
        filepath = os.path.join(templates_dir, filename)
        
        # Skip subdirectories in this pass
        if os.path.isdir(filepath):
            continue
            
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                template_image = Image.open(filepath)
                template_cv = pil_to_cv(template_image)
                
                # Templates are already in the correct color space when saved
                # Do NOT apply colormap here - that would double-colormap them
                print(f"[LOAD] Loaded template: {filename} (shape: {template_cv.shape})")
                
                # Create clean display name
                clean_name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                clean_name = clean_name.replace('_', ' ')
                
                # Add to current templates and active folder
                template_data = {
                    'image': template_cv,
                    'pil_image': template_image,
                    'size': template_image.size,
                    'clean_name': clean_name,
                    'original_name': filename,
                    'source': 'stored_template',
                    'class_folder': None  # Root level templates
                }
                
                loaded_templates[filename] = template_data
                
            except Exception as e:
                continue  # Skip problematic files
    
    # Now load templates from class subdirectories
    for class_name in os.listdir(templates_dir):
        class_path = os.path.join(templates_dir, class_name)
        
        # Only process directories
        if not os.path.isdir(class_path):
            continue
            
        # Load all templates in this class folder
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    filepath = os.path.join(class_path, filename)
                    template_image = Image.open(filepath)
                    template_cv = pil_to_cv(template_image)
                    
                    # Templates are already in the correct color space when saved
                    # Do NOT apply colormap here - that would double-colormap them
                    print(f"[LOAD] Loaded template: {class_name}/{filename} (shape: {template_cv.shape})")
                    
                    # Create clean display name (include class prefix)
                    base_name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    clean_name = f"{class_name}/{base_name}"
                    
                    # Use unique key that includes class folder
                    template_key = f"{class_name}/{filename}"
                    
                    # Add to loaded templates dict
                    template_data = {
                        'image': template_cv,
                        'pil_image': template_image,
                        'size': template_image.size,
                        'clean_name': clean_name,
                        'original_name': filename,
                        'source': 'stored_template',
                        'class_folder': class_name
                    }
                    
                    loaded_templates[template_key] = template_data
                    
                except Exception as e:
                    continue  # Skip problematic files
    
    return loaded_templates

def run_unified_detection(test_images, detection_params, status_callback=None, skip_boundary_check=False, skip_duplicate_check=False):
    """
    Unified detection function that uses the same comprehensive logic as manual detection.
    This ensures consistency between manual and auto-label detection processes.
    
    Args:
        test_images: List of test images to process
        detection_params: Dictionary containing detection parameters
        status_callback: Function to call with status messages
        skip_boundary_check: If True, allows signals at boundaries (for Train Mode)
        skip_duplicate_check: If True, skips duplicate checking (for Train Mode)
        
    Returns:
        tuple: (detection_results, unidentified_found, timing_info)
    """
    # Initialize timing tracker
    timing_tracker = TimingTracker()
    timing_tracker.start_tracking()
    
    detection_results = []
    all_unidentified_found = []
    
    with time_phase(timing_tracker, "initialization"):
        if status_callback:
            status_callback("🚀 Starting detection process...")
    
    with time_phase(timing_tracker, "image_processing"):
        for img_idx, test_file in enumerate(test_images):
            try:
                with time_phase(timing_tracker, f"load_image_{img_idx+1}"):
                    # Load test image
                    test_image = Image.open(test_file)
                    test_cv = pil_to_cv(test_image)
                    filename = getattr(test_file, 'name', f'image_{img_idx+1}')
                    
                    # ALWAYS apply colormap for consistency with template matching
                    # The simple_template_matcher will apply colormap to both image and template
                    # So we need to let it handle the colormap, not pre-process here
                    display_image = test_image
                    
                    # Store the original test_cv - simple_template_matcher will apply colormap consistently
                    image_results = {
                        'filename': filename,
                        'image': display_image,
                        'matches': [],
                        'image_size': test_image.size  # Add image dimensions (width, height) for SigMF
                    }
                
                # TEMPLATE MATCHING - Use simple, reliable matcher
                if st.session_state.templates:
                    with time_phase(timing_tracker, f"template_matching_{img_idx+1}"):
                        if status_callback:
                            status_callback(f"� Matching {len(st.session_state.templates)} templates on {filename}...")
                        
                        # Import and reload the simple matcher to ensure latest code
                        import simple_template_matcher
                        import importlib
                        importlib.reload(simple_template_matcher)
                        
                        # Run template matching
                        template_matches = simple_template_matcher.match_all_templates(
                            image=test_cv,
                            threshold=detection_params['threshold'],
                            min_confidence=detection_params['min_confidence'],
                            apply_nms=True
                        )
                        
                        # Add matches to results
                        image_results['matches'].extend(template_matches)
                        
                        if status_callback and template_matches:
                            status_callback(f"✅ Found {len(template_matches)} template matches")
                
                # Detect colored rectangles for unidentified signals - EXACTLY THE SAME AS MANUAL
                if detection_params['detect_green_rectangles']:
                    with time_phase(timing_tracker, f"unidentified_detection_{img_idx+1}"):
                        if status_callback:
                            status_callback(f"🟢 Detecting unidentified signals in {filename}...")
                        
                        # Detect colored rectangles in the original image - SAME FUNCTION
                        colored_matches = detect_colored_rectangles(test_cv, min_area=detection_params['green_min_area'])
                        
                        if colored_matches:
                            # Remove colored rectangles that overlap with identified drones - SAME FUNCTION
                            filtered_colored = remove_overlapping_detections(
                                image_results['matches'], 
                                colored_matches, 
                                overlap_threshold=detection_params['green_overlap_threshold']
                            )
                            
                            # Merge overlapping colored rectangles to avoid duplicates - SAME FUNCTION
                            merged_colored = merge_colored_rectangles(filtered_colored, merge_threshold=detection_params['colored_merge_threshold'])
                            
                            # Add unidentified signal detections to results - SAME AS MANUAL
                            image_results['matches'].extend(merged_colored)
                            
                            # Process each unidentified drone for storage - SAME AS MANUAL
                            img_h, img_w = test_cv.shape[:2]
                            for unidentified_match in merged_colored:
                                # Apply all validation rules and store if valid - SAME FUNCTION
                                # Pass skip_boundary_check and skip_duplicate_check flags for Train Mode
                                if add_unidentified_drone(unidentified_match, test_image, filename, img_w, img_h, 
                                                         detection_params['min_confidence'], skip_boundary_check, skip_duplicate_check):
                                    # Also add to the found list for auto-labeling
                                    unidentified_match['source_image'] = test_image
                                    unidentified_match['filename'] = filename
                                    all_unidentified_found.append(unidentified_match)
                
                # VARIANT CONSOLIDATION: Consolidate overlapping matches from the same drone variants
                # This ensures DroneA_1 and DroneA_2 don't show duplicate detections in the same area
                if image_results['matches']:
                    with time_phase(timing_tracker, f"variant_consolidation_{img_idx+1}"):
                        if status_callback:
                            status_callback(f"🔄 Consolidating pattern variants for {filename}...")
                        
                        # Apply variant consolidation (only consolidates same base drone variants that overlap)
                        image_results['matches'] = consolidate_pattern_variants(
                            image_results['matches'], 
                            overlap_threshold=0.7  # 70% overlap required for consolidation
                        )
                    
                    # HORIZONTAL FUSION: Fuse horizontally adjacent detections with small overlap
                    # This merges patterns split across boundaries or detected as separate parts
                    with time_phase(timing_tracker, f"horizontal_fusion_{img_idx+1}"):
                        original_count = len(image_results['matches'])
                        image_results['matches'] = fuse_horizontally_adjacent_detections(
                            image_results['matches'],
                            horizontal_overlap_threshold=0.3,  # 30% max horizontal overlap
                            vertical_alignment_threshold=0.7   # 70% vertical alignment required
                        )
                        fused_count = original_count - len(image_results['matches'])
                        if fused_count > 0 and status_callback:
                            status_callback(f"🔗 Fused {fused_count} horizontally adjacent detection(s) in {filename}")
                    
                    # SINGLE DRONE MODE: Classify unidentified signals by similarity matching
                    if detection_params.get('single_drone_mode', False):
                        with time_phase(timing_tracker, f"similarity_matching_{img_idx+1}"):
                            # Separate unidentified from known
                            unidentified = [m for m in image_results['matches'] if m['template_name'] == 'Unidentified Signal']
                            known = [m for m in image_results['matches'] if m['template_name'] != 'Unidentified Signal']
                            
                            if unidentified and known:
                                # Classify by multi-factor similarity matching
                                classified, remaining_unidentified = classify_by_similarity_matching(
                                    unidentified, known, 
                                    height_tolerance=0.15,  # 15% height tolerance
                                    y_position_tolerance=0.2  # 20% vertical position tolerance
                                )
                                
                                # Update matches with classified + remaining unidentified + known
                                image_results['matches'] = known + classified + remaining_unidentified
                                
                                classified_count = len(classified)
                                if classified_count > 0 and status_callback:
                                    status_callback(f"📡 Classified {classified_count} signal(s) by similarity matching in {filename}")
                
                detection_results.append(image_results)
                
            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error processing {getattr(test_file, 'name', 'unknown')}: {str(e)}")
                continue
    
    # Get timing information
    timing_info = timing_tracker.get_timing_summary()
    
    if status_callback:
        total_time = timing_info['formatted_total']
        status_callback(f"✅ Detection completed in {total_time}")
    
    # NOTE: SigMF saving is now handled by the caller (e.g., tab_pattern_detection.py)
    # after chunk_info has been properly added to the results. Don't save here!
    # if SigMFProcessor and detection_results:
    #     sigmf_path_param = detection_params.get('sigmf_meta_path') if detection_params else None
    #     with time_phase(timing_tracker, "sigmf_annotation"):
    #         save_detections_to_sigmf(detection_results, status_callback, sigmf_path_param)
    
    return detection_results, all_unidentified_found, timing_info


def save_detections_to_sigmf(detection_results, status_callback=None, sigmf_meta_path_override=None):
    """
    Save detection results as SigMF annotations.
    
    CRITICAL: Transforms coordinates from chunk/resized space back to ORIGINAL image space!
    
    Pipeline:
    1. Original image (e.g., 101708×1229) → RESIZED to 31778×384 → CHUNKED to 2048×384
    2. Detection runs on CHUNKS with coordinates (0-2048, 0-384)
    3. This function:
       - Groups all chunks by original filename
       - Transforms ALL coordinates: CHUNK coords → RESIZED coords → ORIGINAL coords
       - Creates ONE annotation file per original image with all detections
    
    Args:
        detection_results: List of detection result dictionaries with chunk_info
        status_callback: Function to call with status messages
        sigmf_meta_path_override: Optional explicit path to .sigmf-meta file or directory
    """
    if not SigMFProcessor:
        return
    
    # STEP 1: Group all chunks by original filename
    original_files = {}
    
    for result in detection_results:
        filename = result.get('filename', '')
        original_filename = result.get('original_filename', filename)
        matches = result.get('matches', [])
        chunk_info = result.get('chunk_info', {})
        
        if not matches:
            continue
        
        # Extract the base filename (remove _chunk_N suffix)
        # Example: "example_chunk_0.png" → "example.png"
        # Example: "2025-10-06T11-50-57Z_aaronia_militarybase_007_01_chunk_4.png" → "2025-10-06T11-50-57Z_aaronia_militarybase_007_01.png"
        import re
        base_filename = re.sub(r'_chunk_\d+', '', original_filename)
        
        # Initialize entry for this original file
        if base_filename not in original_files:
            original_files[base_filename] = {
                'chunks': [],
                'original_size': chunk_info.get('original_size'),
                'resized_size': chunk_info.get('resized_size')
            }
        
        # Add this chunk's data
        original_files[base_filename]['chunks'].append({
            'filename': filename,
            'matches': matches,
            'chunk_offset_x': chunk_info.get('offset_x', 0),
            'chunk_offset_y': chunk_info.get('offset_y', 0)
        })
    
    # STEP 2: Prepare all annotations (without writing yet)
    files_to_save = []  # List of (processor, num_annotations) tuples
    
    for base_filename, file_data in original_files.items():
        original_size = file_data['original_size']
        resized_size = file_data['resized_size']
        chunks = file_data['chunks']
        
        # Calculate scaling factors from resized → original
        if original_size and resized_size:
            orig_width, orig_height = original_size
            resized_width, resized_height = resized_size
            scale_x = orig_width / resized_width if resized_width > 0 else 1.0
            scale_y = orig_height / resized_height if resized_height > 0 else 1.0
        else:
            # No transformation needed (legacy behavior for non-chunked images)
            scale_x = 1.0
            scale_y = 1.0
        
        # STEP 3: Find corresponding .sigmf-meta file
        sigmf_meta_path = None
        if sigmf_meta_path_override:
            if os.path.isdir(sigmf_meta_path_override):
                # If directory provided, look for matching file in that directory
                base_name = os.path.splitext(os.path.basename(base_filename))[0]
                candidate = os.path.join(sigmf_meta_path_override, f"{base_name}.sigmf-meta")
                if os.path.exists(candidate):
                    sigmf_meta_path = candidate
            elif os.path.isfile(sigmf_meta_path_override):
                # If specific file provided, use it directly
                sigmf_meta_path = sigmf_meta_path_override
        
        # Fall back to auto-discovery if not found
        if not sigmf_meta_path:
            sigmf_meta_path = find_sigmf_meta_for_image(base_filename)
        
        if not sigmf_meta_path:
            continue
        
        try:
            # Load SigMF metadata
            processor = SigMFProcessor(sigmf_meta_path)
            
            if not processor.load_metadata():
                continue
            
            # Set spectrogram parameters (these should ideally come from metadata or config)
            processor.set_spectrogram_params(fft_size=2048, step_size=1024)
            
            # Get image dimensions - use the ACTUAL IMAGE dimensions (in pixels) from original_size
            if original_size:
                img_width, img_height = original_size
            else:
                # Fallback for legacy non-chunked images
                img_width = 1024
                img_height = 512
            
            # STEP 4: Transform ALL detections from ALL chunks to original coordinates
            # The key is to convert chunk-relative coordinates to original image coordinates
            # Then pass those to SigMFProcessor which will handle the time/freq conversion
            all_detections = []
            total_chunks = len(chunks)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_offset_x = chunk['chunk_offset_x']
                chunk_offset_y = chunk['chunk_offset_y']
                chunk_detections = len(chunk['matches'])
                
                for match in chunk['matches']:
                    # Step 1: Get chunk coordinates (relative to chunk)
                    chunk_x = match.get('x', 0)
                    chunk_y = match.get('y', 0)
                    chunk_w = match.get('width', 0)
                    chunk_h = match.get('height', 0)
                    
                    # Step 2: Transform to resized image coordinates (add chunk offset)
                    resized_x = chunk_x + chunk_offset_x
                    resized_y = chunk_y + chunk_offset_y
                    
                    # Step 3: Transform to original image coordinates (apply scaling)
                    original_x = int(resized_x * scale_x)
                    original_y = int(resized_y * scale_y)
                    original_w = int(chunk_w * scale_x)
                    original_h = int(chunk_h * scale_y)
                    
                    # Store detection with original image pixel coordinates
                    # SigMFProcessor.pixel_to_time_freq() will convert these to samples/Hz
                    detection = {
                        'x': original_x,
                        'y': original_y,
                        'width': original_w,
                        'height': original_h,
                        'label': match.get('template_name', 'unknown'),
                        'confidence': match.get('confidence', 0.0)
                    }
                    all_detections.append(detection)
            
            # Debug: Verify all chunks were processed
            if status_callback:
                status_callback(f"📊 Processed {total_chunks} chunks with {len(all_detections)} total detections for {base_filename}")
            
            # STEP 5: Add ALL detections to processor (don't save yet)
            # Pass original image dimensions in pixels (CRITICAL: must be original size, not chunk size!)
            num_added = processor.add_detections(
                detections=all_detections,
                image_width=orig_width,
                image_height=orig_height,
                replace_existing=True  # Replace to avoid duplicates on re-run
            )
            
            if num_added > 0:
                files_to_save.append(processor)
        
        except Exception:
            continue
    
    # STEP 6: Batch save all files at once - MUCH FASTER!
    processed_count = 0
    for processor in files_to_save:
        if processor.save_metadata():
            processed_count += 1
    
    if status_callback and processed_count > 0:
        status_callback(f"✅ Saved SigMF annotations for {processed_count} file(s)")

def auto_label_process(test_images, detection_params, progress_callback=None, status_callback=None, stop_callback=None):
    """
    Automatically run pattern detection iteratively until all unidentified signals are labeled.
    Now uses the SAME comprehensive detection logic as manual detection for consistency.
    
    Args:
        test_images: List of test images to process
        detection_params: Dictionary containing detection parameters (threshold, min_confidence, etc.)
        progress_callback: Function to call with progress updates (current_iteration, total_labeled)
        status_callback: Function to call with status messages
        stop_callback: Function that returns True if process should be stopped
        
    Returns:
        dict: Results of the auto-labeling process
    """
    if not test_images:
        return {"success": False, "error": "No test images provided"}
    
    total_labeled = 0
    iteration = 0
    max_iterations = 50  # Safety limit to prevent infinite loops
    
    while iteration < max_iterations:
        if stop_callback and stop_callback():
            return {"success": False, "error": "Process stopped by user", "total_labeled": total_labeled, "iterations": iteration}
        
        iteration += 1
        
        if status_callback:
            status_callback(f"Iteration {iteration}: Running comprehensive pattern detection...")
        
        # Run unified detection - SAME AS MANUAL DETECTION!
        iteration_results, unidentified_found, timing_info = run_unified_detection(test_images, detection_params, status_callback)
        
        # Update detection results
        st.session_state.detection_results = iteration_results
        
        # Check if we found any unidentified signals
        if not unidentified_found:
            # No unidentified signals found - process complete
            if status_callback:
                status_callback(f"Auto-labeling complete! No more unidentified signals found after {iteration} iterations.")
            return {
                "success": True, 
                "total_labeled": total_labeled, 
                "iterations": iteration,
                "message": f"Process completed successfully. Labeled {total_labeled} signals in {iteration} iterations."
            }
        
        # Sort unidentified signals by confidence in descending order
        unidentified_found.sort(key=lambda x: x['confidence'], reverse=True)
        
        if status_callback:
            status_callback(f"Found {len(unidentified_found)} unidentified signals. Labeling highest confidence ones...")
        
        # Label the highest confidence unidentified signals as new templates
        labeled_this_iteration = 0
        # Use a reasonable confidence threshold - not too high that nothing gets labeled
        confidence_threshold = max(detection_params['min_confidence'], 0.70)  # At least 70% confidence
        
        if status_callback:
            status_callback(f"Using confidence threshold: {confidence_threshold:.2f} for auto-labeling")
        
        for unidentified_match in unidentified_found:
            # Debug: show what we're evaluating
            if status_callback:
                status_callback(f"Evaluating signal with confidence {unidentified_match['confidence']:.3f} (threshold: {confidence_threshold:.3f})")
            
            # Only auto-label signals with sufficient confidence
            if unidentified_match['confidence'] >= confidence_threshold:
                try:
                    # Enhanced check if this unidentified signal contains other identified models
                    # Use configurable threshold and bidirectional overlap checking
                    current_result = next((r for r in iteration_results if r['filename'] == unidentified_match.get('filename')), None)
                    if current_result:
                        identified_matches = [m for m in current_result['matches'] if m['template_name'] != 'Unidentified Signal']
                        
                        # Check if any identified models are inside this unidentified signal
                        unidentified_box = (unidentified_match['x'], unidentified_match['y'], 
                                          unidentified_match['width'], unidentified_match['height'])
                        
                        has_models_inside = False
                        containment_threshold = 0.75  # Use higher threshold for containment
                        
                        for identified_match in identified_matches:
                            # Skip other unidentified signals in comparison
                            if identified_match['template_name'] == 'Unidentified Signal':
                                continue
                                
                            identified_box = (identified_match['x'], identified_match['y'],
                                            identified_match['width'], identified_match['height'])
                            
                            # Check if unidentified signal contains identified models (main concern)
                            overlap_identified_in_unidentified = calculate_overlap_percentage(identified_box, unidentified_box)
                            
                            # Also check for high mutual overlap (same object detected differently)
                            overlap_unidentified_in_identified = calculate_overlap_percentage(unidentified_box, identified_box)
                            max_mutual_overlap = max(overlap_identified_in_unidentified, overlap_unidentified_in_identified)
                            
                            # Skip if:
                            # 1. Unidentified signal contains identified models (>=75% overlap)
                            # 2. High mutual overlap suggests same object (>=60%)
                            if (overlap_identified_in_unidentified >= containment_threshold or 
                                max_mutual_overlap >= 0.6):  # 60% mutual overlap suggests same object
                                has_models_inside = True
                                if status_callback:
                                    status_callback(f"Skipping signal (conf: {unidentified_match['confidence']:.3f}) - contains/overlaps with {identified_match['template_name']} ({overlap_identified_in_unidentified:.1%}/{overlap_unidentified_in_identified:.1%})")
                                break
                        
                        # Skip this signal if it contains other models
                        if has_models_inside:
                            continue
                    
                    # Create a template from the highest confidence unidentified signal
                    img_array = np.array(unidentified_match['source_image'])
                    x, y, w, h = unidentified_match['x'], unidentified_match['y'], unidentified_match['width'], unidentified_match['height']
                    
                    # Ensure coordinates are within image bounds
                    img_h, img_w = img_array.shape[:2]
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    
                    if w > 0 and h > 0:
                        # Extract signal image crop
                        signal_crop = img_array[y:y+h, x:x+w]
                        signal_image = Image.fromarray(signal_crop)
                        
                        # Extract class name from image filename (remove extension)
                        source_filename = unidentified_match.get('filename', f'image_{total_labeled+1}')
                        class_name = os.path.splitext(source_filename)[0]  # Remove extension
                        
                        # Create template name with Signal instead of Auto Drone
                        template_name = f"Signal_{total_labeled + 1}_Conf{unidentified_match['confidence']:.3f}.png"
                        clean_name = f"{class_name}/Signal {total_labeled + 1}"
                        
                        # Save to stored_templates/<class_name>/ folder for organization
                        try:
                            templates_dir = "stored_templates"
                            class_dir = os.path.join(templates_dir, class_name)
                            if not os.path.exists(class_dir):
                                os.makedirs(class_dir)
                            
                            filepath = os.path.join(class_dir, template_name)
                            signal_image.save(filepath, 'PNG')
                            
                            if status_callback:
                                status_callback(f"Saved template to: {filepath}")
                        except Exception as save_error:
                            if status_callback:
                                status_callback(f"Warning: Could not save template file: {str(save_error)}")
                        
                        # Convert to OpenCV format
                        template_cv = pil_to_cv(signal_image)
                        
                        # Add to current templates and active folder with class folder info
                        template_key = f"{class_name}/{template_name}"  # Use class/filename as key
                        template_data = {
                            'image': template_cv,
                            'pil_image': signal_image,
                            'size': signal_image.size,
                            'clean_name': clean_name,
                            'original_name': template_name,
                            'source': 'auto_labeled',
                            'confidence': unidentified_match['confidence'],
                            'iteration': iteration,
                            'class_folder': class_name  # Store class folder for organization
                        }
                        
                        st.session_state.templates[template_key] = template_data
                        st.session_state.template_folders[st.session_state.active_folder][template_key] = template_data
                        
                        labeled_this_iteration += 1
                        total_labeled += 1
                        
                        if status_callback:
                            status_callback(f"Labeled signal {total_labeled}: {clean_name} (confidence: {unidentified_match['confidence']:.3f})")
                        
                        # Limit how many we label per iteration to prevent overwhelming
                        if labeled_this_iteration >= 3:  # Max 3 new labels per iteration
                            break
                            
                except Exception as e:
                    if status_callback:
                        status_callback(f"Error creating template: {str(e)}")
                    continue
            else:
                if status_callback:
                    status_callback(f"Skipping signal with confidence {unidentified_match['confidence']:.3f} (below threshold {confidence_threshold:.3f})")
        
        if labeled_this_iteration == 0:
            # No high-confidence signals to label - process complete
            if status_callback:
                status_callback(f"Auto-labeling complete! No high-confidence unidentified signals remaining after {iteration} iterations.")
            return {
                "success": True, 
                "total_labeled": total_labeled, 
                "iterations": iteration,
                "message": f"Process completed. Labeled {total_labeled} signals in {iteration} iterations. Remaining signals have confidence < {confidence_threshold}."
            }
        
        if progress_callback:
            progress_callback(iteration, total_labeled)
            
        # Brief pause to allow UI updates
        import time
        time.sleep(0.1)
    
    # Max iterations reached
    return {
        "success": False, 
        "error": f"Maximum iterations ({max_iterations}) reached", 
        "total_labeled": total_labeled, 
        "iterations": iteration
    }


# Helper functions
def iou(boxA, boxB):
    # box: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def filter_duplicates(matches, iou_threshold=0.2):
    """
    Filter duplicate matches, averaging their properties instead of just keeping the highest confidence.
    Uses a lower IOU threshold to catch overlapping detections of the same object.
    """
    if not matches:
        return matches
    
    filtered = []
    processed_indices = set()
    
    for i, match in enumerate(matches):
        if i in processed_indices:
            continue
            
        # Find all duplicates for this match
        duplicates = [match]
        duplicate_indices = {i}
        
        for j, other_match in enumerate(matches[i+1:], i+1):
            if j in processed_indices:
                continue
                
            if match['template_name'] == other_match['template_name']:
                iou_score = iou((match['x'], match['y'], match['width'], match['height']),
                               (other_match['x'], other_match['y'], other_match['width'], other_match['height']))
                
                # Also check for significant overlap (even if IOU is low due to positioning)
                overlap_score = calculate_overlap_percentage(
                    (match['x'], match['y'], match['width'], match['height']),
                    (other_match['x'], other_match['y'], other_match['width'], other_match['height'])
                )
                
                if iou_score > iou_threshold or overlap_score > 0.3:  # 30% overlap threshold
                    duplicates.append(other_match)
                    duplicate_indices.add(j)
        
        # Mark all duplicates as processed
        processed_indices.update(duplicate_indices)
        
        # Create averaged match from duplicates
        if len(duplicates) == 1:
            filtered.append(duplicates[0])
        else:
            # Weighted average based on confidence scores
            total_weight = sum(d['confidence'] for d in duplicates)
            
            # Calculate weighted average position
            weighted_x = sum(d['x'] * d['confidence'] for d in duplicates) / total_weight
            weighted_y = sum(d['y'] * d['confidence'] for d in duplicates) / total_weight
            
            # Use highest confidence as the base confidence, but boost it slightly for merged detection
            max_confidence = max(d['confidence'] for d in duplicates)
            avg_confidence = sum(d['confidence'] for d in duplicates) / len(duplicates)
            # Blend max and average, giving more weight to max but boosting for multiple detections
            boosted_confidence = min(1.0, max_confidence * 0.7 + avg_confidence * 0.3 + 0.05)
            
            # Use the most common detection type, or 'averaged' if mixed
            detection_types = [d.get('detection_type', 'full') for d in duplicates]
            most_common_type = max(set(detection_types), key=detection_types.count)
            
            # Check if any are partial detections
            is_partial = any(d.get('partial', False) for d in duplicates)
            
            averaged_match = {
                'template_name': match['template_name'],
                'confidence': float(boosted_confidence),
                'x': int(weighted_x),
                'y': int(weighted_y),
                'width': match['width'],
                'height': match['height'],
                'center_x': int(weighted_x + match['width']/2),
                'center_y': int(weighted_y + match['height']/2),
                'partial': is_partial,
                'detection_type': f'merged_{most_common_type}' if len(duplicates) > 1 else most_common_type,
                'duplicate_count': len(duplicates),
                'merge_reason': 'overlapping_detections'
            }
            filtered.append(averaged_match)
    
    return filtered

def calculate_overlap_percentage(boxA, boxB):
    """
    Calculate what percentage of the boxes overlap with each other.
    This catches cases where two detections are overlapping the same object
    but might have low IOU due to positioning.
    """
    # boxA and boxB are (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    # Check if there's any intersection
    if xB <= xA or yB <= yA:
        return 0.0
    
    interArea = (xB - xA) * (yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Return the maximum percentage of overlap from either box's perspective
    overlapA = interArea / boxAArea if boxAArea > 0 else 0
    overlapB = interArea / boxBArea if boxBArea > 0 else 0
    
    return max(overlapA, overlapB)


def is_box_completely_inside(inner_box, outer_box, threshold=0.95):
    """
    Check if inner_box is completely (or almost completely) contained within outer_box.
    
    Args:
        inner_box: (x, y, w, h) of potentially nested detection
        outer_box: (x, y, w, h) of potentially containing detection
        threshold: Minimum overlap percentage (0.95 = 95% contained)
    
    Returns:
        True if inner_box is >=95% inside outer_box
    """
    # Calculate intersection
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[0] + inner_box[2], outer_box[0] + outer_box[2])
    yB = min(inner_box[1] + inner_box[3], outer_box[1] + outer_box[3])
    
    # No intersection
    if xB <= xA or yB <= yA:
        return False
    
    # Calculate how much of inner box is inside outer box
    intersection_area = (xB - xA) * (yB - yA)
    inner_area = inner_box[2] * inner_box[3]
    
    if inner_area == 0:
        return False
    
    containment_ratio = intersection_area / inner_area
    return containment_ratio >= threshold


def classify_by_similarity_matching(unidentified_matches, known_matches, height_tolerance=0.15, y_position_tolerance=0.2):
    """
    Classify unidentified signals by matching multiple characteristics with known signals.
    Uses height, vertical position (y), and other features (excluding width for edge cases).
    
    Args:
        unidentified_matches: List of unidentified signal detections
        known_matches: List of known template matches
        height_tolerance: Allowed height difference as fraction (0.15 = 15%)
        y_position_tolerance: Allowed y-position difference as fraction of height (0.2 = 20%)
    
    Returns:
        Tuple: (classified_matches, remaining_unidentified)
    """
    if not unidentified_matches or not known_matches:
        return [], unidentified_matches
    
    classified = []
    remaining = []
    
    for unidentified in unidentified_matches:
        if unidentified['template_name'] != 'Unidentified Signal':
            # Already classified
            classified.append(unidentified)
            continue
        
        us_y = unidentified['y']
        us_height = unidentified['height']
        us_center_y = unidentified.get('center_y', us_y + us_height / 2)
        
        best_match = None
        best_score = 0.0
        best_method = None
        
        # Try to find best matching known signal
        for known in known_matches:
            template_name = known['template_name']
            k_y = known['y']
            k_height = known['height']
            k_center_y = known.get('center_y', k_y + k_height / 2)
            
            # FACTOR 1: Height similarity (most important for signal type)
            height_diff_ratio = abs(us_height - k_height) / max(k_height, 1)
            if height_diff_ratio > height_tolerance:
                continue  # Height too different, skip
            
            height_score = 1.0 - (height_diff_ratio / height_tolerance)
            
            # FACTOR 2: Vertical position alignment (key indicator for same signal)
            # Check if they're at similar y-position (suggesting same horizontal signal line)
            y_center_diff = abs(us_center_y - k_center_y)
            y_diff_ratio = y_center_diff / max(k_height, us_height, 1)
            
            if y_diff_ratio > y_position_tolerance:
                # Not vertically aligned, lower score
                y_position_score = 0.3
            else:
                # Well aligned vertically
                y_position_score = 1.0 - (y_diff_ratio / y_position_tolerance)
            
            # FACTOR 3: Confidence similarity (similar detection strength suggests same type)
            us_conf = unidentified.get('confidence', 0.7)
            k_conf = known.get('confidence', 0.7)
            conf_diff = abs(us_conf - k_conf)
            confidence_score = 1.0 - conf_diff
            
            # COMBINED SCORE with weighted factors
            combined_score = (
                height_score * 0.75 +
                y_position_score * 0.15 +
                confidence_score * 0.10
            )
            
            # Bonus: If very close in y-position AND similar height, boost score
            if y_diff_ratio < 0.1 and height_diff_ratio < 0.1:
                combined_score = min(1.0, combined_score * 1.2)
                method = 'height_and_y_position'
            elif y_diff_ratio < 0.1:
                method = 'y_position_matching'
            elif height_diff_ratio < 0.08:
                method = 'height_matching'
            else:
                method = 'similarity_matching'
            
            # Update best match if this is better
            if combined_score > best_score:
                best_score = combined_score
                best_match = template_name
                best_method = method
        
        # Classify if we found a good match (score > 0.6 threshold)
        if best_match and best_score > 0.6:
            classified_match = unidentified.copy()
            classified_match['template_name'] = best_match
            classified_match['classification_method'] = best_method
            classified_match['similarity_score'] = best_score
            classified_match['original_confidence'] = unidentified.get('confidence', 0.0)
            # Adjust confidence based on similarity score
            base_conf = unidentified.get('confidence', 0.75)
            classified_match['confidence'] = min(0.95, base_conf * (0.9 + best_score * 0.1))
            classified.append(classified_match)
        else:
            remaining.append(unidentified)
    
    return classified, remaining


def remove_nested_detections(matches, containment_threshold=0.95):
    """
    Remove detections that are completely inside other detections.
    This prevents showing smaller drones that are fully contained within larger ones.
    
    Args:
        matches: List of detection matches
        containment_threshold: Minimum overlap to consider nested (0.95 = 95%)
    
    Returns:
        Filtered list with nested detections removed
    """
    if len(matches) <= 1:
        return matches
    
    filtered = []
    
    for i, match in enumerate(matches):
        inner_box = (match['x'], match['y'], match['width'], match['height'])
        inner_area = match['width'] * match['height']
        is_nested = False
        
        # Check if this detection is nested inside any other detection
        for j, other_match in enumerate(matches):
            if i == j:
                continue
            
            outer_box = (other_match['x'], other_match['y'], other_match['width'], other_match['height'])
            outer_area = other_match['width'] * other_match['height']
            
            # Calculate containment in both directions
            inner_containment = is_box_completely_inside(inner_box, outer_box, containment_threshold)
            outer_containment = is_box_completely_inside(outer_box, inner_box, containment_threshold)
            
            # Remove if:
            # 1. This detection is 95-99% inside the other AND smaller
            # 2. 100% overlap (both are inside each other) - remove the inner one
            if inner_containment:
                if outer_containment:
                    # 100% overlap - both completely inside each other
                    # Remove this one (arbitrary but consistent choice)
                    is_nested = True
                    break
                elif inner_area < outer_area:
                    # Smaller detection inside larger one
                    is_nested = True
                    break
        
        # Only keep detections that are NOT nested
        if not is_nested:
            filtered.append(match)
    
    return filtered


def fuse_horizontally_adjacent_detections(matches, horizontal_overlap_threshold=0.3, vertical_alignment_threshold=0.7):
    """
    Fuse detections that are horizontally adjacent with small overlap.
    Useful for detecting patterns split across boundaries or detected as separate parts.
    
    Args:
        matches: List of detection matches
        horizontal_overlap_threshold: Max horizontal overlap as fraction of smaller width (0.3 = 30%)
        vertical_alignment_threshold: Min vertical overlap required for fusion (0.7 = 70%)
    
    Returns:
        List with horizontally adjacent detections fused
    """
    if len(matches) <= 1:
        return matches
    
    # Sort by x position for efficient processing
    sorted_matches = sorted(matches, key=lambda m: m['x'])
    fused = []
    used_indices = set()
    
    for i, match1 in enumerate(sorted_matches):
        if i in used_indices:
            continue
        
        # Start with current match
        current_fused = match1.copy()
        current_fused['fused_from'] = [match1['template_name']]
        fused_any = False
        
        # Look for horizontally adjacent matches
        for j, match2 in enumerate(sorted_matches[i+1:], i+1):
            if j in used_indices:
                continue
            
            # Only fuse detections from the SAME drone type
            if match1['template_name'] != match2['template_name']:
                continue
            
            # Calculate horizontal relationship
            box1_right = current_fused['x'] + current_fused['width']
            box2_left = match2['x']
            box2_right = match2['x'] + match2['width']
            
            # Skip if too far apart (no horizontal adjacency)
            horizontal_gap = box2_left - box1_right
            if horizontal_gap > min(current_fused['width'], match2['width']) * horizontal_overlap_threshold:
                break  # Since sorted by x, no more candidates
            
            # Check vertical alignment
            box1_top = current_fused['y']
            box1_bottom = current_fused['y'] + current_fused['height']
            box2_top = match2['y']
            box2_bottom = match2['y'] + match2['height']
            
            # Calculate vertical overlap
            vertical_overlap_top = max(box1_top, box2_top)
            vertical_overlap_bottom = min(box1_bottom, box2_bottom)
            
            if vertical_overlap_bottom <= vertical_overlap_top:
                continue  # No vertical overlap
            
            vertical_overlap_height = vertical_overlap_bottom - vertical_overlap_top
            min_height = min(current_fused['height'], match2['height'])
            vertical_alignment_ratio = vertical_overlap_height / min_height
            
            # Check if vertically aligned enough
            if vertical_alignment_ratio < vertical_alignment_threshold:
                continue  # Not aligned enough
            
            # Fuse the two detections
            new_x = min(current_fused['x'], match2['x'])
            new_y = min(current_fused['y'], match2['y'])
            new_right = max(box1_right, box2_right)
            new_bottom = max(box1_bottom, box2_bottom)
            new_width = new_right - new_x
            new_height = new_bottom - new_y
            
            # Update fused detection
            current_fused['x'] = new_x
            current_fused['y'] = new_y
            current_fused['width'] = new_width
            current_fused['height'] = new_height
            current_fused['center_x'] = new_x + new_width // 2
            current_fused['center_y'] = new_y + new_height // 2
            
            # Combine confidences (weighted average)
            total_area = current_fused.get('fused_area', match1['width'] * match1['height']) + (match2['width'] * match2['height'])
            current_conf = current_fused.get('confidence', match1['confidence'])
            fused_conf = (current_conf * current_fused.get('fused_area', match1['width'] * match1['height']) + 
                         match2['confidence'] * (match2['width'] * match2['height'])) / total_area
            current_fused['confidence'] = fused_conf
            current_fused['fused_area'] = total_area
            
            # Track fusion
            current_fused['fused_from'].append(match2['template_name'])
            current_fused['detection_type'] = 'fused_horizontal'
            used_indices.add(j)
            fused_any = True
        
        # Add the (possibly fused) detection
        if fused_any:
            current_fused['fused_count'] = len(current_fused['fused_from'])
        
        fused.append(current_fused)
        used_indices.add(i)
    
    return fused

def clean_template_name(template_name):
    """Clean template name by removing file extensions and keeping it concise"""
    # Remove common file extensions
    name = template_name.lower()
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    for ext in extensions:
        if name.endswith(ext):
            template_name = template_name[:-len(ext)]
            break
    
    # Keep original format but remove file extensions only
    # This keeps names short and concise like "dji_mini_2" instead of "Dji Mini 2"
    return template_name

def is_valid_border_detection(x, y, w, h, img_w, img_h, min_border_overlap=0.1):
    """
    Check if a detection is actually at an image border.
    Returns True only if the template would extend outside image bounds
    and has sufficient overlap with the image.
    """
    # Calculate overlap with image
    overlap_x1 = max(0, x)
    overlap_y1 = max(0, y)
    overlap_x2 = min(img_w, x + w)
    overlap_y2 = min(img_h, y + h)
    
    # Check if there's valid overlap
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return False
    
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    template_area = w * h
    overlap_ratio = overlap_area / template_area
    
    # Must have minimum overlap AND extend beyond image bounds
    extends_beyond = (x < 0 or y < 0 or x + w > img_w or y + h > img_h)
    sufficient_overlap = overlap_ratio >= min_border_overlap
    
    return extends_beyond and sufficient_overlap

def detect_border_templates_gpu_optimized(image, template, template_name, threshold):
    """
    Optimized border detection for GPU parallel processing.
    Uses simplified border detection logic for better performance.
    
    Args:
        image: Input image
        template: Template to search for at borders
        template_name: Name of the template
        threshold: Detection threshold for border matches
        
    Returns:
        List of border detection matches
    """
    if len(image.shape) == 3:
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()
    
    if len(template.shape) == 3:
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    else:
        template_gray = template.copy()
    
    img_h, img_w = img_gray.shape
    template_h, template_w = template_gray.shape
    
    border_matches = []
    
    # Only test common crop percentages for performance
    crop_percentages = [0.5]  # 50% visible - optimized for speed (2x faster)
    
    for crop_pct in crop_percentages:
        # Create cropped templates for each border
        crop_h = int(template_h * crop_pct)
        crop_w = int(template_w * crop_pct)
        
        if crop_h < 10 or crop_w < 10:  # Skip if too small
            continue
        
        # Test each border direction
        border_tests = [
            # (cropped_template, border_type, expected_x_offset, expected_y_offset)
            (template_gray[template_h-crop_h:, :], 'top', 0, -(template_h-crop_h)),
            (template_gray[:crop_h, :], 'bottom', 0, img_h - crop_h),
            (template_gray[:, template_w-crop_w:], 'left', -(template_w-crop_w), 0),
            (template_gray[:, :crop_w], 'right', img_w - crop_w, 0)
        ]
        
        for cropped_template, border_type, offset_x, offset_y in border_tests:
            try:
                # Quick template matching on border region
                result = cv.matchTemplate(img_gray, cropped_template, cv.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                
                if len(locations) >= 2 and len(locations[0]) > 0:
                    for y, x in zip(locations[0], locations[1]):
                        confidence = float(result[y, x])
                        
                        # Transform coordinates to represent full template position
                        full_x = int(x + offset_x)
                        full_y = int(y + offset_y)
                        
                        # Validate that this is actually a border detection
                        if is_valid_border_detection(full_x, full_y, template_w, template_h, img_w, img_h):
                            border_match = {
                                'template_name': template_name,
                                'type': 'template_match',
                                'confidence': confidence,
                                'x': full_x,
                                'y': full_y,
                                'width': template_w,
                                'height': template_h,
                                'center_x': int(full_x + template_w/2),
                                'center_y': int(full_y + template_h/2),
                                'partial': True,
                                'border_type': border_type,
                                'crop_percentage': crop_pct,
                                'method': 'TM_CCOEFF_NORMED_BORDER'
                            }
                            border_matches.append(border_match)
                            
            except Exception as e:
                # Skip problematic border detections
                continue
    
    return border_matches

def detect_templates_parallel_gpu(image, detection_settings=None, progress_callback=None):
    """
    High-performance parallel GPU detection for ALL templates simultaneously.
    This includes both standard template matching AND border detection.
    
    Args:
        image: Input image
        detection_settings: Detection configuration (dict with border detection support)
        progress_callback: Optional progress callback
        
    Returns:
        List of detection results from all templates (includes border detections)
    """
    from pathlib import Path
    from smart_backend_selector import SmartBackendSelector
    
    # Default settings with border detection support
    if detection_settings is None:
        detection_settings = {
            'threshold': 0.7,
            'enable_border_detection': True,
            'border_threshold': 0.7
        }
    
    # Initialize backend selector for GPU acceleration
    backend_selector = SmartBackendSelector()
    
    # Load all templates (including those in subdirectories)
    template_files = []
    template_dir = Path("stored_templates")
    if template_dir.exists():
        # Look for PNG files in the main directory and all subdirectories
        template_files = list(template_dir.rglob("*.png"))
    
    if not template_files:
        return []
    
    templates = []
    template_names = []
    
    for template_file in template_files:
        try:
            template = cv.imread(str(template_file), cv.IMREAD_COLOR)
            if template is not None:
                # Apply colormap to grayscale templates to match processed images
                if template.shape[2] == 3:
                    b, g, r = cv.split(template)
                    if np.array_equal(b, g) and np.array_equal(g, r):
                        # Grayscale template - apply colormap
                        gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                        template = cv.applyColorMap(gray, cv.COLORMAP_VIRIDIS)
                
                templates.append(template)
                template_names.append(template_file.stem)
        except Exception as e:
            pass
    
    if not templates:
        return []
    
    all_results = []
    
    try:
        # TRUE PARALLEL GPU PROCESSING - All templates simultaneously
        if (backend_selector.available_backends.get('pytorch_gpu', {}).get('available', False)):
            
            # Get the PyTorch GPU accelerator
            pytorch_backend = backend_selector.available_backends['pytorch_gpu']
            gpu_accelerator = pytorch_backend.get('accelerator')
            
            if gpu_accelerator and gpu_accelerator.is_available():
                # Process ALL templates at once (this is the crucial speedup)
                gpu_results, timing_info = gpu_accelerator.parallel_template_match(
                    image, templates, cv.TM_CCOEFF_NORMED, return_timing=True
                )
                
                # Log performance
                templates_per_sec = timing_info.get('templates_per_second', 0)
                gpu_time = timing_info.get('gpu_time', 0)
                fallback = timing_info.get('fallback', False)
                fallback_reason = timing_info.get('reason', 'unknown')
                
                # Process all results with border detection support
                threshold = detection_settings.get('threshold', 0.7)
                enable_border_detection = detection_settings.get('enable_border_detection', True)
                border_threshold = detection_settings.get('border_threshold', 0.7)
                
                img_h, img_w = image.shape[:2]
                
                for i, (result, template_name) in enumerate(zip(gpu_results, template_names)):
                    if result is None:
                        continue
                    
                    template_h, template_w = templates[i].shape[:2]
                    
                    # Standard template matching results
                    locations = np.where(result >= threshold)
                    
                    # Check if we have valid location results
                    if len(locations) >= 2 and len(locations[0]) > 0 and len(locations[1]) > 0:
                        for y, x in zip(locations[0], locations[1]):
                            confidence = float(result[y, x])
                            
                            detection_result = {
                                'template_name': template_name,
                                'type': 'template_match',
                                'confidence': confidence,
                                'x': int(x),
                                'y': int(y),
                                'width': template_w,
                                'height': template_h,
                                'center_x': int(x + template_w/2),
                                'center_y': int(y + template_h/2),
                                'partial': False,
                                'detection_type': 'gpu_parallel_true' if not fallback else 'cpu_efficient',
                                'method': 'TM_CCOEFF_NORMED',
                                'gpu_timing': timing_info,
                                'processing_time': gpu_time
                            }
                            all_results.append(detection_result)
                    
                    # BORDER DETECTION: Add border detection for this template
                    if enable_border_detection:
                        border_matches = detect_border_templates_gpu_optimized(
                            image, templates[i], template_name, border_threshold
                        )
                        
                        # Add border detections to results
                        for border_match in border_matches:
                            border_match.update({
                                'detection_type': 'gpu_parallel_border' if not fallback else 'cpu_border',
                                'gpu_timing': timing_info,
                                'processing_time': gpu_time
                            })
                            all_results.append(border_match)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(templates))
        
        else:
            
            # CPU fallback with border detection support
            threshold = detection_settings.get('threshold', 0.7)
            enable_border_detection = detection_settings.get('enable_border_detection', True)
            border_threshold = detection_settings.get('border_threshold', 0.7)
            
            for i, (template, template_name) in enumerate(zip(templates, template_names)):
                # Standard template matching
                result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
                
                locations = np.where(result >= threshold)
                
                # Check if we have valid location results
                if len(locations) >= 2 and len(locations[0]) > 0 and len(locations[1]) > 0:
                    for y, x in zip(locations[0], locations[1]):
                        confidence = float(result[y, x])
                        template_h, template_w = template.shape[:2]
                        
                        detection_result = {
                            'template_name': template_name,
                            'type': 'template_match',
                            'confidence': confidence,
                            'x': int(x),
                            'y': int(y),
                            'width': template_w,
                            'height': template_h,
                            'center_x': int(x + template_w/2),
                            'center_y': int(y + template_h/2),
                            'partial': False,
                            'detection_type': 'cpu_sequential',
                            'method': 'TM_CCOEFF_NORMED'
                        }
                        all_results.append(detection_result)
                
                # BORDER DETECTION: Add border detection for CPU fallback
                if enable_border_detection:
                    border_matches = detect_border_templates_gpu_optimized(
                        image, template, template_name, border_threshold
                    )
                    
                    # Add border detections to results
                    for border_match in border_matches:
                        border_match['detection_type'] = 'cpu_border'
                        all_results.append(border_match)
                
                if progress_callback:
                    progress_callback(i + 1, len(templates))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
    
    # Apply non-maximum suppression to remove overlapping detections
    if all_results:
        all_results = apply_nms_to_detections(all_results, overlap_threshold=0.3)

    return all_results

def apply_nms_to_detections(detections, overlap_threshold=0.3):
    """Apply Non-Maximum Suppression to detection results"""
    if len(detections) <= 1:
        return detections
    
    # Group by template name to avoid suppressing different templates
    template_groups = {}
    for detection in detections:
        template_name = detection.get('template_name', 'unknown')
        if template_name not in template_groups:
            template_groups[template_name] = []
        template_groups[template_name].append(detection)
    
    final_results = []
    
    for template_name, group_detections in template_groups.items():
        if len(group_detections) <= 1:
            final_results.extend(group_detections)
            continue
        
        # Convert to format needed for NMS
        boxes = []
        scores = []
        indices = []
        
        for i, detection in enumerate(group_detections):
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            boxes.append([x, y, x + w, y + h])
            scores.append(detection['confidence'])
            indices.append(i)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply OpenCV NMS
        selected_indices = cv.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            score_threshold=0.1, nms_threshold=overlap_threshold
        )
        
        if len(selected_indices) > 0:
            selected_indices = selected_indices.flatten()
            for idx in selected_indices:
                final_results.append(group_detections[indices[idx]])
    
    return final_results


def run_multi_image_gpu_detection(images, image_filenames, detection_params, status_callback=None):
    """
    Run GPU-accelerated detection on multiple images simultaneously.
    Each GPU instance processes one template on one image (horizontal parallelization).
    
    Args:
        images: List of OpenCV images (numpy arrays)
        image_filenames: List of filenames corresponding to images
        detection_params: Detection parameters dict
        status_callback: Optional callback for status updates
        
    Returns:
        List of detection results per image
    """
    if not images:
        return []
    
    try:
        # Get GPU accelerator
        gpu_detector = GPUDetector()
        if not gpu_detector.pytorch_gpu_accelerator or not gpu_detector.pytorch_gpu_accelerator.is_available():
            # Fallback: process images sequentially
            if status_callback:
                status_callback("⚠️ GPU not available, using sequential processing")
            return None
        
        gpu_accelerator = gpu_detector.pytorch_gpu_accelerator
        
        # Load all templates
        from pathlib import Path
        template_dir = Path("stored_templates")
        if not template_dir.exists():
            return []
        
        templates = []
        template_names = []
        template_files = list(template_dir.rglob("*.png"))
        
        for template_file in template_files:
            try:
                template = cv.imread(str(template_file), cv.IMREAD_COLOR)
                if template is not None:
                    templates.append(template)
                    template_names.append(template_file.stem)
            except:
                continue
        
        if not templates:
            return []
        
        if status_callback:
            status_callback(f"🚀 GPU: Processing {len(images)} images × {len(templates)} templates in parallel...")
        
        # Run multi-image parallel GPU detection
        all_results, timing_info = gpu_accelerator.parallel_multi_image_detection(
            images, templates, cv.TM_CCOEFF_NORMED, return_timing=True
        )
        
        # Log performance
        if not timing_info.get('fallback', False):
            ops_per_sec = timing_info.get('operations_per_second', 0)
            imgs_per_sec = timing_info.get('images_per_second', 0)
            concurrent = timing_info.get('max_concurrent_images', 1)
            
            if status_callback:
                status_callback(f"⚡ GPU: {ops_per_sec:.0f} ops/sec, {imgs_per_sec:.1f} imgs/sec ({concurrent} concurrent)")
        
        # Process results into detection format
        processed_results = []
        threshold = detection_params.get('threshold', 0.7)
        min_confidence = detection_params.get('min_confidence', 0.5)
        
        for img_idx, (image_results, image, filename) in enumerate(zip(all_results, images, image_filenames)):
            img_h, img_w = image.shape[:2]
            
            image_detections = {
                'filename': filename,
                'image': cv_to_pil(image),
                'matches': []
            }
            
            # Process each template's results for this image
            for template_idx, (result, template_name) in enumerate(zip(image_results, template_names)):
                if result is None:
                    continue
                
                template = templates[template_idx]
                template_h, template_w = template.shape[:2]
                
                # Find all detections above threshold
                locations = np.where(result >= threshold)
                
                if len(locations) >= 2 and len(locations[0]) > 0:
                    for y, x in zip(locations[0], locations[1]):
                        confidence = float(result[y, x])
                        
                        if confidence >= min_confidence:
                            detection = {
                                'template_name': template_name,
                                'confidence': confidence,
                                'x': int(x),
                                'y': int(y),
                                'width': template_w,
                                'height': template_h,
                                'center_x': int(x + template_w/2),
                                'center_y': int(y + template_h/2),
                                'bbox': [int(x), int(y), template_w, template_h],
                                'partial': False,
                                'detection_type': 'gpu_multi_image_parallel',
                                'method': 'TM_CCOEFF_NORMED'
                            }
                            image_detections['matches'].append(detection)
            
            # Apply NMS to remove overlapping detections
            if image_detections['matches']:
                image_detections['matches'] = apply_nms_to_detections(
                    image_detections['matches'], 
                    overlap_threshold=0.3
                )
            
            processed_results.append(image_detections)
        
        return processed_results
        
    except Exception as e:
        if status_callback:
            status_callback(f"⚠️ GPU processing failed: {str(e)}")
        return None


def detect_colored_rectangles(image, min_area=50):
    """
    Detect signal rectangles using void detection module.
    Replaces old colored rectangle detection with precise void-based detection.
    
    Args:
        image: Input image (BGR or RGB format)
        min_area: Minimum area threshold for valid rectangles
        
    Returns:
        List of detected signal rectangle matches in the same format as template matches
    """
    try:
        from void_detection import VoidDetector
        
        # Initialize detector
        detector = VoidDetector()
        detector.signal_min_area = min_area
        
        # Convert image format if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            bgr_image = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
            
            # Image is already in correct color space (loaded properly via pil_to_cv fix)
            # Do NOT apply colormap - that would double-colormap already colored images
        else:
            # Convert grayscale to BGR
            bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        # Use void detection to find signal regions
        signal_mask, signal_contours, signal_stats = detector.detect_signal_regions(bgr_image)
        
        signal_matches = []
        img_h, img_w = bgr_image.shape[:2]
        
        for i, stats in enumerate(signal_stats):
            if stats['area'] >= min_area:
                x, y, w, h = stats['bbox']
                
                # Get the contour for this region
                contour = signal_contours[i] if i < len(signal_contours) else None
                
                # Decompose irregular shapes into rectangles
                rectangles = decompose_to_rectangles(contour, x, y, w, h, min_area)
                
                for rect in rectangles:
                    # Check if rectangle is at border (for marking, not filtering)
                    at_border = is_at_border(rect['x'], rect['y'], rect['width'], rect['height'], img_w, img_h)
                    
                    # Very basic validation (only filter out extremely invalid rectangles)
                    if rect['area'] >= min_area and rect.get('aspect_ratio', 1.0) <= 50:  # More permissive
                        confidence = min(1.0, rect['area'] / 1000.0)
                        
                        signal_matches.append({
                            'template_name': 'Unidentified Signal',
                            'type': 'unidentified',  # Add type field for filtering
                            'confidence': confidence,
                            'x': int(rect['x']),
                            'y': int(rect['y']),
                            'width': int(rect['width']),
                            'height': int(rect['height']),
                            'center_x': int(rect['x'] + rect['width']/2),
                            'center_y': int(rect['y'] + rect['height']/2),
                            'partial': False,
                            'detection_type': 'void_based_signal',
                            'area': int(rect['area']),
                            'aspect_ratio': round(rect.get('aspect_ratio', 1.0), 2),
                            'validation_method': 'void_detection',
                            'at_border': at_border,  # Mark if at border for storage filtering
                            'storage_eligible': True  # In Train Mode, allow all signals (boundary check happens later)
                        })
        
        return signal_matches
        
    except Exception as e:
        # Fallback to empty list if void detection fails
        return []

def is_at_border(x, y, w, h, img_w, img_h, tolerance=5):
    """Check if rectangle touches image borders"""
    return (x <= tolerance or y <= tolerance or 
            x + w >= img_w - tolerance or y + h >= img_h - tolerance)

def decompose_to_rectangles(contour, x, y, w, h, min_area):
    """
    Decompose irregular shapes into rectangles.
    If shape is approximately rectangular, keep as single rectangle.
    If irregular, decompose into multiple rectangles.
    """
    rectangles = []
    
    if contour is not None and len(contour) > 4:
        # Check if shape is approximately rectangular
        if is_approximately_rectangular(contour):
            # Keep as single rectangle
            rectangles.append({
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': w * h, 'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            })
        else:
            # Decompose into multiple rectangles
            decomposed = decompose_irregular_shape(contour, min_area)
            rectangles.extend(decomposed)
    else:
        # Simple rectangular shape
        rectangles.append({
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': w * h, 'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        })
    
    return rectangles

def is_approximately_rectangular(contour):
    """Check if contour is approximately rectangular"""
    # Approximate contour to polygon
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    # If approximation has 4 vertices, it's rectangular
    return len(approx) == 4

def decompose_irregular_shape(contour, min_area):
    """Decompose irregular shape into multiple rectangles"""
    rectangles = []
    
    # Get bounding rectangle of contour
    x, y, w, h = cv.boundingRect(contour)
    
    # Create mask for this contour
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Translate contour to local coordinates
    local_contour = contour - [x, y]
    cv.fillPoly(mask[1:-1, 1:-1], [local_contour], 255)
    
    # Find connected components within the shape
    num_labels, labels = cv.connectedComponents(mask[1:-1, 1:-1])
    
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        component_contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for comp_contour in component_contours:
            comp_x, comp_y, comp_w, comp_h = cv.boundingRect(comp_contour)
            comp_area = cv.contourArea(comp_contour)
            
            if comp_area >= min_area:
                # Convert back to global coordinates
                global_x = x + comp_x
                global_y = y + comp_y
                
                rectangles.append({
                    'x': global_x, 'y': global_y, 'width': comp_w, 'height': comp_h,
                    'area': comp_area, 'aspect_ratio': max(comp_w, comp_h) / min(comp_w, comp_h) if min(comp_w, comp_h) > 0 else 1.0
                })
    
    # If no valid decomposition, return original bounding rectangle
    if not rectangles:
        rectangles.append({
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': w * h, 'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        })
    
    return rectangles

def is_basic_valid_rectangle(rect):
    """Basic validation for rectangle (aspect ratio, minimum size) - no border check"""
    # Reasonable aspect ratio
    aspect_ratio = rect.get('aspect_ratio', 1.0)
    if aspect_ratio > 20:  # Too thin
        return False
    
    # Minimum size
    if rect['area'] < 25:  # Very small rectangles
        return False
    
    return True

def is_valid_signal_rectangle(rect, img_w, img_h):
    """Full validation including border check (for storage eligibility)"""
    x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
    
    # Basic validation first
    if not is_basic_valid_rectangle(rect):
        return False
    
    # Not at borders (for storage)
    if is_at_border(x, y, w, h, img_w, img_h):
        return False
    
    return True

def validate_and_split_rectangle(x, y, w, h, color_mask, hsv, min_area):
    """
    Validate if a rectangle has colored corners, and if not, split it into separate colored areas.
    
    Args:
        x, y, w, h: Rectangle coordinates and dimensions
        color_mask: Binary mask of colored areas
        hsv: HSV image for color validation
        min_area: Minimum area for valid rectangles
        
    Returns:
        List of validated rectangle dictionaries
    """
    # Check corner colors with a small tolerance around each corner
    corner_tolerance = 3
    corners = [
        (x, y),  # Top-left
        (x + w - 1, y),  # Top-right
        (x, y + h - 1),  # Bottom-left
        (x + w - 1, y + h - 1)  # Bottom-right
    ]
    
    corners_have_color = []
    for corner_x, corner_y in corners:
        # Check a small area around each corner
        corner_has_color = False
        for dx in range(-corner_tolerance, corner_tolerance + 1):
            for dy in range(-corner_tolerance, corner_tolerance + 1):
                check_x = max(0, min(color_mask.shape[1] - 1, corner_x + dx))
                check_y = max(0, min(color_mask.shape[0] - 1, corner_y + dy))
                
                if color_mask[check_y, check_x] > 0:
                    corner_has_color = True
                    break
            if corner_has_color:
                break
        corners_have_color.append(corner_has_color)
    
    # Count how many corners have color
    colored_corners = sum(corners_have_color)
    
    # If most corners have color, it's likely a valid single rectangle
    if colored_corners >= 3:  # At least 3 out of 4 corners have color
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        return [{
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': w * h,
            'aspect_ratio': aspect_ratio,
            'detection_type': 'validated_single_rectangle',
            'validation_method': f'corner_validation_{colored_corners}_corners'
        }]
    
    # If few corners have color, likely multiple rectangles - try to split
    else:
        return split_into_separate_rectangles(x, y, w, h, color_mask, min_area)

def split_into_separate_rectangles(x, y, w, h, color_mask, min_area):
    """
    Split a large rectangle into separate colored areas using connected component analysis.
    
    Args:
        x, y, w, h: Rectangle coordinates and dimensions
        color_mask: Binary mask of colored areas
        min_area: Minimum area for valid rectangles
        
    Returns:
        List of separate rectangle dictionaries
    """
    # Extract the region of interest from the color mask
    roi_mask = color_mask[y:y+h, x:x+w].copy()
    
    if roi_mask.size == 0:
        return []
    
    # Find connected components in the ROI
    num_labels, labels_im = cv.connectedComponents(roi_mask)
    
    separate_rectangles = []
    
    # Process each connected component (skip label 0 which is background)
    for label in range(1, num_labels):
        # Create mask for this component
        component_mask = (labels_im == label).astype(np.uint8) * 255
        
        # Find bounding rectangle for this component
        component_contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if component_contours:
            # Get the largest contour for this component
            largest_contour = max(component_contours, key=cv.contourArea)
            comp_x, comp_y, comp_w, comp_h = cv.boundingRect(largest_contour)
            comp_area = cv.contourArea(largest_contour)
            
            # Convert coordinates back to original image space
            abs_x = x + comp_x
            abs_y = y + comp_y
            
            # Only keep if it meets minimum area requirement
            if comp_area >= min_area:
                aspect_ratio = max(comp_w, comp_h) / min(comp_w, comp_h) if min(comp_w, comp_h) > 0 else float('inf')
                
                # Only keep reasonable rectangles (not extremely thin)
                if aspect_ratio <= 20:
                    separate_rectangles.append({
                        'x': abs_x, 'y': abs_y, 'width': comp_w, 'height': comp_h,
                        'area': comp_area,
                        'aspect_ratio': aspect_ratio,
                        'detection_type': 'split_rectangle',
                        'validation_method': f'connected_component_{label}'
                    })
    
    # If no valid separate rectangles found, return the original as fallback
    if not separate_rectangles:
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        return [{
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': w * h,
            'aspect_ratio': aspect_ratio,
            'detection_type': 'fallback_rectangle',
            'validation_method': 'no_valid_splits_found'
        }]
    
    return separate_rectangles

def remove_overlapping_detections(identified_matches, colored_matches, overlap_threshold=0.3):
    """
    Remove colored rectangle detections that overlap significantly with identified drone detections.
    Enhanced to detect same drone with different reference points (aligned sides, different heights).
    
    Args:
        identified_matches: List of template-based drone detections
        colored_matches: List of colored rectangle detections
        overlap_threshold: Minimum overlap ratio to consider a conflict
        
    Returns:
        Filtered list of colored matches without overlaps with identified drones
    """
    if not identified_matches or not colored_matches:
        return colored_matches
    
    filtered_colored = []
    
    for colored_match in colored_matches:
        colored_box = (colored_match['x'], colored_match['y'], colored_match['width'], colored_match['height'])
        has_overlap = False
        
        for identified_match in identified_matches:
            identified_box = (identified_match['x'], identified_match['y'], 
                            identified_match['width'], identified_match['height'])
            
            # Calculate standard overlap percentage
            overlap_pct = calculate_overlap_percentage(colored_box, identified_box)
            
            # Check for standard overlap
            if overlap_pct >= overlap_threshold:
                has_overlap = True
                break
            
            # Enhanced check for same drone with different reference points
            # Check if sides are aligned but heights are different
            if is_same_drone_different_height(colored_box, identified_box, tolerance=10):
                has_overlap = True
                break
        
        if not has_overlap:
            filtered_colored.append(colored_match)
    
    return filtered_colored

def is_same_drone_different_height(box1, box2, tolerance=10):
    """
    Check if two rectangles represent the same drone detected with different heights.
    This happens when colored area detection and template matching use different reference points.
    
    Args:
        box1: (x, y, width, height) - colored rectangle
        box2: (x, y, width, height) - identified drone rectangle  
        tolerance: Pixel tolerance for alignment checking
        
    Returns:
        True if likely the same drone with different height reference
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate edges
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Check for horizontal alignment (similar x-coordinates and widths)
    left_aligned = abs(left1 - left2) <= tolerance
    right_aligned = abs(right1 - right2) <= tolerance
    width_similar = abs(w1 - w2) <= tolerance * 2
    
    # Check for vertical overlap/containment with different heights
    vertical_overlap = not (bottom1 <= top2 or bottom2 <= top1)
    height_different = abs(h1 - h2) > tolerance
    
    # Case 1: Left and right sides aligned, different heights, vertical overlap
    if left_aligned and right_aligned and height_different and vertical_overlap:
        return True
    
    # Case 2: Similar width and horizontal position, but different heights
    if width_similar and abs((left1 + right1)/2 - (left2 + right2)/2) <= tolerance and height_different and vertical_overlap:
        return True
    
    # Case 3: One rectangle is horizontally contained within the other with similar center
    horizontal_contained = (left2 <= left1 <= right1 <= right2) or (left1 <= left2 <= right2 <= right1)
    center_x1 = (left1 + right1) / 2
    center_x2 = (left2 + right2) / 2
    horizontal_center_close = abs(center_x1 - center_x2) <= tolerance
    
    if horizontal_contained and horizontal_center_close and height_different and vertical_overlap:
        return True
    
    # Case 4: Check for vertical alignment (same drone rotated or detected differently)
    top_aligned = abs(top1 - top2) <= tolerance
    bottom_aligned = abs(bottom1 - bottom2) <= tolerance
    height_similar = abs(h1 - h2) <= tolerance * 2
    
    # Horizontal overlap with different widths but aligned vertically
    horizontal_overlap = not (right1 <= left2 or right2 <= left1)
    width_different = abs(w1 - w2) > tolerance
    
    if top_aligned and bottom_aligned and width_different and horizontal_overlap:
        return True
    
    if height_similar and abs((top1 + bottom1)/2 - (top2 + bottom2)/2) <= tolerance and width_different and horizontal_overlap:
        return True
    
    return False

def merge_colored_rectangles(colored_matches, merge_threshold=0.1):
    """
    Merge overlapping colored rectangles to avoid duplicate detections.
    Uses stricter criteria to prevent merging distinct drones.
    
    Args:
        colored_matches: List of colored rectangle detections
        merge_threshold: Minimum overlap ratio to merge rectangles
        
    Returns:
        List of merged colored rectangle detections
    """
    if len(colored_matches) <= 1:
        return colored_matches
    
    merged = []
    processed = set()
    
    for i, match1 in enumerate(colored_matches):
        if i in processed:
            continue
            
        # Start with the current rectangle
        current_group = [match1]
        current_indices = {i}
        
        # Only look for direct overlaps with the current rectangle (no chain-linking)
        for j, match2 in enumerate(colored_matches):
            if j in processed or j == i:
                continue
                
            box1 = (match1['x'], match1['y'], match1['width'], match1['height'])
            box2 = (match2['x'], match2['y'], match2['width'], match2['height'])
            
            # Calculate overlap percentage
            overlap_pct = max(
                calculate_overlap_percentage(box1, box2),
                calculate_overlap_percentage(box2, box1)
            )
            
            # Only merge if significant overlap AND strict criteria are met
            if overlap_pct >= merge_threshold:
                # Calculate size difference ratio
                area1 = match1['area']
                area2 = match2['area']
                size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
                
                # Calculate distance between centers
                center1_x = match1['center_x']
                center1_y = match1['center_y']
                center2_x = match2['center_x']
                center2_y = match2['center_y']
                distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                
                # Calculate maximum allowed distance based on rectangle sizes
                avg_size = (np.sqrt(area1) + np.sqrt(area2)) / 2
                max_allowed_distance = avg_size * 1.2  # Allow 20% larger distance than average size
                
                # Strict merging criteria:
                # 1. High overlap (> merge_threshold)
                # 2. Similar sizes (ratio > 0.6 instead of 0.3)
                # 3. Close proximity (distance < max_allowed_distance)
                # 4. Very high overlap for different sizes (> 0.7 if size_ratio < 0.6)
                should_merge = False
                
                if size_ratio > 0.6 and distance < max_allowed_distance:
                    # Similar sizes and close proximity
                    should_merge = True
                elif overlap_pct > 0.7 and distance < max_allowed_distance * 0.8:
                    # Very high overlap and very close (for slightly different sizes)
                    should_merge = True
                
                if should_merge:
                    current_group.append(match2)
                    current_indices.add(j)
        
        # Mark all rectangles in current group as processed
        processed.update(current_indices)
        
        if len(current_group) == 1:
            # No merging needed, keep original
            merged.append(current_group[0])
        else:
            # Merge only if the group makes sense (not too spread out)
            # Check if merged rectangle would be reasonable
            min_x = min(r['x'] for r in current_group)
            min_y = min(r['y'] for r in current_group)
            max_x = max(r['x'] + r['width'] for r in current_group)
            max_y = max(r['y'] + r['height'] for r in current_group)
            
            merged_width = max_x - min_x
            merged_height = max_y - min_y
            merged_area_bbox = merged_width * merged_height
            actual_area = sum(r['area'] for r in current_group)
            
            # Only merge if the bounding box isn't too much larger than actual content
            area_efficiency = actual_area / merged_area_bbox if merged_area_bbox > 0 else 0
            
            if area_efficiency > 0.3:  # At least 30% of bounding box should be filled
                # Create merged rectangle
                merged_match = {
                    'template_name': 'Unidentified Signal',
                    'confidence': max(r['confidence'] for r in current_group),
                    'x': int(min_x),
                    'y': int(min_y),
                    'width': int(merged_width),
                    'height': int(merged_height),
                    'center_x': int(min_x + merged_width/2),
                    'center_y': int(min_y + merged_height/2),
                    'partial': False,
                    'detection_type': 'merged_colored_rectangle',
                    'area': int(actual_area),
                    'aspect_ratio': round(max(merged_width, merged_height) / min(merged_width, merged_height), 2) if min(merged_width, merged_height) > 0 else 1.0,
                    'merged_count': len(current_group),
                    'area_efficiency': round(area_efficiency, 3)
                }
                merged.append(merged_match)
            else:
                # Don't merge - bounding box would be too sparse, keep separate
                merged.extend(current_group)
    
    return merged

def preprocess_for_matching(img):
    """Normalize image for template matching."""
    img = img.astype(np.float32)
    img -= np.mean(img)
    std = np.std(img)
    if std > 0:
        img /= std
    return img

def detect_pattern(image, template, method, threshold, template_name, partial_threshold=0.3, 
                  enable_border_detection=True, merge_overlapping=True, overlap_sensitivity=0.3):
    try:
        if len(image.shape) == 3:
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        if len(template.shape) == 3:
            template_gray = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
        else:
            template_gray = template.copy()
        h, w = template_gray.shape
        img_h, img_w = img_gray.shape

        # Validate template size - template must be smaller than or equal to image
        if h > img_h or w > img_w:
            # Template is larger than image - cannot perform template matching
            # This prevents OpenCV assertion error: template size must be ≤ image size
            return []

        # Normalize both images
        img_gray = preprocess_for_matching(img_gray)
        template_gray = preprocess_for_matching(template_gray)

        matches = []
        
        # Standard template matching
        res = cv.matchTemplate(img_gray, template_gray, method)
        
        # For TM_CCOEFF_NORMED, higher values are better matches
        loc = np.where(res >= threshold)
        confidences = res[loc]
        partial_loc = np.where((res < threshold) & (res >= partial_threshold))
        partial_confidences = res[partial_loc]

        # Full matches
        for i, (x, y) in enumerate(zip(loc[1], loc[0])):
            matches.append({
                'template_name': template_name,
                'confidence': float(confidences[i]),
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'center_x': int(x + w/2),
                'center_y': int(y + h/2),
                'partial': False,
                'detection_type': 'full'
            })
            
        # Partial matches (template cropped by image borders)
        for i, (x, y) in enumerate(zip(partial_loc[1], partial_loc[0])):
            if is_partial_border(x, y, w, h, img_w, img_h):
                matches.append({
                    'template_name': template_name,
                    'confidence': float(partial_confidences[i]),
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'center_x': int(x + w/2),
                    'center_y': int(y + h/2),
                    'partial': True,
                    'detection_type': 'border_partial'
                })

        # Multi-scale template matching for border detection (only if enabled)
        if enable_border_detection:
            border_matches = detect_border_templates(img_gray, template_gray, method, partial_threshold, template_name)
            matches.extend(border_matches)

        # Filter duplicates based on user settings
        if merge_overlapping:
            matches = filter_duplicates(matches, iou_threshold=overlap_sensitivity)
        
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches

    except Exception as e:
        st.error(f"Error in pattern detection: {str(e)}")
        return []

def detect_pattern_parallel(image, template, method, threshold, template_name, partial_threshold=0.3, 
                           enable_border_detection=True, merge_overlapping=True, overlap_sensitivity=0.3,
                           parallel_config=None):
    """
    Parallel version of detect_pattern that processes image quadrants simultaneously.
    
    Args:
        image: Input image for detection
        template: Template to search for
        method: OpenCV template matching method
        threshold: Detection threshold
        template_name: Name of the template
        partial_threshold: Threshold for partial detections
        enable_border_detection: Whether to enable border detection
        merge_overlapping: Whether to merge overlapping detections
        overlap_sensitivity: Sensitivity for overlap merging
        parallel_config: ParallelDetectionConfig object
        
    Returns:
        List of detection matches with parallel processing optimizations
    """
    try:
        # Check if parallel processing should be used
        if not parallel_config or not parallel_config.enabled:
            # Fall back to standard detection
            return detect_pattern(image, template, method, threshold, template_name, 
                                partial_threshold, enable_border_detection, merge_overlapping, overlap_sensitivity)
        
        # Check if image is large enough to benefit from parallelization
        img_h, img_w = image.shape[:2] if len(image.shape) > 1 else (image.shape[0], 1)
        if img_h * img_w < parallel_config.min_image_size ** 2:
            # Image too small, use standard detection
            return detect_pattern(image, template, method, threshold, template_name, 
                                partial_threshold, enable_border_detection, merge_overlapping, overlap_sensitivity)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        if len(template.shape) == 3:
            template_gray = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
        else:
            template_gray = template.copy()
        
        h, w = template_gray.shape
        
        # Validate template size
        if h > img_h or w > img_w:
            return []
        
        # Normalize images
        img_gray = preprocess_for_matching(img_gray)
        template_gray = preprocess_for_matching(template_gray)
        
        # Divide image into quadrants
        quadrants = divide_image_into_quadrants(img_gray, parallel_config.overlap_percentage)
        
        def process_quadrant(quad_data):
            """Process a single quadrant"""
            quad_img, quad_offset = quad_data
            quad_matches = []
            
            # Skip quadrants that are too small for the template
            if quad_img.shape[0] < h or quad_img.shape[1] < w:
                return []
            
            try:
                # Standard template matching on quadrant
                res = cv.matchTemplate(quad_img, template_gray, method)
                
                # Full matches
                loc = np.where(res >= threshold)
                confidences = res[loc]
                
                for i, (x, y) in enumerate(zip(loc[1], loc[0])):
                    quad_matches.append({
                        'template_name': template_name,
                        'confidence': float(confidences[i]),
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(x + w/2),
                        'center_y': int(y + h/2),
                        'partial': False,
                        'detection_type': 'parallel_full'
                    })
                
                # Partial matches if enabled
                if enable_border_detection:
                    partial_loc = np.where((res < threshold) & (res >= partial_threshold))
                    partial_confidences = res[partial_loc]
                    
                    for i, (x, y) in enumerate(zip(partial_loc[1], partial_loc[0])):
                        if is_partial_border(x, y, w, h, quad_img.shape[1], quad_img.shape[0]):
                            quad_matches.append({
                                'template_name': template_name,
                                'confidence': float(partial_confidences[i]),
                                'x': int(x),
                                'y': int(y),
                                'width': int(w),
                                'height': int(h),
                                'center_x': int(x + w/2),
                                'center_y': int(y + h/2),
                                'partial': True,
                                'detection_type': 'parallel_partial'
                            })
                
                # Transform coordinates back to full image space
                return transform_coordinates_from_quadrant(quad_matches, quad_offset)
                
            except Exception as e:
                return []
        
        # Process quadrants in parallel
        all_matches = []
        if parallel_config.use_threading:
            # Use ThreadPoolExecutor for I/O bound operations (better for OpenCV)
            with ThreadPoolExecutor(max_workers=parallel_config.max_workers) as executor:
                quadrant_results = list(executor.map(process_quadrant, quadrants))
        else:
            # Use ProcessPoolExecutor for CPU-bound operations
            with ProcessPoolExecutor(max_workers=parallel_config.max_workers) as executor:
                quadrant_results = list(executor.map(process_quadrant, quadrants))
        
        # Combine results from all quadrants
        for quad_matches in quadrant_results:
            all_matches.extend(quad_matches)
        
        # Remove duplicates that occur in overlapping regions
        if merge_overlapping and all_matches:
            all_matches = filter_duplicates_parallel(all_matches, iou_threshold=overlap_sensitivity)
        
        # Sort by confidence
        all_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_matches
        
    except Exception as e:
        # Fall back to standard detection
        return detect_pattern(image, template, method, threshold, template_name, 
                            partial_threshold, enable_border_detection, merge_overlapping, overlap_sensitivity)

def filter_duplicates_parallel(matches, iou_threshold=0.2):
    """
    Enhanced duplicate filtering for parallel processing results.
    Handles cases where the same object is detected in multiple overlapping quadrants.
    """
    if not matches:
        return matches
    
    filtered = []
    processed_indices = set()
    
    for i, match in enumerate(matches):
        if i in processed_indices:
            continue
            
        # Find all duplicates for this match (including cross-quadrant duplicates)
        duplicates = [match]
        duplicate_indices = {i}
        
        for j, other_match in enumerate(matches[i+1:], i+1):
            if j in processed_indices:
                continue
                
            if match['template_name'] == other_match['template_name']:
                iou_score = iou((match['x'], match['y'], match['width'], match['height']),
                               (other_match['x'], other_match['y'], other_match['width'], other_match['height']))
                
                overlap_score = calculate_overlap_percentage(
                    (match['x'], match['y'], match['width'], match['height']),
                    (other_match['x'], other_match['y'], other_match['width'], other_match['height'])
                )
                
                # More aggressive merging for parallel results (lower threshold)
                if iou_score > iou_threshold * 0.7 or overlap_score > 0.25:  # 25% overlap for parallel processing
                    duplicates.append(other_match)
                    duplicate_indices.add(j)
        
        # Mark all duplicates as processed
        processed_indices.update(duplicate_indices)
        
        # Create averaged match from duplicates
        if len(duplicates) == 1:
            filtered.append(duplicates[0])
        else:
            # Enhanced averaging for parallel processing
            total_weight = sum(d['confidence'] for d in duplicates)
            
            # Calculate weighted average position
            weighted_x = sum(d['x'] * d['confidence'] for d in duplicates) / total_weight
            weighted_y = sum(d['y'] * d['confidence'] for d in duplicates) / total_weight
            
            # Boost confidence for cross-quadrant detections (higher reliability)
            max_confidence = max(d['confidence'] for d in duplicates)
            avg_confidence = sum(d['confidence'] for d in duplicates) / len(duplicates)
            cross_quadrant_boost = 0.08 if len(duplicates) > 2 else 0.05  # Extra boost for multiple quadrant detection
            boosted_confidence = min(1.0, max_confidence * 0.6 + avg_confidence * 0.4 + cross_quadrant_boost)
            
            # Determine if this was a cross-quadrant detection
            quadrant_offsets = [d.get('quadrant_offset') for d in duplicates if d.get('quadrant_offset')]
            is_cross_quadrant = len(set(str(offset) for offset in quadrant_offsets)) > 1
            
            averaged_match = {
                'template_name': match['template_name'],
                'confidence': float(boosted_confidence),
                'x': int(weighted_x),
                'y': int(weighted_y),
                'width': match['width'],
                'height': match['height'],
                'center_x': int(weighted_x + match['width']/2),
                'center_y': int(weighted_y + match['height']/2),
                'partial': any(d.get('partial', False) for d in duplicates),
                'detection_type': 'parallel_merged_cross_quadrant' if is_cross_quadrant else 'parallel_merged',
                'duplicate_count': len(duplicates),
                'merge_reason': 'cross_quadrant_overlap' if is_cross_quadrant else 'same_quadrant_overlap',
                'processed_in_quadrant': True,
                'cross_quadrant_detection': is_cross_quadrant
            }
            filtered.append(averaged_match)
    
    return filtered

def process_template_match_result(match_result, threshold, template_name, template):
    """
    Convert OpenCV matchTemplate result to Smart Astra detection format
    """
    detections = []
    
    if match_result is None or match_result.size == 0:
        return detections
    
    # Find all locations above threshold
    locations = np.where(match_result >= threshold)
    
    if len(locations[0]) == 0:
        return detections
    
    # Get template dimensions
    h, w = template.shape[:2]
    
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        confidence = match_result[pt[1], pt[0]]
        
        detection = {
            'template_name': template_name,
            'confidence': float(confidence),
            'x': int(pt[0]),
            'y': int(pt[1]),
            'width': w,
            'height': h,
            'center_x': int(pt[0] + w/2),
            'center_y': int(pt[1] + h/2),
            'detection_type': 'smart_backend',
            'partial': False
        }
        detections.append(detection)
    
    return detections

def detect_pattern_smart(image, template, method, threshold, template_name, parallel_config=None):
    """
    Smart template matching that automatically selects the best performing backend.
    
    Args:
        image: Input image for detection
        template: Template to search for
        method: OpenCV template matching method
        threshold: Detection threshold
        template_name: Name of the template
        parallel_config: ParallelDetectionConfig object
        
    Returns:
        List of detection matches using the optimal backend
    """
    try:
        # Use smart backend selector if available
        if parallel_config and parallel_config.gpu_detector.smart_backend_selector:
            result = parallel_config.gpu_detector.smart_backend_selector.template_match_smart(
                image, template, method
            )
            
            # Convert result to detection format
            if result is not None:
                return process_template_match_result(result, threshold, template_name, template)
        
        # Fallback to GPU detection if smart selector not available
        return detect_pattern_gpu(image, template, method, threshold, template_name, parallel_config)
        
    except Exception as e:
        # Final fallback to CPU
        return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                     threshold, True, True, 0.3, parallel_config)

def detect_pattern_gpu(image, template, method, threshold, template_name, parallel_config=None):
    """
    GPU-accelerated template matching using PyTorch CUDA (preferred) or OpenCV CUDA/OpenCL.
    
    Args:
        image: Input image for detection
        template: Template to search for
        method: OpenCV template matching method
        threshold: Detection threshold
        template_name: Name of the template
        parallel_config: ParallelDetectionConfig object
        
    Returns:
        List of detection matches with GPU acceleration
    """
    try:
        # Check if GPU should be used
        if not parallel_config or not parallel_config.should_use_gpu():
            # Fall back to CPU detection
            return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                         threshold, True, True, 0.3, parallel_config)
        
        # Try PyTorch GPU acceleration first (most reliable and user-friendly)
        if parallel_config.gpu_detector.pytorch_cuda_available:
            return detect_pattern_pytorch_gpu(image, template, method, threshold, template_name, parallel_config)
        
        # Fall back to OpenCV CUDA/OpenCL if PyTorch not available
        return detect_pattern_opencv_gpu(image, template, method, threshold, template_name, parallel_config)
        
    except Exception as e:
        return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                     threshold, True, True, 0.3, parallel_config)


def detect_pattern_pytorch_gpu(image, template, method, threshold, template_name, parallel_config=None):
    """
    PyTorch GPU-accelerated template matching using the optimized PyTorchGPUAccelerator.
    """
    try:
        # Check if PyTorch GPU accelerator is available
        if not parallel_config or not parallel_config.gpu_detector.pytorch_gpu_accelerator:
            # Fall back to CPU detection
            return detect_pattern(image, template, method, threshold, template_name)
        
        # Use the optimized GPU accelerator
        gpu_accelerator = parallel_config.gpu_detector.pytorch_gpu_accelerator
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        if len(template.shape) == 3:
            template_gray = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
        else:
            template_gray = template.copy()
        
        h, w = template_gray.shape
        img_h, img_w = img_gray.shape
        
        # Validate template size
        if h > img_h or w > img_w:
            return []
        
        # Normalize images
        img_gray = preprocess_for_matching(img_gray)
        template_gray = preprocess_for_matching(template_gray)
        
        # Use the optimized GPU template matching - return numpy array result
        result_np = gpu_accelerator.template_match_gpu(img_gray, template_gray, method)
        
        # Find matches above threshold
        matches = []
        loc = np.where(result_np >= threshold)
        confidences = result_np[loc]
        
        for i, (x, y) in enumerate(zip(loc[1], loc[0])):
            matches.append({
                'template_name': template_name,
                'confidence': float(confidences[i]),
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'center_x': int(x + w // 2),
                'center_y': int(y + h // 2),
                'area': int(w * h),
                'match_type': 'gpu_pytorch'
            })
        
        return matches
        
    except Exception as e:
        # Fall back to CPU detection
        return detect_pattern(image, template, method, threshold, template_name)


def detect_pattern_opencv_gpu(image, template, method, threshold, template_name, parallel_config=None):
    """
    GPU-accelerated template matching using OpenCV CUDA or OpenCL.
    
    Args:
        image: Input image for detection
        template: Template to search for
        method: OpenCV template matching method
        threshold: Detection threshold
        template_name: Name of the template
        parallel_config: ParallelDetectionConfig object
        
    Returns:
        List of detection matches with GPU acceleration
    """
    try:
        # Check if GPU should be used
        if not parallel_config or not parallel_config.should_use_gpu():
            # Fall back to CPU detection
            return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                         threshold, True, True, 0.3, parallel_config)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        if len(template.shape) == 3:
            template_gray = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
        else:
            template_gray = template.copy()
        
        h, w = template_gray.shape
        img_h, img_w = img_gray.shape
        
        # Validate template size
        if h > img_h or w > img_w:
            return []
        
        # Normalize images
        img_gray = preprocess_for_matching(img_gray)
        template_gray = preprocess_for_matching(template_gray)
        
        matches = []
        
        # Try CUDA acceleration first
        if parallel_config.prefer_cuda and parallel_config.gpu_detector.cuda_available:
            try:
                # Upload images to GPU
                gpu_img = cv.cuda_GpuMat()
                gpu_template = cv.cuda_GpuMat()
                gpu_result = cv.cuda_GpuMat()
                
                gpu_img.upload(img_gray.astype(np.float32))
                gpu_template.upload(template_gray.astype(np.float32))
                
                # Perform template matching on GPU
                cv.cuda.matchTemplate(gpu_img, gpu_template, method, gpu_result)
                
                # Download result back to CPU
                result = gpu_result.download()
                
                # Process matches
                loc = np.where(result >= threshold)
                confidences = result[loc]
                
                for i, (x, y) in enumerate(zip(loc[1], loc[0])):
                    matches.append({
                        'template_name': template_name,
                        'confidence': float(confidences[i]),
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(x + w/2),
                        'center_y': int(y + h/2),
                        'partial': False,
                        'detection_type': 'gpu_cuda',
                        'gpu_accelerated': True
                    })
                
                return matches
                
            except Exception as cuda_error:
                pass
        # Try OpenCL acceleration
        if parallel_config.gpu_detector.opencl_available:
            try:
                # Enable OpenCL
                cv.ocl.setUseOpenCL(True)
                
                # Convert to UMat for OpenCL acceleration
                img_umat = cv.UMat(img_gray.astype(np.float32))
                template_umat = cv.UMat(template_gray.astype(np.float32))
                
                # Perform template matching with OpenCL
                result = cv.matchTemplate(img_umat, template_umat, method)
                
                # Convert back to numpy array
                result_np = result.get()
                
                # Process matches
                loc = np.where(result_np >= threshold)
                confidences = result_np[loc]
                
                for i, (x, y) in enumerate(zip(loc[1], loc[0])):
                    matches.append({
                        'template_name': template_name,
                        'confidence': float(confidences[i]),
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(x + w/2),
                        'center_y': int(y + h/2),
                        'partial': False,
                        'detection_type': 'gpu_opencl',
                        'gpu_accelerated': True
                    })
                
                return matches
                
            except Exception as opencl_error:
                pass
        
        # Fall back to CPU if GPU acceleration fails
        return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                     threshold, True, True, 0.3, parallel_config)
        
    except Exception as e:
        return detect_pattern_parallel(image, template, method, threshold, template_name, 
                                     threshold, True, True, 0.3, parallel_config)

def detect_pattern_adaptive(image, template, method, threshold, template_name, partial_threshold=0.3, 
                           enable_border_detection=True, merge_overlapping=True, overlap_sensitivity=0.3,
                           parallel_config=None):
    """
    Adaptive pattern detection that chooses the best method based on system state.
    
    Args:
        image: Input image for detection
        template: Template to search for
        method: OpenCV template matching method
        threshold: Detection threshold
        template_name: Name of the template
        partial_threshold: Threshold for partial detections
        enable_border_detection: Whether to enable border detection
        merge_overlapping: Whether to merge overlapping detections
        overlap_sensitivity: Sensitivity for overlap merging
        parallel_config: ParallelDetectionConfig object
        
    Returns:
        List of detection matches using the most appropriate method
    """
    if not parallel_config:
        # No configuration, use standard detection
        return detect_pattern(image, template, method, threshold, template_name, 
                            partial_threshold, enable_border_detection, merge_overlapping, overlap_sensitivity)
    
    # Initialize configuration if needed
    if not hasattr(parallel_config, 'system_monitor'):
        parallel_config.initialize()
    
    # Adapt to current system load
    config_changed = parallel_config.adapt_to_system_load()
    
    # Apply quality factor to threshold
    adjusted_threshold = threshold * parallel_config.quality_factor
    adjusted_threshold = max(0.1, min(0.95, adjusted_threshold))  # Clamp to valid range
    
    # Choose detection method based on configuration
    img_h, img_w = image.shape[:2] if len(image.shape) > 1 else (image.shape[0], 1)
    image_size = img_h * img_w
    
    # Priority 1: GPU acceleration (modern GPUs are fast even for small images)
    if parallel_config.should_use_gpu():
        # For RTX/modern cards, use GPU for all images above 50x50 pixels
        min_gpu_size = 50 * 50  # Much lower threshold for modern GPUs
        if image_size > min_gpu_size:
            return detect_pattern_gpu(image, template, method, adjusted_threshold, template_name, parallel_config)
    
    # Priority 2: Parallel CPU processing for larger images without GPU
    if (parallel_config.enabled and 
        image_size > parallel_config.min_image_size ** 2 and
        parallel_config.get_effective_workers() > 1):
        return detect_pattern_parallel(image, template, method, adjusted_threshold, template_name, 
                                     partial_threshold, enable_border_detection, merge_overlapping, 
                                     overlap_sensitivity, parallel_config)
    
    # Standard single-threaded detection for small images or overloaded systems
    else:
        return detect_pattern(image, template, method, adjusted_threshold, template_name, 
                            partial_threshold, enable_border_detection, merge_overlapping, overlap_sensitivity)

def is_partial_border(x, y, w, h, img_w, img_h):
    """
    Returns True if any part of the template at (x, y, w, h) is outside the image bounds.
    This means the template is cropped by the border.
    """
    return (
        x < 0 or
        y < 0 or
        x + w > img_w or
        y + h > img_h
    )

def detect_border_templates(img_gray, template_gray, method, threshold, template_name):
    """
    Detect templates that are partially cut off at image borders.
    This creates cropped versions of the template and searches for them.
    """
    h, w = template_gray.shape
    img_h, img_w = img_gray.shape
    matches = []
    
    # Define border crop percentages to test
    crop_percentages = [0.3, 0.5, 0.7]  # 30%, 50%, 70% of template visible
    
    for crop_pct in crop_percentages:
        # Create cropped templates for each border
        crop_h = int(h * crop_pct)
        crop_w = int(w * crop_pct)
        
        # Top border (bottom part of template visible)
        if crop_h > 10:  # Minimum size check
            cropped_template = template_gray[h-crop_h:, :]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold, 
                template_name, 'top_border', crop_pct, w, h, img_w, img_h
            ))
        
        # Bottom border (top part of template visible)
        if crop_h > 10:
            cropped_template = template_gray[:crop_h, :]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'bottom_border', crop_pct, w, h, img_w, img_h
            ))
        
        # Left border (right part of template visible)
        if crop_w > 10:
            cropped_template = template_gray[:, w-crop_w:]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'left_border', crop_pct, w, h, img_w, img_h
            ))
        
        # Right border (left part of template visible)  
        if crop_w > 10:
            cropped_template = template_gray[:, :crop_w]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'right_border', crop_pct, w, h, img_w, img_h
            ))
        
        # Corner cases
        if crop_h > 10 and crop_w > 10:
            # Top-left corner (bottom-right part visible)
            cropped_template = template_gray[h-crop_h:, w-crop_w:]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'top_left_corner', crop_pct, w, h, img_w, img_h
            ))
            
            # Top-right corner (bottom-left part visible)
            cropped_template = template_gray[h-crop_h:, :crop_w]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'top_right_corner', crop_pct, w, h, img_w, img_h
            ))
            
            # Bottom-left corner (top-right part visible)
            cropped_template = template_gray[:crop_h, w-crop_w:]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'bottom_left_corner', crop_pct, w, h, img_w, img_h
            ))
            
            # Bottom-right corner (top-left part visible)
            cropped_template = template_gray[:crop_h, :crop_w]
            matches.extend(search_cropped_template(
                img_gray, cropped_template, method, threshold,
                template_name, 'bottom_right_corner', crop_pct, w, h, img_w, img_h
            ))
    
    return matches

def search_cropped_template(img_gray, cropped_template, method, threshold, 
                           template_name, border_type, crop_pct, orig_w, orig_h, img_w, img_h):
    """
    Search for a cropped template in the image and adjust coordinates.
    Only search in the specific border region that matches the border_type.
    """
    matches = []
    crop_h, crop_w = cropped_template.shape
    
    # Normalize cropped template
    cropped_template = preprocess_for_matching(cropped_template)
    
    # Define search regions based on SPECIFIC border type
    search_region = None
    region_offset_x = 0
    region_offset_y = 0
    
    if border_type == 'top_border':
        # Only search TOP edge of image
        border_height = min(orig_h, img_h // 2)
        search_region = img_gray[:border_height, :]
        region_offset_y = 0
        region_offset_x = 0
    elif border_type == 'bottom_border':
        # Only search BOTTOM edge of image
        border_height = min(orig_h, img_h // 2)
        search_region = img_gray[-border_height:, :]
        region_offset_y = img_h - border_height
        region_offset_x = 0
    elif border_type == 'left_border':
        # Only search LEFT edge of image
        border_width = min(orig_w, img_w // 2)
        search_region = img_gray[:, :border_width]
        region_offset_y = 0
        region_offset_x = 0
    elif border_type == 'right_border':
        # Only search RIGHT edge of image
        border_width = min(orig_w, img_w // 2)
        search_region = img_gray[:, -border_width:]
        region_offset_y = 0
        region_offset_x = img_w - border_width
    elif border_type == 'top_left_corner':
        # Only search TOP-LEFT corner
        corner_h = min(orig_h, img_h // 3)
        corner_w = min(orig_w, img_w // 3)
        search_region = img_gray[:corner_h, :corner_w]
        region_offset_y = 0
        region_offset_x = 0
    elif border_type == 'top_right_corner':
        # Only search TOP-RIGHT corner
        corner_h = min(orig_h, img_h // 3)
        corner_w = min(orig_w, img_w // 3)
        search_region = img_gray[:corner_h, -corner_w:]
        region_offset_y = 0
        region_offset_x = img_w - corner_w
    elif border_type == 'bottom_left_corner':
        # Only search BOTTOM-LEFT corner
        corner_h = min(orig_h, img_h // 3)
        corner_w = min(orig_w, img_w // 3)
        search_region = img_gray[-corner_h:, :corner_w]
        region_offset_y = img_h - corner_h
        region_offset_x = 0
    elif border_type == 'bottom_right_corner':
        # Only search BOTTOM-RIGHT corner
        corner_h = min(orig_h, img_h // 3)
        corner_w = min(orig_w, img_w // 3)
        search_region = img_gray[-corner_h:, -corner_w:]
        region_offset_y = img_h - corner_h
        region_offset_x = img_w - corner_w
    
    # Skip if search region is too small or invalid
    if search_region is None or search_region.shape[0] < crop_h or search_region.shape[1] < crop_w:
        return matches
    
    # Perform template matching only in the specific border region
    res = cv.matchTemplate(search_region, cropped_template, method)
    
    # For TM_CCOEFF_NORMED, higher values are better matches
    loc = np.where(res >= threshold)
    confidences = res[loc]
    
    # Process matches and adjust coordinates
    for i, (x, y) in enumerate(zip(loc[1], loc[0])):
        # Adjust coordinates to global image coordinates
        global_x = x + region_offset_x
        global_y = y + region_offset_y
        
        # Adjust coordinates to represent the original template position
        adjusted_x, adjusted_y = adjust_coordinates_for_border(
            global_x, global_y, border_type, crop_pct, orig_w, orig_h, crop_w, crop_h
        )
        
        # CRITICAL: Only add detection if it actually touches the correct border(s)
        if is_detection_on_correct_border(adjusted_x, adjusted_y, orig_w, orig_h, img_w, img_h, border_type):
            matches.append({
                'template_name': template_name,
                'confidence': float(confidences[i]) * crop_pct,  # Reduce confidence for partial matches
                'x': int(adjusted_x),
                'y': int(adjusted_y),
                'width': int(orig_w),
                'height': int(orig_h),
                'center_x': int(adjusted_x + orig_w/2),
                'center_y': int(adjusted_y + orig_h/2),
                'partial': True,
                'detection_type': f'border_{border_type}_{int(crop_pct*100)}pct'
            })
    
    return matches

def is_detection_on_correct_border(x, y, w, h, img_w, img_h, border_type):
    """
    Verify that the detection actually touches the border(s) specified by border_type.
    This prevents false corner detections when template only touches one edge.
    """
    # Calculate which borders the template actually touches
    touches_top = y <= 0
    touches_bottom = y + h >= img_h
    touches_left = x <= 0
    touches_right = x + w >= img_w
    
    # Check if the detection matches the expected border type
    if border_type == 'top_border':
        return touches_top and not (touches_left or touches_right)
    elif border_type == 'bottom_border':
        return touches_bottom and not (touches_left or touches_right)
    elif border_type == 'left_border':
        return touches_left and not (touches_top or touches_bottom)
    elif border_type == 'right_border':
        return touches_right and not (touches_top or touches_bottom)
    elif border_type == 'top_left_corner':
        return touches_top and touches_left
    elif border_type == 'top_right_corner':
        return touches_top and touches_right
    elif border_type == 'bottom_left_corner':
        return touches_bottom and touches_left
    elif border_type == 'bottom_right_corner':
        return touches_bottom and touches_right
    
    return False

def adjust_coordinates_for_border(x, y, border_type, crop_pct, orig_w, orig_h, crop_w, crop_h):
    """
    Adjust detected coordinates to represent the original template position.
    """
    if border_type == 'top_border':
        # Template extends above image, adjust y upward
        return x, y - (orig_h - crop_h)
    elif border_type == 'bottom_border':
        # Template extends below image, y stays same
        return x, y
    elif border_type == 'left_border':
        # Template extends left of image, adjust x leftward
        return x - (orig_w - crop_w), y
    elif border_type == 'right_border':
        # Template extends right of image, x stays same
        return x, y
    elif border_type == 'top_left_corner':
        # Template extends top and left
        return x - (orig_w - crop_w), y - (orig_h - crop_h)
    elif border_type == 'top_right_corner':
        # Template extends top and right
        return x, y - (orig_h - crop_h)
    elif border_type == 'bottom_left_corner':
        # Template extends bottom and left
        return x - (orig_w - crop_w), y
    elif border_type == 'bottom_right_corner':
        # Template extends bottom and right
        return x, y
    else:
        return x, y

def draw_dashed_rectangle(img, pt1, pt2, color, thickness, dash_length=10):
    """
    Draw a dashed rectangle on the image.
    
    Args:
        img: Image to draw on
        pt1: Top-left corner (x1, y1)
        pt2: Bottom-right corner (x2, y2)
        color: Line color
        thickness: Line thickness
        dash_length: Length of each dash
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top line
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)
    # Bottom line
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length)
    # Left line
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length)
    # Right line
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
    """
    Draw a dashed line on the image.
    
    Args:
        img: Image to draw on
        pt1: Start point (x1, y1)
        pt2: End point (x2, y2)
        color: Line color
        thickness: Line thickness
        dash_length: Length of each dash
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx*dx + dy*dy)
    
    if line_length == 0:
        return
    
    # Normalize direction vector
    unit_dx = dx / line_length
    unit_dy = dy / line_length
    
    # Draw dashed line
    current_length = 0
    draw_dash = True
    
    while current_length < line_length:
        start_x = int(x1 + unit_dx * current_length)
        start_y = int(y1 + unit_dy * current_length)
        
        end_length = min(current_length + dash_length, line_length)
        end_x = int(x1 + unit_dx * end_length)
        end_y = int(y1 + unit_dy * end_length)
        
        if draw_dash:
            cv.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        
        current_length = end_length
        draw_dash = not draw_dash

def assign_drone_ids(matches):
    """Assign consistent drone IDs to matches for labeling"""
    # Initialize global template mapping if not exists
    if 'template_id_mapping' not in st.session_state:
        st.session_state.template_id_mapping = {}
    
    # Counter only needed for unidentified signals (which are different instances)
    template_instance_counters = {}
    
    for match in matches:
        template = match['template_name']
        
        # Get or create consistent drone type ID for this template
        if template == 'Unidentified Signal':
            drone_type_id = 'US'
        else:
            # Use the actual template name (clean version) instead of generic "Drone Type X"
            # This preserves the meaningful names that users assigned to their templates
            drone_type_id = clean_template_name(template)
            
            # Store in mapping for consistency, but use the actual template name
            if template not in st.session_state.template_id_mapping:
                st.session_state.template_id_mapping[template] = drone_type_id
        
        # Assign the same drone ID to all instances of the same template
        # No instance numbering - all drones of same type get identical labels
        if template == 'Unidentified Signal':
            # For unidentified signals, we still need some differentiation since they're not the same template
            if template not in template_instance_counters:
                template_instance_counters[template] = 0
            template_instance_counters[template] += 1
            match['drone_id'] = f"{drone_type_id} {template_instance_counters[template]}"
        else:
            # For identified drones, always use the same drone type ID - no numbering
            match['drone_id'] = drone_type_id
    
    return matches

def draw_detection_boxes(image, matches, min_conf=0.0):
    """Draw bounding boxes on image for detected patterns with simplified labels and numbering"""
    img_result = image.copy()
    
    # Define colors for different templates and detection types
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    template_colors = {}
    color_idx = 0
    
    # Filter matches by confidence first
    valid_matches = [match for match in matches if match['confidence'] >= min_conf]
    
    # Assign drone IDs consistently
    valid_matches = assign_drone_ids(valid_matches)
    
    for match in valid_matches:
        template = match['template_name']
        
        # Assign color to template if not assigned
        if template not in template_colors:
            template_colors[template] = colors[color_idx % len(colors)]
            color_idx += 1
        
        base_color = template_colors[template]
        
        # Special handling for unidentified signals (colored rectangles)
        if template == 'Unidentified Signal':
            # Use red color (BGR values for OpenCV) for unidentified signals with dashed line style
            base_color = (0, 80, 255)  # Red in BGR: Blue=0, Green=0, Red=255
            thickness = 2
        
        # Adjust color and line style based on detection type
        if match.get('partial', False):
            # Use dashed line for partial detections by alternating segments
            color = tuple(int(c * 0.7) for c in base_color)  # Darker color for partial
            thickness = 3
        elif template == 'Unidentified Signal':
            color = base_color
            thickness = 2
        else:
            color = base_color
            thickness = 2
        
        # Calculate visible portion of the template
        img_h, img_w = image.shape[:2]
        x, y, w, h = match['x'], match['y'], match['width'], match['height']
        
        # Clip rectangle to image boundaries
        visible_x = max(0, x)
        visible_y = max(0, y)
        visible_x2 = min(img_w, x + w)
        visible_y2 = min(img_h, y + h)
        
        # Draw the visible portion of the rectangle
        if visible_x < visible_x2 and visible_y < visible_y2:
            if template == 'Unidentified Signal':
                # Draw dashed rectangle for unidentified signals
                draw_dashed_rectangle(img_result, (visible_x, visible_y), (visible_x2, visible_y2), color, thickness, 10)
            else:
                # Draw solid rectangle for identified drones
                cv.rectangle(
                    img_result,
                    (visible_x, visible_y),
                    (visible_x2, visible_y2),
                    color,
                    thickness
                )
            
            # For partial detections, draw extended lines to show full template bounds
            if match.get('partial', False):
                # Draw dotted lines to show where template extends beyond image
                if x < 0:  # Extends left
                    cv.line(img_result, (0, visible_y), (0, visible_y2), color, 1)
                if y < 0:  # Extends top
                    cv.line(img_result, (visible_x, 0), (visible_x2, 0), color, 1)
                if x + w > img_w:  # Extends right
                    cv.line(img_result, (img_w-1, visible_y), (img_w-1, visible_y2), color, 1)
                if y + h > img_h:  # Extends bottom
                    cv.line(img_result, (visible_x, img_h-1), (visible_x2, img_h-1), color, 1)
        
        # Create simplified label with drone ID only
        label = match['drone_id']
        
        # Position label within image bounds
        label_x = max(5, min(visible_x, img_w - 100))  # Reduced width since labels are shorter
        label_y = max(15, visible_y - 5 if visible_y > 15 else visible_y + 15)  # Smaller offset
        
        # Draw label background (smaller since text is shorter)
        (text_width, text_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)  # Smaller font
        cv.rectangle(img_result, 
                    (label_x - 2, label_y - text_height - 2), 
                    (label_x + text_width + 2, label_y + 2), 
                    (0, 0, 0), -1)
        
        cv.putText(
            img_result,
            label,
            (label_x, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,  # Smaller font size
            (255, 255, 255),  # White text on black background
            1
        )
    
    return img_result

def draw_detection_boxes_no_labels(image, matches, min_conf=0.0):
    """Draw bounding boxes on image for detected patterns without labels"""
    img_result = image.copy()
    
    # Define colors for different templates and detection types
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    template_colors = {}
    color_idx = 0
    
    # Filter matches by confidence first
    valid_matches = [match for match in matches if match['confidence'] >= min_conf]
    
    for match in valid_matches:
        template = match['template_name']
        
        # Assign color to template if not assigned
        if template not in template_colors:
            template_colors[template] = colors[color_idx % len(colors)]
            color_idx += 1
        
        base_color = template_colors[template]
        
        # Special handling for unidentified signals (colored rectangles)
        if template == 'Unidentified Signal':
            # Use red color (BGR values for OpenCV) for unidentified signals with dashed line style
            base_color = (0, 0, 255)  # Red in BGR: Blue=0, Green=0, Red=255
            thickness = 2
        
        # Adjust color and line style based on detection type
        if match.get('partial', False):
            # Use dashed line for partial detections by alternating segments
            color = tuple(int(c * 0.7) for c in base_color)  # Darker color for partial
            thickness = 3
        elif template == 'Unidentified Signal':
            color = base_color
            thickness = 2
        else:
            color = base_color
            thickness = 2
        
        # Calculate visible portion of the template
        img_h, img_w = image.shape[:2]
        x, y, w, h = match['x'], match['y'], match['width'], match['height']
        
        # Clip rectangle to image boundaries
        visible_x = max(0, x)
        visible_y = max(0, y)
        visible_x2 = min(img_w, x + w)
        visible_y2 = min(img_h, y + h)
        
        # Draw the visible portion of the rectangle (NO LABELS)
        if visible_x < visible_x2 and visible_y < visible_y2:
            if template == 'Unidentified Signal':
                # Draw dashed rectangle for unidentified signals
                draw_dashed_rectangle(img_result, (visible_x, visible_y), (visible_x2, visible_y2), color, thickness, 10)
            else:
                # Draw solid rectangle for identified drones
                cv.rectangle(
                    img_result,
                    (visible_x, visible_y),
                    (visible_x2, visible_y2),
                    color,
                    thickness
                )
            
            # For partial detections, draw extended lines to show full template bounds
            if match.get('partial', False):
                # Draw dotted lines to show where template extends beyond image
                if x < 0:  # Extends left
                    cv.line(img_result, (0, visible_y), (0, visible_y2), color, 1)
                if y < 0:  # Extends top
                    cv.line(img_result, (visible_x, 0), (visible_x2, 0), color, 1)
                if x + w > img_w:  # Extends right
                    cv.line(img_result, (img_w-1, visible_y), (img_w-1, visible_y2), color, 1)
                if y + h > img_h:  # Extends bottom
                    cv.line(img_result, (visible_x, img_h-1), (visible_x2, img_h-1), color, 1)
        
        # NO LABEL DRAWING - this is the key difference from draw_detection_boxes()
    
    return img_result

def cv_to_pil(cv_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))


def save_drone_as_template(drone, custom_name=None):
    """Save an unidentified drone as a new template to computer and load it into the system.
    
    Args:
        drone: Drone dictionary from unidentified_drones
        custom_name: Optional custom name for the template
    
    Returns:
        tuple: (success: bool, message: str, filename: str)
    """
    try:
        # Create templates directory if it doesn't exist
        templates_dir = "stored_templates"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
        
        # Generate filename
        if custom_name:
            clean_name = custom_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"{clean_name}.png"
        else:
            filename = f"US_{drone['id']}_{drone['filename'].replace('.', '_')}.png"
        
        filepath = os.path.join(templates_dir, filename)
        
        # Save image to computer
        drone['image'].save(filepath, 'PNG')
        
        # Load as new template
        template_cv = pil_to_cv(drone['image'])
        clean_display_name = custom_name if custom_name else f"Stored US-{drone['id']}"
        
        # Add to current templates and active folder
        template_data = {
            'image': template_cv,
            'pil_image': drone['image'],
            'size': drone['image'].size,
            'clean_name': clean_display_name,
            'original_name': filename,
            'source': 'unidentified_drone',
            'original_drone_id': drone['id'],
            'class_folder': None  # Root level by default
        }
        
        st.session_state.templates[filename] = template_data
        st.session_state.template_folders[st.session_state.active_folder][filename] = template_data
        
        return True, f"Template '{clean_display_name}' saved to {filepath} and loaded into {st.session_state.active_folder} folder", filename
        
    except Exception as e:
        return False, f"Error saving template: {str(e)}", ""


def save_template_to_class_folder(template_image, template_name, class_name=None):
    """Save a template to an organized class folder in stored_templates.
    
    This function is designed for train mode and organized template management.
    Creates a subdirectory structure: stored_templates/<class_name>/<template_name>.png
    
    If class_name is not provided, it will be extracted from template_name (before dash).
    Example: "nowe-A" -> folder "nowe", "unknown-B" -> folder "unknown"
    
    Args:
        template_image: PIL Image or OpenCV image array
        template_name: Base name for the template (without extension)
        class_name: Optional class/drone type folder name. If None, extracted from template_name
    
    Returns:
        tuple: (success: bool, message: str, filepath: str)
    """
    try:
        # Extract class_name from template_name if not provided
        if class_name is None:
            if '-' in template_name:
                class_name = template_name.rsplit('-', 1)[0]  # "nowe-A" -> "nowe"
            else:
                class_name = template_name
        
        # Create class folder structure
        templates_dir = "stored_templates"
        class_dir = os.path.join(templates_dir, class_name)
        
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Clean the template name
        clean_name = template_name.strip().replace('/', '_').replace('\\', '_')
        filename = f"{clean_name}.png"
        filepath = os.path.join(class_dir, filename)
        
        # Convert to PIL if needed
        if isinstance(template_image, np.ndarray):
            template_pil = cv_to_pil(template_image)
        else:
            template_pil = template_image
        
        # Save to disk
        template_pil.save(filepath, 'PNG')
        
        # Load as template (convert to OpenCV)
        template_cv = pil_to_cv(template_pil)
        
        # Create unique key including class folder
        template_key = f"{class_name}/{filename}"
        clean_display_name = f"{class_name}/{clean_name}"
        
        # Add to current templates and active folder
        template_data = {
            'image': template_cv,
            'pil_image': template_pil,
            'size': template_pil.size,
            'clean_name': clean_display_name,
            'original_name': filename,
            'source': 'train_mode',
            'class_folder': class_name
        }
        
        # Initialize session state if needed
        if 'templates' not in st.session_state:
            st.session_state.templates = {}
        
        st.session_state.templates[template_key] = template_data
        
        # Add to template_folders if it exists and active_folder is set
        if hasattr(st.session_state, 'template_folders') and hasattr(st.session_state, 'active_folder'):
            if st.session_state.active_folder in st.session_state.template_folders:
                st.session_state.template_folders[st.session_state.active_folder][template_key] = template_data
        
        return True, f"Template '{clean_display_name}' saved to {filepath}", filepath
        
    except Exception as e:
        return False, f"Error saving template to class folder: {str(e)}", ""

def is_drone_at_boundary(x, y, w, h, img_w, img_h, border_tolerance=5):
    """Check if a drone detection is cut off at image boundaries.
    Returns True if the drone is cut (should not be stored as unidentified).
    Exception: if touching both top AND bottom, it's considered valid.
    """
    touches_left = x <= border_tolerance
    touches_right = x + w >= img_w - border_tolerance
    touches_top = y <= border_tolerance
    touches_bottom = y + h >= img_h - border_tolerance
    
    # Special case: if touching both top and bottom, allow it
    if touches_top and touches_bottom:
        return False
        
    # If touching any other boundary combination, reject it
    return touches_left or touches_right or touches_top or touches_bottom

def is_inside_other_drone(unidentified_match, identified_matches, overlap_threshold=0.95):
    """Check if an unidentified drone is inside an identified drone.
    Returns True if the unidentified drone significantly overlaps with identified ones.
    LENIENT: Only rejects if 95%+ of the unidentified pattern is inside another (not just touching/bordering).
    """
    unidentified_box = (unidentified_match['x'], unidentified_match['y'], 
                       unidentified_match['width'], unidentified_match['height'])
    
    for identified_match in identified_matches:
        if identified_match['template_name'] == 'Unidentified Signal':
            continue  # Skip other unidentified drones
            
        identified_box = (identified_match['x'], identified_match['y'],
                         identified_match['width'], identified_match['height'])
        
        # Check if unidentified drone is mostly contained within identified drone
        overlap_pct = calculate_overlap_percentage(unidentified_box, identified_box)
        if overlap_pct >= overlap_threshold:
            return True
            
    return False

def matches_existing_unidentified(new_match, existing_unidentified, similarity_threshold=0.7):
    """Check if a new unidentified drone matches existing ones.
    Uses position, size, and confidence similarity to determine matches.
    """
    new_box = (new_match['x'], new_match['y'], new_match['width'], new_match['height'])
    new_area = new_match['width'] * new_match['height']
    
    for existing in existing_unidentified:
        existing_box = (existing['x'], existing['y'], existing['width'], existing['height'])
        existing_area = existing['width'] * existing['height']
        
        # Calculate overlap
        overlap_pct = max(
            calculate_overlap_percentage(new_box, existing_box),
            calculate_overlap_percentage(existing_box, new_box)
        )
        
        # Calculate size similarity
        size_ratio = min(new_area, existing_area) / max(new_area, existing_area) if max(new_area, existing_area) > 0 else 0
        
        # Calculate distance between centers
        new_center = (new_match['center_x'], new_match['center_y'])
        existing_center = (existing['center_x'], existing['center_y'])
        distance = np.sqrt((new_center[0] - existing_center[0])**2 + (new_center[1] - existing_center[1])**2)
        
        # Calculate maximum allowed distance based on size
        avg_size = np.sqrt((new_area + existing_area) / 2)
        max_allowed_distance = avg_size * 0.5  # 50% of average size
        
        # Check if they match based on multiple criteria
        if (overlap_pct >= similarity_threshold or 
            (size_ratio >= 0.8 and distance <= max_allowed_distance)):
            return existing
            
    return None

def get_next_available_drone_id():
    """
    Get the next available ID for an unidentified drone.
    Checks existing IDs to avoid collisions when drones have been deleted.
    
    Returns:
        int: Next available ID that doesn't conflict with existing drones
    """
    if not st.session_state.unidentified_drones:
        return 1
    
    # Get all existing IDs
    existing_ids = set(drone['id'] for drone in st.session_state.unidentified_drones)
    
    # Find the first available ID starting from 1
    next_id = 1
    while next_id in existing_ids:
        next_id += 1
    
    return next_id

def add_unidentified_drone(match, image, filename, img_w, img_h, min_confidence=0.0, skip_boundary_check=False, skip_duplicate_check=False):
    """Add an unidentified drone to storage with validation.
    Returns True if successfully added, False if rejected.
    
    Args:
        skip_boundary_check: If True, allows signals at image boundaries (for Train Mode)
        skip_duplicate_check: If True, skips checking against existing unidentified drones (for Train Mode)
    """
    # Check if this match is eligible for storage (not at borders)
    if not match.get('storage_eligible', True):
        return False
        
    # Apply confidence filter
    if match['confidence'] < min_confidence:
        return False
        
    # Check if at boundary (reject if cut, except top+bottom case)
    # Skip this check in Train Mode since spectrograms often have valid signals at edges
    if not skip_boundary_check:
        if is_drone_at_boundary(match['x'], match['y'], match['width'], match['height'], img_w, img_h):
            return False
        
    # Check if inside identified drones (from current detection results)
    current_result = next((r for r in st.session_state.detection_results if r['filename'] == filename), None)
    if current_result:
        identified_matches = [m for m in current_result['matches'] if m['template_name'] != 'Unidentified Signal']
        if is_inside_other_drone(match, identified_matches):
            return False
    
    # Check if it matches existing unidentified drones (skip in Train Mode)
    if not skip_duplicate_check:
        existing_match = matches_existing_unidentified(match, st.session_state.unidentified_drones)
        if existing_match:
            # Update existing entry with better confidence if this one is higher
            if match['confidence'] > existing_match['confidence']:
                existing_match.update(match)
                existing_match['filename'] = filename
                existing_match['source_image'] = image
            return False  # Don't add duplicate
    
    # Extract drone image crop
    img_array = np.array(image)
    
    # If image is grayscale (PIL), convert to colored version for storage
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 3 and 
                                     np.array_equal(img_array[:,:,0], img_array[:,:,1]) and 
                                     np.array_equal(img_array[:,:,1], img_array[:,:,2])):
        # Apply colormap for better visualization
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        colored_array = cv.applyColorMap(gray, cv.COLORMAP_VIRIDIS)
        img_array = cv.cvtColor(colored_array, cv.COLOR_BGR2RGB)
        image = Image.fromarray(img_array)
    
    x, y, w, h = match['x'], match['y'], match['width'], match['height']
    
    # Add padding around the signal for better template context (10 pixels on each side)
    padding = 10
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    w_padded = min(w + 2 * padding, img_w - x_padded)
    h_padded = min(h + 2 * padding, img_h - y_padded)
    
    # Use padded coordinates for extraction
    x, y, w, h = x_padded, y_padded, w_padded, h_padded
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w > 0 and h > 0:
        drone_crop = img_array[y:y+h, x:x+w]
        drone_image = Image.fromarray(drone_crop)
        
        # Create unidentified drone entry with safe ID assignment
        unidentified_entry = {
            'id': get_next_available_drone_id(),
            'image': drone_image,
            'full_image': image,
            'filename': filename,
            'confidence': match['confidence'],
            'x': x, 'y': y, 'width': w, 'height': h,
            'center_x': match['center_x'],
            'center_y': match['center_y'],
            'detection_type': match.get('detection_type', 'colored_rectangle'),
            'area': match.get('area', w * h),
            'timestamp': pd.Timestamp.now()
        }
        
        st.session_state.unidentified_drones.append(unidentified_entry)
        return True
        
    return False

# Removed duplicate assign_drone_ids function - using the one with template_id_mapping for consistency

def extract_drone_base_name(template_name):
    """
    Extract the base drone name from template filename.
    Examples:
    - DroneA-1.png -> DroneA
    - DroneA-2.png -> DroneA  
    - DJI_Inspire_2_RC-1.png -> DJI_Inspire_2_RC
    - Reaper_attack.png -> Reaper
    - Predator_MQ1_variant1.png -> Predator_MQ1
    
    Note: Numeric variants use hyphen (-), descriptive variants use underscore (_)
    """
    import re
    
    # Remove file extension first
    name = template_name.lower()
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        if name.endswith(ext):
            template_name = template_name[:-len(ext)]
            break
    
    # Pattern to match common variant suffixes
    # Matches: -1, -2 (numeric variants with hyphen)
    # Matches: _attack, _fly, _variant1, etc. (descriptive variants with underscore)
    variant_pattern = r'(-\d+|_[a-zA-Z]+\d*|_variant\d*)$'
    
    base_name = re.sub(variant_pattern, '', template_name, flags=re.IGNORECASE)
    
    # Clean up the base name
    base_name = base_name.strip('_-')
    
    return base_name if base_name else template_name

def get_pattern_priority(template_name):
    """
    Assign priority to different pattern types for conflict resolution.
    Lower numbers = higher priority (will be kept in overlaps).
    """
    name_lower = template_name.lower()
    
    # Define priority based on pattern type
    if any(keyword in name_lower for keyword in ['attack', 'combat', 'armed']):
        return 1  # Highest priority - attack patterns
    elif any(keyword in name_lower for keyword in ['fly', 'flight', 'cruise']):
        return 2  # Flight patterns
    elif any(keyword in name_lower for keyword in ['hover', 'stationary']):
        return 3  # Hover patterns
    elif '_1' in name_lower or 'main' in name_lower or 'primary' in name_lower:
        return 2  # Primary variants
    elif '_2' in name_lower or 'secondary' in name_lower:
        return 4  # Secondary variants
    else:
        return 3  # Default priority
        
def consolidate_pattern_variants(matches, overlap_threshold=0.7):
    """
    Consolidate overlapping matches from the same drone base type.
    When multiple variants of the same drone overlap significantly,
    keep only the best-fitting variant.
    
    Args:
        matches: List of detection matches
        overlap_threshold: Minimum overlap to consider consolidation
        
    Returns:
        List of consolidated matches with variants resolved
    """
    if not matches:
        return matches
    
    # Group matches by base drone name
    base_groups = {}
    for match in matches:
        base_name = extract_drone_base_name(match['template_name'])
        if base_name not in base_groups:
            base_groups[base_name] = []
        base_groups[base_name].append(match)
    
    consolidated_matches = []
    
    # Process each base drone group
    for base_name, group_matches in base_groups.items():
        if len(group_matches) == 1:
            # Single variant - no consolidation needed
            consolidated_matches.extend(group_matches)
        else:
            # Multiple variants - check for overlaps and consolidate
            resolved_matches = resolve_variant_overlaps(group_matches, overlap_threshold)
            consolidated_matches.extend(resolved_matches)
    
    return consolidated_matches

def resolve_variant_overlaps(variant_matches, overlap_threshold):
    """
    Resolve overlaps between variants of the same drone type.
    Keep the best-fitting variant for each overlapping region.
    """
    if len(variant_matches) <= 1:
        return variant_matches
    
    # Sort by confidence initially
    variant_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    resolved = []
    processed_indices = set()
    
    for i, match1 in enumerate(variant_matches):
        if i in processed_indices:
            continue
            
        # Find all overlapping variants
        overlapping_variants = [match1]
        overlapping_indices = {i}
        
        for j, match2 in enumerate(variant_matches[i+1:], i+1):
            if j in processed_indices:
                continue
                
            # Calculate overlap between variants
            box1 = (match1['x'], match1['y'], match1['width'], match1['height'])
            box2 = (match2['x'], match2['y'], match2['width'], match2['height'])
            
            overlap_pct = max(
                calculate_overlap_percentage(box1, box2),
                calculate_overlap_percentage(box2, box1)
            )
            
            if overlap_pct >= overlap_threshold:
                overlapping_variants.append(match2)
                overlapping_indices.add(j)
        
        # Mark all overlapping variants as processed
        processed_indices.update(overlapping_indices)
        
        if len(overlapping_variants) == 1:
            # No overlaps - keep the variant
            resolved.append(overlapping_variants[0])
        else:
            # Multiple overlapping variants - select the best one
            best_variant = select_best_pattern_variant(overlapping_variants)
            resolved.append(best_variant)
    
    return resolved

def select_best_pattern_variant(overlapping_variants):
    """
    Select the best variant from a group of overlapping pattern matches.
    Uses confidence, pattern priority, and detection quality metrics.
    """
    if len(overlapping_variants) == 1:
        return overlapping_variants[0]
    
    # Score each variant
    scored_variants = []
    
    for variant in overlapping_variants:
        score = 0
        
        # Confidence score (0-40 points)
        confidence_score = variant['confidence'] * 40
        score += confidence_score
        
        # Pattern priority score (0-20 points, inverted so lower priority number = higher score)
        priority = get_pattern_priority(variant['template_name'])
        priority_score = max(0, 20 - (priority * 4))
        score += priority_score
        
        # Detection type score (0-20 points)
        detection_type = variant.get('detection_type', 'full')
        if detection_type == 'full':
            type_score = 20
        elif 'merged' in detection_type:
            type_score = 15  # Merged detections are good
        elif 'border' in detection_type:
            type_score = 10  # Border detections are okay
        else:
            type_score = 5   # Other types get low score
        score += type_score
        
        # Partial detection penalty (0-10 points)
        if not variant.get('partial', False):
            score += 10  # Bonus for complete detection
        
        # Template name preference (0-10 points)
        template_name = variant['template_name'].lower()
        if '_1' in template_name or 'main' in template_name:
            score += 8  # Prefer primary variants
        elif 'attack' in template_name or 'combat' in template_name:
            score += 10  # Prefer attack patterns
        elif '_2' in template_name:
            score += 5   # Secondary variants get some bonus
        
        scored_variants.append((score, variant))
    
    # Sort by score (highest first) and return the best variant
    scored_variants.sort(key=lambda x: x[0], reverse=True)
    
    best_variant = scored_variants[0][1].copy()
    
    # Add metadata about the consolidation
    variant_names = [v['template_name'] for v in overlapping_variants]
    best_variant['consolidated_variants'] = variant_names
    best_variant['consolidation_reason'] = f'Best of {len(overlapping_variants)} variants'
    best_variant['base_drone_name'] = extract_drone_base_name(best_variant['template_name'])
    
    return best_variant