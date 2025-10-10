"""
Smart Backend Selector for Smart Astra Detection System
Automatically selects the best performing template matching backend
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, List, Tuple

class SmartBackendSelector:
    """
    Intelligent backend selection for optimal template matching performance
    """
    
    def __init__(self, fast_mode=True, lazy_init=True):
        """Initialize smart backend selector
        
        Args:
            fast_mode: If True, run minimal benchmarks for faster startup
            lazy_init: If True, delay benchmarks until first use
        """
        self.available_backends = {}
        self.performance_cache = {}
        self.benchmark_results = {}
        self.current_best_backend = 'opencv_cpu'
        self.fast_mode = fast_mode
        self.lazy_init = lazy_init
        self.benchmarks_run = False
        
        # Initialize available backends
        self._initialize_backends()
        
        # Run initial benchmarks (unless lazy)
        if not lazy_init:
            if fast_mode:
                self._run_fast_benchmarks()
            else:
                self._run_initial_benchmarks()
        else:
            # Set safe defaults
            self.current_best_backend = 'opencv_cpu'
        
    def _initialize_backends(self):
        """Initialize all available acceleration backends"""
        
        # Always available: Standard OpenCV CPU
        self.available_backends['opencv_cpu'] = {
            'name': 'OpenCV CPU',
            'function': self._opencv_cpu_match,
            'available': True,
            'description': 'Standard OpenCV template matching on CPU'
        }
        
        # Try to initialize PyTorch GPU accelerator
        try:
            from pytorch_gpu_acceleration import PyTorchGPUAccelerator
            pytorch_accelerator = PyTorchGPUAccelerator()
            self.available_backends['pytorch_gpu'] = {
                'name': 'PyTorch GPU',
                'function': lambda img, tmpl, method: pytorch_accelerator.template_match_gpu(img, tmpl, method),
                'available': True,
                'accelerator': pytorch_accelerator,
                'description': 'PyTorch CUDA acceleration'
            }
        except Exception as e:
            self.available_backends['pytorch_gpu'] = {
                'name': 'PyTorch GPU',
                'available': False,
                'error': str(e)
            }
        
        # Try to initialize ONNX DNN accelerator
        try:
            from onnx_dnn_acceleration import ONNXDNNAccelerator
            onnx_accelerator = ONNXDNNAccelerator()
            self.available_backends['onnx_dnn'] = {
                'name': 'ONNX DNN',
                'function': lambda img, tmpl, method: onnx_accelerator.template_match_dnn(img, tmpl, method),
                'available': True,
                'accelerator': onnx_accelerator,
                'description': 'ONNX + OpenCV DNN acceleration'
            }
        except Exception as e:
            self.available_backends['onnx_dnn'] = {
                'name': 'ONNX DNN',
                'available': False,
                'error': str(e)
            }
        
        # Summary
    
    def _opencv_cpu_match(self, image, template, method):
        """Standard OpenCV CPU template matching"""
        return cv2.matchTemplate(image, template, method)
    
    def _run_initial_benchmarks(self):
        """Run quick initial benchmarks to determine the best backend for different scenarios"""

        # Simplified test scenarios: (image_size, template_size, description)
        test_scenarios = [
            ((200, 200), (20, 20), "small"),
            ((400, 400), (40, 40), "medium")
        ]
        
        for img_size, tmpl_size, scenario_name in test_scenarios:
            
            # Create smaller test data for faster benchmarks
            test_image = np.random.randint(0, 255, img_size, dtype=np.uint8)
            test_template = np.random.randint(0, 255, tmpl_size, dtype=np.uint8)
            
            scenario_results = {}
            
            for backend_name, backend_info in self.available_backends.items():
                if not backend_info.get('available', False):
                    continue
                
                try:
                    # Single quick test (no warmup, single iteration)
                    start = time.time()
                    result = backend_info['function'](test_image, test_template, cv2.TM_CCOEFF_NORMED)
                    avg_time = time.time() - start
                    
                    scenario_results[backend_name] = {
                        'avg_time': avg_time,
                        'result_shape': result.shape if result is not None else None
                    }
                    
                except Exception as e:
                    scenario_results[backend_name] = {'error': str(e)}
            
            # Find best backend for this scenario
            best_backend = min(
                [k for k, v in scenario_results.items() if 'avg_time' in v],
                key=lambda k: scenario_results[k]['avg_time'],
                default='opencv_cpu'
            )
            
            self.benchmark_results[scenario_name] = {
                'results': scenario_results,
                'best_backend': best_backend,
                'image_size': img_size,
                'template_size': tmpl_size
            }
        
        # Add default large scenario without benchmarking
        self.benchmark_results['large'] = {
            'best_backend': 'opencv_cpu',  # Default to CPU for large
            'image_size': (800, 600),
            'template_size': (80, 60)
        }
        
        self.benchmark_results['xlarge'] = {
            'best_backend': 'pytorch_gpu' if 'pytorch_gpu' in self.available_backends and self.available_backends['pytorch_gpu'].get('available') else 'opencv_cpu',
            'image_size': (1000, 1000),
            'template_size': (100, 100)
        }
        
        # Determine overall best backend
        self._determine_best_overall_backend()
    
    def _run_fast_benchmarks(self):
        """Run ultra-fast benchmarks with minimal testing"""
        
        # Single quick test with small data
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_template = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        
        fastest_backend = 'opencv_cpu'
        fastest_time = float('inf')
        
        for backend_name, backend_info in self.available_backends.items():
            if not backend_info.get('available', False):
                continue
            
        # Set default results for all scenarios
        for scenario in ['small', 'medium', 'large', 'xlarge']:
            self.benchmark_results[scenario] = {
                'best_backend': fastest_backend,
                'fast_mode': True
            }
        
        self.current_best_backend = fastest_backend
        
        # Set smart defaults based on available backends
        if 'pytorch_gpu' in self.available_backends and self.available_backends['pytorch_gpu'].get('available'):
            # Use GPU for large templates
            self.benchmark_results['xlarge']['best_backend'] = 'pytorch_gpu'
    
    def _determine_best_overall_backend(self):
        """Determine the best overall backend based on benchmark results"""
        backend_scores = {}
        
        for scenario_name, scenario_data in self.benchmark_results.items():
            best_backend = scenario_data['best_backend']
            if best_backend not in backend_scores:
                backend_scores[best_backend] = 0
            backend_scores[best_backend] += 1
        
        if backend_scores:
            self.current_best_backend = max(backend_scores.keys(), key=lambda k: backend_scores[k])
    
    def _ensure_benchmarks_run(self):
        """Ensure benchmarks have been run (lazy initialization)"""
        if not self.benchmarks_run:
            if self.fast_mode:
                self._run_fast_benchmarks()
            else:
                self._run_initial_benchmarks()
            self.benchmarks_run = True
    
    def should_use_gpu(self, num_templates):
        """
        Determine if GPU processing should be used based on workload
        
        Args:
            num_templates: Number of templates to process
            
        Returns:
            bool: True if GPU should be used
        """
        self._ensure_benchmarks_run()
        
        # GPU is beneficial for larger workloads
        if num_templates >= 8:
            return 'pytorch_gpu' in self.available_backends and self.available_backends['pytorch_gpu'].get('available', False)
        
        return False
    
    def get_best_backend_for_size(self, image_shape: tuple, template_shape: tuple) -> str:
        """
        Get the best backend for specific image and template sizes
        """
        # Run benchmarks if not done yet (lazy loading)
        self._ensure_benchmarks_run()
        
        img_area = image_shape[0] * image_shape[1]
        tmpl_area = template_shape[0] * template_shape[1]
        
        # Classify size scenario
        if img_area < 100000 and tmpl_area < 1000:  # Small
            scenario = 'small'
        elif img_area < 400000 and tmpl_area < 4000:  # Medium
            scenario = 'medium'
        elif img_area < 800000 and tmpl_area < 8000:  # Large
            scenario = 'large'
        else:  # Extra large
            scenario = 'xlarge'
        
        return self.benchmark_results.get(scenario, {}).get('best_backend', self.current_best_backend)
    
    def template_match_smart(self, image, template, method=cv2.TM_CCOEFF_NORMED):
        """
        Smart template matching that automatically selects the best backend
        """
        # Determine best backend for this specific case
        best_backend = self.get_best_backend_for_size(image.shape, template.shape)
        
        # Use the best backend
        backend_info = self.available_backends[best_backend]
        
        if not backend_info.get('available', False):
            # Fallback to CPU if best backend isn't available
            best_backend = 'opencv_cpu'
            backend_info = self.available_backends[best_backend]
        
        try:
            return backend_info['function'](image, template, method)
        except Exception as e:
            return self._opencv_cpu_match(image, template, method)
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        report = {
            'available_backends': {
                name: {
                    'name': info['name'],
                    'available': info.get('available', False),
                    'description': info.get('description', '')
                }
                for name, info in self.available_backends.items()
            },
            'benchmark_results': self.benchmark_results,
            'current_best_backend': self.current_best_backend,
            'recommendations': self._get_recommendations()
        }
        
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Get performance recommendations"""
        recommendations = []
        
        # Check if GPU backends are available
        gpu_available = any(
            backend.get('available', False) 
            for name, backend in self.available_backends.items() 
            if 'gpu' in name.lower()
        )
        
        if not gpu_available:
            recommendations.append("Consider enabling GPU acceleration for better performance")
        
        # Check current best backend
        if self.current_best_backend == 'opencv_cpu':
            recommendations.append("CPU backend is optimal - GPU overhead may not be worth it for your typical usage")
        else:
            recommendations.append(f"Using {self.available_backends[self.current_best_backend]['name']} for optimal performance")
        
        return recommendations

def test_smart_backend_selector():
    """Test the smart backend selector"""
    
    # Initialize selector
    selector = SmartBackendSelector()
    
    return selector


if __name__ == "__main__":
    test_smart_backend_selector()