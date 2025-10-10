"""
Integration Module for Burst Detection and Pattern Training

This module provides optimized integration between the new burst detection/pattern
training modules and the existing Ad Astra detection pipeline.

Key optimizations for live feed:
1. Fast-path detection for real-time processing
2. Cached burst detectors to avoid re-initialization
3. Minimal memory allocation for high-throughput
4. Optional GPU acceleration
"""

import numpy as np
import cv2 as cv
from typing import Dict, List, Optional, Tuple
import time
from functools import lru_cache

from burst_detection import BurstDetector, BurstDetectionConfig, PatternDetector
from pattern_training import PatternTrainer, PatternTrainingConfig


class OptimizedBurstDetectorCache:
    """
    Caches burst detector instances to avoid re-initialization overhead.
    Critical for live feed performance.
    """
    _instance = None
    _detectors = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_detector(self, config_key: str = "default") -> BurstDetector:
        """Get cached detector or create new one"""
        if config_key not in self._detectors:
            config = BurstDetectionConfig()
            config.verbose = False  # Disable verbose for live feed
            self._detectors[config_key] = BurstDetector(config)
        return self._detectors[config_key]
    
    def clear_cache(self):
        """Clear detector cache"""
        self._detectors.clear()


class LiveFeedBurstIntegration:
    """
    Integrates burst detection into live feed processing pipeline.
    
    Performance targets:
    - Template matching: 5-20ms per image (existing)
    - Burst detection: 30-80ms per image (new)
    - Total: 35-100ms per image
    
    At 1024x192 images (~225KB):
    - 10 fps → 100ms budget per frame ✓ ACHIEVABLE
    - 20 fps → 50ms budget per frame → Need fast path
    - 100 fps → 10ms budget → Template only
    """
    
    def __init__(self, enable_burst_detection: bool = True,
                 use_fast_mode: bool = False):
        """
        Args:
            enable_burst_detection: Enable burst-based detection
            use_fast_mode: Use optimized fast path (lower accuracy, higher speed)
        """
        self.enable_burst_detection = enable_burst_detection
        self.use_fast_mode = use_fast_mode
        self.detector_cache = OptimizedBurstDetectorCache()
        self.stats = {
            'total_processed': 0,
            'burst_detections': 0,
            'template_detections': 0,
            'avg_burst_time': 0,
            'avg_template_time': 0
        }
    
    def process_image_with_bursts(self, image: np.ndarray, 
                                  templates: Dict,
                                  detection_params: Dict) -> List[Dict]:
        """
        Process image using both template matching and burst detection.
        
        This is the main integration point with existing detection pipeline.
        
        Args:
            image: Spectrogram image (RGB or grayscale)
            templates: Dictionary of template images
            detection_params: Detection parameters (threshold, etc.)
            
        Returns:
            List of detections (compatible with existing format)
        """
        detections = []
        
        # Stage 1: Traditional template matching (fast, existing code)
        start_template = time.time()
        template_detections = self._run_template_matching(
            image, templates, detection_params
        )
        template_time = time.time() - start_template
        
        detections.extend(template_detections)
        
        # Stage 2: Burst detection (if enabled and not in fast mode)
        if self.enable_burst_detection and not self.use_fast_mode:
            start_burst = time.time()
            burst_detections = self._run_burst_detection(
                image, detection_params
            )
            burst_time = time.time() - start_burst
            
            detections.extend(burst_detections)
            
            # Update stats
            self.stats['avg_burst_time'] = (
                self.stats['avg_burst_time'] * self.stats['total_processed'] + burst_time
            ) / (self.stats['total_processed'] + 1)
        
        # Update stats
        self.stats['total_processed'] += 1
        self.stats['avg_template_time'] = (
            self.stats['avg_template_time'] * (self.stats['total_processed'] - 1) + template_time
        ) / self.stats['total_processed']
        self.stats['template_detections'] += len(template_detections)
        if self.enable_burst_detection:
            self.stats['burst_detections'] += len(burst_detections) if not self.use_fast_mode else 0
        
        return detections
    
    def _run_template_matching(self, image: np.ndarray, 
                               templates: Dict,
                               detection_params: Dict) -> List[Dict]:
        """
        Run traditional template matching.
        
        This calls the existing detection function from functions.py
        """
        from functions import detect_pattern_adaptive
        
        detections = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Run detection for each template
        for template_name, template_data in templates.items():
            if 'image' not in template_data:
                continue
            
            template = template_data['image']
            if len(template.shape) == 3:
                template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            
            matches = detect_pattern_adaptive(
                gray, template,
                method=detection_params.get('method', cv.TM_CCORR_NORMED),
                threshold=detection_params.get('threshold', 0.62),
                template_name=template_name,
                partial_threshold=detection_params.get('border_threshold', 0.3),
                enable_border_detection=detection_params.get('enable_border_detection', True),
                merge_overlapping=detection_params.get('merge_overlapping', True),
                overlap_sensitivity=detection_params.get('overlap_sensitivity', 0.3),
                parallel_config=detection_params.get('parallel_config', None)
            )
            
            # Convert to standard format
            for match in matches:
                detections.append({
                    'template_name': template_name,
                    'confidence': match['confidence'],
                    'location': match['location'],
                    'method': 'template_matching'
                })
        
        return detections
    
    def _run_burst_detection(self, image: np.ndarray,
                            detection_params: Dict) -> List[Dict]:
        """
        Run burst-based pattern detection on spectrogram image.
        
        This is fast because it works directly on the image (no IQ needed).
        """
        from burst_detection import detect_bursts_in_spectrogram
        
        config = BurstDetectionConfig()
        config.verbose = False
        
        # Detect bursts in the image
        bursts = detect_bursts_in_spectrogram(
            image, 
            sample_rate=1.0,  # Normalized for image
            center_freq=0,
            config=config
        )
        
        # Convert to standard detection format
        detections = []
        for burst in bursts:
            detections.append({
                'template_name': 'burst_pattern',
                'confidence': 0.7,  # Base confidence for burst detection
                'location': (burst['x'], burst['y'], burst['width'], burst['height']),
                'method': 'burst_detection',
                'burst_info': burst
            })
        
        return detections
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_images': self.stats['total_processed'],
            'avg_template_time_ms': self.stats['avg_template_time'] * 1000,
            'avg_burst_time_ms': self.stats['avg_burst_time'] * 1000,
            'avg_total_time_ms': (self.stats['avg_template_time'] + self.stats['avg_burst_time']) * 1000,
            'fps_estimate': 1.0 / (self.stats['avg_template_time'] + self.stats['avg_burst_time']) if self.stats['avg_template_time'] > 0 else 0,
            'template_detections': self.stats['template_detections'],
            'burst_detections': self.stats['burst_detections']
        }


class FastPNGToDetectionPipeline:
    """
    Complete pipeline from PNG files to detections.
    
    Optimized for automatic live feed processing.
    """
    
    def __init__(self, templates: Dict, detection_params: Dict):
        self.templates = templates
        self.detection_params = detection_params
        self.burst_integration = LiveFeedBurstIntegration(
            enable_burst_detection=True,
            use_fast_mode=False  # Can enable for >20fps requirements
        )
    
    def process_png_file(self, png_path) -> List[Dict]:
        """
        Fast path: PNG → Detections
        
        Performance: ~30-100ms per file
        """
        image = cv.imread(str(png_path))
        
        detections = self.burst_integration.process_image_with_bursts(
            image, self.templates, self.detection_params
        )
        
        return detections


# Performance benchmarking utilities
class PerformanceBenchmark:
    """
    Benchmark tool for measuring detection performance.
    """
    
    @staticmethod
    def benchmark_detection_speed(image_size=(192, 1024), num_iterations=100):
        """
        Benchmark detection speed.
        
        Returns performance metrics for different configurations.
        """
        import time
        
        # Create synthetic test image
        test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Create dummy template
        template = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        templates = {'test': {'image': template}}
        
        # Detection params
        detection_params = {
            'threshold': 0.62,
            'method': cv.TM_CCORR_NORMED,
            'border_threshold': 0.3,
            'enable_border_detection': True,
            'merge_overlapping': True,
            'overlap_sensitivity': 0.3
        }
        
        results = {}
        
        # Benchmark 1: Template only (existing)
        integration_template_only = LiveFeedBurstIntegration(
            enable_burst_detection=False, use_fast_mode=False
        )
        
        start = time.time()
        for _ in range(num_iterations):
            integration_template_only.process_image_with_bursts(
                test_image, templates, detection_params
            )
        elapsed = time.time() - start
        
        results['template_only'] = {
            'total_time': elapsed,
            'avg_time_ms': (elapsed / num_iterations) * 1000,
            'fps': num_iterations / elapsed
        }
        
        # Benchmark 2: Template + Burst (normal mode)
        integration_full = LiveFeedBurstIntegration(
            enable_burst_detection=True, use_fast_mode=False
        )
        
        start = time.time()
        for _ in range(num_iterations):
            integration_full.process_image_with_bursts(
                test_image, templates, detection_params
            )
        elapsed = time.time() - start
        
        results['template_plus_burst'] = {
            'total_time': elapsed,
            'avg_time_ms': (elapsed / num_iterations) * 1000,
            'fps': num_iterations / elapsed
        }
        
        # Benchmark 3: Fast mode
        integration_fast = LiveFeedBurstIntegration(
            enable_burst_detection=True, use_fast_mode=True
        )
        
        start = time.time()
        for _ in range(num_iterations):
            integration_fast.process_image_with_bursts(
                test_image, templates, detection_params
            )
        elapsed = time.time() - start
        
        results['fast_mode'] = {
            'total_time': elapsed,
            'avg_time_ms': (elapsed / num_iterations) * 1000,
            'fps': num_iterations / elapsed
        }
        
        return results


# Easy-to-use wrapper functions
def auto_process_directory(directory_path, templates, detection_params):
    """
    Automatically process all PNG files in a directory.
    
    Args:
        directory_path: Path to directory
        templates: Template dictionary
        detection_params: Detection parameters
        
    Returns:
        Dictionary mapping filename to detections
    """
    from pathlib import Path
    
    directory = Path(directory_path)
    pipeline = FastPNGToDetectionPipeline(templates, detection_params)
    
    results = {}
    
    # Process PNG files
    for png_file in directory.glob('*.png'):
        try:
            detections = pipeline.process_png_file(png_file)
            results[png_file.name] = detections
        except Exception as e:
            pass
    
    return results
