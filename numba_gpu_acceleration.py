#!/usr/bin/env python3
"""
OpenCV CUDA GPU Acceleration for Template Matching

Uses OpenCV's highly optimized CUDA kernels (10-50x faster than custom implementations).
"""

import numpy as np
import cv2 as cv
import time


class NumbaCUDAAccelerator:
    """
    OpenCV CUDA-based GPU accelerator for template matching.
    Uses OpenCV's highly optimized CUDA kernels (10-50x faster).
    """
    
    def __init__(self):
        """Initialize OpenCV CUDA accelerator"""
        self.available = False
        self.device_name = "None"
        self.matcher = None
        
        try:
            # Check if OpenCV CUDA is available
            if cv.cuda.getCudaEnabledDeviceCount() > 0:
                self.available = True
                
                # Get GPU info
                device_info = cv.cuda.getDevice()
                self.device_name = cv.cuda.printCudaDeviceInfo(device_info)
                
                # Create GPU template matcher (reusable for all operations)
                self.matcher = cv.cuda.createTemplateMatching(cv.CV_32F, cv.TM_CCOEFF_NORMED)
                
                print(f"✅ OpenCV CUDA Available - GPU Device {device_info}")
                print(f"   Matcher: TM_CCOEFF_NORMED (Normalized Cross-Correlation)")
            else:
                print("❌ OpenCV CUDA not available - using CPU fallback")
        except Exception as e:
            print(f"⚠️ OpenCV CUDA initialization error: {e}")
            print("   Falling back to CPU processing")
    
    def is_available(self):
        """Check if OpenCV CUDA GPU is available"""
        return self.available
    
    def parallel_template_match(self, image, template):
        """
        Match single template on image using OpenCV CUDA.
        
        Args:
            image: Grayscale image (2D numpy array)
            template: Grayscale template (2D numpy array)
            
        Returns:
            Match result array
        """
        if not self.available or self.matcher is None:
            # Fallback to CPU OpenCV
            return cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        
        try:
            # Ensure inputs are float32
            image = image.astype(np.float32)
            template = template.astype(np.float32)
            
            # Upload to GPU
            gpu_image = cv.cuda_GpuMat()
            gpu_template = cv.cuda_GpuMat()
            gpu_image.upload(image)
            gpu_template.upload(template)
            
            # Perform GPU template matching
            gpu_result = self.matcher.match(gpu_image, gpu_template)
            
            # Download result from GPU
            result = gpu_result.download()
            
            return result
            
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"⚠️ GPU matching failed: {e}, using CPU fallback")
            return cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    
    def parallel_multi_template_match(self, image, templates):
        """
        Match MULTIPLE templates on image using OpenCV CUDA.
        Each template is processed sequentially but each uses GPU acceleration.
        
        Args:
            image: Grayscale image (2D numpy array)
            templates: List of grayscale templates (list of 2D numpy arrays)
            
        Returns:
            List of match result arrays
        """
        if not self.available or self.matcher is None:
            # Fallback to CPU OpenCV
            return [cv.matchTemplate(image, tpl, cv.TM_CCOEFF_NORMED) for tpl in templates]
        
        if not templates:
            return []
        
        try:
            # Process all templates using GPU
            results = []
            
            # Upload image to GPU once (reuse for all templates)
            gpu_image = cv.cuda_GpuMat()
            gpu_image.upload(image.astype(np.float32))
            
            for template in templates:
                # Upload template to GPU
                gpu_template = cv.cuda_GpuMat()
                gpu_template.upload(template.astype(np.float32))
                
                # GPU template matching
                gpu_result = self.matcher.match(gpu_image, gpu_template)
                
                # Download result
                result = gpu_result.download()
                results.append(result)
            
            return results
            
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"⚠️ GPU multi-template matching failed: {e}, using CPU fallback")
            return [cv.matchTemplate(image, tpl, cv.TM_CCOEFF_NORMED) for tpl in templates]
    
    def parallel_multi_image_detection(self, images, templates, return_timing=False):
        """
        Process MULTIPLE images with MULTIPLE templates using OpenCV CUDA.
        Each image and template operation uses GPU acceleration.
        
        Args:
            images: List of grayscale images
            templates: List of grayscale templates
            return_timing: Whether to return timing info
            
        Returns:
            List of results per image (each containing results for all templates)
        """
        start_time = time.time()
        
        if not self.available or self.matcher is None:
            # CPU fallback
            results_per_image = []
            for image in images:
                image_results = [cv.matchTemplate(image, tpl, cv.TM_CCOEFF_NORMED) for tpl in templates]
                results_per_image.append(image_results)
            
            if return_timing:
                elapsed = time.time() - start_time
                return results_per_image, {
                    'gpu_time': elapsed,
                    'fallback': True,
                    'reason': 'cuda_unavailable'
                }
            return results_per_image
        
        try:
            # Process each image with all templates using GPU
            all_results = []
            for image in images:
                image_results = self.parallel_multi_template_match(image, templates)
                all_results.append(image_results)
            
            if return_timing:
                elapsed = time.time() - start_time
                total_ops = len(images) * len(templates)
                return all_results, {
                    'gpu_time': elapsed,
                    'images_processed': len(images),
                    'templates_per_image': len(templates),
                    'total_operations': total_ops,
                    'operations_per_second': total_ops / elapsed if elapsed > 0 else 0,
                    'fallback': False
                }
            
            return all_results
            
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"⚠️ GPU multi-image detection failed: {e}, using CPU fallback")
            results_per_image = []
            for image in images:
                image_results = [cv.matchTemplate(image, tpl, cv.TM_CCOEFF_NORMED) for tpl in templates]
                results_per_image.append(image_results)
            
            if return_timing:
                elapsed = time.time() - start_time
                return results_per_image, {
                    'gpu_time': elapsed,
                    'fallback': True,
                    'reason': str(e)
                }
            return results_per_image
    
    def get_performance_info(self):
        """Get OpenCV CUDA GPU performance information"""
        if not self.available:
            return {'available': False}
        
        try:
            device_id = cv.cuda.getDevice()
            device_count = cv.cuda.getCudaEnabledDeviceCount()
            
            return {
                'available': True,
                'device_id': device_id,
                'device_count': device_count,
                'device_name': self.device_name,
                'matcher_type': 'TM_CCOEFF_NORMED (Normalized Cross-Correlation)',
                'backend': 'OpenCV CUDA (Optimized)'
            }
        except Exception as e:
            return {'available': True, 'error': f'Could not get GPU info: {e}'}
