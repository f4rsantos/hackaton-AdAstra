#!/usr/bin/env python3
"""
Parallel GPU Processor

Processes image chunks in parallel using GPU acceleration.
Optimized for maximum throughput when GPU is available.
"""

import cv2 as cv
import numpy as np
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import streamlit as st

from image_preprocessing_pipeline import pil_to_cv, reconstruct_detections_from_chunks
from numba_gpu_acceleration import NumbaCUDAAccelerator


class ParallelGPUProcessor:
    """
    Process image chunks in parallel using GPU acceleration
    """
    
    def __init__(self, max_workers=4):
        """
        Initialize parallel GPU processor
        
        Args:
            max_workers: Maximum number of parallel workers (default: 4)
        """
        self.max_workers = max_workers
        self.gpu_accelerator = NumbaCUDAAccelerator()
        self.gpu_available = self.gpu_accelerator.is_available()
    
    def process_chunks_parallel(self, preprocessed_data: Dict, detection_params: Dict, 
                               status_callback=None) -> Dict:
        """
        Process all chunks in parallel using GPU
        
        Args:
            preprocessed_data: Dictionary containing preprocessed image data with chunks
            detection_params: Detection parameters (threshold, confidence, etc.)
            status_callback: Optional callback for status updates
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        filename = preprocessed_data['original_filename']
        chunks = preprocessed_data['chunks']
        chunk_info = preprocessed_data['chunk_info']
        
        if status_callback:
            status_callback(f"⚡ GPU Parallel: Processing {chunk_info['total_chunks']} chunks for {filename}")
        
        # Convert all chunks to OpenCV format
        chunk_cv_images = []
        for chunk_data in chunks:
            chunk_pil = chunk_data['chunk']
            chunk_cv = pil_to_cv(chunk_pil)
            chunk_cv_images.append(chunk_cv)
        
        # Process chunks in parallel
        chunk_results = []
        
        if st.session_state.templates:
            # GPU-accelerated template matching with parallel processing
            chunk_results = self._parallel_template_matching(
                chunk_cv_images, chunks, detection_params, status_callback
            )
        
        # Detect colored rectangles in parallel if enabled
        if detection_params.get('detect_green_rectangles', False):
            colored_results = self._parallel_colored_detection(
                chunk_cv_images, chunks, detection_params, status_callback
            )
            
            # Merge colored detections with template detections
            for i, colored_matches in enumerate(colored_results):
                if i < len(chunk_results):
                    chunk_results[i]['matches'].extend(colored_matches.get('matches', []))
                else:
                    chunk_results.append(colored_matches)
        
        # Reconstruct full-image coordinates
        if status_callback:
            status_callback(f"🔧 Reconstructing coordinates from {len(chunk_results)} chunks")
        
        all_detections = reconstruct_detections_from_chunks(chunk_results, chunks)
        
        processing_time = time.time() - start_time
        
        result = {
            'filename': filename,
            'image': preprocessed_data['resized_image'],
            'matches': all_detections,
            'chunk_info': chunk_info,
            'processing_mode': 'parallel_gpu',
            'processing_time': processing_time,
            'gpu_available': self.gpu_available
        }
        
        if status_callback:
            status_callback(f"✅ GPU Parallel complete: {len(all_detections)} detections in {processing_time:.2f}s")
        
        return result
    
    def _parallel_multi_chunk_matching(self, chunk_cv_images: List[np.ndarray],
                                      chunks: List[Dict], detection_params: Dict,
                                   status_callback=None) -> List[Dict]:
        """
        OPTIMIZED: Process chunks in efficient batches
        - Try GPU first (Numba CUDA)
        - If GPU not available or fails, use efficient CPU ThreadPool
        - Process in batches of 8 to avoid memory issues
        
        Args:
            chunk_cv_images: List of ALL chunk images
            chunks: Chunk metadata
            detection_params: Detection parameters
            status_callback: Optional callback
            
        Returns:
            List of detection results per chunk
        """
        from functions import (
            preprocess_for_matching, clean_template_name
        )
        
        # Prepare all templates
        # NOTE: We try GPU processing regardless of availability flag
        # The GPU accelerator will handle fallback internally if needed
        template_list = []
        template_names = []
        template_sizes = []
        
        for template_name, template_data in st.session_state.templates.items():
            # Get template image - could be PIL or already CV format
            template_img = template_data.get('pil_image', template_data.get('image'))
            
            # Convert to CV format if needed
            if isinstance(template_img, Image.Image):
                template_cv = pil_to_cv(template_img)
            else:
                template_cv = template_img
            
            template_gray = cv.cvtColor(template_cv, cv.COLOR_BGR2GRAY)
            template_gray = preprocess_for_matching(template_gray)
            
            template_list.append(template_gray)
            template_names.append(template_data.get('clean_name', clean_template_name(template_name)))
            template_sizes.append(template_gray.shape)
        
        # Convert all chunks to grayscale
        chunk_gray_images = []
        for chunk_img in chunk_cv_images:
            if len(chunk_img.shape) == 3:
                chunk_gray = cv.cvtColor(chunk_img, cv.COLOR_BGR2GRAY)
            else:
                chunk_gray = chunk_img.copy()
            chunk_gray = preprocess_for_matching(chunk_gray)
            chunk_gray_images.append(chunk_gray)
        
        # SMART BATCHING: Process in batches of 3 to avoid memory overflow
        batch_size = 3
        all_results = []
        
        for batch_start in range(0, len(chunk_gray_images), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_gray_images))
            batch_chunks = chunk_gray_images[batch_start:batch_end]
            
            if status_callback:
                status_callback(f"⚡ GPU: Processing batch {batch_start//batch_size + 1} ({batch_end - batch_start} chunks × {len(template_list)} templates)")
            
            try:
                # Try GPU processing for this batch
                batch_results, timing = self.gpu_accelerator.parallel_multi_image_detection(
                    batch_chunks,
                    template_list,
                    return_timing=True
                )
                
                fallback = timing.get('fallback', False)
                if fallback:
                    if status_callback:
                        reason = timing.get('reason', 'unknown')
                        status_callback(f"⚠️ GPU fallback to CPU for batch {batch_start//batch_size + 1}: {reason}")
                else:
                    if status_callback:
                        ops_per_sec = timing.get('operations_per_second', 0)
                        status_callback(f"✅ GPU Batch {batch_start//batch_size + 1}: {ops_per_sec:.0f} ops/sec")
                
                all_results.extend(batch_results)
                
            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ GPU batch {batch_start//batch_size + 1} failed: {str(e)[:50]} - using CPU ThreadPool")
                # Fallback to CPU ThreadPool for this batch
                batch_cv_chunks = chunk_cv_images[batch_start:batch_end]
                batch_chunk_meta = chunks[batch_start:batch_end]
                cpu_results = self._parallel_template_matching(batch_cv_chunks, batch_chunk_meta, detection_params, None)
                all_results.extend([{'matches': r.get('matches', [])} for r in cpu_results])
        
        # Process results into chunk_results format
        chunk_results = []
        
        for chunk_idx, (chunk_template_results, chunk_data) in enumerate(zip(all_results, chunks)):
            chunk_matches = []
            
            # Handle case where chunk_template_results might be a dict from CPU fallback
            if isinstance(chunk_template_results, dict):
                chunk_matches = chunk_template_results.get('matches', [])
            else:
                # Process GPU results
                for template_idx, (result, template_name, (h, w)) in enumerate(zip(chunk_template_results, template_names, template_sizes)):
                    # Find matches above threshold
                    threshold = detection_params['threshold']
                    loc = np.where(result >= threshold)
                    
                    if len(loc[0]) > 0:
                        confidences = result[loc]
                        
                        for i, (x, y) in enumerate(zip(loc[1], loc[0])):
                            confidence = float(confidences[i])
                            
                            if confidence >= detection_params['min_confidence']:
                                chunk_matches.append({
                                    'template_name': template_name,
                                    'confidence': confidence,
                                    'x': int(x),
                                    'y': int(y),
                                    'width': int(w),
                                    'height': int(h),
                                    'center_x': int(x + w/2),
                                    'center_y': int(y + h/2),
                                    'partial': False,
                                    'detection_type': 'gpu_cuda' if self.gpu_available else 'cpu_parallel',
                                    'chunk_index': chunk_idx
                                })
            
            chunk_results.append({'matches': chunk_matches})
        
        return chunk_results
    
    def _parallel_template_matching(self, chunk_cv_images: List[np.ndarray], 
                                   chunks: List[Dict], detection_params: Dict,
                                   status_callback=None) -> List[Dict]:
        """
        Run template matching on all chunks in parallel using GPU
        
        Args:
            chunk_cv_images: List of OpenCV chunk images
            chunks: Chunk metadata
            detection_params: Detection parameters
            status_callback: Optional callback
            
        Returns:
            List of detection results per chunk
        """
        from functions import (
            detect_pattern_adaptive, preprocess_for_matching, 
            clean_template_name, remove_overlapping_detections,
            merge_colored_rectangles, consolidate_pattern_variants
        )
        
        chunk_results = [{'matches': []} for _ in range(len(chunk_cv_images))]
        
        # Check if templates exist and are not empty
        has_templates = hasattr(st.session_state, 'templates') and st.session_state.templates
        
        if not self.gpu_available or not has_templates:
            # Fallback to sequential processing
            if status_callback:
                status_callback("⚠️ GPU not available, using CPU fallback")
            return self._sequential_fallback(chunk_cv_images, detection_params)
        
        # Prepare templates
        template_list = []
        template_names = []
        
        for template_name, template_data in st.session_state.templates.items():
            # Get template image - could be PIL or already CV format
            template_img = template_data.get('pil_image', template_data.get('image'))
            
            # Convert to CV format if needed
            if isinstance(template_img, Image.Image):
                template_cv = pil_to_cv(template_img)
            else:
                # Already in CV/numpy format
                template_cv = template_img
            
            template_gray = cv.cvtColor(template_cv, cv.COLOR_BGR2GRAY)
            template_gray = preprocess_for_matching(template_gray)
            
            template_list.append(template_gray)
            template_names.append(template_data.get('clean_name', clean_template_name(template_name)))
        
        # Process each chunk with all templates in parallel
        def process_single_chunk(chunk_idx, chunk_img):
            """Process one chunk with all templates using GPU"""
            chunk_gray = cv.cvtColor(chunk_img, cv.COLOR_BGR2GRAY)
            chunk_gray = preprocess_for_matching(chunk_gray)
            
            chunk_matches = []
            
            try:
                # Use GPU accelerator for parallel template matching
                match_results, timing_info = self.gpu_accelerator.parallel_template_match(
                    chunk_gray, template_list, method=cv.TM_CCOEFF_NORMED, return_timing=True
                )
                
                # Process results for each template
                for template_idx, (result, template_name) in enumerate(zip(match_results, template_names)):
                    # Find matches above threshold
                    threshold = detection_params['threshold']
                    loc = np.where(result >= threshold)
                    
                    if len(loc[0]) > 0:
                        confidences = result[loc]
                        template_img = template_list[template_idx]
                        h, w = template_img.shape
                        
                        for i, (x, y) in enumerate(zip(loc[1], loc[0])):
                            confidence = float(confidences[i])
                            
                            if confidence >= detection_params['min_confidence']:
                                chunk_matches.append({
                                    'template_name': template_name,
                                    'confidence': confidence,
                                    'x': int(x),
                                    'y': int(y),
                                    'width': int(w),
                                    'height': int(h),
                                    'center_x': int(x + w/2),
                                    'center_y': int(y + h/2),
                                    'partial': False,
                                    'detection_type': 'gpu_parallel',
                                    'chunk_index': chunk_idx
                                })
                
                return chunk_idx, chunk_matches, timing_info
                
            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ GPU error on chunk {chunk_idx}: {str(e)}")
                return chunk_idx, [], {'fallback': True, 'error': str(e)}
        
        # Process all chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single_chunk, idx, chunk_img): idx 
                for idx, chunk_img in enumerate(chunk_cv_images)
            }
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                chunk_idx, matches, timing_info = future.result()
                chunk_results[chunk_idx] = {'matches': matches}
                
                completed += 1
                if status_callback and completed % 5 == 0:
                    status_callback(f"⚡ GPU processed {completed}/{total} chunks")
        
        return chunk_results
    
    def _parallel_colored_detection(self, chunk_cv_images: List[np.ndarray],
                                   chunks: List[Dict], detection_params: Dict,
                                   status_callback=None) -> List[Dict]:
        """
        Detect colored rectangles in all chunks in parallel
        
        Args:
            chunk_cv_images: List of OpenCV chunk images
            chunks: Chunk metadata
            detection_params: Detection parameters
            status_callback: Optional callback
            
        Returns:
            List of colored detection results per chunk
        """
        from functions import detect_colored_rectangles, merge_colored_rectangles
        
        def detect_colored_in_chunk(chunk_idx, chunk_img):
            """Detect colored rectangles in a single chunk"""
            try:
                colored_matches = detect_colored_rectangles(
                    chunk_img, 
                    min_area=detection_params.get('green_min_area', 500)
                )
                
                # Merge overlapping colored rectangles
                if colored_matches:
                    colored_matches = merge_colored_rectangles(
                        colored_matches,
                        merge_threshold=detection_params.get('colored_merge_threshold', 0.3)
                    )
                
                return chunk_idx, colored_matches
                
            except Exception as e:
                if status_callback:
                    status_callback(f"⚠️ Colored detection error on chunk {chunk_idx}: {str(e)}")
                return chunk_idx, []
        
        # Process in parallel
        chunk_colored_results = [{'matches': []} for _ in range(len(chunk_cv_images))]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(detect_colored_in_chunk, idx, chunk_img): idx 
                for idx, chunk_img in enumerate(chunk_cv_images)
            }
            
            for future in as_completed(futures):
                chunk_idx, colored_matches = future.result()
                chunk_colored_results[chunk_idx] = {'matches': colored_matches}
        
        return chunk_colored_results
    
    def _sequential_fallback(self, chunk_cv_images: List[np.ndarray], 
                            detection_params: Dict) -> List[Dict]:
        """
        Fallback to sequential CPU processing if GPU fails
        
        Args:
            chunk_cv_images: List of OpenCV chunk images
            detection_params: Detection parameters
            
        Returns:
            List of detection results per chunk
        """
        from functions import detect_pattern, clean_template_name
        
        chunk_results = []
        
        for chunk_idx, chunk_img in enumerate(chunk_cv_images):
            chunk_matches = []
            
            for template_name, template_data in st.session_state.templates.items():
                # Get template image - could be PIL or already CV format
                template_img = template_data.get('pil_image', template_data.get('image'))
                
                # Convert to CV format if needed
                if isinstance(template_img, Image.Image):
                    template_cv = pil_to_cv(template_img)
                else:
                    # Already in CV/numpy format
                    template_cv = template_img
                
                clean_name = template_data.get('clean_name', clean_template_name(template_name))
                
                try:
                    matches = detect_pattern(
                        chunk_img, template_cv,
                        cv.TM_CCOEFF_NORMED,
                        detection_params['threshold'],
                        clean_name,
                        detection_params.get('border_threshold', 0.5),
                        detection_params.get('enable_border_detection', True),
                        detection_params.get('merge_overlapping', True),
                        detection_params.get('overlap_sensitivity', 0.3)
                    )
                    
                    filtered_matches = [m for m in matches if m['confidence'] >= detection_params['min_confidence']]
                    chunk_matches.extend(filtered_matches)
                    
                except Exception as e:
                    continue
            
            chunk_results.append({'matches': chunk_matches})
        
        return chunk_results
    
    def process_chunks_as_individual_results(self, preprocessed_data: Dict, detection_params: Dict,
                                            status_callback=None) -> List[Dict]:
        """
        Process chunks and return each chunk as an individual result (for separate display)
        
        OPTIMIZED: Uses parallel_multi_image_detection to process ALL chunks at once
        instead of sequentially processing each chunk
        
        Args:
            preprocessed_data: Dictionary containing preprocessed image data with chunks
            detection_params: Detection parameters (threshold, confidence, etc.)
            status_callback: Optional callback for status updates
            
        Returns:
            List of individual chunk results, one per chunk
        """
        start_time = time.time()
        
        filename = preprocessed_data['original_filename']
        chunks = preprocessed_data['chunks']
        chunk_info = preprocessed_data['chunk_info']
        
        if status_callback:
            status_callback(f"⚡ GPU Parallel: Processing ALL {chunk_info['total_chunks']} chunks simultaneously for {filename}")
        
        # Convert all chunks to OpenCV format AT ONCE
        chunk_cv_images = []
        for chunk_data in chunks:
            chunk_pil = chunk_data['chunk']
            chunk_cv = pil_to_cv(chunk_pil)
            chunk_cv_images.append(chunk_cv)
        
        # Initialize chunk_results with empty matches for each chunk
        chunk_results = [{'matches': []} for _ in range(len(chunk_cv_images))]
        
        # Check if templates exist and are not empty
        has_templates = hasattr(st.session_state, 'templates') and st.session_state.templates
        
        if has_templates:
            # OPTIMIZED: Use parallel_multi_image_detection to process ALL chunks at once
            # This is MUCH faster than processing each chunk individually
            template_results = self._parallel_multi_chunk_matching(
                chunk_cv_images, chunks, detection_params, status_callback
            )
            # Update chunk_results with template matches
            for i, template_result in enumerate(template_results):
                if i < len(chunk_results):
                    chunk_results[i]['matches'].extend(template_result.get('matches', []))
        
        # Detect colored rectangles in parallel if enabled
        if detection_params.get('detect_green_rectangles', False):
            colored_results = self._parallel_colored_detection(
                chunk_cv_images, chunks, detection_params, status_callback
            )
            
            # Merge colored detections with existing matches
            for i, colored_matches in enumerate(colored_results):
                if i < len(chunk_results):
                    chunk_results[i]['matches'].extend(colored_matches.get('matches', []))
        
        # Create individual results for each chunk
        individual_results = []
        processing_time = time.time() - start_time
        chunk_processing_time = processing_time / len(chunks) if chunks else 0
        
        for chunk_idx, (chunk_data, chunk_det) in enumerate(zip(chunks, chunk_results)):
            chunk_result = {
                'filename': f"{filename}_chunk_{chunk_idx+1}",
                'original_filename': filename,
                'image': chunk_data['chunk'],  # The actual chunk image
                'matches': chunk_det.get('matches', []),
                'chunk_index': chunk_idx,
                'chunk_info': {
                    'chunk_number': chunk_idx + 1,
                    'total_chunks': len(chunks),
                    'x_offset': chunk_data['x_offset'],
                    'y_offset': chunk_data['y_offset'],
                    'width': chunk_data['width'],
                    'height': chunk_data['height'],
                    'is_partial': chunk_data['width'] < self.chunk_width if hasattr(self, 'chunk_width') else False
                },
                'processing_mode': 'parallel_gpu',
                'processing_time': chunk_processing_time,
                'gpu_available': self.gpu_available
            }
            individual_results.append(chunk_result)
        
        if status_callback:
            status_callback(f"✅ GPU Parallel complete: {len(individual_results)} chunks processed")
        
        return individual_results
    
    def get_gpu_info(self) -> Dict:
        """Get GPU performance information"""
        if self.gpu_available:
            return self.gpu_accelerator.get_performance_info()
        return {'available': False, 'message': 'GPU not available'}
