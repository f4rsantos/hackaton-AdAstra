#!/usr/bin/env python3
"""
Sequential CPU Processor

Processes image chunks sequentially using CPU.
Reliable fallback when GPU is not available or for smaller workloads.
"""

import cv2 as cv
import numpy as np
from typing import List, Dict
import time
from PIL import Image
import streamlit as st

from image_preprocessing_pipeline import pil_to_cv, reconstruct_detections_from_chunks


class SequentialCPUProcessor:
    """
    Process image chunks sequentially using CPU
    """
    
    def __init__(self):
        """Initialize sequential CPU processor"""
        pass
    
    def process_chunks_sequential(self, preprocessed_data: Dict, detection_params: Dict,
                                 status_callback=None) -> Dict:
        """
        Process all chunks sequentially using CPU
        
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
            status_callback(f"🖥️ CPU Sequential: Processing {chunk_info['total_chunks']} chunks for {filename}")
        
        # Convert all chunks to OpenCV format
        chunk_cv_images = []
        for chunk_data in chunks:
            chunk_pil = chunk_data['chunk']
            chunk_cv = pil_to_cv(chunk_pil)
            chunk_cv_images.append(chunk_cv)
        
        # Process chunks sequentially
        chunk_results = []
        
        total_chunks = len(chunk_cv_images)
        
        for chunk_idx, chunk_cv in enumerate(chunk_cv_images):
            if status_callback and chunk_idx % 5 == 0:
                status_callback(f"🖥️ CPU processing chunk {chunk_idx+1}/{total_chunks}")
            
            chunk_matches = []
            
            # Template matching
            if st.session_state.templates:
                template_matches = self._sequential_template_matching(
                    chunk_cv, chunk_idx, detection_params
                )
                chunk_matches.extend(template_matches)
            
            # Colored rectangle detection
            if detection_params.get('detect_green_rectangles', False):
                colored_matches = self._detect_colored_in_chunk(
                    chunk_cv, chunk_idx, detection_params
                )
                chunk_matches.extend(colored_matches)
            
            chunk_results.append({'matches': chunk_matches})
        
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
            'processing_mode': 'sequential_cpu',
            'processing_time': processing_time,
            'gpu_available': False
        }
        
        if status_callback:
            status_callback(f"✅ CPU Sequential complete: {len(all_detections)} detections in {processing_time:.2f}s")
        
        return result
    
    def _sequential_template_matching(self, chunk_img: np.ndarray, chunk_idx: int,
                                     detection_params: Dict) -> List[Dict]:
        """
        Run template matching on a single chunk using simple_template_matcher for speed
        
        Args:
            chunk_img: OpenCV chunk image
            chunk_idx: Index of the chunk
            detection_params: Detection parameters
            
        Returns:
            List of detection matches for this chunk
        """
        # Use the optimized simple_template_matcher
        from simple_template_matcher import match_all_templates
        
        chunk_matches = []
        
        try:
            # Use simple matcher - it's faster and applies VIRIDIS colormap correctly
            matches = match_all_templates(
                image=chunk_img,
                threshold=detection_params['threshold'],
                min_confidence=detection_params['min_confidence'],
                apply_nms=False  # We'll do NMS later at full image level
            )
            
            # Add chunk index to all matches
            for match in matches:
                match['chunk_index'] = chunk_idx
                match['detection_type'] = 'cpu_sequential_fast'
                chunk_matches.append(match)
                
        except Exception as e:
            # Fallback to old method if simple matcher fails
            pass
        
        return chunk_matches
    
    def _detect_colored_in_chunk(self, chunk_img: np.ndarray, chunk_idx: int,
                                detection_params: Dict) -> List[Dict]:
        """
        Detect colored rectangles in a single chunk
        
        Args:
            chunk_img: OpenCV chunk image
            chunk_idx: Index of the chunk
            detection_params: Detection parameters
            
        Returns:
            List of colored detection matches for this chunk
        """
        from functions import detect_colored_rectangles, merge_colored_rectangles
        
        try:
            # Detect colored rectangles
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
            
            # Add chunk index to matches
            for match in colored_matches:
                match['chunk_index'] = chunk_idx
            
            return colored_matches
            
        except Exception as e:
            return []
    
    def process_chunks_with_merging(self, preprocessed_data: Dict, detection_params: Dict,
                                   status_callback=None) -> Dict:
        """
        Process chunks sequentially with overlap-aware merging
        
        This is an enhanced version that handles detections at chunk boundaries
        by checking for overlaps between adjacent chunks.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed image data with chunks
            detection_params: Detection parameters
            status_callback: Optional callback for status updates
            
        Returns:
            Dictionary with detection results
        """
        # Use standard sequential processing
        result = self.process_chunks_sequential(preprocessed_data, detection_params, status_callback)
        
        # Merge overlapping detections at chunk boundaries
        if detection_params.get('merge_overlapping', True):
            result['matches'] = self._merge_boundary_detections(
                result['matches'],
                overlap_threshold=detection_params.get('overlap_sensitivity', 0.3)
            )
        
        return result
    
    def _merge_boundary_detections(self, detections: List[Dict], 
                                  overlap_threshold: float = 0.3) -> List[Dict]:
        """
        Merge detections that overlap at chunk boundaries
        
        Args:
            detections: List of all detections
            overlap_threshold: Threshold for considering detections as overlapping
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence (keep higher confidence detections)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        skip_indices = set()
        
        for i, det1 in enumerate(sorted_detections):
            if i in skip_indices:
                continue
            
            # Check for overlaps with remaining detections
            has_merged = False
            
            for j, det2 in enumerate(sorted_detections[i+1:], start=i+1):
                if j in skip_indices:
                    continue
                
                # Check if same template and overlaps
                if det1['template_name'] == det2['template_name']:
                    overlap = self._calculate_overlap(det1, det2)
                    
                    if overlap > overlap_threshold:
                        # Keep the higher confidence detection
                        skip_indices.add(j)
                        has_merged = True
            
            merged.append(det1)
        
        return merged
    
    def process_chunks_as_individual_results(self, preprocessed_data: Dict, detection_params: Dict,
                                            status_callback=None) -> List[Dict]:
        """
        Process chunks and return each chunk as an individual result (for separate display)
        
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
            status_callback(f"🖥️ CPU Sequential: Processing {chunk_info['total_chunks']} chunks for {filename}")
        
        # Convert all chunks to OpenCV format
        chunk_cv_images = []
        for chunk_data in chunks:
            chunk_pil = chunk_data['chunk']
            chunk_cv = pil_to_cv(chunk_pil)
            chunk_cv_images.append(chunk_cv)
        
        # Create individual results for each chunk
        individual_results = []
        total_chunks = len(chunk_cv_images)
        
        for chunk_idx, (chunk_cv, chunk_data) in enumerate(zip(chunk_cv_images, chunks)):
            if status_callback and chunk_idx % 5 == 0:
                status_callback(f"🖥️ CPU processing chunk {chunk_idx+1}/{total_chunks}")
            
            chunk_start_time = time.time()
            chunk_matches = []
            
            # Template matching
            has_templates = hasattr(st.session_state, 'templates') and st.session_state.templates
            if has_templates:
                template_matches = self._sequential_template_matching(
                    chunk_cv, chunk_idx, detection_params
                )
                chunk_matches.extend(template_matches)
            
            # Colored rectangle detection
            if detection_params.get('detect_green_rectangles', False):
                colored_matches = self._detect_colored_in_chunk(
                    chunk_cv, chunk_idx, detection_params
                )
                chunk_matches.extend(colored_matches)
            
            chunk_processing_time = time.time() - chunk_start_time
            
            # Create individual result for this chunk
            chunk_result = {
                'filename': f"{filename}_chunk_{chunk_idx+1}",
                'original_filename': filename,
                'image': chunk_data['chunk'],  # The actual chunk image
                'matches': chunk_matches,
                'chunk_index': chunk_idx,
                'chunk_info': {
                    'chunk_number': chunk_idx + 1,
                    'total_chunks': total_chunks,
                    'x_offset': chunk_data['x_offset'],
                    'y_offset': chunk_data['y_offset'],
                    'width': chunk_data['width'],
                    'height': chunk_data['height'],
                    'is_partial': chunk_data['width'] < 2048  # Assuming standard chunk width is 2048
                },
                'processing_mode': 'sequential_cpu',
                'processing_time': chunk_processing_time,
                'gpu_available': False
            }
            individual_results.append(chunk_result)
        
        total_time = time.time() - start_time
        
        if status_callback:
            status_callback(f"✅ CPU Sequential complete: {len(individual_results)} chunks processed")
        
        return individual_results
    
    def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
        """
        Calculate overlap ratio between two detections
        
        Args:
            det1: First detection
            det2: Second detection
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Calculate intersection
        x1_left = det1['x']
        y1_top = det1['y']
        x1_right = det1['x'] + det1['width']
        y1_bottom = det1['y'] + det1['height']
        
        x2_left = det2['x']
        y2_top = det2['y']
        x2_right = det2['x'] + det2['width']
        y2_bottom = det2['y'] + det2['height']
        
        # Find intersection rectangle
        x_left = max(x1_left, x2_left)
        y_top = max(y1_top, y2_top)
        x_right = min(x1_right, x2_right)
        y_bottom = min(y1_bottom, y2_bottom)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        det1_area = det1['width'] * det1['height']
        det2_area = det2['width'] * det2['height']
        
        # Calculate overlap as ratio of intersection to smaller detection
        smaller_area = min(det1_area, det2_area)
        
        if smaller_area == 0:
            return 0.0
        
        return intersection_area / smaller_area
