#!/usr/bin/env python3
"""
Image Preprocessing Pipeline

Handles resizing and chunking of large spectrogram images for detection processing.
- Resizes large images (e.g., 101708x1229) to standard size (31778x384)
- Chunks resized images into smaller pieces (2048x384) for efficient processing
- Handles remaining chunks that don't fit the standard size
- Parallel CPU processing for template matching across chunks
"""

import numpy as np
from PIL import Image
import cv2 as cv
from typing import List, Tuple, Dict, Callable, Optional
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class ImagePreprocessor:
    """
    Preprocesses large spectrogram images by resizing and chunking them
    """
    
    def __init__(self, target_width=31778, target_height=384, chunk_width=2048, chunk_height=384):
        """
        Initialize preprocessor with target dimensions
        
        Args:
            target_width: Target width after resizing (default: 31778)
            target_height: Target height after resizing (default: 384)
            chunk_width: Width of each chunk (default: 2048)
            chunk_height: Height of each chunk (default: 384)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target dimensions using fast OpenCV backend
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized PIL Image
        """
        # Convert PIL to OpenCV for faster resize (3-4x speedup)
        img_array = np.array(image)
        
        # Use INTER_AREA for downscaling (best quality) or INTER_LINEAR for speed
        # INTER_AREA is optimal for shrinking images
        resized_array = cv.resize(img_array, (self.target_width, self.target_height), 
                                  interpolation=cv.INTER_AREA)
        
        # Convert back to PIL
        resized = Image.fromarray(resized_array)
        return resized
    
    def chunk_image(self, image: Image.Image) -> List[Dict]:
        """
        Split image into chunks of specified size
        
        Args:
            image: PIL Image to chunk (should already be resized)
            
        Returns:
            List of dictionaries containing:
                - 'chunk': PIL Image chunk
                - 'chunk_index': Index of the chunk
                - 'x_offset': X position in original image
                - 'y_offset': Y position in original image
                - 'width': Width of chunk
                - 'height': Height of chunk
        """
        img_width, img_height = image.size
        chunks = []
        chunk_index = 0
        
        # Process full-width chunks
        x_offset = 0
        while x_offset < img_width:
            # Determine chunk width (last chunk may be smaller)
            current_chunk_width = min(self.chunk_width, img_width - x_offset)
            
            # Extract chunk
            chunk_img = image.crop((x_offset, 0, x_offset + current_chunk_width, img_height))
            
            chunks.append({
                'chunk': chunk_img,
                'chunk_index': chunk_index,
                'x_offset': x_offset,
                'y_offset': 0,
                'width': current_chunk_width,
                'height': img_height
            })
            
            chunk_index += 1
            x_offset += self.chunk_width
        
        return chunks
    
    def process_image(self, image_input) -> Tuple[Image.Image, List[Dict]]:
        """
        Full preprocessing pipeline: resize and chunk with lazy loading optimization
        
        Args:
            image_input: PIL Image or file-like object
            
        Returns:
            Tuple of (resized_image, list_of_chunks)
        """
        # Load image with lazy loading optimization
        if not isinstance(image_input, Image.Image):
            image = Image.open(image_input)
            # Lazy loading: Use draft mode to defer full decode until resize
            # This significantly speeds up loading of large images
            try:
                # Draft mode prepares image for efficient resize
                # Calculate appropriate mode and size hint
                target_size = (self.target_width, self.target_height)
                image.draft(None, target_size)  # Prepare for target size
            except Exception:
                # If draft fails, proceed with normal loading
                pass
        else:
            image = image_input
        
        # Step 1: Resize to target dimensions (now using fast OpenCV backend)
        resized_image = self.resize_image(image)
        
        # Step 2: Chunk into smaller pieces
        chunks = self.chunk_image(resized_image)
        
        return resized_image, chunks
    
    def get_chunk_info(self, chunks: List[Dict]) -> Dict:
        """
        Get summary information about chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'full_size_chunks': 0,
                'partial_chunks': 0,
                'total_width': 0,
                'total_height': 0
            }
        
        full_size_chunks = sum(1 for c in chunks if c['width'] == self.chunk_width)
        partial_chunks = len(chunks) - full_size_chunks
        
        return {
            'total_chunks': len(chunks),
            'full_size_chunks': full_size_chunks,
            'partial_chunks': partial_chunks,
            'chunk_size': f"{self.chunk_width}x{self.chunk_height}",
            'last_chunk_width': chunks[-1]['width'] if chunks else 0,
            'total_width': chunks[-1]['x_offset'] + chunks[-1]['width'] if chunks else 0,
            'total_height': chunks[0]['height'] if chunks else 0
        }


def preprocess_images_batch(image_inputs: List, target_width=31778, target_height=384, 
                            chunk_width=2048, chunk_height=384) -> List[Dict]:
    """
    Preprocess a batch of images
    
    Args:
        image_inputs: List of PIL Images or file-like objects
        target_width: Target width after resizing
        target_height: Target height after resizing
        chunk_width: Width of each chunk
        chunk_height: Height of each chunk
        
    Returns:
        List of dictionaries, one per image, containing:
            - 'original_filename': Original filename
            - 'original_size': Tuple of (width, height)
            - 'resized_image': Resized PIL Image
            - 'chunks': List of chunk dictionaries
            - 'chunk_info': Summary information
    """
    preprocessor = ImagePreprocessor(target_width, target_height, chunk_width, chunk_height)
    
    results = []
    
    for idx, image_input in enumerate(image_inputs):
        # Get filename
        filename = getattr(image_input, 'name', f'image_{idx+1}')
        
        # Load image
        if not isinstance(image_input, Image.Image):
            image = Image.open(image_input)
        else:
            image = image_input
        
        original_size = image.size
        
        # Process image
        resized_image, chunks = preprocessor.process_image(image)
        
        # Get chunk info
        chunk_info = preprocessor.get_chunk_info(chunks)
        
        results.append({
            'original_filename': filename,
            'original_size': original_size,
            'resized_image': resized_image,
            'chunks': chunks,
            'chunk_info': chunk_info
        })
    
    return results


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR)
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image array (BGR format)
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    np_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    cv_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)
    
    return cv_image


def reconstruct_detections_from_chunks(chunk_detections: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """
    Reconstruct full-image detection coordinates from chunk-based detections
    
    Args:
        chunk_detections: List of detection results from each chunk
        chunks: List of chunk metadata (with offsets)
        
    Returns:
        List of detections with adjusted coordinates for the full image
    """
    reconstructed_detections = []
    
    for chunk_idx, (chunk_det, chunk_meta) in enumerate(zip(chunk_detections, chunks)):
        x_offset = chunk_meta['x_offset']
        y_offset = chunk_meta['y_offset']
        
        # Adjust each detection in this chunk
        for detection in chunk_det.get('matches', []):
            adjusted_detection = detection.copy()
            adjusted_detection['x'] += x_offset
            adjusted_detection['y'] += y_offset
            adjusted_detection['center_x'] += x_offset
            adjusted_detection['center_y'] += y_offset
            adjusted_detection['chunk_index'] = chunk_idx
            adjusted_detection['chunk_offset'] = (x_offset, y_offset)
            
            reconstructed_detections.append(adjusted_detection)
    
    return reconstructed_detections


def process_chunk_parallel(chunk_data: Tuple[int, Dict, Dict, Callable]) -> Tuple[int, Dict]:
    """
    Process a single chunk with template matching (worker function for parallel processing)
    
    Args:
        chunk_data: Tuple of (chunk_idx, chunk_info, templates, detection_function)
        
    Returns:
        Tuple of (chunk_idx, detection_results)
    """
    chunk_idx, chunk_info, templates, detection_function = chunk_data
    
    try:
        # Run detection on this chunk
        chunk_image = chunk_info['chunk']
        results = detection_function(chunk_image, templates)
        
        return (chunk_idx, results)
    except Exception as e:
        # Return empty results on error
        return (chunk_idx, {'matches': [], 'error': str(e)})


def process_chunks_parallel(chunks: List[Dict], templates: Dict, detection_function: Callable,
                            max_workers: Optional[int] = None, 
                            progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Process multiple chunks in parallel using CPU workers for template matching
    
    Args:
        chunks: List of chunk dictionaries from preprocess_images_batch
        templates: Dictionary of template images for matching
        detection_function: Function that performs detection on a single chunk
        max_workers: Maximum number of parallel workers (default: CPU count)
        progress_callback: Optional callback function(chunk_idx, total_chunks)
        
    Returns:
        List of detection results for each chunk (in original order)
    """
    if max_workers is None:
        # Use CPU count, but cap at 8 to avoid memory issues
        max_workers = min(os.cpu_count() or 4, 8)
    
    # Prepare chunk data for parallel processing
    chunk_tasks = [
        (idx, chunk, templates, detection_function) 
        for idx, chunk in enumerate(chunks)
    ]
    
    # Results dictionary to maintain order
    results = {}
    
    # Process chunks in parallel using ThreadPoolExecutor (good for I/O and OpenCV operations)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk_parallel, task): task[0] 
            for task in chunk_tasks
        }
        
        # Collect results as they complete
        completed = 0
        total = len(chunks)
        
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_idx, result = future.result()
                results[chunk_idx] = result
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                    
            except Exception as e:
                # Store error result
                results[chunk_idx] = {'matches': [], 'error': str(e)}
    
    # Return results in original chunk order
    return [results[i] for i in range(len(chunks))]


def preprocess_and_detect_parallel(image_inputs: List, templates: Dict, detection_function: Callable,
                                   target_width: int = 31778, target_height: int = 384,
                                   chunk_width: int = 2048, chunk_height: int = 384,
                                   max_workers: Optional[int] = None,
                                   progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Complete pipeline: Preprocess images and run parallel detection on chunks
    
    Args:
        image_inputs: List of PIL Images or file-like objects
        templates: Dictionary of template images for matching
        detection_function: Function that performs detection on a single chunk image
        target_width: Target width after resizing (default: 31778)
        target_height: Target height after resizing (default: 384)
        chunk_width: Width of each chunk (default: 2048)
        chunk_height: Height of each chunk (default: 384)
        max_workers: Maximum number of parallel workers (default: CPU count)
        progress_callback: Optional callback function(current, total, message)
        
    Returns:
        List of dictionaries, one per image, containing:
            - 'original_filename': Original filename
            - 'chunks': List of chunk detection results with adjusted coordinates
            - 'all_detections': All detections combined with full-image coordinates
    """
    # Step 1: Preprocess images (resize + chunk)
    if progress_callback:
        progress_callback(0, len(image_inputs), "Preprocessing images...")
    
    preprocessed_results = preprocess_images_batch(
        image_inputs, target_width, target_height, chunk_width, chunk_height
    )
    
    # Step 2: Process all chunks from all images in parallel
    results = []
    
    for img_idx, img_result in enumerate(preprocessed_results):
        if progress_callback:
            progress_callback(img_idx, len(image_inputs), 
                            f"Processing {img_result['original_filename']}...")
        
        chunks = img_result['chunks']
        
        # Run parallel detection on chunks
        chunk_detections = process_chunks_parallel(
            chunks, templates, detection_function, max_workers,
            lambda curr, total: progress_callback(
                img_idx, len(image_inputs),
                f"Processing chunk {curr}/{total} of {img_result['original_filename']}"
            ) if progress_callback else None
        )
        
        # Reconstruct full-image coordinates
        all_detections = reconstruct_detections_from_chunks(chunk_detections, chunks)
        
        results.append({
            'original_filename': img_result['original_filename'],
            'original_size': img_result['original_size'],
            'resized_image': img_result['resized_image'],
            'chunks': chunk_detections,  # Individual chunk results
            'all_detections': all_detections,  # Combined results with adjusted coordinates
            'chunk_info': img_result['chunk_info']
        })
    
    if progress_callback:
        progress_callback(len(image_inputs), len(image_inputs), "Processing complete!")
    
    return results
