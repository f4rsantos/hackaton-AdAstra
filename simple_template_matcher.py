"""
Simple, reliable template matching using TM_CCOEFF_NORMED.
No GPU, no parallel processing, no complexity - just straightforward matching.
"""

import cv2 as cv
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st


def apply_colormap_always(image: np.ndarray) -> np.ndarray:
    """
    ALWAYS apply VIRIDIS colormap to ensure consistent color space.
    Converts to grayscale first, then applies colormap.
    This normalizes all images to the same color representation.
    
    Args:
        image: Input image (any format)
        
    Returns:
        Colored BGR image with VIRIDIS colormap
    """
    # ALWAYS convert to grayscale first (even if already colored)
    if len(image.shape) == 2:
        # Already grayscale
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale (this normalizes colored images)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image
    
    # Apply VIRIDIS colormap to normalized grayscale
    colored = cv.applyColorMap(gray, cv.COLORMAP_VIRIDIS)
    
    return colored


def match_single_template(
    image: np.ndarray,
    template: np.ndarray,
    template_name: str,
    threshold: float = 0.6,
    method: int = cv.TM_CCOEFF_NORMED
) -> List[Dict]:
    """
    Match a single template against an image using OpenCV's matchTemplate.
    
    Args:
        image: Search image (BGR)
        template: Template image (BGR)
        template_name: Name of the template
        threshold: Minimum correlation threshold (0.0 to 1.0)
        method: OpenCV matching method (default: TM_CCOEFF_NORMED)
        
    Returns:
        List of match dictionaries
    """
    if image is None or template is None:
        return []
    
    # Apply colormap to BOTH image and template
    image = apply_colormap_always(image)
    template = apply_colormap_always(template)
    
    template_h, template_w = template.shape[:2]
    image_h, image_w = image.shape[:2]
    
    # Check if template is larger than image
    if template_h > image_h or template_w > image_w:
        return []
    
    # Ensure images have the same dtype
    if image.dtype != template.dtype:
        image = image.astype(template.dtype)
    
    # Perform template matching
    result = cv.matchTemplate(image, template, method)
    
    # Get statistics
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    
    # Find all matches above threshold
    locations = np.where(result >= threshold)
    
    matches = []
    
    if len(locations[0]) > 0 and len(locations[1]) > 0:
        for y, x in zip(locations[0], locations[1]):
            confidence = float(result[y, x])
            
            match = {
                'template_name': template_name,
                'confidence': confidence,
                'x': int(x),
                'y': int(y),
                'width': template_w,
                'height': template_h,
                'center_x': int(x + template_w / 2),
                'center_y': int(y + template_h / 2),
                'partial': False,
                'detection_type': 'simple_ccoeff',
                'method': 'TM_CCOEFF_NORMED'
            }
            matches.append(match)
    
    return matches


def non_max_suppression(matches: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    Early exit for high-confidence matches.
    
    Args:
        matches: List of match dictionaries
        overlap_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of matches
    """
    if len(matches) <= 1:
        return matches
    
    # Sort by confidence (highest first)
    matches = sorted(matches, key=lambda m: m['confidence'], reverse=True)
    
    # Early exit: if top match has very high confidence (>0.95), skip NMS
    # This assumes near-perfect matches don't need overlap checking
    if matches[0]['confidence'] > 0.95 and len(matches) == 1:
        return matches
    
    keep = []
    
    for i, match in enumerate(matches):
        # Check if this match overlaps too much with any kept match
        should_keep = True
        
        for kept_match in keep:
            # Only suppress if same template
            if match['template_name'] != kept_match['template_name']:
                continue
            
            # Calculate IoU
            x1 = max(match['x'], kept_match['x'])
            y1 = max(match['y'], kept_match['y'])
            x2 = min(match['x'] + match['width'], kept_match['x'] + kept_match['width'])
            y2 = min(match['y'] + match['height'], kept_match['y'] + kept_match['height'])
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = match['width'] * match['height']
            area2 = kept_match['width'] * kept_match['height']
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > overlap_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(match)
    
    return keep


def match_all_templates(
    image: np.ndarray,
    threshold: float = 0.6,
    min_confidence: float = 0.6,
    apply_nms: bool = True
) -> List[Dict]:
    """
    Match all templates from session state against an image.
    
    Args:
        image: Search image (BGR)
        threshold: Matching threshold
        min_confidence: Minimum confidence to include in results
        apply_nms: Whether to apply non-maximum suppression
        
    Returns:
        List of all matches
    """
    if not hasattr(st.session_state, 'templates') or not st.session_state.templates:
        return []
    
    all_matches = []
    
    for template_key, template_data in st.session_state.templates.items():
        try:
            template_image = template_data['image']  # This is OpenCV format (BGR)
            template_name = template_data.get('clean_name', template_key)
            
            # Match this template
            matches = match_single_template(
                image=image,
                template=template_image,
                template_name=template_name,
                threshold=threshold,
                method=cv.TM_CCOEFF_NORMED
            )
            
            # Filter by minimum confidence
            filtered_matches = [m for m in matches if m['confidence'] >= min_confidence]
            
            if filtered_matches:
                all_matches.extend(filtered_matches)
            
        except Exception as e:
            continue
    
    # Apply NMS if requested
    if apply_nms and len(all_matches) > 0:
        all_matches = non_max_suppression(all_matches, overlap_threshold=0.3)
    
    return all_matches


if __name__ == "__main__":
    print("Simple Template Matcher - Test Mode")
    print("This module should be imported and used with Streamlit session state")
