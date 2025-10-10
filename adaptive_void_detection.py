#!/usr/bin/env python3
"""
Adaptive Void Detection for RF Spectrograms

This system detects void/background areas in RF spectrograms regardless of color.
RF spectrograms can have void areas in many different colors (red, blue, green, purple, etc.)
representing background noise or inactive frequency regions.

Key Features:
- Color-agnostic void detection using statistical analysis
- Adaptive thresholding based on image histogram
- Intensity-based background detection
- Edge-based signal area identification
- Works with any colormap (jet, viridis, plasma, etc.)
"""

import cv2 as cv
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from scipy import ndimage
from sklearn.cluster import KMeans


class AdaptiveVoidDetector:
    """
    Advanced void detection that works with any RF spectrogram colormap
    """
    
    def __init__(self, 
                 min_void_area: int = 100,
                 signal_intensity_percentile: float = 70,
                 edge_sensitivity: float = 0.1):
        """
        Initialize adaptive void detector
        
        Args:
            min_void_area: Minimum area for void regions
            signal_intensity_percentile: Percentile threshold for signal detection
            edge_sensitivity: Sensitivity for edge-based signal detection
        """
        self.min_void_area = min_void_area
        self.signal_intensity_percentile = signal_intensity_percentile
        self.edge_sensitivity = edge_sensitivity
        
    def analyze_image_statistics(self, image: np.ndarray) -> Dict:
        """
        Analyze image statistics to understand the distribution of intensities and colors
        
        Args:
            image: Input BGR image
            
        Returns:
            dict: Image statistics including histograms, thresholds, etc.
        """
        # Convert to grayscale for intensity analysis
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        
        # Find intensity statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        median_intensity = np.median(gray)
        
        # Calculate percentile thresholds
        low_threshold = np.percentile(gray, 25)
        high_threshold = np.percentile(gray, self.signal_intensity_percentile)
        
        # Find dominant intensity ranges using histogram analysis
        # Smooth histogram to find peaks
        smooth_hist = ndimage.gaussian_filter1d(hist_norm, sigma=2)
        
        # Find peaks in histogram (likely background and signal intensities)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smooth_hist, height=0.001, distance=10)
        
        # Separate into potential background and signal peaks
        background_peaks = peaks[peaks < len(smooth_hist) * 0.6]  # Lower intensities
        signal_peaks = peaks[peaks >= len(smooth_hist) * 0.6]     # Higher intensities
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'median_intensity': median_intensity,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'histogram': hist_norm,
            'smooth_histogram': smooth_hist,
            'background_peaks': background_peaks,
            'signal_peaks': signal_peaks,
            'intensity_range': (gray.min(), gray.max())
        }
    
    def detect_using_kmeans_clustering(self, image: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Use K-means clustering to separate background from signal areas
        
        Args:
            image: Input BGR image
            n_clusters: Number of clusters (typically 2-4 for background/signal separation)
            
        Returns:
            tuple: (void_mask, cluster_info)
        """
        # Prepare data for clustering
        if len(image.shape) == 3:
            # Use all color channels
            pixels = image.reshape(-1, 3).astype(np.float32)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            # Grayscale image
            gray = image.copy()
            pixels = gray.reshape(-1, 1).astype(np.float32)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        labels = labels.reshape(image.shape[:2])
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            cluster_intensity = np.mean(gray[cluster_mask])
            cluster_size = np.sum(cluster_mask)
            cluster_stats[i] = {
                'intensity': cluster_intensity,
                'size': cluster_size,
                'percentage': (cluster_size / gray.size) * 100
            }
        
        # Identify background clusters (typically lower intensity or largest area)
        # Sort clusters by intensity
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['intensity'])
        
        # Background is typically the lowest intensity cluster(s)
        # or very large uniform areas
        background_clusters = []
        
        # Method 1: Lowest intensity clusters
        background_clusters.append(sorted_clusters[0][0])  # Lowest intensity
        
        # Method 2: Very large uniform areas (likely background)
        for cluster_id, stats in cluster_stats.items():
            if stats['percentage'] > 40:  # Very large areas are likely background
                if cluster_id not in background_clusters:
                    background_clusters.append(cluster_id)
        
        # Create void mask from background clusters
        void_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for cluster_id in background_clusters:
            void_mask[labels == cluster_id] = 255
        
        return void_mask, {
            'cluster_stats': cluster_stats,
            'background_clusters': background_clusters,
            'labels': labels
        }
    
    def detect_using_edge_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Detect signal areas using edge analysis - signals typically have more edges
        
        Args:
            image: Input BGR image
            
        Returns:
            np.ndarray: Signal mask (inverse will be void mask)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate edges using multiple methods
        # Sobel edges
        sobel_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Canny edges
        canny = cv.Canny(blurred, 50, 150)
        
        # Laplacian edges
        laplacian = cv.Laplacian(blurred, cv.CV_64F)
        laplacian = np.abs(laplacian)
        
        # Combine edge information
        edge_strength = (sobel_combined / 255.0 * 0.4 + 
                        canny / 255.0 * 0.4 + 
                        laplacian / np.max(laplacian) * 0.2)
        
        # Apply morphological operations to identify edge-rich regions
        edge_strength_8bit = (edge_strength * 255).astype(np.uint8)
        
        # Dilate edges to create regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        edge_regions = cv.morphologyEx(edge_strength_8bit, cv.MORPH_DILATE, kernel)
        
        # Threshold to create signal mask
        edge_threshold = np.percentile(edge_regions, 60)  # Top 40% of edge activity
        signal_mask = (edge_regions > edge_threshold).astype(np.uint8) * 255
        
        return signal_mask
    
    def detect_using_intensity_analysis(self, image: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Detect void areas using intensity-based analysis
        
        Args:
            image: Input BGR image
            stats: Image statistics from analyze_image_statistics
            
        Returns:
            np.ndarray: Void mask
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method 1: Simple thresholding based on percentiles
        low_intensity_mask = gray < stats['low_threshold']
        
        # Method 2: Adaptive thresholding
        adaptive_thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
        )
        
        # Method 3: Otsu's thresholding
        _, otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        # Method 4: Background peaks analysis
        background_mask = np.zeros_like(gray, dtype=bool)
        if len(stats['background_peaks']) > 0:
            for peak in stats['background_peaks']:
                # Create mask for intensities around this peak
                peak_range = 20  # Range around peak
                peak_mask = (gray >= max(0, peak - peak_range)) & (gray <= min(255, peak + peak_range))
                background_mask |= peak_mask
        
        # Combine methods with weights
        combined_mask = (
            low_intensity_mask.astype(np.float32) * 0.3 +
            (adaptive_thresh > 0).astype(np.float32) * 0.2 +
            (otsu_thresh > 0).astype(np.float32) * 0.2 +
            background_mask.astype(np.float32) * 0.3
        )
        
        # Threshold combined result
        void_mask = (combined_mask > 0.4).astype(np.uint8) * 255
        
        return void_mask
    
    def detect_adaptive_voids(self, image: np.ndarray) -> Tuple[np.ndarray, List, Dict]:
        """
        Main adaptive void detection method that combines multiple approaches
        
        Args:
            image: Input BGR image
            
        Returns:
            tuple: (final_void_mask, void_contours, detection_stats)
        """
        
        # Step 1: Analyze image statistics
        stats = self.analyze_image_statistics(image)

        # Step 2: K-means clustering approach
        kmeans_mask, cluster_info = self.detect_using_kmeans_clustering(image, n_clusters=3)
        
        # Step 3: Edge analysis approach (inverse for void detection)
        signal_mask = self.detect_using_edge_analysis(image)
        edge_void_mask = cv.bitwise_not(signal_mask)
        
        # Step 4: Intensity-based approach
        intensity_void_mask = self.detect_using_intensity_analysis(image, stats)
        
        # Normalize all masks to 0-1 range
        kmeans_norm = kmeans_mask.astype(np.float32) / 255.0
        edge_norm = edge_void_mask.astype(np.float32) / 255.0
        intensity_norm = intensity_void_mask.astype(np.float32) / 255.0
        
        # Weighted combination
        combined_mask = (
            kmeans_norm * 0.4 +          # K-means clustering (most reliable)
            edge_norm * 0.3 +            # Edge analysis
            intensity_norm * 0.3         # Intensity analysis
        )
        
        # Apply final threshold
        final_threshold = 0.5  # At least 2 out of 3 methods must agree
        final_void_mask = (combined_mask > final_threshold).astype(np.uint8) * 255
        
        # Remove small noise
        kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        final_void_mask = cv.morphologyEx(final_void_mask, cv.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_medium = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        final_void_mask = cv.morphologyEx(final_void_mask, cv.MORPH_CLOSE, kernel_medium)
        
        # Find contours
        contours, _ = cv.findContours(final_void_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        void_stats = []
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area >= self.min_void_area:
                x, y, w, h = cv.boundingRect(contour)
                
                # Calculate void characteristics
                if len(image.shape) == 3:
                    region_gray = cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
                else:
                    region_gray = image[y:y+h, x:x+w]
                
                if region_gray.size > 0:
                    avg_intensity = np.mean(region_gray)
                    std_intensity = np.std(region_gray)
                    
                    # Calculate confidence based on uniformity and agreement between methods
                    uniformity = 1.0 - (std_intensity / 255.0)  # More uniform = higher confidence
                    method_agreement = combined_mask[y:y+h, x:x+w].mean()  # How much methods agree
                    
                    confidence = (uniformity * 0.5 + method_agreement * 0.5)
                    confidence = max(0.0, min(1.0, confidence))
                    
                    void_stats.append({
                        'type': 'adaptive_void',
                        'area': area,
                        'bbox': (x, y, w, h),
                        'avg_intensity': avg_intensity,
                        'std_intensity': std_intensity,
                        'uniformity': uniformity,
                        'confidence': confidence
                    })
                    
                    valid_contours.append(contour)
        
        detection_stats = {
            'image_stats': stats,
            'cluster_info': cluster_info,
            'method_agreement': {
                'kmeans_coverage': (kmeans_norm.sum() / kmeans_norm.size) * 100,
                'edge_coverage': (edge_norm.sum() / edge_norm.size) * 100,
                'intensity_coverage': (intensity_norm.sum() / intensity_norm.size) * 100,
                'final_coverage': (final_void_mask.sum() / 255.0 / final_void_mask.size) * 100
            },
            'total_voids': len(void_stats),
            'total_void_area': sum(s['area'] for s in void_stats),
            'void_details': void_stats
        }
        
        return final_void_mask, valid_contours, detection_stats
    
    def visualize_detection_process(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Create a comprehensive visualization of the detection process
        
        Args:
            image: Original image
            results: Results from detect_adaptive_voids
            
        Returns:
            np.ndarray: Visualization image
        """
        void_mask, contours, stats = results
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw void contours in red
        for i, contour in enumerate(contours):
            cv.drawContours(vis_image, [contour], -1, (0, 0, 255), 2)
            
            # Add labels
            x, y, w, h = cv.boundingRect(contour)
            label = f"V{i+1}"
            cv.putText(vis_image, label, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add detection statistics as text overlay
        text_y = 30
        cv.putText(vis_image, f"Adaptive Void Detection", (10, text_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 25
        cv.putText(vis_image, f"Voids: {stats['total_voids']}", (10, text_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += 20
        cv.putText(vis_image, f"Coverage: {stats['method_agreement']['final_coverage']:.1f}%", 
                  (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image


def test_adaptive_void_detection(image_path: str):
    """
    Test the adaptive void detection on a specific image
    
    Args:
        image_path: Path to test image
    """

    # Load image
    image = cv.imread(image_path)
    if image is None:
        return None
    
    # Create detector
    detector = AdaptiveVoidDetector(
        min_void_area=100,
        signal_intensity_percentile=70,
        edge_sensitivity=0.1
    )
    
    # Run detection
    results = detector.detect_adaptive_voids(image)
    
    # Create visualization
    vis_image = detector.visualize_detection_process(image, results)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"adaptive_void_detection_{base_name}.png"
    cv.imwrite(output_path, vis_image)
    
    return results


if __name__ == "__main__":
    # Test on all images in the current directory
    current_dir = os.getcwd()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(current_dir) 
                          if f.lower().endswith(ext.lower()) 
                          and not f.startswith('adaptive_void_detection_')])
    
    if image_files:
        for image_file in image_files[:3]:  # Test first 3 images
            try:
                test_adaptive_void_detection(image_file)
            except Exception as e:
                pass