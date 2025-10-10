"""
SigMF Annotation Processor for Drone Detection
Handles loading, creating, and saving SigMF metadata with drone detection annotations.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sigmf
from sigmf import SigMFFile

class SigMFProcessor:
    """Process SigMF metadata files and add drone detection annotations."""
    
    def __init__(self, sigmf_meta_path: Optional[str] = None):
        """
        Initialize SigMF processor.
        
        Args:
            sigmf_meta_path: Path to .sigmf-meta file (optional)
        """
        self.sigmf_meta_path = sigmf_meta_path
        self.sigmf_file = None
        self.sample_rate = None
        self.center_freq = None
        self.fft_size = None
        self.step_size = None
        self.total_samples = None  # Total samples in the data file
        
        if sigmf_meta_path and os.path.exists(sigmf_meta_path):
            self.load_metadata()
    
    def load_metadata(self, sigmf_meta_path: Optional[str] = None) -> bool:
        """
        Load SigMF metadata file.
        
        Args:
            sigmf_meta_path: Path to .sigmf-meta file (overrides initialization path)
            
        Returns:
            True if successful, False otherwise
        """
        if sigmf_meta_path:
            self.sigmf_meta_path = sigmf_meta_path
            
        if not self.sigmf_meta_path:
            return False
            
        try:
            # Remove .sigmf-meta extension if present (sigmf library handles it)
            base_path = str(self.sigmf_meta_path).replace('.sigmf-meta', '')
            self.sigmf_file = sigmf.sigmffile.fromfile(base_path)
            
            # Extract key parameters from global metadata
            self.sample_rate = self.sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
            
            # Get center frequency from first capture segment
            captures = self.sigmf_file.get_captures()
            if captures:
                self.center_freq = captures[0].get(SigMFFile.FREQUENCY_KEY, 0)
            else:
                self.center_freq = 0
            
            # Try to get total number of samples from the data file
            # Method 1: Check if there's a data file and calculate from its size
            sigmf_data_path = str(self.sigmf_meta_path).replace('.sigmf-meta', '.sigmf-data')
            if os.path.exists(sigmf_data_path):
                try:
                    # Get file size in bytes
                    file_size = os.path.getsize(sigmf_data_path)
                    
                    # Parse datatype to get bytes per sample
                    datatype = self.sigmf_file.get_global_field(SigMFFile.DATATYPE_KEY)
                    
                    # Common datatypes and their sizes (in bytes per I/Q sample)
                    datatype_sizes = {
                        'cf32_le': 8,  # complex float32 = 4 bytes I + 4 bytes Q
                        'cf32_be': 8,
                        'ci16_le': 4,  # complex int16 = 2 bytes I + 2 bytes Q
                        'ci16_be': 4,
                        'ci8': 2,      # complex int8 = 1 byte I + 1 byte Q
                        'cu8': 2,      # complex uint8 = 1 byte I + 1 byte Q
                    }
                    
                    bytes_per_sample = datatype_sizes.get(datatype, 8)  # Default to cf32_le
                    self.total_samples = file_size // bytes_per_sample
                    
                except Exception:
                    self.total_samples = None
            
            return True
            
        except Exception:
            return False
    
    def set_spectrogram_params(self, fft_size: int, step_size: Optional[int] = None):
        """
        Set spectrogram parameters for coordinate conversion.
        
        Args:
            fft_size: FFT size used in ORIGINAL spectrogram generation
            step_size: Step size (overlap) in samples (default: fft_size // 2)
        """
        self.fft_size = fft_size
        self.step_size = step_size if step_size else fft_size // 2
    
    def get_expected_image_dimensions(self) -> Optional[Tuple[int, int]]:
        """
        Calculate the expected spectrogram image dimensions based on SigMF data.
        This returns the TRUE original dimensions that should be used for coordinate mapping.
        
        Returns:
            Tuple of (width, height) in pixels, or None if parameters not set
        """
        if not (self.total_samples and self.fft_size and self.step_size):
            return None
        
        # Width = number of FFT columns
        width = (self.total_samples - self.fft_size) // self.step_size + 1
        
        # Height = FFT size (number of frequency bins)
        height = self.fft_size
        
        return (width, height)
    
    def pixel_to_time_freq(
        self, 
        x: int, 
        y: int, 
        width: int, 
        height: int,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, float, float]:
        """
        Convert pixel coordinates to time-frequency (SigMF annotation format).
        
        CRITICAL: Standard spectrogram orientation:
        - X-axis (horizontal) = TIME (left to right)
        - Y-axis (vertical) = FREQUENCY (bottom to top, but image Y goes top to bottom!)
        
        Args:
            x, y: Top-left corner of detection in pixels
            width, height: Detection box size in pixels
            image_width, image_height: Full spectrogram image dimensions
            
        Returns:
            Tuple of (start_index, length_samples, f_lo, f_hi)
            - start_index: Start sample index in time
            - length_samples: Duration in samples
            - f_lo: Lower frequency bound in Hz (absolute frequency)
            - f_hi: Upper frequency bound in Hz (absolute frequency)
        """
        if not self.sample_rate:
            raise ValueError("Sample rate not set - load metadata first")
            
        if not self.fft_size or not self.step_size:
            raise ValueError("Spectrogram params not set - call set_spectrogram_params() first")
        
        # CRITICAL: Our images use STANDARD orientation (X=time, Y=freq)
        # But annotation script's spectrogram array uses (rows=time, cols=freq)
        # So we need to swap X<->Y when applying their formulas!
        
        # OUR IMAGE ORIENTATION:
        # - X-axis (horizontal, left-to-right) = TIME
        # - Y-axis (vertical, top-to-bottom) = FREQUENCY
        
        # ANNOTATION SCRIPT FORMULAS (their Y=time, X=freq):
        # st = int(y * stp_size)
        # ln = int(h * stp_size)
        # fl = int(x / fft_size * sample_rate - (sample_rate / 2) + center_freq)
        # fu = int((x + w) / fft_size * sample_rate - (sample_rate / 2) + center_freq)
        
        # ADAPTED TO OUR ORIENTATION (our X=time, Y=freq):
        # st = int(x * stp_size)  
        # ln = int(w * stp_size)  
        # fl = int(y / fft_size * sample_rate - (sample_rate / 2) + center_freq)
        # fu = int((y + h) / fft_size * sample_rate - (sample_rate / 2) + center_freq)
        
        # SCALING: If image dimensions don't match spectrogram dimensions, scale first
        
        # TIME AXIS (X-axis, horizontal)
        # Calculate number of FFT columns in the spectrogram
        if self.total_samples:
            num_spectrogram_cols = (self.total_samples - self.fft_size) // self.step_size + 1
            # Scale pixel X to spectrogram column index
            spec_x = x * (num_spectrogram_cols / image_width) if image_width > 0 else x
            spec_w = width * (num_spectrogram_cols / image_width) if image_width > 0 else width
        else:
            # No scaling if we don't know total_samples (assume 1 pixel = 1 column)
            spec_x = x
            spec_w = width
        
        # Convert to sample indices
        start_index = int(spec_x * self.step_size)
        length_samples = int(spec_w * self.step_size)
        
        # FREQUENCY AXIS (Y-axis, vertical)
        # CRITICAL: Image Y-axis is INVERTED compared to frequency axis!
        # - Image: Y=0 at TOP, Y increases DOWNWARD
        # - Spectrogram: Low freq at BOTTOM, high freq at TOP
        # So we need to flip Y coordinate
        y_flipped = image_height - y - height
        
        # Scale flipped pixel Y to FFT bin index (0 to fft_size)
        spec_y = y_flipped * (self.fft_size / image_height) if image_height > 0 else y_flipped
        spec_h = height * (self.fft_size / image_height) if image_height > 0 else height
        
        # Apply frequency formula
        f_lo = int(spec_y / self.fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq)
        f_hi = int((spec_y + spec_h) / self.fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq)
        
        return start_index, length_samples, f_lo, f_hi
    
    def create_annotation(
        self,
        x: int,
        y: int, 
        width: int,
        height: int,
        image_width: int,
        image_height: int,
        label: str,
        confidence: Optional[float] = None,
        generator: str = "Ad_Astra_Drone_Detector"
    ) -> Dict:
        """
        Create a SigMF annotation from pixel coordinates.
        
        Args:
            x, y: Top-left corner of detection in pixels
            width, height: Detection box size in pixels
            image_width, image_height: Full spectrogram image dimensions (ORIGINAL size!)
            label: Drone template label (e.g., "007-Comns1")
            confidence: Detection confidence score (optional)
            generator: Tool name that generated the annotation
            
        Returns:
            Dictionary with SigMF annotation structure
        """
        # Convert pixel coordinates to time-frequency
        start_idx, length_samples, f_lo, f_hi = self.pixel_to_time_freq(
            x, y, width, height, image_width, image_height
        )
        
        # Validate values
        if start_idx < 0:
            start_idx = 0
        
        if length_samples < 0:
            length_samples = 1
        
        # Create SigMF annotation structure
        annotation = {
            SigMFFile.START_INDEX_KEY: int(start_idx),
            SigMFFile.LENGTH_INDEX_KEY: int(length_samples),
            SigMFFile.FLO_KEY: int(f_lo),
            SigMFFile.FHI_KEY: int(f_hi),
            SigMFFile.LABEL_KEY: str(label),
            SigMFFile.GENERATOR_KEY: str(generator)
        }
        
        # Add confidence if provided (custom field)
        if confidence is not None:
            annotation['core:confidence'] = float(confidence)
        
        return annotation
    
    def add_detections(
        self,
        detections: List[Dict],
        image_width: int,
        image_height: int,
        replace_existing: bool = False
    ) -> int:
        """
        Add multiple drone detections as SigMF annotations.
        
        Args:
            detections: List of detection dictionaries with keys:
                       'x', 'y', 'width', 'height', 'label', 'confidence' (optional)
            image_width: Full spectrogram width in pixels
            image_height: Full spectrogram height in pixels
            replace_existing: If True, replace all existing annotations
            
        Returns:
            Number of annotations added
        """
        if self.sigmf_file is None:
            return 0
        
        # Initialize annotations list if needed
        if SigMFFile.ANNOTATION_KEY not in self.sigmf_file._metadata:
            self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY] = []
        
        # Clear existing if requested
        if replace_existing:
            self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY] = []
        
        # Add new annotations
        for det in detections:
            try:
                annotation = self.create_annotation(
                    x=det['x'],
                    y=det['y'],
                    width=det['width'],
                    height=det['height'],
                    image_width=image_width,
                    image_height=image_height,
                    label=det['label'],
                    confidence=det.get('confidence')
                )
                
                self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY].append(annotation)
                
            except Exception:
                pass
        
        # Sort annotations by start index
        if len(self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY]) > 0:
            self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY].sort(
                key=lambda x: x[SigMFFile.START_INDEX_KEY]
            )
        
        return len(self.sigmf_file._metadata[SigMFFile.ANNOTATION_KEY])
    
    def save_metadata(self, output_path: Optional[str] = None) -> bool:
        """
        Save SigMF metadata with annotations to disk.
        
        Args:
            output_path: Path to save .sigmf-meta file (default: overwrite original)
            
        Returns:
            True if successful, False otherwise
        """
        if self.sigmf_file is None:
            return False
        
        try:
            # Use original path if no output specified
            save_path = output_path if output_path else self.sigmf_meta_path
            
            if not save_path:
                return False
            
            # Ensure path ends with .sigmf-meta
            if not save_path.endswith('.sigmf-meta'):
                save_path = save_path + '.sigmf-meta'
            
            # Write metadata to file directly using JSON - FAST!
            import json
            with open(save_path, 'w') as f:
                json.dump(self.sigmf_file._metadata, f, indent=2)
            
            return os.path.exists(save_path)
            
        except Exception:
            return False
    
    def get_annotation_summary(self) -> str:
        """
        Get summary of current annotations.
        
        Returns:
            Formatted string with annotation statistics
        """
        if self.sigmf_file is None:
            return "No SigMF file loaded"
        
        annotations = self.sigmf_file._metadata.get(SigMFFile.ANNOTATION_KEY, [])
        
        if not annotations:
            return "No annotations found"
        
        # Count by label
        label_counts = {}
        for ann in annotations:
            label = ann.get(SigMFFile.LABEL_KEY, 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        summary = f"📊 Total Annotations: {len(annotations)}\n"
        for label, count in sorted(label_counts.items()):
            summary += f"   {label}: {count}\n"
        
        return summary.strip()


def find_sigmf_meta_for_image(image_path: str) -> Optional[str]:
    """
    Find corresponding .sigmf-meta file for a spectrogram image.
    
    Args:
        image_path: Path to spectrogram image (.png)
        
    Returns:
        Path to .sigmf-meta file if found, None otherwise
    """
    # Try same name with .sigmf-meta extension
    base_path = str(image_path).replace('.png', '').replace('.jpg', '')
    meta_path = base_path + '.sigmf-meta'
    
    if os.path.exists(meta_path):
        return meta_path
    
    # Try in same directory with different extensions
    image_file = Path(image_path)
    parent_dir = image_file.parent
    stem = image_file.stem
    
    for meta_file in parent_dir.glob(f"{stem}*.sigmf-meta"):
        return str(meta_file)
    
    return None


# Example usage
if __name__ == "__main__":
    # Example: Process detections and save annotations
    processor = SigMFProcessor("example.sigmf-meta")
    
    if processor.load_metadata():
        # Set spectrogram parameters
        processor.set_spectrogram_params(fft_size=2048, step_size=1024)
        
        # Example detections from template matching
        detections = [
            {'x': 100, 'y': 50, 'width': 80, 'height': 120, 'label': '007-Comns1', 'confidence': 0.95},
            {'x': 300, 'y': 200, 'width': 100, 'height': 150, 'label': '007-Controller', 'confidence': 0.87},
        ]
        
        # Add annotations
        processor.add_detections(
            detections=detections,
            image_width=1024,
            image_height=512,
            replace_existing=True
        )
        
        # Print summary
        print(processor.get_annotation_summary())
        
        # Save
        processor.save_metadata()
