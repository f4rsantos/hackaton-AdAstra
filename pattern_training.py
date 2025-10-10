"""
Pattern-Based Training Module for Ad Astra Drone Detection

This module implements automatic pattern extraction from drone signals:
1. Identifies repeating burst patterns
2. Extracts one complete cycle as a template
3. Trains new templates from single images or streams

Key innovation: Instead of manually creating templates, the system learns
patterns by detecting burst repetition and automatically extracting cycles.
"""

import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from burst_detection import BurstDetector, BurstDetectionConfig, PatternDetector


@dataclass
class PatternTrainingConfig:
    """Configuration for pattern-based template training"""
    # Pattern detection
    min_repetitions: int = 3  # Minimum cycles to confirm pattern
    timing_tolerance: float = 0.15  # 15% tolerance for timing variations
    
    # Cycle extraction
    min_cycle_duration: float = 0.001  # seconds
    max_cycle_duration: float = 1.0  # seconds
    
    # Template quality
    min_snr: float = 10.0  # dB, minimum SNR for good template
    min_pattern_confidence: float = 0.7  # 0-1 scale
    
    # Image processing
    template_width: int = 100  # pixels
    template_height: int = 100  # pixels
    apply_enhancement: bool = True
    
    # Burst detection config
    burst_config: BurstDetectionConfig = None
    
    def __post_init__(self):
        if self.burst_config is None:
            self.burst_config = BurstDetectionConfig()


class CycleExtractor:
    """
    Extracts individual cycles from repeating patterns.
    """
    
    def __init__(self, config: PatternTrainingConfig):
        self.config = config
    
    def find_cycle_boundaries(self, bursts: List[Dict], pattern: Dict) -> List[Tuple[int, int]]:
        """
        Determine the boundaries of each cycle in a repeating pattern.
        
        Args:
            bursts: List of detected bursts
            pattern: Pattern information (period, etc.)
            
        Returns:
            List of (start_sample, end_sample) tuples for each cycle
        """
        if len(bursts) < self.config.min_repetitions:
            return []
        
        cycles = []
        period_samples = pattern.get('period_samples', 0)
        
        if period_samples == 0:
            # Use actual burst spacing
            for i in range(len(bursts) - 1):
                start = bursts[i]['start_sample']
                end = bursts[i+1]['start_sample']
                cycles.append((start, end))
            
            # Include last burst with estimated end
            if len(bursts) > 1:
                avg_duration = np.mean([c[1] - c[0] for c in cycles])
                last_start = bursts[-1]['start_sample']
                last_end = int(last_start + avg_duration)
                cycles.append((last_start, last_end))
        else:
            # Use pattern period
            for burst in bursts:
                start = burst['start_sample']
                end = start + period_samples
                cycles.append((start, end))
        
        return cycles
    
    def extract_best_cycle(self, samples: np.ndarray, cycles: List[Tuple[int, int]], 
                          sample_rate: float) -> Optional[np.ndarray]:
        """
        Extract the cycle with best quality (highest power).
        
        Args:
            samples: IQ samples
            cycles: List of cycle boundaries
            sample_rate: Sample rate in Hz
            
        Returns:
            IQ samples of best cycle or None
        """
        if len(cycles) == 0:
            return None
        
        best_cycle = None
        best_power = -np.inf
        
        for start, end in cycles:
            if end > len(samples):
                continue
            
            cycle_samples = samples[start:end]
            
            # Calculate average power
            power = np.mean(np.abs(cycle_samples)**2)
            
            # Duration check
            duration = len(cycle_samples) / sample_rate
            if not (self.config.min_cycle_duration <= duration <= self.config.max_cycle_duration):
                continue
            
            if power > best_power:
                best_power = power
                best_cycle = cycle_samples
        
        return best_cycle
    
    def cycle_to_spectrogram_image(self, cycle_samples: np.ndarray, 
                                   sample_rate: float) -> np.ndarray:
        """
        Convert cycle IQ samples to spectrogram image for template.
        
        Args:
            cycle_samples: IQ samples of one cycle
            sample_rate: Sample rate in Hz
            
        Returns:
            RGB image array suitable as template
        """
        # Calculate spectrogram
        fft_size = min(256, len(cycle_samples) // 4)
        if fft_size < 16:
            fft_size = 16
        
        hop_size = fft_size // 2
        num_steps = (len(cycle_samples) - fft_size) // hop_size + 1
        
        if num_steps < 2:
            return None
        
        spectrogram = np.zeros((num_steps, fft_size), dtype=np.float32)
        window = np.hanning(fft_size).astype(np.complex64)
        window /= np.sqrt((np.abs(window)**2).sum())
        
        for i in range(num_steps):
            start_idx = i * hop_size
            end_idx = start_idx + fft_size
            if end_idx > len(cycle_samples):
                break
            
            windowed = window * cycle_samples[start_idx:end_idx]
            spectrum = np.fft.fftshift(np.fft.fft(windowed))
            spectrogram[i, :] = np.abs(spectrum)
        
        # Convert to dB
        spectrogram[spectrogram <= 0] = 1e-20
        spectrogram = 20 * np.log10(spectrogram)
        
        # Transpose and flip
        spec = spectrogram.T
        spec = np.flipud(spec)
        
        # Normalize
        vmin = np.percentile(spec, 5)
        vmax = np.percentile(spec, 95)
        spec = np.clip((spec - vmin) / (vmax - vmin + 1e-10), 0, 1)
        spec = (spec * 255).astype(np.uint8)
        
        # Resize to template size
        spec = cv.resize(spec, (self.config.template_width, self.config.template_height))
        
        # Apply enhancement
        if self.config.apply_enhancement:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            spec = clahe.apply(spec)
        
        # Convert to RGB
        spec_rgb = cv.cvtColor(spec, cv.COLOR_GRAY2RGB)
        
        return spec_rgb


class PatternTrainer:
    """
    Main class for automatic pattern-based template training.
    """
    
    def __init__(self, config: PatternTrainingConfig = None):
        self.config = config or PatternTrainingConfig()
        self.burst_detector = BurstDetector(self.config.burst_config)
        self.pattern_detector = PatternDetector(
            min_repetitions=self.config.min_repetitions,
            timing_tolerance=self.config.timing_tolerance
        )
        self.cycle_extractor = CycleExtractor(self.config)
    
    def train_from_iq_samples(self, samples: np.ndarray, sample_rate: float,
                             center_freq: float = 0, label: str = "auto") -> Optional[Dict]:
        """
        Complete training pipeline from IQ samples.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            label: Label for the pattern
            
        Returns:
            Dictionary with template info and image, or None if no pattern found
        """
        # Step 1: Detect bursts
        bursts = self.burst_detector.detect_bursts(samples, sample_rate, center_freq)
        
        if len(bursts) < self.config.min_repetitions:
            return None
        
        # Step 2: Detect patterns
        patterns = self.pattern_detector.find_patterns(bursts, sample_rate)
        
        if len(patterns) == 0:
            return None
        
        # Use best pattern (highest confidence)
        best_pattern = max(patterns, key=lambda p: p.get('confidence', 0))
        
        if best_pattern['confidence'] < self.config.min_pattern_confidence:
            return None
        
        # Step 3: Extract cycle boundaries
        period_samples = int(best_pattern['period'] * sample_rate)
        best_pattern['period_samples'] = period_samples
        
        cycles = self.cycle_extractor.find_cycle_boundaries(bursts, best_pattern)
        
        if len(cycles) == 0:
            return None
        
        # Step 4: Extract best cycle
        best_cycle_samples = self.cycle_extractor.extract_best_cycle(samples, cycles, sample_rate)
        
        if best_cycle_samples is None:
            return None
        
        # Step 5: Convert to template image
        template_image = self.cycle_extractor.cycle_to_spectrogram_image(
            best_cycle_samples, sample_rate
        )
        
        if template_image is None:
            return None
        
        # Generate label if auto
        if label == "auto":
            label = f"Pattern_{int(best_pattern['center_freq']/1e6)}MHz_{int(best_pattern['period']*1000)}ms"
        
        template_info = {
            'label': label,
            'image': template_image,
            'pattern': best_pattern,
            'num_bursts': len(bursts),
            'num_cycles': len(cycles),
            'confidence': best_pattern['confidence'],
            'center_freq': best_pattern['center_freq'],
            'bandwidth': best_pattern['bandwidth'],
            'period': best_pattern['period'],
            'burst_duration': best_pattern['burst_duration']
        }
        
        return template_info
    
    def train_from_spectrogram_image(self, image: np.ndarray, 
                                    label: str = "auto") -> Optional[Dict]:
        """
        Extract patterns from existing spectrogram image.
        
        This analyzes the image to find repeating visual patterns and
        extracts one cycle as a template.
        
        Args:
            image: RGB or grayscale spectrogram image
            label: Label for the pattern
            
        Returns:
            Dictionary with template info, or None if no pattern found
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect repeating patterns using autocorrelation
        height, width = gray.shape
        
        # Compute horizontal autocorrelation to find period
        row_sums = np.mean(gray, axis=0)
        row_sums = row_sums - np.mean(row_sums)
        
        # Autocorrelation
        autocorr = np.correlate(row_sums, row_sums, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (repeating patterns)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr, height=0.3*np.max(autocorr), distance=10)
        
        if len(peaks) < self.config.min_repetitions - 1:
            return None
        
        # Estimate period from first peak
        period_pixels = peaks[0] if len(peaks) > 0 else width // 3
        
        # Extract one cycle
        if period_pixels < 10 or period_pixels > width - 10:
            return None
        
        # Find best starting position (highest energy)
        best_start = 0
        best_energy = 0
        
        for start in range(0, width - period_pixels, period_pixels // 4):
            cycle = gray[:, start:start+period_pixels]
            energy = np.sum(cycle**2)
            if energy > best_energy:
                best_energy = energy
                best_start = start
        
        # Extract template
        template = gray[:, best_start:best_start+period_pixels]
        
        # Resize to standard template size
        template = cv.resize(template, (self.config.template_width, self.config.template_height))
        
        # Apply enhancement
        if self.config.apply_enhancement:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            template = clahe.apply(template)
        
        # Convert to RGB
        template_rgb = cv.cvtColor(template, cv.COLOR_GRAY2RGB)
        
        if label == "auto":
            label = f"ImagePattern_{period_pixels}px"
        
        template_info = {
            'label': label,
            'image': template_rgb,
            'period_pixels': period_pixels,
            'num_repetitions': len(peaks) + 1,
            'confidence': min(1.0, len(peaks) / self.config.min_repetitions)
        }
        
        return template_info


def auto_label_from_patterns(spectrogram_image: np.ndarray, 
                            existing_templates: Dict,
                            min_confidence: float = 0.7) -> List[Dict]:
    """
    Enhanced auto-labeling that looks for pattern cycles rather than just matches.
    
    This is the new auto-label method that:
    1. Identifies repeating patterns in unidentified signals
    2. Extracts one cycle per pattern
    3. Adds as new template if not matching existing ones
    
    Args:
        spectrogram_image: Full spectrogram image to analyze
        existing_templates: Dictionary of existing templates
        min_confidence: Minimum confidence for pattern detection
        
    Returns:
        List of new template dictionaries
    """
    config = PatternTrainingConfig()
    config.min_pattern_confidence = min_confidence
    
    trainer = PatternTrainer(config)
    
    # Try to extract pattern from image
    template_info = trainer.train_from_spectrogram_image(spectrogram_image)
    
    if template_info is None:
        return []
    
    if template_info['confidence'] < min_confidence:
        return []
    
    # Check if similar template already exists
    new_template = template_info['image']
    
    for template_name, template_data in existing_templates.items():
        # Simple similarity check using normalized cross-correlation
        if 'image' in template_data:
            existing = cv.cvtColor(template_data['image'], cv.COLOR_RGB2GRAY)
            new_gray = cv.cvtColor(new_template, cv.COLOR_RGB2GRAY)
            
            # Resize to match
            if existing.shape != new_gray.shape:
                existing = cv.resize(existing, (new_gray.shape[1], new_gray.shape[0]))
            
            # Compute similarity
            result = cv.matchTemplate(new_gray, existing, cv.TM_CCORR_NORMED)
            similarity = np.max(result)
            
            if similarity > 0.85:  # Very similar
                return []  # Don't create duplicate
    
    return [template_info]
