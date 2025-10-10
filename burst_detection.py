"""
Burst Detection Module for Ad Astra Drone Detection

This module implements advanced burst detection algorithms adapted from annotation-scripts
to detect drone signals based on temporal patterns and frequency characteristics rather
than just template matching.

Key improvements:
1. Time-domain burst detection with power thresholding
2. Frequency-domain bandwidth and center frequency estimation
3. Pattern recognition based on burst repetition and timing
4. Adaptive noise floor estimation
"""

import numpy as np
import cv2 as cv
from scipy import fft
from scipy.signal import medfilt, convolve2d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class BurstDetectionConfig:
    """Configuration for burst detection algorithm"""
    # Time domain thresholds
    time_threshold_dB: float = 5.0  # dB above noise floor for burst detection
    freq_threshold_dB: Optional[float] = None  # Frequency domain threshold (auto if None)
    
    # Noise estimation
    navg_per_MHz: int = 1  # Averaging relative to sample rate
    percentile: int = 10  # Percentile for noise estimation
    
    # FFT settings
    rbw: int = 100000  # Resolution bandwidth in Hz
    
    # Burst constraints
    min_duration: float = 1.5e-5  # seconds
    max_duration: float = 10.0  # seconds
    min_bandwidth: float = 1e6  # Hz
    max_bandwidth: float = 1e8  # Hz
    
    # Merge settings
    merge_time_overlap: float = 0.01  # Relative overlap for merging
    merge_freq_overlap: float = -0.08  # Negative allows small gaps
    
    # Spectrogram settings (for spectrogram-based detection)
    spectrogram_avg_len: Tuple[int, int] = (1, 3)  # time x freq averaging
    spectrogram_navg: int = 2  # Number of averaging iterations
    
    verbose: bool = False


class BurstDetector:
    """
    Advanced burst detection for drone signals.
    
    Uses two-stage approach:
    1. Time-domain burst detection with adaptive noise floor
    2. Frequency-domain CF/BW estimation per burst
    """
    
    def __init__(self, config: BurstDetectionConfig):
        self.config = config
    
    def detect_bursts_time_domain(self, samples: np.ndarray, sample_rate: float) -> List[Dict]:
        """
        Detect bursts in time domain using power thresholding.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            
        Returns:
            List of burst dictionaries with start, end indices
        """
        # Estimate noise level using adaptive averaging
        N = int(self.config.navg_per_MHz * sample_rate / 1e6)
        if N < 1:
            N = 1
            
        # Calculate power in dB
        num_segments = len(samples) // N
        if num_segments < 1:
            return []
            
        pwr = np.abs(samples[:num_segments * N])**2
        pwr = np.sum(np.reshape(pwr, (-1, N)), axis=1) / N
        pwr = 10 * np.log10(pwr + 1e-20)  # Avoid log(0)
        
        # Smooth with median filter
        pwr = medfilt(pwr, 31)
        
        # Estimate noise floor
        noise_floor = np.percentile(pwr, self.config.percentile)
        threshold = noise_floor + self.config.time_threshold_dB
        
        # Detect bursts as regions above threshold
        peaks = (pwr >= threshold).astype(np.int8)
        peaks = np.concatenate([np.array([0]), peaks, np.array([0])])
        edges = peaks[1:] - peaks[:-1]
        
        burst_starts = np.nonzero(edges == 1)[0]
        burst_ends = np.nonzero(edges == -1)[0]
        
        if len(burst_starts) != len(burst_ends):
            # Handle edge cases
            min_len = min(len(burst_starts), len(burst_ends))
            burst_starts = burst_starts[:min_len]
            burst_ends = burst_ends[:min_len]
        
        # Convert to sample indices with padding
        bursts = []
        for start, end in zip(burst_starts, burst_ends):
            start_idx = int(max(0, start - 1) * N)
            end_idx = int(min(len(samples), end + 1) * N)
            
            duration = (end_idx - start_idx) / sample_rate
            
            # Filter by duration
            if self.config.min_duration <= duration <= self.config.max_duration:
                bursts.append({
                    'start': start_idx,
                    'end': end_idx,
                    'duration': duration,
                    'noise_floor': noise_floor
                })
        
        return bursts
    
    def estimate_burst_spectrum(self, samples: np.ndarray, sample_rate: float, 
                                center_freq: float, noise_floor_dB: float) -> Optional[Dict]:
        """
        Estimate center frequency and bandwidth of a burst.
        
        Args:
            samples: IQ samples of the burst
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            noise_floor_dB: Noise floor estimate in dB/Hz
            
        Returns:
            Dictionary with f_low, f_high, bandwidth or None if invalid
        """
        fft_size = int(sample_rate // self.config.rbw)
        if fft_size < 4:
            fft_size = 4
            
        stp_size = int(fft_size // 2)
        num_fft = max(1, int((len(samples) - fft_size) // stp_size + 1))
        
        # Calculate averaged spectrum
        spectrum = np.zeros(fft_size, dtype=np.float32)
        win = np.hanning(fft_size).astype(np.complex64)
        win /= np.sqrt((np.abs(win)**2).sum())
        
        for i in range(num_fft):
            start_idx = i * stp_size
            end_idx = start_idx + fft_size
            if end_idx > len(samples):
                break
            spectrum += np.abs(fft.fftshift(fft.fft(win * samples[start_idx:end_idx])))
        
        spectrum /= num_fft * np.sqrt(sample_rate)
        spectrum[spectrum <= 0] = 1e-20
        spectrum = 20 * np.log10(spectrum)
        
        # Determine threshold
        if self.config.freq_threshold_dB is None:
            # Adaptive: midpoint between peak and noise
            threshold = (np.max(spectrum) + noise_floor_dB) / 2
        else:
            threshold = noise_floor_dB + self.config.freq_threshold_dB
        
        # Find bandwidth
        mask = np.nonzero(spectrum >= threshold)[0]
        if len(mask) == 0:
            return None
        
        # Calculate frequencies
        freq_bins = fft.fftshift(fft.fftfreq(fft_size, d=1/sample_rate))
        f_low = int(freq_bins[mask[0]]) + center_freq
        f_high = int(freq_bins[mask[-1]]) + center_freq
        bandwidth = f_high - f_low
        
        # Filter by bandwidth
        if not (self.config.min_bandwidth <= bandwidth <= self.config.max_bandwidth):
            return None
        
        return {
            'f_low': f_low,
            'f_high': f_high,
            'bandwidth': bandwidth,
            'center_freq': (f_low + f_high) / 2
        }
    
    def detect_bursts(self, samples: np.ndarray, sample_rate: float, 
                     center_freq: float = 0) -> List[Dict]:
        """
        Complete burst detection: time domain + frequency estimation.
        
        Args:
            samples: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            
        Returns:
            List of detected bursts with time and frequency info
        """
        # Stage 1: Time domain burst detection
        time_bursts = self.detect_bursts_time_domain(samples, sample_rate)
        
        # Stage 2: Frequency estimation for each burst
        detected_bursts = []
        for burst in time_bursts:
            burst_samples = samples[burst['start']:burst['end']]
            
            freq_info = self.estimate_burst_spectrum(
                burst_samples, sample_rate, center_freq, burst['noise_floor']
            )
            
            if freq_info is not None:
                detected_bursts.append({
                    'start_sample': burst['start'],
                    'length_samples': burst['end'] - burst['start'],
                    'duration': burst['duration'],
                    'f_low': freq_info['f_low'],
                    'f_high': freq_info['f_high'],
                    'bandwidth': freq_info['bandwidth'],
                    'center_freq': freq_info['center_freq']
                })
        
        # Merge overlapping bursts
        detected_bursts = self._merge_bursts(detected_bursts, sample_rate)
        
        return detected_bursts
    
    def _merge_bursts(self, bursts: List[Dict], sample_rate: float) -> List[Dict]:
        """Merge overlapping or nearby bursts"""
        if len(bursts) < 2:
            return bursts
        
        bursts.sort(key=lambda x: x['start_sample'])
        merged = []
        
        for burst in bursts:
            if not merged:
                merged.append(burst)
                continue
            
            last = merged[-1]
            
            # Check for overlap/proximity
            time_overlap_samples = max(burst['length_samples'], last['length_samples']) * self.config.merge_time_overlap
            freq_overlap_hz = max(burst['bandwidth'], last['bandwidth']) * self.config.merge_freq_overlap
            
            last_end = last['start_sample'] + last['length_samples']
            burst_start = burst['start_sample']
            
            time_overlaps = (last_end - burst_start >= time_overlap_samples)
            freq_overlaps = (last['f_high'] - burst['f_low'] >= freq_overlap_hz and
                           burst['f_high'] - last['f_low'] >= freq_overlap_hz)
            
            if time_overlaps and freq_overlaps:
                # Merge
                new_start = min(last['start_sample'], burst['start_sample'])
                new_end = max(last_end, burst_start + burst['length_samples'])
                last['start_sample'] = new_start
                last['length_samples'] = new_end - new_start
                last['duration'] = last['length_samples'] / sample_rate
                last['f_low'] = min(last['f_low'], burst['f_low'])
                last['f_high'] = max(last['f_high'], burst['f_high'])
                last['bandwidth'] = last['f_high'] - last['f_low']
                last['center_freq'] = (last['f_low'] + last['f_high']) / 2
            else:
                merged.append(burst)
        
        return merged


class PatternDetector:
    """
    Detects repeating patterns in burst sequences to identify drone signatures.
    """
    
    def __init__(self, min_repetitions: int = 3, timing_tolerance: float = 0.1):
        """
        Args:
            min_repetitions: Minimum pattern repetitions to consider valid
            timing_tolerance: Relative tolerance for timing variations (0.1 = 10%)
        """
        self.min_repetitions = min_repetitions
        self.timing_tolerance = timing_tolerance
    
    def find_patterns(self, bursts: List[Dict], sample_rate: float) -> List[Dict]:
        """
        Find repeating patterns in burst sequence.
        
        Returns:
            List of detected patterns with timing and frequency characteristics
        """
        if len(bursts) < self.min_repetitions:
            return []
        
        patterns = []
        
        # Calculate inter-burst intervals
        intervals = []
        for i in range(len(bursts) - 1):
            interval = (bursts[i+1]['start_sample'] - 
                       (bursts[i]['start_sample'] + bursts[i]['length_samples'])) / sample_rate
            intervals.append(interval)
        
        # Look for repeating intervals
        if len(intervals) >= self.min_repetitions - 1:
            # Simple pattern: consistent spacing
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if std_interval / mean_interval < self.timing_tolerance:
                # Found consistent pattern
                patterns.append({
                    'type': 'periodic',
                    'period': mean_interval,
                    'burst_duration': np.mean([b['duration'] for b in bursts]),
                    'center_freq': np.mean([b['center_freq'] for b in bursts]),
                    'bandwidth': np.mean([b['bandwidth'] for b in bursts]),
                    'num_bursts': len(bursts),
                    'confidence': 1.0 - (std_interval / mean_interval)
                })
        
        return patterns


def detect_bursts_in_spectrogram(image: np.ndarray, sample_rate: float, 
                                 center_freq: float, config: BurstDetectionConfig) -> List[Dict]:
    """
    Detect bursts directly from spectrogram image (PNG).
    
    This is the bridge function that allows burst detection on
    already-generated spectrogram images.
    """
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold
    noise_level = np.percentile(gray, config.percentile)
    threshold_level = noise_level + config.time_threshold_dB
    
    _, binary = cv.threshold(gray.astype(np.float32), threshold_level, 255, cv.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    # Find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to bursts
    height, width = image.shape[:2]
    bursts = []
    
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        
        # Map image coordinates to signal domain
        # Assuming image represents full time/frequency span
        start_time = x / width  # Normalized
        duration = w / width
        f_low_norm = (height - (y + h)) / height
        f_high_norm = (height - y) / height
        
        # Scale to actual values (requires knowledge of time/freq span)
        # This is approximate and should be adjusted based on actual spectrogram parameters
        burst = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'start_normalized': start_time,
            'duration_normalized': duration,
            'f_low_normalized': f_low_norm,
            'f_high_normalized': f_high_norm
        }
        
        # Filter by size
        if w >= 3 and h >= 3:  # Minimum size in pixels
            bursts.append(burst)
    
    return bursts
