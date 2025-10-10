#!/usr/bin/env python3
"""
High-throughput live feed processing for drone detection
Supports 1 Gbps datastreams and GB-scale ZIP archives
Processes PNG images for drone pattern detection
"""

import os
import time
import threading
import queue
import zipfile
import io
import json
import numpy as np
import cv2 as cv
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, List, Dict, Any, Optional, Callable, Tuple
import requests
import socket
from dataclasses import dataclass
from pathlib import Path

from integration_optimized import LiveFeedBurstIntegration

# Optional imports for advanced features
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

@dataclass
class LiveFeedConfig:
    """Configuration for live feed processing"""
    max_concurrent_streams: int = 4
    frame_skip_factor: int = 10  # Starting point for adaptive skipping
    quality_vs_speed: float = 0.7
    memory_limit_gb: int = 8
    throughput_target_mbps: int = 100
    enable_caching: bool = True
    buffer_size_mb: int = 100
    use_burst_detection: bool = True  # Enable burst detection (AdAstra default)
    fast_mode: bool = True  # Use fast mode for high FPS (template-only detection)
    # Adaptive frame skipping (AdAstra)
    enable_adaptive_skip: bool = True  # Auto-adjust frame_skip_factor based on performance
    min_skip_factor: int = 1  # Best case: process every frame
    max_skip_factor: int = 50  # Emergency mode: process 1 in 50 frames
    target_latency_ms: float = 50.0  # Target processing time per frame
    target_queue_usage: float = 0.6  # Target queue utilization (60%)
    # Multi-drone temporal tracking
    enable_temporal_tracking: bool = False  # Enable for single-drone, disable for multi-drone
    temporal_search_region: int = 100  # Size of temporal search region (pixels)
    temporal_full_search_interval: int = 50  # Do full search every N frames

class HighThroughputProcessor:
    """High-performance processor for live data streams with AdAstra adaptive optimization"""
    
    def __init__(self, config: LiveFeedConfig, detection_callback: Callable = None):
        self.config = config
        self.detection_callback = detection_callback
        self.active = False
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections_found': 0,
            'avg_processing_time': 0,
            'throughput_mbps': 0,
            'start_time': None,
            'bytes_processed': 0,
            'png_files_processed': 0,
            'frames_skipped': 0,
            'adaptive_adjustments': 0
        }
        self.processing_queue = queue.Queue(maxsize=config.buffer_size_mb)
        self.result_queue = queue.Queue()
        self.workers = []
        
        # AdAstra: Adaptive frame skipping
        self.performance_window = []  # Rolling window of recent performance metrics
        self.current_skip_factor = config.frame_skip_factor
        
        # Initialize burst detection integration
        self.burst_integration = LiveFeedBurstIntegration(
            enable_burst_detection=config.use_burst_detection,
            use_fast_mode=config.fast_mode
        )
        
        # Initialize multi-drone temporal tracker
        self.temporal_tracker = MultiDroneTemporalTracker(
            enable_temporal=config.enable_temporal_tracking,
            search_region_size=config.temporal_search_region
        )
        self.temporal_tracker.full_search_interval = config.temporal_full_search_interval
        
    def start_processing(self):
        """Start the high-throughput processing pipeline"""
        self.active = True
        self.stats['start_time'] = time.time()
        
        # Start worker threads
        for i in range(self.config.max_concurrent_streams):
            worker = threading.Thread(target=self._processing_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def stop_processing(self):
        """Stop all processing"""
        self.active = False
        
        # Clear queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def _processing_worker(self):
        """Worker thread for processing individual frames"""
        
        while self.active:
            try:
                # Get work item from queue
                work_item = self.processing_queue.get(timeout=1.0)
                if work_item is None:
                    break
                
                frame_data, templates, detection_params, frame_id = work_item
                
                # Process frame
                start_time = time.time()
                
                # Auto-detect file type and convert to image
                image = self._convert_to_image(frame_data, frame_id)
                
                if image is None:
                    continue
                
                # Run detection using burst integration (handles both template and burst detection)
                detections = []
                
                try:
                    # Use burst integration for unified detection
                    results = self.burst_integration.process_image_with_bursts(
                        image=image,
                        templates=templates,
                        threshold=detection_params.get('threshold', 0.7),
                        method=detection_params.get('method', cv.TM_CCOEFF_NORMED),
                        parallel_config=detection_params.get('parallel_config')
                    )
                    
                    # Add frame and timestamp info to matches
                    for match in results.get('template_matches', []):
                        match['frame_id'] = frame_id
                        match['timestamp'] = time.time()
                        match['processing_time_ms'] = (time.time() - start_time) * 1000
                        detections.append(match)
                    
                    # Add burst detections if available
                    for burst in results.get('bursts', []):
                        burst['frame_id'] = frame_id
                        burst['timestamp'] = time.time()
                        burst['processing_time_ms'] = (time.time() - start_time) * 1000
                        burst['detection_type'] = 'burst'
                        # Add burst characteristics for temporal tracking
                        burst['burst_info'] = {
                            'center_freq': burst.get('center_freq'),
                            'bandwidth': burst.get('bandwidth'),
                            'snr': burst.get('snr')
                        }
                        detections.append(burst)
                    
                    # Multi-drone temporal tracking: Update trackers and add tracker IDs
                    if self.config.enable_temporal_tracking:
                        detections = self.temporal_tracker.update_trackers(
                            detections, 
                            image.shape
                        )
                        
                except Exception as e:
                    pass
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['processed_frames'] += 1
                self.stats['detections_found'] += len(detections)
                
                # Update average processing time
                if self.stats['processed_frames'] > 0:
                    self.stats['avg_processing_time'] = (
                        (self.stats['avg_processing_time'] * (self.stats['processed_frames'] - 1) + 
                         processing_time * 1000) / self.stats['processed_frames']
                    )
                
                # AdAstra: Update adaptive skip factor based on performance
                self._update_adaptive_skip()
                
                # Put results in result queue
                if detections and self.detection_callback:
                    self.result_queue.put(detections)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                continue
    
    def _convert_to_image(self, frame_data: Any, frame_id: str) -> Optional[np.ndarray]:
        """Convert frame data to image"""
        
        # Case 1: Already a numpy array (image)
        if isinstance(frame_data, np.ndarray):
            return frame_data
        
        # Case 3: Bytes (PNG/JPEG image data)
        if isinstance(frame_data, bytes):
            try:
                # Assume it's an image
                image_array = np.frombuffer(frame_data, dtype=np.uint8)
                image = cv.imdecode(image_array, cv.IMREAD_COLOR)
                if image is not None:
                    self.stats['png_files_processed'] += 1
                return image
            except Exception as e:
                return None
        
        # Case 4: File path (string)
        if isinstance(frame_data, (str, Path)):
            file_path = str(frame_data)
            
            # Otherwise, assume it's an image file
            try:
                image = cv.imread(file_path, cv.IMREAD_COLOR)
                if image is not None:
                    self.stats['png_files_processed'] += 1
                return image
            except Exception as e:
                return None
        return None
    
    def _update_adaptive_skip(self):
        """
        AdAstra: Adaptive frame skipping based on real-time performance
        Adjusts frame_skip_factor dynamically based on processing time and queue depth
        """
        if not self.config.enable_adaptive_skip:
            return
        
        # Need at least 10 samples for meaningful statistics
        if self.stats['processed_frames'] < 10:
            return
        
        # Add current metrics to rolling window
        queue_depth = self.processing_queue.qsize()
        queue_capacity = self.config.buffer_size_mb
        queue_usage = queue_depth / max(queue_capacity, 1)
        
        self.performance_window.append({
            'time': self.stats['avg_processing_time'],
            'queue': queue_usage
        })
        
        # Keep only last 100 samples (rolling window)
        if len(self.performance_window) > 100:
            self.performance_window.pop(0)
        
        # Update every 10 frames
        if self.stats['processed_frames'] % 10 != 0:
            return
        
        # Calculate average metrics over window
        avg_time = np.mean([p['time'] for p in self.performance_window])
        avg_queue = np.mean([p['queue'] for p in self.performance_window])
        
        old_skip = self.current_skip_factor
        
        # Adaptation logic based on performance zones
        if avg_time > self.config.target_latency_ms * 2 or avg_queue > 0.9:
            # Zone 5: Critical - Emergency mode
            self.current_skip_factor = min(self.config.max_skip_factor, self.current_skip_factor + 5)
        elif avg_time > self.config.target_latency_ms * 1.5 or avg_queue > 0.8:
            # Zone 4: Struggling - Reduce load
            self.current_skip_factor = min(self.config.max_skip_factor, self.current_skip_factor + 3)
        elif avg_time > self.config.target_latency_ms * 1.2 or avg_queue > 0.7:
            # Zone 3: Acceptable - Slight reduction
            self.current_skip_factor = min(self.config.max_skip_factor, self.current_skip_factor + 1)
        elif avg_time < self.config.target_latency_ms * 0.6 and avg_queue < 0.3:
            # Zone 1: Excellent - Process more frames
            self.current_skip_factor = max(self.config.min_skip_factor, self.current_skip_factor - 1)
        elif avg_time < self.config.target_latency_ms * 0.8 and avg_queue < 0.5:
            # Zone 2: Good - Slowly increase processing
            if self.stats['processed_frames'] % 50 == 0:  # More conservative
                self.current_skip_factor = max(self.config.min_skip_factor, self.current_skip_factor - 1)
        
        # Log adjustments
        if old_skip != self.current_skip_factor:
            self.stats['adaptive_adjustments'] += 1
    
    def get_adaptive_stats(self) -> Dict:
        """Get current adaptive skip statistics"""
        stats = {
            'current_skip_factor': self.current_skip_factor,
            'target_latency_ms': self.config.target_latency_ms,
            'avg_processing_time': self.stats['avg_processing_time'],
            'queue_usage': self.processing_queue.qsize() / max(self.config.buffer_size_mb, 1),
            'total_adjustments': self.stats['adaptive_adjustments'],
            'frames_skipped': self.stats['frames_skipped'],
            'skip_rate': self.stats['frames_skipped'] / max(self.stats['total_frames'], 1) if self.stats['total_frames'] > 0 else 0
        }
        
        # Add temporal tracking stats if enabled
        if self.config.enable_temporal_tracking:
            stats['temporal_tracking'] = self.temporal_tracker.get_stats()
        
        return stats


@dataclass
class DroneTracker:
    """Individual drone tracker with signature matching"""
    tracker_id: int
    last_position: Tuple[int, int]  # (x, y)
    template_signature: Optional[np.ndarray] = None
    burst_characteristics: Optional[Dict] = None  # center_freq, bandwidth, snr
    match_history: List[float] = None  # Recent match scores
    frames_since_seen: int = 0
    total_detections: int = 0
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.match_history is None:
            self.match_history = []
    
    def update_position(self, new_pos: Tuple[int, int], match_score: float):
        """Update tracker with new detection"""
        self.last_position = new_pos
        self.match_history.append(match_score)
        if len(self.match_history) > 10:
            self.match_history.pop(0)
        self.frames_since_seen = 0
        self.total_detections += 1
        # Increase confidence with consistent detections
        self.confidence = min(1.0, self.confidence + 0.05)
    
    def mark_missed(self):
        """Mark that tracker didn't find its drone this frame"""
        self.frames_since_seen += 1
        # Decrease confidence when missing
        self.confidence = max(0.0, self.confidence - 0.1)
    
    def get_search_region(self, image_shape: Tuple[int, int], region_size: int = 100) -> Tuple[int, int, int, int]:
        """Get bounding box for temporal search region (x1, y1, x2, y2)"""
        height, width = image_shape[:2]
        x, y = self.last_position
        
        half_size = region_size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(width, x + half_size)
        y2 = min(height, y + half_size)
        
        return (x1, y1, x2, y2)
    
    def matches_signature(self, detection: Dict, threshold: float = 0.7) -> bool:
        """Check if detection matches this tracker's drone signature"""
        # Template matching score
        if detection.get('match_score', 0) < threshold:
            return False
        
        # If we have burst characteristics, check them too
        if self.burst_characteristics and 'burst_info' in detection:
            burst = detection['burst_info']
            
            # Check center frequency (within 5%)
            if 'center_freq' in self.burst_characteristics and 'center_freq' in burst:
                freq_diff = abs(burst['center_freq'] - self.burst_characteristics['center_freq'])
                if freq_diff > self.burst_characteristics['center_freq'] * 0.05:
                    return False
            
            # Check bandwidth (within 20%)
            if 'bandwidth' in self.burst_characteristics and 'bandwidth' in burst:
                bw_diff = abs(burst['bandwidth'] - self.burst_characteristics['bandwidth'])
                if bw_diff > self.burst_characteristics['bandwidth'] * 0.2:
                    return False
        
        return True
    
    def is_alive(self, max_missed_frames: int = 30) -> bool:
        """Check if tracker should be kept alive"""
        return self.frames_since_seen < max_missed_frames


class MultiDroneTemporalTracker:
    """
    Multi-object temporal tracker using template matching + burst characteristics
    Maintains drone identities across frames for 5-10x speedup
    """
    
    def __init__(self, enable_temporal: bool = False, search_region_size: int = 100):
        self.enable_temporal = enable_temporal
        self.search_region_size = search_region_size
        self.trackers: List[DroneTracker] = []
        self.next_tracker_id = 0
        self.frames_since_full_search = 0
        self.full_search_interval = 50  # Do full search every N frames
        self.stats = {
            'temporal_searches': 0,
            'full_searches': 0,
            'trackers_spawned': 0,
            'trackers_lost': 0,
            'identity_preservations': 0
        }
    
    def should_do_full_search(self) -> bool:
        """Decide if we should do full-frame search"""
        if not self.enable_temporal:
            return True
        
        # Always do full search periodically to find new drones
        if self.frames_since_full_search >= self.full_search_interval:
            return True
        
        # If we have no active trackers, do full search
        if not self.trackers:
            return True
        
        # If all trackers are low confidence, do full search
        if all(t.confidence < 0.3 for t in self.trackers):
            return True
        
        return False
    
    def get_search_regions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
        """Get search regions for all active trackers. Returns [(x1,y1,x2,y2,tracker_id), ...]"""
        regions = []
        for tracker in self.trackers:
            if tracker.is_alive():
                x1, y1, x2, y2 = tracker.get_search_region(image_shape, self.search_region_size)
                regions.append((x1, y1, x2, y2, tracker.tracker_id))
        return regions
    
    def update_trackers(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Update trackers with new detections, maintain identities
        Returns detections with added 'tracker_id' field
        """
        if not self.enable_temporal:
            # Passthrough mode - no tracking
            return detections
        
        # Check if this was a full search frame
        is_full_search = self.should_do_full_search()
        
        if is_full_search:
            self.stats['full_searches'] += 1
            self.frames_since_full_search = 0
        else:
            self.stats['temporal_searches'] += 1
            self.frames_since_full_search += 1
        
        # Hungarian algorithm simplified: greedy matching
        unmatched_detections = detections.copy()
        matched_trackers = set()
        
        # Try to match detections to existing trackers
        for tracker in self.trackers[:]:
            if not tracker.is_alive():
                self.trackers.remove(tracker)
                self.stats['trackers_lost'] += 1
                continue
            
            best_match = None
            best_score = 0
            best_idx = -1
            
            for idx, detection in enumerate(unmatched_detections):
                # Check if detection is in tracker's search region
                det_x = detection.get('x', detection.get('center_x', 0))
                det_y = detection.get('y', detection.get('center_y', 0))
                x1, y1, x2, y2 = tracker.get_search_region(image_shape, self.search_region_size)
                
                if not (x1 <= det_x <= x2 and y1 <= det_y <= y2):
                    continue  # Detection outside search region
                
                # Check signature match
                if not tracker.matches_signature(detection):
                    continue
                
                # Calculate match quality (distance + score)
                dist = np.sqrt((det_x - tracker.last_position[0])**2 + 
                              (det_y - tracker.last_position[1])**2)
                match_score = detection.get('match_score', 0.5)
                
                # Combined score: prefer closer + higher match score
                combined_score = match_score * 0.7 - (dist / 100) * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = detection
                    best_idx = idx
            
            if best_match:
                # Matched! Update tracker
                det_x = best_match.get('x', best_match.get('center_x', 0))
                det_y = best_match.get('y', best_match.get('center_y', 0))
                tracker.update_position((det_x, det_y), best_match.get('match_score', 0.5))
                best_match['tracker_id'] = tracker.tracker_id
                best_match['tracker_confidence'] = tracker.confidence
                matched_trackers.add(tracker.tracker_id)
                unmatched_detections.pop(best_idx)
                self.stats['identity_preservations'] += 1
            else:
                # No match found
                tracker.mark_missed()
        
        # Spawn new trackers for unmatched detections (new drones)
        for detection in unmatched_detections:
            det_x = detection.get('x', detection.get('center_x', 0))
            det_y = detection.get('y', detection.get('center_y', 0))
            
            # Extract burst characteristics if available
            burst_chars = None
            if 'burst_info' in detection:
                burst = detection['burst_info']
                burst_chars = {
                    'center_freq': burst.get('center_freq'),
                    'bandwidth': burst.get('bandwidth'),
                    'snr': burst.get('snr')
                }
            
            new_tracker = DroneTracker(
                tracker_id=self.next_tracker_id,
                last_position=(det_x, det_y),
                burst_characteristics=burst_chars,
                total_detections=1
            )
            self.trackers.append(new_tracker)
            detection['tracker_id'] = self.next_tracker_id
            detection['tracker_confidence'] = 1.0
            self.next_tracker_id += 1
            self.stats['trackers_spawned'] += 1
        
        return detections
    
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            **self.stats,
            'active_trackers': len([t for t in self.trackers if t.is_alive()]),
            'total_trackers': len(self.trackers),
            'avg_confidence': np.mean([t.confidence for t in self.trackers]) if self.trackers else 0
        }


class ZipStreamProcessor:
    """Memory-efficient ZIP archive processor for GB-scale files"""
    
    def __init__(self, processor: HighThroughputProcessor):
        self.processor = processor
        
    def process_zip_stream(self, zip_file_path: str, templates: Dict, detection_params: Dict) -> Generator[Dict, None, None]:
        """Process ZIP archive in streaming mode without full extraction"""
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                file_list = [f for f in zip_ref.namelist() 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                
                self.processor.stats['total_frames'] = len(file_list)
                
                for file_info in file_list:
                    if not self.processor.active:
                        break
                    
                    try:
                        # Extract file data to memory
                        with zip_ref.open(file_info) as file_data:
                            image_bytes = file_data.read()
                            
                        # Update bytes processed
                        self.processor.stats['bytes_processed'] += len(image_bytes)
                        
                        # Calculate throughput
                        if self.processor.stats['start_time']:
                            elapsed = time.time() - self.processor.stats['start_time']
                            if elapsed > 0:
                                self.processor.stats['throughput_mbps'] = (
                                    self.processor.stats['bytes_processed'] / (1024 * 1024) / elapsed * 8
                                )
                        
                        # Add to processing queue
                        work_item = (image_bytes, templates, detection_params, file_info)
                        
                        # Handle queue overflow with backpressure
                        try:
                            self.processor.processing_queue.put(work_item, timeout=5.0)
                        except queue.Full:
                            continue
                        
                        # Yield progress update
                        yield {
                            'type': 'progress',
                            'processed': self.processor.stats['processed_frames'],
                            'total': self.processor.stats['total_frames'],
                            'file': file_info,
                            'throughput_mbps': self.processor.stats['throughput_mbps']
                        }
                        
                        # Check for results
                        while not self.processor.result_queue.empty():
                            try:
                                detections = self.processor.result_queue.get_nowait()
                                yield {
                                    'type': 'detections',
                                    'detections': detections,
                                    'file': file_info
                                }
                            except queue.Empty:
                                break
                        
                        # Apply frame skipping if configured
                        # Use dynamic skip factor from AdAstra if adaptive mode enabled
                        current_skip = self.processor.current_skip_factor if self.processor.config.enable_adaptive_skip else self.processor.config.frame_skip_factor
                        
                        if current_skip > 1:
                            # Count skipped frames for statistics
                            frames_to_skip = current_skip - 1
                            self.processor.stats['frames_skipped'] += frames_to_skip
                            
                            # Skip ahead in the file list
                            for _ in range(frames_to_skip):
                                if file_list.index(file_info) + 1 < len(file_list):
                                    file_list.pop(file_list.index(file_info) + 1)
                    
                    except Exception as e:
                        continue
                        
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e)
            }

class NetworkStreamProcessor:
    """High-throughput network stream processor"""
    
    def __init__(self, processor: HighThroughputProcessor):
        self.processor = processor
        
    def process_http_stream(self, url: str, templates: Dict, detection_params: Dict) -> Generator[Dict, None, None]:
        """Process HTTP/HTTPS streaming data"""
        
        try:
            headers = {
                'User-Agent': 'DroneDetection-LiveFeed/1.0',
                'Accept': 'image/*,application/json,multipart/*'
            }
            
            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                buffer = b''
                frame_count = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if not self.processor.active:
                        break
                    
                    buffer += chunk
                    self.processor.stats['bytes_processed'] += len(chunk)
                    
                    # Try to extract complete images from buffer
                    while b'\xff\xd8' in buffer and b'\xff\xd9' in buffer:
                        # Find JPEG boundaries
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9', start) + 2
                        
                        if end > start:
                            image_data = buffer[start:end]
                            buffer = buffer[end:]
                            
                            # Process image
                            work_item = (image_data, templates, detection_params, f"frame_{frame_count}")
                            
                            try:
                                self.processor.processing_queue.put(work_item, timeout=1.0)
                                frame_count += 1
                                self.processor.stats['total_frames'] = frame_count
                                
                                # Update throughput
                                if self.processor.stats['start_time']:
                                    elapsed = time.time() - self.processor.stats['start_time']
                                    if elapsed > 0:
                                        self.processor.stats['throughput_mbps'] = (
                                            self.processor.stats['bytes_processed'] / (1024 * 1024) / elapsed * 8
                                        )
                                
                                yield {
                                    'type': 'progress',
                                    'processed': self.processor.stats['processed_frames'],
                                    'total': frame_count,
                                    'throughput_mbps': self.processor.stats['throughput_mbps']
                                }
                                
                            except queue.Full:
                                continue
                        else:
                            break
                    
                    # Check for results
                    while not self.processor.result_queue.empty():
                        try:
                            detections = self.processor.result_queue.get_nowait()
                            yield {
                                'type': 'detections',
                                'detections': detections
                            }
                        except queue.Empty:
                            break
                            
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e)
            }

class DirectoryMonitor:
    """Monitor directory for new files and process them (PNG files)"""
    
    def __init__(self, processor: HighThroughputProcessor):
        self.processor = processor
        self.processed_files = set()
        
    def monitor_directory(self, directory_path: str, file_pattern: str, templates: Dict, 
                         detection_params: Dict, polling_interval: int = 5) -> Generator[Dict, None, None]:
        """Monitor directory for new files (PNG)"""
        
        import glob
        
        patterns = [p.strip() for p in file_pattern.split(',')]
        
        while self.processor.active:
            try:
                all_files = []
                for pattern in patterns:
                    full_pattern = os.path.join(directory_path, pattern)
                    all_files.extend(glob.glob(full_pattern))
                
                # Find new files
                new_files = [f for f in all_files if f not in self.processed_files]
                
                # Process image files
                for file_path in new_files:
                    if not self.processor.active:
                        break
                    
                    try:
                        # Read file
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                        
                        # Add to processing queue
                        work_item = (image_data, templates, detection_params, os.path.basename(file_path))
                        
                        try:
                            self.processor.processing_queue.put(work_item, timeout=1.0)
                            self.processed_files.add(file_path)
                            
                            yield {
                                'type': 'progress',
                                'file': os.path.basename(file_path),
                                'file_type': 'image',
                                'total_files': len(self.processed_files)
                            }
                            
                        except queue.Full:
                            continue
                            
                    except Exception as e:
                        continue
                
                # Check for results
                while not self.processor.result_queue.empty():
                    try:
                        detections = self.processor.result_queue.get_nowait()
                        yield {
                            'type': 'detections',
                            'detections': detections
                        }
                    except queue.Empty:
                        break
                
                # Wait before next poll
                time.sleep(polling_interval)
                
            except Exception as e:
                yield {
                    'type': 'error',
                    'error': str(e)
                }
                time.sleep(polling_interval)

def create_live_feed_processor(config: LiveFeedConfig, detection_callback: Callable = None) -> HighThroughputProcessor:
    """Factory function to create live feed processor"""
    return HighThroughputProcessor(config, detection_callback)

def optimize_for_throughput(image: np.ndarray, quality_factor: float = 0.7) -> np.ndarray:
    """Optimize image for high-throughput processing"""
    
    if quality_factor < 1.0:
        # Reduce resolution for speed
        height, width = image.shape[:2]
        new_width = int(width * quality_factor)
        new_height = int(height * quality_factor)
        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    
    # Apply noise reduction for better detection
    if len(image.shape) == 3:
        image = cv.bilateralFilter(image, 5, 50, 50)
    
    return image