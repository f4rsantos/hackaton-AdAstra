"""
Live Feed Processing Tab - Unified Detection Version
Works EXACTLY like Pattern Detection + Unknown Signals tabs but with automatic template saving.
"""

import streamlit as st
import time
import os
import glob
from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import base64
import io
import json

# USE SAME IMPORTS AS PATTERN DETECTION
from functions import (
    pil_to_cv, cv_to_pil, run_unified_detection, save_drone_as_template,
    clean_template_name, draw_detection_boxes, draw_detection_boxes_no_labels,
    save_template_to_class_folder, filter_duplicates
)


def render(threshold, min_confidence, border_threshold, enable_border_detection,
           merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
           green_min_area, green_overlap_threshold, colored_merge_threshold,
           parallel_config, smart_astra_enabled, max_concurrent_streams,
           frame_skip_factor, buffer_size_mb, memory_limit_gb, throughput_target_mbps,
           enable_caching, polling_interval, enable_train_mode):
    """
    Render the Live Feed Processing tab.
    
    This tab works EXACTLY like Pattern Detection tab but:
    - Loads data from a directory (PNG files)
    - Automatically saves unidentified signals as templates
    - Supports Train Mode for class-based training
    """
    
    st.markdown('<h2 class="sub-header">📡 Live Feed Processing</h2>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'live_feed_running' not in st.session_state:
        st.session_state.live_feed_running = False
    if 'live_feed_results' not in st.session_state:
        st.session_state.live_feed_results = []
    if 'live_feed_metrics' not in st.session_state:
        st.session_state.live_feed_metrics = {
            'frames_processed': 0,
            'detections_count': 0,
            'new_patterns_saved': 0,
            'avg_process_time': 0.0
        }
    if 'auto_save_enabled' not in st.session_state:
        st.session_state.auto_save_enabled = True  # Default to enabled
    if 'api_last_processed' not in st.session_state:
        st.session_state.api_last_processed = set()  # Track processed API images
    
    # Configuration Section
    st.markdown("### ⚙️ Configuration")
    
    col_config1, col_config2 = st.columns([2, 1])
    
    with col_config1:
        # Data source selection
        data_source = st.radio(
            "📡 Data Source",
            ["Local Directory", "API Endpoint"],
            horizontal=True,
            help="Choose between local directory or API endpoint as data source"
        )
        
        if data_source == "Local Directory":
            # Directory input
            data_directory = st.text_input(
                "📁 Data Directory",
                value="",
                placeholder="Enter path to directory containing PNG files",
                help="Directory containing spectrograms (PNG files) to process"
            )
            api_url = None
            api_key = None
            api_poll_interval = None
        else:
            # API configuration
            data_directory = None
            api_url = st.text_input(
                "🌐 API Endpoint URL",
                value="",
                placeholder="https://api.example.com/drone-feed",
                help="API endpoint that returns image data (PNG or base64)"
            )
            
            # API authentication (optional)
            with st.expander("🔒 API Authentication (Optional)", expanded=False):
                api_key = st.text_input(
                    "API Key / Token",
                    value="",
                    type="password",
                    placeholder="Enter API key if required",
                    help="Authorization token for API access"
                )
                st.caption("Leave empty if API doesn't require authentication")
            
            # Polling interval
            api_poll_interval = st.slider(
                "⏱️ Polling Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5,
                help="How often to check API for new data"
            )
        
        # Mode selection
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            processing_mode = "Train Mode" if enable_train_mode else "Live Feed"
            st.info(f"**Mode:** {processing_mode}")
        
        with mode_col2:
            # Auto-save toggle
            auto_save = st.checkbox(
                "🤖 Auto-Save Unidentified Signals",
                value=st.session_state.auto_save_enabled,
                help="Automatically save unidentified signals as new templates"
            )
            st.session_state.auto_save_enabled = auto_save
    
    with col_config2:
        st.markdown("**📊 Detection Settings**")
        st.markdown(f"""
        - **Templates:** {len(st.session_state.templates) if st.session_state.templates else 0}
        - **Threshold:** {threshold:.2f}
        - **Min Confidence:** {min_confidence:.2f}
        - **Colored Areas:** {'✅' if detect_green_rectangles else '❌'}
        - **Auto-Save:** {'✅' if auto_save else '❌'}
        """)
        
        # Performance Mode Indicator
        has_templates = bool(st.session_state.get('templates', {}))
        has_colored = detect_green_rectangles
        
        if not has_templates and not has_colored:
            st.success("⚡ **Performance Mode:** Ultra Fast (No Detection)")
            st.caption("✅ Optimal: Images will be processed instantly")
        elif has_colored and not has_templates:
            st.warning("🟡 **Performance Mode:** Colored Detection Only")
            st.caption("💡 Tip: Disable colored detection for maximum speed")
        elif has_templates and not has_colored:
            template_count = len(st.session_state.templates)
            st.info(f"🟠 **Performance Mode:** Template Matching ({template_count} templates)")
            if template_count > 10:
                st.caption("💡 Tip: Consider reducing number of templates for faster processing")
        else:
            template_count = len(st.session_state.templates)
            st.error(f"🔴 **Performance Mode:** Full Detection ({template_count} templates + colored)")
            st.caption("💡 Tip: Disable colored detection or reduce templates for better speed")
    
    st.markdown("---")
    
    # Metrics Display - LIVE UPDATING
    metrics = st.session_state.live_feed_metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Create empty containers for live updates
    metric_frames = metrics_col1.empty()
    metric_detections = metrics_col2.empty()
    metric_patterns = metrics_col3.empty()
    metric_avgtime = metrics_col4.empty()
    
    # Display current metrics
    metric_frames.metric("Frames Processed", metrics['frames_processed'])
    metric_detections.metric("Total Detections", metrics['detections_count'])
    metric_patterns.metric("New Patterns Saved", metrics['new_patterns_saved'])
    metric_avgtime.metric("Avg Time (ms)", f"{metrics['avg_process_time']*1000:.0f}")
    
    # Color Mode Display (if auto-detected from frequency)
    if 'color_mode' in metrics and metrics['color_mode'] != 'auto':
        color_emoji = {'green': '🟢', 'blue': '🔵', 'aqua': '🔷', 'auto': '🎨'}.get(metrics['color_mode'], '🎨')
        st.caption(f"{color_emoji} Signal Color Mode: **{metrics['color_mode'].upper()}** (auto-detected from frequency)")
    
    st.markdown("---")
    
    # Control Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if not st.session_state.live_feed_running:
            # Check if valid source is provided
            valid_source = (data_directory and os.path.exists(data_directory)) or (api_url and api_url.strip())
            
            if st.button("▶️ Start Processing", type="primary", disabled=not valid_source):
                if data_source == "Local Directory":
                    if not data_directory or not os.path.exists(data_directory):
                        st.error("❌ Invalid directory path")
                    else:
                        st.session_state.live_feed_running = True
                        st.session_state.live_feed_source = "directory"
                        st.rerun()
                else:  # API Endpoint
                    if not api_url or not api_url.strip():
                        st.error("❌ Invalid API URL")
                    else:
                        st.session_state.live_feed_running = True
                        st.session_state.live_feed_source = "api"
                        st.session_state.api_url = api_url
                        st.session_state.api_key = api_key
                        st.session_state.api_poll_interval = api_poll_interval
                        st.session_state.api_last_processed = set()  # Reset tracking
                        st.rerun()
        else:
            if st.button("⏸️ Stop Processing", type="secondary"):
                st.session_state.live_feed_running = False
                st.rerun()
    
    with col_btn2:
        if st.button("🔄 Reset Metrics"):
            st.session_state.live_feed_metrics = {
                'frames_processed': 0,
                'detections_count': 0,
                'new_patterns_saved': 0,
                'avg_process_time': 0.0
            }
            st.session_state.live_feed_results = []
            st.session_state.api_last_processed = set()  # Reset API tracking
            st.rerun()
    
    with col_btn3:
        st.empty()  # Spacer
    
    st.markdown("---")
    
    # Processing Section
    if st.session_state.live_feed_running:
        # Prepare detection parameters - EXACTLY THE SAME AS PATTERN DETECTION
        detection_params = {
            'threshold': threshold,
            'min_confidence': min_confidence,
            'border_threshold': border_threshold if enable_border_detection else 0.5,
            'enable_border_detection': enable_border_detection,
            'merge_overlapping': merge_overlapping,
            'overlap_sensitivity': overlap_sensitivity,
            'detect_green_rectangles': detect_green_rectangles,
            'green_min_area': green_min_area,
            'green_overlap_threshold': green_overlap_threshold,
            'colored_merge_threshold': colored_merge_threshold,
            'parallel_config': parallel_config
        }
        
        # Pass metric containers for live updates
        metric_containers = {
            'frames': metric_frames,
            'detections': metric_detections,
            'patterns': metric_patterns,
            'avgtime': metric_avgtime
        }
        
        # Route to appropriate processing function based on source
        source_type = st.session_state.get('live_feed_source', 'directory')
        
        if source_type == "api":
            # API processing
            _process_api_feed(detection_params, auto_save, metric_containers)
        elif enable_train_mode:
            # Directory train mode
            _process_train_mode(data_directory, detection_params, auto_save, metric_containers)
        else:
            # Directory live feed
            _process_live_feed(data_directory, detection_params, auto_save, metric_containers)
    
    # Display Results
    if st.session_state.live_feed_results:
        st.markdown("### 📊 Detection Results")
        
        # Display recent results (last 10)
        recent_results = st.session_state.live_feed_results[-10:]
        
        for result in reversed(recent_results):
            with st.expander(f"📄 {result['filename']} - {len(result['matches'])} detections", expanded=False):
                if result['matches']:
                    # Draw detection boxes on image
                    img_cv = pil_to_cv(result['image'])
                    img_with_boxes = draw_detection_boxes(img_cv, result['matches'])
                    result_img = cv_to_pil(img_with_boxes)
                    
                    st.image(result_img, width='stretch')
                    
                    # Show match details
                    st.markdown("**Detections:**")
                    for idx, match in enumerate(result['matches']):
                        drone_name = match.get('template_name', 'Unknown')
                        confidence = match.get('confidence', 0.0)
                        detection_type = match.get('detection_type', 'template')
                        
                        if detection_type == 'unidentified':
                            st.markdown(f"{idx+1}. 🟢 **Unidentified Signal** - Confidence: {confidence:.3f}")
                        else:
                            st.markdown(f"{idx+1}. ✅ **{drone_name}** - Confidence: {confidence:.3f}")
                else:
                    st.image(result['image'], width='stretch')
                    st.info("No detections in this frame")

def _process_api_feed(detection_params, auto_save, metric_containers):
    """
    Process API Feed mode - polls API endpoint for new images.
    Supports JSON responses with base64 images, URLs, or direct image data.
    """
    status_text = st.empty()
    image_display = st.empty()
    
    # Get API configuration from session state
    api_url = st.session_state.get('api_url', '')
    api_key = st.session_state.get('api_key', '')
    poll_interval = st.session_state.get('api_poll_interval', 5)
    
    try:
        status_text.text("🌐 Connecting to API endpoint...")
        
        processed_ids = st.session_state.api_last_processed
        
        # Continuous polling loop
        while st.session_state.live_feed_running:
            poll_start_time = time.time()
            
            try:
                # Prepare headers
                headers = {}
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                
                # Make API request
                status_text.text(f"📡 Polling API... ({len(processed_ids)} processed)")
                response = requests.get(api_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Handle different response formats
                images_to_process = _parse_api_response(data, processed_ids)
                
                if not images_to_process:
                    status_text.text(f"⏳ No new images. Waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue
                
                # Process each new image
                for img_data in images_to_process:
                    if not st.session_state.live_feed_running:
                        break
                    
                    file_start_time = time.time()
                    
                    try:
                        # Extract image and metadata
                        img = img_data['image']
                        filename = img_data['filename']
                        img_id = img_data['id']
                        
                        # Display current image
                        status_text.text(f"🔍 Processing: {filename}")
                        image_display.image(img, caption=filename, width='stretch')
                        
                        # Run detection
                        test_image = _create_file_object(img, filename)
                        detection_results, unidentified_found, timing_info = run_unified_detection(
                            [test_image],
                            detection_params,
                            status_callback=None
                        )
                        
                        file_elapsed = time.time() - file_start_time
                        
                        # Process results
                        if detection_results:
                            result = detection_results[0]
                            
                            # Remove duplicates
                            if result['matches']:
                                result['matches'] = _remove_exact_duplicates(result['matches'])
                                result['matches'] = _remove_stacked_variants(result['matches'])
                                result['matches'] = filter_duplicates(result['matches'])
                            
                            # Update metrics
                            st.session_state.live_feed_metrics['frames_processed'] += 1
                            st.session_state.live_feed_metrics['detections_count'] += len(result['matches'])
                            
                            # Update average time
                            current_avg = st.session_state.live_feed_metrics['avg_process_time']
                            frames_count = st.session_state.live_feed_metrics['frames_processed']
                            new_avg = ((current_avg * (frames_count - 1)) + file_elapsed) / frames_count
                            st.session_state.live_feed_metrics['avg_process_time'] = new_avg
                            
                            # Update metric displays LIVE
                            metrics = st.session_state.live_feed_metrics
                            metric_containers['frames'].metric("Frames Processed", metrics['frames_processed'])
                            metric_containers['detections'].metric("Total Detections", metrics['detections_count'])
                            metric_containers['patterns'].metric("New Patterns Saved", metrics['new_patterns_saved'])
                            metric_containers['avgtime'].metric("Avg Time (ms)", f"{metrics['avg_process_time']*1000:.0f}")
                            
                            # Auto-save unidentified signals
                            if auto_save and result['matches']:
                                for match in result['matches']:
                                    if match.get('detection_type') == 'unidentified':
                                        # Extract pattern from image
                                        bbox = match.get('bbox', [0, 0, 0, 0])
                                        x, y, w, h = bbox
                                        
                                        # Find unidentified drone data
                                        drone_data = _find_unidentified_drone(match)
                                        
                                        if drone_data:
                                            # Generate unique pattern name
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            pattern_name = f"api_pattern_{timestamp}"
                                            
                                            # Save as template
                                            success = save_drone_as_template(
                                                drone_data['image'],
                                                pattern_name,
                                                confidence=match.get('confidence', 0.0)
                                            )
                                            
                                            if success:
                                                st.session_state.live_feed_metrics['new_patterns_saved'] += 1
                                                # Update metrics display
                                                metrics = st.session_state.live_feed_metrics
                                                metric_containers['patterns'].metric("New Patterns Saved", metrics['new_patterns_saved'])
                            
                            # Store result
                            result['filename'] = filename
                            result['image'] = img
                            st.session_state.live_feed_results.append(result)
                            
                            # Keep only last 50 results in memory
                            if len(st.session_state.live_feed_results) > 50:
                                st.session_state.live_feed_results = st.session_state.live_feed_results[-50:]
                        
                        # Mark as processed
                        processed_ids.add(img_id)
                        
                    except Exception as e:
                        status_text.error(f"❌ Error processing image: {str(e)}")
                        time.sleep(1)
                        continue
                
                # Wait before next poll
                poll_elapsed = time.time() - poll_start_time
                wait_time = max(0, poll_interval - poll_elapsed)
                if wait_time > 0 and st.session_state.live_feed_running:
                    status_text.text(f"✅ Batch complete. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                status_text.error(f"🌐 API Connection Error: {str(e)}")
                time.sleep(poll_interval)
                continue
            except json.JSONDecodeError:
                status_text.error("❌ Invalid JSON response from API")
                time.sleep(poll_interval)
                continue
                
    except Exception as e:
        status_text.error(f"❌ Fatal error: {str(e)}")
        st.session_state.live_feed_running = False


def _parse_api_response(data, processed_ids):
    """
    Parse API response and extract images.
    Supports multiple response formats:
    - Single image: {"image": "base64...", "filename": "...", "id": "..."}
    - Multiple images: {"images": [{...}, {...}]}
    - Direct array: [{...}, {...}]
    """
    images = []
    
    try:
        # Format 1: Direct array
        if isinstance(data, list):
            items = data
        # Format 2: Object with 'images' key
        elif 'images' in data:
            items = data['images']
        # Format 3: Single image
        elif 'image' in data or 'data' in data:
            items = [data]
        else:
            return []
        
        # Process each item
        for idx, item in enumerate(items):
            # Generate ID if not provided
            item_id = item.get('id', item.get('filename', f'image_{idx}'))
            
            # Skip if already processed
            if item_id in processed_ids:
                continue
            
            # Extract image
            img = None
            filename = item.get('filename', f'api_image_{item_id}.png')
            
            # Try different image formats
            if 'image' in item:
                # Base64 encoded image
                img_data = item['image']
                if isinstance(img_data, str):
                    # Remove data URL prefix if present
                    if ',' in img_data:
                        img_data = img_data.split(',', 1)[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
            
            elif 'url' in item:
                # Image URL
                img_url = item['url']
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()
                img = Image.open(io.BytesIO(img_response.content))
            
            elif 'data' in item:
                # Direct binary data
                img_bytes = base64.b64decode(item['data'])
                img = Image.open(io.BytesIO(img_bytes))
            
            if img:
                # Maintain original resolution - do not resize
                # Original spectrograms are typically 2048x384
                
                images.append({
                    'image': img,
                    'filename': filename,
                    'id': item_id
                })
    
    except Exception as e:
        print(f"Error parsing API response: {e}")
    
    return images


def _process_live_feed(directory, detection_params, auto_save, metric_containers):
    """
    Process Live Feed mode - with GPU BATCH PROCESSING for horizontal parallelization.
    When GPU is enabled, batches multiple images for parallel processing.
    Otherwise, processes files ONE-BY-ONE with live updates.
    """
    status_text = st.empty()
    progress_bar = st.progress(0)
    image_display = st.empty()
    
    # Check if GPU batch processing is enabled
    parallel_config = detection_params.get('parallel_config')
    use_gpu_batch = (parallel_config and 
                     getattr(parallel_config, 'gpu_enabled', False) and 
                     getattr(parallel_config, 'gpu_batch_images', 1) > 1)
    gpu_batch_size = getattr(parallel_config, 'gpu_batch_images', 1) if use_gpu_batch else 1
    
    try:
        status_text.text("🔍 Scanning directory for files...")
        
        # Collect PNG file paths
        png_files = list(Path(directory).rglob("*.png"))
        
        all_files = png_files
        total_files = len(all_files)
        
        if total_files == 0:
            status_text.error("❌ No PNG files found in directory")
            st.session_state.live_feed_running = False
            return
        
        status_text.text(f"📂 Found {len(png_files)} PNG files")
        if use_gpu_batch:
            status_text.text(f"🎮 GPU Batch Mode: Processing {gpu_batch_size} images at a time")
        time.sleep(0.3)
        
        # GPU BATCH PROCESSING MODE
        if use_gpu_batch:
            _process_live_feed_gpu_batch(
                all_files, total_files, detection_params, 
                auto_save, metric_containers, gpu_batch_size,
                status_text, progress_bar, image_display
            )
        else:
            # SEQUENTIAL PROCESSING MODE (original behavior)
            _process_live_feed_sequential(
                all_files, total_files, detection_params,
                auto_save, metric_containers,
                status_text, progress_bar, image_display
            )
        
        # Complete
        progress_bar.progress(1.0)
        metrics = st.session_state.live_feed_metrics
        status_text.text(f"✅ Completed: {metrics['frames_processed']} frames, {metrics['detections_count']} detections, {metrics['new_patterns_saved']} saved")
        
        # Stop processing
        st.session_state.live_feed_running = False
        
    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
        st.session_state.live_feed_running = False


def _process_live_feed_gpu_batch(all_files, total_files, detection_params,
                                  auto_save, metric_containers, gpu_batch_size,
                                  status_text, progress_bar, image_display):
    """
    Process live feed with GPU batch mode - horizontal parallelization.
    Loads multiple PNG images, processes them all at once on GPU.
    """
    from functions import run_multi_image_gpu_detection, pil_to_cv, cv_to_pil, draw_detection_boxes, filter_duplicates
    
    file_idx = 0
    
    while file_idx < total_files and st.session_state.live_feed_running:
        batch_start_time = time.time()
        
        # Collect batch of images
        batch_files = all_files[file_idx:file_idx + gpu_batch_size]
        batch_images = []
        batch_filenames = []
        batch_pil_images = []
        
        # Load batch
        for file_path in batch_files:
            try:
                img = Image.open(file_path)
                filename = file_path.name
                
                img_cv = pil_to_cv(img)
                batch_images.append(img_cv)
                batch_filenames.append(filename)
                batch_pil_images.append(img)
            except Exception as e:
                status_text.warning(f"⚠️ Error loading {file_path.name}: {str(e)}")
                continue
        
        if not batch_images:
            file_idx += len(batch_files)
            continue
        
        # Run GPU batch detection
        status_text.text(f"🎮 GPU processing batch of {len(batch_images)} images... [{file_idx+1}-{file_idx+len(batch_images)}/{total_files}]")
        
        gpu_results = run_multi_image_gpu_detection(
            batch_images, batch_filenames, detection_params, 
            status_callback=lambda msg: status_text.text(msg)
        )
        
        batch_elapsed = time.time() - batch_start_time
        
        # Process each image's results
        if gpu_results:
            for img_idx, result in enumerate(gpu_results):
                # Remove duplicates
                if result['matches']:
                    result['matches'] = _remove_exact_duplicates(result['matches'])
                    result['matches'] = _remove_stacked_variants(result['matches'])
                    result['matches'] = filter_duplicates(result['matches'], iou_threshold=0.3)
                
                num_detections = len(result['matches'])
                
                # Update metrics
                st.session_state.live_feed_metrics['frames_processed'] += 1
                st.session_state.live_feed_metrics['detections_count'] += num_detections
                
                # Update average time (per image)
                total_frames = st.session_state.live_feed_metrics['frames_processed']
                prev_avg = st.session_state.live_feed_metrics['avg_process_time']
                img_time = batch_elapsed / len(batch_images)
                new_avg = ((prev_avg * (total_frames - 1)) + img_time) / total_frames
                st.session_state.live_feed_metrics['avg_process_time'] = new_avg
                
                # Update metrics display
                metrics = st.session_state.live_feed_metrics
                metric_containers['frames'].metric("Frames Processed", metrics['frames_processed'])
                metric_containers['detections'].metric("Total Detections", metrics['detections_count'])
                metric_containers['patterns'].metric("New Patterns Saved", metrics['new_patterns_saved'])
                metric_containers['avgtime'].metric("Avg Time (ms)", f"{metrics['avg_process_time']*1000:.0f}")
                
                # Store result
                st.session_state.live_feed_results.append(result)
                
                # Display last image in batch with detections
                if img_idx == len(gpu_results) - 1:
                    if result['matches']:
                        img_cv = pil_to_cv(result['image'])
                        img_with_boxes = draw_detection_boxes(img_cv, result['matches'])
                        image_display.image(
                            cv_to_pil(img_with_boxes),
                            caption=f"🎮 GPU Batch [{file_idx+1}-{file_idx+len(batch_images)}/{total_files}] - Last: {result['filename']} ({num_detections} detections)",
                            width='stretch'
                        )
                    else:
                        image_display.image(
                            result['image'],
                            caption=f"🎮 GPU Batch [{file_idx+1}-{file_idx+len(batch_images)}/{total_files}] - Last: {result['filename']} (no detections)",
                            width='stretch'
                        )
        
        # Update progress
        file_idx += len(batch_files)
        progress_bar.progress(min(file_idx / total_files, 1.0))
        
        # Show batch completion
        imgs_per_sec = len(batch_images) / batch_elapsed if batch_elapsed > 0 else 0
        status_text.text(f"✅ GPU batch complete: {len(batch_images)} images in {batch_elapsed:.2f}s ({imgs_per_sec:.1f} imgs/sec)")
        time.sleep(0.1)


def _process_live_feed_sequential(all_files, total_files, detection_params,
                                   auto_save, metric_containers,
                                   status_text, progress_bar, image_display):
    """
    Process live feed sequentially with proper preprocessing:
    - Resize images to 31778x384
    - Cut into 2048x384 chunks
    - Process each chunk individually
    """
    from functions import pil_to_cv, cv_to_pil, run_unified_detection, draw_detection_boxes, filter_duplicates
    from image_preprocessing_pipeline import preprocess_images_batch
    
    # Preprocessing parameters (same as Pattern Detection)
    target_width = 31778
    target_height = 384
    chunk_width = 2048
    chunk_height = 384
    
    # Process files ONE-BY-ONE for true streaming
    for file_idx, file_path in enumerate(all_files):
        # Check if user stopped
        if not st.session_state.live_feed_running:
            status_text.warning("⏸️ Processing stopped by user")
            break
        
        file_start_time = time.time()
        
        try:
            # Load PNG file
            status_text.text(f"📷 Loading PNG [{file_idx+1}/{total_files}]: {file_path.name}")
            img = Image.open(file_path)
            filename = file_path.name
            
            # Check if we can skip expensive detection processing
            should_skip_detection = (
                not st.session_state.get('templates', {}) and  # No templates loaded
                not detection_params.get('detect_green_rectangles', False)  # No colored detection
            )
            
            if should_skip_detection:
                # Fast path: No detection needed, just validate the image
                status_text.text(f"⚡ Fast mode: No detection configured for {filename}")
                
                # Create minimal result without expensive processing
                detection_results = [{
                    'filename': filename,
                    'image': img,  # Keep the image for display
                    'matches': []  # Empty matches
                }]
                unidentified_found = []
                timing_info = {
                    'total_time': time.time() - file_start_time,
                    'mode': 'fast_skip',
                    'images_processed': 1,
                    'templates_matched': 0,
                    'colored_detection': False
                }
                
            else:
                # Normal path: Preprocess (resize + chunk) then run detection
                status_text.text(f"🔧 Preprocessing {filename}...")
                
                # Preprocess: resize to 31778x384, then chunk into 2048x384 pieces
                test_image = _create_file_object(img, filename)
                preprocessed_results = preprocess_images_batch(
                    [test_image],
                    target_width=target_width,
                    target_height=target_height,
                    chunk_width=chunk_width,
                    chunk_height=chunk_height
                )
                
                if not preprocessed_results or not preprocessed_results[0]['chunks']:
                    status_text.warning(f"⚠️ Failed to preprocess {filename}")
                    continue
                
                # Get chunks for this image
                chunks_data = preprocessed_results[0]['chunks']
                status_text.text(f"🔍 Detecting patterns in {len(chunks_data)} chunks from {filename}...")
                
                # Convert chunks to file objects for unified detection
                chunk_images = []
                for chunk_idx, chunk_info in enumerate(chunks_data):
                    chunk_img = chunk_info['image']
                    chunk_filename = f"{filename}_chunk_{chunk_idx}"
                    chunk_file = _create_file_object(chunk_img, chunk_filename)
                    chunk_images.append(chunk_file)
                
                # Run unified detection on ALL chunks
                detection_results, unidentified_found, timing_info = run_unified_detection(
                    chunk_images,
                    detection_params,
                    status_callback=lambda msg: status_text.text(msg)
                )
                
                # Merge chunk results back to full image (adjust coordinates)
                # Combine all chunk matches into single result with adjusted coordinates
                merged_matches = []
                for chunk_idx, chunk_result in enumerate(detection_results):
                    chunk_info = chunks_data[chunk_idx]
                    offset_x = chunk_info['offset_x']
                    offset_y = chunk_info['offset_y']
                    
                    for match in chunk_result.get('matches', []):
                        # Adjust coordinates back to full image space
                        adjusted_match = match.copy()
                        adjusted_match['x'] += offset_x
                        adjusted_match['bbox'] = [
                            match['bbox'][0] + offset_x,
                            match['bbox'][1] + offset_y,
                            match['bbox'][2],
                            match['bbox'][3]
                        ]
                        merged_matches.append(adjusted_match)
                
                # Create single merged result with full image
                detection_results = [{
                    'filename': filename,
                    'image': img,  # Use original full image
                    'matches': merged_matches
                }]
            
            file_elapsed = time.time() - file_start_time
            
            # Get result for this image
            if detection_results:
                result = detection_results[0]
                
                # LIVE-FEED: Aggressive duplicate removal with variant handling
                # Step 1: Remove exact duplicates (same template, exact same box)
                # Step 2: Handle template variants (-A, -B) on top of each other
                # Step 3: Merge similar detections (same template, overlapping boxes)
                if result['matches']:
                    result['matches'] = _remove_exact_duplicates(result['matches'])
                    result['matches'] = _remove_stacked_variants(result['matches'])  # 🆕 NEW
                    result['matches'] = filter_duplicates(result['matches'], iou_threshold=0.3)
                
                num_detections = len(result['matches'])
                
                # Update metrics LIVE
                st.session_state.live_feed_metrics['frames_processed'] += 1
                st.session_state.live_feed_metrics['detections_count'] += num_detections
                
                # Update average time
                total_frames = st.session_state.live_feed_metrics['frames_processed']
                prev_avg = st.session_state.live_feed_metrics['avg_process_time']
                new_avg = ((prev_avg * (total_frames - 1)) + file_elapsed) / total_frames
                st.session_state.live_feed_metrics['avg_process_time'] = new_avg
                
                # Update TOP metrics display LIVE
                metrics = st.session_state.live_feed_metrics
                metric_containers['frames'].metric("Frames Processed", metrics['frames_processed'])
                metric_containers['detections'].metric("Total Detections", metrics['detections_count'])
                metric_containers['patterns'].metric("New Patterns Saved", metrics['new_patterns_saved'])
                metric_containers['avgtime'].metric("Avg Time (ms)", f"{metrics['avg_process_time']*1000:.0f}")
                
                # Store result
                st.session_state.live_feed_results.append(result)
                
                # Display image with detections LIVE
                if result['matches']:
                    img_cv = pil_to_cv(result['image'])
                    img_with_boxes = draw_detection_boxes(img_cv, result['matches'])
                    image_display.image(
                        cv_to_pil(img_with_boxes), 
                        caption=f"[{file_idx+1}/{total_files}] {filename} - {num_detections} detections",
                        width='stretch'
                    )
                    status_text.text(f"✅ {filename}: {num_detections} detections in {file_elapsed:.2f}s")
                else:
                    image_display.image(
                        result['image'], 
                        caption=f"[{file_idx+1}/{total_files}] {filename} - No detections",
                        width='stretch'
                    )
                    status_text.text(f"✅ {filename}: No detections ({file_elapsed:.2f}s)")
                
                # NOTE: Live Feed mode does NOT auto-save patterns
                # Use Train Mode for automatic template creation
            
            # Update progress bar
            progress_bar.progress((file_idx + 1) / total_files)
            
            # Small delay for visual feedback
            time.sleep(0.1)
            
        except Exception as e:
            status_text.error(f"❌ Error processing {file_path.name}: {str(e)}")
            continue  # Continue with next file instead of stopping
    
    # Complete (MOVED OUTSIDE THE LOOP)
    progress_bar.progress(1.0)
    metrics = st.session_state.live_feed_metrics
    status_text.text(f"✅ Completed: {metrics['frames_processed']} frames, {metrics['detections_count']} detections, {metrics['new_patterns_saved']} saved")
    
    # Stop processing
    st.session_state.live_feed_running = False


def _process_train_mode(directory, detection_params, auto_save, metric_containers):
    """
    Process Train Mode - class-based training with auto-save.
    Only saves UNIDENTIFIED signals that meet the same standards as Pattern Detection.
    """
    status_text = st.empty()
    progress_bar = st.progress(0)
    image_display = st.empty()
    
    try:
        status_text.text("🔍 Scanning for class folders...")
        
        # Get all subdirectories (each is a class)
        class_folders = [d for d in Path(directory).iterdir() if d.is_dir()]
        
        if not class_folders:
            status_text.error("❌ No class folders found. Train Mode expects subdirectories (one per drone class)")
            st.session_state.live_feed_running = False
            return
        
        status_text.text(f"📂 Found {len(class_folders)} class folders")
        time.sleep(0.5)
        
        total_saved = 0
        
        # Process each class folder
        for folder_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name.lower()
            
            status_text.text(f"🎯 Processing class: {class_name} ({folder_idx+1}/{len(class_folders)})")
            
            # Collect PNG FILE PATHS (don't load yet for streaming)
            png_files = list(class_folder.glob("*.png"))
            all_files = png_files
            
            if not all_files:
                status_text.warning(f"⚠️ No PNG files in {class_name}, skipping...")
                continue
            
            # Filter templates to only those matching this class
            if st.session_state.templates:
                class_templates = {
                    k: v for k, v in st.session_state.templates.items()
                    if (v.get('class_folder') or '').lower() == class_name
                }
            else:
                class_templates = {}
            
            status_text.text(f"🔍 {class_name}: Processing {len(all_files)} files with {len(class_templates)} templates...")
            
            # Find next variant letter
            variant_letter = _get_next_variant_letter(class_name)
            saved_this_class = 0
            saved_patterns = set()  # Track saved patterns for THIS CLASS to avoid duplicates (lenient: >50% overlap)
            
            # Preprocessing parameters (same as Live Feed)
            target_width = 31778
            target_height = 384
            chunk_width = 2048
            chunk_height = 384
            
            # Process files ONE-BY-ONE for true streaming with live display
            for file_idx, file_path in enumerate(all_files):
                # Check if user stopped
                if not st.session_state.live_feed_running:
                    break
                
                file_start_time = time.time()
                
                try:
                    # Load PNG file
                    status_text.text(f"📷 [{file_idx+1}/{len(all_files)}] Loading: {file_path.name}")
                    img = Image.open(file_path)
                    filename = file_path.name
                    
                    # Run detection on THIS image
                    status_text.text(f"🔍 [{file_idx+1}/{len(all_files)}] Detecting in {filename}...")
                    
                    # Check if we can skip expensive detection processing
                    should_skip_detection = (
                        not class_templates and  # No templates for this class
                        not detection_params.get('detect_green_rectangles', False)  # No colored detection
                    )
                    
                    if should_skip_detection:
                        # Fast path for Train Mode: No detection needed
                        status_text.text(f"⚡ [{file_idx+1}/{len(all_files)}] Fast mode: No detection for {filename}")
                        
                        detection_results = [{
                            'filename': filename,
                            'image': img,
                            'matches': []
                        }]
                        unidentified_found = []
                        timing_info = {
                            'total_time': 0.001,
                            'mode': 'train_fast_skip'
                        }
                    else:
                        # Normal Train Mode detection with preprocessing
                        # Temporarily use class-specific templates
                        original_templates = st.session_state.templates
                        st.session_state.templates = class_templates
                        
                        status_text.text(f"🔧 [{file_idx+1}/{len(all_files)}] Preprocessing {filename}...")
                        
                        # Preprocess: resize to 31778x384, then chunk into 2048x384 pieces
                        from image_preprocessing_pipeline import preprocess_images_batch
                        test_image = _create_file_object(img, filename)
                        preprocessed_results = preprocess_images_batch(
                            [test_image],
                            target_width=target_width,
                            target_height=target_height,
                            chunk_width=chunk_width,
                            chunk_height=chunk_height
                        )
                        
                        if not preprocessed_results or not preprocessed_results[0]['chunks']:
                            status_text.warning(f"⚠️ Failed to preprocess {filename}")
                            st.session_state.templates = original_templates
                            continue
                        
                        # Get chunks for this image
                        chunks_data = preprocessed_results[0]['chunks']
                        status_text.text(f"🔍 [{file_idx+1}/{len(all_files)}] Processing {len(chunks_data)} chunks...")
                        
                        # Convert chunks to file objects
                        chunk_images = []
                        for chunk_idx, chunk_info in enumerate(chunks_data):
                            chunk_img = chunk_info['image']
                            chunk_filename = f"{filename}_chunk_{chunk_idx}"
                            chunk_file = _create_file_object(chunk_img, chunk_filename)
                            chunk_images.append(chunk_file)
                        
                        # Run unified detection on chunks
                        # Train Mode: ENABLE boundary check with top-bottom exception
                        detection_results, unidentified_found, timing_info = run_unified_detection(
                            chunk_images,
                            detection_params,
                            status_callback=lambda msg: status_text.text(msg),
                            skip_boundary_check=False,  # Train Mode: Enforce boundaries
                            skip_duplicate_check=True  # We handle duplicates ourselves
                        )
                        
                        # Merge chunk results back to full image
                        merged_matches = []
                        for chunk_idx, chunk_result in enumerate(detection_results):
                            chunk_info = chunks_data[chunk_idx]
                            offset_x = chunk_info['offset_x']
                            offset_y = chunk_info['offset_y']
                            
                            for match in chunk_result.get('matches', []):
                                adjusted_match = match.copy()
                                adjusted_match['x'] += offset_x
                                adjusted_match['bbox'] = [
                                    match['bbox'][0] + offset_x,
                                    match['bbox'][1] + offset_y,
                                    match['bbox'][2],
                                    match['bbox'][3]
                                ]
                                merged_matches.append(adjusted_match)
                        
                        # Create single merged result
                        detection_results = [{
                            'filename': filename,
                            'image': img,
                            'matches': merged_matches
                        }]
                        
                        # Restore original templates
                        st.session_state.templates = original_templates
                    
                    # Get result
                    if detection_results:
                        result = detection_results[0]
                        
                        # TRAIN MODE: Aggressive duplicate removal
                        # Step 1: Remove exact duplicates (same template, exact same box)
                        # Step 2: Merge similar detections (same template, overlapping boxes)
                        if result['matches']:
                            result['matches'] = _remove_exact_duplicates(result['matches'])
                            result['matches'] = filter_duplicates(result['matches'], iou_threshold=0.3)
                        
                        num_detections = len(result['matches'])
                        num_unidentified = len(unidentified_found)
                        
                        file_elapsed = time.time() - file_start_time
                        
                        # Update metrics LIVE
                        st.session_state.live_feed_metrics['frames_processed'] += 1
                        st.session_state.live_feed_metrics['detections_count'] += num_detections
                        
                        # Update average time
                        total_frames = st.session_state.live_feed_metrics['frames_processed']
                        prev_avg = st.session_state.live_feed_metrics['avg_process_time']
                        new_avg = ((prev_avg * (total_frames - 1)) + file_elapsed) / total_frames
                        st.session_state.live_feed_metrics['avg_process_time'] = new_avg
                        
                        # Update TOP metrics display LIVE
                        metrics = st.session_state.live_feed_metrics
                        metric_containers['frames'].metric("Frames Processed", metrics['frames_processed'])
                        metric_containers['detections'].metric("Total Detections", metrics['detections_count'])
                        metric_containers['patterns'].metric("New Patterns Saved", metrics['new_patterns_saved'])
                        metric_containers['avgtime'].metric("Avg Time (ms)", f"{metrics['avg_process_time']*1000:.0f}")
                        
                        # Store result
                        st.session_state.live_feed_results.append(result)
                        
                        # Display image with detections LIVE
                        if result['matches']:
                            img_cv = pil_to_cv(result['image'])
                            img_with_boxes = draw_detection_boxes(img_cv, result['matches'])
                            image_display.image(
                                cv_to_pil(img_with_boxes),
                                caption=f"[{class_name}] {filename} - {num_detections} detections ({num_unidentified} unidentified)",
                                width='stretch'
                            )
                        else:
                            image_display.image(
                                result['image'],
                                caption=f"[{class_name}] {filename} - No detections",
                                width='stretch'
                            )
                        
                        # Auto-save HIGH confidence unidentified signals (good detections)
                        # Logic: Save patterns with confidence >= 0.5 (TRAIN MODE: Higher threshold for quality)
                        # Duplicate detection: Simple bbox comparison to avoid saving exact same pattern twice
                        if auto_save and unidentified_found:
                            min_conf_threshold = 0.5  # Train Mode: Require 50% confidence minimum
                            
                            for idx, unidentified in enumerate(unidentified_found):
                                # Save HIGH confidence patterns (good quality detections)
                                confidence = unidentified.get('confidence', 0.0)
                                
                                if confidence < min_conf_threshold:  # Skip LOW confidence
                                    continue  # Skip patterns below 0.5 threshold
                                
                                # SIMILARITY-based duplicate check: Skip if similar pattern already saved
                                # This merges visually similar patterns instead of requiring exact bbox match
                                bbox = unidentified.get('bbox', [0, 0, 0, 0])
                                x, y, w, h = bbox
                                
                                # Check if this pattern is similar to any already-saved pattern
                                is_duplicate = False
                                for saved_bbox in saved_patterns:
                                    saved_x, saved_y, saved_w, saved_h = saved_bbox
                                    
                                    # Calculate size similarity
                                    saved_area = saved_w * saved_h
                                    new_area = w * h
                                    size_ratio = min(saved_area, new_area) / max(saved_area, new_area) if max(saved_area, new_area) > 0 else 0
                                    
                                    # Calculate center distance
                                    saved_center = (saved_x + saved_w/2, saved_y + saved_h/2)
                                    new_center = (x + w/2, y + h/2)
                                    distance = np.sqrt((saved_center[0] - new_center[0])**2 + (saved_center[1] - new_center[1])**2)
                                    
                                    # Calculate overlap percentage
                                    x_left = max(x, saved_x)
                                    y_top = max(y, saved_y)
                                    x_right = min(x + w, saved_x + saved_w)
                                    y_bottom = min(y + h, saved_y + saved_h)
                                    
                                    overlap_pct = 0
                                    if x_right > x_left and y_bottom > y_top:
                                        intersection = (x_right - x_left) * (y_bottom - y_top)
                                        overlap_pct = intersection / new_area if new_area > 0 else 0
                                    
                                    # Consider duplicate if:
                                    # - High overlap (70%+) OR
                                    # - Similar size (80%+) AND close distance (within 1x avg size)
                                    avg_size = np.sqrt((saved_area + new_area) / 2)
                                    if overlap_pct >= 0.7 or (size_ratio >= 0.8 and distance <= avg_size):
                                        is_duplicate = True
                                        break
                                
                                if is_duplicate:
                                    continue
                                
                                bbox_tuple = (x, y, w, h)
                                
                                drone_data = _find_unidentified_drone(unidentified)
                                
                                if not drone_data:
                                    continue
                                
                                if drone_data:
                                    # Use class-based naming: drone-name-A, drone-name-B, etc.
                                    pattern_name = f"{class_name}-{variant_letter}"
                                    
                                    # Use save_template_to_class_folder to properly set class_folder attribute
                                    # This ensures templates are properly associated with their class for filtering
                                    success, message, saved_filepath = save_template_to_class_folder(
                                        drone_data['image'],
                                        pattern_name,
                                        class_name
                                    )
                                    
                                    if success:
                                        saved_this_class += 1
                                        total_saved += 1
                                        saved_patterns.add(bbox_tuple)  # Store bbox tuple to avoid exact duplicates
                                        st.session_state.live_feed_metrics['new_patterns_saved'] += 1
                                        # Update TOP metrics live
                                        metric_containers['patterns'].metric("New Patterns Saved", st.session_state.live_feed_metrics['new_patterns_saved'])
                                        # Move to next letter
                                        variant_letter = chr(ord(variant_letter) + 1)
                            
                            if saved_this_class > 0:
                                status_text.text(f"💾 Saved {saved_this_class} patterns from {filename}")
                        
                        # Show simple completion message
                        if not auto_save or saved_this_class == 0:
                            status_text.text(f"✅ Processed {filename}")
                    
                    # Small delay for visual feedback
                    time.sleep(0.1)
                    
                except Exception as e:
                    status_text.warning(f"⚠️ Error processing {file_path.name}: {str(e)}")
                    continue
            
            if saved_this_class > 0:
                status_text.text(f"✅ Class {class_name}: Saved {saved_this_class} new variants")
                
                # TRAIN MODE: Deduplicate templates for THIS class immediately after saving
                if auto_save:
                    status_text.text(f"🔍 Running deduplication for {class_name}...")
                    removed_count = _deduplicate_class_templates(class_name)
                    if removed_count > 0:
                        status_text.text(f"🧹 Removed {removed_count} duplicate templates from {class_name}")
                        saved_this_class -= removed_count
                        total_saved -= removed_count
                
                time.sleep(0.3)
            
            # Update progress
            progress_bar.progress((folder_idx + 1) / len(class_folders))
        
        # Complete
        progress_bar.progress(1.0)
        metrics = st.session_state.live_feed_metrics
        
        status_text.text(f"✅ Train Mode complete: {total_saved} new patterns saved")
        
        if total_saved > 0:
            st.success(f"🎉 Training complete! Saved {total_saved} new pattern variants")
        
        st.session_state.live_feed_running = False
        
    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
        st.session_state.live_feed_running = False


def _create_file_object(img, filename):
    """Create a file-like object from PIL Image with name attribute"""
    import io
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_buffer.name = filename
    
    return img_buffer


def _remove_exact_duplicates(matches):
    """
    Remove exact duplicate detections (same template, same exact bounding box).
    This handles cases where the same detection appears multiple times.
    
    Args:
        matches: List of detection matches
        
    Returns:
        List of matches with exact duplicates removed (keeps highest confidence)
    """
    if not matches or len(matches) <= 1:
        return matches
    
    # Group by template name and bounding box
    unique_detections = {}
    
    for match in matches:
        # Create key from template name and bounding box
        key = (
            match['template_name'],
            match['x'],
            match['y'],
            match['width'],
            match['height']
        )
        
        # Keep the match with highest confidence for each unique key
        if key not in unique_detections:
            unique_detections[key] = match
        else:
            # Keep the one with higher confidence
            if match['confidence'] > unique_detections[key]['confidence']:
                unique_detections[key] = match
    
    return list(unique_detections.values())


def _remove_stacked_variants(matches):
    """
    Remove stacked template variants (e.g., drone-A, drone-B on top of each other).
    When multiple variants of the same drone class are detected at similar positions,
    keep only the one with the best confidence that matches the identified area.
    
    Args:
        matches: List of detection matches
        
    Returns:
        List of matches with stacked variants removed (keeps best match per position)
    """
    if not matches or len(matches) <= 1:
        return matches
    
    # Extract base name (remove variant suffix like -A, -B, -1, -2)
    def get_base_name(template_name):
        """Extract base drone name without variant suffix"""
        import re
        # Remove variants: -A, -B, -1, -2, etc.
        base = re.sub(r'-[A-Z0-9]+$', '', template_name, flags=re.IGNORECASE)
        return base.lower()
    
    # Group matches by position (similar x, similar width, any height)
    position_groups = []
    
    for match in matches:
        x, y, w, h = match['x'], match['y'], match['width'], match['height']
        center_x = x + w/2
        
        # Find if this match belongs to an existing position group
        found_group = False
        for group in position_groups:
            # Check against first match in group
            ref = group[0]
            ref_x, ref_w = ref['x'], ref['width']
            ref_center_x = ref_x + ref_w/2
            
            # Similar position if:
            # - Centers are close (within 20% of width)
            # - Widths are similar (within 20%)
            x_tolerance = max(w, ref_w) * 0.2
            w_tolerance = 0.2
            
            if (abs(center_x - ref_center_x) <= x_tolerance and
                abs(w - ref_w) / max(w, ref_w) <= w_tolerance):
                group.append(match)
                found_group = True
                break
        
        if not found_group:
            position_groups.append([match])
    
    # For each position group, keep only best match per drone class
    filtered_matches = []
    
    for group in position_groups:
        if len(group) == 1:
            # Single detection, keep it
            filtered_matches.append(group[0])
        else:
            # Multiple detections at same position
            # Group by base drone name
            drone_groups = {}
            for match in group:
                base_name = get_base_name(match['template_name'])
                if base_name not in drone_groups:
                    drone_groups[base_name] = []
                drone_groups[base_name].append(match)
            
            # For each drone class, keep best match
            for base_name, variants in drone_groups.items():
                if len(variants) == 1:
                    filtered_matches.append(variants[0])
                else:
                    # Multiple variants of same drone - keep highest confidence
                    best_match = max(variants, key=lambda m: m['confidence'])
                    filtered_matches.append(best_match)
    
    return filtered_matches



def _find_unidentified_drone(unidentified_match):
    """
    Find the corresponding drone data in session state for an unidentified signal.
    Returns the drone data structure needed for save_drone_as_template().
    """
    # The unidentified_match from run_unified_detection contains the data we need
    # We need to convert it to the format expected by save_drone_as_template
    
    if 'source_image' not in unidentified_match:
        return None
    
    # Extract the pattern from the source image
    source_img = unidentified_match['source_image']
    x, y = unidentified_match['x'], unidentified_match['y']
    w, h = unidentified_match['width'], unidentified_match['height']
    
    # Create drone data structure
    img_cv = pil_to_cv(source_img)
    pattern_img = img_cv[y:y+h, x:x+w]
    pattern_pil = cv_to_pil(pattern_img)
    
    drone_data = {
        'id': unidentified_match.get('id', 0),
        'image': pattern_pil,
        'full_image': source_img,
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'confidence': unidentified_match.get('confidence', 0.0),
        'detection_type': 'unidentified',
        'filename': unidentified_match.get('filename', 'unknown'),
        'timestamp': datetime.now(),
        'area': w * h
    }
    
    return drone_data


def _deduplicate_class_templates(class_name):
    """
    Deduplicate templates for a single class folder.
    Removes templates that are too similar (>92% match) to existing ones.
    Optimized to reduce RAM usage by not loading all images at once.
    
    Returns:
        int: Number of templates removed
    """
    from pathlib import Path
    
    templates_dir = Path("stored_templates")
    class_folder = templates_dir / class_name
    
    if not class_folder.exists():
        return 0
    
    # Get all templates for this class
    template_files = list(class_folder.glob("*.png"))
    if len(template_files) <= 1:
        return 0  # Nothing to deduplicate
    
    removed_count = 0
    templates_to_remove = set()
    
    # Sort by modification time (keep older ones, remove newer duplicates)
    template_files.sort(key=lambda f: f.stat().st_mtime)
    
    # Process templates one at a time to reduce RAM usage
    for i, template_path in enumerate(template_files):
        if template_path in templates_to_remove:
            continue
        
        # Load current template only when needed
        current_img = None
        try:
            current_img = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
            if current_img is None:
                continue
        except Exception:
            continue
        
        # Compare with remaining templates (only load comparison images as needed)
        for j in range(i + 1, len(template_files)):
            compare_path = template_files[j]
            if compare_path in templates_to_remove:
                continue
            
            try:
                # Load comparison image
                compare_img = cv.imread(str(compare_path), cv.IMREAD_GRAYSCALE)
                if compare_img is None:
                    continue
                
                # Check if images are same size
                if current_img.shape != compare_img.shape:
                    # Release memory immediately
                    del compare_img
                    continue
                
                # Calculate similarity using template matching
                result = cv.matchTemplate(current_img, compare_img, cv.TM_CCOEFF_NORMED)
                similarity = result[0][0]
                
                # Release memory immediately after comparison
                del compare_img, result
                
                # If too similar (>92%), mark for removal
                if similarity > 0.92:
                    templates_to_remove.add(compare_path)
                    
            except Exception:
                continue
        
        # Release current image memory before moving to next
        del current_img
    
    # Remove duplicate templates
    for template_path in templates_to_remove:
        try:
            # Remove from disk
            template_path.unlink()
            removed_count += 1
            
            # Remove from session state
            template_key = f"{class_name}/{template_path.name}"
            if template_key in st.session_state.templates:
                del st.session_state.templates[template_key]
                
        except Exception as e:
            print(f"Error removing {template_path}: {e}")
    
    return removed_count


def _get_next_variant_letter(class_name):
    """
    Find the next available variant letter for a class.
    Checks stored_templates for existing variants (A, B, C, etc.)
    """
    import string
    
    # Check what variants already exist for this class
    class_name_lower = class_name.lower()
    existing_variants = set()
    
    if st.session_state.templates:
        for template_key in st.session_state.templates.keys():
            # Key format: "class_name/filename.png"
            if '/' in template_key:
                folder, filename = template_key.split('/', 1)
                folder_lower = folder.lower()
                
                if folder_lower == class_name_lower:
                    # Extract variant letter from filename
                    # Expected: drone-name-A.png, drone-name-B.png, etc.
                    if '-' in filename:
                        parts = filename.rsplit('-', 1)
                        if len(parts) == 2:
                            variant_part = parts[1].replace('.png', '')
                            if len(variant_part) == 1 and variant_part.isalpha():
                                existing_variants.add(variant_part.upper())
    
    # Find next available letter
    for letter in string.ascii_uppercase:
        if letter not in existing_variants:
            return letter
    
    # If all letters used, start with AA, AB, etc.
    return 'AA'


def _deduplicate_saved_templates(class_folders):
    """
    Deduplicate saved templates at the end of train mode.
    Compares templates visually and removes duplicates, keeping only the best one.
    
    Args:
        class_folders: List of class folder paths that were processed
        
    Returns:
        int: Number of duplicate templates removed
    """
    removed_count = 0
    templates_dir = "stored_templates"
    
    try:
        for class_folder in class_folders:
            class_name = class_folder.name.lower()
            class_path = os.path.join(templates_dir, class_name)
            
            if not os.path.exists(class_path):
                continue
            
            # Get all templates in this class
            template_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
            
            if len(template_files) < 2:
                continue  # No duplicates possible
            
            # Load all templates
            templates = []
            for filename in template_files:
                filepath = os.path.join(class_path, filename)
                try:
                    img = Image.open(filepath)
                    img_cv = pil_to_cv(img)
                    templates.append({
                        'filename': filename,
                        'filepath': filepath,
                        'image': img_cv,
                        'size': img.size
                    })
                except Exception as e:
                    continue
            
            # Find and remove duplicates
            to_remove = set()
            
            for i, template1 in enumerate(templates):
                if template1['filename'] in to_remove:
                    continue
                    
                for j, template2 in enumerate(templates[i+1:], i+1):
                    if template2['filename'] in to_remove:
                        continue
                    
                    # Check if templates are similar
                    if _are_templates_duplicate(template1['image'], template2['image']):
                        # Keep the one with better quality (larger file size typically = less compression)
                        size1 = os.path.getsize(template1['filepath'])
                        size2 = os.path.getsize(template2['filepath'])
                        
                        # Remove the smaller file (likely more compressed/lower quality)
                        if size1 >= size2:
                            to_remove.add(template2['filename'])
                        else:
                            to_remove.add(template1['filename'])
            
            # Delete duplicate files
            for filename in to_remove:
                filepath = os.path.join(class_path, filename)
                try:
                    os.remove(filepath)
                    removed_count += 1
                    
                    # Also remove from session state templates
                    template_key = f"{class_name}/{filename}"
                    if template_key in st.session_state.templates:
                        del st.session_state.templates[template_key]
                    
                    # Remove from template_folders if it exists
                    if hasattr(st.session_state, 'template_folders'):
                        for folder_name, folder_templates in st.session_state.template_folders.items():
                            if template_key in folder_templates:
                                del folder_templates[template_key]
                                
                except Exception as e:
                    continue
    
    except Exception as e:
        pass  # Silently handle errors, not critical
    
    return removed_count


def _are_templates_duplicate(img1_cv, img2_cv, similarity_threshold=0.92):
    """
    Check if two templates are duplicates using template matching.
    
    Args:
        img1_cv: First template image (OpenCV format)
        img2_cv: Second template image (OpenCV format)
        similarity_threshold: Threshold for considering templates as duplicates (0.92 = 92% similar)
        
    Returns:
        bool: True if templates are duplicates
    """
    try:
        # Resize to same size for comparison if needed
        h1, w1 = img1_cv.shape[:2]
        h2, w2 = img2_cv.shape[:2]
        
        if (h1, w1) != (h2, w2):
            # Different sizes - try to match smaller in larger
            if h1 * w1 < h2 * w2:
                template = img1_cv
                image = img2_cv
            else:
                template = img2_cv
                image = img1_cv
            
            # Ensure template is smaller than image
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                return False  # Can't compare
            
            # Convert to grayscale
            if len(template.shape) == 3:
                template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            else:
                template_gray = template
                
            if len(image.shape) == 3:
                image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            else:
                image_gray = image
            
            # Template match
            result = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv.minMaxLoc(result)
            
            return max_val >= similarity_threshold
        else:
            # Same size - compare directly using correlation
            if len(img1_cv.shape) == 3:
                img1_gray = cv.cvtColor(img1_cv, cv.COLOR_BGR2GRAY)
            else:
                img1_gray = img1_cv
                
            if len(img2_cv.shape) == 3:
                img2_gray = cv.cvtColor(img2_cv, cv.COLOR_BGR2GRAY)
            else:
                img2_gray = img2_cv
            
            # Normalize and compute correlation
            img1_norm = cv.normalize(img1_gray, None, 0, 255, cv.NORM_MINMAX)
            img2_norm = cv.normalize(img2_gray, None, 0, 255, cv.NORM_MINMAX)
            
            # Calculate structural similarity
            correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
            
            return correlation >= similarity_threshold
            
    except Exception as e:
        return False  # If comparison fails, assume not duplicate
