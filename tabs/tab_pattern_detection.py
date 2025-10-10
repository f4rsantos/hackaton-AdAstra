import streamlit as st
import zipfile
import io
import time
import os
from PIL import Image
from pathlib import Path
from functions import (
    pil_to_cv, clean_template_name, run_unified_detection,
    auto_label_process, save_template_to_class_folder
)
import cv2 as cv
from image_preprocessing_pipeline import preprocess_images_batch
from parallel_gpu_processor import ParallelGPUProcessor
from sequential_cpu_processor import SequentialCPUProcessor

def render(threshold, min_confidence, border_threshold, enable_border_detection,
           merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
           green_min_area, green_overlap_threshold, colored_merge_threshold,
           parallel_config, show_all_matches, draw_boxes):
    """Render the Pattern Detection tab"""
    st.markdown('<h2 class="sub-header">🔍 Drone Pattern Detection</h2>', unsafe_allow_html=True)
    
    # Train Mode checkbox at the top
    col_train, col_mode = st.columns(2)
    
    with col_train:
        enable_train_mode = st.checkbox(
            "🎓 Enable Train Mode",
            value=False,
            help="Save unique patterns from each image with filename-based naming (e.g., image-A, image-B)"
        )
    
    with col_mode:
        single_drone_mode = st.checkbox(
            "📡 Smart Classification",
            value=False,
            help="Auto-classify unidentified signals by comparing height, vertical position, and signal characteristics with known detections"
        )
    
    if enable_train_mode:
        st.info("🎓 **Train Mode Active:** Will save unique patterns from each image as templates (filename-A, filename-B, etc.)")
    
    if single_drone_mode:
        st.info("📡 **Smart Classification Active:** Unidentified signals will be auto-classified by matching height, y-position, and other characteristics with known signals")
    
    # SigMF Metadata Path Input
    with st.expander("📡 SigMF Metadata Configuration (Optional)", expanded=False):
        st.markdown("""
        If your spectrogram images have corresponding `.sigmf-meta` files, specify where to find them.
        Detection results will be automatically saved as SigMF annotations.
        """)
        
        sigmf_meta_path = st.text_input(
            "SigMF .sigmf-meta Path",
            value="",
            placeholder="e.g., /path/to/metadata/ or /path/to/file.sigmf-meta",
            help="Path to directory containing .sigmf-meta files, or path to specific .sigmf-meta file. Leave blank for auto-detection in same directory as images."
        )
        
        if sigmf_meta_path:
            if os.path.isdir(sigmf_meta_path):
                st.success(f"✅ Using metadata directory: `{sigmf_meta_path}`")
            elif os.path.isfile(sigmf_meta_path) and sigmf_meta_path.endswith('.sigmf-meta'):
                st.success(f"✅ Using specific metadata file: `{os.path.basename(sigmf_meta_path)}`")
            else:
                st.warning("⚠️ Path doesn't exist or isn't a valid .sigmf-meta file/directory")
    
    # Allow detection even without templates if colored rectangle detection is enabled
    can_run_detection = st.session_state.templates or detect_green_rectangles
    
    if not can_run_detection:
        st.warning("⚠️ Please either upload template patterns or enable 'Detect Colored Areas as Unidentified Signals' to run detection")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Upload Test Spectrograms**")
            
            # Upload method selection
            upload_method = st.radio(
                "Upload Method",
                ["Multiple Images", "ZIP Archive"],
                horizontal=True
            )
            
            test_images = []
            
            if upload_method == "Multiple Images":
                uploaded_images = st.file_uploader(
                    "Upload spectrogram images for analysis",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="Upload spectrogram images to search for drone patterns"
                )
                test_images = uploaded_images if uploaded_images else []
            
            else:  # ZIP Archive
                uploaded_zip = st.file_uploader(
                    "Upload ZIP file containing spectrogram images",
                    type=['zip'],
                    help="Upload a ZIP file containing spectrogram images"
                )
                
                if uploaded_zip:
                    try:
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            img_files = [f for f in zip_ref.namelist() 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            
                            if img_files:
                                st.success(f"✅ Found {len(img_files)} images in ZIP")
                                
                                test_images = []
                                for img_file in img_files:
                                    with zip_ref.open(img_file) as f:
                                        img_content = f.read()
                                        img_obj = io.BytesIO(img_content)
                                        img_obj.name = img_file
                                        test_images.append(img_obj)
                            else:
                                st.warning("⚠️ No image files found in ZIP")
                    except Exception as e:
                        st.error(f"❌ Error reading ZIP: {str(e)}")
        
        with col2:
            st.markdown("**Detection Status**")
            
            # Auto-Label information box
            with st.expander("ℹ️ How Auto-Label Works", expanded=False):
                st.markdown("""
                **Auto-Label Process:**
                1. 🔍 Runs pattern detection on all test images
                2. 📊 Finds unidentified signals (colored rectangles)
                3. 📈 Sorts unidentified signals by confidence (highest first)
                4. 🏷️ Automatically creates templates from high-confidence drones (≥70%)
                5. 🔄 Re-runs detection with updated templates
                6. ⏰ Repeats until no high-confidence unidentified signals remain
                
                **Requirements:**
                - ✅ 'Detect Colored Areas as Unidentified Signals' must be enabled
                - ✅ Test images must be uploaded
                - ⚡ Process can be stopped at any time
                """)
            
            if test_images:
                # Determine detection mode
                if st.session_state.templates and detect_green_rectangles:
                    detection_mode = 'Template + Colored Areas'
                elif st.session_state.templates:
                    detection_mode = 'Template Only'
                else:
                    detection_mode = 'Colored Areas Only'
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>📊 Ready for Detection</h4>
                    <p><strong>Test Images:</strong> {len(test_images)}</p>
                    <p><strong>Detection Mode:</strong> {detection_mode}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detection mode buttons - Full width button
                # Regular detection button
                if st.button("🚀 Start Pattern Detection", type="primary", width='stretch'):
                    # Store test images in session state for auto-label access
                    st.session_state.current_test_images = test_images
                    st.session_state.detection_results = []
                    
                    # Check if auto-label should run after detection
                    auto_label_after_detection = getattr(st.session_state, 'auto_label_enabled', False)
                    run_auto_label = auto_label_after_detection and detect_green_rectangles
                    
                    # Debug info for auto-label
                    if auto_label_after_detection:
                        st.info(f"🔄 Auto-Label is enabled. Colored areas detection: {detect_green_rectangles}")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    timing_text = st.empty()
                    
                    # Fixed preprocessing parameters (always resize and chunk)
                    target_width = 31778
                    target_height = 384
                    chunk_width = 2048
                    chunk_height = 384
                    
                    # Prepare detection parameters
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
                        'parallel_config': parallel_config,
                        'sigmf_meta_path': sigmf_meta_path if sigmf_meta_path else None,  # Add SigMF path
                        'single_drone_mode': single_drone_mode  # Add single drone mode
                    }
                    
                    # STEP 1: Preprocess images (resize to 31778x384, then chunk into 2048x384 pieces)
                    status_text.text("🔧 Preprocessing images (resize + chunk)...")
                    
                    preprocessed_results = preprocess_images_batch(
                        test_images,
                        target_width=target_width,
                        target_height=target_height,
                        chunk_width=chunk_width,
                        chunk_height=chunk_height
                    )
                    
                    if not preprocessed_results:
                        status_text.error("❌ Failed to preprocess images")
                        st.stop()
                    
                    # Convert preprocessed chunks to file objects for detection
                    all_chunks = []
                    chunk_metadata = []  # Store metadata for reassembly
                    
                    for img_result in preprocessed_results:
                        original_filename = img_result['original_filename']  # Fixed: use 'original_filename'
                        base_name = os.path.splitext(original_filename)[0]
                        original_size = img_result['original_size']  # (width, height) of ORIGINAL image
                        resized_size = img_result['resized_image'].size  # (width, height) of RESIZED image
                        
                        for chunk_idx, chunk_info in enumerate(img_result['chunks']):
                            chunk_img = chunk_info['chunk']  # Fixed: use 'chunk' not 'image'
                            chunk_filename = f"{base_name}_chunk_{chunk_idx}.png"
                            
                            # Convert PIL to file object
                            chunk_buffer = io.BytesIO()
                            chunk_img.save(chunk_buffer, format='PNG')
                            chunk_buffer.seek(0)
                            chunk_buffer.name = chunk_filename
                            
                            all_chunks.append(chunk_buffer)
                            chunk_metadata.append({
                                'original_filename': original_filename,
                                'chunk_idx': chunk_idx,
                                'offset_x': chunk_info['x_offset'],  # Fixed: use 'x_offset'
                                'offset_y': chunk_info['y_offset'],  # Fixed: use 'y_offset'
                                'resized_image': img_result['resized_image'],  # Fixed: use 'resized_image'
                                'original_size': original_size,  # CRITICAL: for SigMF coordinate transformation
                                'resized_size': resized_size  # CRITICAL: for SigMF coordinate transformation
                            })
                    
                    status_text.text(f"✅ Preprocessed into {len(all_chunks)} chunks from {len(test_images)} images")
                    time.sleep(0.5)
                    
                    # STEP 2: Run detection on chunks
                    def manual_status_callback(message):
                        status_text.text(message)
                        # Update progress for manual detection
                        if "Detecting" in message and "/" in message:
                            progress_bar.progress(min(0.9, len(st.session_state.detection_results) / len(all_chunks)))
                    
                    # Run detection on ALL chunks
                    chunk_detection_results, unidentified_found, timing_info = run_unified_detection(
                        all_chunks, 
                        detection_params, 
                        manual_status_callback
                    )
                    
                    # STEP 3: Add chunk metadata to results for Results Analysis tab
                    status_text.text("✅ Processing complete! Adding chunk metadata...")
                    
                    # Enhance each chunk result with metadata for Results Analysis tab
                    detection_results = []
                    for chunk_idx, chunk_result in enumerate(chunk_detection_results):
                        metadata = chunk_metadata[chunk_idx]
                        
                        # Add chunk info to result for Results Analysis tab to display properly
                        enhanced_result = chunk_result.copy()
                        enhanced_result['chunk_info'] = {
                            'chunk_number': chunk_idx + 1,
                            'total_chunks': len(all_chunks),
                            'offset_x': metadata['offset_x'],
                            'offset_y': metadata['offset_y'],
                            'original_size': metadata['original_size'],  # CRITICAL: (orig_width, orig_height)
                            'resized_size': metadata['resized_size']  # CRITICAL: (resized_width, resized_height)
                        }
                        enhanced_result['original_filename'] = metadata['original_filename']
                        
                        detection_results.append(enhanced_result)
                    
                    # Update session state with results
                    st.session_state.detection_results = detection_results
                    
                    # Save to SigMF if path was provided (do it AFTER chunk_info is added!)
                    if sigmf_meta_path and detection_results:
                        status_text.text("📡 Saving SigMF annotations...")
                        from functions import save_detections_to_sigmf
                        save_detections_to_sigmf(
                            detection_results,
                            sigmf_meta_path_override=sigmf_meta_path,
                            status_callback=lambda msg: status_text.text(msg)
                        )
                    
                    # Train Mode: Save unique patterns from unidentified_drones (what the gallery uses)
                    if enable_train_mode and detect_green_rectangles and st.session_state.unidentified_drones:
                        status_text.text("🎓 Train Mode: Clustering unidentified signals by dimensions...")
                        total_patterns_saved = 0
                        
                        # Group all unidentified drones from the current detection by filename
                        filename_groups = {}
                        for drone in st.session_state.unidentified_drones:
                            fname = drone['filename']
                            if fname not in filename_groups:
                                filename_groups[fname] = []
                            filename_groups[fname].append(drone)
                        
                        # Process each file's drones
                        for filename, drones in filename_groups.items():
                            base_name = os.path.splitext(filename)[0]
                            
                            # Filter for high confidence
                            high_conf_drones = [d for d in drones if d.get('confidence', 0.0) >= 0.5]
                            
                            if not high_conf_drones:
                                continue
                            
                            # Cluster by dimensions (width, height, area)
                            clusters = []
                            
                            for drone in high_conf_drones:
                                w = drone['width']
                                h = drone['height']
                                area = drone['area']
                                
                                # Find matching cluster by similar dimensions
                                found_cluster = False
                                for cluster in clusters:
                                    rep = cluster[0]
                                    rep_w = rep['width']
                                    rep_h = rep['height']
                                    rep_area = rep['area']
                                    
                                    # Check if dimensions are similar (within 20% tolerance)
                                    w_ratio = min(w, rep_w) / max(w, rep_w) if max(w, rep_w) > 0 else 0
                                    h_ratio = min(h, rep_h) / max(h, rep_h) if max(h, rep_h) > 0 else 0
                                    area_ratio = min(area, rep_area) / max(area, rep_area) if max(area, rep_area) > 0 else 0
                                    
                                    # Similar if width, height, and area are all within 80% similarity
                                    if w_ratio >= 0.8 and h_ratio >= 0.8 and area_ratio >= 0.8:
                                        cluster.append(drone)
                                        found_cluster = True
                                        break
                                
                                if not found_cluster:
                                    clusters.append([drone])
                            
                            # Sort clusters by size (larger = more duplicates = better pattern)
                            clusters.sort(key=lambda c: len(c), reverse=True)
                            
                            # Save ONE representative from each cluster
                            for idx, cluster in enumerate(clusters):
                                suffix = chr(ord('A') + idx)
                                pattern_name = f"{base_name}-{suffix}"
                                
                                # Pick highest confidence drone from cluster
                                representative = max(cluster, key=lambda d: d.get('confidence', 0.0))
                                
                                # Use the drone's image directly (it's already cropped with padding)
                                drone_image = representative['image']
                                
                                # Save to "unidentified" class folder
                                success, message, filepath = save_template_to_class_folder(
                                    drone_image,
                                    pattern_name,
                                    "unidentified"
                                )
                                
                                if success:
                                    total_patterns_saved += 1
                                    cluster_size = len(cluster)
                                    if cluster_size > 1:
                                        status_text.text(f"💾 Saved {pattern_name} (found {cluster_size}x - good pattern!)")
                                    time.sleep(0.1)
                        
                        if total_patterns_saved > 0:
                            status_text.text(f"🎓 Train Mode: Saved {total_patterns_saved} representative patterns")
                            time.sleep(1)
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("✅ Detection complete!")
                    
                    # Display timing information below progress bar
                    if timing_info:
                        total_time = timing_info.get('total_time', 0)
                        mode_name = timing_info.get('mode', 'Standard')
                        avg_time = total_time / len(detection_results) if detection_results else 0
                        total_detections = sum(len(r.get('matches', [])) for r in detection_results)
                        
                        timing_text.markdown(f"""
                        **⏱️ Performance Metrics:**
                        - **Processing Mode:** {mode_name}
                        - **Total Time:** {total_time:.2f}s
                        - **Images Processed:** {len(test_images)}
                        - **Results Generated:** {len(detection_results)}
                        - **Average per Image:** {avg_time:.2f}s
                        - **Total Detections:** {total_detections}
                        """)
                        
                        # Success message
                        st.success(f"🎉 {mode_name}: Processed {len(test_images)} images in {total_time:.2f}s with {total_detections} detections!")
                        
                        # Store unidentified_found for auto-label compatibility
                        # Note: unidentified_found already comes from run_unified_detection, but we can also extract from results if needed
                        if not unidentified_found:
                            unidentified_found = []
                            for result in detection_results:
                                for match in result.get('matches', []):
                                    if match.get('template_name') == 'Unidentified Drone':
                                        unidentified_found.append(match)
                    
                    # Auto-label disabled for chunked processing (would overwrite chunk results with full images)
                    if False and run_auto_label:
                        st.warning("⚠️ Auto-Label is disabled during chunked processing to preserve individual chunk results.")
                        
                        # Create containers for auto-label progress
                        auto_progress_container = st.empty()
                        auto_status_container = st.empty()
                        
                        try:
                            def auto_progress_callback(iteration, total_labeled):
                                with auto_progress_container.container():
                                    st.progress(min(iteration / 20, 1.0))
                                    st.text(f"Auto-Label Iteration: {iteration}, Labeled: {total_labeled}")
                            
                            def auto_status_callback(message):
                                with auto_status_container.container():
                                    st.text(message)
                            
                            # Prepare detection parameters for auto-label
                            detection_params_auto = {
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
                                'parallel_config': parallel_config,
                                'sigmf_meta_path': sigmf_meta_path if sigmf_meta_path else None  # Add SigMF path
                            }
                            
                            # Run auto-label process
                            auto_result = auto_label_process(
                                test_images,
                                detection_params_auto,
                                progress_callback=auto_progress_callback,
                                status_callback=auto_status_callback
                            )
                            
                            # Display auto-label results
                            if auto_result and auto_result.get("success"):
                                with auto_status_container.container():
                                    st.success(f"✅ Auto-Label Complete: {auto_result['message']}")
                                with auto_progress_container.container():
                                    st.progress(1.0)
                                
                                if auto_result.get('total_labeled', 0) > 0:
                                    st.success(f"""
                                    🎉 Auto-Label Results:
                                    - **Iterations:** {auto_result.get('iterations', 0)}
                                    - **New Signals:** {auto_result.get('total_labeled', 0)}
                                    """)
                            else:
                                with auto_status_container.container():
                                    st.warning("⚠️ Auto-Label completed with no new signals created")
                                    
                        except Exception as e:
                            with auto_status_container.container():
                                st.error(f"❌ Auto-Label failed: {str(e)}")
            
            else:
                st.info("👆 Upload test images to begin detection")

