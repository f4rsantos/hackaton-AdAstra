import streamlit as st
import time
import os
import pandas as pd
import numpy as np
from PIL import Image
from functions import (
    pil_to_cv, cv_to_pil, draw_detection_boxes, draw_detection_boxes_no_labels,
    assign_drone_ids, add_unidentified_drone
)

def render(show_all_matches, min_confidence, draw_boxes):
    """Render the Results Analysis tab"""
    st.markdown('<h2 class="sub-header">📊 Detection Results Analysis</h2>', unsafe_allow_html=True)
    
    # Ensure live_feed_results is initialized
    if 'live_feed_results' not in st.session_state:
        st.session_state.live_feed_results = []
    
    # Results source selection - both options should always be available
    result_source = st.radio(
        "📊 Results Source",
        ["🔍 Pattern Detection Results", "📡 Live Feed Results"],
        horizontal=True,
        help="Choose which results to analyze",
        key="results_source_selector"
    )
    
    if result_source == "🔍 Pattern Detection Results":
        _render_pattern_detection_results(show_all_matches, min_confidence, draw_boxes)
    elif result_source == "📡 Live Feed Results":
        _render_live_feed_results()


def _render_pattern_detection_results(show_all_matches, min_confidence, draw_boxes):
    """Render pattern detection results"""
    # Check if detection results exist and are not empty
    has_results = (hasattr(st.session_state, 'detection_results') and 
                   st.session_state.detection_results and 
                   len(st.session_state.detection_results) > 0)
    
    if not has_results:
        st.info("🔍 Run pattern detection first to see results here")
        return
    
    # Summary statistics
    total_images = len(st.session_state.detection_results)
    total_matches = sum(len(result.get('matches', [])) for result in st.session_state.detection_results)
    images_with_matches = sum(1 for result in st.session_state.detection_results if result.get('matches', []))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images Processed", total_images)
    with col2:
        st.metric("Total Matches", total_matches)
    with col3:
        st.metric("Images with Matches", images_with_matches)
    
    st.markdown("---")
    
    # Debug info (can be removed later)
    with st.expander("🔍 Debug Info", expanded=True):
        st.write(f"Total results in session: {len(st.session_state.detection_results)}")
        st.write(f"Sample result keys: {list(st.session_state.detection_results[0].keys()) if st.session_state.detection_results else 'No results'}")
        if st.session_state.detection_results:
            first_result = st.session_state.detection_results[0]
            st.write(f"First result filename: {first_result.get('filename', 'No filename')}")
            st.write(f"First result has chunk_info: {'chunk_info' in first_result}")
            if first_result.get('image'):
                img_size = first_result['image'].size if hasattr(first_result['image'], 'size') else 'Unknown'
                st.write(f"**First result image size: {img_size}**")
                st.write(f"Expected chunk size: 2048x384 or smaller")
            if 'chunk_info' in first_result:
                st.write(f"Chunk info: {first_result['chunk_info']}")
    
    # Display results for each image/chunk
    for result_idx, result in enumerate(st.session_state.detection_results):
        # Check if this is a chunk result (from chunked processing)
        is_chunk_result = '_chunk_' in result.get('filename', '') or 'chunk_info' in result
        
        # Show chunks always, or show images if they have matches or show_all_matches is enabled
        should_display = is_chunk_result or result.get('matches', []) or show_all_matches
        
        if should_display:
            # Display chunk info if this is a chunk
            if is_chunk_result and 'chunk_info' in result:
                chunk_info = result['chunk_info']
                chunk_num = chunk_info.get('chunk_number', '?')
                total_chunks = chunk_info.get('total_chunks', '?')
                st.markdown(f"### 📷 Chunk {chunk_num}/{total_chunks} - {result.get('original_filename', result.get('filename', 'Unknown'))}")
            else:
                st.markdown(f"### 📷 {result.get('filename', 'Unknown')}")
            
            # Filter matches by confidence
            filtered_matches = [m for m in result.get('matches', []) if m.get('confidence', 0) >= min_confidence]
            
            # Add toggle for showing/hiding labels
            show_labels = True
            if filtered_matches:
                show_labels = st.checkbox("🏷️ Show Labels", value=True, key=f"labels_{result_idx}_{result['filename']}", 
                                        help="Toggle visibility of drone ID labels on the detection image")
            
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("**Original Spectrogram**")
                display_name = result.get('filename', 'Unknown')
                if result.get('image'):
                    st.image(result['image'], caption=f"Original: {display_name}", width='stretch')
                else:
                    st.warning("⚠️ Image data not available")
            
            with col_img2:
                st.markdown("**Detection Results**")
                
                if filtered_matches and draw_boxes and result.get('image'):
                    # Draw bounding boxes
                    test_cv = pil_to_cv(result['image'])
                    if show_labels:
                        result_image = draw_detection_boxes(test_cv, filtered_matches, min_confidence)
                    else:
                        result_image = draw_detection_boxes_no_labels(test_cv, filtered_matches, min_confidence)
                    result_pil = cv_to_pil(result_image)
                    st.image(result_pil, caption=f"Detections: {display_name}", width='stretch')
                elif result.get('image'):
                    st.image(result['image'], caption=f"No detections: {display_name}", width='stretch')
                else:
                    st.warning("⚠️ Image data not available")
            
            # Detection statistics and manual store button
            if filtered_matches:
                confidences = [m.get('confidence', 0) for m in filtered_matches if 'confidence' in m]
                avg_conf = np.mean(confidences) if confidences else 0
                templates = set(m.get('template_name', 'Unknown') for m in filtered_matches if 'template_name' in m)
                
                st.markdown(f"""
                <div class="detection-stats">
                    <h4>🎯 Detection Statistics</h4>
                    <p><strong>Total Matches:</strong> {len(filtered_matches)}</p>
                    <p><strong>Average Confidence:</strong> {avg_conf:.3f}</p>
                    <p><strong>Templates Matched:</strong> {', '.join(templates) if templates else 'None'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Manual store button for unidentified drones
                unidentified_in_result = [m for m in filtered_matches if m.get('template_name') == 'Unidentified Drone']
                if unidentified_in_result:
                    st.markdown("### 🔄 Manual Storage Control")
                    col_store1, col_store2 = st.columns([2, 1])
                    
                    with col_store1:
                        st.write(f"Found {len(unidentified_in_result)} unidentified drone(s) in this chunk/image.")
                        
                    with col_store2:
                        filename_key = result.get('filename', f'result_{result_idx}')
                        if st.button(f"📦 Store Unidentified Drones", key=f"store_{result_idx}_{filename_key}"):
                            stored_count = 0
                            if result.get('image'):
                                img_h, img_w = result['image'].size[1], result['image'].size[0]
                                
                                for unidentified_match in unidentified_in_result:
                                    if add_unidentified_drone(unidentified_match, result['image'], filename_key, img_w, img_h, min_confidence):
                                        stored_count += 1
                                
                                if stored_count > 0:
                                    st.success(f"✅ Stored {stored_count} new unidentified drone(s)!")
                                else:
                                    st.info("ℹ️ No new drones stored (duplicates or validation failures)")
                                st.rerun()
                            else:
                                st.error("❌ Image data not available for storage")
                
                # Detailed results table
                result_name = result.get('filename', 'Unknown')
                with st.expander(f"📊 Detailed Results for {result_name}"):
                    # Assign drone IDs for consistent labeling with images
                    filtered_matches_with_ids = assign_drone_ids(filtered_matches.copy())
                    
                    results_data = []
                    for m in filtered_matches_with_ids:
                        # Simplify template name for display
                        display_template = 'US' if m['template_name'] == 'Unidentified Drone' else m['template_name']
                        
                        row = {
                            'Drone ID': m['drone_id'],
                            'Template': display_template,
                            'Confidence': f"{m['confidence']:.3f}",
                            'Detection Type': m.get('detection_type', 'full'),
                            'Position': f"({m['x']}, {m['y']})",
                            'Center': f"({m['center_x']}, {m['center_y']})",
                            'Size': f"{m['width']}x{m['height']}",
                            'Partial': 'Yes' if m.get('partial', False) else 'No',
                            'At Border': 'Yes' if m.get('at_border', False) else 'No',
                            'Storage': 'Eligible' if m.get('storage_eligible', True) else 'Excluded'
                        }
                        
                        # Add duplicate count if this is an averaged result
                        if m.get('duplicate_count', 0) > 1:
                            row['Duplicates Merged'] = str(m['duplicate_count'])
                        
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, width="stretch")
                    
                    # Show detection type statistics
                    _show_detection_statistics(filtered_matches)
            else:
                result_name = result.get('filename', 'this chunk/image')
                st.info(f"✅ Chunk processed - No drone patterns detected in {result_name} with confidence ≥ {min_confidence:.2f}")
            
            st.markdown("---")


def _render_live_feed_results():
    """Render live feed results"""
    
    # Always show content, even if empty
    result_count = len(st.session_state.live_feed_results) if st.session_state.live_feed_results else 0
    
    if result_count == 0:
        st.info("""
        📡 **No Live Feed Detections Yet**
        
        Live feed detections appear here when:
        - Running **Live Feed Processing** (non-train mode) on files
        - Running **Train Mode** on data that matches existing templates
        
        💡 **To see detections in Train Mode:**
        1. Run Train Mode once to extract templates from your labeled data
        2. Run Train Mode again on the same data - it will now match against the templates you just created
        3. Or switch to regular Live Feed processing to detect patterns in new files
        
        **Current Status:**
        - Detections stored: {result_count}
        - Templates in library: {len(st.session_state.get('templates', {}))}
        """)
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        return
    
    # Show detections if we have any
    st.caption(f"📊 Showing {result_count} detections")
    
    if result_count > 0:
        # Display live feed detection results
        st.markdown("### 📊 Live Feed Detection Summary")
        
        # Group detections by class_folder
        detections_by_class = {}
        for detection in st.session_state.live_feed_results:
            class_folder = detection.get('class_folder', 'Unknown')
            if class_folder not in detections_by_class:
                detections_by_class[class_folder] = []
            detections_by_class[class_folder].append(detection)
        
        # Live feed statistics
        total_live_detections = len(st.session_state.live_feed_results)
        unique_templates = set()
        confidence_scores = []
        
        for detection in st.session_state.live_feed_results:
            if 'template_name' in detection:
                unique_templates.add(detection['template_name'])
            if 'confidence' in detection:
                confidence_scores.append(detection['confidence'])
        
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
        
        with col_summary1:
            st.metric("📈 Total Detections", total_live_detections)
        
        with col_summary2:
            st.metric("📂 Classes Processed", len(detections_by_class))
        
        with col_summary3:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            st.metric("📊 Avg Confidence", f"{avg_confidence:.3f}")
        
        with col_summary4:
            st.metric("🎯 Unique Templates", len(unique_templates))
        
        st.markdown("---")
        
        # Display detections grouped by class folder
        st.markdown("### 🔴 Live Feed by Class")
        
        for class_folder, class_detections in detections_by_class.items():
            with st.expander(f"📂 **{class_folder}** ({len(class_detections)} detections)", expanded=True):
                st.markdown(f"**Class:** `{class_folder}` | **Detections:** {len(class_detections)}")
                
                # Get unique files in this class
                files_with_detections = {}
                for det in class_detections:
                    filename = det.get('filename', 'unknown')
                    if filename not in files_with_detections:
                        files_with_detections[filename] = []
                    files_with_detections[filename].append(det)
                
                # Show sample of recent detections (limit to last 10 files to avoid clutter)
                recent_files = list(files_with_detections.items())[-10:]
                
                st.markdown(f"**Files with detections:** {len(files_with_detections)} (showing last {len(recent_files)})")
                
                # Create a live feed style display
                for filename, file_dets in recent_files:
                    col_img, col_info = st.columns([2, 1])
                    
                    with col_img:
                        # Try to load and display the image with detection boxes
                        try:
                            file_path = file_dets[0].get('file_path')
                            if file_path and os.path.exists(file_path):
                                # Load the PNG image
                                from PIL import Image
                                img = Image.open(file_path)
                                
                                # Draw detection boxes
                                img_cv = pil_to_cv(img)
                                img_with_boxes = draw_detection_boxes(img_cv, file_dets, 0.6)
                                img_pil = cv_to_pil(img_with_boxes)
                                
                                st.image(img_pil, caption=f"📷 {filename}", width='stretch')
                            else:
                                st.caption(f"📷 {filename} (file not found)")
                        except Exception as e:
                            st.caption(f"📷 {filename} (display error: {str(e)})")
                    
                    with col_info:
                        st.markdown("**Detection Info:**")
                        st.markdown(f"**Matches:** {len(file_dets)}")
                        
                        # Show templates matched (handle both template and unidentified detections)
                        templates_matched = list(set(
                            d.get('template_name', 'Unidentified Signal') for d in file_dets
                        ))
                        st.markdown(f"**Templates:**")
                        for tmpl in templates_matched:
                            st.markdown(f"- {tmpl}")
                        
                        # Average confidence
                        confidences = [d.get('confidence', 0) for d in file_dets]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        st.markdown(f"**Avg Conf:** {avg_conf:.3f}")
                    
                    st.markdown("---")
        
        # Show detailed table
        st.markdown("### 📋 Detailed Detection Table")
        
        with st.expander("View all detections in table format"):
            detection_data = []
            # Iterate through result objects, then through matches in each result
            for result in st.session_state.live_feed_results:
                filename = result.get('filename', 'unknown')
                for match in result.get('matches', []):
                    detection_data.append({
                        'Filename': filename,
                        'Template': match.get('template_name', 'Unidentified Signal'),
                        'Confidence': f"{match.get('confidence', 0):.3f}",
                        'Position': f"({match.get('x', 0)}, {match.get('y', 0)})",
                        'Size': f"{match.get('width', 0)}x{match.get('height', 0)}",
                        'Type': match.get('detection_type', 'template')
                    })
            
            if detection_data:
                detection_df = pd.DataFrame(detection_data)
                st.dataframe(detection_df, width='stretch')
            else:
                st.info("No detections to display")
        
        # Live feed controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        with col_ctrl1:
            if st.button("🗑️ Clear Live Results"):
                st.session_state.live_feed_results = []
                st.success("✅ Live feed results cleared")
                st.rerun()
        
        with col_ctrl2:
            if st.button("💾 Save Current Results"):
                # Save results to session for later analysis
                if 'saved_live_results' not in st.session_state:
                    st.session_state.saved_live_results = []
                
                import datetime
                save_timestamp = datetime.datetime.now().isoformat()
                
                st.session_state.saved_live_results.append({
                    'timestamp': save_timestamp,
                    'detection_count': len(st.session_state.live_feed_results),
                    'results': st.session_state.live_feed_results.copy(),
                    'stats': st.session_state.live_feed_stats.copy()
                })
                
                st.success(f"✅ Results saved with {len(st.session_state.live_feed_results)} detections")


def _show_detection_statistics(filtered_matches):
    """Show detection type statistics"""
    detection_types = {}
    merged_count = 0
    total_duplicates_merged = 0
    border_count = 0
    storage_excluded_count = 0
    
    for m in filtered_matches:
        det_type = m.get('detection_type', 'full')
        detection_types[det_type] = detection_types.get(det_type, 0) + 1
        
        if 'merged' in det_type or m.get('merge_reason'):
            merged_count += 1
        
        if m.get('duplicate_count', 0) > 1:
            total_duplicates_merged += m['duplicate_count'] - 1
            
        if m.get('at_border', False):
            border_count += 1
            
        if not m.get('storage_eligible', True):
            storage_excluded_count += 1
    
    if len(detection_types) > 1 or merged_count > 0 or total_duplicates_merged > 0 or border_count > 0:
        st.markdown("**Detection Statistics:**")
        for det_type, count in detection_types.items():
            display_name = det_type.replace('_', ' ').title()
            if 'merged' in det_type:
                display_name += " (Overlapping Combined)"
            st.write(f"- {display_name}: {count}")
        
        if total_duplicates_merged > 0:
            st.write(f"- **Total Overlapping Detections Merged:** {total_duplicates_merged}")
            st.write(f"- **Merged Detection Groups:** {merged_count}")
            
        if border_count > 0:
            st.write(f"- **Border Detections:** {border_count}")
            st.write(f"- **Excluded from Storage:** {storage_excluded_count}")
            
        if total_duplicates_merged > 0:
            st.info("ℹ️ Overlapping detections were automatically merged to prevent duplicate results from the same object.")
        if border_count > 0:
            st.info("ℹ️ Border detections are shown for analysis but excluded from storage to avoid incomplete signal captures.")
