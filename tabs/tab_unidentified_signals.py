import streamlit as st
import zipfile
import io
import numpy as np
import pandas as pd
import cv2 as cv
from functions import save_drone_as_template, cv_to_pil, draw_dashed_rectangle

def render(min_confidence, green_min_area, detect_green_rectangles):
    """Render the Unidentified Signals tab"""
    st.markdown('<h2 class="sub-header">🚁 Unidentified Drones</h2>', unsafe_allow_html=True)
    
    if not st.session_state.unidentified_drones:
        st.info("🔍 No unidentified drones stored yet. Run pattern detection with 'Detect Colored Areas as Unidentified Signals' enabled to find and store unidentified drones.")
        
        # Show current detection settings
        st.markdown("### ⚙️ Current Detection Settings")
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown(f"""
            <div class="info-box">
                <h4>🎯 Detection Configuration</h4>
                <p><strong>Colored Area Detection:</strong> {'✅ Enabled' if detect_green_rectangles else '❌ Disabled'}</p>
                <p><strong>Min Confidence:</strong> {min_confidence:.1f}</p>
                <p><strong>Min Area:</strong> {green_min_area if detect_green_rectangles else 'N/A'} pixels</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            st.markdown(f"""
            <div class="info-box">
                <h4>🛡️ Validation Rules</h4>
                <p><strong>Boundary Check:</strong> ✅ Enabled</p>
                <p><strong>Overlap Filter:</strong> ✅ Enabled</p>
                <p><strong>Duplicate Check:</strong> ✅ Enabled</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Show statistics
        total_stored = len(st.session_state.unidentified_drones)
        avg_confidence = np.mean([drone['confidence'] for drone in st.session_state.unidentified_drones])
        source_files = set(drone['filename'] for drone in st.session_state.unidentified_drones)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Stored Drones", total_stored)
        with col_stat2:
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col_stat3:
            st.metric("Source Files", len(source_files))
        
        st.markdown("---")
        
        # Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
        
        with col_ctrl1:
            st.markdown("**🎛️ Display Options**")
            
        with col_ctrl2:
            sort_by = st.selectbox("Sort by", ["ID", "Confidence", "Size", "Filename"], index=0)
            
        with col_ctrl3:
            if st.button("🗑️ Clear All", help="Remove all stored unidentified drones"):
                st.session_state.unidentified_drones = []
                st.success("Cleared all unidentified drones!")
                st.rerun()
        
        # Sort drones based on selection
        sorted_drones = st.session_state.unidentified_drones.copy()
        if sort_by == "Confidence":
            sorted_drones.sort(key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "Size":
            sorted_drones.sort(key=lambda x: x['area'], reverse=True)
        elif sort_by == "Filename":
            sorted_drones.sort(key=lambda x: x['filename'])
        # ID is already in order
        
        st.markdown("### 🖼️ Unidentified Drone Gallery")
        
        # Display drones in a grid
        cols_per_row = 4
        for i in range(0, len(sorted_drones), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(sorted_drones):
                    drone = sorted_drones[idx]
                    
                    with cols[j]:
                        # Display drone image with forced 100px sizing
                        drone_caption = drone.get('custom_name', f"US-{drone['id']}")
                        st.image(drone['image'], 
                                caption=drone_caption, 
                                width=100)
                        
                        # Rename functionality
                        if st.button(f"✏️", key=f"rename_{drone['id']}", help="Rename this drone"):
                            new_name = st.text_input(f"New name", 
                                                     value=drone.get('custom_name', f"Drone_{drone['id']}"),
                                                     key=f"new_name_{drone['id']}")
                            if st.button(f"✅", key=f"confirm_rename_{drone['id']}"):
                                # Update drone in session state
                                for i, stored_drone in enumerate(st.session_state.unidentified_drones):
                                    if stored_drone['id'] == drone['id']:
                                        st.session_state.unidentified_drones[i]['custom_name'] = new_name
                                        break
                                st.success(f"Renamed!")
                                st.rerun()
                        
                        # Save button and name input for each drone
                        st.markdown("**🏷️ Save:**")
                        display_name = drone.get('custom_name', f"Drone_Type_{drone['id']}")
                        store_name = st.text_input(f"Template Name", 
                                                  value=display_name,
                                                  key=f"name_{drone['id']}",
                                                  help="Name for the new template",
                                                  label_visibility="collapsed")
                        
                        col_store_btn, col_remove_btn = st.columns([1, 1])
                        with col_store_btn:
                            if st.button(f"💾 Save", key=f"store_{drone['id']}", 
                                       help="Save as template to computer and load into system",
                                       width='stretch'):
                                success, message, filename = save_drone_as_template(drone, store_name)
                                if success:
                                    st.success(f"✅ {message}")
                                    # Remove from unidentified drones after successful storage
                                    st.session_state.unidentified_drones = [
                                        d for d in st.session_state.unidentified_drones 
                                        if d['id'] != drone['id']
                                    ]
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        
                        with col_remove_btn:
                            if st.button(f"🗑️", key=f"remove_gallery_{drone['id']}", 
                                       help="Remove this unidentified drone",
                                       width='stretch'):
                                st.session_state.unidentified_drones = [
                                    d for d in st.session_state.unidentified_drones 
                                    if d['id'] != drone['id']
                                ]
                                st.success(f"Removed US-{drone['id']}")
                                st.rerun()
                        
                        # Drone info
                        with st.expander(f"ℹ️ US-{drone['id']} Details"):
                            st.markdown(f"""
                            **Source:** {drone['filename']}  
                            **Confidence:** {drone['confidence']:.3f}  
                            **Position:** ({drone['x']}, {drone['y']})  
                            **Size:** {drone['width']} × {drone['height']}  
                            **Area:** {drone['area']} pixels  
                            **Type:** {drone['detection_type']}  
                            **Found:** {drone['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                            """)
                            
                            # Show context (full image with highlight)
                            if st.checkbox(f"Show Context", key=f"context_{drone['id']}"):
                                st.markdown("**Context in Original Image:**")
                                
                                # Create a highlighted version of the full image
                                full_img_array = np.array(drone['full_image'])
                                full_img_cv = cv.cvtColor(full_img_array, cv.COLOR_RGB2BGR)
                                
                                # Draw red dashed rectangle around this drone
                                draw_dashed_rectangle(
                                    full_img_cv,
                                    (drone['x'], drone['y']),
                                    (drone['x'] + drone['width'], drone['y'] + drone['height']),
                                    (0, 0, 255),  # Red in BGR
                                    2, 10
                                )
                                
                                highlighted_img = cv_to_pil(full_img_cv)
                                st.image(highlighted_img, 
                                        caption=f"US-{drone['id']} in {drone['filename']}",
                                        width='stretch')
                            
                            # Additional actions in details
                            col_detail1, col_detail2 = st.columns(2)
                            with col_detail1:
                                if st.button(f"💾 Save from Details", key=f"store_detail_{drone['id']}", 
                                           help="Alternative save button from details"):
                                    success, message, filename = save_drone_as_template(drone, f"Detail_Save_{drone['id']}")
                                    if success:
                                        st.success(f"✅ {message}")
                                        st.session_state.unidentified_drones = [
                                            d for d in st.session_state.unidentified_drones 
                                            if d['id'] != drone['id']
                                        ]
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {message}")
                            
                            with col_detail2:
                                if st.button(f"🗑️ Remove from Details", key=f"remove_detail_{drone['id']}", 
                                           help="Remove this unidentified drone"):
                                    st.session_state.unidentified_drones = [
                                        d for d in st.session_state.unidentified_drones 
                                        if d['id'] != drone['id']
                                    ]
                                    st.success(f"Removed US-{drone['id']}")
                                    st.rerun()
        
        # Export functionality
        if st.session_state.unidentified_drones:
            st.markdown("---")
            st.markdown("### 📤 Export Options")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("📊 Export as CSV"):
                    # Create CSV data
                    csv_data = []
                    for drone in st.session_state.unidentified_drones:
                        csv_data.append({
                            'ID': f"US-{drone['id']}",
                            'Filename': drone['filename'],
                            'Confidence': drone['confidence'],
                            'X': drone['x'],
                            'Y': drone['y'],
                            'Width': drone['width'],
                            'Height': drone['height'],
                            'Area': drone['area'],
                            'Detection_Type': drone['detection_type'],
                            'Timestamp': drone['timestamp'].isoformat()
                        })
                    
                    df = pd.DataFrame(csv_data)
                    csv_string = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_string,
                        file_name="unidentified_drones.csv",
                        mime="text/csv"
                    )
            
            with col_export2:
                if st.button("🖼️ Export Images as ZIP"):
                    # Create ZIP file in memory
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for drone in st.session_state.unidentified_drones:
                            # Save drone image
                            img_buffer = io.BytesIO()
                            drone['image'].save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            filename = f"US-{drone['id']}_{drone['filename'].replace('.', '_')}.png"
                            zip_file.writestr(filename, img_buffer.getvalue())
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Download Images ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="unidentified_drones_images.zip",
                        mime="application/zip"
                    )
