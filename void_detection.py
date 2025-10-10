#!/usr/bin/env python3
"""
Simplified Void and Signal Detection System

Fixed color scheme based on typical spectrogram colormaps:
- VOID REGIONS: Cyan/Turquoise background (low activity, both B and G high, R low)
- SIGNAL REGIONS: Lime-green/Yellow-green (high activity, G high, R moderate, B low)

Optimized for consistent spectrogram colormap without manual color selection.
"""

import cv2 as cv
import numpy as np
import os
import streamlit as st


def analyze_image_colors(image):
    """
    Analyze the color distribution in an image to help debug detection issues.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        dict: Color statistics
    """
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Split channels
    b, g, r = cv.split(image)
    h, s, v = cv.split(hsv)
    
    # Calculate statistics
    stats = {
        'bgr': {
            'b_mean': float(np.mean(b)),
            'g_mean': float(np.mean(g)),
            'r_mean': float(np.mean(r)),
            'b_max': int(np.max(b)),
            'g_max': int(np.max(g)),
            'r_max': int(np.max(r))
        },
        'hsv': {
            'h_mean': float(np.mean(h)),
            's_mean': float(np.mean(s)),
            'v_mean': float(np.mean(v)),
            'h_max': int(np.max(h)),
            's_max': int(np.max(s)),
            'v_max': int(np.max(v))
        },
        'pixels_with_high_green': int(np.sum(g > 100)),
        'pixels_with_g_gt_b': int(np.sum(g > b)),
        'total_pixels': image.shape[0] * image.shape[1]
    }
    
    return stats


class VoidDetector:
    """
    Simplified detector for void and signal regions with fixed color scheme.
    
    - VOID: Cyan/Turquoise background (B≈G, both high, R low)
    - SIGNAL: Lime-green/Yellow-green (G high, R moderate, B low)
    """
    
    def __init__(self, debug=False):
        """Initialize detector with optimized parameters.
        
        Args:
            debug: If True, saves intermediate masks for debugging
        """
        # Void detection parameters (blue/cyan areas)
        self.void_min_area = 100
        self.void_morphology_kernel = 5
        
        # Signal detection parameters (green/yellow areas) 
        self.signal_min_area = 50
        self.signal_morphology_kernel = 3
        
        # Debug mode
        self.debug = debug
    
    def detect_void_regions(self, image):
        """
        Detect blue/cyan VOID regions (background/inactive areas).
        
        Fixed color detection for blue/cyan background areas in spectrograms.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            tuple: (void_mask, void_contours, void_stats)
        """
        if len(image.shape) == 2:
            bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        else:
            bgr_image = image.copy()
        
        # Convert to HSV for robust detection
        hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
        
        # VIRIDIS COLORMAP VOID DETECTION
        # Dark blue-green background (void): Hue 85-100, Saturation 130-255, Value 80-160
        # Based on actual newo.png analysis: void pixels have H~86-94, S~187-205, V~142-153
        lower_void = np.array([80, 130, 80])   # Relaxed lower bounds
        upper_void = np.array([105, 255, 165]) # Upper bounds for dark VIRIDIS void
        
        void_mask = cv.inRange(hsv, lower_void, upper_void)
        
        # Additional refinement: VIRIDIS void in BGR
        # Void criteria: Green ~128-160, Blue ~130-145, Red ~30-40 (dark teal/cyan)
        b, g, r = cv.split(bgr_image)
        
        # VIRIDIS void: Green and Blue similar and high, Red very low
        void_green_range = (g >= 120) & (g <= 165)
        void_blue_range = (b >= 120) & (b <= 150)
        void_red_low = (r >= 25) & (r <= 45)
        
        bgr_void_mask = (void_green_range & void_blue_range & void_red_low).astype(np.uint8) * 255
        
        # Combine both methods for robust detection
        void_mask = cv.bitwise_or(void_mask, bgr_void_mask)
        
        # Clean up void mask with morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
                                        (self.void_morphology_kernel, self.void_morphology_kernel))
        void_mask = cv.morphologyEx(void_mask, cv.MORPH_OPEN, kernel)
        void_mask = cv.morphologyEx(void_mask, cv.MORPH_CLOSE, kernel)
        
        # Find void contours
        contours, _ = cv.findContours(void_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        void_contours = []
        void_stats = []
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area >= self.void_min_area:
                x, y, w, h = cv.boundingRect(contour)
                
                # Calculate void characteristics
                if len(image.shape) == 3:
                    region_gray = cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
                else:
                    region_gray = image[y:y+h, x:x+w]
                
                if region_gray.size > 0:
                    avg_intensity = np.mean(region_gray)
                    
                    # Void confidence based on blue/cyan characteristics
                    region_bgr = bgr_image[y:y+h, x:x+w]
                    b_mean = np.mean(region_bgr[:,:,0])
                    g_mean = np.mean(region_bgr[:,:,1])
                    r_mean = np.mean(region_bgr[:,:,2])
                    
                    # Blue/cyan dominance (blue or green high, red low)
                    blue_cyan_score = (max(b_mean, g_mean) - r_mean) / 255.0
                    blue_cyan_score = max(0.0, min(1.0, blue_cyan_score))
                    
                    # Color saturation (how "colored" vs gray)
                    saturation = (max(b_mean, g_mean) - min(b_mean, g_mean, r_mean)) / max(1, max(b_mean, g_mean))
                    saturation = max(0.0, min(1.0, saturation))
                    
                    confidence = blue_cyan_score * 0.6 + saturation * 0.4
                    
                    void_stats.append({
                        'type': 'void_region',
                        'area': area,
                        'bbox': (x, y, w, h),
                        'avg_intensity': avg_intensity,
                        'confidence': confidence,
                        'avg_color': (int(b_mean), int(g_mean), int(r_mean))
                    })
                    
                    void_contours.append(contour)
        
        return void_mask, void_contours, void_stats
    
    def detect_signal_regions(self, image):
        """
        Detect green/yellow SIGNAL regions (active noise/signal areas).
        
        Fixed color detection for green/yellowish signal areas in spectrograms.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            tuple: (signal_mask, signal_contours, signal_stats)
        """
        if len(image.shape) == 2:
            bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        else:
            bgr_image = image.copy()
        
        # Convert to HSV for robust detection
        hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
        
        # VIRIDIS SIGNAL DETECTION - Must be BRIGHTER and MORE YELLOW than void
        # Void: H~86-94, S~187-205, V~142-153 (dark cyan-green)
        # Signal: H~43, S~182, V~216 (bright yellow-green)
        # Key difference: Value (brightness) must be HIGH (180+)
        lower_signal = np.array([28, 100, 180])  # Must be bright! V >= 180
        upper_signal = np.array([80, 255, 255])  # Yellow-green hues only
        
        hsv_signal_mask = cv.inRange(hsv, lower_signal, upper_signal)
        
        # Additional BGR detection: BRIGHT signals only (G > 180, not dark cyan)
        b, g, r = cv.split(bgr_image)
        
        # VIRIDIS signals are BRIGHT with high green
        # Void: G~128-160, B~130-145, R~30-40
        # Signal: G~200+, B~60, R~150
        
        # Method 1: Very bright green (distinguishes from void)
        very_bright_green = (g >= 180) & (b < 100)
        
        # Method 2: Yellow-green (high G and R, low B)
        bright_yellow_green = (g >= 180) & (r >= 100) & (b < 120)
        
        bgr_signal_mask = (very_bright_green | bright_yellow_green).astype(np.uint8) * 255
        
        # Combine both methods for robust detection
        signal_mask = cv.bitwise_or(hsv_signal_mask, bgr_signal_mask)
        
        # Clean up signal mask with morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, 
                                        (self.signal_morphology_kernel, self.signal_morphology_kernel))
        signal_mask = cv.morphologyEx(signal_mask, cv.MORPH_OPEN, 
                                    cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
        signal_mask = cv.morphologyEx(signal_mask, cv.MORPH_CLOSE, kernel)
        
        # Find signal contours
        contours, _ = cv.findContours(signal_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        signal_contours = []
        signal_stats = []
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area >= self.signal_min_area:
                x, y, w, h = cv.boundingRect(contour)
                
                # Calculate signal characteristics
                if len(image.shape) == 3:
                    region_gray = cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
                else:
                    region_gray = image[y:y+h, x:x+w]
                
                if region_gray.size > 0:
                    avg_intensity = np.mean(region_gray)
                    
                    # Signal confidence based on green/yellow characteristics
                    region_bgr = bgr_image[y:y+h, x:x+w]
                    b_mean = np.mean(region_bgr[:,:,0])
                    g_mean = np.mean(region_bgr[:,:,1])
                    r_mean = np.mean(region_bgr[:,:,2])
                    
                    # Green/yellow dominance
                    green_score = (g_mean - min(b_mean, r_mean)) / 255.0
                    green_score = max(0.0, min(1.0, green_score))
                    
                    # Brightness (signals should be bright)
                    brightness = g_mean / 255.0
                    
                    # Yellow bonus (green + red, low blue)
                    yellow_bonus = 0.0
                    if r_mean > 80 and g_mean > 100 and b_mean < 100 and r_mean < g_mean:
                        yellow_bonus = 0.15
                    
                    confidence = min(1.0, green_score * 0.5 + brightness * 0.35 + yellow_bonus + 0.15)
                    
                    signal_stats.append({
                        'type': 'signal_region',
                        'area': area,
                        'bbox': (x, y, w, h),
                        'avg_intensity': avg_intensity,
                        'confidence': confidence,
                        'avg_color': (int(b_mean), int(g_mean), int(r_mean))
                    })
                    
                    signal_contours.append(contour)
        
        return signal_mask, signal_contours, signal_stats
    
    def detect_all(self, image):
        """
        Detect both void and signal regions.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            dict: Complete detection results with voids, signals, and summary stats
        """
        # Detect voids (blue/cyan background)
        void_mask, void_contours, void_stats = self.detect_void_regions(image)
        
        # Detect signals (green/yellow noise)
        signal_mask, signal_contours, signal_stats = self.detect_signal_regions(image)
        
        return {
            'voids': {
                'mask': void_mask,
                'contours': void_contours,
                'stats': void_stats
            },
            'signals': {
                'mask': signal_mask,
                'contours': signal_contours,
                'stats': signal_stats
            },
            'summary': {
                'total_voids': len(void_stats),
                'total_signals': len(signal_stats),
                'void_area': sum(s['area'] for s in void_stats),
                'signal_area': sum(s['area'] for s in signal_stats)
            }
        }
    
    def visualize_results(self, image, results):
        """
        Create visualization of detection results.
        
        Args:
            image: Original input image
            results: Detection results from detect_all()
            
        Returns:
            np.ndarray: Annotated visualization image
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv.cvtColor(vis_image, cv.COLOR_GRAY2BGR)
        
        # Draw voids in red contours
        for contour in results['voids']['contours']:
            cv.drawContours(vis_image, [contour], -1, (0, 0, 255), 2)
        
        # Draw signals in bright magenta contours (to stand out against green)
        for contour in results['signals']['contours']:
            cv.drawContours(vis_image, [contour], -1, (255, 0, 255), 2)
        
        # Add void labels
        for i, stat in enumerate(results['voids']['stats']):
            x, y, w, h = stat['bbox']
            label = f"V{i+1}"
            cv.putText(vis_image, label, (x, y-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add signal labels
        for i, stat in enumerate(results['signals']['stats']):
            x, y, w, h = stat['bbox']
            label = f"S{i+1}"
            cv.putText(vis_image, label, (x, y-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return vis_image


# ====================================
# STREAMLIT WEB INTERFACE
# ====================================

def run_streamlit_interface():
    """
    Streamlit web interface for interactive void and signal detection.
    
    Fixed color scheme:
    - Blue/Cyan = Void (background)
    - Green/Yellow = Signal (noise)
    """
    st.title("🔍 Void & Signal Detection System")
    st.markdown("""
    **Fixed Color Scheme Detection:**
    - 🔵 **Void Regions**: Blue/Cyan background (inactive areas)
    - 🟢 **Signal Regions**: Green/Yellow noise (active signals)
    """)
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, 
                                     help="Save intermediate masks and show color analysis")
    
    detector = VoidDetector(debug=debug_mode)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Detection Parameters")
    
    # Void detection parameters
    st.sidebar.subheader("🔵 Void Detection (Blue/Cyan)")
    detector.void_min_area = st.sidebar.slider("Min Void Area", 50, 500, 100)
    detector.void_morphology_kernel = st.sidebar.slider("Void Kernel Size", 3, 9, 5)
    
    # Signal detection parameters  
    st.sidebar.subheader("🟢 Signal Detection (Green/Yellow)")
    detector.signal_min_area = st.sidebar.slider("Min Signal Area", 25, 200, 50)
    detector.signal_morphology_kernel = st.sidebar.slider("Signal Kernel Size", 2, 6, 3)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image for detection analysis",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    # Process images from workspace
    current_dir = os.getcwd()
    workspace_images = [f for f in os.listdir(current_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                       and not f.startswith('detection_result_')]
    
    if workspace_images:
        st.subheader("📁 Images in Workspace")
        selected_image = st.selectbox("Select workspace image:", 
                                    ["None"] + workspace_images)
        
        if selected_image != "None":
            image_path = os.path.join(current_dir, selected_image)
            image = cv.imread(image_path)
            if image is not None:
                st.image(image, caption=f"Selected: {selected_image}", 
                        use_column_width=True, channels="BGR")
                
                if st.button("🔍 Analyze Selected Image"):
                    with st.spinner("Analyzing image..."):
                        # Color analysis first (if debug mode)
                        if debug_mode:
                            st.subheader("🎨 Color Analysis")
                            color_stats = analyze_image_colors(image)
                            
                            col_debug1, col_debug2 = st.columns(2)
                            with col_debug1:
                                st.write("**BGR Channels:**")
                                st.write(f"- Blue mean: {color_stats['bgr']['b_mean']:.1f} (max: {color_stats['bgr']['b_max']})")
                                st.write(f"- Green mean: {color_stats['bgr']['g_mean']:.1f} (max: {color_stats['bgr']['g_max']})")
                                st.write(f"- Red mean: {color_stats['bgr']['r_mean']:.1f} (max: {color_stats['bgr']['r_max']})")
                            
                            with col_debug2:
                                st.write("**HSV Channels:**")
                                st.write(f"- Hue mean: {color_stats['hsv']['h_mean']:.1f} (max: {color_stats['hsv']['h_max']})")
                                st.write(f"- Saturation mean: {color_stats['hsv']['s_mean']:.1f} (max: {color_stats['hsv']['s_max']})")
                                st.write(f"- Value mean: {color_stats['hsv']['v_mean']:.1f} (max: {color_stats['hsv']['v_max']})")
                            
                            st.write(f"**Signal Analysis:** {color_stats['pixels_with_high_green']} pixels with G>100 ({100*color_stats['pixels_with_high_green']/color_stats['total_pixels']:.1f}%)")
                            st.write(f"**Green>Blue:** {color_stats['pixels_with_g_gt_b']} pixels ({100*color_stats['pixels_with_g_gt_b']/color_stats['total_pixels']:.1f}%)")
                        
                        results = detector.detect_all(image)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🔵 Void Regions", results['summary']['total_voids'])
                        with col2:
                            st.metric("🟢 Signal Regions", results['summary']['total_signals'])
                        with col3:
                            total_pixels = image.shape[0] * image.shape[1]
                            coverage = (results['summary']['void_area'] + results['summary']['signal_area']) / total_pixels * 100
                            st.metric("Coverage %", f"{coverage:.1f}")
                        
                        # Visualization
                        vis_image = detector.visualize_results(image, results)
                        st.image(vis_image, caption="Detection Results (Red=Voids, Magenta=Signals)", 
                                use_column_width=True, channels="BGR")
                        
                        # Detailed results in columns
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if results['voids']['stats']:
                                st.subheader("🔵 Void Regions (Blue/Cyan)")
                                void_df = []
                                for i, stat in enumerate(results['voids']['stats']):
                                    void_df.append({
                                        'ID': f"V{i+1}",
                                        'Area': stat['area'],
                                        'Confidence': f"{stat['confidence']:.3f}",
                                        'Color (BGR)': f"{stat['avg_color']}"
                                    })
                                st.dataframe(void_df, use_container_width=True)
                        
                        with col_b:
                            if results['signals']['stats']:
                                st.subheader("🟢 Signal Regions (Green/Yellow)")
                                signal_df = []
                                for i, stat in enumerate(results['signals']['stats']):
                                    signal_df.append({
                                        'ID': f"S{i+1}",
                                        'Area': stat['area'],
                                        'Confidence': f"{stat['confidence']:.3f}",
                                        'Color (BGR)': f"{stat['avg_color']}"
                                    })
                                st.dataframe(signal_df, use_container_width=True)
    
    # Handle uploaded file
    if uploaded_file is not None:
        # Load uploaded image
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", 
                use_column_width=True, channels="BGR")
        
        if st.button("🔍 Analyze Uploaded Image"):
            with st.spinner("Analyzing uploaded image..."):
                # Color analysis first (if debug mode)
                if debug_mode:
                    st.subheader("🎨 Color Analysis")
                    color_stats = analyze_image_colors(image)
                    
                    col_debug1, col_debug2 = st.columns(2)
                    with col_debug1:
                        st.write("**BGR Channels:**")
                        st.write(f"- Blue mean: {color_stats['bgr']['b_mean']:.1f} (max: {color_stats['bgr']['b_max']})")
                        st.write(f"- Green mean: {color_stats['bgr']['g_mean']:.1f} (max: {color_stats['bgr']['g_max']})")
                        st.write(f"- Red mean: {color_stats['bgr']['r_mean']:.1f} (max: {color_stats['bgr']['r_max']})")
                    
                    with col_debug2:
                        st.write("**HSV Channels:**")
                        st.write(f"- Hue mean: {color_stats['hsv']['h_mean']:.1f} (max: {color_stats['hsv']['h_max']})")
                        st.write(f"- Saturation mean: {color_stats['hsv']['s_mean']:.1f} (max: {color_stats['hsv']['s_max']})")
                        st.write(f"- Value mean: {color_stats['hsv']['v_mean']:.1f} (max: {color_stats['hsv']['v_max']})")
                    
                    st.write(f"**Signal Analysis:** {color_stats['pixels_with_high_green']} pixels with G>100 ({100*color_stats['pixels_with_high_green']/color_stats['total_pixels']:.1f}%)")
                    st.write(f"**Green>Blue:** {color_stats['pixels_with_g_gt_b']} pixels ({100*color_stats['pixels_with_g_gt_b']/color_stats['total_pixels']:.1f}%)")
                
                results = detector.detect_all(image)
                
                # Display results (same as above)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔵 Void Regions", results['summary']['total_voids'])
                with col2:
                    st.metric("🟢 Signal Regions", results['summary']['total_signals'])
                with col3:
                    total_pixels = image.shape[0] * image.shape[1]
                    coverage = (results['summary']['void_area'] + results['summary']['signal_area']) / total_pixels * 100
                    st.metric("Coverage %", f"{coverage:.1f}")
                
                # Visualization
                vis_image = detector.visualize_results(image, results)
                st.image(vis_image, caption="Detection Results (Red=Voids, Magenta=Signals)", 
                        use_column_width=True, channels="BGR")
                
                # Detailed results in columns
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if results['voids']['stats']:
                        st.subheader("🔵 Void Regions (Blue/Cyan)")
                        void_df = []
                        for i, stat in enumerate(results['voids']['stats']):
                            void_df.append({
                                'ID': f"V{i+1}",
                                'Area': stat['area'],
                                'Confidence': f"{stat['confidence']:.3f}",
                                'Color (BGR)': f"{stat['avg_color']}"
                            })
                        st.dataframe(void_df, use_container_width=True)
                
                with col_b:
                    if results['signals']['stats']:
                        st.subheader("🟢 Signal Regions (Green/Yellow)")
                        signal_df = []
                        for i, stat in enumerate(results['signals']['stats']):
                            signal_df.append({
                                'ID': f"S{i+1}",
                                'Area': stat['area'],
                                'Confidence': f"{stat['confidence']:.3f}",
                                'Color (BGR)': f"{stat['avg_color']}"
                            })
                        st.dataframe(signal_df, use_container_width=True)


if __name__ == "__main__":
    run_streamlit_interface()
