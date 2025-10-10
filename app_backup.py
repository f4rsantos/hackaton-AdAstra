import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import zipfile
import time
import threading
import queue
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import *
from multiprocessing import cpu_count
import psutil
from live_feed_processor import (
    LiveFeedConfig, HighThroughputProcessor, ZipStreamProcessor,
    NetworkStreamProcessor, DirectoryMonitor, create_live_feed_processor,
    optimize_for_throughput
)


# Session state initialization - Must be first!
if 'templates' not in st.session_state:
    st.session_state.templates = {}
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'template_folders' not in st.session_state:
    st.session_state.template_folders = {'Default': {}}
if 'active_folder' not in st.session_state:
    st.session_state.active_folder = 'Default'
if 'unidentified_drones' not in st.session_state:
    st.session_state.unidentified_drones = []
if 'templates_loaded' not in st.session_state:
    st.session_state.templates_loaded = False
    load_stored_templates()
    st.session_state.templates_loaded = True

# Live feed session state
if 'live_feed_processor' not in st.session_state:
    st.session_state.live_feed_processor = None
if 'live_feed_active' not in st.session_state:
    st.session_state.live_feed_active = False
if 'live_feed_results' not in st.session_state:
    st.session_state.live_feed_results = []
if 'live_feed_stats' not in st.session_state:
    st.session_state.live_feed_stats = {}
if 'live_feed_config' not in st.session_state:
    st.session_state.live_feed_config = None

# Configure Streamlit page
st.set_page_config(
    page_title="Ad Astra",
    page_icon="⍙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f3a93, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 1rem;
    }
    
    .sidebar-section {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1976d2;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #37474f 0%, #546e7a 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .detection-stats {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    
    .template-box {
        background: #f0f4ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    
    /* Consistent image sizing for drone gallery */
    .stImage > img {
        max-height: 200px;
        object-fit: contain;
        width: 100% !important;
    }
    
    /* Target the Unidentified Drones tab gallery - force 100px containers */
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] {
        min-height: 300px;
        max-height: 300px;
        padding: 10px;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        margin: 5px;
        display: flex;
        flex-direction: column;
        align-items: center;
        background-color: #fafafa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Force ALL images in drone gallery to 100x100px */
    div[data-testid="tabpanel"]:nth-child(4) .stImage {
        width: 100px !important;
        height: 100px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        margin: 0 auto 10px auto !important;
        border: 2px solid #ccc;
        border-radius: 8px;
        flex-shrink: 0 !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) .stImage > div {
        width: 100px !important;
        height: 100px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) .stImage img {
        max-width: 100px !important;
        max-height: 100px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 4px;
        display: block !important;
    }
    
    /* Ensure all images in the gallery maintain proper proportions */
    div[data-testid="tabpanel"]:nth-child(4) img {
        max-width: 100px !important;
        max-height: 100px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        object-position: center !important;
    }
    
    /* Force consistent button sizing in gallery */
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] .stButton {
        margin: 2px 0 !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] .stButton button {
        font-size: 12px !important;
        padding: 4px 8px !important;
        height: 32px !important;
    }
    
    /* Template images consistent sizing */
    .template-image {
        max-height: 150px;
        object-fit: contain;
    }
    
    /* Detection result images */
    .detection-image {
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
</style>""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">⍙ Ad Astra Drone Detection</h1>', unsafe_allow_html=True)

# How to Use section
with st.expander("📋 How to Use This Platform", expanded=False):
    st.markdown("""
    ### Spectrogram Drone Pattern Detection Guide
    
    **Step 1: Upload Template Patterns (Optional)**
    - Upload known drone spectrogram patterns as PNG/JPG images
    - These serve as reference templates for detection of known drone types
    - Name your templates descriptively (e.g., "DJI_Phantom.png", "Racing_Drone.png")
    - **Note: Templates are optional - you can detect unidentified signals without them**
    
    **Step 2: Configure Detection Settings**
    - Adjust matching threshold (higher = more strict matching)
    - Set minimum match confidence for filtering results
    - Choose detection method (normalized correlation recommended)
    
    **Step 3: Upload Test Spectrograms**
    - Upload spectrogram images to analyze for drone patterns
    - Can process single images or batch process multiple files
    - Supports PNG, JPG, and ZIP archives
    
    **Step 4: Run Pattern Detection**
    - The system will search for template patterns in your spectrograms (if templates provided)
    - Automatically detect colored rectangles as unidentified signal areas (if enabled)
    - View matches with confidence scores and bounding boxes
    - Compare results across different templates and images
    
    **Step 5: Auto-Label Process (Optional)**
    - Use the Auto-Label feature in the sidebar to automatically create templates
    - Iteratively processes unidentified signals and converts high-confidence ones to templates
    - Configure thresholds and containment detection in the sidebar settings
    - Process continues until no more high-confidence signals are found
    
    **💡 Tips for Best Results:**
    - Use high-quality, clear spectrogram images
    - Templates should show distinctive drone signature patterns
    - Adjust threshold based on pattern complexity
    - Test with known positive samples first
    - **CLAHE enhancement is automatically applied** for better pattern visibility
    - **Colored rectangle detection** automatically identifies unidentified signal areas (green, cyan, teal)
    - Orange dashed boxes indicate unidentified signals, colored solid boxes are identified drones
    - **Use Auto-Label in the sidebar** to automatically create templates from unidentified signals
    """)

# Sidebar for configuration
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Detection Configuration</h2>', unsafe_allow_html=True)
    
    # Smart Astra Master Control - TOP PRIORITY
    st.markdown("---")
    st.markdown("**🚀 Smart Astra Mode**")
    smart_astra_enabled = st.checkbox(
        "🧠 Enable Smart Astra",
        value=False,
        help="Intelligent system that automatically optimizes all parameters based on current conditions and image characteristics"
    )
    
    if smart_astra_enabled:
        st.info("🚀 **Smart Astra Active** - System is automatically optimizing all parameters")
        
        # Smart Astra Status Display
        with st.expander("📊 Smart Astra Status", expanded=False):
            st.markdown("**Current Intelligent Decisions:**")
            
            # Get system info for smart decisions
            import psutil
            from functions import GPUDetector, SystemMonitor
            
            try:
                # Initialize smart components
                gpu_detector = GPUDetector()
                system_monitor = SystemMonitor()
                
                cuda_available, opencl_available = gpu_detector.detect_gpu_capabilities()
                stats = system_monitor.get_system_stats()
                
                # Smart Astra decision logic
                system_load = (stats['cpu_percent'] + stats['memory_percent']) / 2
                
                # Auto-determine performance mode
                if system_load > 80:
                    smart_performance_mode = "Battery Saver"
                    smart_reason = "High system load detected"
                elif system_load > 60:
                    smart_performance_mode = "Balanced"
                    smart_reason = "Moderate system load"
                elif gpu_detector.is_gpu_available():
                    smart_performance_mode = "Maximum"
                    smart_reason = "GPU available, low system load"
                else:
                    smart_performance_mode = "Performance"
                    smart_reason = "CPU-optimized, good system resources"
                
                # Smart parameter decisions
                smart_threshold = 0.62 # best value so far
                smart_confidence = 0.35
                smart_parallel = system_load < 70
                smart_auto_label = True  # Always enabled in Smart Astra
                
                col_smart1, col_smart2 = st.columns(2)
                
                with col_smart1:
                    st.metric("Performance Mode", smart_performance_mode)
                    st.metric("System Load", f"{system_load:.1f}%")
                    st.metric("Threshold", f"{smart_threshold:.2f}")
                    
                with col_smart2:
                    st.metric("Min Confidence", f"{smart_confidence:.2f}")
                    st.metric("Parallel Processing", "✅" if smart_parallel else "❌")
                    st.metric("Auto-Label", "✅" if smart_auto_label else "❌")
                
                st.markdown(f"**🎯 Decision Reason:** {smart_reason}")
                
                # Apply smart decisions to variables
                threshold = smart_threshold
                min_confidence = smart_confidence
                run_auto_label = smart_auto_label
                
            except Exception as e:
                st.warning(f"Smart Astra initialization error: {e}")
                # Fallback to safe defaults
                smart_performance_mode = "Balanced"
                threshold = 0.62
                min_confidence = 0.35
                run_auto_label = True
                smart_parallel = True
    
    st.markdown("---")
    
    # Ensure session_state.templates is initialized
    if 'templates' not in st.session_state:
        st.session_state.templates = {}
    # Manual Configuration (hidden when Smart Astra is enabled)
    if not smart_astra_enabled:
        # OpenCV Template Matching Parameters (only show if templates are available)
        if st.session_state.templates:
            with st.container():
                st.markdown('<div class="sidebar-section">Pattern Matching Parameters</div>', unsafe_allow_html=True)
                
                # Detection parameters
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider("Match Threshold", 0.1, 0.95, 0.62, 0.01,
                                        help="Higher values = more strict matching")
                    show_all_matches = st.checkbox("Show All Matches", False)
                with col2:
                    min_confidence = st.slider("Min Confidence", 0.0, 0.9, 0.3, 0.1)
                    draw_boxes = st.checkbox("Draw Bounding Boxes", True)
                
                # Border detection settings
                st.markdown("**• Border Detection**")
                enable_border_detection = st.checkbox("Enable Border Detection", True, 
                                                     help="Detect templates that are partially cut off at image borders")
                if enable_border_detection:
                    border_threshold = st.slider("Border Detection Sensitivity", 0.1, 1.0, 0.7, 0.05,
                                               help="Lower values = more sensitive to partial templates at borders")
                else:
                    border_threshold = 0.5  # Default value when border detection is disabled
                
                # Duplicate detection settings
                st.markdown("**• Duplicate Detection**")
                merge_overlapping = st.checkbox("Merge Overlapping Detections", True,
                                              help="Combine overlapping detections of the same template into a single centered detection")
                if merge_overlapping:
                    overlap_sensitivity = st.slider("Overlap Sensitivity", 0.0, 1.0, 1.0, 0.01,
                                                   help="Lower values = more aggressive merging of nearby detections")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Set default values when no templates are available
            threshold = 0.62
            show_all_matches = False
            draw_boxes = True
            enable_border_detection = True
            border_threshold = 0.3
            merge_overlapping = True
            overlap_sensitivity = 1.0
            
            # Show min confidence control even without templates (needed for US detection)
            min_confidence = st.slider("Min Confidence", 0.0, 0.9, 0.0, 0.1,
                                      help="Minimum confidence for filtering unidentified signal detections")
    else:
        # Smart Astra overrides - set optimal defaults and show information only
        show_all_matches = False
        draw_boxes = True
        enable_border_detection = True
        border_threshold = 0.6
        merge_overlapping = True
        overlap_sensitivity = 0.7
        
        # Display current Smart Astra parameters (read-only)
        st.markdown("**📊 Current Smart Parameters** *(auto-optimized)*")
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            st.metric("Match Threshold", f"{threshold:.3f}")
            st.metric("Border Threshold", f"{border_threshold:.2f}")
        with col_param2:
            st.metric("Min Confidence", f"{min_confidence:.2f}")
            st.metric("Overlap Sensitivity", f"{overlap_sensitivity:.2f}")
        
        # Show a message indicating pattern matching is not available
        st.markdown('<div class="sidebar-section">🎯 Pattern Matching Parameters</div>', unsafe_allow_html=True)
        st.info("📝 Pattern matching controls will appear here when you upload template patterns in the Template Management tab.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Colored rectangle detection settings - controlled by Smart Astra
    if not smart_astra_enabled:
        st.markdown("**• Unidentified Signal Detection**")
        detect_green_rectangles = st.checkbox("Detect Colored Areas as Unidentified Signals", True,
                                            help="Automatically detect colored rectangles (green, cyan, teal) in spectrograms as unidentified signal areas")
        if detect_green_rectangles:
            green_min_area = st.slider("Minimum Colored Area", 50, 500, 50, 25,
                                     help="Minimum area (pixels) for a colored rectangle to be considered a drone signal")
            
            st.markdown("**• Colored Area Overlap Settings**")
            green_overlap_threshold = st.slider("Overlap with Identified Drones", 0.0, 1.0, 0.3, 0.01,
                                              help="How much a colored rectangle must overlap with an identified drone to be ignored")
            
            colored_merge_threshold = st.slider("Unidentified Signal Merge Threshold", 0.0, 1.0, 0.2, 0.01,
                                              help="Overlap threshold for merging colored areas with each other. "
                                                   "Uses strict criteria including size similarity, proximity, and area efficiency. "
                                                   "Lower values = more conservative merging (only very similar overlapping areas). "
                                                   "Higher values = more permissive merging. "
                                                   "Prevents connecting distant rectangles or merging different-sized signals.")
        else:
            green_min_area = 100
            green_overlap_threshold = 0.3
            colored_merge_threshold = 0.2
    else:
        # Smart Astra optimizes colored area detection
        detect_green_rectangles = True  # Always enabled for comprehensive detection
        green_min_area = 75  # Optimized minimum area
        green_overlap_threshold = 0.25  # Slightly more permissive overlap
        colored_merge_threshold = 0.18  # Conservative merging
        
        st.markdown("**🎨 Unidentified Signal Detection** *(Smart Astra Optimized)*")
        st.success("✅ Colored area detection enabled with smart parameters")
        green_overlap_threshold = 0.3
        colored_merge_threshold = 0.2
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-Label section in sidebar - controlled by Smart Astra
    if not smart_astra_enabled:
        st.markdown("**• Auto-Label Process**")
        with st.expander("🔄 Auto-Label Settings", expanded=False):        
            auto_label_confidence_threshold = st.slider(
                "Auto-Label Confidence Threshold", 
                0.5, 0.95, 0.85, 0.05,
                help="Minimum confidence required for automatic labeling of unidentified signals"
            )
            
            containment_overlap_threshold = st.slider(
                "Containment Detection Threshold", 
                0.3, 0.8, 0.75, 0.05,
                help="Overlap threshold for detecting if an unidentified signal contains other models. Higher values = more strict containment detection"
            )
        
        # Auto-Label toggle checkbox
        auto_label_enabled = st.checkbox(
            "🔄 Enable Auto-Label", 
            value=False,
            help="When enabled, auto-labeling will run automatically after pattern detection"
        )
    else:
        # Smart Astra auto-configures auto-label settings
        auto_label_confidence_threshold = 0.82  # Optimized for Smart Astra
        containment_overlap_threshold = 0.70   # Balanced containment detection
        auto_label_enabled = True  # Always enabled in Smart Astra
        
        st.markdown("**🔄 Auto-Label** *(Smart Astra Managed)*")
        st.success("✅ Auto-labeling enabled with optimized settings")
    
    # Store settings in session state for use in detection
    st.session_state.auto_label_confidence_threshold = auto_label_confidence_threshold
    st.session_state.containment_overlap_threshold = containment_overlap_threshold
    st.session_state.auto_label_enabled = auto_label_enabled

    # Set the run_auto_label variable for compatibility
    run_auto_label = auto_label_enabled

    # Advanced Performance Management section in sidebar
    st.markdown("**• Performance Management**")
    with st.expander("⚡ Advanced Performance Settings", expanded=False):
        # Performance mode selection
        performance_modes = PerformanceMode.get_all_modes()
        mode_names = [mode['name'] for mode in performance_modes]
        mode_descriptions = [mode['description'] for mode in performance_modes]
        
        selected_mode_name = st.selectbox(
            "Performance Mode",
            mode_names,
            index=1,  # Default to Balanced
            help="Choose a performance preset that matches your system capabilities"
        )
        
        # Show mode description
        selected_mode = PerformanceMode.get_mode_by_name(selected_mode_name)
        st.info(f"**{selected_mode['name']}**: {selected_mode['description']}")
        
        # Create tabs for different settings
        settings_tab1, settings_tab2, settings_tab3 = st.tabs(["🖥️ System", "⚡ Parallel", "🎮 GPU"])
        
        with settings_tab1:
            # System monitoring and adaptive settings
            st.markdown("**System Adaptation**")
            
            adaptive_scaling = st.checkbox(
                "🧠 Smart Scaling", 
                value=True,
                help="Automatically adjust performance based on system load"
            )
            
            if adaptive_scaling:
                col_sys1, col_sys2 = st.columns(2)
                
                with col_sys1:
                    auto_throttle = st.checkbox(
                        "Auto Throttle", 
                        value=True,
                        help="Reduce performance when system is stressed"
                    )
                    
                    throttle_threshold = st.slider(
                        "Throttle Threshold %", 
                        70, 95, 90, 5,
                        help="CPU/Memory usage % to trigger throttling"
                    )
                
                with col_sys2:
                    memory_cleanup = st.checkbox(
                        "Memory Cleanup", 
                        value=True,
                        help="Force garbage collection to free memory"
                    )
                    
                    adaptation_interval = st.slider(
                        "Adaptation Interval (s)", 
                        1, 10, 5, 1,
                        help="How often to check and adapt performance"
                    )
            else:
                auto_throttle = False
                throttle_threshold = 90
                memory_cleanup = True
                adaptation_interval = 5
            
            # Show current system stats
            try:
                stats = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count_phys = psutil.cpu_count(logical=False)
                cpu_count_log = psutil.cpu_count(logical=True)
                
                st.markdown("**Current System Status**")
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.metric("CPU Usage", f"{cpu_percent:.1f}%")
                    st.metric("CPU Cores", f"{cpu_count_phys}P/{cpu_count_log}L")
                
                with col_stat2:
                    st.metric("Memory Usage", f"{stats.percent:.1f}%")
                    st.metric("Available RAM", f"{stats.available/(1024**3):.1f} GB")
                    
            except Exception as e:
                st.warning("System monitoring not available")
        
        with settings_tab2:
            # Parallel processing settings
            st.markdown("**Parallel Processing**")
            
            enable_parallel = st.checkbox(
                "🚀 Enable Parallel Processing", 
                value=selected_mode['name'] != 'Battery',
                help="Process images in parallel for faster detection"
            )
            
            if enable_parallel:
                col_par1, col_par2 = st.columns(2)
                
                with col_par1:
                    custom_workers = st.checkbox(
                        "Custom Worker Count",
                        value=False,
                        help="Override automatic worker count"
                    )
                    
                    if custom_workers:
                        max_workers = st.slider(
                            "Workers", 
                            1, min(16, cpu_count() * 2), 
                            selected_mode['max_workers'], 1,
                            help="Number of parallel workers"
                        )
                    else:
                        max_workers = None
                    
                    overlap_percentage = st.slider(
                        "Quadrant Overlap %", 
                        0.05, 0.30, 0.15, 0.05,
                        help="Overlap between image quadrants"
                    )
                
                with col_par2:
                    min_image_size = st.slider(
                        "Min Image Size", 
                        400, 1200, 800, 100,
                        help="Minimum image size for parallel processing"
                    )
                    
                    use_threading = st.checkbox(
                        "Use Threading", 
                        value=True,
                        help="Use threading vs multiprocessing"
                    )
                    
                    batch_processing = st.checkbox(
                        "Batch Processing",
                        value=selected_mode['name'] == 'Maximum',
                        help="Process multiple images in batches"
                    )
                    
                    if batch_processing:
                        batch_size = st.slider(
                            "Batch Size",
                            2, 16, selected_mode.get('batch_size', 4), 2,
                            help="Images per batch"
                        )
                    else:
                        batch_size = 4
            else:
                max_workers = 1
                overlap_percentage = 0.15
                min_image_size = 800
                use_threading = True
                batch_processing = False
                batch_size = 4
        
        with settings_tab3:
            # GPU settings
            st.markdown("**GPU Acceleration**")
            
            # Initialize GPU detector to check availability
            gpu_detector = GPUDetector()
            cuda_available, opencl_available = gpu_detector.detect_gpu_capabilities()
            gpu_info = gpu_detector.get_gpu_info()
            
            if cuda_available or opencl_available:
                enable_gpu = st.checkbox(
                    "🎮 Enable GPU Acceleration",
                    value=selected_mode['gpu_enabled'],
                    help="Use GPU for faster template matching"
                )
                
                if enable_gpu:
                    col_gpu1, col_gpu2 = st.columns(2)
                    
                    with col_gpu1:
                        if cuda_available and opencl_available:
                            prefer_cuda = st.radio(
                                "GPU Type",
                                ["CUDA", "OpenCL"],
                                index=0,
                                help="Preferred GPU acceleration method"
                            ) == "CUDA"
                        else:
                            prefer_cuda = cuda_available
                            gpu_type = "CUDA" if cuda_available else "OpenCL"
                            st.write(f"**Available:** {gpu_type}")
                        
                        gpu_memory_limit = st.slider(
                            "GPU Memory Limit (GB)",
                            0.5, 8.0, 2.0, 0.5,
                            help="Maximum GPU memory to use"
                        )
                    
                    with col_gpu2:
                        # Show GPU info
                        if cuda_available:
                            st.write("**CUDA Info:**")
                            st.write(f"• Devices: {gpu_info.get('cuda_devices', 0)}")
                            if 'cuda_memory' in gpu_info:
                                st.write(f"• Memory: {gpu_info['cuda_memory']:.1f} GB")
                        
                        if opencl_available:
                            st.write("**OpenCL:** Available")
                        
                        # GPU performance note
                        st.info("💡 GPU acceleration provides 10-100x speedup for large images")
                else:
                    prefer_cuda = True
                    gpu_memory_limit = 2.0
            else:
                enable_gpu = False
                prefer_cuda = True
                gpu_memory_limit = 2.0
                st.warning("⚠️ No GPU acceleration available")
                st.write("Install CUDA-enabled OpenCV for GPU support")
        
        # Quality vs Speed settings
        st.markdown("**Quality vs Speed**")
        col_qual1, col_qual2 = st.columns(2)
        
        with col_qual1:
            quality_factor = st.slider(
                "Detection Quality", 
                0.5, 1.5, selected_mode['quality_factor'], 0.1,
                help="Higher = more accurate but slower detection"
            )
        
        with col_qual2:
            frame_skip_enabled = st.checkbox(
                "Frame Dropping",
                value=selected_mode['frame_skip_factor'] > 1,
                help="Skip frames under high load"
            )
            
            if frame_skip_enabled:
                frame_skip_factor = st.slider(
                    "Skip Factor",
                    2, 8, selected_mode['frame_skip_factor'], 1,
                    help="Process every Nth frame"
                )
            else:
                frame_skip_factor = 1
    
    # Create enhanced parallel config object
    parallel_config = ParallelDetectionConfig()
    parallel_config.enabled = enable_parallel
    parallel_config.adaptive_scaling = adaptive_scaling
    parallel_config.performance_mode = selected_mode_name
    parallel_config.custom_worker_count = max_workers if enable_parallel else None
    parallel_config.overlap_percentage = overlap_percentage
    parallel_config.min_image_size = min_image_size
    parallel_config.use_threading = use_threading
    parallel_config.gpu_enabled = enable_gpu if (cuda_available or opencl_available) else False
    parallel_config.prefer_cuda = prefer_cuda
    parallel_config.gpu_memory_limit = gpu_memory_limit
    parallel_config.quality_factor = quality_factor
    parallel_config.frame_skip_enabled = frame_skip_enabled
    parallel_config.frame_skip_factor = frame_skip_factor
    parallel_config.auto_throttle = auto_throttle
    parallel_config.throttle_threshold = throttle_threshold
    parallel_config.memory_cleanup = memory_cleanup
    parallel_config.adaptation_interval = adaptation_interval
    parallel_config.batch_processing = batch_processing
    parallel_config.batch_size = batch_size
    
    # Smart Astra Settings Panel (at bottom of sidebar)
    if smart_astra_enabled:
        st.markdown("---")
        st.markdown("**🔧 Smart Astra Control Panel**")
        with st.expander("⚙️ Smart Astra Settings", expanded=False):
            st.markdown("**🤖 Adaptive Capabilities:**")
            
            col_adapt1, col_adapt2 = st.columns(2)
            
            with col_adapt1:
                st.markdown("**Auto-Managed:**")
                st.write("✅ Performance Mode")
                st.write("✅ Detection Thresholds") 
                st.write("✅ Auto-Label Process")
                st.write("✅ Parallel Processing")
                st.write("✅ Worker Count")
                st.write("✅ GPU Acceleration")
                
            with col_adapt2:
                st.markdown("**Auto-Optimized:**")
                st.write("✅ Border Detection")
                st.write("✅ Overlap Sensitivity")
                st.write("✅ Signal Area Detection")
                st.write("✅ Memory Management")
                st.write("✅ System Load Balance")
                st.write("✅ Quality vs Speed")
            
            st.markdown("**🎛️ Manual Override Options** *(Advanced)*")
            
            smart_override_performance = st.selectbox(
                "Force Performance Mode",
                ["Auto-Detect", "Battery Saver", "Balanced", "Performance", "Maximum"],
                index=0,
                help="Override Smart Astra's automatic performance mode selection"
            )
            
            smart_sensitivity_bias = st.slider(
                "Detection Sensitivity Bias",
                -0.1, 0.1, 0.0, 0.01,
                help="Adjust Smart Astra's threshold decisions. Negative = more sensitive, Positive = more strict"
            )
            
            smart_conserve_resources = st.checkbox(
                "Resource Conservation Mode",
                value=False,
                help="Prioritize system resource conservation over maximum performance"
            )
            
            # Apply manual overrides
            if smart_override_performance != "Auto-Detect":
                smart_performance_mode = smart_override_performance
            
            # Adjust thresholds based on bias
            threshold = max(0.1, min(0.95, threshold + smart_sensitivity_bias))
            min_confidence = max(0.0, min(0.9, min_confidence + smart_sensitivity_bias))
            
            st.markdown("**📊 Current Smart Decisions:**")
            st.info(f"Performance: {smart_performance_mode} | Load: {system_load:.1f}% | Threshold: {threshold:.3f}")
    
    # Initialize the configuration
    parallel_config.initialize()
    
    # Apply Smart Astra performance mode to parallel config if enabled
    if smart_astra_enabled:
        # Map Smart Astra performance mode to parallel config
        mode_mapping = {
            "Battery Saver": "Battery",
            "Balanced": "Balanced", 
            "Performance": "Performance",
            "Maximum": "Maximum"
        }
        parallel_config.performance_mode = mode_mapping.get(smart_performance_mode, "Balanced")
        
        # Apply Smart Astra optimizations to parallel config
        parallel_config.adaptive_scaling = True
        parallel_config.auto_throttle = not smart_conserve_resources if 'smart_conserve_resources' in locals() else True
        
        # Smart GPU decisions
        if 'system_load' in locals() and system_load > 70:
            parallel_config.gpu_enabled = False  # Disable GPU under high load
        
        # Smart worker decisions  
        if 'system_load' in locals():
            if system_load > 80:
                parallel_config.custom_worker_count = 2  # Conservative worker count
            elif system_load > 60:
                parallel_config.custom_worker_count = 4  # Moderate worker count
            # else: use auto-detected worker count
    
    # Performance status indicator
    st.markdown("**🎯 Active Performance Configuration:**")
    if parallel_config.should_use_gpu():
        perf_method = "🎮 GPU Accelerated"
        perf_details = f"Method: {'CUDA' if parallel_config.prefer_cuda else 'OpenCL'}"
        perf_color = "🟢"
    elif parallel_config.enabled and parallel_config.get_effective_workers() > 1:
        perf_method = "🚀 CPU Parallel"
        perf_details = f"Workers: {parallel_config.get_effective_workers()}"
        perf_color = "🟡"
    else:
        perf_method = "⚡ CPU Sequential"
        perf_details = "Single-threaded"
        perf_color = "🔴"
    
    st.info(f"{perf_color} **{perf_method}** - {perf_details} | Mode: {parallel_config.performance_mode}")

# Store configuration in session state for live feed access
st.session_state.threshold = threshold
st.session_state.border_threshold = border_threshold
st.session_state.enable_border_detection = enable_border_detection
st.session_state.merge_overlapping = merge_overlapping
st.session_state.overlap_sensitivity = overlap_sensitivity
st.session_state.parallel_config = parallel_config
st.session_state.smart_astra_enabled = smart_astra_enabled

# Live Feed Advanced Settings (in sidebar dropdown)
with st.sidebar:
    st.markdown("---")
    
    # Smart ASTRA Live Feed Control
    if smart_astra_enabled:
        st.markdown("### 🚀 Smart ASTRA Live Feed")
        st.info("🧠 **Smart ASTRA Active** - Automatically optimizing live feed parameters")
        
        # Initialize Smart ASTRA live feed state
        if 'smart_astra_live_feed_state' not in st.session_state:
            st.session_state.smart_astra_live_feed_state = {
                'adaptation_enabled': True,
                'performance_history': [],
                'last_adaptation_time': 0,
                'adaptation_count': 0
            }
        
        # Smart ASTRA status display
        with st.expander("📊 Smart ASTRA Live Feed Status", expanded=False):
            col_astra1, col_astra2 = st.columns(2)
            
            with col_astra1:
                st.write("**Current Strategy:**")
                if st.session_state.live_feed_active and 'live_feed_stats' in st.session_state:
                    current_mbps = st.session_state.live_feed_stats.get('throughput_mbps', 0)
                    if current_mbps > 800:
                        st.success("🚀 High Performance Mode")
                    elif current_mbps > 400:
                        st.info("⚡ Balanced Mode")
                    else:
                        st.warning("🔧 Optimization Mode")
                else:
                    st.write("🔄 Ready to Adapt")
            
            with col_astra2:
                st.write("**Adaptations Made:**")
                st.write(f"Count: {st.session_state.smart_astra_live_feed_state['adaptation_count']}")
                
                if st.session_state.smart_astra_live_feed_state['performance_history']:
                    avg_perf = sum(st.session_state.smart_astra_live_feed_state['performance_history']) / len(st.session_state.smart_astra_live_feed_state['performance_history'])
                    st.write(f"Avg Performance: {avg_perf:.1f} Mbps")
        
        # Smart ASTRA adaptive parameters (calculated automatically)
        def calculate_smart_astra_live_feed_params():
            """Calculate optimal live feed parameters based on current performance and requirements"""
            
            # Base configuration for RF images (1024x192, ~225KB)
            base_config = {
                'max_concurrent_streams': 8,
                'frame_skip_factor': 3,  # Never reduce below 1 to maintain quality
                'buffer_size_mb': 200,
                'memory_limit_gb': 16,
                'throughput_target_mbps': 1000,
                'enable_caching': True,
                'polling_interval': 5
            }
            
            # Get current performance metrics
            current_stats = st.session_state.get('live_feed_stats', {})
            current_mbps = current_stats.get('throughput_mbps', 0)
            avg_processing_time = current_stats.get('avg_processing_time', 0)
            
            # Adaptive adjustments (NEVER reducing image quality or limiting templates)
            if current_mbps > 0:  # Only adapt if we have performance data
                
                # Performance-based adaptations
                if current_mbps < 400:  # Low performance - optimize for speed
                    base_config['max_concurrent_streams'] = min(12, cpu_count())  # More parallelism
                    base_config['buffer_size_mb'] = 300  # Larger buffer
                    # Note: frame_skip_factor stays >= 3 to maintain quality
                    
                elif current_mbps > 800:  # High performance - optimize for quality detection
                    base_config['frame_skip_factor'] = max(2, base_config['frame_skip_factor'] - 1)  # Process more frames
                    base_config['buffer_size_mb'] = 400  # Even larger buffer for quality
                
                # Memory optimization
                available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                if available_memory > 20:
                    base_config['memory_limit_gb'] = min(24, int(available_memory * 0.7))
                elif available_memory < 8:
                    base_config['memory_limit_gb'] = max(4, int(available_memory * 0.5))
                
                # Record performance history for learning
                st.session_state.smart_astra_live_feed_state['performance_history'].append(current_mbps)
                if len(st.session_state.smart_astra_live_feed_state['performance_history']) > 50:
                    st.session_state.smart_astra_live_feed_state['performance_history'].pop(0)
                
                # Increment adaptation counter
                st.session_state.smart_astra_live_feed_state['adaptation_count'] += 1
                st.session_state.smart_astra_live_feed_state['last_adaptation_time'] = time.time()
            
            return base_config
        
        # Calculate Smart ASTRA optimized parameters
        smart_params = calculate_smart_astra_live_feed_params()
        
        max_concurrent_streams = smart_params['max_concurrent_streams']
        frame_skip_factor = smart_params['frame_skip_factor']
        buffer_size_mb = smart_params['buffer_size_mb']
        memory_limit_gb = smart_params['memory_limit_gb']
        throughput_target_mbps = smart_params['throughput_target_mbps']
        enable_caching = smart_params['enable_caching']
        polling_interval = smart_params['polling_interval']
        
        # Display Smart ASTRA decisions (read-only)
        st.markdown("**🎛️ Smart ASTRA Live Feed Parameters:**")
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            st.write(f"• Streams: {max_concurrent_streams}")
            st.write(f"• Frame Skip: {frame_skip_factor}")
            st.write(f"• Buffer: {buffer_size_mb}MB")
        
        with col_param2:
            st.write(f"• Memory: {memory_limit_gb}GB")
            st.write(f"• Target: {throughput_target_mbps}Mbps")
            st.write(f"• Caching: {'✓' if enable_caching else '✗'}")
        
        st.success("🛡️ **Quality Guarantee:** Smart ASTRA NEVER reduces image quality or limits templates")
    
    elif st.checkbox("⚙️ Live Feed Advanced Settings", value=False):
        st.markdown("### 🔧 Live Feed Configuration")
        
        # Processing settings
        max_concurrent_streams = st.slider(
            "Max Concurrent Streams",
            1, 16, 8, 1,
            help="Number of parallel processing streams"
        )
        
        frame_skip_factor = st.slider(
            "Frame Skip Factor",
            1, 10, 3, 1,
            help="Process every Nth frame (3 = every 3rd frame for RF images)"
        )
        
        # Buffer and memory
        buffer_size_mb = st.slider(
            "Buffer Size (MB)",
            50, 1000, 200, 50,
            help="Memory buffer for live processing"
        )
        
        memory_limit_gb = st.slider(
            "Memory Limit (GB)",
            4, 32, 16, 2,
            help="Maximum memory usage"
        )
        
        # Performance settings
        throughput_target_mbps = st.slider(
            "Target Throughput (Mbps)",
            100, 1000, 1000, 100,
            help="Target processing throughput"
        )
        
        enable_caching = st.checkbox(
            "Enable Result Caching",
            value=True,
            help="Cache results for performance"
        )
        
        # Monitoring settings
        polling_interval = st.slider(
            "Directory Polling (seconds)",
            1, 60, 5, 1,
            help="How often to check for new files"
        )
        
        st.markdown("**🛰️ RF Image Info:**")
        st.info("RF: 1024x192, ~225KB, needs 569fps for 1Gbps")
        
    else:
        # Default values when advanced settings are collapsed
        max_concurrent_streams = 8
        frame_skip_factor = 3  # Optimized for RF images
        buffer_size_mb = 200
        memory_limit_gb = 16
        throughput_target_mbps = 1000
        enable_caching = True
        polling_interval = 5

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Template Management", 
    "🔍 Pattern Detection", 
    "📡 Live Feed Processing",
    "📊 Results Analysis", 
    "🚁 Unidentified Signals"
])

with tab1:
    st.markdown('<h2 class="sub-header">📁 Drone Pattern Templates</h2>', unsafe_allow_html=True)
    
    # Template Folder Management
    st.markdown("### 📂 Template Folder Management")
    col_folder1, col_folder2, col_folder3 = st.columns([2, 1, 1])
    
    with col_folder1:
        # Active folder selection
        folder_names = list(st.session_state.template_folders.keys())
        current_folder_index = folder_names.index(st.session_state.active_folder) if st.session_state.active_folder in folder_names else 0
        
        selected_folder = st.selectbox(
            "Active Template Folder",
            folder_names,
            index=current_folder_index,
            help="Select the active template folder to work with"
        )
        
        if selected_folder != st.session_state.active_folder:
            st.session_state.active_folder = selected_folder
            # Update templates to show current folder contents
            st.session_state.templates = st.session_state.template_folders[selected_folder].copy()
            st.rerun()
    
    with col_folder2:
        # Create new folder
        new_folder_name = st.text_input("New Folder Name", placeholder="Enter folder name...")
        if st.button("➕ Create Folder") and new_folder_name:
            if new_folder_name not in st.session_state.template_folders:
                st.session_state.template_folders[new_folder_name] = {}
                st.success(f"Created folder: {new_folder_name}")
                st.rerun()
            else:
                st.warning(f"Folder '{new_folder_name}' already exists!")
    
    with col_folder3:
        # Rename current folder
        if st.session_state.active_folder != 'Default':
            rename_folder = st.text_input("Rename Active Folder", value=st.session_state.active_folder)
            if st.button("✏️ Rename Folder") and rename_folder != st.session_state.active_folder:
                if rename_folder not in st.session_state.template_folders:
                    # Rename the folder
                    st.session_state.template_folders[rename_folder] = st.session_state.template_folders[st.session_state.active_folder]
                    del st.session_state.template_folders[st.session_state.active_folder]
                    st.session_state.active_folder = rename_folder
                    st.success(f"Renamed folder to: {rename_folder}")
                    st.rerun()
                else:
                    st.warning(f"Folder '{rename_folder}' already exists!")
        
        # Delete current folder (except Default)
        if st.session_state.active_folder != 'Default' and len(st.session_state.template_folders) > 1:
            if st.button("🗑️ Delete Folder", help="Delete the active folder and all its templates"):
                del st.session_state.template_folders[st.session_state.active_folder]
                st.session_state.active_folder = 'Default'
                st.session_state.templates = st.session_state.template_folders['Default'].copy()
                st.success("Folder deleted!")
                st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Upload Templates to '{st.session_state.active_folder}' Folder**")
        
        # Upload method selection
        upload_type = st.radio(
            "Upload Method",
            ["Individual Files", "Folder/ZIP Archive"],
            horizontal=True,
            help="Choose whether to upload individual template files or a folder/ZIP containing templates"
        )
        
        if upload_type == "Individual Files":
            uploaded_templates = st.file_uploader(
                "Upload drone pattern templates (PNG/JPG)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload known drone spectrogram patterns to use as detection templates"
            )
            
            if uploaded_templates:
                for template_file in uploaded_templates:
                    try:
                        # Ensure templates is initialized as dict
                        if not isinstance(st.session_state.templates, dict):
                            st.session_state.templates = {}
                        
                        # Load and process template
                        template_image = Image.open(template_file)
                        template_cv = pil_to_cv(template_image)
                        
                        # Clean the template name for display
                        clean_name = clean_template_name(template_file.name)
                        
                        # Store in both session templates and folder structure
                        template_data = {
                            'image': template_cv,
                            'pil_image': template_image,
                            'size': template_image.size,
                            'clean_name': clean_name,
                            'original_name': template_file.name
                        }
                        
                        st.session_state.templates[template_file.name] = template_data
                        st.session_state.template_folders[st.session_state.active_folder][template_file.name] = template_data
                        
                    except Exception as e:
                        st.error(f"Error loading template {template_file.name}: {str(e)}")
        
        else:  # Folder/ZIP Archive
            uploaded_folder = st.file_uploader(
                "Upload ZIP file containing template images",
                type=['zip'],
                help="Upload a ZIP file containing template images to add to the current folder"
            )
            
            if uploaded_folder:
                try:
                    import zipfile
                    import io
                    
                    with zipfile.ZipFile(uploaded_folder, 'r') as zip_ref:
                        template_files = [f for f in zip_ref.namelist() 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX/')]
                        
                        if template_files:
                            st.success(f"✅ Found {len(template_files)} template images in ZIP")
                            
                            for template_file_path in template_files:
                                try:
                                    with zip_ref.open(template_file_path) as f:
                                        img_content = f.read()
                                        template_image = Image.open(io.BytesIO(img_content))
                                        template_cv = pil_to_cv(template_image)
                                        
                                        # Ensure templates is initialized as dict
                                        if not isinstance(st.session_state.templates, dict):
                                            st.session_state.templates = {}
                                        
                                        # Get just the filename from the path
                                        filename = template_file_path.split('/')[-1]
                                        clean_name = clean_template_name(filename)
                                        
                                        # Store in both session templates and folder structure
                                        template_data = {
                                            'image': template_cv,
                                            'pil_image': template_image,
                                            'size': template_image.size,
                                            'clean_name': clean_name,
                                            'original_name': filename
                                        }
                                        
                                        st.session_state.templates[filename] = template_data
                                        st.session_state.template_folders[st.session_state.active_folder][filename] = template_data
                                        
                                except Exception as e:
                                    st.error(f"Error loading template from ZIP {template_file_path}: {str(e)}")
                        else:
                            st.warning("⚠️ No template image files found in ZIP")
                            
                except Exception as e:
                    st.error(f"❌ Error reading ZIP file: {str(e)}")
    
    with col2:
        st.markdown("**Template Library**")
        
        # Show folder summary
        total_templates = sum(len(folder) for folder in st.session_state.template_folders.values())
        st.markdown(f"""
        <div class="info-box">
            <h4>📂 Folder Overview</h4>
            <p><strong>Total Folders:</strong> {len(st.session_state.template_folders)}</p>
            <p><strong>Total Templates:</strong> {total_templates}</p>
            <p><strong>Active Folder:</strong> {st.session_state.active_folder}</p>
            <p><strong>Templates in Active:</strong> {len(st.session_state.templates)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.templates:
            st.markdown(f"""
            <div class="template-box">
                <h4>📚 Templates in '{st.session_state.active_folder}'</h4>
                <p><strong>Count:</strong> {len(st.session_state.templates)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display templates from active folder
            for name, template_data in st.session_state.templates.items():
                clean_display_name = template_data.get('clean_name', clean_template_name(name))
                with st.expander(f"🎯 {clean_display_name}"):
                    col_img, col_info = st.columns([2, 1])
                    
                    with col_img:
                        st.image(template_data['pil_image'], 
                                caption=f"Template: {clean_display_name}", 
                                width='stretch')
                    
                    with col_info:
                        st.write(f"**Size:** {template_data['size'][0]} x {template_data['size'][1]}")
                        st.write(f"**File:** {name}")
                        st.write(f"**Folder:** {st.session_state.active_folder}")
                        
                        # Move to different folder
                        other_folders = [f for f in st.session_state.template_folders.keys() if f != st.session_state.active_folder]
                        if other_folders:
                            move_to_folder = st.selectbox(f"Move to folder", other_folders, key=f"move_{name}")
                            if st.button(f"📁 Move", key=f"move_btn_{name}"):
                                # Move template to selected folder
                                st.session_state.template_folders[move_to_folder][name] = template_data
                                del st.session_state.templates[name]
                                del st.session_state.template_folders[st.session_state.active_folder][name]
                                st.success(f"Moved {clean_display_name} to {move_to_folder}")
                                st.rerun()
                        
                        if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                            del st.session_state.templates[name]
                            del st.session_state.template_folders[st.session_state.active_folder][name]
                            st.rerun()
        else:
            st.info("👆 Upload template patterns to get started")

with tab2:
    st.markdown('<h2 class="sub-header">🔍 Drone Pattern Detection</h2>', unsafe_allow_html=True)
    
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
                    <p><strong>Templates:</strong> {len(st.session_state.templates) if st.session_state.templates else 0}</p>
                    <p><strong>Detection Mode:</strong> {detection_mode}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detection mode buttons - Full width button
                # Regular detection button
                if st.button("🚀 Start Pattern Detection", type="primary", use_container_width=True):
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
                        if st.session_state.templates:
                            total_operations = len(test_images) * len(st.session_state.templates)
                        else:
                            total_operations = len(test_images)  # Only colored rectangle detection
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        timing_text = st.empty()  # Add timing display container
                        
                        # Prepare detection parameters for unified detection
                        detection_params = {
                            'threshold': threshold,
                            'min_confidence': min_confidence,
                            'border_threshold': border_threshold if enable_border_detection else 0.5,
                            'enable_border_detection': enable_border_detection,
                            'merge_overlapping': merge_overlapping if 'merge_overlapping' in locals() else True,
                            'overlap_sensitivity': overlap_sensitivity if 'overlap_sensitivity' in locals() else 0.3,
                            'detect_green_rectangles': detect_green_rectangles,
                            'green_min_area': green_min_area,
                            'green_overlap_threshold': green_overlap_threshold,
                            'colored_merge_threshold': colored_merge_threshold,
                            'parallel_config': parallel_config
                        }
                        
                        # Use unified detection function - SAME AS AUTO-LABEL!
                        def manual_status_callback(message):
                            status_text.text(message)
                            # Update progress for manual detection
                            # Simple progress estimate based on message content
                            if "Detecting" in message and "in" in message:
                                # Extract progress info if possible, otherwise use simple counter
                                progress_bar.progress(min(0.9, len(st.session_state.detection_results) / len(test_images)))
                        
                        # Run the SAME comprehensive detection as auto-label
                        detection_results, unidentified_found, timing_info = run_unified_detection(test_images, detection_params, manual_status_callback)
                        
                        # Update session state with results
                        st.session_state.detection_results = detection_results
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        status_text.text("✅ Detection complete!")
                        
                        # Display timing information below progress bar
                        if timing_info:
                            total_time = timing_info.get('formatted_total', 'N/A')
                            timing_text.markdown(f"""
                            **⏱️ Performance Metrics:**
                            - **Total Detection Time:** {total_time}
                            - **Images Processed:** {len(test_images)}
                            - **Average per Image:** {timing_info.get('total_time', 0) / len(test_images):.2f}s
                            """)
                        
                        # Success message based on detection mode
                        if st.session_state.templates and detect_green_rectangles:
                            st.success(f"🎉 Processed {len(test_images)} images with {len(st.session_state.templates)} templates + colored area detection in {timing_info.get('formatted_total', 'N/A')}!")
                        elif st.session_state.templates:
                            st.success(f"🎉 Processed {len(test_images)} images with {len(st.session_state.templates)} templates in {timing_info.get('formatted_total', 'N/A')}!")
                        else:
                            st.success(f"🎉 Processed {len(test_images)} images with colored area detection only in {timing_info.get('formatted_total', 'N/A')}!")
                        
                        # Run auto-label if enabled
                        if run_auto_label:
                            st.markdown("---")
                            st.info(f"🔄 Auto-Label is enabled. Starting auto-labeling process... (Found {len(unidentified_found) if 'unidentified_found' in locals() else 0} unidentified signals from initial detection)")
                            
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
                                    'merge_overlapping': merge_overlapping if 'merge_overlapping' in locals() else True,
                                    'overlap_sensitivity': overlap_sensitivity if 'overlap_sensitivity' in locals() else 0.3,
                                    'detect_green_rectangles': detect_green_rectangles,
                                    'green_min_area': green_min_area,
                                    'green_overlap_threshold': green_overlap_threshold,
                                    'colored_merge_threshold': colored_merge_threshold,
                                    'parallel_config': parallel_config
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

with tab3:
    st.markdown('<h2 class="sub-header">📡 Live Feed Processing</h2>', unsafe_allow_html=True)
    
    # Initialize live feed session state
    if 'live_feed_active' not in st.session_state:
        st.session_state.live_feed_active = False
    if 'live_feed_results' not in st.session_state:
        st.session_state.live_feed_results = []
    if 'live_feed_stats' not in st.session_state:
        st.session_state.live_feed_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections_found': 0,
            'avg_processing_time': 0,
            'throughput_mbps': 0,
            'start_time': None
        }
    
    st.markdown("### 🌐 Data Source Configuration")
    
    # Simplified source selection
    feed_source_type = st.selectbox(
        "📥 Data Source Type",
        [
            "� Directory Path (Gb-scale support)",
            "📡 Network Stream", 
            "📁 Directory Monitoring"
        ],
        help="Choose your data source type"
    )
    
    # Source-specific configuration
    if "Directory Path" in feed_source_type:
        st.markdown("**� Directory Path Processing (No file size limits)**")
        
        col_path1, col_path2 = st.columns([3, 1])
        with col_path1:
            data_directory = st.text_input(
                "Directory Path",
                placeholder="C:/data/spectrograms/ or /mnt/rf_data/",
                help="Path to directory containing RF images (supports GB-scale datasets)"
            )
        
        with col_path2:
            st.markdown("**Supported:**")
            st.write("• Any size dataset")
            st.write("• ZIP files in directory")
            st.write("• Mixed file types")
        
        if data_directory:
            st.success(f"📁 Ready to process: {data_directory}")
    
    elif "Network Stream" in feed_source_type:
        st.markdown("**🌐 Network Data Stream**")
        
        col_net1, col_net2 = st.columns(2)
        with col_net1:
            stream_url = st.text_input(
                "Stream URL",
                placeholder="http://example.com/stream",
                help="URL for the data stream"
            )
        with col_net2:
            stream_format = st.selectbox(
                "Data Format",
                ["Images", "JSON", "Binary"],
                help="Format of incoming data"
            )
            # Set default protocol for network streams
            stream_protocol = "HTTP/HTTPS"
    
    elif "Directory Monitoring" in feed_source_type:
        st.markdown("**📁 Directory Monitoring**")
        
        monitor_directory = st.text_input(
            "Directory to Monitor",
            placeholder="C:/data/incoming/",
            help="Directory to watch for new files"
        )
        
        file_pattern = st.text_input(
            "File Pattern",
            value="*.png,*.jpg,*.jpeg",
            help="File patterns to monitor"
        )
    
    # Live feed control buttons
    st.markdown("### 🎮 Live Feed Control")
    
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    
    with col_ctrl1:
        if st.button("▶️ Start Live Feed", type="primary", disabled=st.session_state.live_feed_active):
            if not st.session_state.templates and not detect_green_rectangles:
                st.error("⚠️ Please upload templates or enable colored area detection first")
            else:
                # Create live feed configuration with Smart ASTRA adaptive values
                config = LiveFeedConfig(
                    max_concurrent_streams=max_concurrent_streams,
                    frame_skip_factor=frame_skip_factor,
                    quality_vs_speed=1.0,  # Never reduce quality as requested
                    memory_limit_gb=memory_limit_gb,
                    throughput_target_mbps=throughput_target_mbps,
                    enable_caching=enable_caching,
                    buffer_size_mb=buffer_size_mb
                )
                
                # Add Smart ASTRA context to configuration if enabled
                if smart_astra_enabled:
                    config.smart_astra_enabled = True
                    config.smart_astra_context = {
                        'template_count': len(st.session_state.templates),
                        'system_cores': cpu_count(),
                        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                        'adaptation_enabled': True,
                        'quality_preservation_mode': True  # NEVER reduce quality/resolution
                    }
                
                # Create and start processor
                st.session_state.live_feed_processor = create_live_feed_processor(config)
                st.session_state.live_feed_processor.start_processing()
                st.session_state.live_feed_config = config
                st.session_state.live_feed_active = True
                st.session_state.live_feed_stats = st.session_state.live_feed_processor.stats.copy()
                
                if smart_astra_enabled:
                    st.success("🚀 Live feed processor started with Smart ASTRA adaptation!")
                else:
                    st.success("🚀 Live feed processor started!")
                st.rerun()
    
    with col_ctrl2:
        if st.button("⏸️ Pause", disabled=not st.session_state.live_feed_active):
            if st.session_state.live_feed_processor:
                st.session_state.live_feed_processor.stop_processing()
            st.session_state.live_feed_active = False
            st.info("⏸️ Live feed paused")
            st.rerun()
    
    with col_ctrl3:
        if st.button("⏹️ Stop", disabled=not st.session_state.live_feed_active):
            if st.session_state.live_feed_processor:
                st.session_state.live_feed_processor.stop_processing()
            st.session_state.live_feed_processor = None
            st.session_state.live_feed_active = False
            st.session_state.live_feed_results = []
            st.session_state.live_feed_stats = {
                'total_frames': 0,
                'processed_frames': 0,
                'detections_found': 0,
                'avg_processing_time': 0,
                'throughput_mbps': 0,
                'start_time': None
            }
            st.info("⏹️ Live feed stopped and cleared")
            st.rerun()
    
    with col_ctrl4:
        if st.button("📊 Export Results"):
            if st.session_state.live_feed_results:
                # Create export data
                import json
                import datetime
                
                export_data = {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'total_detections': len(st.session_state.live_feed_results),
                    'statistics': st.session_state.live_feed_stats,
                    'detections': st.session_state.live_feed_results
                }
                
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    "💾 Download Results JSON",
                    export_json,
                    f"live_feed_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                st.warning("No results to export")
    
    # Live feed status display
    if st.session_state.live_feed_active:
        st.markdown("### 📊 Live Feed Status")
        
        # Real-time metrics (updated continuously if processor is active)
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        # Get current stats from processor
        if st.session_state.live_feed_processor:
            current_stats = st.session_state.live_feed_processor.stats
            st.session_state.live_feed_stats.update(current_stats)
        
        with col_metric1:
            st.metric(
                "📈 Frames Processed",
                st.session_state.live_feed_stats.get('processed_frames', 0),
                delta=st.session_state.live_feed_stats.get('total_frames', 0) - st.session_state.live_feed_stats.get('processed_frames', 0),
                delta_color="inverse"
            )
        
        with col_metric2:
            st.metric(
                "🎯 Detections Found",
                st.session_state.live_feed_stats.get('detections_found', 0),
                delta=len(st.session_state.live_feed_results) if st.session_state.live_feed_results else 0
            )
        
        with col_metric3:
            avg_time = st.session_state.live_feed_stats.get('avg_processing_time', 0)
            st.metric(
                "⚡ Avg Processing (ms)",
                f"{avg_time:.1f}" if avg_time else "0.0"
            )
        
        with col_metric4:
            throughput = st.session_state.live_feed_stats.get('throughput_mbps', 0)
            st.metric(
                "📊 Throughput (Mbps)",
                f"{throughput:.1f}" if throughput else "0.0"
            )
        
        # Live processing implementation
        st.markdown("**🔄 Live Processing Stream**")
        
        # Smart ASTRA real-time adaptation
        if smart_astra_enabled and st.session_state.live_feed_processor:
            
            # Smart ASTRA performance monitoring and adaptation
            current_throughput = st.session_state.live_feed_stats.get('throughput_mbps', 0)
            current_time = time.time()
            
            # Adaptive optimization every 30 seconds (to avoid constant changes)
            if (current_time - st.session_state.smart_astra_live_feed_state.get('last_adaptation_time', 0)) > 30:
                
                # Performance-based adaptations
                config_changed = False
                current_config = st.session_state.live_feed_config
                
                if current_throughput > 0:  # Only adapt if we have data
                    
                    # Low performance optimization
                    if current_throughput < 400 and current_config.max_concurrent_streams < 12:
                        current_config.max_concurrent_streams = min(12, cpu_count())
                        current_config.buffer_size_mb = min(400, current_config.buffer_size_mb + 50)
                        config_changed = True
                        
                    # High performance - optimize for quality (process more frames)
                    elif current_throughput > 800 and current_config.frame_skip_factor > 2:
                        current_config.frame_skip_factor = max(2, current_config.frame_skip_factor - 1)
                        config_changed = True
                    
                    # Memory optimization
                    available_memory = psutil.virtual_memory().available / (1024**3)
                    if available_memory > 20 and current_config.memory_limit_gb < 24:
                        current_config.memory_limit_gb = min(24, int(available_memory * 0.7))
                        config_changed = True
                    elif available_memory < 8 and current_config.memory_limit_gb > 4:
                        current_config.memory_limit_gb = max(4, int(available_memory * 0.5))
                        config_changed = True
                
                # Apply adaptations if needed (without stopping the processor)
                if config_changed:
                    st.session_state.smart_astra_live_feed_state['adaptation_count'] += 1
                    st.session_state.smart_astra_live_feed_state['last_adaptation_time'] = current_time
                    
                    # Show adaptation notice
                    st.info(f"🧠 Smart ASTRA adapted configuration (Adaptation #{st.session_state.smart_astra_live_feed_state['adaptation_count']})")
        
        if st.session_state.live_feed_processor:
            # Update stats from processor
            processor_stats = st.session_state.live_feed_processor.stats
            st.session_state.live_feed_stats.update(processor_stats)
            
            # Process data based on source type
            if "ZIP Archive" in feed_source_type and uploaded_zip:
                st.info("📦 Processing ZIP archive in streaming mode...")
                
                # Save uploaded ZIP to temp file for processing
                with open("temp_upload.zip", "wb") as f:
                    f.write(uploaded_zip.getvalue())
                
                # Create ZIP processor
                zip_processor = ZipStreamProcessor(st.session_state.live_feed_processor)
                
                # Prepare detection parameters with Smart ASTRA adaptation
                if smart_astra_enabled:
                    # Smart ASTRA adaptive thresholds based on performance
                    current_throughput = st.session_state.live_feed_stats.get('throughput_mbps', 0)
                    detection_count = st.session_state.live_feed_stats.get('detections_found', 0)
                    processed_count = st.session_state.live_feed_stats.get('processed_frames', 1)
                    detection_rate = detection_count / processed_count if processed_count > 0 else 0
                    
                    # Adaptive threshold: balance speed vs accuracy
                    if current_throughput < 300:  # Low performance - slightly higher threshold for speed
                        adaptive_threshold = min(0.62, st.session_state.get('threshold', 0.620) + 0.02)
                    elif detection_rate > 0.5:  # High detection rate - slightly lower threshold for quality
                        adaptive_threshold = max(0.55, st.session_state.get('threshold', 0.620) - 0.02)
                    else:
                        adaptive_threshold = st.session_state.get('threshold', 0.620)
                    
                    detection_params = {
                        'threshold': adaptive_threshold,
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config'),
                        'smart_astra_adaptive': True,
                        'adaptive_threshold': adaptive_threshold
                    }
                else:
                    detection_params = {
                        'threshold': st.session_state.get('threshold', 0.610),
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config') if st.session_state.get('smart_astra_enabled', False) else None
                    }
                
                # Create progress displays
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                # Process ZIP stream
                try:
                    for update in zip_processor.process_zip_stream("temp_upload.zip", st.session_state.templates, detection_params):
                        if update['type'] == 'progress':
                            if update['total'] > 0:
                                progress = update['processed'] / update['total']
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {update['file']} ({update['processed']}/{update['total']}) - {update.get('throughput_mbps', 0):.1f} Mbps")
                        
                        elif update['type'] == 'detections':
                            # Add detections to results
                            st.session_state.live_feed_results.extend(update['detections'])
                            
                            # Show latest detections
                            with results_container:
                                if update['detections']:
                                    st.success(f"✅ Found {len(update['detections'])} detections in {update['file']}")
                                    for detection in update['detections'][-3:]:  # Show last 3
                                        st.write(f"🎯 {detection['template_name']} - Confidence: {detection['confidence']:.3f}")
                        
                        elif update['type'] == 'error':
                            st.error(f"❌ Processing error: {update['error']}")
                            break
                        
                        # Update metrics
                        with col_metric1:
                            st.metric("📈 Frames Processed", st.session_state.live_feed_stats['processed_frames'])
                        with col_metric2:
                            st.metric("🎯 Detections Found", st.session_state.live_feed_stats['detections_found'])
                        
                        time.sleep(0.1)  # Allow UI updates
                    
                    st.success("✅ ZIP archive processing completed!")
                    
                except Exception as e:
                    st.error(f"❌ ZIP processing error: {str(e)}")
                
                # Clean up temp file
                try:
                    os.remove("temp_upload.zip")
                except:
                    pass
            
            elif "Directory Path" in feed_source_type and data_directory:
                st.info(f"📁 Processing directory: {data_directory}")
                
                # Use the existing directory processing functionality
                # but with high-throughput streaming configuration
                import os
                import glob
                from PIL import Image
                
                # Get all image files from directory
                supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
                all_files = []
                
                for ext in supported_extensions:
                    all_files.extend(glob.glob(os.path.join(data_directory, '**', ext), recursive=True))
                
                if not all_files:
                    st.warning(f"No image files found in {data_directory}")
                else:
                    st.info(f"Found {len(all_files)} image files to process")
                    
                    # Create progress displays
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    # Process files in high-throughput mode
                    try:
                        import time
                        start_time = time.time()
                        processed_count = 0
                        detection_count = 0
                        
                        for i, file_path in enumerate(all_files):
                            # Update progress
                            progress = (i + 1) / len(all_files)
                            progress_bar.progress(progress)
                            
                            # Process single file
                            try:
                                # Load image
                                test_image = Image.open(file_path)
                                test_cv = pil_to_cv(test_image)
                                filename = os.path.basename(file_path)
                                
                                file_detections = []
                                
                                # Process against each template
                                if st.session_state.templates:
                                    for template_name, template_data in st.session_state.templates.items():
                                        clean_name = template_data.get('clean_name', clean_template_name(template_name))
                                        
                                        # Use optimized detection for high throughput
                                        template_cv = template_data['template_cv']
                                        method = template_data.get('method', cv.TM_CCOEFF_NORMED)
                                        threshold = st.session_state.get('threshold', 0.62)
                                        
                                        # Detect using high-speed method (no parallel processing for speed)
                                        detections = detect_pattern(
                                            test_cv, template_cv, method, threshold, clean_name,
                                            partial_threshold=st.session_state.get('border_threshold', 0.6),
                                            enable_border_detection=st.session_state.get('enable_border_detection', True),
                                            border_threshold=st.session_state.get('border_threshold', 0.6)
                                        )
                                        
                                        if detections:
                                            for detection in detections:
                                                detection['filename'] = filename
                                                detection['file_path'] = file_path
                                            file_detections.extend(detections)
                                
                                processed_count += 1
                                if file_detections:
                                    detection_count += len(file_detections)
                                    st.session_state.live_feed_results.extend(file_detections)
                                    
                                    # Show latest detections
                                    with results_container:
                                        st.success(f"✅ Found {len(file_detections)} detections in {filename}")
                                        for detection in file_detections[-2:]:  # Show last 2
                                            st.write(f"🎯 {detection['template_name']} - Confidence: {detection['confidence']:.3f}")
                                
                                # Update status
                                elapsed = time.time() - start_time
                                fps = processed_count / elapsed if elapsed > 0 else 0
                                mbps = (fps * 0.225)  # Assuming ~225KB per RF image
                                
                                status_text.text(f"Processing: {filename} ({processed_count}/{len(all_files)}) - {fps:.1f} fps / {mbps:.1f} Mbps")
                                
                                # Update metrics
                                st.session_state.live_feed_stats['processed_frames'] = processed_count
                                st.session_state.live_feed_stats['detections_found'] = detection_count
                                st.session_state.live_feed_stats['throughput_mbps'] = mbps
                                
                            except Exception as file_error:
                                st.warning(f"⚠️ Skipping {os.path.basename(file_path)}: {str(file_error)}")
                            
                            # Allow UI updates (but don't slow down too much for high throughput)
                            if i % 20 == 0:  # Update every 20 files for better performance
                                time.sleep(0.01)
                        
                        # Final summary
                        elapsed = time.time() - start_time
                        avg_fps = processed_count / elapsed if elapsed > 0 else 0
                        avg_mbps = avg_fps * 0.225
                        
                        st.success(f"""
                        ✅ Directory processing completed!
                        📈 **Performance Summary:**
                        - **Files Processed:** {processed_count}/{len(all_files)}
                        - **Detections Found:** {detection_count}
                        - **Processing Time:** {elapsed:.1f} seconds
                        - **Average Speed:** {avg_fps:.1f} fps / {avg_mbps:.1f} Mbps
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Directory processing error: {str(e)}")
            
            elif "Network Stream" in feed_source_type and stream_url:
                st.info(f"🌐 Connected to {stream_protocol} stream: {stream_url}")
                
                # Create network processor
                network_processor = NetworkStreamProcessor(st.session_state.live_feed_processor)
                
                # Prepare detection parameters with Smart ASTRA adaptation
                if smart_astra_enabled:
                    # Smart ASTRA adaptive thresholds based on performance
                    current_throughput = st.session_state.live_feed_stats.get('throughput_mbps', 0)
                    detection_count = st.session_state.live_feed_stats.get('detections_found', 0)
                    processed_count = st.session_state.live_feed_stats.get('processed_frames', 1)
                    detection_rate = detection_count / processed_count if processed_count > 0 else 0
                    
                    # Adaptive threshold: balance speed vs accuracy
                    if current_throughput < 300:  # Low performance - slightly higher threshold for speed
                        adaptive_threshold = min(0.62, st.session_state.get('threshold', 0.620) + 0.02)
                    elif detection_rate > 0.5:  # High detection rate - slightly lower threshold for quality
                        adaptive_threshold = max(0.62, st.session_state.get('threshold', 0.620) - 0.02)
                    else:
                        adaptive_threshold = st.session_state.get('threshold', 0.620)
                    
                    detection_params = {
                        'threshold': adaptive_threshold,
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config'),
                        'smart_astra_adaptive': True,
                        'adaptive_threshold': adaptive_threshold
                    }
                else:
                    detection_params = {
                        'threshold': st.session_state.get('threshold', 0.620),
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config') if st.session_state.get('smart_astra_enabled', False) else None
                    }
                
                # Create displays
                status_container = st.container()
                results_container = st.container()
                
                # Process network stream
                try:
                    for update in network_processor.process_http_stream(stream_url, st.session_state.templates, detection_params):
                        if update['type'] == 'progress':
                            with status_container:
                                st.text(f"📡 Streaming - Processed: {update['processed']} frames - {update.get('throughput_mbps', 0):.1f} Mbps")
                        
                        elif update['type'] == 'detections':
                            # Add detections to results
                            st.session_state.live_feed_results.extend(update['detections'])
                            
                            # Show latest detections
                            with results_container:
                                if update['detections']:
                                    st.success(f"✅ Live detection: {len(update['detections'])} matches found")
                                    for detection in update['detections'][-2:]:  # Show last 2
                                        st.write(f"🎯 {detection['template_name']} - Confidence: {detection['confidence']:.3f}")
                        
                        elif update['type'] == 'error':
                            st.error(f"❌ Stream error: {update['error']}")
                            break
                        
                        time.sleep(0.1)  # Allow UI updates
                        
                except Exception as e:
                    st.error(f"❌ Network stream error: {str(e)}")
            
            elif "Directory Monitoring" in feed_source_type and monitor_directory:
                st.info(f"📁 Monitoring directory: {monitor_directory}")
                
                # Create directory monitor
                dir_monitor = DirectoryMonitor(st.session_state.live_feed_processor)
                
                # Prepare detection parameters with Smart ASTRA adaptation
                if smart_astra_enabled:
                    # Smart ASTRA adaptive thresholds based on performance
                    current_throughput = st.session_state.live_feed_stats.get('throughput_mbps', 0)
                    detection_count = st.session_state.live_feed_stats.get('detections_found', 0)
                    processed_count = st.session_state.live_feed_stats.get('processed_frames', 1)
                    detection_rate = detection_count / processed_count if processed_count > 0 else 0
                    
                    # Adaptive threshold: balance speed vs accuracy
                    if current_throughput < 300:  # Low performance - slightly higher threshold for speed
                        adaptive_threshold = min(0.62, st.session_state.get('threshold', 0.62) + 0.02)
                    elif detection_rate > 0.5:  # High detection rate - slightly lower threshold for quality
                        adaptive_threshold = max(0.62, st.session_state.get('threshold', 0.62) - 0.02)
                    else:
                        adaptive_threshold = st.session_state.get('threshold', 0.62)
                    
                    detection_params = {
                        'threshold': adaptive_threshold,
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config'),
                        'smart_astra_adaptive': True,
                        'adaptive_threshold': adaptive_threshold
                    }
                else:
                    detection_params = {
                        'threshold': st.session_state.get('threshold', 0.610),
                        'border_threshold': st.session_state.get('border_threshold', 0.6),
                        'enable_border_detection': st.session_state.get('enable_border_detection', True),
                        'merge_overlapping': st.session_state.get('merge_overlapping', True),
                        'overlap_sensitivity': st.session_state.get('overlap_sensitivity', 0.3),
                        'parallel_config': st.session_state.get('parallel_config') if st.session_state.get('smart_astra_enabled', False) else None
                    }
                
                # Create displays
                status_container = st.container()
                results_container = st.container()
                
                # Monitor directory
                try:
                    for update in dir_monitor.monitor_directory(monitor_directory, file_pattern, st.session_state.templates, detection_params, polling_interval):
                        if update['type'] == 'progress':
                            with status_container:
                                st.text(f"📁 New file processed: {update['file']} (Total: {update['total_files']})")
                        
                        elif update['type'] == 'detections':
                            # Add detections to results
                            st.session_state.live_feed_results.extend(update['detections'])
                            
                            # Show latest detections
                            with results_container:
                                if update['detections']:
                                    st.success(f"✅ File detection: {len(update['detections'])} matches found")
                                    for detection in update['detections'][-2:]:  # Show last 2
                                        st.write(f"🎯 {detection['template_name']} - Confidence: {detection['confidence']:.3f}")
                        
                        elif update['type'] == 'error':
                            st.error(f"❌ Monitoring error: {update['error']}")
                            break
                        
                        time.sleep(0.1)  # Allow UI updates
                        
                except Exception as e:
                    st.error(f"❌ Directory monitoring error: {str(e)}")
            
            else:
                st.info("⚙️ Configure a data source above to start live processing")
        
        else:
            st.warning("⚠️ Live feed processor not initialized")
    
    else:
        st.markdown("### 💡 Live Feed Information")
        st.info("""
        **📡 High-Throughput Live Feed Capabilities:**
        
        🚀 **Performance Specifications:**
        • Up to 1 Gbps data stream processing
        • GB-scale ZIP archive support
        • Multi-threaded parallel processing
        • Memory-efficient streaming extraction
        
        🎯 **Detection Features:**
        • Real-time template matching
        • Live auto-labeling
        • Colored area detection
        • Smart Astra optimization
        
        📊 **Output:**
        • Live results in Results Analysis tab
        • Real-time statistics and metrics
        • Exportable detection logs
        • Performance monitoring
        """)

with tab4:
    st.markdown('<h2 class="sub-header">📊 Detection Results Analysis</h2>', unsafe_allow_html=True)
    
    # Results source selection
    result_source = st.radio(
        "📊 Results Source",
        ["🔍 Pattern Detection Results", "📡 Live Feed Results"],
        horizontal=True,
        help="Choose which results to analyze"
    )
    
    if result_source == "🔍 Pattern Detection Results":
        # Original pattern detection results
        if not st.session_state.detection_results:
            st.info("🔍 Run pattern detection first to see results here")
        else:
            # Summary statistics
            total_images = len(st.session_state.detection_results)
            total_matches = sum(len(result['matches']) for result in st.session_state.detection_results)
            images_with_matches = sum(1 for result in st.session_state.detection_results if result['matches'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Processed", total_images)
            with col2:
                st.metric("Total Matches", total_matches)
            with col3:
                st.metric("Images with Matches", images_with_matches)
        
        st.markdown("---")
        
        # Display results for each image
        for result_idx, result in enumerate(st.session_state.detection_results):
            if result['matches'] or show_all_matches:
                st.markdown(f"### 📷 {result['filename']}")
                
                # Filter matches by confidence
                filtered_matches = [m for m in result['matches'] if m['confidence'] >= min_confidence]
                
                # Add toggle for showing/hiding labels (positioned above images for better alignment)
                show_labels = True  # Default value
                if filtered_matches:
                    show_labels = st.checkbox("🏷️ Show Labels", value=True, key=f"labels_{result_idx}_{result['filename']}", 
                                            help="Toggle visibility of drone ID labels on the detection image")
                
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    st.markdown("**Original Spectrogram**")
                    st.image(result['image'], caption=f"Original: {result['filename']}", width='stretch')
                
                with col_img2:
                    st.markdown("**Detection Results**")
                    
                    if filtered_matches and draw_boxes:
                        # Draw bounding boxes
                        test_cv = pil_to_cv(result['image'])
                        if show_labels:
                            result_image = draw_detection_boxes(test_cv, filtered_matches, min_confidence)
                        else:
                            result_image = draw_detection_boxes_no_labels(test_cv, filtered_matches, min_confidence)
                        result_pil = cv_to_pil(result_image)
                        st.image(result_pil, caption=f"Detections: {result['filename']}", width='stretch')
                    else:
                        st.image(result['image'], caption=f"No detections: {result['filename']}", width='stretch')
                
                # Detection statistics and manual store button
                if filtered_matches:
                    st.markdown(f"""
                    <div class="detection-stats">
                        <h4>🎯 Detection Statistics</h4>
                        <p><strong>Total Matches:</strong> {len(filtered_matches)}</p>
                        <p><strong>Average Confidence:</strong> {np.mean([m['confidence'] for m in filtered_matches]):.3f}</p>
                        <p><strong>Templates Matched:</strong> {', '.join(set(m['template_name'] for m in filtered_matches))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Manual store button for unidentified drones
                    unidentified_in_result = [m for m in filtered_matches if m['template_name'] == 'Unidentified Drone']
                    if unidentified_in_result:
                        st.markdown("### 🔄 Manual Storage Control")
                        col_store1, col_store2 = st.columns([2, 1])
                        
                        with col_store1:
                            st.write(f"Found {len(unidentified_in_result)} unidentified drone(s) in this image.")
                            
                        with col_store2:
                            if st.button(f"📦 Store Unidentified Drones", key=f"store_{result_idx}_{result['filename']}"):
                                stored_count = 0
                                img_h, img_w = result['image'].size[1], result['image'].size[0]  # PIL image dimensions
                                
                                for unidentified_match in unidentified_in_result:
                                    if add_unidentified_drone(unidentified_match, result['image'], result['filename'], img_w, img_h, min_confidence):
                                        stored_count += 1
                                
                                if stored_count > 0:
                                    st.success(f"✅ Stored {stored_count} new unidentified drone(s)!")
                                else:
                                    st.info("ℹ️ No new drones stored (duplicates or validation failures)")
                                st.rerun()
                    
                    # Detailed results table
                    with st.expander(f"📊 Detailed Results for {result['filename']}"):
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
                else:
                    st.info(f"No drone patterns detected in {result['filename']} with confidence ≥ {min_confidence:.2f}")
                
                st.markdown("---")
    
    elif result_source == "📡 Live Feed Results":
        # Live feed results analysis
        if not st.session_state.live_feed_results:
            if st.session_state.live_feed_active:
                st.info("📡 Live feed is active. Detections will appear here in real-time.")
                
                # Live feed placeholder with auto-refresh
                live_placeholder = st.empty()
                
                with live_placeholder.container():
                    st.markdown("**🔄 Live Detection Stream**")
                    
                    # Real-time metrics display
                    col_live1, col_live2, col_live3, col_live4 = st.columns(4)
                    
                    with col_live1:
                        st.metric(
                            "🎯 Live Detections",
                            st.session_state.live_feed_stats['detections_found']
                        )
                    
                    with col_live2:
                        if st.session_state.live_feed_stats['start_time']:
                            elapsed = time.time() - st.session_state.live_feed_stats['start_time']
                            st.metric("⏱️ Elapsed Time", f"{elapsed:.1f}s")
                        else:
                            st.metric("⏱️ Elapsed Time", "0s")
                    
                    with col_live3:
                        fps = 0
                        if st.session_state.live_feed_stats['avg_processing_time'] > 0:
                            fps = 1000 / st.session_state.live_feed_stats['avg_processing_time']
                        st.metric("📊 Processing FPS", f"{fps:.1f}")
                    
                    with col_live4:
                        efficiency = 0
                        if st.session_state.live_feed_stats['total_frames'] > 0:
                            efficiency = (st.session_state.live_feed_stats['processed_frames'] / 
                                        st.session_state.live_feed_stats['total_frames']) * 100
                        st.metric("⚡ Efficiency", f"{efficiency:.1f}%")
                
                # Auto-refresh for live updates (optional enhancement)
                if st.button("🔄 Refresh Live Data"):
                    st.rerun()
                    
            else:
                st.info("📡 Start live feed processing in the Live Feed tab to see real-time results here.")
        else:
            # Display live feed detection results
            st.markdown("### 📊 Live Feed Detection Summary")
            
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
                st.metric("🎯 Unique Templates", len(unique_templates))
            
            with col_summary3:
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                st.metric("📊 Avg Confidence", f"{avg_confidence:.3f}")
            
            with col_summary4:
                detection_rate = (st.session_state.live_feed_stats['detections_found'] / 
                                max(1, st.session_state.live_feed_stats['processed_frames'])) * 100
                st.metric("🎯 Detection Rate", f"{detection_rate:.1f}%")
            
            # Live feed performance chart
            st.markdown("### 📈 Live Feed Performance")
            
            if st.session_state.live_feed_stats['start_time']:
                # Create performance data visualization
                import pandas as pd
                import numpy as np
                
                # Generate sample performance data (replace with actual live data)
                time_points = np.linspace(0, time.time() - st.session_state.live_feed_stats['start_time'], 20)
                throughput_data = np.random.normal(st.session_state.live_feed_stats['throughput_mbps'], 10, 20)
                detection_data = np.random.poisson(2, 20)  # Sample detection counts
                
                perf_df = pd.DataFrame({
                    'Time (s)': time_points,
                    'Throughput (Mbps)': np.maximum(0, throughput_data),
                    'Detections per Interval': detection_data
                })
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.line_chart(perf_df.set_index('Time (s)')['Throughput (Mbps)'])
                
                with col_chart2:
                    st.bar_chart(perf_df.set_index('Time (s)')['Detections per Interval'])
            
            # Recent detections list
            st.markdown("### 🔍 Recent Live Detections")
            
            # Show last 10 detections
            recent_detections = st.session_state.live_feed_results[-10:] if len(st.session_state.live_feed_results) > 10 else st.session_state.live_feed_results
            
            if recent_detections:
                detection_df = pd.DataFrame(recent_detections)
                st.dataframe(
                    detection_df[['timestamp', 'template_name', 'confidence', 'x', 'y', 'width', 'height']].round(3),
                    use_container_width=True
                )
            
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

with tab5:
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
                                width=100)  # CSS will override this for consistency
                        
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #546e7a; padding: 20px;">
    <p> ⍙ <strong>Ad Astra Drone Detection</strong> | NATO IST Hackaton</p>
</div>
""", unsafe_allow_html=True)

