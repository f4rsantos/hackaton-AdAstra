"""
Sidebar Configuration Module for Ad Astra Drone Detection

This module handles all sidebar configuration including:
- Smart Astra intelligent optimization
- Template matching parameters
- Border and duplicate detection
- Colored rectangle detection
- Auto-label settings
- Performance management (System/Parallel/GPU)
- Live feed advanced settings (with AdAstra adaptive frame skipping)
"""

import streamlit as st
import psutil
from os import cpu_count
import time
from functions import GPUDetector, SystemMonitor, ParallelDetectionConfig, PerformanceMode


def render_sidebar():
    """
    Render the complete sidebar configuration and return all parameters.
    
    Returns:
        tuple: All configuration parameters needed by the tab modules:
            (threshold, min_confidence, border_threshold, enable_border_detection,
             merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
             green_min_area, green_overlap_threshold, colored_merge_threshold,
             parallel_config, show_all_matches, draw_boxes, smart_astra_enabled,
             max_concurrent_streams, frame_skip_factor, buffer_size_mb, 
             memory_limit_gb, throughput_target_mbps, enable_caching, polling_interval,
             enable_temporal_tracking, temporal_search_region, temporal_full_search_interval,
             enable_train_mode)
    """
    
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Detection Configuration</h2>', unsafe_allow_html=True)
        
        # Smart Astra Master Control - TOP PRIORITY
        st.markdown("---")
        st.markdown("**🚀 Smart Astra Mode**")
        smart_astra_enabled = st.checkbox(
            "🧠 Enable Smart Astra",
            value=True,
            help="Intelligent system that automatically optimizes all parameters based on current conditions and image characteristics"
        )
        
        # Training Mode - Top Priority Feature
        st.markdown("**🎓 Live Feed Training Mode**")
        enable_train_mode = st.checkbox(
            "Enable Live Train Mode",
            value=False,
            help="📚 Process subdirectories as template sources. Each subfolder becomes a template class with automatic naming."
        )
        
        if enable_train_mode:
            st.info("""
            **Train Mode Active:**
            - Each subfolder = template class
            - Subfolder name = template name
            - Multiple patterns → `foldername_1`, `foldername_2`, etc.
            - Templates saved to `stored_templates/<class>/`
            - Persistent across sessions
            """)
        
        if smart_astra_enabled:
            st.info("🚀 **Smart Astra Active** - System is automatically optimizing all parameters")
            
            # Smart Astra Status Display
            with st.expander("📊 Smart Astra Status", expanded=False):
                st.markdown("**Current Intelligent Decisions:**")
                
                # Get system info for smart decisions
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
                    
                    # Smart parameter decisions (can be overridden by user)
                    smart_threshold = 0.44  # Lowered for better multi-instance detection
                    smart_confidence = 0.5  # Lowered to allow more matches
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
                    
                    # Apply smart decisions to variables (can be overridden below)
                    threshold = smart_threshold
                    min_confidence = smart_confidence
                    run_auto_label = smart_auto_label
                    
                except Exception as e:
                    st.warning(f"Smart Astra initialization error: {e}")
                    # Fallback to improved defaults based on analysis
                    smart_performance_mode = "Balanced"
                    threshold = 0.55 # improved default
                    min_confidence = 0.60 # improved default
                    run_auto_label = True
                    smart_parallel = True
            
            # 🆕 OVERRIDE CONTROLS - User can override Smart Astra's decisions
            with st.expander("🎯 Live feed Parameters (Override Smart Astra)", expanded=False):
                st.markdown("**Manual Override Controls**")
                st.caption("Smart Astra sets optimal values, but you can override them here")
                
                col1, col2 = st.columns(2)
                with col1:
                    threshold_override = st.slider(
                        "Match Threshold", 
                        0.1, 0.95, threshold, 0.01,
                        help="Override Smart Astra's threshold. Higher = stricter matching",
                        key="threshold_override"
                    )
                    threshold = threshold_override  # User value overrides Smart Astra
                    
                with col2:
                    confidence_override = st.slider(
                        "Min Confidence", 
                        0.0, 0.9, min_confidence, 0.05,
                        help="Override Smart Astra's min confidence",
                        key="confidence_override"
                    )
                    min_confidence = confidence_override  # User value overrides Smart Astra
                
                st.info(f"✓ Using Threshold: **{threshold:.2f}**, Min Confidence: **{min_confidence:.2f}**")
        
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
                        threshold = st.slider("Match Threshold", 0.1, 0.95, 0.40, 0.01,
                                            help="Higher values = more strict matching")
                        show_all_matches = st.checkbox("Show All Matches", False)
                    with col2:
                        min_confidence = st.slider("Min Confidence", 0.0, 0.9, 0.5, 0.1)
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
            detect_green_rectangles = st.checkbox("Enable Unidentified Signal Detection", True,
                                                help="Automatically detect colored rectangles (green, cyan, teal) in spectrograms as unidentified signal areas. Manual verification recommended.")
            if detect_green_rectangles:
                green_min_area = st.slider("Minimum Colored Area", 20, 500, 20, 25,
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
            # Smart Astra optimizes colored area detection BUT allows user control
            st.markdown("**🎨 Unidentified Signal Detection** *(Smart Astra Optimized)*")
            
            detect_green_rectangles = st.checkbox(
                "🟢 Enable Unidentified Signal Detection", 
                value=True,  # Default to True for automatic signal detection
                help="⚠️ Experimental: Detect colored rectangles (green, cyan, teal) as unidentified signals. May cause false positives on certain spectrograms."
            )
            
            if detect_green_rectangles:
                green_min_area = 75  # Optimized minimum area
                green_overlap_threshold = 0.25  # Slightly more permissive overlap
                colored_merge_threshold = 0.18  # Conservative merging
                st.success("✅ Colored area detection enabled with smart parameters")
            else:
                green_min_area = 75
                green_overlap_threshold = 0.25
                colored_merge_threshold = 0.18
                st.info("ℹ️ Colored area detection disabled - only template-based detection active")
        
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
                        
                        # GPU Batch Processing
                        st.markdown("**Batch Processing (GPU)**")
                        gpu_batch_images = st.slider(
                            "Images per Batch",
                            min_value=1,
                            max_value=16,
                            value=4 if gpu_info.get('cuda_memory', 2) > 6 else 2,
                            help="Number of images to process simultaneously on GPU. Higher values increase throughput but use more GPU memory."
                        )
                        
                        st.info(f"💡 Horizontal parallelization: Each instance = 1 template × 1 image")
                    else:
                        prefer_cuda = True
                        gpu_memory_limit = 2.0
                        gpu_batch_images = 2
                else:
                    enable_gpu = False
                    prefer_cuda = True
                    gpu_memory_limit = 2.0
                    gpu_batch_images = 2
                    st.warning("⚠️ No GPU acceleration available")
                    st.write("Install CUDA-enabled OpenCV for GPU support")
            
            # Manual Frame Skip Control
            if not smart_astra_enabled:
                st.markdown("**Frame Processing**")
                frame_skip_enabled = st.checkbox(
                    "Enable Frame Skipping",
                    value=selected_mode['frame_skip_factor'] > 1,
                    help="Manually skip frames to reduce processing load"
                )
                
                if frame_skip_enabled:
                    frame_skip_factor = st.slider(
                        "Skip Factor",
                        2, 8, selected_mode['frame_skip_factor'], 1,
                        help="Process every Nth frame (higher = faster but may miss detections)"
                    )
                else:
                    frame_skip_factor = 1
            else:
                # Smart Astra mode: AdAstra handles frame skipping automatically
                frame_skip_enabled = False
                frame_skip_factor = 1
        
        # Standard quality for all modes
        quality_factor = 1.0
        
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
        parallel_config.gpu_batch_images = gpu_batch_images if enable_gpu else 1
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
                    st.write("✅ Adaptive Frame Skipping")
                
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
            enable_temporal_tracking = False  # Smart ASTRA: Safe for multi-drone by default
            temporal_search_region = 100
            temporal_full_search_interval = 50
            # Note: enable_train_mode is set at top level, not overridden here
            
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
            
            # Multi-drone temporal tracking
            st.markdown("**🎯 Temporal Tracking (Advanced)**")
            enable_temporal_tracking = st.checkbox(
                "Enable Multi-Drone Temporal Tracking",
                value=False,
                help="🚀 5-10x speed boost using temporal tracking. Uses identity-preserving tracking for multiple drones."
            )
            
            if enable_temporal_tracking:
                temporal_search_region = st.slider(
                    "Search Region Size (px)",
                    50, 200, 100, 10,
                    help="Size of temporal search region around last position"
                )
                temporal_full_search_interval = st.slider(
                    "Full Search Interval",
                    10, 100, 50, 10,
                    help="Do full-frame search every N frames to find new drones"
                )
            else:
                # Meaningless values considering it's disabled
                temporal_search_region = 100
                temporal_full_search_interval = 50
            
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
            enable_temporal_tracking = False  # Safe default for multi-drone
            temporal_search_region = 100
            temporal_full_search_interval = 50
            # Note: enable_train_mode is set at top level, not here
    
    # Return all configuration parameters
    return (
        threshold, min_confidence, border_threshold, enable_border_detection,
        merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
        green_min_area, green_overlap_threshold, colored_merge_threshold,
        parallel_config, show_all_matches, draw_boxes, smart_astra_enabled,
        max_concurrent_streams, frame_skip_factor, buffer_size_mb, 
        memory_limit_gb, throughput_target_mbps, enable_caching, polling_interval,
        enable_temporal_tracking, temporal_search_region, temporal_full_search_interval,
        enable_train_mode
    )
