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
from ui_styles import apply_custom_styles
from sidebar_config import render_sidebar


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
    # Load templates from disk and save them to session state
    loaded_templates = load_stored_templates()
    if loaded_templates:
        st.session_state.templates.update(loaded_templates)
        print(f"[APP] Loaded {len(loaded_templates)} templates from stored_templates/")
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

# Apply custom CSS styling
apply_custom_styles()

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

# Render sidebar and get all configuration parameters
(threshold, min_confidence, border_threshold, enable_border_detection,
 merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
 green_min_area, green_overlap_threshold, colored_merge_threshold,
 parallel_config, show_all_matches, draw_boxes, smart_astra_enabled,
 max_concurrent_streams, frame_skip_factor, buffer_size_mb, 
 memory_limit_gb, throughput_target_mbps, enable_caching, polling_interval,
 enable_temporal_tracking, temporal_search_region, temporal_full_search_interval,
 enable_train_mode) = render_sidebar()

# Import tab modules
from tabs import tab_template_management, tab_pattern_detection, tab_live_feed, tab_results_analysis, tab_unidentified_signals

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Template Management", 
    "🔍 Pattern Detection", 
    "📡 Live Feed Processing",
    "📊 Results Analysis", 
    "🚁 Unidentified Signals"
])

with tab1:
    tab_template_management.render()

with tab2:
    tab_pattern_detection.render(
        threshold, min_confidence, border_threshold, enable_border_detection,
        merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
        green_min_area, green_overlap_threshold, colored_merge_threshold,
        parallel_config, show_all_matches, draw_boxes
    )

with tab3:
    tab_live_feed.render(
        threshold, min_confidence, border_threshold, enable_border_detection,
        merge_overlapping, overlap_sensitivity, detect_green_rectangles, 
        green_min_area, green_overlap_threshold, colored_merge_threshold,
        parallel_config, smart_astra_enabled, max_concurrent_streams,
        frame_skip_factor, buffer_size_mb, memory_limit_gb, throughput_target_mbps,
        enable_caching, polling_interval, enable_train_mode
    )

with tab4:
    tab_results_analysis.render(show_all_matches, min_confidence, draw_boxes)

with tab5:
    tab_unidentified_signals.render(min_confidence, green_min_area, detect_green_rectangles)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #546e7a; padding: 20px;">
    <p> ⍙ <strong>Ad Astra Drone Detection</strong> | NATO IST Hackaton</p>
</div>
""", unsafe_allow_html=True)
