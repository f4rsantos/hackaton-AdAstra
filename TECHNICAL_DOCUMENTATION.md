# Technical Documentation: RF Signal Detection and Classification System

## How to Run

Run `streamlit run app.py` on cmd to start

---

## Table of Contents

1. [Component Explanations](#component-explanations)
2. [Complete Workflow and Block Diagram](#complete-workflow-and-block-diagram)
3. [Metric Definitions and Formulas](#metric-definitions-and-formulas)

---

## 1. Component Explanations

### 1.1 Image Preprocessing Pipeline

**Purpose**: Standardize input images to optimal dimensions for template matching while preserving signal characteristics.

**Operations**:

1. **Image Loading**

   - Single pass through pixel data
   - Memory allocation for image buffer

2. **Resize Operation**

   - Original dimensions → Target dimensions (31,778 × 384)
   - Uses Lanczos resampling (high-quality interpolation)

3. **Image Chunking**
   - Linear scan to extract 2048×384 chunks
   - Number of chunks: ⌈W₂ / 2048⌉ ≈ 16 chunks per image

### 1.2 Template Matching (Core Algorithm)

**Method**: Normalized Cross-Correlation with Multi-Scale Matching

**Mathematical Foundation**:

The normalized cross-correlation coefficient between image region I and template T is:

```
NCC(x,y) = Σᵢⱼ [(I(x+i, y+j) - Ī) × (T(i,j) - T̄)] / (σᵢ × σₜ)
```

Where:

- I(x,y): Image intensity at position (x,y)
- T(i,j): Template intensity at position (i,j)
- Ī, T̄: Mean intensities of image region and template
- σᵢ, σₜ: Standard deviations of image region and template

**Process**:

1. **Single Template Match**

   - For each position (x,y) in image, compute correlation over template size

2. **OpenCV Optimization**

   - Uses Fast Fourier Transform (FFT) for large templates
   - Our implementation uses cv2.TM_CCOEFF_NORMED method

3. **Multi-Template Matching**

   - M templates processed sequentially
   - Each template match is independent

4. **Threshold Filtering**
   - Filter detection candidates above threshold

### 1.3 Non-Maximum Suppression (NMS)

**Algorithm**: Greedy Selection with IoU Threshold

**Pseudocode**:

```
NMS(detections, iou_threshold):
    Sort detections by confidence (descending)
    selected = []
    while detections not empty:
        best = detections[0]
        selected.append(best)
        detections.remove(best)
        for each det in detections:
            if IoU(best, det) > iou_threshold:
                detections.remove(det)
    return selected
```

**IoU Calculation**:

```
IoU = Area(intersection) / Area(union)
```

### 1.4 Post-Processing Operations

**1. Consolidation of Overlapping Detections**

**Complexity**: O(K²)

- Groups detections with IoU > threshold
- Averages bounding boxes and confidences

**2. Horizontal Fusion**

**Purpose**: Merge adjacent signal segments split across chunk boundaries.

**Conditions for Fusion**:

- Horizontal overlap < 30% (prevents over-merging)
- Vertical alignment > 70% (same frequency band)
- Same template class

**3. Similarity-Based Classification**

**Purpose**: Classify "Unidentified" signals by matching to known templates.

**Features Compared**:

- Height ratio: |h₁ - h₂| / max(h₁, h₂)
- Y-position similarity: |y₁ - y₂| / image_height

**Decision Rule**:

```
Classify as Template T if:
    height_similarity > 0.85 AND
    y_position_similarity > 0.80
```

### 1.5 Coordinate Transformation (Chunk → Original)

**Purpose**: Map detection coordinates from chunk space back to original image space for SigMF annotations.

**Three-Stage Transformation**:

1. **Chunk Space**: (0-2048, 0-384)

   - Detection coordinates relative to chunk

2. **Resized Image Space**: (0-31778, 0-384)

   ```
   resized_x = chunk_x + chunk_offset_x
   resized_y = chunk_y + chunk_offset_y
   ```

   **Complexity**: O(1) per detection

3. **Original Image Space**: (0-W_orig, 0-H_orig)
   ```
   original_x = resized_x × (W_orig / W_resized)
   original_y = resized_y × (H_orig / H_resized)
   ```

### 1.6 SigMF Annotation Generation

**Purpose**: Convert pixel coordinates to time-frequency domain for RF signal analysis tools.

**Coordinate Mapping Algorithm**:

1. **Time Axis (X-axis → Sample Index)**:

   ```
   num_spectrogram_cols = (total_samples - fft_size) / step_size + 1
   spec_x = x × (num_spectrogram_cols / image_width)
   start_sample = spec_x × step_size
   length_samples = spec_w × step_size
   ```

2. **Frequency Axis (Y-axis → Hz)**:

   ```
   # Flip Y-axis (image Y=0 at top, freq=0 at bottom)
   y_flipped = image_height - y - height

   spec_y = y_flipped × (fft_size / image_height)
   f_lo = (spec_y / fft_size) × sample_rate - (sample_rate / 2) + center_freq
   f_hi = ((spec_y + spec_h) / fft_size) × sample_rate - (sample_rate / 2) + center_freq
   ```

**Complexity**: O(D)

- Simple arithmetic operations per detection
- SigMF file I/O: O(D × log D) for insertion into sorted annotation list

**Metadata Extraction**:

- Read .sigmf-data file size: O(1)
- Calculate total_samples from file size and datatype

---

## 2. Complete Workflow and Block Diagram

### 2.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA ACQUISITION                       │
├─────────────────────────────────────────────────────────────────┤
│  • RF Spectrogram Images (PNG format)                           │
│  • Template Library (Stored drone signatures)                   │
│  • SigMF Metadata Files (.sigmf-meta, .sigmf-data)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Image Loading │───▶│ Resize Image │───▶│ Image Chunking│  │
│  │  (PIL/OpenCV)  │    │ (31778×384)  │    │ (2048×384)    │  │
│  └────────────────┘    └──────────────┘    └───────────────┘  │
│         │                      │                     │          │
│         │ Original: 101708×1229│                     │          │
│         │ Resized:  31778×384  │                     │          │
│         │ Chunks:   16 × 2048×384                   │          │
│         │                      │                     │          │
│  ┌──────▼──────────────────────▼─────────────────────▼───────┐ │
│  │          Chunk Metadata Generation                         │ │
│  │  • offset_x, offset_y (position in resized image)         │ │
│  │  • original_size (101708, 1229)                           │ │
│  │  • resized_size (31778, 384)                              │ │
│  │  • chunk_index (0-15)                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPLATE MATCHING ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each chunk (parallel processing):                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 1: Apply Colormap Normalization                      │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ • Convert to grayscale if needed                      │ │ │
│  │  │ • Apply cv2.COLORMAP_JET for consistent color scheme │ │ │
│  │  │ • Ensures templates and images use same color space  │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 2: Multi-Template Matching                           │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ For each template T in template library:             │ │ │
│  │  │                                                       │ │ │
│  │  │  1. Load template from disk (cached)                 │ │ │
│  │  │  2. Apply same colormap to template                  │ │ │
│  │  │  3. Run cv2.matchTemplate(chunk, T, TM_CCOEFF_NORMED)│ │ │
│  │  │     Result: Correlation map (W-Tw+1) × (H-Th+1)      │ │ │
│  │  │  4. Find peaks above threshold (0.7-0.9)             │ │ │
│  │  │  5. Extract bounding boxes and confidence scores     │ │ │
│  │  │                                                       │ │ │
│  │  │  Detection Format:                                    │ │ │
│  │  │  {                                                    │ │ │
│  │  │    'x': int,           # Left edge                   │ │ │
│  │  │    'y': int,           # Top edge                    │ │ │
│  │  │    'width': int,       # Box width                   │ │ │
│  │  │    'height': int,      # Box height                  │ │ │
│  │  │    'confidence': float, # NCC score [0-1]            │ │ │
│  │  │    'template_name': str # Drone class label          │ │ │
│  │  │  }                                                    │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 3: Non-Maximum Suppression (NMS)                     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Algorithm:                                            │ │ │
│  │  │  1. Sort detections by confidence (descending)        │ │ │
│  │  │  2. Select highest confidence detection               │ │ │
│  │  │  3. Remove all detections with IoU > 0.3             │ │ │
│  │  │  4. Repeat until no detections remain                │ │ │
│  │  │                                                       │ │ │
│  │  │ IoU Formula:                                          │ │ │
│  │  │  IoU(A,B) = Area(A ∩ B) / Area(A ∪ B)               │ │ │
│  │  │                                                       │ │ │
│  │  │ where:                                                │ │ │
│  │  │  Area(A ∩ B) = intersection_area                     │ │ │
│  │  │  Area(A ∪ B) = area_A + area_B - intersection_area  │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 4: Colored Rectangle Detection (Optional)            │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Purpose: Detect unidentified signals marked by users │ │ │
│  │  │                                                       │ │ │
│  │  │ 1. Define HSV color ranges for green/red/blue        │ │ │
│  │  │    Green: [40-80, 50-255, 50-255]                   │ │ │
│  │  │ 2. Create binary mask using cv2.inRange()            │ │ │
│  │  │ 3. Find contours with cv2.findContours()             │ │ │
│  │  │ 4. Filter by area (min_area threshold)               │ │ │
│  │  │ 5. Extract bounding rectangles                       │ │ │
│  │  │ 6. Label as "Unidentified Signal"                    │ │ │
│  │  │ 7. Remove overlaps with known templates              │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING STAGE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Aggregate detections from all chunks:                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 1: Variant Consolidation                             │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Purpose: Merge detections from same drone variants    │ │ │
│  │  │ Example: "007-Controller", "007-Comns1" → "007"       │ │ │
│  │  │                                                        │ │ │
│  │  │ Algorithm:                                             │ │ │
│  │  │  For each pair of detections (A, B):                  │ │ │
│  │  │    if same_base_class(A, B) AND IoU(A,B) > 0.5:       │ │ │
│  │  │      keep highest_confidence(A, B)                     │ │ │
│  │  │      remove other                                      │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │

│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 2: Horizontal Fusion                                  │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Purpose: Merge adjacent detections across chunk edges │ │ │
│  │  │                                                        │ │ │
│  │  │ Fusion Criteria:                                       │ │ │
│  │  │  1. Same template class                               │ │ │
│  │  │  2. Horizontal overlap ≤ 30%                          │ │ │
│  │  │  3. Vertical alignment ≥ 70%                          │ │ │
│  │  │                                                        │ │ │
│  │  │ Alignment Formula:                                     │ │ │
│  │  │  y_center_diff = |y_center_A - y_center_B|           │ │ │
│  │  │  avg_height = (height_A + height_B) / 2              │ │ │
│  │  │  alignment = 1 - (y_center_diff / avg_height)        │ │ │
│  │  │                                                        │ │ │
│  │  │ Merged Box:                                            │ │ │
│  │  │  x = min(A.x, B.x)                                    │ │ │
│  │  │  width = max(A.x+A.width, B.x+B.width) - x           │ │ │
│  │  │  y = min(A.y, B.y)                                    │ │ │
│  │  │  height = max(A.y+A.height, B.y+B.height) - y        │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  STEP 3: Similarity-Based Classification                   │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Purpose: Classify "Unidentified" signals              │ │ │
│  │  │                                                        │ │ │
│  │  │ For each unidentified signal U:                       │ │ │
│  │  │   For each known template detection K:                │ │ │
│  │  │                                                        │ │ │
│  │  │     height_ratio = |U.height - K.height| /           │ │ │
│  │  │                    max(U.height, K.height)            │ │ │
│  │  │     height_sim = 1 - height_ratio                     │ │ │
│  │  │                                                        │ │ │
│  │  │     y_diff = |U.y_center - K.y_center|               │ │ │
│  │  │     y_sim = 1 - (y_diff / image_height)              │ │ │
│  │  │                                                        │ │ │
│  │  │     if height_sim > 0.85 AND y_sim > 0.80:           │ │ │
│  │  │       classify U as K.template_name                   │ │ │
│  │  │       confidence = min(height_sim, y_sim)             │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              COORDINATE TRANSFORMATION PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Purpose: Transform coordinates for SigMF annotation generation │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Stage 1: Chunk Space → Resized Image Space                │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Input: Detection in chunk coordinates (0-2048, 0-384) │ │ │
│  │  │                                                        │ │ │
│  │  │ Transformation:                                        │ │ │
│  │  │   resized_x = chunk_x + chunk_offset_x               │ │ │
│  │  │   resized_y = chunk_y + chunk_offset_y               │ │ │
│  │  │                                                        │ │ │
│  │  │ Output: Coordinates in resized image (0-31778, 0-384)│ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Stage 2: Resized Image Space → Original Image Space       │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Input: Coordinates in resized image (31778×384)       │ │ │
│  │  │                                                        │ │ │
│  │  │ Scaling Factors:                                       │ │ │
│  │  │   scale_x = original_width / resized_width           │ │ │
│  │  │           = 101708 / 31778 ≈ 3.20                    │ │ │
│  │  │   scale_y = original_height / resized_height          │ │ │
│  │  │           = 1229 / 384 ≈ 3.20                        │ │ │
│  │  │                                                        │ │ │
│  │  │ Transformation:                                        │ │ │
│  │  │   original_x = round(resized_x × scale_x)            │ │ │
│  │  │   original_y = round(resized_y × scale_y)            │ │ │
│  │  │   original_width = round(width × scale_x)            │ │ │
│  │  │   original_height = round(height × scale_y)          │ │ │
│  │  │                                                        │ │ │
│  │  │ Output: Coordinates in original image (0-101708,     │ │ │
│  │  │                                         0-1229)       │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Stage 3: Original Image Space → Time-Frequency Domain     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │ Input: Pixel coordinates in original image            │ │ │
│  │  │                                                        │ │ │
│  │  │ TIME AXIS (X → Sample Index):                         │ │ │
│  │  │                                                        │ │ │
│  │  │  num_cols = (total_samples - fft_size) / step_size + 1│ │ │
│  │  │           = (125M - 2048) / 1024 + 1                 │ │ │
│  │  │           = 122,069 columns                           │ │ │
│  │  │                                                        │ │ │
│  │  │  spec_x = x × (num_cols / image_width)               │ │ │
│  │  │         = x × (122069 / 101708)                      │ │ │
│  │  │         = x × 1.2002                                  │ │ │
│  │  │                                                        │ │ │
│  │  │  start_sample = round(spec_x × step_size)            │ │ │
│  │  │  length_samples = round(spec_width × step_size)      │ │ │
│  │  │                                                        │ │ │
│  │  │ FREQUENCY AXIS (Y → Hz):                              │ │ │
│  │  │                                                        │ │ │
│  │  │  # Flip Y-axis (image Y=0 at top, spectrum at bottom)│ │ │
│  │  │  y_flipped = image_height - y - height               │ │ │
│  │  │                                                        │ │ │
│  │  │  spec_y = y_flipped × (fft_size / image_height)      │ │ │
│  │  │         = y_flipped × (2048 / 1229)                  │ │ │
│  │  │         = y_flipped × 1.666                           │ │ │
│  │  │                                                        │ │ │
│  │  │  bin_fraction = spec_y / fft_size                     │ │ │
│  │  │  f_lo = bin_fraction × sample_rate                   │ │ │
│  │  │        - (sample_rate / 2)                           │ │ │
│  │  │        + center_freq                                  │ │ │
│  │  │                                                        │ │ │
│  │  │  For sample_rate = 20 MHz, center_freq = 2.4 GHz:   │ │ │
│  │  │    f_lo = bin_fraction × 20MHz - 10MHz + 2.4GHz     │ │ │
│  │  │                                                        │ │ │
│  │  │ Output: SigMF Annotation                              │ │ │
│  │  │  {                                                     │ │ │
│  │  │    "core:sample_start": start_sample,                │ │ │
│  │  │    "core:sample_count": length_samples,              │ │ │
│  │  │    "core:freq_lower_edge": f_lo,                     │ │ │
│  │  │    "core:freq_upper_edge": f_hi,                     │ │ │
│  │  │    "core:label": template_name,                      │ │ │
│  │  │    "core:comment": f"Confidence: {confidence:.2f}"   │ │ │
│  │  │  }                                                     │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT GENERATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Visual Results (Streamlit UI)                          │ │
│  │     • Annotated spectrogram images                         │ │
│  │     • Bounding boxes with labels and confidence scores     │ │
│  │     • Detection statistics and timing information          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  2. SigMF Annotation Files (.sigmf-meta)                   │ │
│  │     • Updated JSON metadata with annotations array         │ │
│  │     • Compatible with standard RF analysis tools           │ │
│  │     • Includes timing, frequency, and classification data  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  3. Training Data (Optional)                               │ │
│  │     • Unidentified signals saved as new templates          │ │
│  │     • Automatic template generation for iterative learning │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### 2.2.1 Input Data Specifications

**RF Spectrogram Images**:

- **Format**: PNG (24-bit RGB)
- **Typical Dimensions**: 101,708 × 1,229 pixels
- **Color Space**: JET colormap (blue=low, red=high)
- **Content**: Time-frequency representation of RF signals
  - X-axis: Time progression (left to right)
  - Y-axis: Frequency (bottom to top)
  - Color intensity: Signal power

**Template Library**:

- **Location**: `stored_templates/` directory
- **Organization**: Subdirectories per drone class
  - Example: `stored_templates/007/007-Controller.png`
- **Format**: PNG images with transparency (RGBA) or RGB
- **Typical Size**: 40×40 to 100×100 pixels
- **Naming Convention**: `{class}-{variant}.png`

**SigMF Metadata**:

- **Files**:
  - `.sigmf-meta`: JSON metadata file
  - `.sigmf-data`: Binary I/Q sample data
- **Required Fields**:
  ```json
  {
    "global": {
      "core:datatype": "ci16_le",
      "core:sample_rate": 20000000,
      "core:version": "1.0.0"
    },
    "captures": [
      {
        "core:sample_start": 0,
        "core:frequency": 2400000000
      }
    ],
    "annotations": []
  }
  ```

#### 2.2.2 Preprocessing Pipeline Details

**Step 1: Image Loading**

- **Library**: PIL (Python Imaging Library)
- **Color Space**: Automatic RGB conversion
- **Validation**: Check dimensions and format

**Step 2: Resize Operation**

- **Target Size**: 31,778 × 384 pixels
- **Rationale**:
  - Width: Divisible by 2048 (chunk size) with minimal remainder
  - Height: 384 maintains aspect ratio and fits GPU memory
- **Algorithm**: Lanczos resampling (high-quality downsampling)
- **Aspect Ratio**: Approximately preserved (scale factor ~3.2×)

**Step 3: Image Chunking**

- **Chunk Size**: 2048 × 384 pixels
- **Number of Chunks**: ⌈31778 / 2048⌉ = 16 chunks per image
- **Overlap**: None (contiguous chunks)
- **Last Chunk**: Smaller width (31778 mod 2048 = 1058 pixels)
- **Metadata Tracked**:
  - `chunk_index`: 0 to 15
  - `offset_x`: Position in resized image (0, 2048, 4096, ...)
  - `offset_y`: Always 0 (no vertical chunking)
  - `original_size`: (101708, 1229)
  - `resized_size`: (31778, 384)

#### 2.2.3 Template Matching Details

**Colormap Normalization**:

- **Purpose**: Ensure visual consistency between templates and test images
- **Method**: Apply `cv2.COLORMAP_JET` to both
- **Effect**: Converts grayscale intensity to color spectrum
  - 0 (low) → Blue
  - 127 (medium) → Green
  - 255 (high) → Red

**Matching Algorithm**:

- **Method**: `cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)`
- **Output**: Correlation coefficient map ranging [-1, 1]
- **Threshold**: Typically 0.7-0.9 for reliable matches
- **Peak Detection**: `cv2.minMaxLoc()` finds local maxima

**Multi-Scale Matching** (Optional):

- **Scales Tested**: [0.8, 0.9, 1.0, 1.1, 1.2]
- **Purpose**: Account for size variations in signals
- **Implementation**: Resize template, run matching, rescale coordinates

#### 2.2.4 Post-Processing Details

**Variant Consolidation Logic**:

```python
def same_base_class(label_a, label_b):
    # Extract base class (e.g., "007" from "007-Controller")
    base_a = label_a.split('-')[0]
    base_b = label_b.split('-')[0]
    return base_a == base_b

def consolidate_variants(detections):
    for i, det_a in enumerate(detections):
        for j, det_b in enumerate(detections[i+1:], start=i+1):
            if same_base_class(det_a['label'], det_b['label']):
                iou = calculate_iou(det_a, det_b)
                if iou > 0.5:
                    # Keep higher confidence detection
                    if det_a['confidence'] > det_b['confidence']:
                        remove_detection(det_b)
                    else:
                        remove_detection(det_a)
```

**Horizontal Fusion Detailed Algorithm**:

```python
def fuse_horizontally_adjacent(detections):
    # Sort by X-coordinate
    sorted_dets = sorted(detections, key=lambda d: d['x'])

    fused = []
    i = 0
    while i < len(sorted_dets):
        current = sorted_dets[i]
        j = i + 1

        # Look for adjacent detection
        while j < len(sorted_dets):
            next_det = sorted_dets[j]

            # Check same class
            if current['label'] != next_det['label']:
                break

            # Check horizontal adjacency
            gap = next_det['x'] - (current['x'] + current['width'])
            if gap > 100:  # Too far apart
                break

            # Check vertical alignment
            y_center_curr = current['y'] + current['height'] / 2
            y_center_next = next_det['y'] + next_det['height'] / 2
            avg_height = (current['height'] + next_det['height']) / 2
            alignment = 1 - abs(y_center_curr - y_center_next) / avg_height

            if alignment < 0.7:  # Poor vertical alignment
                break

            # Merge the two detections
            merged = {
                'x': min(current['x'], next_det['x']),
                'y': min(current['y'], next_det['y']),
                'width': max(current['x'] + current['width'],
                            next_det['x'] + next_det['width']) -
                        min(current['x'], next_det['x']),
                'height': max(current['y'] + current['height'],
                             next_det['y'] + next_det['height']) -
                         min(current['y'], next_det['y']),
                'confidence': max(current['confidence'],
                                 next_det['confidence']),
                'label': current['label']
            }
            current = merged
            j += 1

        fused.append(current)
        i = j

    return fused
```

### 2.3 Data Flow Summary

```
Input: Spectrogram (101708×1229)
   ↓
Resize: (31778×384)
   ↓
Chunk: 16 × (2048×384)
   ↓
Template Match: M templates × 16 chunks → K detections/chunk
   ↓
NMS: K → K' detections (K' < K)
   ↓
Aggregate: 16 × K' → D total detections
   ↓
Post-Process: D → D' filtered detections
   ↓
Transform: Chunk space → Original space (101708×1229)
   ↓
SigMF: Pixel coordinates → (samples, Hz)
   ↓
Output: Annotations + Visualizations
```

---

## 3. Metric Definitions and Formulas

### 3.1 Detection Quality Metrics

#### 3.1.1 Normalized Cross-Correlation (NCC) Score

**Purpose**: Measure similarity between template and image region

**Formula**:

```
NCC(I, T, x, y) = Σᵢ Σⱼ [(I(x+i,y+j) - Ī) × (T(i,j) - T̄)] / (σᵢ × σₜ × n)
```

**Where**:

- `I(x+i, y+j)`: Pixel intensity in image at position (x+i, y+j)
- `T(i,j)`: Pixel intensity in template at position (i,j)
- `Ī`: Mean intensity of image region
- `T̄`: Mean intensity of template
- `σᵢ`: Standard deviation of image region
- `σₜ`: Standard deviation of template
- `n`: Number of pixels in template

**Range**: [-1, 1]

- **1.0**: Perfect positive correlation (exact match)
- **0.0**: No correlation
- **-1.0**: Perfect negative correlation (inverted match)

**Interpretation**:

- **NCC > 0.9**: Excellent match (high confidence)
- **0.7 < NCC < 0.9**: Good match (medium confidence)
- **NCC < 0.7**: Poor match (typically rejected)

**Usage in System**:

- Primary confidence score for template matches
- Threshold parameter: `min_confidence` (default: 0.7)

#### 3.1.2 Intersection over Union (IoU)

**Purpose**: Measure overlap between two bounding boxes

**Formula**:

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
```

**Detailed Calculation**:

```
# Intersection coordinates
x_left = max(A.x1, B.x1)
x_right = min(A.x2, B.x2)
y_top = max(A.y1, B.y1)
y_bottom = min(A.y2, B.y2)

# Intersection area
if x_right > x_left and y_bottom > y_top:
    intersection = (x_right - x_left) × (y_bottom - y_top)
else:
    intersection = 0

# Union area
area_A = (A.x2 - A.x1) × (A.y2 - A.y1)
area_B = (B.x2 - B.x1) × (B.y2 - B.y1)
union = area_A + area_B - intersection

# IoU
IoU = intersection / union
```

**Range**: [0, 1]

- **IoU = 1**: Perfect overlap (identical boxes)
- **IoU > 0.5**: Significant overlap
- **IoU < 0.3**: Minimal overlap
- **IoU = 0**: No overlap

**Usage in System**:

- NMS threshold: IoU > 0.3 triggers suppression
- Consolidation threshold: IoU > 0.5 triggers merging
- Fusion condition: IoU < 0.3 (low overlap allowed)

#### 3.1.3 Confidence Score

**Purpose**: Overall reliability measure for a detection

**Formula** (Template Matches):

```
confidence = NCC_score
```

**Formula** (Similarity Classification):

```
height_similarity = 1 - |height_A - height_B| / max(height_A, height_B)
y_similarity = 1 - |y_center_A - y_center_B| / image_height
confidence = min(height_similarity, y_similarity)
```

**Range**: [0, 1]

- **confidence > 0.9**: Very high confidence
- **0.7 < confidence < 0.9**: High confidence
- **0.5 < confidence < 0.7**: Medium confidence
- **confidence < 0.5**: Low confidence (may be filtered)

**Usage in System**:

- Sorting criterion for NMS
- Display threshold in UI
- Training mode: Only save high-confidence templates (>0.7)

### 3.2 Coordinate Transformation Metrics

#### 3.2.1 Scaling Factor

**Purpose**: Convert between coordinate spaces

**Formula** (Resized → Original):

```
scale_x = original_width / resized_width
scale_y = original_height / resized_height
```

**Example Values**:

```
scale_x = 101708 / 31778 = 3.200645
scale_y = 1229 / 384 = 3.200521
```

**Usage**:

```
original_x = round(resized_x × scale_x)
original_y = round(resized_y × scale_y)
```

**Error Analysis**:

- Rounding error: ±0.5 pixels in original space
- Relative error: ±0.5 / 101708 ≈ 0.0005% (negligible)

#### 3.2.2 Spectrogram Column Mapping

**Purpose**: Map pixel X-coordinate to spectrogram column index

**Formula**:

```
num_columns = (total_samples - fft_size) / step_size + 1
spec_x = x × (num_columns / image_width)
```

**Example**:

```
total_samples = 125,000,000
fft_size = 2048
step_size = 1024

num_columns = (125,000,000 - 2,048) / 1,024 + 1 = 122,069

For image_width = 101,708:
  scale_factor = 122,069 / 101,708 = 1.200165

For pixel x = 1000:
  spec_x = 1000 × 1.200165 = 1200.165 → column 1200
```

**Interpretation**:

- Each pixel represents ~1.2 spectrogram columns
- Slight upscaling due to FFT overlap (step_size < fft_size)

#### 3.2.3 Frequency Mapping

**Purpose**: Map pixel Y-coordinate to frequency in Hz

**Formula**:

```
# Step 1: Flip Y-axis
y_flipped = image_height - y - height

# Step 2: Map to FFT bin
spec_y = y_flipped × (fft_size / image_height)

# Step 3: Convert to frequency
bin_fraction = spec_y / fft_size
f_Hz = bin_fraction × sample_rate - (sample_rate / 2) + center_freq
```

**Example**:

```
image_height = 1229
fft_size = 2048
sample_rate = 20,000,000 Hz (20 MHz)
center_freq = 2,400,000,000 Hz (2.4 GHz)

For pixel y = 100, height = 50:
  y_flipped = 1229 - 100 - 50 = 1079
  spec_y = 1079 × (2048 / 1229) = 1797.6
  bin_fraction = 1797.6 / 2048 = 0.8775
  f_Hz = 0.8775 × 20M - 10M + 2.4G
       = 17.55M - 10M + 2.4G
       = 2,407,550,000 Hz
       = 2.40755 GHz
```

**Frequency Resolution**:

```
freq_resolution = sample_rate / fft_size
                = 20,000,000 / 2048
                = 9,765.625 Hz per bin
                ≈ 9.77 kHz per bin
```

### 3.3 Performance Metrics

#### 3.3.1 Processing Time

**Measurement**: Wall-clock time for each pipeline stage

**Breakdown**:

```
Total_Time = T_preprocess + T_detection + T_postprocess + T_transform + T_sigmf
```

**Typical Values** (single image, M=2 templates):

- `T_preprocess` ≈ 200 ms (loading + resize + chunk)
- `T_detection` ≈ 800 ms (16 chunks × 50 ms/chunk)
- `T_postprocess` ≈ 50 ms (NMS + consolidation + fusion)
- `T_transform` ≈ 5 ms (coordinate mapping)
- `T_sigmf` ≈ 50 ms (file I/O + JSON update)
- **`Total_Time` ≈ 1.1 seconds**

**Formula for Multiple Images**:

```
Total_Time = N × (T_preprocess + T_detection + T_postprocess + T_transform + T_sigmf)
```

**Parallel Processing Speedup**:

```
Speedup = T_sequential / T_parallel
        ≈ N × T_per_image / (T_per_image + (N-1) × T_per_image / num_cores)
```

For `num_cores = 4` and `N = 16` images:

```
Speedup ≈ 16 / (1 + 15/4) = 16 / 4.75 ≈ 3.37×
```

#### 3.3.2 Detection Density

**Purpose**: Measure number of detections per unit area

**Formula**:

```
density = num_detections / (image_width × image_height)
```

**Units**: Detections per pixel²

**Example**:

```
num_detections = 150
image_area = 101708 × 1229 = 125,000,332 pixels

density = 150 / 125,000,332
        = 1.2 × 10⁻⁶ detections/pixel²
        = 1.2 detections per million pixels
```

**Interpretation**:

- **High density**: >100 detections per image (complex signal environment)
- **Medium density**: 10-100 detections per image (typical)
- **Low density**: <10 detections per image (sparse signals)

#### 3.3.3 False Positive Rate (Estimated)

**Purpose**: Estimate spurious detections

**Formula** (with ground truth):

```
FPR = False_Positives / (False_Positives + True_Negatives)
```

**Practical Estimation** (without ground truth):

```
estimated_FPR = low_confidence_count / total_detections

where low_confidence_count = detections with confidence < 0.75
```

**Usage**:

- Adjust `min_confidence` threshold to reduce FPR
- Higher threshold → Lower FPR, but may miss true positives

### 3.4 Signal Quality Metrics

#### 3.4.1 Signal Duration

**Purpose**: Temporal extent of detected signal

**Formula**:

```
duration_seconds = length_samples / sample_rate
```

**Example**:

```
length_samples = 388,362
sample_rate = 20,000,000

duration = 388,362 / 20,000,000 = 0.01942 seconds ≈ 19.4 ms
```

**Typical Values**:

- **Short pulse**: <10 ms
- **Medium burst**: 10-50 ms
- **Long transmission**: >50 ms

#### 3.4.2 Bandwidth

**Purpose**: Frequency span of detected signal

**Formula**:

```
bandwidth_Hz = f_upper - f_lower
```

**Example**:

```
f_lower = 2,400,500,000 Hz
f_upper = 2,401,200,000 Hz

bandwidth = 2,401,200,000 - 2,400,500,000 = 700,000 Hz = 700 kHz
```

**Typical Values**:

- **Narrowband**: <100 kHz
- **Wideband**: 100 kHz - 1 MHz
- **Ultra-wideband**: >1 MHz

#### 3.4.3 Duty Cycle

**Purpose**: Fraction of time signal is active

**Formula**:

```
duty_cycle = Σ(signal_durations) / total_recording_duration
```

**Example**:

```
Total recording: 6.25 seconds
Signal durations: [19.4 ms, 15.7 ms, 22.1 ms, ...] (20 detections)
Total signal time: 450 ms

duty_cycle = 0.450 / 6.25 = 0.072 = 7.2%
```

**Interpretation**:

- **Low duty cycle** (<10%): Intermittent transmissions
- **High duty cycle** (>50%): Continuous or frequent transmissions

### 3.5 System Performance Indicators

#### 3.5.1 Throughput

**Formula**:

```
throughput = num_images_processed / total_time

units: images/second or images/hour
```

**Example**:

```
100 images processed in 180 seconds

throughput = 100 / 180 = 0.556 images/second
           = 33.3 images/minute
           = 2000 images/hour
```

#### 3.5.2 Memory Efficiency

**Formula**:

```
memory_per_image = peak_memory / num_images_in_memory
```

**Example**:

```
Peak memory: 800 MB
Batch size: 8 images

memory_per_image = 800 / 8 = 100 MB/image
```

**Optimization Strategy**:

- Process images sequentially to minimize memory
- Release chunk data after detection
- Stream results to disk rather than accumulating in memory

#### 3.5.3 Annotation Accuracy

**Formula** (with ground truth):

```
accuracy = correct_annotations / total_ground_truth_annotations
```

**Distance Metric** (box center error):

```
distance_error = sqrt((x_pred - x_true)² + (y_pred - y_true)²)
```

**Acceptable Error Threshold**: <5% of image dimension

```
For 101708×1229 image:
  acceptable_x_error < 0.05 × 101708 = 5085 pixels
  acceptable_y_error < 0.05 × 1229 = 61 pixels
```
