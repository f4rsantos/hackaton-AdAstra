import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import cv2
from PIL import Image
import io

def load_audio(file_path, sr=22050, duration=None):
    """
    Load audio file and return time series and sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Duration to load (None for full file)
    
    Returns:
        y: Audio time series
        sr: Sample rate
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    return y, sr

def generate_spectrogram(y, sr, n_fft=2048, hop_length=512, window='hann'):
    """
    Generate spectrogram from audio time series.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        window: Window function
    
    Returns:
        S: Magnitude spectrogram
        freqs: Frequency bins
        times: Time bins
    """
    # Compute Short-Time Fourier Transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    S = np.abs(D)
    
    # Convert to dB scale
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Generate frequency and time axes
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    
    return S_db, freqs, times

def generate_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Generate mel spectrogram from audio time series.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    
    Returns:
        S_mel: Mel spectrogram in dB
        freqs: Mel frequency bins
        times: Time bins
    """
    # Compute mel spectrogram
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    
    # Convert to dB scale
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    
    # Generate frequency and time axes
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    times = librosa.frames_to_time(np.arange(S_mel.shape[1]), sr=sr, hop_length=hop_length)
    
    return S_mel_db, freqs, times

def spectrogram_to_image(S_db, cmap='viridis', figsize=(12, 8), dpi=100):
    """
    Convert spectrogram to image format suitable for machine learning processing.
    
    Args:
        S_db: Spectrogram in dB scale
        cmap: Colormap for visualization
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
    
    Returns:
        img_array: Image as numpy array (RGB)
        fig: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot spectrogram
    img = ax.imshow(
        S_db, 
        origin='lower', 
        aspect='auto', 
        cmap=cmap,
        interpolation='nearest'
    )
    
    # Remove axes for clean image
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    
    # Convert to image array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    
    # Load as PIL image and convert to numpy array
    pil_img = Image.open(buf)
    img_array = np.array(pil_img.convert('RGB'))
    
    buf.close()
    
    return img_array, fig

def enhance_spectrogram(S_db, method='histogram_eq'):
    """
    Enhance spectrogram for better feature detection.
    
    Args:
        S_db: Spectrogram in dB scale
        method: Enhancement method ('histogram_eq', 'contrast_stretch', 'gamma')
    
    Returns:
        S_enhanced: Enhanced spectrogram
    """
    # Normalize to 0-255 range
    S_norm = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8)
    
    if method == 'histogram_eq':
        # Histogram equalization
        S_enhanced = cv2.equalizeHist(S_norm)
    elif method == 'contrast_stretch':
        # Contrast stretching
        p2, p98 = np.percentile(S_norm, (2, 98))
        S_enhanced = cv2.convertScaleAbs(S_norm, alpha=255/(p98-p2), beta=-p2*255/(p98-p2))
    elif method == 'gamma':
        # Gamma correction
        gamma = 0.7
        lookupTable = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        S_enhanced = cv2.LUT(S_norm, lookupTable)
    else:
        S_enhanced = S_norm
    
    return S_enhanced

def extract_audio_features(y, sr):
    """
    Extract various audio features for analysis.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        features: Dictionary of extracted features
    """
    features = {}
    
    # Basic statistics
    features['duration'] = len(y) / sr
    features['rms_energy'] = np.sqrt(np.mean(y**2))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    return features

def enhance_yellow_colors(image, intensity=1.5, preserve_brightness=True):
    """
    Enhance yellow colors in an image to improve contrast and visibility.
    
    This function specifically boosts yellow hues while optionally preserving
    overall brightness, which can help with better detection of spectrograms
    or other images where yellow features are important.
    
    Args:
        image: Input image as numpy array (RGB or BGR)
        intensity: Enhancement intensity factor (1.0 = no change, >1.0 = more yellow)
        preserve_brightness: Whether to preserve overall brightness after enhancement
    
    Returns:
        enhanced_image: Image with enhanced yellow colors
    """
    # Convert image to float32 for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Convert to HSV color space for better color manipulation
    if len(img_float.shape) == 3:
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    else:
        # If grayscale, convert to RGB first
        img_float = cv2.cvtColor(img_float, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    
    h, s, v = cv2.split(hsv)
    
    # Define yellow hue range (in OpenCV HSV: 0-179 for hue)
    # Yellow is around 30 degrees, which maps to ~15 in OpenCV HSV
    yellow_hue_center = 30 / 2  # ~15 in OpenCV scale
    yellow_hue_range = 25 / 2   # ~12.5 in OpenCV scale
    
    # Create mask for yellow regions
    # Handle hue wrapping around 0/180
    yellow_mask = np.logical_or(
        np.abs(h * 180 - yellow_hue_center) <= yellow_hue_range,
        np.abs((h * 180 + 180) - yellow_hue_center) <= yellow_hue_range
    )
    
    # Also check for yellow-green and orange-yellow regions
    yellow_green_center = 45 / 2  # ~22.5
    orange_yellow_center = 15 / 2  # ~7.5
    
    yellow_extended_mask = np.logical_or(
        yellow_mask,
        np.logical_or(
            np.abs(h * 180 - yellow_green_center) <= yellow_hue_range,
            np.abs(h * 180 - orange_yellow_center) <= yellow_hue_range
        )
    )
    
    # Enhance saturation and value for yellow regions
    s_enhanced = s.copy()
    v_enhanced = v.copy()
    
    # Boost saturation for yellow regions
    s_enhanced[yellow_extended_mask] = np.clip(s[yellow_extended_mask] * intensity, 0, 1)
    
    # Slightly boost brightness for yellow regions
    v_enhanced[yellow_extended_mask] = np.clip(v[yellow_extended_mask] * np.sqrt(intensity), 0, 1)
    
    # Recombine HSV channels
    hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
    
    # Convert back to RGB
    rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    # Preserve brightness if requested
    if preserve_brightness:
        original_brightness = np.mean(cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY))
        enhanced_brightness = np.mean(cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2GRAY))
        
        if enhanced_brightness > 0:
            brightness_factor = original_brightness / enhanced_brightness
            rgb_enhanced = np.clip(rgb_enhanced * brightness_factor, 0, 1)
    
    # Convert back to uint8
    enhanced_image = (rgb_enhanced * 255).astype(np.uint8)
    
    return enhanced_image

def apply_yellow_filter_to_spectrogram(S_db, intensity=1.3):
    """
    Apply yellow enhancement to a spectrogram after converting it to image format.
    
    Args:
        S_db: Spectrogram in dB scale
        intensity: Yellow enhancement intensity
    
    Returns:
        enhanced_spectrogram: Yellow-enhanced spectrogram as image array
    """
    # Convert spectrogram to image first
    img_array, fig = spectrogram_to_image(S_db, cmap='viridis')
    plt.close(fig)  # Close the figure to free memory
    
    # Apply yellow enhancement
    enhanced_img = enhance_yellow_colors(img_array, intensity=intensity)
    
    return enhanced_img

def preprocess_for_ml(spectrogram_image, target_size=(640, 640)):
    """
    Preprocess spectrogram image for machine learning input.
    
    Args:
        spectrogram_image: Spectrogram as numpy array (RGB)
        target_size: Target size for ML model input
    
    Returns:
        processed_image: Preprocessed image ready for ML processing
    """
    # Resize image
    h, w = spectrogram_image.shape[:2]
    processed_image = cv2.resize(spectrogram_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to [0, 1]
    processed_image = processed_image.astype(np.float32) / 255.0
    
    return processed_image