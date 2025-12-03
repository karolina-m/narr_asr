"""
Audio Quality Metrics Pipeline for ASR Performance Analysis

This script extracts multiple audio quality metrics from MP3 files
to help predict ASR (Automatic Speech Recognition) performance.

Requirements:
    pip install librosa soundfile numpy scipy pydub

Usage:
    python audio_quality_metrics.py input.mp3
    # Or use as a module:
    from audio_quality_metrics import extract_audio_quality_metrics
    metrics = extract_audio_quality_metrics('audio.mp3')
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.stats import kurtosis, skew
import re
import warnings
warnings.filterwarnings('ignore')


def extract_key_from_filename(filename):
    
    name_without_ext = Path(filename).stem
    
    # assume pattern of 
    match = re.match(r'^(.+?)_clean(?:_s)?$', name_without_ext)
    
    if match:
        return match.group(1)
    else:
        # 
        return name_without_ext


def extract_audio_quality_metrics(audio_path, sr=16000):
    """
    Extract comprehensive audio quality metrics from an audio file.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the audio file (supports mp3, wav, etc.)
    sr : int
        Target sampling rate (default: 16000 Hz, common for speech)
    
    Returns:
    --------
    dict : Dictionary containing all quality metrics
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    metrics = {}
    
    # 1. SIGNAL-TO-NOISE RATIO (SNR)
    metrics['snr_db'] = estimate_snr(y, sr)
    
    # 2. RMS ENERGY (Overall loudness)
    rms = librosa.feature.rms(y=y)[0]
    metrics['rms_mean'] = float(np.mean(rms))
    metrics['rms_std'] = float(np.std(rms))
    metrics['rms_db'] = float(20 * np.log10(np.mean(rms) + 1e-10))
    
    # 3. CLIPPING DETECTION
    metrics['clipping_rate'] = detect_clipping(y)
    
    # 4. DYNAMIC RANGE
    metrics['dynamic_range_db'] = calculate_dynamic_range(y)
    
    # 5. ZERO-CROSSING RATE (indicates noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    metrics['zcr_mean'] = float(np.mean(zcr))
    
    # 6. SPECTRAL FEATURES (focus on speech frequencies)
    spectral_metrics = extract_spectral_features(y, sr)
    metrics.update(spectral_metrics)
    
    # 7. SPEECH-RELEVANT FREQUENCY ENERGY
    metrics['speech_band_energy'] = speech_frequency_energy(y, sr)
    
    # 8. SILENCE/NOISE RATIO
    metrics['silence_ratio'] = estimate_silence_ratio(y, sr)
    
    # 9. SPECTRAL FLUX (temporal variation in spectrum)
    metrics['spectral_flux'] = calculate_spectral_flux(y)
    
    # 10. FILE METADATA
    metrics['duration_seconds'] = float(len(y) / sr)
    metrics['sample_rate'] = int(sr)
    
    return metrics


def estimate_snr(y, sr, frame_length=2048, hop_length=512):
    """
    Estimate Signal-to-Noise Ratio using a simple energy-based method.
    Assumes lower energy frames are noise.
    """
    # Calculate frame energies
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    
    # Use lower 20th percentile as noise estimate
    noise_threshold = np.percentile(energy, 20)
    noise_frames = energy[energy <= noise_threshold]
    signal_frames = energy[energy > noise_threshold]
    
    if len(noise_frames) == 0 or len(signal_frames) == 0:
        return 0.0
    
    noise_power = np.mean(noise_frames)
    signal_power = np.mean(signal_frames)
    
    if noise_power == 0:
        return 100.0
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def detect_clipping(y, threshold=0.99):
    """
    Detect clipping (distortion from over-amplification).
    Returns the proportion of samples near maximum amplitude.
    """
    clipped_samples = np.sum(np.abs(y) >= threshold)
    clipping_rate = clipped_samples / len(y)
    return float(clipping_rate)


def calculate_dynamic_range(y):
    """
    Calculate dynamic range (difference between loudest and quietest parts).
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Use percentiles to avoid outliers
    loud = np.percentile(rms_db, 95)
    quiet = np.percentile(rms_db, 5)
    
    return float(loud - quiet)


def extract_spectral_features(y, sr):
    """
    Extract spectral features relevant to speech clarity.
    """
    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Spectral flatness (how noise-like vs tone-like)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    return {
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_centroid_std': float(np.std(spectral_centroid)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'spectral_flatness_mean': float(np.mean(spectral_flatness)),
        'spectral_flatness_std': float(np.std(spectral_flatness))
    }


def speech_frequency_energy(y, sr):
    """
    Calculate energy in speech-relevant frequency band (300-3400 Hz).
    Higher values indicate better speech clarity.
    """
    # Create bandpass filter for speech frequencies
    sos = signal.butter(4, [300, 3400], btype='band', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    
    speech_energy = np.mean(y_filtered**2)
    total_energy = np.mean(y**2)
    
    if total_energy == 0:
        return 0.0
    
    # Ratio of speech-band energy to total energy
    ratio = speech_energy / total_energy
    return float(ratio)


def estimate_silence_ratio(y, sr, top_db=30):
    """
    Estimate the proportion of the recording that is silence/quiet.
    """
    # Split into frames and determine which are silent
    intervals = librosa.effects.split(y, top_db=top_db)
    
    if len(intervals) == 0:
        return 1.0  # All silence
    
    speech_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)
    
    silence_ratio = 1 - (speech_samples / total_samples)
    return float(silence_ratio)


def calculate_spectral_flux(y):
    """
    Calculate spectral flux (rate of change in spectrum).
    High flux can indicate noisy or unstable audio.
    """
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    return float(np.mean(flux))


def batch_process_directory(directory, output_csv='audio_quality_metrics.csv', return_dataframe=True):
    """
    Process all audio files in a directory and save results to CSV.
    
    Parameters:
    -----------
    directory : str or Path
        Path to directory containing audio files
    output_csv : str or None
        Output CSV file path. If None, doesn't save to CSV.
    return_dataframe : bool
        Whether to return the DataFrame (default: True)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with metrics and 'key' column
    """
    import pandas as pd
    from tqdm import tqdm
    
    directory = Path(directory)
    audio_files = list(directory.glob('*.mp3')) + list(directory.glob('*.wav'))
    
    results = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            metrics = extract_audio_quality_metrics(audio_file)
            metrics['filename'] = audio_file.name
            metrics['filepath'] = str(audio_file)
            # Extract key from filename
            metrics['key'] = extract_key_from_filename(audio_file.name)
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to put key and filename first
    cols = ['key', 'filename'] + [col for col in df.columns if col not in ['key', 'filename']]
    df = df[cols]
    
    # Save to CSV if output path is provided
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"\nProcessed {len(results)} files. Results saved to {output_csv}")
    else:
        print(f"\nProcessed {len(results)} files.")
    
    if return_dataframe:
        return df


def get_audio_metrics_dataframe(directory, save_csv=False, output_csv='audio_quality_metrics.csv'):
    """
    Convenience function to get audio metrics as a DataFrame.
    Designed for easy integration into other scripts.
    
    Parameters:
    -----------
    directory : str or Path
        Path to directory containing audio files
    save_csv : bool
        Whether to save results to CSV (default: False)
    output_csv : str
        Output CSV filename if save_csv is True
    
    Returns:
    --------
    pd.DataFrame : DataFrame with 'key' column and all metrics
    
    Example:
    --------
    >>> from audio_quality_metrics import get_audio_metrics_dataframe
    >>> df = get_audio_metrics_dataframe('path/to/audio/files')
    >>> # Now use df in your analysis
    >>> merged = pd.merge(asr_results, df, on='key')
    """
    output_path = output_csv if save_csv else None
    return batch_process_directory(directory, output_csv=output_path, return_dataframe=True)


def print_metrics_summary(metrics):
    """
    Print a formatted summary of audio quality metrics.
    """
    print("\n" + "="*60)
    print("AUDIO QUALITY METRICS SUMMARY")
    print("="*60)
    
    print(f"\n📊 OVERALL QUALITY:")
    print(f"  SNR:                    {metrics['snr_db']:.2f} dB")
    print(f"  RMS Level:              {metrics['rms_db']:.2f} dB")
    print(f"  Dynamic Range:          {metrics['dynamic_range_db']:.2f} dB")
    
    print(f"\n🔊 SIGNAL CHARACTERISTICS:")
    print(f"  Speech Band Energy:     {metrics['speech_band_energy']:.4f}")
    print(f"  Clipping Rate:          {metrics['clipping_rate']*100:.2f}%")
    print(f"  Silence Ratio:          {metrics['silence_ratio']*100:.2f}%")
    
    print(f"\n🎵 SPECTRAL FEATURES:")
    print(f"  Spectral Centroid:      {metrics['spectral_centroid_mean']:.1f} Hz")
    print(f"  Spectral Flatness:      {metrics['spectral_flatness_mean']:.4f}")
    print(f"  Spectral Flux:          {metrics['spectral_flux']:.2f}")
    
    print(f"\n⏱️  FILE INFO:")
    print(f"  Duration:               {metrics['duration_seconds']:.2f} seconds")
    print(f"  Sample Rate:            {metrics['sample_rate']} Hz")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process single file:   python audio_quality_metrics.py input.mp3")
        print("  Process directory:     python audio_quality_metrics.py /path/to/audio/files/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if input_path.is_file():
        # Process single file
        print(f"Processing: {input_path}")
        metrics = extract_audio_quality_metrics(input_path)
        print_metrics_summary(metrics)
        print(f"Extracted key: {extract_key_from_filename(input_path.name)}")
        
    elif input_path.is_dir():
        # Process directory
        output_csv = 'audio_quality_metrics.csv'
        if len(sys.argv) > 2:
            output_csv = sys.argv[2]
        
        df = batch_process_directory(input_path, output_csv)
        print(f"\nSummary statistics:")
        print(df[['key', 'snr_db', 'rms_db', 'clipping_rate', 'speech_band_energy']].describe())
        print(f"\nUnique keys found: {df['key'].nunique()}")
        print(df[['key', 'filename']].head(10))
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)