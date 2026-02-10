"""
Signal preprocessing module for EMG data
Includes filtering, artifact removal, and normalization
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class EMGPreprocessor:
    """Comprehensive EMG signal preprocessing."""
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.fs = config['signal_processing']['sampling_rate']
        
    def preprocess(self, emg: np.ndarray, remove_artifacts: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        emg : np.ndarray
            Raw EMG signals [n_samples, n_channels]
        remove_artifacts : bool
            Whether to apply artifact removal
        
        Returns
        -------
        processed : np.ndarray
            Preprocessed EMG signals
        """
        logger.info("Starting EMG preprocessing")
        
        # Step 1: Bandpass filtering
        filtered = self.bandpass_filter(emg)
        
        # Step 2: Notch filter (if enabled)
        if self.config['signal_processing']['notch_filter']['enabled']:
            filtered = self.notch_filter(filtered)
        
        # Step 3: Artifact removal (optional)
        if remove_artifacts:
            filtered = self.remove_artifacts(filtered)
        
        # Step 4: Rectification
        rectified = np.abs(filtered)
        
        # Step 5: Envelope extraction
        envelope = self.extract_envelope(rectified)
        
        # Step 6: Normalization
        normalized = self.normalize(envelope)
        
        logger.info("Preprocessing complete")
        return normalized
    
    def bandpass_filter(self, emg: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to remove DC offset and high-frequency noise.
        
        Parameters
        ----------
        emg : np.ndarray
            Input EMG signals
        
        Returns
        -------
        filtered : np.ndarray
            Bandpass filtered signals
        """
        bp_config = self.config['signal_processing']['bandpass_filter']
        
        # Design Butterworth bandpass filter
        nyquist = self.fs / 2
        low = bp_config['low_cutoff'] / nyquist
        high = bp_config['high_cutoff'] / nyquist
        
        b, a = butter(bp_config['order'], [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(emg)
        for ch in range(emg.shape[1]):
            filtered[:, ch] = filtfilt(b, a, emg[:, ch])
        
        logger.debug(f"Applied bandpass filter: {bp_config['low_cutoff']}-{bp_config['high_cutoff']} Hz")
        return filtered
    
    def notch_filter(self, emg: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Parameters
        ----------
        emg : np.ndarray
            Input EMG signals
        
        Returns
        -------
        filtered : np.ndarray
            Notch filtered signals
        """
        notch_config = self.config['signal_processing']['notch_filter']
        freq = notch_config['frequency']
        Q = notch_config['quality_factor']
        
        # Design notch filter
        b, a = iirnotch(freq, Q, self.fs)
        
        # Apply filter
        filtered = np.zeros_like(emg)
        for ch in range(emg.shape[1]):
            filtered[:, ch] = filtfilt(b, a, emg[:, ch])
        
        logger.debug(f"Applied notch filter at {freq} Hz")
        return filtered
    
    def extract_envelope(self, rectified: np.ndarray) -> np.ndarray:
        """
        Extract linear envelope using low-pass filtering.
        
        Parameters
        ----------
        rectified : np.ndarray
            Rectified EMG signals
        
        Returns
        -------
        envelope : np.ndarray
            EMG envelopes
        """
        env_config = self.config['signal_processing']['envelope_filter']
        
        # Design low-pass filter
        nyquist = self.fs / 2
        cutoff = env_config['cutoff'] / nyquist
        
        b, a = butter(env_config['order'], cutoff, btype='low')
        
        # Apply filter
        envelope = np.zeros_like(rectified)
        for ch in range(rectified.shape[1]):
            envelope[:, ch] = filtfilt(b, a, rectified[:, ch])
        
        return envelope
    
    def normalize(self, emg: np.ndarray) -> np.ndarray:
        """
        Normalize EMG signals.
        
        Parameters
        ----------
        emg : np.ndarray
            Input EMG signals
        
        Returns
        -------
        normalized : np.ndarray
            Normalized signals
        """
        method = self.config['signal_processing']['normalization']
        
        normalized = np.zeros_like(emg)
        
        for ch in range(emg.shape[1]):
            if method == 'max':
                # Normalize by maximum value
                max_val = np.max(np.abs(emg[:, ch]))
                normalized[:, ch] = emg[:, ch] / (max_val + 1e-10)
                
            elif method == 'rms':
                # Normalize by RMS value
                rms = np.sqrt(np.mean(emg[:, ch] ** 2))
                normalized[:, ch] = emg[:, ch] / (rms + 1e-10)
                
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(emg[:, ch])
                std = np.std(emg[:, ch])
                normalized[:, ch] = (emg[:, ch] - mean) / (std + 1e-10)
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        logger.debug(f"Applied {method} normalization")
        return normalized
    
    def remove_artifacts(self, emg: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """
        Remove motion artifacts using threshold-based detection.
        
        Parameters
        ----------
        emg : np.ndarray
            Input EMG signals
        threshold : float
            Threshold in standard deviations
        
        Returns
        -------
        cleaned : np.ndarray
            Artifact-free signals
        """
        cleaned = emg.copy()
        
        for ch in range(emg.shape[1]):
            # Compute signal statistics
            mean = np.mean(emg[:, ch])
            std = np.std(emg[:, ch])
            
            # Detect artifacts
            artifacts = np.abs(emg[:, ch] - mean) > (threshold * std)
            
            if np.any(artifacts):
                # Interpolate artifact regions
                artifact_indices = np.where(artifacts)[0]
                valid_indices = np.where(~artifacts)[0]
                
                cleaned[artifact_indices, ch] = np.interp(
                    artifact_indices,
                    valid_indices,
                    emg[valid_indices, ch]
                )
                
                logger.debug(f"Removed {np.sum(artifacts)} artifact samples from channel {ch}")
        
        return cleaned
    
    def compute_quality_metrics(self, emg: np.ndarray) -> dict:
        """
        Compute signal quality metrics.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals
        
        Returns
        -------
        metrics : dict
            Quality metrics for each channel
        """
        metrics = {}
        
        for ch in range(emg.shape[1]):
            # Signal-to-noise ratio
            f, psd = welch(emg[:, ch], fs=self.fs, nperseg=1024)
            
            # Power in EMG band (20-450 Hz) vs. total power
            emg_band = (f >= 20) & (f <= 450)
            signal_power = np.sum(psd[emg_band])
            total_power = np.sum(psd)
            snr = 10 * np.log10(signal_power / (total_power - signal_power + 1e-10))
            
            # RMS
            rms = np.sqrt(np.mean(emg[:, ch] ** 2))
            
            # Peak-to-peak amplitude
            peak_to_peak = np.ptp(emg[:, ch])
            
            metrics[f'channel_{ch}'] = {
                'snr_db': snr,
                'rms': rms,
                'peak_to_peak': peak_to_peak
            }
        
        return metrics