"""
Feature extraction module for EMG network analysis
Includes coherence, correlation, mutual information, and Granger causality
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import itertools
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class CoherenceAnalyzer:
    """Compute intermuscular coherence across frequency bands."""
    
    def __init__(self, config: dict):
        """
        Initialize coherence analyzer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.fs = config['signal_processing']['sampling_rate']
        self.coherence_config = config['coherence']
        
    def compute_pairwise_coherence(self, emg: np.ndarray, 
                                   band: str = 'full') -> np.ndarray:
        """
        Compute pairwise coherence between all muscle pairs.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals [n_samples, n_channels]
        band : str
            Frequency band name (e.g., 'alpha', 'beta', 'full')
        
        Returns
        -------
        coherence_matrix : np.ndarray
            Pairwise coherence matrix [n_channels, n_channels]
        """
        n_channels = emg.shape[1]
        coherence_matrix = np.eye(n_channels)  # Diagonal is 1
        
        # Get frequency band limits
        freq_range = self.coherence_config['frequency_bands'][band]
        
        # Compute for all pairs
        for i, j in itertools.combinations(range(n_channels), 2):
            coh = self._compute_coherence(emg[:, i], emg[:, j], freq_range)
            coherence_matrix[i, j] = coh
            coherence_matrix[j, i] = coh  # Symmetric
        
        logger.debug(f"Computed coherence matrix for {band} band")
        return coherence_matrix
    
    def _compute_coherence(self, x: np.ndarray, y: np.ndarray, 
                          freq_range: list) -> float:
        """
        Compute coherence between two signals in a frequency band.
        
        Parameters
        ----------
        x, y : np.ndarray
            Input signals
        freq_range : list
            [low_freq, high_freq] in Hz
        
        Returns
        -------
        coherence : float
            Mean coherence in the frequency band
        """
        method = self.coherence_config['method']
        
        if method == 'welch':
            f, Cxy = signal.coherence(
                x, y,
                fs=self.fs,
                nperseg=self.coherence_config['nperseg'],
                noverlap=self.coherence_config['noverlap']
            )
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
        # Extract coherence in frequency band
        band_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        coherence_value = np.mean(Cxy[band_mask])
        
        return coherence_value
    
    def compute_multiband_coherence(self, emg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute coherence across all frequency bands.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals
        
        Returns
        -------
        coherence_bands : dict
            Dictionary mapping band names to coherence matrices
        """
        bands = self.coherence_config['frequency_bands'].keys()
        coherence_bands = {}
        
        for band in bands:
            coherence_bands[band] = self.compute_pairwise_coherence(emg, band)
            logger.info(f"Computed {band} band coherence: "
                       f"mean={np.mean(coherence_bands[band]):.3f}")
        
        return coherence_bands


class ConnectivityAnalyzer:
    """Compute various connectivity measures between muscles."""
    
    def __init__(self, config: dict):
        """Initialize connectivity analyzer."""
        self.config = config
        self.fs = config['signal_processing']['sampling_rate']
    
    def compute_correlation(self, emg: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation matrix.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals [n_samples, n_channels]
        
        Returns
        -------
        correlation_matrix : np.ndarray
            Correlation matrix [n_channels, n_channels]
        """
        return np.corrcoef(emg.T)
    
    def compute_mutual_information(self, emg: np.ndarray) -> np.ndarray:
        """
        Compute mutual information matrix.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals
        
        Returns
        -------
        mi_matrix : np.ndarray
            Mutual information matrix [n_channels, n_channels]
        """
        n_channels = emg.shape[1]
        mi_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                mi = mutual_info_regression(
                    emg[:, i].reshape(-1, 1),
                    emg[:, j],
                    random_state=42
                )[0]
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # Normalize by joint entropy approximation
        mi_matrix = mi_matrix / (np.max(mi_matrix) + 1e-10)
        
        return mi_matrix
    
    def compute_granger_causality(self, emg: np.ndarray, 
                                  max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Granger causality matrix.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals
        max_lag : int
            Maximum lag for causality
        
        Returns
        -------
        gc_matrix : np.ndarray
            Granger causality matrix [n_channels, n_channels]
        p_values : np.ndarray
            P-values for statistical significance
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        n_channels = emg.shape[1]
        gc_matrix = np.zeros((n_channels, n_channels))
        p_values = np.ones((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                
                # Test if i Granger-causes j
                data = np.column_stack([emg[:, j], emg[:, i]])
                
                try:
                    # Perform Granger causality test
                    result = grangercausalitytests(data, max_lag, verbose=False)
                    
                    # Extract F-test results (using lag=1 for simplicity)
                    f_stat = result[1][0]['ssr_ftest'][0]
                    p_val = result[1][0]['ssr_ftest'][1]
                    
                    gc_matrix[i, j] = f_stat
                    p_values[i, j] = p_val
                    
                except:
                    logger.warning(f"Granger causality test failed for pair ({i}, {j})")
        
        # Normalize GC values
        gc_matrix = gc_matrix / (np.max(gc_matrix) + 1e-10)
        
        return gc_matrix, p_values


class TemporalCoherenceAnalyzer:
    """Analyze time-varying coherence patterns."""
    
    def __init__(self, config: dict):
        """Initialize temporal coherence analyzer."""
        self.config = config
        self.fs = config['signal_processing']['sampling_rate']
        
    def compute_sliding_window_coherence(self, emg: np.ndarray, 
                                        window_size: float = 1.0,
                                        overlap: float = 0.5) -> Dict:
        """
        Compute coherence in sliding windows.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals [n_samples, n_channels]
        window_size : float
            Window size in seconds
        overlap : float
            Overlap fraction (0-1)
        
        Returns
        -------
        temporal_coherence : dict
            Time-varying coherence matrices
        """
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))
        
        n_windows = (emg.shape[0] - window_samples) // step_samples + 1
        n_channels = emg.shape[1]
        
        # Initialize storage
        coherence_time_series = np.zeros((n_windows, n_channels, n_channels))
        time_points = np.zeros(n_windows)
        
        # Compute coherence for each window
        coherence_analyzer = CoherenceAnalyzer(self.config)
        
        for w in range(n_windows):
            start_idx = w * step_samples
            end_idx = start_idx + window_samples
            
            window_emg = emg[start_idx:end_idx, :]
            coherence_time_series[w] = coherence_analyzer.compute_pairwise_coherence(
                window_emg, band='full'
            )
            time_points[w] = (start_idx + end_idx) / (2 * self.fs)
        
        return {
            'coherence': coherence_time_series,
            'time': time_points,
            'window_size': window_size,
            'overlap': overlap
        }
    
    def compute_coherence_variability(self, temporal_coherence: Dict) -> np.ndarray:
        """
        Compute variability of coherence over time.
        
        Parameters
        ----------
        temporal_coherence : dict
            Output from compute_sliding_window_coherence
        
        Returns
        -------
        variability : np.ndarray
            Standard deviation of coherence over time [n_channels, n_channels]
        """
        coherence = temporal_coherence['coherence']
        variability = np.std(coherence, axis=0)
        
        return variability