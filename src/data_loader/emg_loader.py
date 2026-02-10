"""
Data loading module for EMG signals
Handles WFDB format and gait cycle extraction
"""

import numpy as np
import wfdb
from pathlib import Path
import logging
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


class EMGDataLoader:
    """Load and manage EMG data from PhysioNet dataset."""
    
    def __init__(self, data_dir: str, config: dict):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_dir : str
            Path to data directory
        config : dict
            Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.muscle_names = config['muscles']['names']
        
        # Channel indices based on dataset structure
        self.emg_indices = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
        self.footswitch_indices = [0, 7]  # Left and right foot switches
        self.goniometer_indices = [1, 8]  # Knee angles
        
    def load_subject(self, subject_id: str) -> Dict[str, np.ndarray]:
        """
        Load data for a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'S2', 'S15')
        
        Returns
        -------
        data : dict
            Dictionary containing EMG signals, foot switches, metadata
        """
        record_path = self.data_dir / subject_id
        
        try:
            # Load WFDB record
            record = wfdb.rdrecord(str(record_path))
            
            # Extract signals
            signals = record.p_signal
            
            data = {
                'emg': signals[:, self.emg_indices],
                'footswitch': signals[:, self.footswitch_indices],
                'goniometer': signals[:, self.goniometer_indices],
                'fs': record.fs,
                'subject_id': subject_id,
                'signal_names': record.sig_name,
                'duration': len(signals) / record.fs
            }
            
            logger.info(f"Loaded subject {subject_id}: {data['duration']:.2f}s, "
                       f"{data['emg'].shape[1]} channels")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load subject {subject_id}: {str(e)}")
            raise
    
    def detect_gait_events(self, footswitch: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
        """
        Detect gait events (heel strikes, toe offs) from foot switch signals.
        
        Parameters
        ----------
        footswitch : np.ndarray
            Foot switch signals [n_samples, 2] (left, right)
        fs : float
            Sampling frequency
        
        Returns
        -------
        events : dict
            Dictionary with heel strikes and toe offs for each foot
        """
        threshold = self.config['gait']['foot_switch_threshold']
        min_distance = int(self.config['gait']['min_stride_duration'] * fs)
        
        events = {}
        foot_names = ['left', 'right']
        
        for i, foot in enumerate(foot_names):
            # Binarize foot switch signal
            binary = (footswitch[:, i] > threshold).astype(int)
            
            # Find transitions
            diff = np.diff(binary, prepend=0)
            heel_strikes = np.where(diff == 1)[0]  # 0 -> 1 transitions
            toe_offs = np.where(diff == -1)[0]     # 1 -> 0 transitions
            
            # Filter events based on minimum distance
            heel_strikes = self._filter_events(heel_strikes, min_distance)
            toe_offs = self._filter_events(toe_offs, min_distance)
            
            events[f'{foot}_heel_strike'] = heel_strikes
            events[f'{foot}_toe_off'] = toe_offs
        
        logger.info(f"Detected {len(events['left_heel_strike'])} left strides, "
                   f"{len(events['right_heel_strike'])} right strides")
        
        return events
    
    def _filter_events(self, events: np.ndarray, min_distance: int) -> np.ndarray:
        """Filter events to ensure minimum distance between consecutive events."""
        if len(events) == 0:
            return events
        
        filtered = [events[0]]
        for event in events[1:]:
            if event - filtered[-1] >= min_distance:
                filtered.append(event)
        
        return np.array(filtered)
    
    def extract_gait_cycles(self, emg: np.ndarray, events: Dict[str, np.ndarray], 
                           fs: float, foot: str = 'left') -> List[np.ndarray]:
        """
        Extract individual gait cycles from EMG data.
        
        Parameters
        ----------
        emg : np.ndarray
            EMG signals [n_samples, n_channels]
        events : dict
            Gait events dictionary
        fs : float
            Sampling frequency
        foot : str
            Which foot to use ('left' or 'right')
        
        Returns
        -------
        cycles : list of np.ndarray
            List of EMG cycles, each [cycle_length, n_channels]
        """
        heel_strikes = events[f'{foot}_heel_strike']
        
        max_duration = int(self.config['gait']['max_stride_duration'] * fs)
        cycles = []
        
        for i in range(len(heel_strikes) - 1):
            start = heel_strikes[i]
            end = heel_strikes[i + 1]
            
            # Validate cycle duration
            if end - start > max_duration:
                logger.warning(f"Skipping abnormally long cycle: {(end-start)/fs:.2f}s")
                continue
            
            cycle = emg[start:end, :]
            cycles.append(cycle)
        
        logger.info(f"Extracted {len(cycles)} valid gait cycles for {foot} foot")
        return cycles
    
    def normalize_cycles(self, cycles: List[np.ndarray], 
                        n_points: int = 101) -> np.ndarray:
        """
        Time-normalize gait cycles to standard length.
        
        Parameters
        ----------
        cycles : list of np.ndarray
            List of variable-length cycles
        n_points : int
            Number of points for normalized cycle (default: 101 for 0-100%)
        
        Returns
        -------
        normalized : np.ndarray
            Normalized cycles [n_cycles, n_points, n_channels]
        """
        n_channels = cycles[0].shape[1]
        normalized = np.zeros((len(cycles), n_points, n_channels))
        
        for i, cycle in enumerate(cycles):
            # Interpolate each channel to standard length
            old_time = np.linspace(0, 1, len(cycle))
            new_time = np.linspace(0, 1, n_points)
            
            for ch in range(n_channels):
                normalized[i, :, ch] = np.interp(new_time, old_time, cycle[:, ch])
        
        return normalized
    
    def get_cycle_phases(self, events: Dict[str, np.ndarray], 
                        heel_strike_idx: int, foot: str = 'left') -> Dict[str, Tuple[int, int]]:
        """
        Get stance and swing phase boundaries for a gait cycle.
        
        Parameters
        ----------
        events : dict
            Gait events dictionary
        heel_strike_idx : int
            Index of the heel strike event
        foot : str
            Which foot ('left' or 'right')
        
        Returns
        -------
        phases : dict
            Dictionary with 'stance' and 'swing' phase boundaries
        """
        heel_strikes = events[f'{foot}_heel_strike']
        toe_offs = events[f'{foot}_toe_off']
        
        if heel_strike_idx >= len(heel_strikes) - 1:
            return None
        
        cycle_start = heel_strikes[heel_strike_idx]
        cycle_end = heel_strikes[heel_strike_idx + 1]
        
        # Find toe-off within this cycle
        toe_off_in_cycle = toe_offs[(toe_offs > cycle_start) & (toe_offs < cycle_end)]
        
        if len(toe_off_in_cycle) == 0:
            return None
        
        toe_off = toe_off_in_cycle[0]
        
        phases = {
            'stance': (cycle_start, toe_off),
            'swing': (toe_off, cycle_end)
        }
        
        return phases