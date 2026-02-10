"""
Utility functions for EMG Network Analysis
"""

import os
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    
    # Create logs directory
    if config['logging']['log_to_file']:
        log_dir = Path(config['logging']['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config['logging']['log_file']) 
            if config['logging']['log_to_file'] else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_output_directory(base_dir='outputs', run_name=None):
    """Create timestamped output directory."""
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"run_{timestamp}"
    
    output_dir = Path(base_dir) / run_name
    
    # Create subdirectories
    subdirs = ['graphs', 'features', 'figures', 'reports', 'logs']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_results(data, filename, output_dir, format='json'):
    """Save analysis results to file."""
    output_path = Path(output_dir) / filename
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'npy':
        np.save(output_path, data)
    elif format == 'npz':
        np.savez_compressed(output_path, **data)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def ensure_dir(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_subject_ids(data_dir):
    """Get list of available subject IDs from data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all .hea files (WFDB header files)
    hea_files = list(data_path.glob('*.hea'))
    subject_ids = [f.stem for f in hea_files]
    
    return sorted(subject_ids)


def validate_config(config):
    """Validate configuration parameters."""
    required_keys = ['dataset', 'signal_processing', 'coherence', 'graph']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate sampling rate
    if config['signal_processing']['sampling_rate'] <= 0:
        raise ValueError("Sampling rate must be positive")
    
    # Validate filter parameters
    bp = config['signal_processing']['bandpass_filter']
    if bp['low_cutoff'] >= bp['high_cutoff']:
        raise ValueError("Bandpass low cutoff must be less than high cutoff")
    
    return True


def format_time(seconds):
    """Format seconds into readable time string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"