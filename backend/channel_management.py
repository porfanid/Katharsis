#!/usr/bin/env python3
"""
Advanced EEG Channel Management System
=====================================

Comprehensive channel management capabilities:
- Bad channel detection (automatic and manual)
- Channel interpolation (spherical splines)
- Channel location management
- Montage loading and editing
- Channel grouping and selection

Author: porfanid
Version: 1.0
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from scipy.spatial.distance import cdist

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class ChannelInfo:
    """Information about a single EEG channel"""
    
    def __init__(
        self,
        name: str,
        ch_type: str = 'eeg',
        loc: Optional[np.ndarray] = None,
        unit: str = 'V',
        is_bad: bool = False,
        impedance: Optional[float] = None
    ):
        """
        Initialize channel information
        
        Args:
            name: Channel name
            ch_type: Channel type ('eeg', 'ecg', 'eog', etc.)
            loc: 3D location coordinates
            unit: Signal unit
            is_bad: Whether channel is marked as bad
            impedance: Channel impedance (if available)
        """
        self.name = name
        self.ch_type = ch_type
        self.loc = loc
        self.unit = unit
        self.is_bad = is_bad
        self.impedance = impedance


class BadChannelDetector:
    """Automatic detection of bad EEG channels"""
    
    def __init__(
        self,
        flat_threshold: float = 1e-6,
        noisy_threshold: float = 5.0,
        correlation_threshold: float = 0.1,
        hf_noise_threshold: float = 3.0
    ):
        """
        Initialize bad channel detector
        
        Args:
            flat_threshold: Threshold for detecting flat channels (μV)
            noisy_threshold: Z-score threshold for noisy channels  
            correlation_threshold: Minimum correlation with neighbors
            hf_noise_threshold: High-frequency noise threshold
        """
        self.flat_threshold = flat_threshold
        self.noisy_threshold = noisy_threshold
        self.correlation_threshold = correlation_threshold
        self.hf_noise_threshold = hf_noise_threshold
    
    def detect_bad_channels(
        self, 
        raw: mne.io.Raw,
        eeg_channels: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Detect bad channels using multiple criteria
        
        Args:
            raw: Raw EEG data
            eeg_channels: List of EEG channels to check (if None, auto-detect)
            
        Returns:
            Dictionary with lists of bad channels by type
        """
        if eeg_channels is None:
            eeg_channels = self._get_eeg_channels(raw)
        
        if not eeg_channels:
            return {'flat': [], 'noisy': [], 'uncorrelated': [], 'high_freq_noise': []}
        
        # Get data for analysis
        data, times = raw[eeg_channels, :]
        
        bad_channels = {
            'flat': self._detect_flat_channels(data, eeg_channels),
            'noisy': self._detect_noisy_channels(data, eeg_channels),
            'uncorrelated': self._detect_uncorrelated_channels(data, eeg_channels),
            'high_freq_noise': self._detect_hf_noise_channels(raw, eeg_channels)
        }
        
        return bad_channels
    
    def _detect_flat_channels(
        self, 
        data: np.ndarray, 
        channel_names: List[str]
    ) -> List[str]:
        """Detect flat (no signal variation) channels"""
        flat_channels = []
        
        for i, ch_name in enumerate(channel_names):
            ch_data = data[i, :]
            signal_std = np.std(ch_data)
            
            if signal_std < self.flat_threshold:
                flat_channels.append(ch_name)
        
        return flat_channels
    
    def _detect_noisy_channels(
        self, 
        data: np.ndarray, 
        channel_names: List[str]
    ) -> List[str]:
        """Detect noisy channels using z-score of standard deviation"""
        noisy_channels = []
        
        # Calculate standard deviation for each channel
        channel_stds = np.std(data, axis=1)
        
        # Calculate z-scores
        mean_std = np.mean(channel_stds)
        std_std = np.std(channel_stds)
        
        if std_std > 0:
            z_scores = (channel_stds - mean_std) / std_std
            
            for i, ch_name in enumerate(channel_names):
                if z_scores[i] > self.noisy_threshold:
                    noisy_channels.append(ch_name)
        
        return noisy_channels
    
    def _detect_uncorrelated_channels(
        self, 
        data: np.ndarray, 
        channel_names: List[str]
    ) -> List[str]:
        """Detect channels with low correlation to neighbors"""
        uncorrelated_channels = []
        
        if len(channel_names) < 3:
            return uncorrelated_channels
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data)
        
        for i, ch_name in enumerate(channel_names):
            # Calculate mean correlation with other channels (excluding self)
            other_correlations = correlation_matrix[i, :]
            other_correlations = other_correlations[np.arange(len(other_correlations)) != i]
            
            # Remove NaN values
            other_correlations = other_correlations[~np.isnan(other_correlations)]
            
            if len(other_correlations) > 0:
                mean_correlation = np.mean(np.abs(other_correlations))
                
                if mean_correlation < self.correlation_threshold:
                    uncorrelated_channels.append(ch_name)
        
        return uncorrelated_channels
    
    def _detect_hf_noise_channels(
        self, 
        raw: mne.io.Raw, 
        channel_names: List[str]
    ) -> List[str]:
        """Detect channels with high-frequency noise"""
        hf_noise_channels = []
        
        sfreq = raw.info['sfreq']
        
        # High-pass filter to isolate high-frequency content
        try:
            raw_hf = raw.copy().filter(l_freq=50, h_freq=None, verbose=False)
            hf_data, _ = raw_hf[channel_names, :]
            
            # Calculate high-frequency power for each channel
            hf_powers = np.var(hf_data, axis=1)
            
            # Z-score normalization
            mean_hf_power = np.mean(hf_powers)
            std_hf_power = np.std(hf_powers)
            
            if std_hf_power > 0:
                hf_z_scores = (hf_powers - mean_hf_power) / std_hf_power
                
                for i, ch_name in enumerate(channel_names):
                    if hf_z_scores[i] > self.hf_noise_threshold:
                        hf_noise_channels.append(ch_name)
        
        except Exception:
            # If filtering fails, return empty list
            pass
        
        return hf_noise_channels
    
    def _get_eeg_channels(self, raw: mne.io.Raw) -> List[str]:
        """Get list of EEG channels from raw data"""
        return [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) 
                if ch_type == 'eeg']


class ChannelInterpolator:
    """Channel interpolation using spherical splines"""
    
    def __init__(self, method: str = 'spline'):
        """
        Initialize channel interpolator
        
        Args:
            method: Interpolation method ('spline' or 'MNE')
        """
        self.method = method
        self.interpolation_history = []
    
    def interpolate_channels(
        self, 
        raw: mne.io.Raw, 
        bad_channels: List[str],
        copy: bool = True
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Interpolate bad channels
        
        Args:
            raw: Raw EEG data
            bad_channels: List of channels to interpolate
            copy: Whether to copy data before interpolation
            
        Returns:
            Tuple of (interpolated_raw, interpolation_info)
        """
        if copy:
            raw = raw.copy()
        
        # Validate channels exist
        invalid_channels = [ch for ch in bad_channels if ch not in raw.ch_names]
        if invalid_channels:
            raise ValueError(f"Channels not found: {invalid_channels}")
        
        # Store original bad channels
        original_bads = raw.info['bads'].copy()
        
        # Mark channels as bad
        raw.info['bads'] = list(set(raw.info['bads'] + bad_channels))
        
        interpolation_info = {
            'method': self.method,
            'interpolated_channels': bad_channels,
            'n_interpolated': len(bad_channels),
            'original_bads': original_bads,
            'success': False,
            'error': None
        }
        
        try:
            # Perform interpolation
            if self.method == 'spline':
                raw = self._interpolate_spline(raw)
            else:
                raw.interpolate_bads(reset_bads=False, verbose=False)
            
            interpolation_info['success'] = True
            self.interpolation_history.append(interpolation_info.copy())
            
        except Exception as e:
            interpolation_info['error'] = str(e)
            raise RuntimeError(f"Channel interpolation failed: {str(e)}")
        
        return raw, interpolation_info
    
    def _interpolate_spline(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Interpolate using spherical splines (MNE implementation)"""
        return raw.interpolate_bads(reset_bads=False, verbose=False)
    
    def get_interpolation_history(self) -> List[Dict]:
        """Get history of interpolations"""
        return self.interpolation_history.copy()


class MontageManager:
    """Management of EEG montages and channel locations"""
    
    def __init__(self):
        """Initialize montage manager"""
        self.current_montage = None
        self.montage_history = []
    
    def load_standard_montage(
        self, 
        raw: mne.io.Raw, 
        montage_name: str = 'standard_1020'
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Load standard EEG montage
        
        Args:
            raw: Raw EEG data
            montage_name: Name of standard montage
            
        Returns:
            Tuple of (raw_with_montage, montage_info)
        """
        raw = raw.copy()
        
        montage_info = {
            'montage_name': montage_name,
            'success': False,
            'error': None,
            'matched_channels': [],
            'unmatched_channels': []
        }
        
        try:
            # Load standard montage
            montage = mne.channels.make_standard_montage(montage_name)
            
            # Find matching channels
            montage_channels = set(montage.ch_names)
            raw_channels = set(raw.ch_names)
            
            matched = list(montage_channels.intersection(raw_channels))
            unmatched = list(raw_channels - montage_channels)
            
            montage_info['matched_channels'] = matched
            montage_info['unmatched_channels'] = unmatched
            
            # Set montage
            raw.set_montage(montage, match_case=False, verbose=False)
            
            self.current_montage = montage
            montage_info['success'] = True
            self.montage_history.append(montage_info.copy())
            
        except Exception as e:
            montage_info['error'] = str(e)
            raise RuntimeError(f"Montage loading failed: {str(e)}")
        
        return raw, montage_info
    
    def load_custom_montage(
        self, 
        raw: mne.io.Raw, 
        montage_file: Union[str, Path]
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Load custom montage from file
        
        Args:
            raw: Raw EEG data
            montage_file: Path to montage file
            
        Returns:
            Tuple of (raw_with_montage, montage_info)
        """
        raw = raw.copy()
        montage_file = Path(montage_file)
        
        montage_info = {
            'montage_file': str(montage_file),
            'success': False,
            'error': None
        }
        
        try:
            # Load montage based on file extension
            if montage_file.suffix.lower() in ['.sfp', '.elc']:
                montage = mne.channels.read_custom_montage(montage_file)
            elif montage_file.suffix.lower() == '.locs':
                montage = mne.channels.read_custom_montage(montage_file, head_size=0.095)
            else:
                raise ValueError(f"Unsupported montage file format: {montage_file.suffix}")
            
            raw.set_montage(montage, match_case=False, verbose=False)
            
            self.current_montage = montage
            montage_info['success'] = True
            self.montage_history.append(montage_info.copy())
            
        except Exception as e:
            montage_info['error'] = str(e)
            raise RuntimeError(f"Custom montage loading failed: {str(e)}")
        
        return raw, montage_info
    
    def get_available_montages(self) -> List[str]:
        """Get list of available standard montages"""
        standard_montages = [
            'standard_1020',
            'standard_1005', 
            'standard_postfixed',
            'biosemi16',
            'biosemi32',
            'biosemi64',
            'biosemi128',
            'biosemi256',
            'easycap-M1',
            'easycap-M10',
            'EGI_256',
            'GSN-HydroCel-128',
            'GSN-HydroCel-129',
            'GSN-HydroCel-256',
            'GSN-HydroCel-257'
        ]
        return standard_montages
    
    def validate_montage_match(
        self, 
        raw: mne.io.Raw, 
        montage_name: str
    ) -> Dict[str, List[str]]:
        """
        Validate how well a montage matches the data channels
        
        Args:
            raw: Raw EEG data
            montage_name: Name of montage to validate
            
        Returns:
            Dictionary with matched/unmatched channels
        """
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            montage_channels = set(montage.ch_names)
            raw_channels = set(raw.ch_names)
            
            return {
                'matched': list(montage_channels.intersection(raw_channels)),
                'unmatched_in_data': list(raw_channels - montage_channels),
                'unmatched_in_montage': list(montage_channels - raw_channels),
                'match_percentage': len(montage_channels.intersection(raw_channels)) / len(raw_channels) * 100
            }
        except Exception as e:
            return {
                'matched': [],
                'unmatched_in_data': list(raw.ch_names),
                'unmatched_in_montage': [],
                'match_percentage': 0.0,
                'error': str(e)
            }


class ChannelGroupManager:
    """Management of channel groups and selections"""
    
    def __init__(self):
        """Initialize channel group manager"""
        self.channel_groups = {}
    
    def create_anatomical_groups(self, raw: mne.io.Raw) -> Dict[str, List[str]]:
        """
        Create anatomical channel groups
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Dictionary of anatomical groups
        """
        ch_names = raw.ch_names
        groups = {}
        
        # Frontal channels
        frontal_patterns = ['FP', 'AF', 'F']
        groups['frontal'] = [ch for ch in ch_names 
                           if any(ch.startswith(pattern) for pattern in frontal_patterns)]
        
        # Central channels
        central_patterns = ['FC', 'C', 'CP']
        groups['central'] = [ch for ch in ch_names 
                           if any(ch.startswith(pattern) for pattern in central_patterns)]
        
        # Parietal channels
        parietal_patterns = ['P', 'PO']
        groups['parietal'] = [ch for ch in ch_names 
                            if any(ch.startswith(pattern) for pattern in parietal_patterns)]
        
        # Occipital channels
        occipital_patterns = ['O']
        groups['occipital'] = [ch for ch in ch_names 
                             if any(ch.startswith(pattern) for pattern in occipital_patterns)]
        
        # Temporal channels
        temporal_patterns = ['FT', 'T', 'TP']
        groups['temporal'] = [ch for ch in ch_names 
                            if any(ch.startswith(pattern) for pattern in temporal_patterns)]
        
        # Left hemisphere
        groups['left'] = [ch for ch in ch_names if ch.endswith(('1', '3', '5', '7', '9'))]
        
        # Right hemisphere  
        groups['right'] = [ch for ch in ch_names if ch.endswith(('2', '4', '6', '8', '10'))]
        
        # Midline
        groups['midline'] = [ch for ch in ch_names if ch.endswith('z') or 'z' in ch.lower()]
        
        self.channel_groups.update(groups)
        return groups
    
    def create_frequency_groups(self, raw: mne.io.Raw) -> Dict[str, List[str]]:
        """
        Create channel groups optimized for frequency analysis
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Dictionary of frequency-optimized groups
        """
        ch_names = raw.ch_names
        eeg_channels = [ch for ch, ch_type in zip(ch_names, raw.get_channel_types()) 
                       if ch_type == 'eeg']
        
        groups = {
            'alpha_sites': [],  # Posterior channels for alpha analysis
            'beta_sites': [],   # Central channels for beta analysis
            'theta_sites': [],  # Frontal channels for theta analysis
            'gamma_sites': []   # Temporal channels for gamma analysis
        }
        
        # Alpha sites (posterior)
        alpha_patterns = ['P', 'PO', 'O']
        groups['alpha_sites'] = [ch for ch in eeg_channels 
                               if any(ch.startswith(pattern) for pattern in alpha_patterns)]
        
        # Beta sites (central)
        beta_patterns = ['C', 'FC', 'CP']
        groups['beta_sites'] = [ch for ch in eeg_channels 
                              if any(ch.startswith(pattern) for pattern in beta_patterns)]
        
        # Theta sites (frontal)
        theta_patterns = ['F', 'FP', 'AF']
        groups['theta_sites'] = [ch for ch in eeg_channels 
                               if any(ch.startswith(pattern) for pattern in theta_patterns)]
        
        # Gamma sites (temporal)
        gamma_patterns = ['T', 'FT', 'TP']
        groups['gamma_sites'] = [ch for ch in eeg_channels 
                               if any(ch.startswith(pattern) for pattern in gamma_patterns)]
        
        self.channel_groups.update(groups)
        return groups
    
    def add_custom_group(self, group_name: str, channels: List[str]):
        """Add custom channel group"""
        self.channel_groups[group_name] = channels
    
    def get_group(self, group_name: str) -> List[str]:
        """Get channels in a specific group"""
        return self.channel_groups.get(group_name, [])
    
    def list_groups(self) -> List[str]:
        """List all available groups"""
        return list(self.channel_groups.keys())


class EEGChannelManager:
    """Main channel management system combining all functionality"""
    
    def __init__(self):
        """Initialize channel manager"""
        self.bad_channel_detector = BadChannelDetector()
        self.channel_interpolator = ChannelInterpolator()
        self.montage_manager = MontageManager()
        self.group_manager = ChannelGroupManager()
    
    def analyze_channels(self, raw: mne.io.Raw) -> Dict:
        """
        Comprehensive channel analysis
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Dictionary with comprehensive channel analysis
        """
        analysis = {
            'n_channels': len(raw.ch_names),
            'channel_types': {},
            'bad_channels': {},
            'montage_info': {},
            'groups': {}
        }
        
        # Channel types
        ch_types = raw.get_channel_types()
        for ch_type in set(ch_types):
            analysis['channel_types'][ch_type] = ch_types.count(ch_type)
        
        # Bad channel detection
        eeg_channels = [ch for ch, ch_type in zip(raw.ch_names, ch_types) 
                       if ch_type == 'eeg']
        
        if eeg_channels:
            analysis['bad_channels'] = self.bad_channel_detector.detect_bad_channels(
                raw, eeg_channels
            )
        
        # Montage validation
        if raw.info.get_montage() is not None:
            analysis['montage_info'] = {
                'has_montage': True,
                'montage_kind': str(type(raw.info.get_montage())),
                'n_positions': len(raw.info.get_montage().get_positions()['ch_pos'])
            }
        else:
            analysis['montage_info'] = {'has_montage': False}
        
        # Channel groups
        analysis['groups'] = {
            'anatomical': self.group_manager.create_anatomical_groups(raw),
            'frequency': self.group_manager.create_frequency_groups(raw)
        }
        
        return analysis
    
    def get_channel_recommendations(self, raw: mne.io.Raw) -> Dict:
        """
        Get recommendations for channel processing
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Dictionary with processing recommendations
        """
        analysis = self.analyze_channels(raw)
        recommendations = {
            'bad_channels_to_remove': [],
            'bad_channels_to_interpolate': [],
            'montage_suggestions': [],
            'reference_suggestions': []
        }
        
        # Bad channel recommendations
        bad_channels = analysis['bad_channels']
        all_bad = set()
        for bad_type, channels in bad_channels.items():
            all_bad.update(channels)
        
        # Recommend removal for channels that are bad in multiple ways
        bad_counts = {}
        for bad_type, channels in bad_channels.items():
            for ch in channels:
                bad_counts[ch] = bad_counts.get(ch, 0) + 1
        
        for ch, count in bad_counts.items():
            if count >= 2:  # Bad in multiple ways
                recommendations['bad_channels_to_remove'].append(ch)
            else:
                recommendations['bad_channels_to_interpolate'].append(ch)
        
        # Montage suggestions
        if not analysis['montage_info']['has_montage']:
            montages = self.montage_manager.get_available_montages()
            for montage in montages[:3]:  # Top 3 suggestions
                match_info = self.montage_manager.validate_montage_match(raw, montage)
                if match_info['match_percentage'] > 50:
                    recommendations['montage_suggestions'].append({
                        'montage': montage,
                        'match_percentage': match_info['match_percentage']
                    })
        
        return recommendations