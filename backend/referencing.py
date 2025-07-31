#!/usr/bin/env python3
"""
EEG Re-referencing System
========================

Comprehensive re-referencing capabilities for EEG preprocessing:
- Average reference
- Common reference (specific channel)
- Bipolar referencing
- Linked ears/mastoids referencing
- Custom reference combinations

Author: porfanid
Version: 1.0
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class ReferenceConfig:
    """Configuration class for re-referencing parameters"""
    
    def __init__(
        self,
        ref_type: str,
        ref_channels: Optional[Union[str, List[str]]] = None,
        exclude_channels: Optional[List[str]] = None,
        copy: bool = True
    ):
        """
        Initialize reference configuration
        
        Args:
            ref_type: Type of reference ('average', 'common', 'bipolar', 'linked_ears', 'custom')
            ref_channels: Reference channel(s) for common/custom referencing
            exclude_channels: Channels to exclude from referencing
            copy: Whether to copy data before re-referencing
        """
        self.ref_type = ref_type
        self.ref_channels = ref_channels
        self.exclude_channels = exclude_channels or []
        self.copy = copy
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate reference configuration parameters"""
        valid_types = ['average', 'common', 'bipolar', 'linked_ears', 'custom']
        if self.ref_type not in valid_types:
            raise ValueError(f"Invalid ref_type. Must be one of {valid_types}")
        
        if self.ref_type in ['common', 'custom'] and not self.ref_channels:
            raise ValueError(f"{self.ref_type} reference requires ref_channels parameter")


class EEGReferenceProcessor:
    """
    Advanced EEG re-referencing processor
    
    Provides various re-referencing methods for EEG preprocessing,
    including standard and custom reference schemes.
    """
    
    def __init__(self):
        """Initialize the reference processor"""
        self.reference_history = []
        self.original_reference = None
    
    def apply_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Apply re-referencing to EEG data
        
        Args:
            raw: Raw EEG data
            config: Reference configuration
            
        Returns:
            Tuple of (re-referenced_raw, reference_info)
        """
        if config.copy:
            raw = raw.copy()
        
        # Store original reference info
        if self.original_reference is None:
            self.original_reference = {
                'ref_channels': raw.info.get('custom_ref_applied', 'unknown'),
                'n_channels': len(raw.ch_names),
                'channel_names': raw.ch_names.copy()
            }
        
        # Store reference parameters for history
        ref_info = {
            'type': config.ref_type,
            'ref_channels': config.ref_channels,
            'exclude_channels': config.exclude_channels,
            'n_channels': len(raw.ch_names),
            'success': False,
            'error': None
        }
        
        try:
            if config.ref_type == 'average':
                raw = self._apply_average_reference(raw, config)
            elif config.ref_type == 'common':
                raw = self._apply_common_reference(raw, config)
            elif config.ref_type == 'bipolar':
                raw = self._apply_bipolar_reference(raw, config)
            elif config.ref_type == 'linked_ears':
                raw = self._apply_linked_ears_reference(raw, config)
            elif config.ref_type == 'custom':
                raw = self._apply_custom_reference(raw, config)
            
            ref_info['success'] = True
            self.reference_history.append(ref_info.copy())
            
        except Exception as e:
            ref_info['error'] = str(e)
            raise RuntimeError(f"Reference application failed: {str(e)}")
        
        return raw, ref_info
    
    def _apply_average_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> mne.io.Raw:
        """Apply average reference"""
        # Get EEG channels only
        eeg_channels = self._get_eeg_channels(raw)
        
        # Exclude specified channels
        channels_to_use = [ch for ch in eeg_channels if ch not in config.exclude_channels]
        
        if not channels_to_use:
            raise ValueError("No channels available for average referencing")
        
        # Apply average reference
        raw.set_eeg_reference('average', ch_type='eeg', verbose=False)
        
        return raw
    
    def _apply_common_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> mne.io.Raw:
        """Apply common reference to a specific channel"""
        ref_channel = config.ref_channels
        if isinstance(ref_channel, list):
            ref_channel = ref_channel[0]  # Use first channel if list provided
        
        if ref_channel not in raw.ch_names:
            raise ValueError(f"Reference channel '{ref_channel}' not found in data")
        
        # Apply common reference
        raw.set_eeg_reference([ref_channel], ch_type='eeg', verbose=False)
        
        return raw
    
    def _apply_bipolar_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> mne.io.Raw:
        """Apply bipolar reference (channel pairs)"""
        if not isinstance(config.ref_channels, list) or len(config.ref_channels) < 2:
            raise ValueError("Bipolar reference requires at least 2 channels in ref_channels")
        
        # Create bipolar montage
        bipolar_pairs = []
        ref_channels = config.ref_channels
        
        # Create pairs from consecutive channels
        for i in range(0, len(ref_channels) - 1, 2):
            if i + 1 < len(ref_channels):
                ch1, ch2 = ref_channels[i], ref_channels[i + 1]
                if ch1 in raw.ch_names and ch2 in raw.ch_names:
                    bipolar_pairs.append((ch1, ch2))
        
        if not bipolar_pairs:
            raise ValueError("No valid bipolar pairs found in specified channels")
        
        # Create new channel names and data
        new_ch_names = []
        new_data = []
        
        for ch1, ch2 in bipolar_pairs:
            new_ch_name = f"{ch1}-{ch2}"
            new_ch_names.append(new_ch_name)
            
            # Get data indices
            ch1_idx = raw.ch_names.index(ch1)
            ch2_idx = raw.ch_names.index(ch2)
            
            # Compute bipolar difference
            data1 = raw.get_data([ch1_idx])[0]
            data2 = raw.get_data([ch2_idx])[0]
            bipolar_data = data1 - data2
            new_data.append(bipolar_data)
        
        # Create new Raw object with bipolar data
        new_data = np.array(new_data)
        info = mne.create_info(
            ch_names=new_ch_names,
            sfreq=raw.info['sfreq'],
            ch_types='eeg',
            verbose=False
        )
        
        raw_bipolar = mne.io.RawArray(new_data, info, verbose=False)
        
        return raw_bipolar
    
    def _apply_linked_ears_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> mne.io.Raw:
        """Apply linked ears/mastoids reference"""
        # Common ear/mastoid channel names
        ear_channels = ['A1', 'A2', 'M1', 'M2', 'TP9', 'TP10', 'P9', 'P10']
        
        # Find available ear channels
        available_ears = [ch for ch in ear_channels if ch in raw.ch_names]
        
        if not available_ears:
            # Fallback to specified channels if no standard ear channels found
            if config.ref_channels:
                available_ears = [ch for ch in config.ref_channels if ch in raw.ch_names]
        
        if not available_ears:
            raise ValueError("No ear/mastoid channels found for linked ears reference")
        
        # Apply linked ears reference
        raw.set_eeg_reference(available_ears, ch_type='eeg', verbose=False)
        
        return raw
    
    def _apply_custom_reference(
        self, 
        raw: mne.io.Raw, 
        config: ReferenceConfig
    ) -> mne.io.Raw:
        """Apply custom reference using specified channels"""
        ref_channels = config.ref_channels
        if isinstance(ref_channels, str):
            ref_channels = [ref_channels]
        
        # Validate reference channels
        invalid_channels = [ch for ch in ref_channels if ch not in raw.ch_names]
        if invalid_channels:
            raise ValueError(f"Reference channels not found: {invalid_channels}")
        
        # Apply custom reference
        raw.set_eeg_reference(ref_channels, ch_type='eeg', verbose=False)
        
        return raw
    
    def _get_eeg_channels(self, raw: mne.io.Raw) -> List[str]:
        """Get list of EEG channels from raw data"""
        return [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) 
                if ch_type == 'eeg']
    
    def get_reference_suggestions(self, raw: mne.io.Raw) -> Dict[str, Dict]:
        """
        Get reference suggestions based on available channels
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Dictionary of reference suggestions with descriptions
        """
        suggestions = {}
        ch_names = raw.ch_names
        eeg_channels = self._get_eeg_channels(raw)
        
        # Average reference
        if len(eeg_channels) >= 3:
            suggestions['average'] = {
                'config': ReferenceConfig('average'),
                'description': f"Average reference using {len(eeg_channels)} EEG channels",
                'recommended': True
            }
        
        # Common reference options
        if eeg_channels:
            for ch in eeg_channels[:5]:  # Show first 5 channels as options
                suggestions[f'common_{ch}'] = {
                    'config': ReferenceConfig('common', ref_channels=ch),
                    'description': f"Common reference to channel {ch}",
                    'recommended': False
                }
        
        # Linked ears reference
        ear_channels = ['A1', 'A2', 'M1', 'M2', 'TP9', 'TP10', 'P9', 'P10']
        available_ears = [ch for ch in ear_channels if ch in ch_names]
        
        if len(available_ears) >= 2:
            suggestions['linked_ears'] = {
                'config': ReferenceConfig('linked_ears'),
                'description': f"Linked ears reference using: {', '.join(available_ears)}",
                'recommended': True
            }
        
        # Bipolar reference for neighboring channels
        if len(eeg_channels) >= 4:
            suggestions['bipolar'] = {
                'config': ReferenceConfig('bipolar', ref_channels=eeg_channels[:4]),
                'description': "Bipolar reference using first 4 EEG channels",
                'recommended': False
            }
        
        return suggestions
    
    def create_montage_reference(
        self, 
        raw: mne.io.Raw, 
        montage_pairs: List[Tuple[str, str]]
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Create a custom montage-based reference
        
        Args:
            raw: Raw EEG data
            montage_pairs: List of (channel, reference) pairs
            
        Returns:
            Tuple of (referenced_raw, montage_info)
        """
        raw = raw.copy()
        
        # Validate all channels exist
        all_channels = set()
        for ch, ref in montage_pairs:
            all_channels.update([ch, ref])
        
        missing_channels = [ch for ch in all_channels if ch not in raw.ch_names]
        if missing_channels:
            raise ValueError(f"Channels not found: {missing_channels}")
        
        # Create new montage data
        new_ch_names = []
        new_data = []
        
        for ch, ref in montage_pairs:
            new_ch_name = f"{ch}-{ref}"
            new_ch_names.append(new_ch_name)
            
            # Get data
            ch_idx = raw.ch_names.index(ch)
            ref_idx = raw.ch_names.index(ref)
            
            ch_data = raw.get_data([ch_idx])[0]
            ref_data = raw.get_data([ref_idx])[0]
            montage_data = ch_data - ref_data
            new_data.append(montage_data)
        
        # Create new Raw object
        new_data = np.array(new_data)
        info = mne.create_info(
            ch_names=new_ch_names,
            sfreq=raw.info['sfreq'],
            ch_types='eeg',
            verbose=False
        )
        
        raw_montage = mne.io.RawArray(new_data, info, verbose=False)
        
        montage_info = {
            'type': 'montage',
            'pairs': montage_pairs,
            'n_channels': len(new_ch_names),
            'success': True
        }
        
        return raw_montage, montage_info
    
    def get_reference_history(self) -> List[Dict]:
        """Get history of applied references"""
        return self.reference_history.copy()
    
    def clear_history(self):
        """Clear reference history"""
        self.reference_history.clear()
    
    def restore_original_reference(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Attempt to restore original reference (limited functionality)
        
        Args:
            raw: Current raw data
            
        Returns:
            Raw data with attempted original reference restoration
        """
        if self.original_reference is None:
            raise ValueError("No original reference information available")
        
        # This is a simplified restoration - in practice, true restoration
        # would require storing the original data
        raw = raw.copy()
        
        # Reset to no reference (if possible)
        try:
            raw.set_eeg_reference([], verbose=False)
        except Exception:
            # Fallback to average reference
            raw.set_eeg_reference('average', verbose=False)
        
        return raw


class ReferencePresets:
    """Predefined reference presets for common EEG scenarios"""
    
    @staticmethod
    def get_clinical_preset() -> ReferenceConfig:
        """Clinical EEG reference (average reference)"""
        return ReferenceConfig('average')
    
    @staticmethod
    def get_research_preset() -> ReferenceConfig:
        """Research EEG reference (average reference)"""
        return ReferenceConfig('average')
    
    @staticmethod
    def get_sleep_preset() -> ReferenceConfig:
        """Sleep study reference (linked mastoids)"""
        return ReferenceConfig('linked_ears')
    
    @staticmethod
    def get_erp_preset() -> ReferenceConfig:
        """ERP analysis reference (average reference)"""
        return ReferenceConfig('average')
    
    @staticmethod
    def get_bipolar_preset(channels: List[str]) -> ReferenceConfig:
        """Bipolar montage preset"""
        return ReferenceConfig('bipolar', ref_channels=channels)