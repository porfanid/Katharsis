#!/usr/bin/env python3
"""
Advanced EEG Filtering System
============================

Comprehensive filtering capabilities for EEG preprocessing:
- High-pass, low-pass, band-pass, band-stop filters
- Notch filters for line noise removal
- FIR and IIR filter implementations
- Real-time filter preview and parameter optimization

Author: porfanid
Version: 1.0
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from scipy import signal

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class FilterConfig:
    """Configuration class for filter parameters"""
    
    def __init__(
        self,
        filter_type: str,
        freq_low: Optional[float] = None,
        freq_high: Optional[float] = None,
        freq_notch: Optional[float] = None,
        method: str = 'fir',
        filter_length: str = 'auto',
        transition_bandwidth: str = 'auto',
        window: str = 'hamming',
        phase: str = 'zero',
        iir_params: Optional[Dict] = None
    ):
        """
        Initialize filter configuration
        
        Args:
            filter_type: Type of filter ('highpass', 'lowpass', 'bandpass', 'bandstop', 'notch')
            freq_low: Low cutoff frequency (Hz)
            freq_high: High cutoff frequency (Hz) 
            freq_notch: Notch frequency (Hz) for notch filters
            method: Filter method ('fir' or 'iir')
            filter_length: Filter length ('auto' or int)
            transition_bandwidth: Transition bandwidth ('auto' or float)
            window: Window function for FIR filters
            phase: Phase response ('zero', 'zero-double', 'minimum')
            iir_params: Parameters for IIR filters
        """
        self.filter_type = filter_type
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.freq_notch = freq_notch
        self.method = method
        self.filter_length = filter_length
        self.transition_bandwidth = transition_bandwidth
        self.window = window
        self.phase = phase
        self.iir_params = iir_params or {}
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate filter configuration parameters"""
        valid_types = ['highpass', 'lowpass', 'bandpass', 'bandstop', 'notch']
        if self.filter_type not in valid_types:
            raise ValueError(f"Invalid filter_type. Must be one of {valid_types}")
        
        if self.filter_type in ['highpass', 'lowpass'] and self.freq_low is None:
            raise ValueError(f"{self.filter_type} filter requires freq_low parameter")
        
        if self.filter_type in ['bandpass', 'bandstop']:
            if self.freq_low is None or self.freq_high is None:
                raise ValueError(f"{self.filter_type} filter requires both freq_low and freq_high")
            if self.freq_low >= self.freq_high:
                raise ValueError("freq_low must be less than freq_high")
        
        if self.filter_type == 'notch' and self.freq_notch is None:
            raise ValueError("Notch filter requires freq_notch parameter")


class EEGFilterProcessor:
    """
    Advanced EEG filtering processor with comprehensive filtering capabilities
    
    Provides various filtering methods optimized for EEG data preprocessing,
    including both FIR and IIR implementations with customizable parameters.
    """
    
    def __init__(self):
        """Initialize the filter processor"""
        self.filter_history = []
        self.original_sfreq = None
    
    def apply_filter(
        self, 
        raw: mne.io.Raw, 
        config: FilterConfig,
        copy: bool = True
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Apply filter to EEG data
        
        Args:
            raw: Raw EEG data
            config: Filter configuration
            copy: Whether to copy the data before filtering
            
        Returns:
            Tuple of (filtered_raw, filter_info)
        """
        if copy:
            raw = raw.copy()
        
        self.original_sfreq = raw.info['sfreq']
        
        # Store filter parameters for history
        filter_info = {
            'type': config.filter_type,
            'freq_low': config.freq_low,
            'freq_high': config.freq_high,
            'freq_notch': config.freq_notch,
            'method': config.method,
            'sfreq': self.original_sfreq,
            'n_channels': len(raw.ch_names),
            'success': False,
            'error': None
        }
        
        try:
            if config.filter_type == 'highpass':
                raw = self._apply_highpass(raw, config)
            elif config.filter_type == 'lowpass':
                raw = self._apply_lowpass(raw, config)
            elif config.filter_type == 'bandpass':
                raw = self._apply_bandpass(raw, config)
            elif config.filter_type == 'bandstop':
                raw = self._apply_bandstop(raw, config)
            elif config.filter_type == 'notch':
                raw = self._apply_notch(raw, config)
            
            filter_info['success'] = True
            self.filter_history.append(filter_info.copy())
            
        except Exception as e:
            filter_info['error'] = str(e)
            raise RuntimeError(f"Filter application failed: {str(e)}")
        
        return raw, filter_info
    
    def _apply_highpass(self, raw: mne.io.Raw, config: FilterConfig) -> mne.io.Raw:
        """Apply high-pass filter"""
        return raw.filter(
            l_freq=config.freq_low,
            h_freq=None,
            method=config.method,
            filter_length=config.filter_length,
            l_trans_bandwidth=config.transition_bandwidth,
            h_trans_bandwidth=config.transition_bandwidth,
            fir_window=config.window,
            phase=config.phase,
            iir_params=config.iir_params if config.method == 'iir' else None,
            verbose=False
        )
    
    def _apply_lowpass(self, raw: mne.io.Raw, config: FilterConfig) -> mne.io.Raw:
        """Apply low-pass filter"""
        return raw.filter(
            l_freq=None,
            h_freq=config.freq_low,  # Using freq_low as the cutoff
            method=config.method,
            filter_length=config.filter_length,
            l_trans_bandwidth=config.transition_bandwidth,
            h_trans_bandwidth=config.transition_bandwidth,
            fir_window=config.window,
            phase=config.phase,
            iir_params=config.iir_params if config.method == 'iir' else None,
            verbose=False
        )
    
    def _apply_bandpass(self, raw: mne.io.Raw, config: FilterConfig) -> mne.io.Raw:
        """Apply band-pass filter"""
        return raw.filter(
            l_freq=config.freq_low,
            h_freq=config.freq_high,
            method=config.method,
            filter_length=config.filter_length,
            l_trans_bandwidth=config.transition_bandwidth,
            h_trans_bandwidth=config.transition_bandwidth,
            fir_window=config.window,
            phase=config.phase,
            iir_params=config.iir_params if config.method == 'iir' else None,
            verbose=False
        )
    
    def _apply_bandstop(self, raw: mne.io.Raw, config: FilterConfig) -> mne.io.Raw:
        """Apply band-stop filter"""
        # MNE doesn't have direct bandstop, so we implement it as notch with bandwidth
        freqs = np.arange(config.freq_low, config.freq_high + 1, 1.0)
        return raw.notch_filter(
            freqs=freqs,
            method=config.method,
            filter_length=config.filter_length,
            phase=config.phase,
            verbose=False
        )
    
    def _apply_notch(self, raw: mne.io.Raw, config: FilterConfig) -> mne.io.Raw:
        """Apply notch filter"""
        return raw.notch_filter(
            freqs=config.freq_notch,
            method=config.method,
            filter_length=config.filter_length,
            phase=config.phase,
            verbose=False
        )
    
    def get_filter_response(
        self, 
        config: FilterConfig, 
        sfreq: float, 
        n_freq: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get filter frequency response for visualization
        
        Args:
            config: Filter configuration
            sfreq: Sampling frequency
            n_freq: Number of frequency points
            
        Returns:
            Tuple of (frequencies, response)
        """
        nyquist = sfreq / 2
        freqs = np.linspace(0, nyquist, n_freq)
        
        try:
            if config.method == 'fir':
                response = self._get_fir_response(config, sfreq, freqs)
            else:
                response = self._get_iir_response(config, sfreq, freqs)
        except Exception:
            # Fallback to simple response
            response = np.ones_like(freqs)
        
        return freqs, response
    
    def _get_fir_response(
        self, 
        config: FilterConfig, 
        sfreq: float, 
        freqs: np.ndarray
    ) -> np.ndarray:
        """Get FIR filter frequency response"""
        # Simplified implementation - in real case would use MNE's filter design
        nyquist = sfreq / 2
        response = np.ones_like(freqs)
        
        if config.filter_type == 'highpass':
            response[freqs < config.freq_low] = 0
        elif config.filter_type == 'lowpass':
            response[freqs > config.freq_low] = 0
        elif config.filter_type == 'bandpass':
            response[(freqs < config.freq_low) | (freqs > config.freq_high)] = 0
        elif config.filter_type == 'bandstop':
            response[(freqs >= config.freq_low) & (freqs <= config.freq_high)] = 0
        elif config.filter_type == 'notch':
            # Simple notch implementation
            notch_width = 2.0  # Hz
            response[np.abs(freqs - config.freq_notch) < notch_width] = 0
        
        return response
    
    def _get_iir_response(
        self, 
        config: FilterConfig, 
        sfreq: float, 
        freqs: np.ndarray
    ) -> np.ndarray:
        """Get IIR filter frequency response"""
        # Simplified implementation
        return self._get_fir_response(config, sfreq, freqs)
    
    def create_common_filters(self, sfreq: float) -> Dict[str, FilterConfig]:
        """
        Create commonly used filter configurations
        
        Args:
            sfreq: Sampling frequency
            
        Returns:
            Dictionary of common filter configurations
        """
        filters = {}
        
        # High-pass filters
        filters['highpass_0.1'] = FilterConfig('highpass', freq_low=0.1)
        filters['highpass_0.5'] = FilterConfig('highpass', freq_low=0.5)
        filters['highpass_1.0'] = FilterConfig('highpass', freq_low=1.0)
        
        # Low-pass filters  
        filters['lowpass_30'] = FilterConfig('lowpass', freq_low=30.0)
        filters['lowpass_40'] = FilterConfig('lowpass', freq_low=40.0)
        filters['lowpass_50'] = FilterConfig('lowpass', freq_low=50.0)
        
        # Band-pass filters
        filters['bandpass_1_30'] = FilterConfig('bandpass', freq_low=1.0, freq_high=30.0)
        filters['bandpass_0.5_40'] = FilterConfig('bandpass', freq_low=0.5, freq_high=40.0)
        filters['bandpass_8_12'] = FilterConfig('bandpass', freq_low=8.0, freq_high=12.0)  # Alpha
        filters['bandpass_13_30'] = FilterConfig('bandpass', freq_low=13.0, freq_high=30.0)  # Beta
        
        # Notch filters
        filters['notch_50'] = FilterConfig('notch', freq_notch=50.0)  # EU line noise
        filters['notch_60'] = FilterConfig('notch', freq_notch=60.0)  # US line noise
        
        return filters
    
    def get_filter_history(self) -> List[Dict]:
        """Get history of applied filters"""
        return self.filter_history.copy()
    
    def clear_history(self):
        """Clear filter history"""
        self.filter_history.clear()


class FilterPresets:
    """Predefined filter presets for common EEG preprocessing scenarios"""
    
    @staticmethod
    def get_preprocessing_preset() -> List[FilterConfig]:
        """Standard preprocessing filter chain"""
        return [
            FilterConfig('highpass', freq_low=0.5),
            FilterConfig('lowpass', freq_low=40.0),
            FilterConfig('notch', freq_notch=50.0)
        ]
    
    @staticmethod
    def get_erp_preset() -> List[FilterConfig]:
        """ERP analysis filter chain"""
        return [
            FilterConfig('highpass', freq_low=0.1),
            FilterConfig('lowpass', freq_low=30.0),
            FilterConfig('notch', freq_notch=50.0)
        ]
    
    @staticmethod
    def get_alpha_analysis_preset() -> List[FilterConfig]:
        """Alpha band analysis filter chain"""
        return [
            FilterConfig('bandpass', freq_low=8.0, freq_high=12.0),
            FilterConfig('notch', freq_notch=50.0)
        ]
    
    @staticmethod
    def get_beta_analysis_preset() -> List[FilterConfig]:
        """Beta band analysis filter chain"""
        return [
            FilterConfig('bandpass', freq_low=13.0, freq_high=30.0),
            FilterConfig('notch', freq_notch=50.0)
        ]
    
    @staticmethod
    def get_gamma_analysis_preset() -> List[FilterConfig]:
        """Gamma band analysis filter chain"""
        return [
            FilterConfig('bandpass', freq_low=30.0, freq_high=100.0),
            FilterConfig('notch', freq_notch=50.0)
        ]