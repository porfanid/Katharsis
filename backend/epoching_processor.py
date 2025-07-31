#!/usr/bin/env python3
"""
Epoching and Segmentation Processor
==================================

Phase 3.1: Advanced epoching and segmentation system for time-domain analysis.
Provides comprehensive tools for event-based epoching, time-based segmentation,
baseline correction, and epoch quality assessment.

Features:
- Event-based epoching with multiple event types
- Time-based segmentation with flexible window definitions
- Multiple baseline correction methods
- Epoch rejection based on amplitude, gradient, and statistical criteria
- Condition-based epoch sorting and organization
- Comprehensive epoch quality metrics

Author: porfanid
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import mne
from mne.epochs import BaseEpochs


class BaselineCorrectionMethod(Enum):
    """Baseline correction methods"""
    NONE = "none"
    MEAN = "mean" 
    MEDIAN = "median"
    RESCALE = "rescale"
    ZSCORE = "zscore"
    LOGRATIO = "logratio"


class EpochRejectionCriteria(Enum):
    """Epoch rejection criteria types"""
    AMPLITUDE = "amplitude"
    PEAK_TO_PEAK = "peak_to_peak"
    GRADIENT = "gradient"
    VARIANCE = "variance"
    KURTOSIS = "kurtosis"
    MUSCLE = "muscle"
    EOG = "eog"


@dataclass
class EpochingConfig:
    """Configuration for epoching operations"""
    
    # Event-based epoching
    event_ids: Dict[str, int] = None  # Event ID mapping {'stimulus': 1, 'response': 2}
    tmin: float = -0.2  # Start time before event (seconds)
    tmax: float = 0.8   # End time after event (seconds)
    
    # Baseline correction
    baseline_method: BaselineCorrectionMethod = BaselineCorrectionMethod.MEAN
    baseline_tmin: Optional[float] = None  # Start of baseline period (None = tmin)
    baseline_tmax: Optional[float] = 0.0   # End of baseline period
    
    # Epoch rejection
    rejection_criteria: Dict[EpochRejectionCriteria, float] = None
    flat_criteria: Dict[str, float] = None  # Flat epoch detection {'eeg': 1e-6}
    
    # Advanced options
    preload: bool = True
    detrend: Optional[int] = None  # Polynomial order for detrending (None = no detrend)
    picks: Optional[List[str]] = None  # Channel selection
    proj: bool = True  # Apply projections
    decim: int = 1  # Decimation factor
    
    def __post_init__(self):
        """Initialize default values"""
        if self.event_ids is None:
            self.event_ids = {'stimulus': 1}
        
        if self.rejection_criteria is None:
            self.rejection_criteria = {
                EpochRejectionCriteria.AMPLITUDE: 150e-6,  # 150 µV
                EpochRejectionCriteria.PEAK_TO_PEAK: 200e-6,  # 200 µV
                EpochRejectionCriteria.GRADIENT: 75e-6,  # 75 µV
            }
        
        if self.flat_criteria is None:
            self.flat_criteria = {'eeg': 1e-6}


@dataclass
class SegmentationConfig:
    """Configuration for time-based segmentation"""
    
    segment_length: float = 2.0  # Segment length in seconds
    overlap: float = 0.0  # Overlap between segments (0.0 - 1.0)
    window_function: str = 'hann'  # Window function to apply
    detrend_segments: bool = True
    
    # Quality criteria for segments
    min_good_channels: float = 0.8  # Minimum fraction of good channels
    max_amplitude: float = 200e-6  # Maximum amplitude threshold
    max_gradient: float = 100e-6  # Maximum gradient threshold


@dataclass
class EpochQualityMetrics:
    """Quality metrics for epochs"""
    
    n_epochs_total: int
    n_epochs_good: int  
    n_epochs_rejected: int
    rejection_rate: float
    
    # Per-criterion rejection counts
    rejection_reasons: Dict[str, int]
    
    # Signal quality metrics
    mean_amplitude: float
    std_amplitude: float
    snr_estimate: float
    
    # Channel-wise statistics
    channel_rejection_rates: Dict[str, float]
    worst_channels: List[str]


class EpochingProcessor:
    """
    Advanced epoching and segmentation processor for EEG data.
    
    This class provides comprehensive epoching capabilities including:
    - Event-based epoching with flexible event handling
    - Time-based segmentation for continuous analysis
    - Multiple baseline correction methods
    - Advanced epoch rejection criteria
    - Quality assessment and reporting
    """
    
    def __init__(self):
        """Initialize the epoching processor"""
        self.epochs_ = None
        self.events_ = None
        self.quality_metrics_ = None
        self.rejection_log_ = []
    
    def create_epochs_from_events(self, raw: mne.io.Raw, events: np.ndarray, 
                                config: EpochingConfig) -> mne.Epochs:
        """
        Create epochs from event markers in the data.
        
        Args:
            raw: MNE Raw object with EEG data
            events: Event array (n_events, 3) with [sample, prev_value, event_id]
            config: Epoching configuration
            
        Returns:
            MNE Epochs object with epoched data
        """
        try:
            # Create epochs
            epochs = mne.Epochs(
                raw=raw,
                events=events,
                event_id=config.event_ids,
                tmin=config.tmin,
                tmax=config.tmax,
                baseline=(config.baseline_tmin, config.baseline_tmax),
                picks=config.picks,
                preload=config.preload,
                proj=config.proj,
                detrend=config.detrend,
                decim=config.decim,
                verbose='WARNING'
            )
            
            # Apply baseline correction
            if config.baseline_method != BaselineCorrectionMethod.NONE:
                epochs = self._apply_baseline_correction(epochs, config)
            
            # Apply rejection criteria
            epochs = self._apply_rejection_criteria(epochs, config)
            
            # Store results
            self.epochs_ = epochs
            self.events_ = events
            
            # Calculate quality metrics
            self.quality_metrics_ = self._calculate_quality_metrics(epochs, config)
            
            return epochs
            
        except Exception as e:
            raise RuntimeError(f"Failed to create epochs from events: {str(e)}")
    
    def create_fixed_length_epochs(self, raw: mne.io.Raw, config: SegmentationConfig) -> mne.Epochs:
        """
        Create fixed-length epochs for continuous data analysis.
        
        Args:
            raw: MNE Raw object with EEG data
            config: Segmentation configuration
            
        Returns:
            MNE Epochs object with fixed-length epochs
        """
        try:
            # Create fixed-length epochs
            epochs = mne.make_fixed_length_epochs(
                raw=raw,
                duration=config.segment_length,
                overlap=config.overlap * config.segment_length,
                preload=True,
                verbose='WARNING'
            )
            
            # Apply windowing if specified
            if config.window_function != 'none':
                epochs = self._apply_windowing(epochs, config.window_function)
            
            # Apply detrending if specified
            if config.detrend_segments:
                epochs.detrend(1)  # Linear detrending
            
            # Apply quality criteria
            epochs = self._apply_segment_quality_criteria(epochs, config)
            
            self.epochs_ = epochs
            
            return epochs
            
        except Exception as e:
            raise RuntimeError(f"Failed to create fixed-length epochs: {str(e)}")
    
    def find_events_from_raw(self, raw: mne.io.Raw, stim_channel: str = 'STI 014',
                           min_duration: float = 0.002) -> np.ndarray:
        """
        Find events from stimulus channel in raw data.
        
        Args:
            raw: MNE Raw object
            stim_channel: Name of stimulus channel
            min_duration: Minimum event duration
            
        Returns:
            Event array (n_events, 3)
        """
        try:
            events = mne.find_events(
                raw=raw,
                stim_channel=stim_channel,
                min_duration=min_duration,
                verbose='WARNING'
            )
            
            return events
            
        except Exception as e:
            raise RuntimeError(f"Failed to find events: {str(e)}")
    
    def create_events_from_annotations(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Create events from annotations in the raw data.
        
        Args:
            raw: MNE Raw object with annotations
            
        Returns:
            Tuple of (events array, event_id dictionary)
        """
        try:
            events, event_id = mne.events_from_annotations(raw, verbose='WARNING')
            return events, event_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create events from annotations: {str(e)}")
    
    def _apply_baseline_correction(self, epochs: mne.Epochs, config: EpochingConfig) -> mne.Epochs:
        """Apply specified baseline correction method"""
        
        if epochs.baseline is None:
            return epochs
            
        if config.baseline_method == BaselineCorrectionMethod.MEAN:
            # Default MNE baseline correction (already applied)
            return epochs
            
        elif config.baseline_method == BaselineCorrectionMethod.MEDIAN:
            # Median baseline correction
            baseline_data = epochs.copy().crop(
                tmin=config.baseline_tmin or config.tmin,
                tmax=config.baseline_tmax
            ).get_data()
            
            baseline_median = np.median(baseline_data, axis=2, keepdims=True)
            epochs._data -= baseline_median
            
        elif config.baseline_method == BaselineCorrectionMethod.RESCALE:
            # Rescale baseline to unit variance
            baseline_data = epochs.copy().crop(
                tmin=config.baseline_tmin or config.tmin,
                tmax=config.baseline_tmax
            ).get_data()
            
            baseline_std = np.std(baseline_data, axis=2, keepdims=True)
            baseline_std[baseline_std == 0] = 1  # Avoid division by zero
            epochs._data /= baseline_std
            
        elif config.baseline_method == BaselineCorrectionMethod.ZSCORE:
            # Z-score normalization
            baseline_data = epochs.copy().crop(
                tmin=config.baseline_tmin or config.tmin,
                tmax=config.baseline_tmax
            ).get_data()
            
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_std = np.std(baseline_data, axis=2, keepdims=True)
            baseline_std[baseline_std == 0] = 1
            
            epochs._data = (epochs._data - baseline_mean) / baseline_std
            
        elif config.baseline_method == BaselineCorrectionMethod.LOGRATIO:
            # Log-ratio baseline correction
            baseline_data = epochs.copy().crop(
                tmin=config.baseline_tmin or config.tmin,
                tmax=config.baseline_tmax
            ).get_data()
            
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_mean[baseline_mean <= 0] = 1e-10  # Avoid log(0)
            
            epochs._data = np.log(np.abs(epochs._data) / np.abs(baseline_mean))
        
        return epochs
    
    def _apply_rejection_criteria(self, epochs: mne.Epochs, config: EpochingConfig) -> mne.Epochs:
        """Apply epoch rejection criteria"""
        
        n_epochs_initial = len(epochs)
        rejection_log = []
        
        for criterion, threshold in config.rejection_criteria.items():
            n_before = len(epochs)
            
            if criterion == EpochRejectionCriteria.AMPLITUDE:
                # Amplitude-based rejection - only include channel types present in data
                available_types = set(epochs.get_channel_types())
                reject_dict = {}
                for ch_type in ['eeg', 'meg', 'eog', 'ecg']:
                    if ch_type in available_types:
                        reject_dict[ch_type] = threshold
                
                if reject_dict:  # Only apply if we have relevant channel types
                    epochs.drop_bad(reject=reject_dict, flat=config.flat_criteria)
                
            elif criterion == EpochRejectionCriteria.PEAK_TO_PEAK:
                # Peak-to-peak rejection
                data = epochs.get_data()
                pp_values = np.max(data, axis=2) - np.min(data, axis=2)
                bad_epochs = np.any(pp_values > threshold, axis=1)
                epochs.drop(np.where(bad_epochs)[0])
                
            elif criterion == EpochRejectionCriteria.GRADIENT:
                # Gradient-based rejection
                data = epochs.get_data()
                gradients = np.abs(np.diff(data, axis=2))
                max_gradients = np.max(gradients, axis=2)
                bad_epochs = np.any(max_gradients > threshold, axis=1)
                epochs.drop(np.where(bad_epochs)[0])
                
            elif criterion == EpochRejectionCriteria.VARIANCE:
                # Variance-based rejection
                data = epochs.get_data()
                variances = np.var(data, axis=2)
                bad_epochs = np.any(variances > threshold, axis=1)
                epochs.drop(np.where(bad_epochs)[0])
                
            elif criterion == EpochRejectionCriteria.KURTOSIS:
                # Kurtosis-based rejection
                data = epochs.get_data()
                kurtosis_values = stats.kurtosis(data, axis=2)
                bad_epochs = np.any(np.abs(kurtosis_values) > threshold, axis=1)
                epochs.drop(np.where(bad_epochs)[0])
            
            n_after = len(epochs)
            n_rejected = n_before - n_after
            
            if n_rejected > 0:
                rejection_log.append({
                    'criterion': criterion.value,
                    'threshold': threshold,
                    'n_rejected': n_rejected,
                    'rejection_rate': n_rejected / n_epochs_initial
                })
        
        self.rejection_log_ = rejection_log
        return epochs
    
    def _apply_windowing(self, epochs: mne.Epochs, window_function: str) -> mne.Epochs:
        """Apply windowing function to epochs"""
        
        n_times = epochs.get_data().shape[2]
        
        if window_function == 'hann':
            window = np.hanning(n_times)
        elif window_function == 'hamming':
            window = np.hamming(n_times)
        elif window_function == 'blackman':
            window = np.blackman(n_times)
        elif window_function == 'kaiser':
            window = np.kaiser(n_times, beta=8.6)
        else:
            return epochs  # No windowing
        
        # Apply window to each epoch and channel
        epochs._data *= window[np.newaxis, np.newaxis, :]
        
        return epochs
    
    def _apply_segment_quality_criteria(self, epochs: mne.Epochs, config: SegmentationConfig) -> mne.Epochs:
        """Apply quality criteria to fixed-length segments"""
        
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        bad_epochs = []
        
        for epoch_idx in range(n_epochs):
            epoch_data = data[epoch_idx]
            
            # Check amplitude threshold
            if np.any(np.abs(epoch_data) > config.max_amplitude):
                bad_epochs.append(epoch_idx)
                continue
            
            # Check gradient threshold
            gradients = np.abs(np.diff(epoch_data, axis=1))
            if np.any(gradients > config.max_gradient):
                bad_epochs.append(epoch_idx)
                continue
            
            # Check minimum good channels
            channel_ok = np.sum(np.max(np.abs(epoch_data), axis=1) < config.max_amplitude)
            if channel_ok / n_channels < config.min_good_channels:
                bad_epochs.append(epoch_idx)
        
        if bad_epochs:
            epochs.drop(bad_epochs)
        
        return epochs
    
    def _calculate_quality_metrics(self, epochs: mne.Epochs, config: EpochingConfig) -> EpochQualityMetrics:
        """Calculate comprehensive quality metrics for epochs"""
        
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Basic counts
        n_epochs_total = len(epochs.selection) + len(epochs.drop_log)
        n_epochs_good = len(epochs)
        n_epochs_rejected = n_epochs_total - n_epochs_good
        rejection_rate = n_epochs_rejected / n_epochs_total if n_epochs_total > 0 else 0
        
        # Rejection reasons
        rejection_reasons = {}
        for log_entry in self.rejection_log_:
            rejection_reasons[log_entry['criterion']] = log_entry['n_rejected']
        
        # Signal quality metrics
        mean_amplitude = np.mean(np.abs(data))
        std_amplitude = np.std(data)
        
        # Estimate SNR (ratio of signal variance to noise variance)
        # Use baseline period as noise estimate
        if epochs.baseline is not None:
            baseline_data = epochs.copy().crop(
                tmin=epochs.baseline[0], 
                tmax=epochs.baseline[1]
            ).get_data()
            noise_var = np.var(baseline_data)
            signal_var = np.var(data)
            snr_estimate = signal_var / noise_var if noise_var > 0 else np.inf
        else:
            snr_estimate = np.nan
        
        # Channel-wise rejection rates
        channel_rejection_rates = {}
        worst_channels = []
        
        # This would need more detailed tracking of per-channel rejections
        # For now, provide placeholder
        for ch_name in epochs.ch_names:
            channel_rejection_rates[ch_name] = 0.0
        
        return EpochQualityMetrics(
            n_epochs_total=n_epochs_total,
            n_epochs_good=n_epochs_good,
            n_epochs_rejected=n_epochs_rejected,
            rejection_rate=rejection_rate,
            rejection_reasons=rejection_reasons,
            mean_amplitude=mean_amplitude,
            std_amplitude=std_amplitude,
            snr_estimate=snr_estimate,
            channel_rejection_rates=channel_rejection_rates,
            worst_channels=worst_channels
        )
    
    def get_epoch_conditions(self, epochs: mne.Epochs) -> Dict[str, List[int]]:
        """
        Get epoch indices organized by condition/event type.
        
        Args:
            epochs: MNE Epochs object
            
        Returns:
            Dictionary mapping condition names to epoch indices
        """
        conditions = {}
        
        for event_name, event_id in epochs.event_id.items():
            # Find epochs with this event ID
            condition_epochs = [i for i, event in enumerate(epochs.events) 
                              if event[2] == event_id]
            conditions[event_name] = condition_epochs
        
        return conditions
    
    def export_epochs_info(self, epochs: mne.Epochs) -> Dict[str, Any]:
        """
        Export comprehensive information about epochs.
        
        Args:
            epochs: MNE Epochs object
            
        Returns:
            Dictionary with epoch information
        """
        info = {
            'n_epochs': len(epochs),
            'n_channels': len(epochs.ch_names),
            'n_times': len(epochs.times),
            'sampling_frequency': epochs.info['sfreq'],
            'time_range': (epochs.tmin, epochs.tmax),
            'baseline': epochs.baseline,
            'event_id': epochs.event_id,
            'ch_names': epochs.ch_names,
            'times': epochs.times.tolist(),
            'quality_metrics': self.quality_metrics_.__dict__ if self.quality_metrics_ else None,
            'rejection_log': self.rejection_log_
        }
        
        return info