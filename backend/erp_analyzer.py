#!/usr/bin/env python3
"""
Event-Related Potential (ERP) Analysis System
============================================

Phase 3.2: Comprehensive ERP analysis with advanced statistical capabilities.
Provides tools for ERP computation, grand averaging, peak detection, 
area measurements, difference waves, and statistical comparisons.

Features:
- Average ERP computation with robust statistics
- Grand average across subjects with confidence intervals
- Automatic and manual peak detection with multiple algorithms
- Area under curve analysis with flexible time windows
- Difference wave computation and analysis
- Statistical comparison of ERPs (parametric and non-parametric)
- Topographic analysis of ERP components
- Export capabilities for publication-ready figures

Author: porfanid
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats, signal, integrate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.epochs import BaseEpochs


class PeakDetectionMethod(Enum):
    """Peak detection methods"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    TEMPLATE_MATCHING = "template_matching"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


class ERPComponent(Enum):
    """Common ERP components"""
    P1 = "P1"
    N1 = "N1"  
    P2 = "P2"
    N2 = "N2"
    P3 = "P3"
    N400 = "N400"
    P600 = "P600"
    MMN = "MMN"  # Mismatch Negativity
    CNV = "CNV"  # Contingent Negative Variation
    CUSTOM = "custom"


class StatisticalTest(Enum):
    """Statistical tests for ERP comparison"""
    TTEST_PAIRED = "ttest_paired"
    TTEST_INDEPENDENT = "ttest_independent" 
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    PERMUTATION_TEST = "permutation_test"
    CLUSTER_PERMUTATION = "cluster_permutation"


@dataclass
class ERPConfig:
    """Configuration for ERP analysis"""
    
    # Averaging parameters
    method: str = 'mean'  # 'mean', 'median', 'trimmed_mean'
    trim_percent: float = 0.1  # For trimmed mean
    confidence_level: float = 0.95  # For confidence intervals
    
    # Peak detection
    peak_detection_method: PeakDetectionMethod = PeakDetectionMethod.AUTOMATIC
    component_windows: Dict[ERPComponent, Tuple[float, float]] = None
    
    # Baseline correction for ERPs
    baseline_window: Tuple[float, float] = (-0.1, 0.0)
    
    # Smoothing
    apply_smoothing: bool = False
    smoothing_window: float = 0.01  # seconds
    
    # Export settings
    export_format: str = 'png'
    dpi: int = 300
    
    def __post_init__(self):
        """Initialize default component windows"""
        if self.component_windows is None:
            self.component_windows = {
                ERPComponent.P1: (0.08, 0.12),
                ERPComponent.N1: (0.12, 0.20),  
                ERPComponent.P2: (0.15, 0.25),
                ERPComponent.N2: (0.20, 0.35),
                ERPComponent.P3: (0.30, 0.60),
                ERPComponent.N400: (0.30, 0.50),
                ERPComponent.P600: (0.50, 0.70),
                ERPComponent.MMN: (0.10, 0.25),
                ERPComponent.CNV: (-0.50, 0.0)
            }


@dataclass 
class PeakInfo:
    """Information about detected ERP peaks"""
    
    component: ERPComponent
    latency: float  # Peak latency in seconds
    amplitude: float  # Peak amplitude in µV
    channel: str  # Channel name
    confidence: float  # Detection confidence (0-1)
    
    # Additional metrics
    onset_latency: Optional[float] = None
    offset_latency: Optional[float] = None
    area_under_curve: Optional[float] = None
    peak_width: Optional[float] = None


@dataclass
class ERPStatistics:
    """Statistical results for ERP analysis"""
    
    # Basic descriptive statistics
    mean_amplitude: float
    std_amplitude: float
    sem_amplitude: float  # Standard error of the mean
    confidence_interval: Tuple[float, float]
    
    # Peak statistics
    peak_info: List[PeakInfo]
    
    # Area measurements
    area_measurements: Dict[str, float]
    
    # Signal quality
    snr: float
    reliability: float  # Test-retest reliability estimate


@dataclass
class ERPComparison:
    """Results of statistical comparison between ERPs"""
    
    condition1: str
    condition2: str
    test_type: StatisticalTest
    
    # Statistical results
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Time-resolved results (for cluster permutation)
    significant_timepoints: Optional[np.ndarray] = None
    cluster_info: Optional[List[Dict]] = None


class ERPAnalyzer:
    """
    Comprehensive Event-Related Potential (ERP) analysis system.
    
    This class provides advanced ERP analysis capabilities including:
    - Robust ERP averaging with multiple methods
    - Grand averaging across subjects
    - Automatic and manual peak detection
    - Area under curve analysis
    - Statistical comparisons between conditions
    - Difference wave analysis
    - Topographic mapping of ERP components
    """
    
    def __init__(self, config: ERPConfig = None):
        """Initialize the ERP analyzer"""
        self.config = config or ERPConfig()
        self.erp_data_ = {}
        self.grand_averages_ = {}
        self.statistics_ = {}
        self.comparisons_ = {}
    
    def compute_erp(self, epochs: mne.Epochs, condition_name: str = None) -> mne.Evoked:
        """
        Compute Event-Related Potential from epochs.
        
        Args:
            epochs: MNE Epochs object
            condition_name: Name for this condition
            
        Returns:
            MNE Evoked object with ERP data
        """
        try:
            # Apply baseline correction if specified
            if self.config.baseline_window:
                epochs_copy = epochs.copy()
                epochs_copy.apply_baseline(self.config.baseline_window)
            else:
                epochs_copy = epochs.copy()
            
            # Compute ERP based on method
            if self.config.method == 'mean':
                erp = epochs_copy.average()
            elif self.config.method == 'median':
                # Compute median ERP
                data = epochs_copy.get_data()
                median_data = np.median(data, axis=0)
                
                erp = mne.EvokedArray(
                    median_data, 
                    epochs_copy.info,
                    tmin=epochs_copy.tmin,
                    comment=f'median_{condition_name or "condition"}'
                )
            elif self.config.method == 'trimmed_mean':
                # Compute trimmed mean ERP
                data = epochs_copy.get_data()
                n_epochs = data.shape[0]
                n_trim = int(n_epochs * self.config.trim_percent / 2)
                
                if n_trim > 0:
                    # Sort along epoch dimension and trim
                    sorted_data = np.sort(data, axis=0)
                    trimmed_data = sorted_data[n_trim:-n_trim] if n_trim > 0 else sorted_data
                    mean_data = np.mean(trimmed_data, axis=0)
                else:
                    mean_data = np.mean(data, axis=0)
                
                erp = mne.EvokedArray(
                    mean_data,
                    epochs_copy.info, 
                    tmin=epochs_copy.tmin,
                    comment=f'trimmed_mean_{condition_name or "condition"}'
                )
            else:
                raise ValueError(f"Unknown ERP method: {self.config.method}")
            
            # Apply smoothing if requested
            if self.config.apply_smoothing:
                erp = self._smooth_erp(erp, self.config.smoothing_window)
            
            # Store ERP data
            if condition_name:
                self.erp_data_[condition_name] = erp
                
                # Compute statistics
                self.statistics_[condition_name] = self._compute_erp_statistics(epochs_copy, erp)
            
            return erp
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute ERP: {str(e)}")
    
    def compute_grand_average(self, erp_list: List[mne.Evoked], 
                            condition_name: str = None) -> mne.Evoked:
        """
        Compute grand average ERP across subjects.
        
        Args:
            erp_list: List of MNE Evoked objects (one per subject)
            condition_name: Name for this condition
            
        Returns:
            Grand average MNE Evoked object
        """
        try:
            # Ensure all ERPs have same structure
            self._validate_erp_compatibility(erp_list)
            
            # Stack ERP data
            erp_data = np.stack([erp.data for erp in erp_list], axis=0)
            
            # Compute grand average
            if self.config.method == 'mean':
                grand_avg_data = np.mean(erp_data, axis=0)
            elif self.config.method == 'median':  
                grand_avg_data = np.median(erp_data, axis=0)
            elif self.config.method == 'trimmed_mean':
                n_subjects = erp_data.shape[0]
                n_trim = int(n_subjects * self.config.trim_percent / 2)
                
                if n_trim > 0:
                    sorted_data = np.sort(erp_data, axis=0)
                    trimmed_data = sorted_data[n_trim:-n_trim]
                    grand_avg_data = np.mean(trimmed_data, axis=0)
                else:
                    grand_avg_data = np.mean(erp_data, axis=0)
            
            # Create grand average evoked object
            grand_avg = mne.EvokedArray(
                grand_avg_data,
                erp_list[0].info,
                tmin=erp_list[0].tmin,
                comment=f'grand_avg_{condition_name or "condition"}'
            )
            
            # Compute confidence intervals
            grand_avg = self._add_confidence_intervals(grand_avg, erp_data)
            
            if condition_name:
                self.grand_averages_[condition_name] = grand_avg
            
            return grand_avg
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute grand average: {str(e)}")
    
    def detect_peaks(self, erp: mne.Evoked, components: List[ERPComponent] = None,
                    channels: List[str] = None) -> List[PeakInfo]:
        """
        Detect ERP peaks and components.
        
        Args:
            erp: MNE Evoked object
            components: List of ERP components to detect
            channels: List of channels to analyze
            
        Returns:
            List of detected peaks with information
        """
        try:
            if components is None:
                components = list(self.config.component_windows.keys())
            
            if channels is None:
                channels = erp.ch_names
            
            detected_peaks = []
            
            for component in components:
                if component not in self.config.component_windows:
                    continue
                    
                time_window = self.config.component_windows[component]
                
                for channel in channels:
                    if channel not in erp.ch_names:
                        continue
                    
                    peak_info = self._detect_component_peak(
                        erp, component, channel, time_window
                    )
                    
                    if peak_info:
                        detected_peaks.append(peak_info)
            
            return detected_peaks
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect peaks: {str(e)}")
    
    def compute_area_under_curve(self, erp: mne.Evoked, time_windows: Dict[str, Tuple[float, float]],
                               channels: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute area under curve for specified time windows.
        
        Args:
            erp: MNE Evoked object
            time_windows: Dictionary mapping window names to (tmin, tmax) tuples
            channels: List of channels to analyze
            
        Returns:
            Dictionary with AUC values per window and channel
        """
        try:
            if channels is None:
                channels = erp.ch_names
            
            auc_results = {}
            
            for window_name, (tmin, tmax) in time_windows.items():
                auc_results[window_name] = {}
                
                # Extract data for time window
                erp_cropped = erp.copy().crop(tmin=tmin, tmax=tmax)
                times = erp_cropped.times
                
                for channel in channels:
                    if channel not in erp.ch_names:
                        continue
                    
                    ch_idx = erp.ch_names.index(channel)
                    data = erp_cropped.data[ch_idx, :]
                    
                    # Compute area under curve using trapezoidal integration
                    auc = np.trapz(data, times)
                    auc_results[window_name][channel] = auc
            
            return auc_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute area under curve: {str(e)}")
    
    def compute_difference_wave(self, erp1: mne.Evoked, erp2: mne.Evoked,
                              name: str = None) -> mne.Evoked:
        """
        Compute difference wave between two ERPs.
        
        Args:
            erp1: First ERP (minuend)
            erp2: Second ERP (subtrahend)  
            name: Name for the difference wave
            
        Returns:
            Difference wave as MNE Evoked object
        """
        try:
            # Ensure ERPs are compatible
            self._validate_erp_compatibility([erp1, erp2])
            
            # Compute difference
            diff_data = erp1.data - erp2.data
            
            # Create difference wave evoked object
            diff_wave = mne.EvokedArray(
                diff_data,
                erp1.info,
                tmin=erp1.tmin,
                comment=name or f'{erp1.comment}_minus_{erp2.comment}'
            )
            
            return diff_wave
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute difference wave: {str(e)}")
    
    def compare_erps(self, erp1_data: np.ndarray, erp2_data: np.ndarray,
                    condition1: str, condition2: str, 
                    test_type: StatisticalTest = StatisticalTest.TTEST_PAIRED) -> ERPComparison:
        """
        Statistical comparison between two ERP conditions.
        
        Args:
            erp1_data: ERP data for condition 1 (n_subjects, n_channels, n_times)
            erp2_data: ERP data for condition 2 (n_subjects, n_channels, n_times)
            condition1: Name of first condition
            condition2: Name of second condition
            test_type: Type of statistical test
            
        Returns:
            Statistical comparison results
        """
        try:
            if test_type == StatisticalTest.TTEST_PAIRED:
                # Paired t-test
                statistic, p_value = stats.ttest_rel(
                    erp1_data.reshape(erp1_data.shape[0], -1),
                    erp2_data.reshape(erp2_data.shape[0], -1),
                    axis=0
                )
                
                # Cohen's d for effect size
                diff = erp1_data - erp2_data
                effect_size = np.mean(diff) / np.std(diff, ddof=1)
                
            elif test_type == StatisticalTest.TTEST_INDEPENDENT:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(
                    erp1_data.reshape(erp1_data.shape[0], -1),
                    erp2_data.reshape(erp2_data.shape[0], -1),
                    axis=0
                )
                
                # Cohen's d for independent samples
                pooled_std = np.sqrt(
                    ((erp1_data.shape[0] - 1) * np.var(erp1_data, axis=0, ddof=1) +
                     (erp2_data.shape[0] - 1) * np.var(erp2_data, axis=0, ddof=1)) /
                    (erp1_data.shape[0] + erp2_data.shape[0] - 2)
                )
                effect_size = (np.mean(erp1_data, axis=0) - np.mean(erp2_data, axis=0)) / pooled_std
                
            elif test_type == StatisticalTest.WILCOXON:
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(
                    erp1_data.reshape(erp1_data.shape[0], -1),
                    erp2_data.reshape(erp2_data.shape[0], -1),
                    axis=0
                )
                
                # Rank-biserial correlation as effect size
                effect_size = statistic / (erp1_data.shape[0] * (erp1_data.shape[0] + 1) / 2)
                
            elif test_type == StatisticalTest.MANN_WHITNEY:
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(
                    erp1_data.reshape(erp1_data.shape[0], -1),
                    erp2_data.reshape(erp2_data.shape[0], -1),
                    axis=0
                )
                
                # Effect size for Mann-Whitney
                n1, n2 = erp1_data.shape[0], erp2_data.shape[0]
                effect_size = (statistic - n1 * n2 / 2) / (n1 * n2)
                
            elif test_type == StatisticalTest.PERMUTATION_TEST:
                # Permutation test
                statistic, p_value, effect_size = self._permutation_test(erp1_data, erp2_data)
                
            else:
                raise ValueError(f"Unsupported statistical test: {test_type}")
            
            # Convert arrays to scalars for summary statistics
            if isinstance(statistic, np.ndarray):
                statistic = np.mean(statistic)
            if isinstance(p_value, np.ndarray):
                p_value = np.mean(p_value)
            if isinstance(effect_size, np.ndarray):
                effect_size = np.mean(effect_size)
            
            comparison = ERPComparison(
                condition1=condition1,
                condition2=condition2,
                test_type=test_type,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size
            )
            
            # Store comparison
            comparison_key = f"{condition1}_vs_{condition2}_{test_type.value}"
            self.comparisons_[comparison_key] = comparison
            
            return comparison
            
        except Exception as e:
            raise RuntimeError(f"Failed to compare ERPs: {str(e)}")
    
    def _smooth_erp(self, erp: mne.Evoked, window_size: float) -> mne.Evoked:
        """Apply smoothing to ERP data"""
        
        # Convert window size from seconds to samples
        sfreq = erp.info['sfreq']
        window_samples = int(window_size * sfreq)
        
        if window_samples < 3:
            return erp  # Too small to smooth
        
        # Apply Gaussian smoothing
        erp_smooth = erp.copy()
        for ch_idx in range(erp_smooth.data.shape[0]):
            erp_smooth.data[ch_idx, :] = signal.gaussian_filter1d(
                erp_smooth.data[ch_idx, :], 
                sigma=window_samples/3
            )
        
        return erp_smooth
    
    def _validate_erp_compatibility(self, erp_list: List[mne.Evoked]):
        """Validate that ERPs have compatible structure"""
        
        if len(erp_list) < 2:
            return
        
        reference = erp_list[0]
        for i, erp in enumerate(erp_list[1:], 1):
            if erp.data.shape != reference.data.shape:
                raise ValueError(f"ERP {i} has incompatible shape: {erp.data.shape} vs {reference.data.shape}")
            
            if not np.allclose(erp.times, reference.times):
                raise ValueError(f"ERP {i} has incompatible time points")
            
            if erp.ch_names != reference.ch_names:
                raise ValueError(f"ERP {i} has incompatible channel names")
    
    def _add_confidence_intervals(self, grand_avg: mne.Evoked, erp_data: np.ndarray) -> mne.Evoked:
        """Add confidence interval information to grand average"""
        
        n_subjects = erp_data.shape[0]
        alpha = 1 - self.config.confidence_level
        
        # Compute standard error
        sem = np.std(erp_data, axis=0, ddof=1) / np.sqrt(n_subjects)
        
        # Compute confidence intervals using t-distribution
        t_value = stats.t.ppf(1 - alpha/2, n_subjects - 1)
        ci_lower = grand_avg.data - t_value * sem
        ci_upper = grand_avg.data + t_value * sem
        
        # Store CI information (this would need custom metadata handling)
        grand_avg.metadata = {
            'confidence_level': self.config.confidence_level,
            'n_subjects': n_subjects,
            'standard_error': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        return grand_avg
    
    def _detect_component_peak(self, erp: mne.Evoked, component: ERPComponent,
                             channel: str, time_window: Tuple[float, float]) -> Optional[PeakInfo]:
        """Detect a specific ERP component peak in a channel"""
        
        try:
            # Extract data for channel and time window
            ch_idx = erp.ch_names.index(channel)
            erp_cropped = erp.copy().crop(tmin=time_window[0], tmax=time_window[1])
            data = erp_cropped.data[ch_idx, :]
            times = erp_cropped.times
            
            if self.config.peak_detection_method == PeakDetectionMethod.AUTOMATIC:
                # Find peaks/troughs based on component polarity
                if component.value.startswith('P'):  # Positive component
                    peaks, properties = find_peaks(data, prominence=np.std(data)*0.5)
                    if len(peaks) > 0:
                        peak_idx = peaks[np.argmax(data[peaks])]  # Highest peak
                else:  # Negative component  
                    peaks, properties = find_peaks(-data, prominence=np.std(data)*0.5)
                    if len(peaks) > 0:
                        peak_idx = peaks[np.argmax(-data[peaks])]  # Most negative peak
                
                if len(peaks) == 0:
                    return None
                
                peak_latency = times[peak_idx]
                peak_amplitude = data[peak_idx]
                confidence = min(1.0, properties['prominences'][np.where(peaks == peak_idx)[0][0]] / np.std(data))
                
            elif self.config.peak_detection_method == PeakDetectionMethod.ADAPTIVE_THRESHOLD:
                # Adaptive threshold based on data statistics
                threshold = np.mean(np.abs(data)) + 2 * np.std(data)
                
                if component.value.startswith('P'):
                    candidate_indices = np.where(data > threshold)[0]
                else:
                    candidate_indices = np.where(data < -threshold)[0]
                
                if len(candidate_indices) == 0:
                    return None
                
                # Select peak closest to expected latency
                expected_latency = np.mean(time_window)
                expected_idx = np.argmin(np.abs(times - expected_latency))
                peak_idx = candidate_indices[np.argmin(np.abs(candidate_indices - expected_idx))]
                
                peak_latency = times[peak_idx]
                peak_amplitude = data[peak_idx]
                confidence = min(1.0, np.abs(peak_amplitude) / threshold)
                
            else:
                # Manual detection would require user interaction
                return None
            
            # Compute additional metrics
            area_under_curve = np.trapz(np.abs(data), times)
            
            return PeakInfo(
                component=component,
                latency=peak_latency,
                amplitude=peak_amplitude,
                channel=channel,
                confidence=confidence,
                area_under_curve=area_under_curve
            )
            
        except Exception as e:
            print(f"Failed to detect peak for {component.value} in {channel}: {str(e)}")
            return None
    
    def _compute_erp_statistics(self, epochs: mne.Epochs, erp: mne.Evoked) -> ERPStatistics:
        """Compute comprehensive statistics for ERP"""
        
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Basic descriptive statistics
        mean_amplitude = np.mean(erp.data)
        std_amplitude = np.std(data)
        sem_amplitude = std_amplitude / np.sqrt(n_epochs)
        
        # Confidence interval
        alpha = 1 - self.config.confidence_level
        t_value = stats.t.ppf(1 - alpha/2, n_epochs - 1)
        ci_lower = mean_amplitude - t_value * sem_amplitude
        ci_upper = mean_amplitude + t_value * sem_amplitude
        
        # Detect peaks
        peak_info = self.detect_peaks(erp)
        
        # Compute area measurements
        time_windows = {
            'early': (erp.tmin, 0.2),
            'late': (0.2, erp.tmax),
            'full': (erp.tmin, erp.tmax)
        }
        area_measurements = {}
        auc_results = self.compute_area_under_curve(erp, time_windows)
        for window, channels in auc_results.items():
            area_measurements[window] = np.mean(list(channels.values()))
        
        # Estimate SNR
        if epochs.baseline is not None:
            baseline_data = epochs.copy().crop(
                tmin=epochs.baseline[0],
                tmax=epochs.baseline[1] 
            ).get_data()
            noise_power = np.var(baseline_data)
            signal_power = np.var(data)
            snr = signal_power / noise_power if noise_power > 0 else np.inf
        else:
            snr = np.nan
        
        # Reliability estimate (split-half correlation)
        if n_epochs >= 4:
            half1 = data[:n_epochs//2].mean(axis=0)
            half2 = data[n_epochs//2:].mean(axis=0)
            reliability = np.corrcoef(half1.flatten(), half2.flatten())[0, 1]
            # Spearman-Brown correction
            reliability = 2 * reliability / (1 + reliability)
        else:
            reliability = np.nan
        
        return ERPStatistics(
            mean_amplitude=mean_amplitude,
            std_amplitude=std_amplitude,
            sem_amplitude=sem_amplitude,
            confidence_interval=(ci_lower, ci_upper),
            peak_info=peak_info,
            area_measurements=area_measurements,
            snr=snr,
            reliability=reliability
        )
    
    def _permutation_test(self, data1: np.ndarray, data2: np.ndarray, 
                         n_permutations: int = 1000) -> Tuple[float, float, float]:
        """Perform permutation test between two datasets"""
        
        # Flatten data for analysis
        flat1 = data1.reshape(data1.shape[0], -1)
        flat2 = data2.reshape(data2.shape[0], -1)
        
        # Observed difference
        observed_diff = np.mean(flat1) - np.mean(flat2)
        
        # Combine data
        combined = np.vstack([flat1, flat2])
        n1, n2 = flat1.shape[0], flat2.shape[0]
        
        # Permutation test
        perm_diffs = []
        for _ in range(n_permutations):
            # Shuffle combined data
            shuffled = combined.copy()
            np.random.shuffle(shuffled)
            
            # Split into two groups
            perm1 = shuffled[:n1]
            perm2 = shuffled[n1:]
            
            # Compute difference
            perm_diff = np.mean(perm1) - np.mean(perm2)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        
        # P-value (two-tailed)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.var(flat1, ddof=1) + np.var(flat2, ddof=1)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0
        
        return observed_diff, p_value, effect_size
    
    def export_results(self, output_dir: str = "erp_results"):
        """
        Export ERP analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export ERP data
        erp_summary = []
        for condition, erp in self.erp_data_.items():
            erp_info = {
                'condition': condition,
                'n_channels': len(erp.ch_names),
                'n_times': len(erp.times),
                'tmin': erp.tmin,
                'tmax': erp.tmax,
                'sampling_rate': erp.info['sfreq']
            }
            
            if condition in self.statistics_:
                stats = self.statistics_[condition]
                erp_info.update({
                    'mean_amplitude': stats.mean_amplitude,
                    'std_amplitude': stats.std_amplitude,
                    'snr': stats.snr,
                    'reliability': stats.reliability,
                    'n_peaks_detected': len(stats.peak_info)
                })
            
            erp_summary.append(erp_info)
        
        # Save summary
        pd.DataFrame(erp_summary).to_csv(
            os.path.join(output_dir, 'erp_summary.csv'),
            index=False
        )
        
        # Export comparisons
        if self.comparisons_:
            comp_summary = []
            for comp_name, comp in self.comparisons_.items():
                comp_summary.append({
                    'comparison': comp_name,
                    'condition1': comp.condition1,
                    'condition2': comp.condition2,
                    'test_type': comp.test_type.value,
                    'statistic': comp.statistic,
                    'p_value': comp.p_value,
                    'effect_size': comp.effect_size
                })
            
            pd.DataFrame(comp_summary).to_csv(
                os.path.join(output_dir, 'erp_comparisons.csv'),
                index=False
            )
        
        print(f"ERP analysis results exported to {output_dir}")