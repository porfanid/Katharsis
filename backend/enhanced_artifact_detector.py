#!/usr/bin/env python3
"""
Enhanced Artifact Detection System - Phase 2
============================================

Advanced artifact detection capabilities:
- Multiple detection algorithms for different artifact types
- Statistical outlier detection
- Pattern-based artifact recognition
- Integration with enhanced ICA classification

Author: porfanid
Version: 2.0
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass

import mne
import numpy as np
from scipy import stats, signal
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class ArtifactType(Enum):
    """Types of artifacts that can be detected"""
    EYE_BLINK = "eye_blink"
    EYE_MOVEMENT = "eye_movement"
    MUSCLE = "muscle"
    MOVEMENT = "movement"
    HEART = "heart"
    LINE_NOISE = "line_noise"
    CHANNEL_JUMP = "channel_jump"
    FLATLINE = "flatline"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class ArtifactDetection:
    """Result of artifact detection"""
    artifact_type: ArtifactType
    channel_idx: int
    channel_name: str
    start_sample: int
    end_sample: int
    confidence: float
    severity: float
    features: Dict[str, float]
    description: str


@dataclass
class DetectionConfig:
    """Configuration for artifact detection"""
    # Detection thresholds
    eye_blink_threshold: float = 100e-6  # 100 μV
    muscle_threshold: float = 50e-6       # 50 μV
    movement_threshold: float = 200e-6    # 200 μV
    heart_threshold: float = 20e-6        # 20 μV
    
    # Statistical thresholds
    outlier_z_threshold: float = 4.0
    flatline_threshold: float = 1e-6     # 1 μV
    jump_threshold: float = 100e-6       # 100 μV
    
    # Frequency band thresholds (Hz)
    muscle_freq_range: Tuple[float, float] = (30.0, 100.0)
    line_noise_freqs: List[float] = [50.0, 60.0]
    heart_freq_range: Tuple[float, float] = (0.8, 2.0)
    
    # Detection windows (samples)
    min_artifact_duration: int = 10
    max_artifact_duration: int = 1000
    
    # Channel-specific settings
    frontal_channels: Optional[List[str]] = None
    temporal_channels: Optional[List[str]] = None
    
    # Advanced settings
    use_ica_classification: bool = True
    use_statistical_outliers: bool = True
    use_pattern_detection: bool = True


class EnhancedArtifactDetector:
    """
    Enhanced artifact detection system with multiple algorithms
    
    Features:
    - Multiple artifact type detection
    - Statistical outlier detection
    - Pattern-based recognition
    - Channel-specific analysis
    - Integration with ICA classification
    """
    
    def __init__(self, config: DetectionConfig = None):
        """Initialize enhanced artifact detector"""
        self.config = config or DetectionConfig()
        self.raw_data: Optional[mne.io.Raw] = None
        self.detections: List[ArtifactDetection] = []
        
        # Cached computations
        self._channel_statistics: Dict[str, Dict[str, float]] = {}
        self._frequency_features: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Default channel groups if not specified
        if self.config.frontal_channels is None:
            self.config.frontal_channels = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'F4']
        
        if self.config.temporal_channels is None:
            self.config.temporal_channels = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8']
    
    def detect_artifacts(self, raw: mne.io.Raw, ica_processor=None) -> Dict[str, Any]:
        """
        Comprehensive artifact detection
        
        Args:
            raw: Raw EEG data
            ica_processor: Optional enhanced ICA processor for component-based detection
            
        Returns:
            Dictionary with detection results and statistics
        """
        self.raw_data = raw.copy()
        self.detections = []
        
        try:
            # Pre-compute features
            self._compute_channel_statistics()
            self._compute_frequency_features()
            
            # Detect different types of artifacts
            self._detect_eye_blinks()
            self._detect_eye_movements()
            self._detect_muscle_artifacts()
            self._detect_movement_artifacts()
            self._detect_heart_artifacts()
            self._detect_line_noise()
            self._detect_channel_jumps()
            self._detect_flatlines()
            
            if self.config.use_statistical_outliers:
                self._detect_statistical_outliers()
            
            if self.config.use_pattern_detection:
                self._detect_patterns()
            
            # Integrate ICA-based detection if available
            if ica_processor and self.config.use_ica_classification:
                self._integrate_ica_detections(ica_processor)
            
            # Post-process detections
            self._merge_overlapping_detections()
            self._filter_detections()
            
            return self._generate_detection_summary()
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'n_detections': 0
            }
    
    def _compute_channel_statistics(self):
        """Compute basic statistics for each channel"""
        data = self.raw_data.get_data()
        ch_names = self.raw_data.ch_names
        
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]
            
            self._channel_statistics[ch_name] = {
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'min': np.min(ch_data),
                'max': np.max(ch_data),
                'range': np.ptp(ch_data),
                'rms': np.sqrt(np.mean(ch_data**2)),
                'kurtosis': stats.kurtosis(ch_data),
                'skewness': stats.skew(ch_data),
                'variance': np.var(ch_data)
            }
    
    def _compute_frequency_features(self):
        """Compute frequency domain features for each channel"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]
            
            # Compute power spectral density
            freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(1024, len(ch_data)//4))
            
            # Compute band powers
            delta_power = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
            theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
            beta_power = np.mean(psd[(freqs >= 12) & (freqs <= 30)])
            gamma_power = np.mean(psd[(freqs >= 30) & (freqs <= 100)])
            
            # Muscle band power
            muscle_power = np.mean(psd[(freqs >= self.config.muscle_freq_range[0]) & 
                                     (freqs <= self.config.muscle_freq_range[1])])
            
            # Line noise power
            line_noise_power = 0
            for line_freq in self.config.line_noise_freqs:
                freq_idx = np.argmin(np.abs(freqs - line_freq))
                line_noise_power += psd[freq_idx]
            
            # Heart rate band power  
            heart_power = np.mean(psd[(freqs >= self.config.heart_freq_range[0]) & 
                                    (freqs <= self.config.heart_freq_range[1])])
            
            self._frequency_features[ch_name] = {
                'freqs': freqs,
                'psd': psd,
                'delta_power': delta_power,
                'theta_power': theta_power,
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'gamma_power': gamma_power,
                'muscle_power': muscle_power,
                'line_noise_power': line_noise_power,
                'heart_power': heart_power,
                'total_power': np.sum(psd)
            }
    
    def _detect_eye_blinks(self):
        """Detect eye blink artifacts in frontal channels"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        # Focus on frontal channels
        frontal_ch_indices = [i for i, ch in enumerate(ch_names) 
                            if ch in self.config.frontal_channels]
        
        for ch_idx in frontal_ch_indices:
            ch_name = ch_names[ch_idx]
            ch_data = data[ch_idx, :]
            
            # Detect sharp positive deflections (typical of eye blinks)
            # Apply high-pass filter to emphasize blinks
            filtered_data = signal.filtfilt(*signal.butter(4, 1.0, btype='high', fs=sfreq), ch_data)
            
            # Find peaks above threshold
            peaks, properties = signal.find_peaks(
                filtered_data,
                height=self.config.eye_blink_threshold,
                distance=int(0.5 * sfreq),  # At least 0.5s between blinks
                width=5  # Minimum width
            )
            
            for peak_idx in peaks:
                # Determine blink boundaries
                peak_height = filtered_data[peak_idx]
                
                # Find start and end of blink
                start_idx = max(0, peak_idx - int(0.2 * sfreq))
                end_idx = min(len(ch_data), peak_idx + int(0.2 * sfreq))
                
                # Refine boundaries based on signal return to baseline
                baseline_level = np.median(filtered_data[max(0, peak_idx - int(sfreq)):
                                                       min(len(filtered_data), peak_idx + int(sfreq))])
                
                threshold_level = baseline_level + 0.3 * (peak_height - baseline_level)
                
                # Find actual start
                for i in range(peak_idx, start_idx, -1):
                    if filtered_data[i] < threshold_level:
                        start_idx = i
                        break
                
                # Find actual end
                for i in range(peak_idx, end_idx):
                    if filtered_data[i] < threshold_level:
                        end_idx = i
                        break
                
                # Calculate features
                duration = (end_idx - start_idx) / sfreq
                amplitude = np.max(filtered_data[start_idx:end_idx]) - baseline_level
                
                # Only keep reasonable blinks (50ms - 500ms duration)
                if 0.05 <= duration <= 0.5:
                    detection = ArtifactDetection(
                        artifact_type=ArtifactType.EYE_BLINK,
                        channel_idx=ch_idx,
                        channel_name=ch_name,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        confidence=min(amplitude / self.config.eye_blink_threshold, 1.0),
                        severity=amplitude / self.config.eye_blink_threshold,
                        features={
                            'amplitude': float(amplitude),
                            'duration': duration,
                            'peak_sample': int(peak_idx),
                            'baseline_level': float(baseline_level)
                        },
                        description=f"Eye blink: {amplitude*1e6:.1f} μV, {duration*1000:.0f}ms"
                    )
                    self.detections.append(detection)
    
    def _detect_eye_movements(self):
        """Detect horizontal and vertical eye movements"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        # Look for differential activity between left/right frontal channels
        left_channels = [i for i, ch in enumerate(ch_names) 
                        if any(l in ch for l in ['1', '3', '5', '7']) and 
                        any(f in ch for f in ['F', 'Fp'])]
        right_channels = [i for i, ch in enumerate(ch_names) 
                         if any(r in ch for r in ['2', '4', '6', '8']) and 
                         any(f in ch for f in ['F', 'Fp'])]
        
        if len(left_channels) > 0 and len(right_channels) > 0:
            # Average left and right frontal activity
            left_avg = np.mean(data[left_channels, :], axis=0)
            right_avg = np.mean(data[right_channels, :], axis=0)
            
            # Calculate differential signal
            diff_signal = left_avg - right_avg
            
            # Apply low-pass filter to focus on eye movements (< 10 Hz)
            filtered_diff = signal.filtfilt(*signal.butter(4, 10.0, btype='low', fs=sfreq), diff_signal)
            
            # Detect significant deviations from baseline
            diff_std = np.std(filtered_diff)
            threshold = 3 * diff_std
            
            # Find sustained deviations
            above_threshold = np.abs(filtered_diff) > threshold
            
            # Find continuous segments
            segments = self._find_continuous_segments(above_threshold)
            
            for start_idx, end_idx in segments:
                duration = (end_idx - start_idx) / sfreq
                
                # Only keep eye movements of reasonable duration (0.1 - 2.0 seconds)
                if 0.1 <= duration <= 2.0:
                    max_deviation = np.max(np.abs(filtered_diff[start_idx:end_idx]))
                    
                    # Determine dominant channel
                    left_activity = np.mean(np.abs(left_avg[start_idx:end_idx]))
                    right_activity = np.mean(np.abs(right_avg[start_idx:end_idx]))
                    
                    if left_activity > right_activity:
                        dominant_ch_idx = left_channels[0]
                        dominant_ch_name = ch_names[dominant_ch_idx]
                    else:
                        dominant_ch_idx = right_channels[0]
                        dominant_ch_name = ch_names[dominant_ch_idx]
                    
                    detection = ArtifactDetection(
                        artifact_type=ArtifactType.EYE_MOVEMENT,
                        channel_idx=dominant_ch_idx,
                        channel_name=dominant_ch_name,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        confidence=min(max_deviation / threshold, 1.0),
                        severity=max_deviation / threshold,
                        features={
                            'max_deviation': float(max_deviation),
                            'duration': duration,
                            'left_activity': float(left_activity),
                            'right_activity': float(right_activity)
                        },
                        description=f"Eye movement: {max_deviation*1e6:.1f} μV differential, {duration:.1f}s"
                    )
                    self.detections.append(detection)
    
    def _detect_muscle_artifacts(self):
        """Detect muscle artifacts using high-frequency power"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        # Focus on temporal and frontal channels
        target_channels = [i for i, ch in enumerate(ch_names) 
                          if ch in self.config.temporal_channels or ch in self.config.frontal_channels]
        
        for ch_idx in target_channels:
            ch_name = ch_names[ch_idx]
            ch_data = data[ch_idx, :]
            
            # Calculate sliding window high-frequency power
            window_size = int(0.5 * sfreq)  # 0.5 second windows
            step_size = int(0.1 * sfreq)    # 0.1 second steps
            
            muscle_power = []
            window_centers = []
            
            for i in range(0, len(ch_data) - window_size, step_size):
                window_data = ch_data[i:i + window_size]
                
                # Band-pass filter for muscle frequencies
                filtered_window = signal.filtfilt(
                    *signal.butter(4, self.config.muscle_freq_range, btype='band', fs=sfreq),
                    window_data
                )
                
                # Calculate RMS power
                rms_power = np.sqrt(np.mean(filtered_window**2))
                muscle_power.append(rms_power)
                window_centers.append(i + window_size // 2)
            
            muscle_power = np.array(muscle_power)
            
            # Detect periods of elevated muscle activity
            baseline_power = np.median(muscle_power)
            threshold_power = baseline_power + 3 * np.std(muscle_power)
            
            above_threshold = muscle_power > threshold_power
            
            # Find continuous segments
            segments = self._find_continuous_segments_from_indices(
                np.where(above_threshold)[0], window_centers, step_size
            )
            
            for start_idx, end_idx in segments:
                duration = (end_idx - start_idx) / sfreq
                
                # Only keep muscle artifacts of reasonable duration
                if 0.1 <= duration <= 5.0:
                    max_power = np.max(ch_data[start_idx:end_idx])
                    
                    detection = ArtifactDetection(
                        artifact_type=ArtifactType.MUSCLE,
                        channel_idx=ch_idx,
                        channel_name=ch_name,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        confidence=min((max_power - baseline_power) / threshold_power, 1.0),
                        severity=max_power / threshold_power,
                        features={
                            'max_power': float(max_power),
                            'baseline_power': float(baseline_power),
                            'duration': duration,
                            'frequency_range': self.config.muscle_freq_range
                        },
                        description=f"Muscle artifact: {max_power*1e6:.1f} μV RMS, {duration:.1f}s"
                    )
                    self.detections.append(detection)
    
    def _detect_movement_artifacts(self):
        """Detect large movement artifacts affecting multiple channels"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        # Calculate global field power (GFP)
        gfp = np.std(data, axis=0)
        
        # Detect periods of excessive GFP
        baseline_gfp = np.median(gfp)
        threshold_gfp = baseline_gfp + 5 * np.std(gfp)
        
        above_threshold = gfp > threshold_gfp
        segments = self._find_continuous_segments(above_threshold)
        
        for start_idx, end_idx in segments:
            duration = (end_idx - start_idx) / sfreq
            
            # Only keep movement artifacts of reasonable duration
            if 0.1 <= duration <= 10.0:
                max_gfp = np.max(gfp[start_idx:end_idx])
                
                # Find the channel with maximum deviation
                segment_data = data[:, start_idx:end_idx]
                max_deviations = np.max(np.abs(segment_data), axis=1)
                max_ch_idx = np.argmax(max_deviations)
                
                detection = ArtifactDetection(
                    artifact_type=ArtifactType.MOVEMENT,
                    channel_idx=max_ch_idx,
                    channel_name=ch_names[max_ch_idx],
                    start_sample=start_idx,
                    end_sample=end_idx,
                    confidence=min(max_gfp / threshold_gfp, 1.0),
                    severity=max_gfp / threshold_gfp,
                    features={
                        'max_gfp': float(max_gfp),
                        'baseline_gfp': float(baseline_gfp),
                        'duration': duration,
                        'affected_channels': int(np.sum(max_deviations > self.config.movement_threshold))
                    },
                    description=f"Movement artifact: GFP {max_gfp*1e6:.1f} μV, {duration:.1f}s"
                )
                self.detections.append(detection)
    
    def _detect_heart_artifacts(self):
        """Detect cardiac artifacts (ECG contamination)"""
        # Implementation for heart artifact detection
        # This would involve detecting periodic patterns consistent with heart rate
        pass
    
    def _detect_line_noise(self):
        """Detect line noise (50/60 Hz) artifacts"""
        # Implementation for line noise detection
        # This would analyze power at specific frequencies
        pass
    
    def _detect_channel_jumps(self):
        """Detect sudden jumps in channel values"""
        data = self.raw_data.get_data()
        ch_names = self.raw_data.ch_names
        
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]
            
            # Calculate first derivative (differences)
            diff_data = np.diff(ch_data)
            
            # Detect jumps
            jump_threshold = self.config.jump_threshold
            jump_indices = np.where(np.abs(diff_data) > jump_threshold)[0]
            
            for jump_idx in jump_indices:
                detection = ArtifactDetection(
                    artifact_type=ArtifactType.CHANNEL_JUMP,
                    channel_idx=ch_idx,
                    channel_name=ch_name,
                    start_sample=jump_idx,
                    end_sample=jump_idx + 1,
                    confidence=min(np.abs(diff_data[jump_idx]) / jump_threshold, 1.0),
                    severity=np.abs(diff_data[jump_idx]) / jump_threshold,
                    features={
                        'jump_amplitude': float(np.abs(diff_data[jump_idx])),
                        'jump_direction': int(np.sign(diff_data[jump_idx]))
                    },
                    description=f"Channel jump: {diff_data[jump_idx]*1e6:.1f} μV"
                )
                self.detections.append(detection)
    
    def _detect_flatlines(self):
        """Detect flatline periods (channel disconnection)"""
        data = self.raw_data.get_data()
        sfreq = self.raw_data.info['sfreq']
        ch_names = self.raw_data.ch_names
        
        min_duration = int(1.0 * sfreq)  # Minimum 1 second
        
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]
            
            # Find periods with very low variance
            window_size = int(0.5 * sfreq)
            step_size = int(0.1 * sfreq)
            
            for i in range(0, len(ch_data) - window_size, step_size):
                window_data = ch_data[i:i + window_size]
                variance = np.var(window_data)
                
                if variance < self.config.flatline_threshold**2:
                    # Extend the flatline period
                    start_idx = i
                    end_idx = i + window_size
                    
                    # Extend backwards
                    while start_idx > 0 and np.var(ch_data[start_idx-step_size:start_idx]) < self.config.flatline_threshold**2:
                        start_idx -= step_size
                    
                    # Extend forwards
                    while end_idx < len(ch_data) and np.var(ch_data[end_idx:end_idx+step_size]) < self.config.flatline_threshold**2:
                        end_idx += step_size
                    
                    duration = (end_idx - start_idx) / sfreq
                    
                    if duration >= 1.0:  # At least 1 second
                        detection = ArtifactDetection(
                            artifact_type=ArtifactType.FLATLINE,
                            channel_idx=ch_idx,
                            channel_name=ch_name,
                            start_sample=start_idx,
                            end_sample=end_idx,
                            confidence=1.0 - variance / self.config.flatline_threshold**2,
                            severity=1.0,
                            features={
                                'variance': float(variance),
                                'duration': duration,
                                'mean_level': float(np.mean(window_data))
                            },
                            description=f"Flatline: {duration:.1f}s, variance {variance*1e12:.1f} pV²"
                        )
                        self.detections.append(detection)
    
    def _detect_statistical_outliers(self):
        """Detect statistical outliers using multiple methods"""
        data = self.raw_data.get_data()
        ch_names = self.raw_data.ch_names
        
        # Z-score based outlier detection
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]
            
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(ch_data))
            outlier_indices = np.where(z_scores > self.config.outlier_z_threshold)[0]
            
            # Group consecutive outliers
            if len(outlier_indices) > 0:
                segments = self._find_continuous_segments_from_indices(outlier_indices, outlier_indices, 1)
                
                for start_idx, end_idx in segments:
                    max_z_score = np.max(z_scores[start_idx:end_idx])
                    
                    detection = ArtifactDetection(
                        artifact_type=ArtifactType.STATISTICAL_OUTLIER,
                        channel_idx=ch_idx,
                        channel_name=ch_name,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        confidence=min(max_z_score / self.config.outlier_z_threshold, 1.0),
                        severity=max_z_score / self.config.outlier_z_threshold,
                        features={
                            'max_z_score': float(max_z_score),
                            'n_outlier_samples': int(end_idx - start_idx)
                        },
                        description=f"Statistical outlier: z-score {max_z_score:.1f}"
                    )
                    self.detections.append(detection)
    
    def _detect_patterns(self):
        """Detect artifact patterns using machine learning"""
        # This would use more advanced pattern recognition
        # For now, we'll implement a simple version
        pass
    
    def _integrate_ica_detections(self, ica_processor):
        """Integrate artifact detections from ICA component classification"""
        if not hasattr(ica_processor, 'component_classifications'):
            return
        
        # Add detections based on ICA component classifications
        for classification in ica_processor.component_classifications:
            if classification.component_type.value in ['eye_blink', 'eye_movement', 'muscle', 'heart', 'line_noise']:
                # This would require mapping ICA components back to time-domain detections
                # For now, we'll skip this implementation
                pass
    
    def _find_continuous_segments(self, boolean_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments where boolean_array is True"""
        segments = []
        start_idx = None
        
        for i, value in enumerate(boolean_array):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        
        # Handle case where segment continues to end
        if start_idx is not None:
            segments.append((start_idx, len(boolean_array)))
        
        return segments
    
    def _find_continuous_segments_from_indices(self, indices: np.ndarray, 
                                             centers: np.ndarray, 
                                             step_size: int) -> List[Tuple[int, int]]:
        """Find continuous segments from sparse indices"""
        if len(indices) == 0:
            return []
        
        segments = []
        start_idx = centers[indices[0]]
        end_idx = start_idx
        
        for i in range(1, len(indices)):
            current_center = centers[indices[i]]
            
            # Check if this index is continuous with the previous
            if indices[i] - indices[i-1] <= 1:  # Allow small gaps
                end_idx = current_center
            else:
                # End current segment and start new one
                segments.append((start_idx, end_idx + step_size))
                start_idx = current_center
                end_idx = current_center
        
        # Add final segment
        segments.append((start_idx, end_idx + step_size))
        
        return segments
    
    def _merge_overlapping_detections(self):
        """Merge overlapping detections of the same type"""
        # Sort detections by start time
        self.detections.sort(key=lambda x: (x.channel_idx, x.start_sample))
        
        merged_detections = []
        
        for detection in self.detections:
            # Check if this detection overlaps with the last merged detection
            if (merged_detections and 
                merged_detections[-1].channel_idx == detection.channel_idx and
                merged_detections[-1].artifact_type == detection.artifact_type and
                merged_detections[-1].end_sample >= detection.start_sample):
                
                # Merge with previous detection
                last_detection = merged_detections[-1]
                last_detection.end_sample = max(last_detection.end_sample, detection.end_sample)
                last_detection.confidence = max(last_detection.confidence, detection.confidence)
                last_detection.severity = max(last_detection.severity, detection.severity)
                
            else:
                merged_detections.append(detection)
        
        self.detections = merged_detections
    
    def _filter_detections(self):
        """Filter detections based on duration and severity criteria"""
        filtered_detections = []
        
        for detection in self.detections:
            duration_samples = detection.end_sample - detection.start_sample
            
            # Apply duration filters
            if (duration_samples >= self.config.min_artifact_duration and 
                duration_samples <= self.config.max_artifact_duration):
                
                # Apply confidence threshold
                if detection.confidence > 0.3:  # Minimum confidence
                    filtered_detections.append(detection)
        
        self.detections = filtered_detections
    
    def _generate_detection_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of artifact detections"""
        if not self.detections:
            return {
                'success': True,
                'n_detections': 0,
                'artifact_types': {},
                'channels_affected': 0,
                'total_artifact_duration': 0.0
            }
        
        # Count detections by type
        type_counts = {}
        for artifact_type in ArtifactType:
            type_counts[artifact_type.value] = 0
        
        for detection in self.detections:
            type_counts[detection.artifact_type.value] += 1
        
        # Calculate statistics
        affected_channels = set(d.channel_name for d in self.detections)
        
        total_duration = sum((d.end_sample - d.start_sample) / self.raw_data.info['sfreq'] 
                           for d in self.detections)
        
        avg_confidence = np.mean([d.confidence for d in self.detections])
        avg_severity = np.mean([d.severity for d in self.detections])
        
        return {
            'success': True,
            'n_detections': len(self.detections),
            'artifact_types': type_counts,
            'channels_affected': len(affected_channels),
            'affected_channel_names': list(affected_channels),
            'total_artifact_duration': total_duration,
            'average_confidence': float(avg_confidence),
            'average_severity': float(avg_severity),
            'detections': [self._detection_to_dict(d) for d in self.detections]
        }
    
    def _detection_to_dict(self, detection: ArtifactDetection) -> Dict[str, Any]:
        """Convert detection to dictionary for serialization"""
        return {
            'artifact_type': detection.artifact_type.value,
            'channel_idx': detection.channel_idx,
            'channel_name': detection.channel_name,
            'start_sample': detection.start_sample,
            'end_sample': detection.end_sample,
            'start_time': detection.start_sample / self.raw_data.info['sfreq'],
            'end_time': detection.end_sample / self.raw_data.info['sfreq'],
            'confidence': detection.confidence,
            'severity': detection.severity,
            'features': detection.features,
            'description': detection.description
        }
    
    def get_detections_by_type(self, artifact_type: ArtifactType) -> List[ArtifactDetection]:
        """Get all detections of a specific type"""
        return [d for d in self.detections if d.artifact_type == artifact_type]
    
    def get_detections_by_channel(self, channel_name: str) -> List[ArtifactDetection]:
        """Get all detections for a specific channel"""
        return [d for d in self.detections if d.channel_name == channel_name]
    
    def get_detection_mask(self, artifact_types: Optional[List[ArtifactType]] = None) -> np.ndarray:
        """
        Get boolean mask indicating artifact periods
        
        Args:
            artifact_types: Types of artifacts to include (None for all)
            
        Returns:
            Boolean array with True for artifact periods
        """
        if self.raw_data is None:
            return np.array([])
        
        mask = np.zeros(self.raw_data.n_times, dtype=bool)
        
        for detection in self.detections:
            if artifact_types is None or detection.artifact_type in artifact_types:
                mask[detection.start_sample:detection.end_sample] = True
        
        return mask