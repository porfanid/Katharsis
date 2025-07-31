#!/usr/bin/env python3
"""
Enhanced ICA Processing System - Phase 2
========================================

Advanced ICA capabilities including:
- Multiple ICA algorithms (FastICA, Extended Infomax, AMICA)
- Automated component classification (eye, muscle, heart, line noise)
- ICA stability analysis and validation
- Advanced artifact detection methods

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
from scipy.spatial.distance import cdist
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class ICAMethod(Enum):
    """Available ICA algorithms"""
    FASTICA = "fastica"
    EXTENDED_INFOMAX = "extended-infomax" 
    PICARD = "picard"
    MNE_DEFAULT = "mne-default"


class ComponentType(Enum):
    """Types of ICA components"""
    EYE_BLINK = "eye_blink"
    EYE_MOVEMENT = "eye_movement"
    MUSCLE = "muscle"
    HEART = "heart"
    LINE_NOISE = "line_noise"
    BRAIN = "brain"
    UNKNOWN = "unknown"


@dataclass
class ComponentClassification:
    """Classification result for an ICA component"""
    component_idx: int
    component_type: ComponentType
    confidence: float
    features: Dict[str, float]
    rejection_reason: Optional[str] = None


@dataclass
class ICAConfig:
    """Configuration for ICA processing"""
    method: ICAMethod = ICAMethod.FASTICA
    n_components: Optional[int] = None
    max_iter: int = 1000
    random_state: int = 42
    fit_params: Optional[Dict] = None
    
    # Component classification settings
    enable_auto_classification: bool = True
    classification_thresholds: Optional[Dict[str, float]] = None
    
    # Stability analysis
    enable_stability_analysis: bool = False
    n_stability_runs: int = 5
    stability_threshold: float = 0.8
    
    # Additional processing options
    compute_sources: bool = True


class EnhancedICAProcessor:
    """
    Enhanced ICA processor with multiple algorithms and automated classification
    
    Features:
    - Multiple ICA algorithms (FastICA, Extended Infomax, etc.)
    - Automated component classification
    - ICA stability analysis
    - Advanced artifact detection
    """
    
    def __init__(self, config: ICAConfig = None):
        """Initialize enhanced ICA processor"""
        self.config = config or ICAConfig()
        self.ica: Optional[mne.preprocessing.ICA] = None
        self.raw_data: Optional[mne.io.Raw] = None
        self.fitted: bool = False
        self.components_: Optional[np.ndarray] = None
        self.mixing_matrix_: Optional[np.ndarray] = None
        
        # Classification results
        self.component_classifications: List[ComponentClassification] = []
        self.auto_reject_indices: List[int] = []
        
        # Stability analysis results
        self.stability_scores: Optional[np.ndarray] = None
        self.stability_matrix: Optional[np.ndarray] = None
        
        # Component features for classification
        self.component_features: Dict[int, Dict[str, float]] = {}
        
        # Set default classification thresholds
        if self.config.classification_thresholds is None:
            self.config.classification_thresholds = {
                'eye_blink_threshold': 3.0,      # Kurtosis threshold for eye blinks
                'muscle_threshold': 30.0,        # High frequency power threshold
                'heart_threshold': 0.7,          # Heart rate correlation threshold
                'line_noise_threshold': 0.8,     # 50/60 Hz power ratio threshold
                'frontal_eye_threshold': 0.6,    # Frontal channel correlation for eye artifacts
                'temporal_muscle_threshold': 0.5  # Temporal channel correlation for muscle
            }
    
    def fit_ica(self, raw: mne.io.Raw, picks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit ICA model using specified algorithm
        
        Args:
            raw: Raw EEG data
            picks: Channels to use for ICA (None for all EEG channels)
            
        Returns:
            Dictionary with fitting results and statistics
        """
        try:
            self.raw_data = raw.copy()
            
            # Prepare data
            if picks is None:
                picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            
            # Initialize ICA with specified method
            n_components = self.config.n_components
            if n_components is None:
                # Use number of channels, not time points
                n_components = min(len(picks), raw.get_data(picks=picks).shape[0])
                n_components = max(n_components, min(10, len(picks)))  # At least 10 components or number of channels
            
            # Create ICA object based on method
            fit_params = self.config.fit_params or {}
            
            if self.config.method == ICAMethod.FASTICA:
                # Add tolerance parameter for better convergence
                if 'tol' not in fit_params:
                    fit_params['tol'] = 1e-3  # More lenient tolerance
                self.ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method='fastica',
                    random_state=self.config.random_state,
                    max_iter=self.config.max_iter,
                    fit_params=fit_params
                )
            elif self.config.method == ICAMethod.EXTENDED_INFOMAX:
                self.ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method='infomax',
                    random_state=self.config.random_state,
                    max_iter=self.config.max_iter,
                    fit_params=self.config.fit_params or {'extended': True}
                )
            elif self.config.method == ICAMethod.PICARD:
                self.ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method='picard',
                    random_state=self.config.random_state,
                    max_iter=self.config.max_iter,
                    fit_params=self.config.fit_params or {}
                )
            else:  # MNE_DEFAULT
                self.ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    random_state=self.config.random_state,
                    max_iter=self.config.max_iter
                )
            
            # Fit ICA
            self.ica.fit(raw, picks=picks)
            
            # Check if ICA was fitted properly - handle different MNE versions
            try:
                # Try to get components using the method (newer MNE versions)
                components = self.ica.get_components()
                if components is not None and components.size > 0:
                    self.components_ = components
                    # Get mixing matrix
                    if hasattr(self.ica, 'mixing_matrix_'):
                        self.mixing_matrix_ = self.ica.mixing_matrix_
                    else:
                        # Calculate mixing matrix from components
                        self.mixing_matrix_ = np.linalg.pinv(components)
                    
                    self.fitted = True
                    
                    return {
                        'success': True,
                        'n_components': self.ica.n_components_,
                        'method': self.config.method.value,
                        'explained_variance': self._calculate_explained_variance(),
                        'auto_classifications': len(self.component_classifications),
                        'auto_reject_count': len(self.auto_reject_indices),
                        'fitted': True
                    }
                else:
                    raise RuntimeError("ICA fitting failed - no components extracted.")
                    
            except Exception as get_error:
                # Fallback: try direct attribute access (older MNE versions)
                if hasattr(self.ica, 'components_') and self.ica.components_ is not None:
                    self.components_ = self.ica.components_
                    self.mixing_matrix_ = getattr(self.ica, 'mixing_', None)
                    self.fitted = True
                    
                    return {
                        'success': True,
                        'n_components': self.ica.n_components_,
                        'method': self.config.method.value,
                        'explained_variance': self._calculate_explained_variance(),
                        'auto_classifications': len(self.component_classifications),
                        'auto_reject_count': len(self.auto_reject_indices),
                        'fitted': True
                    }
                else:
                    raise RuntimeError(f"ICA fitting failed - could not access components: {str(get_error)}")
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'n_components': 0
            }
    
    def _classify_components(self):
        """Automatically classify ICA components"""
        if self.ica is None or self.raw_data is None:
            return
        
        self.component_classifications = []
        self.auto_reject_indices = []
        
        # Get component activations
        sources = self.ica.get_sources(self.raw_data)
        
        # Analyze each component
        for comp_idx in range(self.ica.n_components_):
            classification = self._classify_single_component(comp_idx, sources)
            self.component_classifications.append(classification)
            
            # Auto-reject if not brain activity
            if classification.component_type != ComponentType.BRAIN:
                if classification.confidence > 0.7:  # High confidence threshold
                    self.auto_reject_indices.append(comp_idx)
    
    def _classify_single_component(self, comp_idx: int, sources: mne.io.Raw) -> ComponentClassification:
        """Classify a single ICA component"""
        
        # Extract component time series and topography
        component_ts = sources.get_data()[comp_idx, :]
        component_topo = self.ica.mixing_[:, comp_idx]
        
        # Calculate features
        features = self._extract_component_features(component_ts, component_topo, comp_idx)
        self.component_features[comp_idx] = features
        
        # Classification logic
        component_type, confidence, reason = self._determine_component_type(features)
        
        return ComponentClassification(
            component_idx=comp_idx,
            component_type=component_type,
            confidence=confidence,
            features=features,
            rejection_reason=reason
        )
    
    def _extract_component_features(self, component_ts: np.ndarray, 
                                  component_topo: np.ndarray, comp_idx: int) -> Dict[str, float]:
        """Extract features for component classification"""
        
        features = {}
        
        # Temporal features
        features['kurtosis'] = stats.kurtosis(component_ts)
        features['skewness'] = stats.skew(component_ts)
        features['variance'] = np.var(component_ts)
        features['mean_abs'] = np.mean(np.abs(component_ts))
        
        # Frequency features
        sfreq = self.raw_data.info['sfreq']
        freqs, psd = signal.welch(component_ts, sfreq, nperseg=min(1024, len(component_ts)//4))
        
        # Band power ratios
        delta_power = np.mean(psd[(freqs >= 1) & (freqs <= 4)])
        theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
        beta_power = np.mean(psd[(freqs >= 12) & (freqs <= 30)])
        gamma_power = np.mean(psd[(freqs >= 30) & (freqs <= 100)])
        
        total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
        if total_power > 0:
            features['delta_ratio'] = delta_power / total_power
            features['theta_ratio'] = theta_power / total_power
            features['alpha_ratio'] = alpha_power / total_power
            features['beta_ratio'] = beta_power / total_power
            features['gamma_ratio'] = gamma_power / total_power
        else:
            features.update({f'{band}_ratio': 0.0 for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']})
        
        # High frequency power (muscle indicator)
        high_freq_power = np.mean(psd[freqs >= 50])
        features['high_freq_power'] = high_freq_power
        features['high_freq_ratio'] = high_freq_power / (total_power + 1e-10)
        
        # Line noise indicators (50/60 Hz)
        line_50_idx = np.argmin(np.abs(freqs - 50))
        line_60_idx = np.argmin(np.abs(freqs - 60))
        features['line_50_power'] = psd[line_50_idx] if line_50_idx < len(psd) else 0
        features['line_60_power'] = psd[line_60_idx] if line_60_idx < len(psd) else 0
        
        # Topographical features
        if len(component_topo) == len(self.raw_data.ch_names):
            ch_names = self.raw_data.ch_names
            
            # Frontal bias (eye artifacts)
            frontal_channels = [i for i, ch in enumerate(ch_names) if any(fp in ch.upper() for fp in ['FP', 'AF', 'F'])]
            if frontal_channels:
                frontal_weights = np.abs(component_topo[frontal_channels])
                features['frontal_bias'] = np.mean(frontal_weights)
                features['max_frontal'] = np.max(frontal_weights)
            else:
                features['frontal_bias'] = 0.0
                features['max_frontal'] = 0.0
            
            # Temporal bias (muscle artifacts)
            temporal_channels = [i for i, ch in enumerate(ch_names) if any(t in ch.upper() for t in ['T', 'TP', 'FT'])]
            if temporal_channels:
                temporal_weights = np.abs(component_topo[temporal_channels])
                features['temporal_bias'] = np.mean(temporal_weights)
                features['max_temporal'] = np.max(temporal_weights)
            else:
                features['temporal_bias'] = 0.0
                features['max_temporal'] = 0.0
            
            # Symmetry (eye movements vs blinks)
            left_channels = [i for i, ch in enumerate(ch_names) if any(l in ch.upper() for l in ['1', '3', '5', '7'])]
            right_channels = [i for i, ch in enumerate(ch_names) if any(r in ch.upper() for r in ['2', '4', '6', '8'])]
            
            if left_channels and right_channels:
                left_weights = component_topo[left_channels]
                right_weights = component_topo[right_channels]
                features['asymmetry'] = np.abs(np.mean(left_weights) - np.mean(right_weights))
            else:
                features['asymmetry'] = 0.0
            
            # Overall spatial concentration
            features['spatial_concentration'] = np.max(np.abs(component_topo)) / (np.mean(np.abs(component_topo)) + 1e-10)
        
        return features
    
    def _determine_component_type(self, features: Dict[str, float]) -> Tuple[ComponentType, float, Optional[str]]:
        """Determine component type based on features"""
        
        thresholds = self.config.classification_thresholds
        
        # Eye blink detection (high kurtosis + frontal bias)
        if (features['kurtosis'] > thresholds['eye_blink_threshold'] and 
            features['frontal_bias'] > thresholds['frontal_eye_threshold']):
            confidence = min(features['kurtosis'] / 10.0, 0.95)
            return ComponentType.EYE_BLINK, confidence, "High kurtosis + frontal topography"
        
        # Eye movement detection (frontal bias + asymmetry)
        if (features['frontal_bias'] > thresholds['frontal_eye_threshold'] and 
            features['asymmetry'] > 0.3):
            confidence = min(features['frontal_bias'] * 2, 0.9)
            return ComponentType.EYE_MOVEMENT, confidence, "Frontal asymmetric topography"
        
        # Muscle artifact detection (high frequency power + temporal bias)
        if (features['high_freq_ratio'] > 0.3 and 
            features['temporal_bias'] > thresholds['temporal_muscle_threshold']):
            confidence = min(features['high_freq_ratio'] * 2, 0.9)
            return ComponentType.MUSCLE, confidence, "High frequency power + temporal topography"
        
        # Line noise detection
        line_noise_ratio = (features['line_50_power'] + features['line_60_power']) / (features.get('alpha_power', 1) + 1e-10)
        if line_noise_ratio > thresholds['line_noise_threshold']:
            confidence = min(line_noise_ratio / 2.0, 0.9)
            return ComponentType.LINE_NOISE, confidence, "Strong 50/60 Hz power"
        
        # Heart artifact detection (low frequency + specific temporal pattern)
        if (features['delta_ratio'] > 0.5 and 
            features['kurtosis'] < 2.0 and 
            features['temporal_bias'] > 0.3):
            confidence = 0.7
            return ComponentType.HEART, confidence, "Low frequency + temporal pattern"
        
        # Default to brain activity if no artifacts detected
        # Higher confidence for components with good brain-like features
        brain_confidence = 0.5
        if (features['alpha_ratio'] > 0.2 or features['beta_ratio'] > 0.2):
            brain_confidence += 0.2
        if features['spatial_concentration'] < 3.0:  # Not too spatially concentrated
            brain_confidence += 0.1
        
        return ComponentType.BRAIN, min(brain_confidence, 0.9), "Default brain activity"
    
    def _analyze_stability(self, raw: mne.io.Raw, picks: Optional[List[str]] = None):
        """Analyze ICA stability across multiple runs"""
        
        if self.ica is None:
            return
        
        n_runs = self.config.n_stability_runs
        n_components = self.ica.n_components_
        
        # Store components from multiple runs
        all_components = []
        
        for run in range(n_runs):
            # Fit ICA with different random state
            temp_ica = mne.preprocessing.ICA(
                n_components=n_components,
                method=self.config.method.value,
                random_state=self.config.random_state + run,
                max_iter=self.config.max_iter
            )
            
            temp_ica.fit(raw, picks=picks)
            all_components.append(temp_ica.components_)
        
        # Calculate stability matrix (correlation between components across runs)
        self.stability_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(n_components):
                correlations = []
                for run in range(n_runs - 1):
                    corr = np.corrcoef(all_components[run][i, :], all_components[run + 1][j, :])[0, 1]
                    correlations.append(abs(corr))
                
                self.stability_matrix[i, j] = np.mean(correlations)
        
        # Calculate stability scores (max correlation for each component)
        self.stability_scores = np.max(self.stability_matrix, axis=1)
    
    def _calculate_explained_variance(self) -> float:
        """Calculate explained variance by ICA components"""
        if self.ica is None or self.raw_data is None:
            return 0.0
        
        # Get original data
        original_data = self.raw_data.get_data()
        
        # Reconstruct data from all components
        reconstructed = self.ica.apply(self.raw_data.copy(), exclude=[])
        reconstructed_data = reconstructed.get_data()
        
        # Calculate explained variance
        original_var = np.var(original_data)
        residual_var = np.var(original_data - reconstructed_data)
        
        explained_var = 1 - (residual_var / original_var)
        return max(0.0, min(1.0, explained_var))
    
    def get_component_info(self, component_idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific component"""
        if self.ica is None or component_idx >= self.ica.n_components_:
            return {}
        
        info = {
            'component_index': component_idx,
            'mixing_weights': self.mixing_matrix_[:, component_idx].tolist(),
            'component_pattern': self.components_[component_idx, :].tolist()
        }
        
        # Add classification info if available
        if component_idx < len(self.component_classifications):
            classification = self.component_classifications[component_idx]
            info.update({
                'classification': classification.component_type.value,
                'confidence': classification.confidence,
                'features': classification.features,
                'rejection_reason': classification.rejection_reason,
                'auto_reject': component_idx in self.auto_reject_indices
            })
        
        # Add stability info if available
        if self.stability_scores is not None:
            info['stability_score'] = float(self.stability_scores[component_idx])
        
        return info
    
    def get_auto_reject_components(self) -> List[int]:
        """Get list of components recommended for automatic rejection"""
        return self.auto_reject_indices.copy()
    
    def apply_component_removal(self, components_to_remove: List[int]) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """
        Apply component removal and return cleaned data
        
        Args:
            components_to_remove: List of component indices to remove
            
        Returns:
            Tuple of (cleaned_raw, removal_info)
        """
        if self.ica is None or self.raw_data is None:
            raise ValueError("ICA model not fitted")
        
        # Apply ICA exclusion
        cleaned_raw = self.ica.apply(self.raw_data.copy(), exclude=components_to_remove)
        
        # Calculate removal statistics
        removal_info = {
            'removed_components': components_to_remove,
            'n_removed': len(components_to_remove),
            'remaining_components': self.ica.n_components_ - len(components_to_remove),
            'removed_types': []
        }
        
        # Add classification info for removed components
        for comp_idx in components_to_remove:
            if comp_idx < len(self.component_classifications):
                classification = self.component_classifications[comp_idx]
                removal_info['removed_types'].append({
                    'component': comp_idx,
                    'type': classification.component_type.value,
                    'confidence': classification.confidence
                })
        
        return cleaned_raw, removal_info
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of ICA processing"""
        if self.ica is None:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'method': self.config.method.value,
            'n_components': self.ica.n_components_,
            'explained_variance': self._calculate_explained_variance(),
            'classifications': {ct.value: 0 for ct in ComponentType},
            'auto_reject_count': len(self.auto_reject_indices),
            'high_confidence_artifacts': 0
        }
        
        # Count classifications
        for classification in self.component_classifications:
            summary['classifications'][classification.component_type.value] += 1
            if classification.confidence > 0.8 and classification.component_type != ComponentType.BRAIN:
                summary['high_confidence_artifacts'] += 1
        
        # Add stability info if available
        if self.stability_scores is not None:
            summary['stability'] = {
                'mean_stability': float(np.mean(self.stability_scores)),
                'min_stability': float(np.min(self.stability_scores)),
                'stable_components': int(np.sum(self.stability_scores > self.config.stability_threshold))
            }
        
        return summary
    
    def run_ica_analysis(self, raw: mne.io.Raw, picks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete ICA analysis workflow - fits ICA, classifies components, and provides results
        
        Args:
            raw: Raw EEG data
            picks: Channels to use for ICA (None for all EEG channels)
            
        Returns:
            Dictionary with complete ICA analysis results including:
            - ica: Fitted ICA object
            - component_classification: List of component classifications
            - auto_reject_indices: Components recommended for removal
            - summary: Processing summary
        """
        try:
            # Step 1: Fit ICA model
            fitting_results = self.fit_ica(raw, picks)
            
            if not fitting_results.get('success', False) or self.ica is None:
                error_msg = fitting_results.get('error', 'Unknown ICA fitting error')
                return {
                    'error': error_msg,
                    'status': 'failed',
                    'ica': None,
                    'component_classification': [],
                    'auto_reject_indices': [],
                    'summary': {'status': 'failed', 'error': error_msg}
                }
            
            # Step 2: Classify components (only if enabled and ICA fitted successfully)
            if self.config.enable_auto_classification and self.fitted:
                try:
                    self._classify_components()
                except Exception as e:
                    print(f"Warning: Component classification failed: {e}")
            
            # Step 3: Analyze stability if configured
            if self.config.enable_stability_analysis and self.fitted:
                try:
                    self._analyze_stability(raw, picks)
                except Exception as e:
                    print(f"Warning: Stability analysis failed: {e}")
            
            # Step 4: Compile results
            results = {
                'ica': self.ica,
                'component_classification': self.component_classifications,
                'auto_reject_indices': self.auto_reject_indices,
                'fitting_results': fitting_results,
                'summary': self.get_processing_summary(),
                'status': 'success'
            }
            
            # Add sources if requested
            if self.config.compute_sources and self.fitted:
                try:
                    sources = self.ica.get_sources(raw)
                    results['sources'] = sources
                except Exception as e:
                    print(f"Warning: Could not compute sources: {e}")
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'ica': None,
                'component_classification': [],
                'auto_reject_indices': [],
                'summary': {'status': 'failed', 'error': str(e)}
            }
        
        return summary