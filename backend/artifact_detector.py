#!/usr/bin/env python3
"""
Artifact Detector - Detection of artifacts in ICA and PCA components
=====================================================================

Implements algorithms for automatic artifact detection in EEG data:
- EOG artifact detection (eye blinks) - for ICA
- Statistical component analysis - for ICA and PCA
- Explained variance analysis - for PCA
- Multiple detection methods
- Generic detection with fallback methods
- Wavelet processor handling (automatic denoising, no component selection)

Author: porfanid
Version: 1.2
"""

from typing import Dict, List, Tuple, Union

import mne
import numpy as np
from scipy import stats

from .base_processor import BaseComponentProcessor
from .ica_processor import ICAProcessor
from .pca_processor import PCAProcessor
from .wavelet_processor import WaveletProcessor


class ArtifactDetector:
    """
    Artifact detection in ICA and PCA components using multiple methods.

    Uses various algorithms for artifact detection such as:
    - EOG artifacts (eye blinks) via frontal channels (ICA only)
    - Statistical analysis (variance, kurtosis, range)
    - Explained variance analysis (PCA specific)
    - Combined detection algorithms

    Attributes:
        variance_threshold (float): Variance threshold for artifacts
        kurtosis_threshold (float): Kurtosis threshold for artifacts
        range_threshold (float): Range threshold for artifacts
    """

    def __init__(
        self,
        variance_threshold: float = 2.0,
        kurtosis_threshold: float = 2.0,
        range_threshold: float = 3.0,
    ):
        """
        Initialize artifact detector.

        Args:
            variance_threshold (float): Variance threshold for artifacts
            kurtosis_threshold (float): Kurtosis threshold for artifacts
            range_threshold (float): Range threshold for artifacts
        """
        self.variance_threshold = variance_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.range_threshold = range_threshold

    def detect_eog_artifacts(
        self, ica: mne.preprocessing.ICA, raw: mne.io.Raw
    ) -> List[int]:
        """
        Detect EOG artifacts using MNE (ICA only).

        Args:
            ica: Trained ICA object
            raw: Raw EEG data

        Returns:
            List of EOG artifact component indices
        """
        try:
            # Use frontal channels as EOG proxy
            frontal_channels = [ch for ch in ["AF3", "AF4"] if ch in raw.ch_names]

            if not frontal_channels:
                return []

            # Detect EOG artifacts
            eog_indices, _ = ica.find_bads_eog(
                raw, ch_name=frontal_channels, threshold=2.0, verbose=False
            )

            return eog_indices

        except Exception as e:
            print(f"EOG detection error: {str(e)}")
            return []

    def detect_statistical_artifacts(
        self, processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor]
    ) -> List[int]:
        """
        Detect artifacts using statistical criteria.

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA) with computed components

        Returns:
            List of artifact component indices
        """
        artifacts = []
        components_info = processor.get_all_components_info()

        if not components_info:
            return []

        # Calculate thresholds based on distribution
        variances = [info["variance"] for info in components_info.values()]
        kurtoses = [info["kurtosis"] for info in components_info.values()]
        ranges = [info["range"] for info in components_info.values()]

        var_mean, var_std = np.mean(variances), np.std(variances)
        kurt_mean, kurt_std = np.mean(kurtoses), np.std(kurtoses)
        range_mean, range_std = np.mean(ranges), np.std(ranges)

        # Detect outliers
        for comp_idx, info in components_info.items():
            is_artifact = False

            # Variance criterion
            if info["variance"] > var_mean + self.variance_threshold * var_std:
                is_artifact = True

            # Kurtosis criterion
            if info["kurtosis"] > kurt_mean + self.kurtosis_threshold * kurt_std:
                is_artifact = True

            # Range criterion
            if info["range"] > range_mean + self.range_threshold * range_std:
                is_artifact = True

            if is_artifact:
                artifacts.append(comp_idx)

        return artifacts

    def detect_muscle_artifacts(
        self,
        processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor],
        frequency_threshold: float = 20.0,
    ) -> List[int]:
        """
        Detect muscle artifacts (high frequencies).

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA)
            frequency_threshold: Frequency threshold (Hz)

        Returns:
            List of muscle artifact component indices
        """
        artifacts: List[int] = []

        if processor.raw_data is None:
            return []

        try:
            sources_data = processor.get_sources_data()
            if sources_data is None:
                return []

            sfreq = processor.raw_data.info["sfreq"]

            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]

                # FFT for frequency analysis
                freqs = np.fft.fftfreq(len(comp_data), 1 / sfreq)
                fft_data = np.abs(np.fft.fft(comp_data))

                # Calculate power in high frequencies
                high_freq_mask = freqs > frequency_threshold
                high_freq_power = np.sum(fft_data[high_freq_mask])
                total_power = np.sum(fft_data)

                # If power in high frequencies is >50% of total
                if high_freq_power / total_power > 0.5:
                    artifacts.append(i)

            return artifacts

        except Exception as e:
            print(f"Muscle artifact detection error: {str(e)}")
            return []

    def detect_drift_artifacts(
        self,
        processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor],
        drift_threshold: float = 0.1,
    ) -> List[int]:
        """
        Detect drift artifacts (low frequencies).

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA)
            drift_threshold: Threshold for drift (Hz)

        Returns:
            List of drift artifact component indices
        """
        artifacts: List[int] = []

        if processor.raw_data is None:
            return []

        try:
            sources_data = processor.get_sources_data()
            if sources_data is None:
                return []

            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]

                # Calculate trend
                x = np.arange(len(comp_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, comp_data
                )

                # If there is a significant trend
                if abs(r_value) > 0.7 and p_value < 0.05:
                    artifacts.append(i)

            return artifacts

        except Exception as e:
            print(f"Drift artifact detection error: {str(e)}")
            return []

    def detect_pca_variance_artifacts(
        self, pca_processor: PCAProcessor, variance_ratio_threshold: float = 0.3
    ) -> List[int]:
        """
        Detect artifacts based on explained variance (PCA specific).

        In PCA, artifacts often appear as components with excessively high
        explained variance (e.g., eye blinks) or very low (noise).

        Args:
            pca_processor: PCA processor
            variance_ratio_threshold: Variance ratio threshold

        Returns:
            List of artifact component indices
        """
        artifacts: List[int] = []

        try:
            explained_variance = pca_processor.get_explained_variance_ratio()
            if explained_variance is None:
                return []

            # If a component explains excessively large percentage of variance
            # (> threshold), it may be an artifact (e.g., eye blinks)
            for i, var_ratio in enumerate(explained_variance):
                # First components with excessive variance
                if var_ratio > variance_ratio_threshold:
                    artifacts.append(i)

            return artifacts

        except Exception as e:
            print(f"PCA variance artifact detection error: {str(e)}")
            return []

    def detect_pca_spatial_artifacts(
        self, pca_processor: PCAProcessor, raw: mne.io.Raw
    ) -> List[int]:
        """
        Detect artifacts based on spatial patterns (PCA specific).

        Checks if PCA components have high weights in frontal channels
        (indicates EOG artifacts).

        Args:
            pca_processor: PCA processor
            raw: Raw EEG data

        Returns:
            List of artifact component indices
        """
        artifacts: List[int] = []

        try:
            components = pca_processor.get_components()
            if components is None:
                return []

            ch_names = raw.ch_names

            # Find frontal channels (potential EOG artifact sources)
            frontal_indices = []
            frontal_patterns = ["Fp", "AF", "F3", "F4", "F7", "F8", "Fz"]
            for i, ch in enumerate(ch_names):
                if any(pattern in ch for pattern in frontal_patterns):
                    frontal_indices.append(i)

            if not frontal_indices:
                return []

            # For each component, check if it has high weights in frontal channels
            n_components = components.shape[1]
            for comp_idx in range(n_components):
                comp_weights = np.abs(components[:, comp_idx])

                # Calculate ratio of frontal vs total weights
                frontal_weights = np.sum(comp_weights[frontal_indices])
                total_weights = np.sum(comp_weights)

                # If > 50% of weights are in frontal channels, possible EOG artifact
                # Guard against division by zero
                if total_weights > 0 and frontal_weights / total_weights > 0.5:
                    artifacts.append(comp_idx)

            return artifacts

        except Exception as e:
            print(f"PCA spatial artifact detection error: {str(e)}")
            return []

    def detect_artifacts_multi_method(
        self,
        processor: Union[
            ICAProcessor, PCAProcessor, WaveletProcessor, BaseComponentProcessor
        ],
        raw: mne.io.Raw,
        max_components: int = 3,
    ) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Multiple artifact detection with method combination.

        Supports ICA, PCA, and Wavelet processors with appropriate methods.
        For Wavelet processors, returns empty results as denoising is automatic.

        Args:
            processor: Component processor (ICA, PCA, or Wavelet)
            raw: Raw EEG data
            max_components: Maximum number of components to remove

        Returns:
            Tuple with:
            - Final artifact list
            - Dictionary with results from each method
        """
        methods_results: Dict[str, List[int]] = {}

        # Determine processor type and apply appropriate methods
        is_ica = isinstance(processor, ICAProcessor)
        is_pca = isinstance(processor, PCAProcessor)
        is_wavelet = isinstance(processor, WaveletProcessor)

        # Wavelet denoising is automatic - no artifact detection needed
        if is_wavelet:
            # Return empty results - Wavelet denoising handles all channels automatically
            # All channels will be denoised, no component selection needed
            methods_results["wavelet_auto"] = []
            return [], methods_results

        if is_ica:
            ica = processor.get_ica_object()
            if ica is None:
                return [], {}

            # ICA-specific methods
            methods_results["eog"] = self.detect_eog_artifacts(ica, raw)

        if is_pca:
            # PCA-specific methods
            methods_results["variance"] = self.detect_pca_variance_artifacts(processor)
            methods_results["spatial"] = self.detect_pca_spatial_artifacts(
                processor, raw
            )

        # Common methods for both ICA and PCA
        methods_results["statistical"] = self.detect_statistical_artifacts(processor)
        methods_results["muscle"] = self.detect_muscle_artifacts(processor)
        methods_results["drift"] = self.detect_drift_artifacts(processor)

        # Combine results with weights
        artifact_scores = {}

        for comp_idx in range(processor.n_components):
            score = 0

            if is_ica:
                # EOG detection (weight 3) - ICA only
                if comp_idx in methods_results.get("eog", []):
                    score += 3

            if is_pca:
                # PCA variance detection (weight 3)
                if comp_idx in methods_results.get("variance", []):
                    score += 3

                # PCA spatial detection (weight 2)
                if comp_idx in methods_results.get("spatial", []):
                    score += 2

            # Statistical detection (weight 2)
            if comp_idx in methods_results.get("statistical", []):
                score += 2

            # Muscle detection (weight 2)
            if comp_idx in methods_results.get("muscle", []):
                score += 2

            # Drift detection (weight 1)
            if comp_idx in methods_results.get("drift", []):
                score += 1

            artifact_scores[comp_idx] = score

        # Select top artifact components
        sorted_components = sorted(
            artifact_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Keep only components with score > 0
        final_artifacts = [
            comp_idx for comp_idx, score in sorted_components if score > 0
        ][:max_components]

        return final_artifacts, methods_results

    def get_artifact_explanation(
        self, component_idx: int, methods_results: Dict[str, List[int]]
    ) -> str:
        """
        Explain why a component is considered an artifact.

        Args:
            component_idx: Component index
            methods_results: Results from detection methods

        Returns:
            Explanation text
        """
        # Wavelet-specific - automatic denoising, no component artifacts
        if "wavelet_auto" in methods_results:
            return "Wavelet denoising applied to all channels"

        reasons = []

        # ICA-specific
        if component_idx in methods_results.get("eog", []):
            reasons.append("EOG (eye movement)")

        # PCA-specific
        if component_idx in methods_results.get("variance", []):
            reasons.append("High variance (PCA)")

        if component_idx in methods_results.get("spatial", []):
            reasons.append("Spatial pattern (frontal)")

        # Common methods
        if component_idx in methods_results.get("statistical", []):
            reasons.append("Statistical outlier")

        if component_idx in methods_results.get("muscle", []):
            reasons.append("Muscle activity")

        if component_idx in methods_results.get("drift", []):
            reasons.append("Signal drift")

        if not reasons:
            return "Clean brain signal"

        return f"Possible artifact: {', '.join(reasons)}"
