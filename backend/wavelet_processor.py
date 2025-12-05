#!/usr/bin/env python3
"""
Wavelet Processor - Discrete Wavelet Transform Denoising for EEG artifact cleaning
==================================================================================

Implements Discrete Wavelet Transform (DWT) denoising for:
- Signal denoising using wavelet decomposition
- Automatic threshold calculation (Universal Thresholding based on MAD)
- Soft thresholding for noise reduction
- Signal reconstruction

This method is particularly useful for low-channel EEG systems (≤8 channels)
where ICA/PCA may not be effective due to insufficient spatial information.

Author: porfanid
Version: 1.0
"""

import warnings
from typing import Dict, List, Optional

import mne
import numpy as np
import pywt

from .base_processor import BaseComponentProcessor

warnings.filterwarnings("ignore", category=RuntimeWarning)


class WaveletProcessor(BaseComponentProcessor):
    """
    Wavelet Processor for EEG signal denoising using Discrete Wavelet Transform.

    Uses DWT decomposition and soft thresholding to remove noise from EEG signals.
    Unlike ICA/PCA, this method works on individual channels and doesn't require
    spatial decomposition, making it ideal for low-channel-count EEG systems.

    The workflow treats channels as "components" for compatibility with the
    existing GUI infrastructure, but the actual denoising is applied to all
    channels automatically.

    Attributes:
        n_components (int): Number of channels (treated as components for compatibility)
        wavelet (str): Wavelet family to use (e.g., 'db4', 'sym8')
        level (int): Decomposition level (None for automatic calculation)
        threshold_mode (str): Thresholding mode ('soft' or 'hard')
        raw_data (mne.io.Raw): The training data
        components_info (dict): Statistical information about each channel
        _denoised_data (np.ndarray): Cached denoised data
    """

    def __init__(
        self,
        n_components: int = None,
        random_state: int = 42,
        wavelet: str = "db4",
        level: Optional[int] = None,
        threshold_mode: str = "soft",
    ):
        """
        Initialize Wavelet processor.

        Args:
            n_components (int, optional): Number of components (ignored, set to channel count).
            random_state (int): Seed for reproducibility (not used, kept for interface compatibility).
            wavelet (str): Wavelet family to use. Common choices:
                - 'db4': Daubechies 4 (good general-purpose choice)
                - 'db8': Daubechies 8 (smoother)
                - 'sym8': Symlet 8 (more symmetric)
                - 'coif3': Coiflet 3
            level (int, optional): Decomposition level. If None, automatically calculated
                based on signal length.
            threshold_mode (str): Thresholding mode:
                - 'soft': Soft thresholding (shrinks coefficients towards zero)
                - 'hard': Hard thresholding (sets coefficients below threshold to zero)
        """
        super().__init__(n_components, random_state)
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self._denoised_data: Optional[np.ndarray] = None
        self._original_data: Optional[np.ndarray] = None

    def fit(self, raw: mne.io.Raw) -> bool:
        """
        Fit the Wavelet processor to EEG data.

        Stores the filtered signal and prepares for denoising.
        Sets n_components equal to the number of channels (architectural adaptation
        to maintain compatibility with the component-based GUI).

        Args:
            raw (mne.io.Raw): Filtered Raw EEG data

        Returns:
            bool: True if fitting was successful, False otherwise
        """
        try:
            self.raw_data = raw.copy()

            # Get data matrix (channels x timepoints)
            data = raw.get_data()
            self._original_data = data.copy()

            # Set n_components equal to number of channels
            # This is an architectural adaptation for GUI compatibility
            self.n_components = len(raw.ch_names)

            # Automatically calculate decomposition level if not provided
            if self.level is None:
                # Calculate maximum level based on signal length and wavelet
                max_level = pywt.dwt_max_level(data.shape[1], self.wavelet)
                # Use a reasonable level (not too high to avoid over-smoothing)
                self.level = min(max_level, 5)

            # Pre-compute denoised data for all channels
            self._denoised_data = self._denoise_all_channels(data)

            # Calculate component information (based on original data)
            self._calculate_component_info()

            return True

        except Exception as e:
            print(f"Error during Wavelet fitting: {str(e)}")
            return False

    def _denoise_all_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Apply DWT denoising to all channels.

        Args:
            data: Data matrix (n_channels, n_timepoints)

        Returns:
            np.ndarray: Denoised data matrix (n_channels, n_timepoints)
        """
        denoised = np.zeros_like(data)

        for i in range(data.shape[0]):
            denoised[i] = self._denoise_signal(data[i])

        return denoised

    def _denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Denoise a single signal using DWT.

        Applies Universal Thresholding (VisuShrink) based on the median
        absolute deviation (MAD) of the detail coefficients.

        Args:
            signal: 1D signal array

        Returns:
            np.ndarray: Denoised signal
        """
        # Decompose signal
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # Calculate threshold using Universal Thresholding (VisuShrink)
        # Based on MAD (Median Absolute Deviation) of finest detail coefficients
        # σ = MAD(d1) / 0.6745, where d1 is the first level detail coefficients
        detail_coeffs = coeffs[-1]  # Finest level detail coefficients
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745

        # Universal threshold: sqrt(2 * log(n)) * σ
        n = len(signal)
        threshold = sigma * np.sqrt(2 * np.log(n))

        # Apply thresholding to detail coefficients (not approximation)
        thresholded_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged

        for i in range(1, len(coeffs)):
            if self.threshold_mode == "soft":
                thresholded_coeffs.append(pywt.threshold(coeffs[i], threshold, "soft"))
            else:
                thresholded_coeffs.append(pywt.threshold(coeffs[i], threshold, "hard"))

        # Reconstruct signal
        denoised_signal = pywt.waverec(thresholded_coeffs, self.wavelet)

        # Handle potential length mismatch due to padding
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[: len(signal)]
        elif len(denoised_signal) < len(signal):
            # Pad with zeros if needed (rare case)
            denoised_signal = np.pad(
                denoised_signal, (0, len(signal) - len(denoised_signal))
            )

        return denoised_signal

    def get_sources_data(self) -> Optional[np.ndarray]:
        """
        Get the source (channel) signals.

        Returns channel data as "sources" for GUI compatibility.
        In Wavelet denoising, there are no true independent sources,
        so we return the original channel data.

        Returns:
            Optional[np.ndarray]: Array of shape (n_channels, n_timepoints) or None
        """
        if self._original_data is None:
            return None
        return self._original_data

    def get_components(self) -> Optional[np.ndarray]:
        """
        Get the spatial patterns (mixing weights) for each component.

        Returns None as Wavelet denoising doesn't produce spatial components
        like ICA/PCA. This is handled gracefully by the GUI.

        Returns:
            Optional[np.ndarray]: None (no spatial components for Wavelet)
        """
        return None

    def apply_artifact_removal(
        self, components_to_remove: List[int]
    ) -> Optional[mne.io.Raw]:
        """
        Apply artifact removal using DWT denoising.

        This is the main denoising method. It applies DWT-based denoising
        to all channels. The components_to_remove parameter is ignored
        as Wavelet denoising works on all channels automatically.

        Args:
            components_to_remove (List[int]): Ignored - denoising is applied
                to all channels regardless of this parameter.

        Returns:
            Optional[mne.io.Raw]: Cleaned Raw data or None if failed
        """
        if self.raw_data is None or self._denoised_data is None:
            return None

        try:
            # Create cleaned raw object
            cleaned_raw = self.raw_data.copy()
            cleaned_raw._data = self._denoised_data.copy()

            return cleaned_raw

        except Exception as e:
            print(f"Error during Wavelet denoising: {str(e)}")
            return None

    def get_method_name(self) -> str:
        """
        Get the name of this analysis method.

        Returns:
            str: "WAVELETS"
        """
        return "WAVELETS"

    def get_wavelet_info(self) -> Dict[str, any]:
        """
        Get information about the current wavelet configuration.

        Returns:
            Dict with wavelet configuration details
        """
        return {
            "wavelet": self.wavelet,
            "level": self.level,
            "threshold_mode": self.threshold_mode,
        }

    def set_wavelet_params(
        self,
        wavelet: Optional[str] = None,
        level: Optional[int] = None,
        threshold_mode: Optional[str] = None,
    ):
        """
        Update wavelet parameters.

        Args:
            wavelet: New wavelet family
            level: New decomposition level
            threshold_mode: New threshold mode ('soft' or 'hard')
        """
        if wavelet is not None:
            self.wavelet = wavelet
        if level is not None:
            self.level = level
        if threshold_mode is not None:
            if threshold_mode not in ["soft", "hard"]:
                raise ValueError("threshold_mode must be 'soft' or 'hard'")
            self.threshold_mode = threshold_mode

        # Re-compute denoised data if we have raw data
        if self._original_data is not None:
            self._denoised_data = self._denoise_all_channels(self._original_data)

    def get_noise_reduction_stats(self) -> Optional[Dict[str, float]]:
        """
        Calculate noise reduction statistics.

        Returns:
            Dict with noise reduction metrics or None if no data
        """
        if self._original_data is None or self._denoised_data is None:
            return None

        original_rms = np.sqrt(np.mean(self._original_data**2))
        denoised_rms = np.sqrt(np.mean(self._denoised_data**2))

        # Calculate noise estimate (difference between original and denoised)
        noise_estimate = self._original_data - self._denoised_data
        noise_rms = np.sqrt(np.mean(noise_estimate**2))

        # SNR improvement estimation
        if noise_rms > 0:
            snr_improvement_db = 10 * np.log10(original_rms / noise_rms)
        else:
            snr_improvement_db = float("inf")

        return {
            "original_rms": float(original_rms),
            "denoised_rms": float(denoised_rms),
            "noise_rms": float(noise_rms),
            "rms_reduction_percent": float(
                (1 - denoised_rms / original_rms) * 100 if original_rms > 0 else 0
            ),
            "estimated_snr_improvement_db": float(snr_improvement_db),
        }

    @staticmethod
    def get_available_wavelets() -> List[str]:
        """
        Get list of available wavelet families suitable for EEG.

        Returns:
            List of wavelet names
        """
        return [
            "db4",  # Daubechies 4 - good general choice
            "db8",  # Daubechies 8 - smoother
            "sym4",  # Symlet 4 - more symmetric
            "sym8",  # Symlet 8 - more symmetric, smoother
            "coif3",  # Coiflet 3 - good for signals with discontinuities
            "bior3.5",  # Biorthogonal - good reconstruction properties
        ]
