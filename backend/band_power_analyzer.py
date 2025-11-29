#!/usr/bin/env python3
"""
Band Power Analyzer - EEG Frequency Band Power Analysis
========================================================

Calculates the percentage distribution of power in different EEG
frequency bands (Delta, Theta, Alpha, Beta, Gamma).

EEG Frequency Bands:
- Delta (0.5–4 Hz): Deep sleep
- Theta (4–8 Hz): Light sleep and deep relaxation
- Alpha (8–12 Hz): Relaxation and daydreaming
- Beta (12–30 Hz): Focused, alert thinking
- Gamma (30–40 Hz): Higher cognitive functions

Author: porfanid
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import signal

# EEG Frequency Band Definitions
EEG_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 40.0),
}


class BandPowerAnalyzer:
    """
    Analyzer for computing EEG band power percentages.

    Computes the relative power in each frequency band (Delta, Theta, Alpha,
    Beta, Gamma) for EEG signals. Can analyze individual time windows to
    provide real-time band power percentages.

    Attributes:
        bands (Dict[str, Tuple[float, float]]): Frequency band definitions
        sfreq (float): Sampling frequency of the data
    """

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the BandPowerAnalyzer.

        Args:
            bands: Dictionary of band names to (low_freq, high_freq) tuples.
                   Uses default EEG bands if None.
        """
        self.bands = bands if bands is not None else EEG_BANDS.copy()
        self.sfreq: Optional[float] = None

    def compute_band_power_welch(
        self,
        data: np.ndarray,
        sfreq: float,
        nperseg: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute relative band power using Welch's method.

        Args:
            data: 1D array of EEG signal data
            sfreq: Sampling frequency in Hz
            nperseg: Length of each segment for Welch's method.
                     If None, uses min(256, len(data)//4)

        Returns:
            Dictionary mapping band names to relative power percentages (0-100)
        """
        if len(data) == 0:
            return {band: 0.0 for band in self.bands}

        # Ensure we have enough data for spectral analysis
        if nperseg is None:
            nperseg = min(256, max(64, len(data) // 4))

        # Ensure nperseg is not larger than data length
        nperseg = min(nperseg, len(data))

        try:
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(
                data,
                fs=sfreq,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling="density",
            )

            # Calculate total power in the EEG-relevant range (0.5 - 40 Hz)
            eeg_mask = (freqs >= 0.5) & (freqs <= 40.0)
            total_power = np.trapezoid(psd[eeg_mask], freqs[eeg_mask])

            if total_power == 0:
                return {band: 0.0 for band in self.bands}

            # Calculate power in each band
            band_powers = {}
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_mask = (freqs >= low_freq) & (freqs < high_freq)
                if np.any(band_mask):
                    band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                    band_powers[band_name] = (band_power / total_power) * 100.0
                else:
                    band_powers[band_name] = 0.0

            return band_powers

        except Exception as e:
            print(f"Error computing band power: {e}")
            return {band: 0.0 for band in self.bands}

    def compute_band_power_for_raw(
        self,
        raw: mne.io.Raw,
        channel_idx: int = 0,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute band power percentages for a specific channel of Raw data.

        Args:
            raw: MNE Raw object containing EEG data
            channel_idx: Index of the channel to analyze
            tmin: Start time in seconds (None for beginning)
            tmax: End time in seconds (None for end)

        Returns:
            Dictionary mapping band names to relative power percentages (0-100)
        """
        sfreq = raw.info["sfreq"]

        # Get data for the specified channel and time window
        data = raw.get_data(picks=[channel_idx])

        if tmin is not None or tmax is not None:
            start_sample = int((tmin or 0) * sfreq)
            end_sample = int((tmax or raw.times[-1]) * sfreq)
            data = data[:, start_sample:end_sample]

        if data.size == 0:
            return {band: 0.0 for band in self.bands}

        return self.compute_band_power_welch(data[0], sfreq)

    def compute_band_power_time_series(
        self,
        raw: mne.io.Raw,
        channel_idx: int = 0,
        window_duration: float = 1.0,
        overlap: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute band power percentages over time using sliding windows.

        This is useful for real-time or dynamic visualization of band power
        changes over time.

        Args:
            raw: MNE Raw object containing EEG data
            channel_idx: Index of the channel to analyze
            window_duration: Duration of each analysis window in seconds
            overlap: Overlap fraction between windows (0.0 to 0.9)

        Returns:
            Tuple of:
            - time_points: Array of time points (center of each window)
            - band_powers: Dictionary mapping band names to arrays of
                          percentages over time
        """
        sfreq = raw.info["sfreq"]
        data = raw.get_data(picks=[channel_idx])[0]
        n_samples = len(data)
        duration = n_samples / sfreq

        window_samples = int(window_duration * sfreq)
        step_samples = int(window_samples * (1 - overlap))

        if step_samples < 1:
            step_samples = 1

        # Initialize arrays for results
        n_windows = (n_samples - window_samples) // step_samples + 1
        if n_windows < 1:
            # Not enough data for even one window
            return np.array([duration / 2]), {
                band: np.array([power])
                for band, power in self.compute_band_power_welch(data, sfreq).items()
            }

        time_points = np.zeros(n_windows)
        band_powers = {band: np.zeros(n_windows) for band in self.bands}

        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            window_data = data[start:end]

            # Time point is center of window
            time_points[i] = (start + end) / 2 / sfreq

            # Compute band powers for this window
            powers = self.compute_band_power_welch(window_data, sfreq)
            for band in self.bands:
                band_powers[band][i] = powers[band]

        return time_points, band_powers

    def compute_average_band_power(
        self,
        raw: mne.io.Raw,
        channel_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute average band power across multiple channels.

        Args:
            raw: MNE Raw object containing EEG data
            channel_indices: List of channel indices to average.
                            If None, uses all channels.

        Returns:
            Dictionary mapping band names to average power percentages
        """
        if channel_indices is None:
            channel_indices = list(range(len(raw.ch_names)))

        if not channel_indices:
            return {band: 0.0 for band in self.bands}

        # Compute band powers for each channel
        all_powers = []
        for ch_idx in channel_indices:
            powers = self.compute_band_power_for_raw(raw, channel_idx=ch_idx)
            all_powers.append(powers)

        # Average across channels
        avg_powers = {}
        for band in self.bands:
            avg_powers[band] = np.mean([p[band] for p in all_powers])

        return avg_powers

    def compute_band_power_comparison(
        self,
        original_raw: mne.io.Raw,
        cleaned_raw: mne.io.Raw,
        channel_idx: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare band power between original and cleaned signals.

        This is useful for showing the effect of artifact removal on
        the frequency composition of the signal.

        Args:
            original_raw: Original MNE Raw object
            cleaned_raw: Cleaned MNE Raw object
            channel_idx: Index of the channel to analyze

        Returns:
            Dictionary with 'original' and 'cleaned' keys, each containing
            band power percentages
        """
        original_powers = self.compute_band_power_for_raw(
            original_raw, channel_idx=channel_idx
        )
        cleaned_powers = self.compute_band_power_for_raw(
            cleaned_raw, channel_idx=channel_idx
        )

        return {
            "original": original_powers,
            "cleaned": cleaned_powers,
        }

    def get_band_colors(self) -> Dict[str, str]:
        """
        Get suggested colors for each band for visualization.

        Returns:
            Dictionary mapping band names to hex color codes
        """
        return {
            "Delta": "#2E86AB",  # Blue - Deep sleep
            "Theta": "#A23B72",  # Magenta - Light sleep
            "Alpha": "#F18F01",  # Orange - Relaxation
            "Beta": "#C73E1D",  # Red - Focus
            "Gamma": "#6B2737",  # Dark red - Cognition
        }

    def get_band_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for each frequency band.

        Returns:
            Dictionary mapping band names to descriptions
        """
        return {
            "Delta": "Deep sleep (0.5-4 Hz)",
            "Theta": "Light sleep/relaxation (4-8 Hz)",
            "Alpha": "Relaxation/daydreaming (8-12 Hz)",
            "Beta": "Focused thinking (12-30 Hz)",
            "Gamma": "Higher cognition (30-40 Hz)",
        }
