#!/usr/bin/env python3
"""
ICA Processor - Independent Component Analysis for EEG artifact cleaning
=========================================================================

Implements Independent Component Analysis (ICA) method for:
- Training ICA models on EEG data
- Identifying artifacts (blinks, movement, muscle)
- Removing selected components
- Reconstructing clean signals

Author: porfanid
Version: 1.1
"""

import warnings
from typing import Dict, List, Optional

import mne
import numpy as np
from scipy import stats

from .base_processor import BaseComponentProcessor

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ICAProcessor(BaseComponentProcessor):
    """
    ICA Processor for artifact detection and removal.

    Uses the FastICA algorithm for analyzing EEG signals into
    independent components, enabling the detection and removal of
    artifacts such as eye blinks, movement and muscle signals.

    Attributes:
        n_components (int): Number of ICA components
        random_state (int): Seed for reproducibility
        ica (mne.preprocessing.ICA): The trained ICA model
        raw_data (mne.io.Raw): The training data
        components_info (dict): Information about the components
    """

    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Initialize ICA processor.

        Args:
            n_components (int, optional): Number of ICA components.
                                        If None, determined automatically.
            random_state (int): Seed for reproducibility
        """
        super().__init__(n_components, random_state)
        self.ica: Optional[mne.preprocessing.ICA] = None

    def fit(self, raw: mne.io.Raw) -> bool:
        """
        Fit ICA model to EEG data (implements BaseComponentProcessor interface)

        Args:
            raw (mne.io.Raw): Filtered Raw EEG data

        Returns:
            bool: True if fitting was successful, False otherwise
        """
        return self.fit_ica(raw)

    def fit_ica(self, raw: mne.io.Raw) -> bool:
        """
        Train ICA model.

        Trains an ICA model on the provided EEG data using
        the FastICA algorithm. The model decomposes signals into
        independent components representing different activity sources.

        Args:
            raw (mne.io.Raw): Filtered Raw EEG data

        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            self.raw_data = raw.copy()

            # Automatic determination of component count if not provided
            if self.n_components is None:
                self.n_components = min(len(raw.ch_names), len(raw.ch_names))
            else:
                # Ensure we don't exceed the number of channels
                self.n_components = min(self.n_components, len(raw.ch_names))

            # Create and train ICA
            self.ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                method="fastica",
                random_state=self.random_state,
                max_iter=1000,
                verbose=False,
            )

            if self.ica is not None:
                self.ica.fit(raw, verbose=False)
            else:
                raise RuntimeError("ICA initialization failed")

            # Calculate component information
            self._calculate_component_info()

            return True

        except Exception as e:
            print(f"Error during ICA training: {str(e)}")
            return False

    def _calculate_component_info(self):
        """
        Calculate statistical information for each ICA component.

        Calculates basic statistics for each component such as variance,
        kurtosis, range, etc. used for artifact detection.
        """
        if self.ica is None or self.raw_data is None:
            return

        sources = self.ica.get_sources(self.raw_data).get_data()

        for i in range(self.n_components):
            comp_data = sources[i]

            self.components_info[i] = {
                "variance": float(np.var(comp_data)),
                "kurtosis": float(abs(stats.kurtosis(comp_data))),
                "range": float(np.ptp(comp_data)),
                "std": float(np.std(comp_data)),
                "mean": float(np.mean(comp_data)),
                "rms": float(np.sqrt(np.mean(comp_data**2))),
                "skewness": float(abs(stats.skew(comp_data))),
            }

    def get_component_info(self, component_idx: int) -> Dict[str, float]:
        """
        Return information for a specific component.

        Args:
            component_idx (int): Component index (0-based)

        Returns:
            Dict[str, float]: Dictionary with statistical information such as
                            variance, kurtosis, range, std, mean, rms, skewness
        """
        default_info: Dict[str, float] = {}
        return self.components_info.get(component_idx, default_info)

    def get_all_components_info(self) -> Dict[int, Dict[str, float]]:
        """
        Return information for all components.

        Returns:
            Dict[int, Dict[str, float]]: Dictionary with information for all components
        """
        return self.components_info

    def get_component_data(self, component_idx: int) -> Optional[np.ndarray]:
        """
        Return data for a specific component.

        Extracts the time series of the selected ICA component.

        Args:
            component_idx (int): Component index

        Returns:
            Optional[np.ndarray]: Component data as 1D array or None if failed
        """
        if self.ica is None or self.raw_data is None:
            return None

        try:
            sources = self.ica.get_sources(self.raw_data).get_data()
            return sources[component_idx]
        except IndexError:
            return None

    def apply_artifact_removal(
        self, components_to_remove: List[int]
    ) -> Optional[mne.io.Raw]:
        """
        Apply artifact removal.

        Removes the selected ICA components from the original data,
        reconstructing the clean signal without the artifacts.

        Args:
            components_to_remove (List[int]): List of component indices to remove

        Returns:
            Optional[mne.io.Raw]: Cleaned Raw data or None if failed
        """
        if self.ica is None or self.raw_data is None:
            return None

        try:
            # Create copy for cleaning
            cleaned_raw = self.raw_data.copy()

            # Set components to remove
            self.ica.exclude = components_to_remove

            # Apply cleaning
            cleaned_raw = self.ica.apply(cleaned_raw, verbose=False)

            return cleaned_raw

        except Exception as e:
            print(f"Error during cleaning: {str(e)}")
            return None

    def get_ica_object(self) -> Optional[mne.preprocessing.ICA]:
        """
        Return the ICA object.

        Returns:
            Optional[mne.preprocessing.ICA]: The trained ICA model or None
        """
        return self.ica

    def get_sources_data(self) -> Optional[np.ndarray]:
        """
        Return all ICA sources.

        Extracts all ICA components as a data matrix.

        Returns:
            Optional[np.ndarray]: Matrix with shape (n_components, n_timepoints) or None
        """
        if self.ica is None or self.raw_data is None:
            return None

        return self.ica.get_sources(self.raw_data).get_data()

    def get_mixing_matrix(self) -> Optional[np.ndarray]:
        """Return the mixing matrix."""
        if self.ica is None:
            return None
        return self.ica.mixing_

    def get_unmixing_matrix(self) -> Optional[np.ndarray]:
        """Return the unmixing matrix."""
        if self.ica is None:
            return None
        return self.ica.unmixing_

    def get_components(self) -> Optional[np.ndarray]:
        """
        Get the ICA component vectors (spatial patterns)

        Returns the component matrix showing how each component
        contributes to each channel.

        Returns:
            Optional[np.ndarray]: Array of shape (n_channels, n_components) or None
        """
        if self.ica is None:
            return None
        return self.ica.get_components()

    def get_method_name(self) -> str:
        """
        Get the name of this analysis method

        Returns:
            str: "ICA"
        """
        return "ICA"
