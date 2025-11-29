#!/usr/bin/env python3
"""
PCA Processor - Principal Component Analysis for EEG artifact cleaning
======================================================================

Implements Principal Component Analysis (PCA) for:
- Training PCA models on EEG data
- Identifying artifacts through variance analysis
- Removing selected components
- Reconstructing clean signals

PCA is faster than ICA and can be useful for quick preliminary analysis
or when ICA assumptions may not hold.

Author: porfanid
Version: 1.0
"""

import warnings
from typing import List, Optional

import mne
import numpy as np
from sklearn.decomposition import PCA

from .base_processor import BaseComponentProcessor

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PCAProcessor(BaseComponentProcessor):
    """
    PCA Processor for EEG component analysis and artifact removal

    Uses Principal Component Analysis to decompose EEG signals into
    orthogonal components ordered by variance. This allows identification
    and removal of high-variance artifacts while preserving the main
    signal characteristics.

    Attributes:
        n_components (int): Number of PCA components
        random_state (int): Seed for reproducibility
        pca (sklearn.decomposition.PCA): The fitted PCA model
        raw_data (mne.io.Raw): The training data
        components_info (dict): Information about each component
        _sources (np.ndarray): Cached component signals
        _original_data (np.ndarray): Original data for reconstruction
    """

    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Initialize PCA processor

        Args:
            n_components (int, optional): Number of PCA components.
                                         If None, determined automatically.
            random_state (int): Seed for reproducibility
        """
        super().__init__(n_components, random_state)
        self.pca: Optional[PCA] = None
        self._sources: Optional[np.ndarray] = None
        self._original_data: Optional[np.ndarray] = None

    def fit(self, raw: mne.io.Raw) -> bool:
        """
        Fit PCA model to EEG data

        Trains a PCA model on the provided EEG data. PCA decomposes
        the signals into orthogonal components ordered by variance,
        which can help identify artifact-related components.

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

            # Determine number of components
            # PCA can have at most min(n_channels, n_samples) components
            n_channels = len(raw.ch_names)
            n_samples = data.shape[1]
            max_components = min(n_channels, n_samples)

            if self.n_components is None:
                self.n_components = n_channels
            else:
                self.n_components = min(self.n_components, max_components)

            # Create and fit PCA
            self.pca = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                svd_solver="full",
            )

            # Transform data - transpose because sklearn expects (samples, features)
            # For EEG: samples=timepoints, features=channels
            self._sources = self.pca.fit_transform(data.T).T

            # Calculate component information
            self._calculate_component_info()

            return True

        except Exception as e:
            print(f"Error during PCA fitting: {str(e)}")
            return False

    def get_sources_data(self) -> Optional[np.ndarray]:
        """
        Get all PCA component signals

        Returns:
            Optional[np.ndarray]: Array with shape (n_components, n_timepoints) or None
        """
        return self._sources

    def get_components(self) -> Optional[np.ndarray]:
        """
        Get the PCA component vectors (spatial patterns)

        Returns the mixing matrix that shows how each component
        contributes to each channel.

        Returns:
            Optional[np.ndarray]: Array of shape (n_channels, n_components) or None
        """
        if self.pca is None:
            return None
        # PCA components_ is (n_components, n_features) = (n_components, n_channels)
        # We need (n_channels, n_components) for compatibility with ICA
        return self.pca.components_.T

    def get_mixing_matrix(self) -> Optional[np.ndarray]:
        """Get the mixing matrix (how components combine to form channels)"""
        if self.pca is None:
            return None
        return self.pca.components_.T

    def get_unmixing_matrix(self) -> Optional[np.ndarray]:
        """Get the unmixing matrix (how channels decompose into components)"""
        if self.pca is None:
            return None
        return self.pca.components_

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get the explained variance ratio for each component

        Returns:
            Optional[np.ndarray]: Array with variance ratio for each component
        """
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def apply_artifact_removal(
        self, components_to_remove: List[int]
    ) -> Optional[mne.io.Raw]:
        """
        Apply artifact removal by zeroing out selected components

        Reconstructs the signal without the specified components,
        effectively removing their contribution from the data.

        Args:
            components_to_remove (List[int]): List of component indices to remove

        Returns:
            Optional[mne.io.Raw]: Cleaned Raw data or None if failed
        """
        if self.pca is None or self.raw_data is None or self._sources is None:
            return None

        try:
            # Create copy of sources
            cleaned_sources = self._sources.copy()

            # Zero out components to remove
            for comp_idx in components_to_remove:
                if 0 <= comp_idx < self.n_components:
                    cleaned_sources[comp_idx] = 0

            # Inverse transform back to channel space
            # cleaned_sources is (n_components, n_timepoints)
            # We need (n_timepoints, n_components) for inverse_transform
            cleaned_data = self.pca.inverse_transform(cleaned_sources.T).T

            # Create cleaned raw object
            cleaned_raw = self.raw_data.copy()
            cleaned_raw._data = cleaned_data

            return cleaned_raw

        except Exception as e:
            print(f"Error during artifact removal: {str(e)}")
            return None

    def get_method_name(self) -> str:
        """
        Get the name of this analysis method

        Returns:
            str: "PCA"
        """
        return "PCA"

    def get_pca_object(self) -> Optional[PCA]:
        """
        Get the underlying sklearn PCA object

        Returns:
            Optional[PCA]: The fitted PCA model or None
        """
        return self.pca
