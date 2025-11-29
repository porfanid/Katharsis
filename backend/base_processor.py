#!/usr/bin/env python3
"""
Base Processor - Abstract base class for component analysis processors
===================================================================

Provides the common interface and functionality for ICA and PCA processors.
This enables modular design where different analysis methods can be used
interchangeably in the application.

Author: porfanid
Version: 1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import mne
import numpy as np
from scipy import stats


class BaseComponentProcessor(ABC):
    """
    Abstract base class for component analysis processors

    Provides a common interface for different decomposition methods (ICA, PCA)
    used in EEG artifact cleaning. Subclasses must implement the specific
    analysis methods while using the common infrastructure for component
    management and visualization.

    Attributes:
        n_components (int): Number of components to extract
        random_state (int): Seed for reproducibility
        raw_data (mne.io.Raw): The training data
        components_info (dict): Statistical information about each component
    """

    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Initialize the component processor

        Args:
            n_components (int, optional): Number of components to extract.
                                         If None, determined automatically.
            random_state (int): Seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.raw_data: Optional[mne.io.Raw] = None
        self.components_info: Dict[int, Dict[str, float]] = {}

    @abstractmethod
    def fit(self, raw: mne.io.Raw) -> bool:
        """
        Fit the component analysis model to the data

        Args:
            raw (mne.io.Raw): Filtered Raw EEG data

        Returns:
            bool: True if fitting was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_sources_data(self) -> Optional[np.ndarray]:
        """
        Get the source (component) signals

        Returns:
            Optional[np.ndarray]: Array of shape (n_components, n_timepoints) or None
        """
        pass

    @abstractmethod
    def get_components(self) -> Optional[np.ndarray]:
        """
        Get the spatial patterns (mixing weights) for each component

        Returns:
            Optional[np.ndarray]: Array of shape (n_channels, n_components) or None
        """
        pass

    @abstractmethod
    def apply_artifact_removal(
        self, components_to_remove: List[int]
    ) -> Optional[mne.io.Raw]:
        """
        Apply artifact removal by excluding specified components

        Args:
            components_to_remove (List[int]): Indices of components to remove

        Returns:
            Optional[mne.io.Raw]: Cleaned Raw data or None if failed
        """
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """
        Get the name of the analysis method

        Returns:
            str: Name of the method (e.g., "ICA", "PCA")
        """
        pass

    def _calculate_component_info(self):
        """
        Calculate statistical information for each component

        Computes variance, kurtosis, range, std, mean, rms, and skewness
        for each component which are used for artifact detection.
        """
        sources = self.get_sources_data()
        if sources is None:
            return

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
        Get information for a specific component

        Args:
            component_idx (int): Component index (0-based)

        Returns:
            Dict[str, float]: Dictionary with statistical information
        """
        default_info: Dict[str, float] = {}
        return self.components_info.get(component_idx, default_info)

    def get_all_components_info(self) -> Dict[int, Dict[str, float]]:
        """
        Get information for all components

        Returns:
            Dict[int, Dict[str, float]]: Dictionary with information for all components
        """
        return self.components_info

    def get_component_data(self, component_idx: int) -> Optional[np.ndarray]:
        """
        Get data for a specific component

        Args:
            component_idx (int): Component index

        Returns:
            Optional[np.ndarray]: Component data as 1D array or None if failed
        """
        sources = self.get_sources_data()
        if sources is None:
            return None

        try:
            return sources[component_idx]
        except IndexError:
            return None
