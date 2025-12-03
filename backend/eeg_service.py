#!/usr/bin/env python3
"""
EEG Artifact Cleaning Service - Central backend service
========================================================

The central service that unifies all EEG cleaning functions:
- File loading and processing management
- ICA or PCA analysis execution
- Automatic artifact detection
- Data cleaning and saving
- Progress tracking and status updates

Author: porfanid
Version: 1.1
"""

from typing import Any, Callable, Dict, List, Optional

import mne

from .artifact_detector import ArtifactDetector
from .base_processor import BaseComponentProcessor
from .eeg_backend import EEGBackendCore
from .ica_processor import ICAProcessor
from .pca_processor import PCAProcessor


class EEGArtifactCleaningService:
    """
    Central service for EEG artifact cleaning.

    Combines all EEG cleaning functions into a unified service:
    - Data loading and preprocessing
    - ICA or PCA analysis and model training
    - Automatic artifact detection
    - Cleaning and saving results
    - Progress tracking and callback system

    Attributes:
        backend_core (EEGBackendCore): Central backend for I/O and preprocessing
        component_processor (BaseComponentProcessor): ICA or PCA processor
        artifact_detector (ArtifactDetector): Artifact detector
        current_file (str): Current file being processed
        is_processing (bool): Processing state
        analysis_fitted (bool): Whether the model has been trained
        analysis_method (str): Analysis method ("ICA" or "PCA")
    """

    def __init__(
        self,
        n_components: int = None,
        variance_threshold: float = 2.0,
        kurtosis_threshold: float = 2.0,
        range_threshold: float = 3.0,
        analysis_method: str = "ICA",
    ):
        """
        Initialize the EEG cleaning service.

        Args:
            n_components (int, optional): Number of components.
                                        If None, determined automatically.
            variance_threshold (float): Variance threshold for artifact detection
            kurtosis_threshold (float): Kurtosis threshold for artifact detection
            range_threshold (float): Range threshold for artifact detection
            analysis_method (str): Analysis method ("ICA" or "PCA"), default "ICA"
        """
        self.backend_core = EEGBackendCore()
        self._n_components = n_components
        self._analysis_method = analysis_method.upper()

        # Create the appropriate processor based on method
        self._create_processor()

        self.artifact_detector = ArtifactDetector(
            variance_threshold=variance_threshold,
            kurtosis_threshold=kurtosis_threshold,
            range_threshold=range_threshold,
        )

        # Callbacks for progress updates
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None

        # State
        self.current_file: Optional[str] = None
        self.is_processing = False
        self.analysis_fitted = False
        self.suggested_artifacts: List[int] = []
        self.detection_methods_results: Dict[str, List[int]] = {}

    def _create_processor(self):
        """Create the appropriate component processor based on analysis method"""
        if self._analysis_method == "PCA":
            self.component_processor: BaseComponentProcessor = PCAProcessor(
                n_components=self._n_components
            )
        else:
            self.component_processor = ICAProcessor(n_components=self._n_components)

    @property
    def analysis_method(self) -> str:
        """Get the current analysis method"""
        return self._analysis_method

    @analysis_method.setter
    def analysis_method(self, value: str):
        """Set the analysis method and recreate processor if needed"""
        new_method = value.upper()
        if new_method not in ["ICA", "PCA"]:
            raise ValueError("Analysis method must be 'ICA' or 'PCA'")
        if new_method != self._analysis_method:
            self._analysis_method = new_method
            self._create_processor()
            self.analysis_fitted = False

    def set_analysis_method(self, method: str):
        """
        Set the analysis method

        Args:
            method (str): "ICA" or "PCA"
        """
        self.analysis_method = method

    # Backward compatibility property
    @property
    def ica_processor(self) -> ICAProcessor:
        """Backward compatibility: returns component_processor as ICAProcessor"""
        if isinstance(self.component_processor, ICAProcessor):
            return self.component_processor
        # If PCA is being used, return a new ICAProcessor for compatibility
        # This is mainly for artifact detection which uses ICA-specific methods
        return ICAProcessor(n_components=self._n_components)

    @ica_processor.setter
    def ica_processor(self, value: ICAProcessor):
        """Backward compatibility setter"""
        self.component_processor = value

    # Backward compatibility property
    @property
    def ica_fitted(self) -> bool:
        """Backward compatibility: returns analysis_fitted"""
        return self.analysis_fitted

    @ica_fitted.setter
    def ica_fitted(self, value: bool):
        """Backward compatibility setter"""
        self.analysis_fitted = value

    def set_progress_callback(self, callback: Callable[[int], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self.status_callback = callback

    def _update_progress(self, progress: int):
        """Update progress."""
        if self.progress_callback:
            self.progress_callback(progress)

    def _update_status(self, status: str):
        """Update status."""
        if self.status_callback:
            self.status_callback(status)

    def load_and_prepare_file(
        self,
        file_path: str,
        selected_channels: Optional[List[str]] = None,
        analysis_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load and prepare file for processing.

        Args:
            file_path: File path
            selected_channels: List of selected channels (None for auto detection)
            analysis_method: Analysis method ("ICA" or "PCA"), None to use default

        Returns:
            Dictionary with loading results
        """
        self.is_processing = True
        self.current_file = file_path
        self.analysis_fitted = False

        # Set analysis method if provided
        if analysis_method:
            self.set_analysis_method(analysis_method)

        try:
            self._update_status("Loading data...")
            self._update_progress(10)

            # Load file with selected channels
            result = self.backend_core.load_file(file_path, selected_channels)

            if not result["success"]:
                self.is_processing = False
                return result

            # Update processor with channel count
            self._n_components = None  # Auto detection
            self._create_processor()

            self._update_progress(30)
            self._update_status("File loaded successfully")

            return result

        except Exception as e:
            self.is_processing = False
            return {"success": False, "error": f"Loading error: {str(e)}"}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file information for channel selection.

        Args:
            file_path: File path

        Returns:
            Dictionary with file information
        """
        return self.backend_core.get_file_info(file_path)

    def load_from_raw(
        self,
        raw: mne.io.Raw,
        analysis_method: Optional[str] = None,
        already_filtered: bool = False,
    ) -> Dict[str, Any]:
        """
        Load and prepare from a pre-loaded Raw object.

        This is used when the signal has been modified (e.g., regions cut)
        in the signal preview screen before processing.

        Args:
            raw: Pre-loaded MNE Raw object
            analysis_method: Analysis method ("ICA" or "PCA"), None to use default
            already_filtered: If True, skip band-pass filtering (data already preprocessed)

        Returns:
            Dictionary with loading results
        """
        self.is_processing = True
        self.current_file = "modified_signal"
        self.analysis_fitted = False

        # Set analysis method if provided
        if analysis_method:
            self.set_analysis_method(analysis_method)

        try:
            self._update_status("Preparing signal data...")
            self._update_progress(10)

            # Load raw data directly into backend core
            # Pass already_filtered flag to skip redundant filtering
            result = self.backend_core.load_from_raw(raw, already_filtered=already_filtered)

            if not result["success"]:
                self.is_processing = False
                return result

            # Update processor with channel count
            self._n_components = None  # Auto detection
            self._create_processor()

            self._update_progress(30)
            self._update_status("Signal data loaded successfully")

            return result

        except Exception as e:
            self.is_processing = False
            return {"success": False, "error": f"Loading error: {str(e)}"}

    def fit_analysis(self) -> Dict[str, Any]:
        """
        Execute analysis (ICA or PCA).

        Returns:
            Dictionary with analysis results
        """
        if not self.is_processing:
            return {"success": False, "error": "No file loaded"}

        try:
            method_name = self.component_processor.get_method_name()
            self._update_status(f"Running {method_name} analysis...")
            self._update_progress(50)

            # Get filtered data
            filtered_data = self.backend_core.get_filtered_data()
            if filtered_data is None:
                return {
                    "success": False,
                    "error": "No filtered data available",
                }

            # Train model
            success = self.component_processor.fit(filtered_data)

            if not success:
                return {
                    "success": False,
                    "error": f"{method_name} training failed",
                }

            self.analysis_fitted = True
            self._update_progress(70)

            return {
                "success": True,
                "method": method_name,
                "n_components": self.component_processor.n_components,
                "components_info": self.component_processor.get_all_components_info(),
            }

        except Exception as e:
            return {"success": False, "error": f"Analysis error: {str(e)}"}

    def fit_ica_analysis(self) -> Dict[str, Any]:
        """
        Execute ICA analysis (backward compatible method).

        Returns:
            Dictionary with ICA results
        """
        # Ensure we're using ICA
        if self._analysis_method != "ICA":
            self.set_analysis_method("ICA")
        return self.fit_analysis()

    def fit_pca_analysis(self) -> Dict[str, Any]:
        """
        Execute PCA analysis.

        Returns:
            Dictionary with PCA results
        """
        # Ensure we're using PCA
        if self._analysis_method != "PCA":
            self.set_analysis_method("PCA")
        return self.fit_analysis()

    def detect_artifacts(self, max_components: int = 3) -> Dict[str, Any]:
        """
        Detect artifacts using multiple methods.

        Args:
            max_components: Maximum number of components to remove

        Returns:
            Dictionary with detection results
        """
        if not self.analysis_fitted:
            return {"success": False, "error": "Analysis has not been executed"}

        try:
            self._update_status("Detecting artifacts...")
            self._update_progress(80)

            # Get filtered data
            filtered_data = self.backend_core.get_filtered_data()

            # Use full multi-method artifact detection for both ICA and PCA
            # The artifact detector automatically uses appropriate methods for each
            suggested_artifacts, methods_results = (
                self.artifact_detector.detect_artifacts_multi_method(
                    self.component_processor, filtered_data, max_components
                )
            )

            self.suggested_artifacts = suggested_artifacts
            self.detection_methods_results = methods_results

            # Create explanations
            explanations = {}
            for i in range(self.component_processor.n_components):
                explanations[i] = self.artifact_detector.get_artifact_explanation(
                    i, methods_results
                )

            self._update_progress(90)

            return {
                "success": True,
                "suggested_artifacts": suggested_artifacts,
                "methods_results": methods_results,
                "explanations": explanations,
                "components_info": self.component_processor.get_all_components_info(),
            }

        except Exception as e:
            return {"success": False, "error": f"Artifact detection error: {str(e)}"}

    def apply_artifact_removal(self, components_to_remove: List[int]) -> Dict[str, Any]:
        """
        Apply artifact removal.

        Args:
            components_to_remove: List of components to remove

        Returns:
            Dictionary with results
        """
        if not self.analysis_fitted:
            return {"success": False, "error": "Analysis has not been executed"}

        try:
            self._update_status("Applying cleaning...")
            self._update_progress(95)

            # Apply cleaning
            cleaned_data = self.component_processor.apply_artifact_removal(
                components_to_remove
            )

            if cleaned_data is None:
                return {"success": False, "error": "Data cleaning failed"}

            # Calculate before/after statistics
            original_stats = self.backend_core.preprocessor.get_data_statistics(
                self.backend_core.get_filtered_data()
            )
            cleaned_stats = self.backend_core.preprocessor.get_data_statistics(
                cleaned_data
            )

            self._update_progress(100)
            self._update_status("Cleaning completed")

            return {
                "success": True,
                "cleaned_data": cleaned_data,
                "components_removed": components_to_remove,
                "original_stats": original_stats,
                "cleaned_stats": cleaned_stats,
            }

        except Exception as e:
            return {"success": False, "error": f"Cleaning error: {str(e)}"}

    def save_cleaned_data(self, cleaned_data: mne.io.Raw, output_path: str) -> bool:
        """
        Save cleaned data.

        Args:
            cleaned_data: Cleaned data
            output_path: Output path

        Returns:
            bool: True if saving was successful
        """
        return self.backend_core.save_cleaned_data(cleaned_data, output_path)

    def get_component_visualization_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data for component visualization.

        Returns:
            Dictionary with data for plots or None
        """
        if not self.analysis_fitted:
            return None

        result = {
            "raw": self.backend_core.get_filtered_data(),
            "components_info": self.component_processor.get_all_components_info(),
            "suggested_artifacts": self.suggested_artifacts,
            "explanations": {
                i: self.artifact_detector.get_artifact_explanation(
                    i, self.detection_methods_results
                )
                for i in range(self.component_processor.n_components)
            },
            "analysis_method": self._analysis_method,
        }

        # Add method-specific data
        if isinstance(self.component_processor, ICAProcessor):
            result["ica"] = self.component_processor.get_ica_object()
            result["processor"] = self.component_processor
        elif isinstance(self.component_processor, PCAProcessor):
            result["pca"] = self.component_processor.get_pca_object()
            result["processor"] = self.component_processor

        return result

    def reset_state(self):
        """Reset service state."""
        self.is_processing = False
        self.analysis_fitted = False
        self.current_file = None
        self.suggested_artifacts = []
        self.detection_methods_results = {}

        # Reset backend components
        self.backend_core = EEGBackendCore()
        self._n_components = None
        self._create_processor()

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Return processing summary.

        Returns:
            Dictionary with summary
        """
        return {
            "current_file": self.current_file,
            "is_processing": self.is_processing,
            "ica_fitted": self.analysis_fitted,  # Backward compatibility
            "analysis_fitted": self.analysis_fitted,
            "analysis_method": self._analysis_method,
            "n_components": self.component_processor.n_components,
            "suggested_artifacts": self.suggested_artifacts,
            "detection_methods": (
                list(self.detection_methods_results.keys())
                if self.detection_methods_results
                else []
            ),
        }
