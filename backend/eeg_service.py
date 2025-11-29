#!/usr/bin/env python3
"""
EEG Artifact Cleaning Service - Κεντρική υπηρεσία backend
========================================================

Η κεντρική υπηρεσία που ενοποιεί όλες τις λειτουργίες καθαρισμού EEG:
- Διαχείριση φόρτωσης και επεξεργασίας αρχείων
- Εκτέλεση ICA ή PCA ανάλυσης
- Αυτόματος εντοπισμός artifacts
- Καθαρισμός και αποθήκευση δεδομένων
- Progress tracking και status updates

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
    Κεντρική υπηρεσία για EEG artifact cleaning

    Συνδυάζει όλες τις λειτουργίες καθαρισμού EEG σε μια ενιαία υπηρεσία:
    - Φόρτωση και προεπεξεργασία δεδομένων
    - ICA ή PCA ανάλυση και εκπαίδευση μοντέλου
    - Αυτόματος εντοπισμός artifacts
    - Καθαρισμός και αποθήκευση αποτελεσμάτων
    - Progress tracking και callback system

    Attributes:
        backend_core (EEGBackendCore): Κεντρικό backend για I/O και preprocessing
        component_processor (BaseComponentProcessor): Επεξεργαστής ICA ή PCA
        artifact_detector (ArtifactDetector): Ανιχνευτής artifacts
        current_file (str): Τρέχον αρχείο που επεξεργάζεται
        is_processing (bool): Κατάσταση επεξεργασίας
        analysis_fitted (bool): Αν το μοντέλο έχει εκπαιδευτεί
        analysis_method (str): Η μέθοδος ανάλυσης ("ICA" ή "PCA")
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
        Αρχικοποίηση της υπηρεσίας καθαρισμού EEG

        Args:
            n_components (int, optional): Αριθμός συνιστωσών.
                                        Αν None, καθορίζεται αυτόματα.
            variance_threshold (float): Κατώφλι διακύμανσης για artifact detection
            kurtosis_threshold (float): Κατώφλι κύρτωσης για artifact detection
            range_threshold (float): Κατώφλι εύρους για artifact detection
            analysis_method (str): Μέθοδος ανάλυσης ("ICA" ή "PCA"), default "ICA"
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

        # Callbacks για progress updates
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
        """Ορισμός callback για progress updates"""
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable[[str], None]):
        """Ορισμός callback για status updates"""
        self.status_callback = callback

    def _update_progress(self, progress: int):
        """Ενημέρωση progress"""
        if self.progress_callback:
            self.progress_callback(progress)

    def _update_status(self, status: str):
        """Ενημέρωση status"""
        if self.status_callback:
            self.status_callback(status)

    def load_and_prepare_file(
        self,
        file_path: str,
        selected_channels: Optional[List[str]] = None,
        analysis_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Φόρτωση και προετοιμασία αρχείου για επεξεργασία

        Args:
            file_path: Διαδρομή αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)
            analysis_method: Μέθοδος ανάλυσης ("ICA" ή "PCA"), None για χρήση της προεπιλεγμένης

        Returns:
            Dictionary με αποτελέσματα φόρτωσης
        """
        self.is_processing = True
        self.current_file = file_path
        self.analysis_fitted = False

        # Set analysis method if provided
        if analysis_method:
            self.set_analysis_method(analysis_method)

        try:
            self._update_status("Φόρτωση δεδομένων...")
            self._update_progress(10)

            # Φόρτωση αρχείου με επιλεγμένα κανάλια
            result = self.backend_core.load_file(file_path, selected_channels)

            if not result["success"]:
                self.is_processing = False
                return result

            # Ενημερώνουμε τον processor με τον αριθμό καναλιών
            self._n_components = None  # Αυτόματη ανίχνευση
            self._create_processor()

            self._update_progress(30)
            self._update_status("Αρχείο φορτώθηκε επιτυχώς")

            return result

        except Exception as e:
            self.is_processing = False
            return {"success": False, "error": f"Σφάλμα φόρτωσης: {str(e)}"}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Λήψη πληροφοριών αρχείου για επιλογή καναλιών

        Args:
            file_path: Διαδρομή αρχείου

        Returns:
            Dictionary με πληροφορίες αρχείου
        """
        return self.backend_core.get_file_info(file_path)

    def fit_analysis(self) -> Dict[str, Any]:
        """
        Εκτέλεση ανάλυσης (ICA ή PCA)

        Returns:
            Dictionary με αποτελέσματα ανάλυσης
        """
        if not self.is_processing:
            return {"success": False, "error": "Δεν έχει φορτωθεί αρχείο"}

        try:
            method_name = self.component_processor.get_method_name()
            self._update_status(f"Εκτέλεση {method_name} ανάλυσης...")
            self._update_progress(50)

            # Λήψη φιλτραρισμένων δεδομένων
            filtered_data = self.backend_core.get_filtered_data()
            if filtered_data is None:
                return {
                    "success": False,
                    "error": "Δεν υπάρχουν φιλτραρισμένα δεδομένα",
                }

            # Εκπαίδευση μοντέλου
            success = self.component_processor.fit(filtered_data)

            if not success:
                return {
                    "success": False,
                    "error": f"Αποτυχία εκπαίδευσης {method_name}",
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
            return {"success": False, "error": f"Σφάλμα ανάλυσης: {str(e)}"}

    def fit_ica_analysis(self) -> Dict[str, Any]:
        """
        Εκτέλεση ICA ανάλυσης (backward compatible method)

        Returns:
            Dictionary με αποτελέσματα ICA
        """
        # Ensure we're using ICA
        if self._analysis_method != "ICA":
            self.set_analysis_method("ICA")
        return self.fit_analysis()

    def fit_pca_analysis(self) -> Dict[str, Any]:
        """
        Εκτέλεση PCA ανάλυσης

        Returns:
            Dictionary με αποτελέσματα PCA
        """
        # Ensure we're using PCA
        if self._analysis_method != "PCA":
            self.set_analysis_method("PCA")
        return self.fit_analysis()

    def detect_artifacts(self, max_components: int = 3) -> Dict[str, Any]:
        """
        Εντοπισμός artifacts με πολλαπλές μεθόδους

        Args:
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση

        Returns:
            Dictionary με αποτελέσματα εντοπισμού
        """
        if not self.analysis_fitted:
            return {"success": False, "error": "Η ανάλυση δεν έχει εκτελεστεί"}

        try:
            self._update_status("Εντοπισμός artifacts...")
            self._update_progress(80)

            # Λήψη φιλτραρισμένων δεδομένων
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

            # Δημιουργία επεξηγήσεων
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
            return {"success": False, "error": f"Σφάλμα εντοπισμού artifacts: {str(e)}"}

    def apply_artifact_removal(self, components_to_remove: List[int]) -> Dict[str, Any]:
        """
        Εφαρμογή αφαίρεσης artifacts

        Args:
            components_to_remove: Λίστα συνιστωσών προς αφαίρεση

        Returns:
            Dictionary με αποτελέσματα
        """
        if not self.analysis_fitted:
            return {"success": False, "error": "Η ανάλυση δεν έχει εκτελεστεί"}

        try:
            self._update_status("Εφαρμογή καθαρισμού...")
            self._update_progress(95)

            # Εφαρμογή καθαρισμού
            cleaned_data = self.component_processor.apply_artifact_removal(
                components_to_remove
            )

            if cleaned_data is None:
                return {"success": False, "error": "Αποτυχία καθαρισμού δεδομένων"}

            # Υπολογισμός στατιστικών πριν/μετά
            original_stats = self.backend_core.preprocessor.get_data_statistics(
                self.backend_core.get_filtered_data()
            )
            cleaned_stats = self.backend_core.preprocessor.get_data_statistics(
                cleaned_data
            )

            self._update_progress(100)
            self._update_status("Καθαρισμός ολοκληρώθηκε")

            return {
                "success": True,
                "cleaned_data": cleaned_data,
                "components_removed": components_to_remove,
                "original_stats": original_stats,
                "cleaned_stats": cleaned_stats,
            }

        except Exception as e:
            return {"success": False, "error": f"Σφάλμα καθαρισμού: {str(e)}"}

    def save_cleaned_data(self, cleaned_data: mne.io.Raw, output_path: str) -> bool:
        """
        Αποθήκευση καθαρισμένων δεδομένων

        Args:
            cleaned_data: Καθαρισμένα δεδομένα
            output_path: Διαδρομή εξόδου

        Returns:
            bool: True εάν η αποθήκευση ήταν επιτυχής
        """
        return self.backend_core.save_cleaned_data(cleaned_data, output_path)

    def get_component_visualization_data(self) -> Optional[Dict[str, Any]]:
        """
        Λήψη δεδομένων για οπτικοποίηση συνιστωσών

        Returns:
            Dictionary με δεδομένα για plots ή None
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
        """Επαναφορά κατάστασης service"""
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
        Επιστροφή περίληψης επεξεργασίας

        Returns:
            Dictionary με περίληψη
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
