#!/usr/bin/env python3
"""
EEG Artifact Cleaning Service - Κεντρική υπηρεσία backend
========================================================

Η κεντρική υπηρεσία που ενοποιεί όλες τις λειτουργίες καθαρισμού EEG:
- Διαχείριση φόρτωσης και επεξεργασίας αρχείων
- Εκτέλεση ICA ανάλυσης
- Αυτόματος εντοπισμός artifacts
- Καθαρισμός και αποθήκευση δεδομένων
- Progress tracking και status updates

Author: porfanid
Version: 1.0
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import mne

from .artifact_detector import ArtifactDetector
from .eeg_backend import EEGBackendCore
from .ica_processor import ICAProcessor
from .data_consistency_utils import validate_raw_consistency, fix_raw_consistency, diagnose_ica_data_issues


class EEGArtifactCleaningService:
    """
    Κεντρική υπηρεσία για EEG artifact cleaning

    Συνδυάζει όλες τις λειτουργίες καθαρισμού EEG σε μια ενιαία υπηρεσία:
    - Φόρτωση και προεπεξεργασία δεδομένων
    - ICA ανάλυση και εκπαίδευση μοντέλου
    - Αυτόματος εντοπισμός artifacts
    - Καθαρισμός και αποθήκευση αποτελεσμάτων
    - Progress tracking και callback system

    Attributes:
        backend_core (EEGBackendCore): Κεντρικό backend για I/O και preprocessing
        ica_processor (ICAProcessor): Επεξεργαστής ICA
        artifact_detector (ArtifactDetector): Ανιχνευτής artifacts
        current_file (str): Τρέχον αρχείο που επεξεργάζεται
        is_processing (bool): Κατάσταση επεξεργασίας
        ica_fitted (bool): Αν το ICA μοντέλο έχει εκπαιδευτεί
    """

    def __init__(
        self,
        n_components: int = None,
        variance_threshold: float = 2.0,
        kurtosis_threshold: float = 2.0,
        range_threshold: float = 3.0,
    ):
        """
        Αρχικοποίηση της υπηρεσίας καθαρισμού EEG

        Args:
            n_components (int, optional): Αριθμός ICA συνιστωσών.
                                        Αν None, καθορίζεται αυτόματα.
            variance_threshold (float): Κατώφλι διακύμανσης για artifact detection
            kurtosis_threshold (float): Κατώφλι κύρτωσης για artifact detection
            range_threshold (float): Κατώφλι εύρους για artifact detection
        """
        self.backend_core = EEGBackendCore()
        self.ica_processor = ICAProcessor(n_components=n_components)
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
        self.ica_fitted = False
        self.suggested_artifacts: List[int] = []
        self.detection_methods_results: Dict[str, List[int]] = {}

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

    def load_preprocessed_data(self, raw_data: mne.io.Raw) -> Dict[str, Any]:
        """
        Set preprocessed raw data for ICA analysis with data consistency validation
        
        Args:
            raw_data: Preprocessed MNE Raw object
            
        Returns:
            Dictionary with success status and any fixes applied
        """
        self.is_processing = True
        self.ica_fitted = False
        
        try:
            self._update_status("Validating preprocessed data...")
            self._update_progress(10)
            
            # Critical: Validate and fix data consistency before using
            consistency_check = validate_raw_consistency(raw_data)
            fix_info = {'applied_fixes': []}
            
            if not consistency_check['valid']:
                self._update_status(f"Fixing data consistency issue: {consistency_check['error']}")
                raw_data, fix_result = fix_raw_consistency(raw_data, strategy='auto')
                fix_info = fix_result
                
                if fix_result['status'] != 'fixed':
                    return {
                        "success": False, 
                        "error": f"Could not fix data consistency: {fix_result.get('error', 'Unknown error')}\n💡 Try reloading your data or using different preprocessing settings"
                    }
                
                # Re-validate after fix
                final_check = validate_raw_consistency(raw_data)
                if not final_check['valid']:
                    return {
                        "success": False,
                        "error": f"Data consistency fix failed: {final_check['error']}\n💡 Data may be corrupted - try reloading from original file"
                    }
            
            self._update_status("Accepting preprocessed data...")
            self._update_progress(20)
            
            # Set the preprocessed data in backend core
            self.backend_core.raw_data = raw_data
            # Since the data is already preprocessed (filtered), set it as filtered_data too
            self.backend_core.filtered_data = raw_data
            self.backend_core.data = raw_data.get_data()
            self.backend_core.info = raw_data.info
            self.backend_core.sfreq = raw_data.info['sfreq']
            self.backend_core.channels = raw_data.ch_names
            
            # Update ICA processor with channel count
            n_channels = len(raw_data.ch_names)
            self.ica_processor = ICAProcessor(n_components=None)  # Auto-detect
            
            self._update_progress(30)
            self._update_status("Preprocessed data loaded successfully")
            
            result = {
                "success": True,
                "channels": raw_data.ch_names,
                "sampling_rate": raw_data.info['sfreq'],
                "n_samples": raw_data.n_times,
                "consistency_check": consistency_check
            }
            
            # Add fix information if any fixes were applied
            if fix_info.get('applied_fixes') or fix_info.get('changes'):
                result['fixes_applied'] = fix_info
                result['message'] = f"Data loaded with {len(fix_info.get('changes', []))} consistency fixes applied"
            
            return result
            
        except Exception as e:
            self.is_processing = False
            return {"success": False, "error": f"Failed to load preprocessed data: {str(e)}"}

    def load_and_prepare_file(
        self, file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Φόρτωση και προετοιμασία αρχείου για επεξεργασία

        Args:
            file_path: Διαδρομή αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)

        Returns:
            Dictionary με αποτελέσματα φόρτωσης
        """
        self.is_processing = True
        self.current_file = file_path
        self.ica_fitted = False

        try:
            self._update_status("Φόρτωση δεδομένων...")
            self._update_progress(10)

            # Φόρτωση αρχείου με επιλεγμένα κανάλια
            result = self.backend_core.load_file(file_path, selected_channels)

            if not result["success"]:
                self.is_processing = False
                return result

            # Ενημερώνουμε τον ICA processor με τον αριθμό καναλιών
            n_channels = len(result["channels"])
            self.ica_processor = ICAProcessor(n_components=None)  # Αυτόματη ανίχνευση

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

    def fit_ica_analysis(self) -> Dict[str, Any]:
        """
        Εκτέλεση ICA ανάλυσης με ενισχυμένο χειρισμό σφαλμάτων και data consistency checks
        
        Returns:
            Dictionary με αποτελέσματα ICA
        """
        if not self.is_processing:
            return {"success": False, "error": "Δεν έχει φορτωθεί αρχείο"}

        try:
            self._update_status("Εκτέλεση ICA ανάλυσης...")
            self._update_progress(50)

            # Λήψη φιλτραρισμένων δεδομένων
            filtered_data = self.backend_core.get_filtered_data()
            if filtered_data is None:
                return {
                    "success": False,
                    "error": "Δεν υπάρχουν φιλτραρισμένα δεδομένα",
                }

            # Enhanced: Run comprehensive data diagnosis before ICA
            self._update_status("Διάγνωση δεδομένων για ICA...")
            diagnosis = diagnose_ica_data_issues(filtered_data)
            
            if not diagnosis['can_proceed_with_ica']:
                error_msg = "Τα δεδομένα δεν είναι κατάλληλα για ICA:\n"
                for rec in diagnosis['recommendations']:
                    error_msg += f"• {rec}\n"
                
                # Add specific consistency information
                if not diagnosis['consistency']['valid']:
                    error_msg += f"\n🚫 Πρόβλημα συνέπειας: {diagnosis['consistency']['error']}"
                
                return {"success": False, "error": error_msg}

            # Use Enhanced ICA Processor instead of basic one
            from .enhanced_ica_processor import EnhancedICAProcessor, ICAConfig, ICAMethod
            
            # Configure enhanced ICA with better parameters
            ica_config = ICAConfig(
                method=ICAMethod.FASTICA,  # Use FastICA as default
                n_components=None,  # Auto-detect
                max_iter=2000,  # Increase iterations for better convergence
                enable_auto_classification=True,
                random_state=42
            )
            
            enhanced_ica = EnhancedICAProcessor(ica_config)
            
            # Fit ICA with enhanced processor
            self._update_status("Εκπαίδευση Enhanced ICA...")
            results = enhanced_ica.fit_ica(filtered_data)
            
            if not results.get('success', False):
                return {
                    "success": False,
                    "error": results.get('error', 'Άγνωστο σφάλμα Enhanced ICA')
                }

            # Store the enhanced ICA processor for later use
            self.enhanced_ica_processor = enhanced_ica
            self.ica_fitted = True
            
            self._update_progress(70)
            self._update_status("✅ Enhanced ICA εκπαίδευση επιτυχής")

            return {
                "success": True,
                "n_components": results['n_components'],
                "method": results['method'],
                "explained_variance": results.get('explained_variance', 0.0),
                "components_info": {}, # Enhanced ICA handles this differently
                "data_info": {
                    "n_channels": diagnosis['data_quality']['shape'][0],
                    "n_samples": diagnosis['data_quality']['shape'][1],
                    "duration": diagnosis['data_quality']['shape'][1] / filtered_data.info['sfreq'],
                    "sampling_rate": filtered_data.info['sfreq']
                },
                "diagnosis": diagnosis,
                "auto_classifications": results.get('auto_classifications', 0),
                "auto_reject_count": results.get('auto_reject_count', 0)
            }

        except Exception as e:
            error_msg = str(e)
            
            # Enhanced error handling with specific solutions
            if "Number of channels" in error_msg and "do not match" in error_msg:
                enhanced_error = f"Σφάλμα συνέπειας δεδομένων: {error_msg}\n\n💡 Λύση: Τα δεδομένα έχουν πρόβλημα συνέπειας μεταξύ info και data array\n🔧 Δοκιμάστε να επαναφορτώσετε τα δεδομένα ή να χρησιμοποιήσετε διαφορετικές ρυθμίσεις προεπεξεργασίας"
            elif "component" in error_msg.lower() and "1" in error_msg:
                enhanced_error = f"Σφάλμα ICA: {error_msg}\n\n💡 Λύση: Προσθέστε περισσότερα κανάλια EEG"
            elif "nan" in error_msg.lower():
                enhanced_error = f"Σφάλμα ICA: {error_msg}\n\n💡 Λύση: Τα δεδομένα περιέχουν NaN τιμές - εφαρμόστε καλύτερο φιλτράρισμα"
            elif "inf" in error_msg.lower():
                enhanced_error = f"Σφάλμα ICA: {error_msg}\n\n💡 Λύση: Τα δεδομένα περιέχουν άπειρες τιμές - ελέγξτε το φιλτράρισμα"
            elif "converge" in error_msg.lower():
                enhanced_error = f"Σφάλμα ICA: {error_msg}\n\n💡 Λύση: Προβλήματα σύγκλισης - εφαρμόστε καλύτερο προεπεξεργασία"
            else:
                enhanced_error = f"Σφάλμα ICA: {error_msg}"
            
            return {"success": False, "error": enhanced_error}

    def detect_artifacts(self, max_components: int = 3) -> Dict[str, Any]:
        """
        Εντοπισμός artifacts με πολλαπλές μεθόδους

        Args:
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση

        Returns:
            Dictionary με αποτελέσματα εντοπισμού
        """
        if not self.ica_fitted:
            return {"success": False, "error": "ICA δεν έχει εκπαιδευτεί"}

        try:
            self._update_status("Εντοπισμός artifacts...")
            self._update_progress(80)

            # Validate ICA processor state before artifact detection
            if self.ica_processor.n_components is None:
                return {
                    "success": False, 
                    "error": "Σφάλμα ICA: Το ICA δεν έχει εκπαιδευτεί σωστά - n_components είναι None\n💡 Δοκιμάστε να επαναλάβετε την εκπαίδευση ICA"
                }

            # Λήψη φιλτραρισμένων δεδομένων
            filtered_data = self.backend_core.get_filtered_data()

            # Εντοπισμός artifacts
            suggested_artifacts, methods_results = (
                self.artifact_detector.detect_artifacts_multi_method(
                    self.ica_processor, filtered_data, max_components
                )
            )

            self.suggested_artifacts = suggested_artifacts
            self.detection_methods_results = methods_results

            # Δημιουργία επεξηγήσεων - με έλεγχο για n_components
            explanations = {}
            if self.ica_processor.n_components is not None:
                for i in range(self.ica_processor.n_components):
                    explanations[i] = self.artifact_detector.get_artifact_explanation(
                        i, methods_results
                    )

            self._update_progress(90)

            return {
                "success": True,
                "suggested_artifacts": suggested_artifacts,
                "methods_results": methods_results,
                "explanations": explanations,
                "components_info": self.ica_processor.get_all_components_info(),
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
        if not self.ica_fitted:
            return {"success": False, "error": "ICA δεν έχει εκπαιδευτεί"}

        try:
            self._update_status("Εφαρμογή καθαρισμού...")
            self._update_progress(95)

            # Εφαρμογή καθαρισμού
            cleaned_data = self.ica_processor.apply_artifact_removal(
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
        if not self.ica_fitted:
            return None

        return {
            "ica": self.ica_processor.get_ica_object(),
            "raw": self.backend_core.get_filtered_data(),
            "components_info": self.ica_processor.get_all_components_info(),
            "suggested_artifacts": self.suggested_artifacts,
            "explanations": {
                i: self.artifact_detector.get_artifact_explanation(
                    i, self.detection_methods_results
                )
                for i in range(self.ica_processor.n_components)
            },
        }

    def reset_state(self):
        """Επαναφορά κατάστασης service"""
        self.is_processing = False
        self.ica_fitted = False
        self.current_file = None
        self.suggested_artifacts = []
        self.detection_methods_results = {}

        # Reset backend components
        self.backend_core = EEGBackendCore()
        self.ica_processor = ICAProcessor(n_components=None)  # Use automatic detection

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Επιστροφή περίληψης επεξεργασίας

        Returns:
            Dictionary με περίληψη
        """
        return {
            "current_file": self.current_file,
            "is_processing": self.is_processing,
            "ica_fitted": self.ica_fitted,
            "n_components": self.ica_processor.n_components,
            "suggested_artifacts": self.suggested_artifacts,
            "detection_methods": (
                list(self.detection_methods_results.keys())
                if self.detection_methods_results
                else []
            ),
        }
