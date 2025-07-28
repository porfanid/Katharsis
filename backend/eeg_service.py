#!/usr/bin/env python3
"""
Backend Service - Κεντρική υπηρεσία backend για EEG artifact cleaning
Backend Service - Main backend service for EEG artifact cleaning
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from .eeg_backend import EEGBackendCore
from .ica_processor import ICAProcessor
from .artifact_detector import ArtifactDetector
import mne


class EEGArtifactCleaningService:
    """Κεντρική υπηρεσία για EEG artifact cleaning"""
    
    def __init__(self, 
                 n_components: int = None,
                 variance_threshold: float = 2.0,
                 kurtosis_threshold: float = 2.0,
                 range_threshold: float = 3.0):
        """
        Αρχικοποίηση service
        
        Args:
            n_components: Αριθμός ICA συνιστωσών (None για αυτόματη ανίχνευση)
            variance_threshold: Κατώφλι διακύμανσης
            kurtosis_threshold: Κατώφλι κύρτωσης
            range_threshold: Κατώφλι εύρους
        """
        self.backend_core = EEGBackendCore()
        self.ica_processor = ICAProcessor(n_components=n_components)
        self.artifact_detector = ArtifactDetector(
            variance_threshold=variance_threshold,
            kurtosis_threshold=kurtosis_threshold,
            range_threshold=range_threshold
        )
        
        # Callbacks για progress updates
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None
        
        # State
        self.current_file = None
        self.is_processing = False
        self.ica_fitted = False
        self.suggested_artifacts = []
        self.detection_methods_results = {}
        
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
    
    def load_and_prepare_file(self, file_path: str) -> Dict[str, Any]:
        """
        Φόρτωση και προετοιμασία αρχείου για επεξεργασία
        
        Args:
            file_path: Διαδρομή αρχείου
            
        Returns:
            Dictionary με αποτελέσματα φόρτωσης
        """
        self.is_processing = True
        self.current_file = file_path
        self.ica_fitted = False
        
        try:
            self._update_status("Φόρτωση δεδομένων...")
            self._update_progress(10)
            
            # Φόρτωση αρχείου
            result = self.backend_core.load_file(file_path)
            
            if not result['success']:
                self.is_processing = False
                return result
            
            self._update_progress(30)
            self._update_status("Αρχείο φορτώθηκε επιτυχώς")
            
            return result
            
        except Exception as e:
            self.is_processing = False
            return {
                'success': False,
                'error': f"Σφάλμα φόρτωσης: {str(e)}"
            }
    
    def fit_ica_analysis(self) -> Dict[str, Any]:
        """
        Εκτέλεση ICA ανάλυσης
        
        Returns:
            Dictionary με αποτελέσματα ICA
        """
        if not self.is_processing:
            return {'success': False, 'error': 'Δεν έχει φορτωθεί αρχείο'}
        
        try:
            self._update_status("Εκτέλεση ICA ανάλυσης...")
            self._update_progress(50)
            
            # Λήψη φιλτραρισμένων δεδομένων
            filtered_data = self.backend_core.get_filtered_data()
            if filtered_data is None:
                return {'success': False, 'error': 'Δεν υπάρχουν φιλτραρισμένα δεδομένα'}
            
            # Εκπαίδευση ICA
            success = self.ica_processor.fit_ica(filtered_data)
            
            if not success:
                return {'success': False, 'error': 'Αποτυχία εκπαίδευσης ICA'}
            
            self.ica_fitted = True
            self._update_progress(70)
            
            return {
                'success': True,
                'n_components': self.ica_processor.n_components,
                'components_info': self.ica_processor.get_all_components_info()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Σφάλμα ICA: {str(e)}"
            }
    
    def detect_artifacts(self, max_components: int = 3) -> Dict[str, Any]:
        """
        Εντοπισμός artifacts με πολλαπλές μεθόδους
        
        Args:
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση
            
        Returns:
            Dictionary με αποτελέσματα εντοπισμού
        """
        if not self.ica_fitted:
            return {'success': False, 'error': 'ICA δεν έχει εκπαιδευτεί'}
        
        try:
            self._update_status("Εντοπισμός artifacts...")
            self._update_progress(80)
            
            # Λήψη φιλτραρισμένων δεδομένων
            filtered_data = self.backend_core.get_filtered_data()
            
            # Εντοπισμός artifacts
            suggested_artifacts, methods_results = \
                self.artifact_detector.detect_artifacts_multi_method(
                    self.ica_processor, 
                    filtered_data, 
                    max_components
                )
            
            self.suggested_artifacts = suggested_artifacts
            self.detection_methods_results = methods_results
            
            # Δημιουργία επεξηγήσεων
            explanations = {}
            for i in range(self.ica_processor.n_components):
                explanations[i] = self.artifact_detector.get_artifact_explanation(
                    i, methods_results
                )
            
            self._update_progress(90)
            
            return {
                'success': True,
                'suggested_artifacts': suggested_artifacts,
                'methods_results': methods_results,
                'explanations': explanations,
                'components_info': self.ica_processor.get_all_components_info()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Σφάλμα εντοπισμού artifacts: {str(e)}"
            }
    
    def apply_artifact_removal(self, 
                             components_to_remove: List[int]) -> Dict[str, Any]:
        """
        Εφαρμογή αφαίρεσης artifacts
        
        Args:
            components_to_remove: Λίστα συνιστωσών προς αφαίρεση
            
        Returns:
            Dictionary με αποτελέσματα
        """
        if not self.ica_fitted:
            return {'success': False, 'error': 'ICA δεν έχει εκπαιδευτεί'}
        
        try:
            self._update_status("Εφαρμογή καθαρισμού...")
            self._update_progress(95)
            
            # Εφαρμογή καθαρισμού
            cleaned_data = self.ica_processor.apply_artifact_removal(components_to_remove)
            
            if cleaned_data is None:
                return {'success': False, 'error': 'Αποτυχία καθαρισμού δεδομένων'}
            
            # Υπολογισμός στατιστικών πριν/μετά
            original_stats = self.backend_core.preprocessor.get_data_statistics(
                self.backend_core.get_filtered_data()
            )
            cleaned_stats = self.backend_core.preprocessor.get_data_statistics(cleaned_data)
            
            self._update_progress(100)
            self._update_status("Καθαρισμός ολοκληρώθηκε")
            
            return {
                'success': True,
                'cleaned_data': cleaned_data,
                'components_removed': components_to_remove,
                'original_stats': original_stats,
                'cleaned_stats': cleaned_stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Σφάλμα καθαρισμού: {str(e)}"
            }
    
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
            'ica': self.ica_processor.get_ica_object(),
            'raw': self.backend_core.get_filtered_data(),
            'components_info': self.ica_processor.get_all_components_info(),
            'suggested_artifacts': self.suggested_artifacts,
            'explanations': {
                i: self.artifact_detector.get_artifact_explanation(
                    i, self.detection_methods_results
                ) for i in range(self.ica_processor.n_components)
            }
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
            'current_file': self.current_file,
            'is_processing': self.is_processing,
            'ica_fitted': self.ica_fitted,
            'n_components': self.ica_processor.n_components,
            'suggested_artifacts': self.suggested_artifacts,
            'detection_methods': list(self.detection_methods_results.keys()) if self.detection_methods_results else []
        }