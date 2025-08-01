#!/usr/bin/env python3
"""
ICA Processor - Independent Component Analysis για EEG artifact cleaning
======================================================================

Υλοποιεί τη μέθοδο Ανάλυσης Ανεξάρτητων Συνιστωσών (ICA) για:
- Εκπαίδευση ICA μοντέλων σε EEG δεδομένα
- Αναγνώριση artifacts (βλεφαρισμοί, κίνηση, μυικά)
- Απομάκρυνση επιλεγμένων συνιστωσών
- Αποκατάσταση καθαρών σημάτων

Author: porfanid
Version: 1.0
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ICAProcessor:
    """
    Επεξεργαστής ICA για εντοπισμό και αφαίρεση artifacts

    Χρησιμοποιεί την FastICA αλγόριθμο για την ανάλυση των EEG σημάτων σε
    ανεξάρτητες συνιστώσες, επιτρέποντας τον εντοπισμό και την αφαίρεση
    artifacts όπως βλεφαρισμοί, κίνηση και μυικά σήματα.

    Attributes:
        n_components (int): Αριθμός ICA συνιστωσών
        random_state (int): Seed για αναπαραγωγιμότητα
        ica (mne.preprocessing.ICA): Το εκπαιδευμένο ICA μοντέλο
        raw_data (mne.io.Raw): Τα δεδομένα εκπαίδευσης
        components_info (dict): Πληροφορίες για τις συνιστώσες
        last_error (str): Το τελευταίο σφάλμα που προέκυψε
    """

    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Αρχικοποίηση ICA processor

        Args:
            n_components (int, optional): Αριθμός ICA συνιστωσών.
                                        Αν None, καθορίζεται αυτόματα.
            random_state (int): Seed για αναπαραγωγιμότητα
        """
        self.n_components = n_components
        self.random_state = random_state
        self.ica: Optional[mne.preprocessing.ICA] = None
        self.raw_data: Optional[mne.io.Raw] = None
        self.components_info: Dict[int, Dict[str, float]] = {}
        self.last_error: Optional[str] = None

    def fit_ica(self, raw: mne.io.Raw) -> bool:
        """
        Εκπαίδευση ICA μοντέλου

        Εκπαιδεύει ένα ICA μοντέλο στα παρεχόμενα EEG δεδομένα χρησιμοποιώντας
        τον FastICA αλγόριθμο. Το μοντέλο αναλύει τα σήματα σε ανεξάρτητες
        συνιστώσες που αντιπροσωπεύουν διαφορετικές πηγές δραστηριότητας.

        Args:
            raw (mne.io.Raw): Φιλτραρισμένα Raw EEG δεδομένα

        Returns:
            bool: True εάν η εκπαίδευση ήταν επιτυχής, False αλλιώς
        """
        try:
            # Comprehensive data validation before ICA training
            validation_result = self._validate_data_for_ica(raw)
            if not validation_result["valid"]:
                self.last_error = f"Σφάλμα επικύρωσης δεδομένων: {validation_result['error']}"
                print(self.last_error)
                return False
            
            self.raw_data = raw.copy()

            # Determine number of components with proper validation
            n_channels = len(raw.ch_names)
            if n_channels < 2:
                self.last_error = "Σφάλμα: Απαιτούνται τουλάχιστον 2 κανάλια για ICA ανάλυση"
                print(self.last_error)
                return False
                
            if self.n_components is None:
                # Use 90% of channels, but at least 2 and at most n_channels-1
                self.n_components = max(2, min(n_channels - 1, int(0.9 * n_channels)))
            else:
                # Ensure n_components is valid
                self.n_components = max(2, min(self.n_components, n_channels - 1))

            # Enhanced ICA initialization with robust parameters
            self.ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                method="fastica",
                random_state=self.random_state,
                max_iter=2000,  # Increased iterations for better convergence
                verbose=False
            )

            if self.ica is None:
                raise RuntimeError("ICA initialization failed")

            # Train ICA with error handling for convergence issues
            print(f"Εκπαίδευση ICA με {self.n_components} συνιστώσες...")
            self.ica.fit(raw, verbose=False)
            
            # Verify ICA was fitted successfully - handle different MNE versions
            try:
                # Try newer MNE version method first
                if hasattr(self.ica, 'get_components') and callable(self.ica.get_components):
                    components = self.ica.get_components()
                    if components is not None and components.size > 0:
                        # Successfully fitted
                        pass
                    else:
                        raise RuntimeError("ICA get_components() returned empty result")
                elif hasattr(self.ica, 'components_') and self.ica.components_ is not None:
                    # Legacy MNE version
                    if self.ica.components_.size > 0:
                        # Successfully fitted
                        pass
                    else:
                        raise RuntimeError("ICA components_ is empty")
                else:
                    raise RuntimeError("Cannot access ICA components - unknown MNE version")
                    
            except Exception as comp_error:
                raise RuntimeError(f"ICA fitting failed - component access error: {str(comp_error)}")

            # Calculate component information
            self._calculate_component_info()

            print(f"✅ ICA εκπαίδευση επιτυχής: {self.n_components} συνιστώσες")
            return True

        except Exception as e:
            error_msg = str(e)
            self.last_error = f"Σφάλμα κατά την εκπαίδευση ICA: {error_msg}"
            print(self.last_error)
            
            # Provide specific error guidance
            if "component" in error_msg.lower() and "1" in error_msg:
                print("💡 Λύση: Προσθέστε περισσότερα κανάλια ή χρησιμοποιήστε δεδομένα με περισσότερα κανάλια")
            elif "nan" in error_msg.lower() or "inf" in error_msg.lower():
                print("💡 Λύση: Ελέγξτε τα δεδομένα για NaN ή άπειρες τιμές και φιλτράρετε ή αντικαταστήστε τες")
            elif "converge" in error_msg.lower():
                print("💡 Λύση: Τα δεδομένα ενδέχεται να χρειάζονται καλύτερο προεπεξεργασία ή φιλτράρισμα")
            
            return False
            
    def _validate_data_for_ica(self, raw: mne.io.Raw) -> dict:
        """
        Comprehensive data validation for ICA training
        
        Args:
            raw: MNE Raw object to validate
            
        Returns:
            dict: Validation result with 'valid' boolean and 'error' message
        """
        try:
            data = raw.get_data()
            
            # Check minimum requirements
            n_channels, n_samples = data.shape
            
            if n_channels < 2:
                return {
                    "valid": False, 
                    "error": f"Ανεπαρκή κανάλια: {n_channels} (απαιτούνται ≥2)"
                }
            
            if n_samples < 1000:  # At least 4 seconds at 250Hz
                return {
                    "valid": False,
                    "error": f"Ανεπαρκή δεδομένα: {n_samples} δείγματα (απαιτούνται ≥1000)"
                }
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data)):
                nan_channels = []
                for i, ch_name in enumerate(raw.ch_names):
                    if np.any(np.isnan(data[i])):
                        nan_channels.append(ch_name)
                return {
                    "valid": False,
                    "error": f"Δεδομένα περιέχουν NaN τιμές στα κανάλια: {nan_channels[:5]}{'...' if len(nan_channels) > 5 else ''}"
                }
                
            if np.any(np.isinf(data)):
                return {
                    "valid": False,
                    "error": "Δεδομένα περιέχουν άπειρες τιμές"
                }
            
            # Check for channels with zero variance (constant channels)
            variances = np.var(data, axis=1)
            zero_var_channels = np.where(variances < 1e-12)[0]
            if len(zero_var_channels) > 0:
                channel_names = [raw.ch_names[i] for i in zero_var_channels]
                print(f"⚠️  Προειδοποίηση: Κανάλια με μηδενική διακύμανση: {channel_names}")
                # Don't fail, just warn - ICA can handle this
            
            # Check data range (should be reasonable for EEG)
            data_range = np.ptp(data)
            if data_range < 1e-6:
                return {
                    "valid": False,
                    "error": "Δεδομένα έχουν πολύ μικρό εύρος τιμών"
                }
            
            if data_range > 1e6:
                print("⚠️  Προειδοποίηση: Δεδομένα έχουν πολύ μεγάλο εύρος - ενδέχεται να χρειάζονται κανονικοποίηση")
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Σφάλμα επικύρωσης: {str(e)}"
            }

    def _calculate_component_info(self):
        """
        Υπολογισμός στατιστικών πληροφοριών για κάθε ICA συνιστώσα

        Υπολογίζει βασικά στατιστικά για κάθε συνιστώσα όπως διακύμανση,
        κύρτωση, εύρος, κλπ. που χρησιμοποιούνται για τον εντοπισμό artifacts.
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
        Επιστροφή πληροφοριών για συγκεκριμένη συνιστώσα

        Args:
            component_idx (int): Δείκτης συνιστώσας (0-based)

        Returns:
            Dict[str, float]: Dictionary με στατιστικές πληροφορίες όπως
                            variance, kurtosis, range, std, mean, rms, skewness
        """
        default_info: Dict[str, float] = {}
        return self.components_info.get(component_idx, default_info)

    def get_all_components_info(self) -> Dict[int, Dict[str, float]]:
        """
        Επιστροφή πληροφοριών όλων των συνιστωσών

        Returns:
            Dict[int, Dict[str, float]]: Dictionary με πληροφορίες όλων των συνιστωσών
        """
        return self.components_info

    def get_component_data(self, component_idx: int) -> Optional[np.ndarray]:
        """
        Επιστροφή δεδομένων συγκεκριμένης συνιστώσας

        Εξάγει τη χρονοσειρά της επιλεγμένης ICA συνιστώσας.

        Args:
            component_idx (int): Δείκτης συνιστώσας

        Returns:
            Optional[np.ndarray]: Δεδομένα συνιστώσας ως 1D array ή None αν αποτύχει
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
        Εφαρμογή αφαίρεσης artifacts

        Αφαιρεί τις επιλεγμένες ICA συνιστώσες από τα αρχικά δεδομένα,
        αποκαθιστώντας το καθαρό σήμα χωρίς τα artifacts.

        Args:
            components_to_remove (List[int]): Λίστα με δείκτες συνιστωσών προς αφαίρεση

        Returns:
            Optional[mne.io.Raw]: Καθαρισμένα Raw δεδομένα ή None αν αποτύχει
        """
        if self.ica is None or self.raw_data is None:
            return None

        try:
            # Δημιουργία αντιγράφου για καθαρισμό
            cleaned_raw = self.raw_data.copy()

            # Ορισμός συνιστωσών προς αφαίρεση
            self.ica.exclude = components_to_remove

            # Εφαρμογή καθαρισμού
            cleaned_raw = self.ica.apply(cleaned_raw, verbose=False)

            return cleaned_raw

        except Exception as e:
            print(f"Σφάλμα κατά τον καθαρισμό: {str(e)}")
            return None

    def get_ica_object(self) -> Optional[mne.preprocessing.ICA]:
        """
        Επιστροφή του ICA αντικειμένου

        Returns:
            Optional[mne.preprocessing.ICA]: Το εκπαιδευμένο ICA μοντέλο ή None
        """
        return self.ica

    def get_sources_data(self) -> Optional[np.ndarray]:
        """
        Επιστροφή όλων των ICA sources

        Εξάγει όλες τις ICA συνιστώσες ως πίνακα δεδομένων.

        Returns:
            Optional[np.ndarray]: Πίνακας με shape (n_components, n_timepoints) ή None
        """
        if self.ica is None or self.raw_data is None:
            return None

        return self.ica.get_sources(self.raw_data).get_data()

    def get_mixing_matrix(self) -> Optional[np.ndarray]:
        """Επιστροφή του mixing matrix"""
        if self.ica is None:
            return None
        return self.ica.mixing_

    def get_unmixing_matrix(self) -> Optional[np.ndarray]:
        """Επιστροφή του unmixing matrix"""
        if self.ica is None:
            return None
        return self.ica.unmixing_
