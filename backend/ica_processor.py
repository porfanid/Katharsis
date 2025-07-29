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
        self.ica = None
        self.raw_data = None
        self.components_info = {}

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
            self.raw_data = raw.copy()

            # Αυτόματος προσδιορισμός αριθμού συνιστωσών αν δεν δοθεί
            if self.n_components is None:
                self.n_components = min(len(raw.ch_names), len(raw.ch_names))
            else:
                # Βεβαιώνουμε ότι δεν υπερβαίνουμε τον αριθμό των καναλιών
                self.n_components = min(self.n_components, len(raw.ch_names))

            # Δημιουργία και εκπαίδευση ICA
            self.ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                method="fastica",
                random_state=self.random_state,
                max_iter=1000,
                verbose=False,
            )

            self.ica.fit(raw, verbose=False)

            # Υπολογισμός πληροφοριών συνιστωσών
            self._calculate_component_info()

            return True

        except Exception as e:
            print(f"Σφάλμα κατά την εκπαίδευση ICA: {str(e)}")
            return False

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
        return self.components_info.get(component_idx, {})

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
