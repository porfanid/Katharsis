#!/usr/bin/env python3
"""
ICA Processor - Independent Component Analysis για EEG artifact cleaning
Επεξεργαστής ICA - Ανάλυση Ανεξάρτητων Συνιστωσών για καθαρισμό EEG artifacts
"""

import numpy as np
import mne
from sklearn.decomposition import FastICA
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ICAProcessor:
    """Επεξεργαστής ICA για εντοπισμό και αφαίρεση artifacts"""
    
    def __init__(self, n_components: int = 5, random_state: int = 42):
        """
        Αρχικοποίηση ICA processor
        
        Args:
            n_components: Αριθμός ICA συνιστωσών
            random_state: Seed για αναπαραγωγιμότητα
        """
        self.n_components = n_components
        self.random_state = random_state
        self.ica = None
        self.raw_data = None
        self.components_info = {}
        
    def fit_ica(self, raw: mne.io.Raw) -> bool:
        """
        Εκπαίδευση ICA μοντέλου
        
        Args:
            raw: Φιλτραρισμένα Raw EEG δεδομένα
            
        Returns:
            bool: True εάν η εκπαίδευση ήταν επιτυχής
        """
        try:
            self.raw_data = raw.copy()
            
            # Δημιουργία και εκπαίδευση ICA
            self.ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                method='fastica',
                random_state=self.random_state,
                max_iter=1000,
                verbose=False
            )
            
            self.ica.fit(raw, verbose=False)
            
            # Υπολογισμός πληροφοριών συνιστωσών
            self._calculate_component_info()
            
            return True
            
        except Exception as e:
            print(f"Σφάλμα κατά την εκπαίδευση ICA: {str(e)}")
            return False
    
    def _calculate_component_info(self):
        """Υπολογισμός στατιστικών πληροφοριών για κάθε ICA συνιστώσα"""
        if self.ica is None or self.raw_data is None:
            return
            
        sources = self.ica.get_sources(self.raw_data).get_data()
        
        for i in range(self.n_components):
            comp_data = sources[i]
            
            self.components_info[i] = {
                'variance': float(np.var(comp_data)),
                'kurtosis': float(abs(stats.kurtosis(comp_data))),
                'range': float(np.ptp(comp_data)),
                'std': float(np.std(comp_data)),
                'mean': float(np.mean(comp_data)),
                'rms': float(np.sqrt(np.mean(comp_data**2))),
                'skewness': float(abs(stats.skew(comp_data)))
            }
    
    def get_component_info(self, component_idx: int) -> Dict[str, float]:
        """
        Επιστροφή πληροφοριών για συγκεκριμένη συνιστώσα
        
        Args:
            component_idx: Δείκτης συνιστώσας
            
        Returns:
            Dictionary με στατιστικές πληροφορίες
        """
        return self.components_info.get(component_idx, {})
    
    def get_all_components_info(self) -> Dict[int, Dict[str, float]]:
        """Επιστροφή πληροφοριών όλων των συνιστωσών"""
        return self.components_info
    
    def get_component_data(self, component_idx: int) -> Optional[np.ndarray]:
        """
        Επιστροφή δεδομένων συγκεκριμένης συνιστώσας
        
        Args:
            component_idx: Δείκτης συνιστώσας
            
        Returns:
            Δεδομένα συνιστώσας ή None
        """
        if self.ica is None or self.raw_data is None:
            return None
            
        try:
            sources = self.ica.get_sources(self.raw_data).get_data()
            return sources[component_idx]
        except IndexError:
            return None
    
    def apply_artifact_removal(self, 
                             components_to_remove: List[int]) -> Optional[mne.io.Raw]:
        """
        Εφαρμογή αφαίρεσης artifacts
        
        Args:
            components_to_remove: Λίστα με δείκτες συνιστωσών προς αφαίρεση
            
        Returns:
            Καθαρισμένα Raw δεδομένα ή None
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
        """Επιστροφή του ICA αντικειμένου"""
        return self.ica
    
    def get_sources_data(self) -> Optional[np.ndarray]:
        """Επιστροφή όλων των ICA sources"""
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