#!/usr/bin/env python3
"""
Artifact Detector - Εντοπισμός artifacts σε ICA συνιστώσες
Ανιχνευτής Ατελειών - Αυτόματος εντοπισμός artifacts σε EEG δεδομένα
"""

import numpy as np
import mne
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from .ica_processor import ICAProcessor


class ArtifactDetector:
    """Εντοπισμός artifacts σε ICA συνιστώσες με πολλαπλές μεθόδους"""
    
    def __init__(self, 
                 variance_threshold: float = 2.0,
                 kurtosis_threshold: float = 2.0,
                 range_threshold: float = 3.0):
        """
        Αρχικοποίηση detector
        
        Args:
            variance_threshold: Κατώφλι διακύμανσης για artifacts
            kurtosis_threshold: Κατώφλι κύρτωσης για artifacts
            range_threshold: Κατώφλι εύρους για artifacts
        """
        self.variance_threshold = variance_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.range_threshold = range_threshold
        
    def detect_eog_artifacts(self, 
                           ica: mne.preprocessing.ICA, 
                           raw: mne.io.Raw) -> List[int]:
        """
        Εντοπισμός EOG artifacts χρησιμοποιώντας MNE
        
        Args:
            ica: Εκπαιδευμένο ICA αντικείμενο
            raw: Raw EEG δεδομένα
            
        Returns:
            Λίστα με δείκτες EOG artifact συνιστωσών
        """
        try:
            # Χρήση των frontal καναλιών ως EOG proxy
            frontal_channels = [ch for ch in ['AF3', 'AF4'] if ch in raw.ch_names]
            
            if not frontal_channels:
                return []
                
            # Εντοπισμός EOG artifacts
            eog_indices, _ = ica.find_bads_eog(
                raw, 
                ch_name=frontal_channels,
                threshold=2.0,
                verbose=False
            )
            
            return eog_indices
            
        except Exception as e:
            print(f"Σφάλμα EOG detection: {str(e)}")
            return []
    
    def detect_statistical_artifacts(self, 
                                   ica_processor: ICAProcessor) -> List[int]:
        """
        Εντοπισμός artifacts με στατιστικά κριτήρια
        
        Args:
            ica_processor: ICA processor με υπολογισμένες συνιστώσες
            
        Returns:
            Λίστα με δείκτες artifact συνιστωσών
        """
        artifacts = []
        components_info = ica_processor.get_all_components_info()
        
        if not components_info:
            return []
        
        # Υπολογισμός κατωφλίων βασισμένων στη διανομή
        variances = [info['variance'] for info in components_info.values()]
        kurtoses = [info['kurtosis'] for info in components_info.values()]
        ranges = [info['range'] for info in components_info.values()]
        
        var_mean, var_std = np.mean(variances), np.std(variances)
        kurt_mean, kurt_std = np.mean(kurtoses), np.std(kurtoses)
        range_mean, range_std = np.mean(ranges), np.std(ranges)
        
        # Εντοπισμός outliers
        for comp_idx, info in components_info.items():
            is_artifact = False
            
            # Κριτήριο διακύμανσης
            if info['variance'] > var_mean + self.variance_threshold * var_std:
                is_artifact = True
                
            # Κριτήριο κύρτωσης
            if info['kurtosis'] > kurt_mean + self.kurtosis_threshold * kurt_std:
                is_artifact = True
                
            # Κριτήριο εύρους
            if info['range'] > range_mean + self.range_threshold * range_std:
                is_artifact = True
            
            if is_artifact:
                artifacts.append(comp_idx)
        
        return artifacts
    
    def detect_muscle_artifacts(self, 
                              ica_processor: ICAProcessor,
                              frequency_threshold: float = 20.0) -> List[int]:
        """
        Εντοπισμός μυϊκών artifacts (υψηλές συχνότητες)
        
        Args:
            ica_processor: ICA processor
            frequency_threshold: Κατώφλι συχνότητας (Hz)
            
        Returns:
            Λίστα με δείκτες muscle artifact συνιστωσών
        """
        artifacts = []
        
        if ica_processor.raw_data is None:
            return []
        
        try:
            sources_data = ica_processor.get_sources_data()
            sfreq = ica_processor.raw_data.info['sfreq']
            
            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]
                
                # FFT για ανάλυση συχνοτήτων
                freqs = np.fft.fftfreq(len(comp_data), 1/sfreq)
                fft_data = np.abs(np.fft.fft(comp_data))
                
                # Υπολογισμός ισχύος σε υψηλές συχνότητες
                high_freq_mask = freqs > frequency_threshold
                high_freq_power = np.sum(fft_data[high_freq_mask])
                total_power = np.sum(fft_data)
                
                # Εάν η ισχύς σε υψηλές συχνότητες είναι >50% του συνόλου
                if high_freq_power / total_power > 0.5:
                    artifacts.append(i)
            
            return artifacts
            
        except Exception as e:
            print(f"Σφάλμα muscle artifact detection: {str(e)}")
            return []
    
    def detect_drift_artifacts(self, 
                             ica_processor: ICAProcessor,
                             drift_threshold: float = 0.1) -> List[int]:
        """
        Εντοπισμός drift artifacts (χαμηλές συχνότητες)
        
        Args:
            ica_processor: ICA processor
            drift_threshold: Κατώφλι για drift (Hz)
            
        Returns:
            Λίστα με δείκτες drift artifact συνιστωσών
        """
        artifacts = []
        
        if ica_processor.raw_data is None:
            return []
        
        try:
            sources_data = ica_processor.get_sources_data()
            sfreq = ica_processor.raw_data.info['sfreq']
            
            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]
                
                # Υπολογισμός τάσης (trend)
                x = np.arange(len(comp_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, comp_data)
                
                # Εάν υπάρχει σημαντική τάση
                if abs(r_value) > 0.7 and p_value < 0.05:
                    artifacts.append(i)
            
            return artifacts
            
        except Exception as e:
            print(f"Σφάλμα drift artifact detection: {str(e)}")
            return []
    
    def detect_artifacts_multi_method(self, 
                                    ica_processor: ICAProcessor,
                                    raw: mne.io.Raw,
                                    max_components: int = 3) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Πολλαπλός εντοπισμός artifacts με συνδυασμό μεθόδων
        
        Args:
            ica_processor: ICA processor
            raw: Raw EEG δεδομένα
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση
            
        Returns:
            Tuple με:
            - Τελική λίστα artifacts
            - Dictionary με αποτελέσματα κάθε μεθόδου
        """
        ica = ica_processor.get_ica_object()
        if ica is None:
            return [], {}
        
        # Εφαρμογή όλων των μεθόδων
        methods_results = {
            'eog': self.detect_eog_artifacts(ica, raw),
            'statistical': self.detect_statistical_artifacts(ica_processor),
            'muscle': self.detect_muscle_artifacts(ica_processor),
            'drift': self.detect_drift_artifacts(ica_processor)
        }
        
        # Συνδυασμός αποτελεσμάτων με βάρη
        artifact_scores = {}
        
        for comp_idx in range(ica_processor.n_components):
            score = 0
            
            # EOG detection (βάρος 3)
            if comp_idx in methods_results['eog']:
                score += 3
            
            # Statistical detection (βάρος 2)
            if comp_idx in methods_results['statistical']:
                score += 2
            
            # Muscle detection (βάρος 2)
            if comp_idx in methods_results['muscle']:
                score += 2
            
            # Drift detection (βάρος 1)
            if comp_idx in methods_results['drift']:
                score += 1
            
            artifact_scores[comp_idx] = score
        
        # Επιλογή των top artifact συνιστωσών
        sorted_components = sorted(artifact_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
        
        # Κρατάμε μόνο συνιστώσες με score > 0
        final_artifacts = [comp_idx for comp_idx, score in sorted_components 
                          if score > 0][:max_components]
        
        return final_artifacts, methods_results
    
    def get_artifact_explanation(self, 
                               component_idx: int,
                               methods_results: Dict[str, List[int]]) -> str:
        """
        Επεξήγηση γιατί μια συνιστώσα θεωρείται artifact
        
        Args:
            component_idx: Δείκτης συνιστώσας
            methods_results: Αποτελέσματα των μεθόδων εντοπισμού
            
        Returns:
            Κείμενο επεξήγησης
        """
        reasons = []
        
        if component_idx in methods_results.get('eog', []):
            reasons.append("EOG (κίνηση ματιών)")
        
        if component_idx in methods_results.get('statistical', []):
            reasons.append("Στατιστικά outlier")
        
        if component_idx in methods_results.get('muscle', []):
            reasons.append("Μυϊκή δραστηριότητα")
        
        if component_idx in methods_results.get('drift', []):
            reasons.append("Drift σήματος")
        
        if not reasons:
            return "Καθαρό εγκεφαλικό σήμα"
        
        return f"Πιθανό artifact: {', '.join(reasons)}"