#!/usr/bin/env python3
"""
Artifact Detector - Εντοπισμός artifacts σε ICA συνιστώσες
========================================================

Υλοποιεί αλγορίθμους για τον αυτόματο εντοπισμό artifacts σε EEG δεδομένα:
- Εντοπισμός EOG artifacts (βλεφαρισμοί)
- Στατιστική ανάλυση συνιστωσών
- Πολλαπλές μέθοδοι εντοπισμού
- Γενερικός εντοπισμός με fallback μεθόδους

Author: porfanid
Version: 1.0
"""

from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import stats

try:
    import mne_icalabel
    ICALABEL_AVAILABLE = True
except ImportError:
    ICALABEL_AVAILABLE = False

from .ica_processor import ICAProcessor


class ArtifactDetector:
    """
    Εντοπισμός artifacts σε ICA συνιστώσες με πολλαπλές μεθόδους

    Χρησιμοποιεί διάφορους αλγορίθμους για τον εντοπισμό artifacts όπως:
    - EOG artifacts (βλεφαρισμοί) μέσω frontal καναλιών
    - Στατιστική ανάλυση (διακύμανση, κύρτωση, εύρος)
    - Συνδυαστικούς αλγορίθμους εντοπισμού

    Attributes:
        variance_threshold (float): Κατώφλι διακύμανσης για artifacts
        kurtosis_threshold (float): Κατώφλι κύρτωσης για artifacts
        range_threshold (float): Κατώφλι εύρους για artifacts
    """

    def __init__(
        self,
        variance_threshold: float = 2.0,
        kurtosis_threshold: float = 2.0,
        range_threshold: float = 3.0,
    ):
        """
        Αρχικοποίηση artifact detector

        Args:
            variance_threshold (float): Κατώφλι διακύμανσης για artifacts
            kurtosis_threshold (float): Κατώφλι κύρτωσης για artifacts
            range_threshold (float): Κατώφλι εύρους για artifacts
        """
        self.variance_threshold = variance_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.range_threshold = range_threshold

    def detect_eog_artifacts(
        self, ica: mne.preprocessing.ICA, raw: mne.io.Raw
    ) -> List[int]:
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
            frontal_channels = [ch for ch in ["AF3", "AF4"] if ch in raw.ch_names]

            if not frontal_channels:
                return []

            # Εντοπισμός EOG artifacts
            eog_indices, _ = ica.find_bads_eog(
                raw, ch_name=frontal_channels, threshold=2.0, verbose=False
            )

            return eog_indices

        except Exception as e:
            print(f"Σφάλμα EOG detection: {str(e)}")
            return []

    def detect_statistical_artifacts(self, ica_processor: ICAProcessor) -> List[int]:
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
        variances = [info["variance"] for info in components_info.values()]
        kurtoses = [info["kurtosis"] for info in components_info.values()]
        ranges = [info["range"] for info in components_info.values()]

        var_mean, var_std = np.mean(variances), np.std(variances)
        kurt_mean, kurt_std = np.mean(kurtoses), np.std(kurtoses)
        range_mean, range_std = np.mean(ranges), np.std(ranges)

        # Εντοπισμός outliers
        for comp_idx, info in components_info.items():
            is_artifact = False

            # Κριτήριο διακύμανσης
            if info["variance"] > var_mean + self.variance_threshold * var_std:
                is_artifact = True

            # Κριτήριο κύρτωσης
            if info["kurtosis"] > kurt_mean + self.kurtosis_threshold * kurt_std:
                is_artifact = True

            # Κριτήριο εύρους
            if info["range"] > range_mean + self.range_threshold * range_std:
                is_artifact = True

            if is_artifact:
                artifacts.append(comp_idx)

        return artifacts

    def detect_muscle_artifacts(
        self, ica_processor: ICAProcessor, frequency_threshold: float = 20.0
    ) -> List[int]:
        """
        Εντοπισμός μυϊκών artifacts (υψηλές συχνότητες)

        Args:
            ica_processor: ICA processor
            frequency_threshold: Κατώφλι συχνότητας (Hz)

        Returns:
            Λίστα με δείκτες muscle artifact συνιστωσών
        """
        artifacts: List[int] = []

        if ica_processor.raw_data is None:
            return []

        try:
            sources_data = ica_processor.get_sources_data()
            sfreq = ica_processor.raw_data.info["sfreq"]

            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]

                # FFT για ανάλυση συχνοτήτων
                freqs = np.fft.fftfreq(len(comp_data), 1 / sfreq)
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

    def detect_drift_artifacts(
        self, ica_processor: ICAProcessor, drift_threshold: float = 0.1
    ) -> List[int]:
        """
        Εντοπισμός drift artifacts (χαμηλές συχνότητες)

        Args:
            ica_processor: ICA processor
            drift_threshold: Κατώφλι για drift (Hz)

        Returns:
            Λίστα με δείκτες drift artifact συνιστωσών
        """
        artifacts: List[int] = []

        if ica_processor.raw_data is None:
            return []

        try:
            sources_data = ica_processor.get_sources_data()
            sfreq = ica_processor.raw_data.info["sfreq"]

            for i in range(sources_data.shape[0]):
                comp_data = sources_data[i]

                # Υπολογισμός τάσης (trend)
                x = np.arange(len(comp_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, comp_data
                )

                # Εάν υπάρχει σημαντική τάση
                if abs(r_value) > 0.7 and p_value < 0.05:
                    artifacts.append(i)

            return artifacts

        except Exception as e:
            print(f"Σφάλμα drift artifact detection: {str(e)}")
            return []

    def detect_artifacts_multi_method(
        self, ica_processor: ICAProcessor, raw: mne.io.Raw, max_components: int = 3
    ) -> Tuple[List[int], Dict[str, List[int]], Dict[int, Dict[str, Any]]]:
        """
        Πολλαπλός εντοπισμός artifacts με συνδυασμό μεθόδων, συμπεριλαμβανομένου του ICLabel

        Args:
            ica_processor: ICA processor
            raw: Raw EEG δεδομένα
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση

        Returns:
            Tuple με:
            - Τελική λίστα artifacts
            - Dictionary με αποτελέσματα κάθε μεθόδου
            - Dictionary με πληροφορίες ICLabel (αν διαθέσιμο)
        """
        ica = ica_processor.get_ica_object()
        if ica is None:
            return [], {}, {}

        # Προσπάθεια χρήσης ICLabel πρώτα
        icalabel_artifacts, icalabel_info = self.detect_with_icalabel(ica, raw)
        
        if icalabel_artifacts and icalabel_info:
            # Αν το ICLabel είναι διαθέσιμο, χρησιμοποιούμε κυρίως αυτό
            print(f"Χρήση ICLabel - εντοπίστηκαν {len(icalabel_artifacts)} artifacts")
            
            methods_results = {
                "icalabel": icalabel_artifacts,
                "eog": self.detect_eog_artifacts(ica, raw),
                "statistical": self.detect_statistical_artifacts(ica_processor),
                "muscle": self.detect_muscle_artifacts(ica_processor),
                "drift": self.detect_drift_artifacts(ica_processor),
            }
            
            # Κρατάμε τα ICLabel artifacts μέχρι το max_components
            final_artifacts = icalabel_artifacts[:max_components]
            
            return final_artifacts, methods_results, icalabel_info
        
        # Αν το ICLabel δεν είναι διαθέσιμο ή απέτυχε, χρησιμοποιούμε τις παραδοσιακές μεθόδους
        print("Χρήση παραδοσιακών μεθόδων εντοπισμού artifacts")
        
        # Εφαρμογή όλων των παραδοσιακών μεθόδων
        methods_results = {
            "eog": self.detect_eog_artifacts(ica, raw),
            "statistical": self.detect_statistical_artifacts(ica_processor),
            "muscle": self.detect_muscle_artifacts(ica_processor),
            "drift": self.detect_drift_artifacts(ica_processor),
        }

        # Συνδυασμός αποτελεσμάτων με βάρη
        artifact_scores = {}

        for comp_idx in range(ica_processor.n_components):
            score = 0

            # EOG detection (βάρος 3)
            if comp_idx in methods_results["eog"]:
                score += 3

            # Statistical detection (βάρος 2)
            if comp_idx in methods_results["statistical"]:
                score += 2

            # Muscle detection (βάρος 2)
            if comp_idx in methods_results["muscle"]:
                score += 2

            # Drift detection (βάρος 1)
            if comp_idx in methods_results["drift"]:
                score += 1

            artifact_scores[comp_idx] = score

        # Επιλογή των top artifact συνιστωσών
        sorted_components = sorted(
            artifact_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Κρατάμε μόνο συνιστώσες με score > 0
        final_artifacts = [
            comp_idx for comp_idx, score in sorted_components if score > 0
        ][:max_components]

        return final_artifacts, methods_results, {}

    def detect_with_icalabel(
        self, ica: mne.preprocessing.ICA, raw: mne.io.Raw
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        """
        Εντοπισμός artifacts χρησιμοποιώντας ICLabel deep learning μοντέλο

        Args:
            ica: Εκπαιδευμένο ICA αντικείμενο
            raw: Raw EEG δεδομένα

        Returns:
            Tuple με:
            - Λίστα με δείκτες artifact συνιστωσών
            - Dictionary με λεπτομερείς πληροφορίες ICLabel για κάθε συνιστώσα
        """
        if not ICALABEL_AVAILABLE:
            print("mne-icalabel δεν είναι διαθέσιμο. Επιστροφή σε στατιστική μέθοδο.")
            return [], {}

        try:
            # Κλήση ICLabel για αυτόματη κατηγοριοποίηση
            component_dict = mne_icalabel.label_components(raw, ica, method="iclabel")

            # Λήψη ετικετών και πιθανοτήτων
            labels = component_dict['labels']
            probabilities = component_dict['y_pred_proba']

            # Δημιουργία λεπτομερών πληροφοριών για κάθε συνιστώσα
            components_info = {}
            artifact_components = []

            # Ορισμός κατωφλίου εμπιστοσύνης για artifacts
            confidence_threshold = 0.7

            # Κατηγορίες που θεωρούνται artifacts
            artifact_categories = {'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise'}

            for i, (label, prob) in enumerate(zip(labels, probabilities)):
                # Δημιουργία emoji για την κατηγορία
                category_emoji = self._get_category_emoji(label)
                
                components_info[i] = {
                    'icalabel_category': label,
                    'icalabel_probability': prob,
                    'icalabel_emoji': category_emoji,
                    'is_artifact': label in artifact_categories and prob >= confidence_threshold,
                    'description': f"{category_emoji} {label} ({prob:.1%})"
                }

                # Προσθήκη στη λίστα artifacts αν χρειάζεται
                if components_info[i]['is_artifact']:
                    artifact_components.append(i)

            print(f"ICLabel εντόπισε {len(artifact_components)} artifact συνιστώσες")
            return artifact_components, components_info

        except Exception as e:
            print(f"Σφάλμα ICLabel detection: {str(e)}")
            return [], {}

    def _get_category_emoji(self, category: str) -> str:
        """
        Επιστρέφει το κατάλληλο emoji για κάθε κατηγορία ICLabel

        Args:
            category: Η κατηγορία ICLabel

        Returns:
            Emoji string
        """
        emoji_map = {
            'brain': '🧠',
            'muscle': '💪', 
            'eye blink': '👁️',
            'heart beat': '❤️',
            'line noise': '⚡',
            'channel noise': '📻',
            'muscle artifact': '💪',
            'eye': '👁️',
            'heart': '❤️',
            'other': '❓',
            # Add title case variants
            'Brain': '🧠',
            'Muscle': '💪', 
            'Eye': '👁️',
            'Heart': '❤️',
            'Line Noise': '⚡',
            'Channel Noise': '📻',
            'Other': '❓'
        }
        return emoji_map.get(category, '❓')

    def get_artifact_explanation(
        self, component_idx: int, methods_results: Dict[str, List[int]], 
        icalabel_info: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> str:
        """
        Επεξήγηση γιατί μια συνιστώσα θεωρείται artifact

        Args:
            component_idx: Δείκτης συνιστώσας
            methods_results: Αποτελέσματα των μεθόδων εντοπισμού
            icalabel_info: Πληροφορίες από ICLabel (προαιρετικό)

        Returns:
            Κείμενο επεξήγησης
        """
        # Προτεραιότητα στο ICLabel αν είναι διαθέσιμο
        if icalabel_info and component_idx in icalabel_info:
            info = icalabel_info[component_idx]
            return info['description']

        # Fallback στις παραδοσιακές μεθόδους
        reasons = []

        if component_idx in methods_results.get("eog", []):
            reasons.append("EOG (κίνηση ματιών)")

        if component_idx in methods_results.get("statistical", []):
            reasons.append("Στατιστικά outlier")

        if component_idx in methods_results.get("muscle", []):
            reasons.append("Μυϊκή δραστηριότητα")

        if component_idx in methods_results.get("drift", []):
            reasons.append("Drift σήματος")

        if not reasons:
            return "🧠 Καθαρό εγκεφαλικό σήμα"

        return f"Πιθανό artifact: {', '.join(reasons)}"
