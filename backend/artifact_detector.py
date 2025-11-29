#!/usr/bin/env python3
"""
Artifact Detector - Εντοπισμός artifacts σε ICA και PCA συνιστώσες
================================================================

Υλοποιεί αλγορίθμους για τον αυτόματο εντοπισμό artifacts σε EEG δεδομένα:
- Εντοπισμός EOG artifacts (βλεφαρισμοί) - για ICA
- Στατιστική ανάλυση συνιστωσών - για ICA και PCA
- Ανάλυση εξηγούμενης διακύμανσης - για PCA
- Πολλαπλές μέθοδοι εντοπισμού
- Γενερικός εντοπισμός με fallback μεθόδους

Author: porfanid
Version: 1.1
"""

from typing import Dict, List, Tuple, Union

import mne
import numpy as np
from scipy import stats

from .base_processor import BaseComponentProcessor
from .ica_processor import ICAProcessor
from .pca_processor import PCAProcessor


class ArtifactDetector:
    """
    Εντοπισμός artifacts σε ICA και PCA συνιστώσες με πολλαπλές μεθόδους

    Χρησιμοποιεί διάφορους αλγορίθμους για τον εντοπισμό artifacts όπως:
    - EOG artifacts (βλεφαρισμοί) μέσω frontal καναλιών (ICA only)
    - Στατιστική ανάλυση (διακύμανση, κύρτωση, εύρος)
    - Ανάλυση εξηγούμενης διακύμανσης (PCA specific)
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
        Εντοπισμός EOG artifacts χρησιμοποιώντας MNE (ICA only)

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

    def detect_statistical_artifacts(
        self, processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor]
    ) -> List[int]:
        """
        Εντοπισμός artifacts με στατιστικά κριτήρια

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA) με υπολογισμένες συνιστώσες

        Returns:
            Λίστα με δείκτες artifact συνιστωσών
        """
        artifacts = []
        components_info = processor.get_all_components_info()

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
        self,
        processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor],
        frequency_threshold: float = 20.0,
    ) -> List[int]:
        """
        Εντοπισμός μυϊκών artifacts (υψηλές συχνότητες)

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA)
            frequency_threshold: Κατώφλι συχνότητας (Hz)

        Returns:
            Λίστα με δείκτες muscle artifact συνιστωσών
        """
        artifacts: List[int] = []

        if processor.raw_data is None:
            return []

        try:
            sources_data = processor.get_sources_data()
            if sources_data is None:
                return []

            sfreq = processor.raw_data.info["sfreq"]

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
        self,
        processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor],
        drift_threshold: float = 0.1,
    ) -> List[int]:
        """
        Εντοπισμός drift artifacts (χαμηλές συχνότητες)

        Works for both ICA and PCA processors.

        Args:
            processor: Component processor (ICA or PCA)
            drift_threshold: Κατώφλι για drift (Hz)

        Returns:
            Λίστα με δείκτες drift artifact συνιστωσών
        """
        artifacts: List[int] = []

        if processor.raw_data is None:
            return []

        try:
            sources_data = processor.get_sources_data()
            if sources_data is None:
                return []

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

    def detect_pca_variance_artifacts(
        self, pca_processor: PCAProcessor, variance_ratio_threshold: float = 0.3
    ) -> List[int]:
        """
        Εντοπισμός artifacts βασισμένος στην εξηγούμενη διακύμανση (PCA specific)

        Στο PCA, artifacts συχνά εμφανίζονται ως συνιστώσες με υπερβολικά υψηλή
        εξηγούμενη διακύμανση (π.χ. eye blinks) ή πολύ χαμηλή (θόρυβος).

        Args:
            pca_processor: PCA processor
            variance_ratio_threshold: Κατώφλι αναλογίας διακύμανσης

        Returns:
            Λίστα με δείκτες artifact συνιστωσών
        """
        artifacts: List[int] = []

        try:
            explained_variance = pca_processor.get_explained_variance_ratio()
            if explained_variance is None:
                return []

            # Αν μια συνιστώσα εξηγεί υπερβολικά μεγάλο ποσοστό της διακύμανσης
            # (> threshold), μπορεί να είναι artifact (π.χ. eye blinks)
            for i, var_ratio in enumerate(explained_variance):
                # Πρώτες συνιστώσες με υπερβολική διακύμανση
                if var_ratio > variance_ratio_threshold:
                    artifacts.append(i)

            return artifacts

        except Exception as e:
            print(f"Σφάλμα PCA variance artifact detection: {str(e)}")
            return []

    def detect_pca_spatial_artifacts(
        self, pca_processor: PCAProcessor, raw: mne.io.Raw
    ) -> List[int]:
        """
        Εντοπισμός artifacts βασισμένος στα χωρικά patterns (PCA specific)

        Ελέγχει αν τα PCA components έχουν υψηλά βάρη σε frontal καναλιά
        (υποδηλώνει EOG artifacts).

        Args:
            pca_processor: PCA processor
            raw: Raw EEG δεδομένα

        Returns:
            Λίστα με δείκτες artifact συνιστωσών
        """
        artifacts: List[int] = []

        try:
            components = pca_processor.get_components()
            if components is None:
                return []

            ch_names = raw.ch_names

            # Εύρεση frontal καναλιών (πιθανές πηγές EOG artifacts)
            frontal_indices = []
            frontal_patterns = ["Fp", "AF", "F3", "F4", "F7", "F8", "Fz"]
            for i, ch in enumerate(ch_names):
                if any(pattern in ch for pattern in frontal_patterns):
                    frontal_indices.append(i)

            if not frontal_indices:
                return []

            # Για κάθε component, έλεγχος αν έχει υψηλά βάρη σε frontal καναλιά
            n_components = components.shape[1]
            for comp_idx in range(n_components):
                comp_weights = np.abs(components[:, comp_idx])

                # Υπολογισμός αναλογίας frontal vs total weights
                frontal_weights = np.sum(comp_weights[frontal_indices])
                total_weights = np.sum(comp_weights)

                # Αν > 50% των βαρών είναι σε frontal καναλιά, πιθανό EOG artifact
                if frontal_weights / total_weights > 0.5:
                    artifacts.append(comp_idx)

            return artifacts

        except Exception as e:
            print(f"Σφάλμα PCA spatial artifact detection: {str(e)}")
            return []

    def detect_artifacts_multi_method(
        self,
        processor: Union[ICAProcessor, PCAProcessor, BaseComponentProcessor],
        raw: mne.io.Raw,
        max_components: int = 3,
    ) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Πολλαπλός εντοπισμός artifacts με συνδυασμό μεθόδων

        Supports both ICA and PCA processors with appropriate methods.

        Args:
            processor: Component processor (ICA or PCA)
            raw: Raw EEG δεδομένα
            max_components: Μέγιστος αριθμός συνιστωσών προς αφαίρεση

        Returns:
            Tuple με:
            - Τελική λίστα artifacts
            - Dictionary με αποτελέσματα κάθε μεθόδου
        """
        methods_results: Dict[str, List[int]] = {}

        # Determine processor type and apply appropriate methods
        is_ica = isinstance(processor, ICAProcessor)
        is_pca = isinstance(processor, PCAProcessor)

        if is_ica:
            ica = processor.get_ica_object()
            if ica is None:
                return [], {}

            # ICA-specific methods
            methods_results["eog"] = self.detect_eog_artifacts(ica, raw)

        if is_pca:
            # PCA-specific methods
            methods_results["variance"] = self.detect_pca_variance_artifacts(processor)
            methods_results["spatial"] = self.detect_pca_spatial_artifacts(
                processor, raw
            )

        # Common methods for both ICA and PCA
        methods_results["statistical"] = self.detect_statistical_artifacts(processor)
        methods_results["muscle"] = self.detect_muscle_artifacts(processor)
        methods_results["drift"] = self.detect_drift_artifacts(processor)

        # Συνδυασμός αποτελεσμάτων με βάρη
        artifact_scores = {}

        for comp_idx in range(processor.n_components):
            score = 0

            if is_ica:
                # EOG detection (βάρος 3) - ICA only
                if comp_idx in methods_results.get("eog", []):
                    score += 3

            if is_pca:
                # PCA variance detection (βάρος 3)
                if comp_idx in methods_results.get("variance", []):
                    score += 3

                # PCA spatial detection (βάρος 2)
                if comp_idx in methods_results.get("spatial", []):
                    score += 2

            # Statistical detection (βάρος 2)
            if comp_idx in methods_results.get("statistical", []):
                score += 2

            # Muscle detection (βάρος 2)
            if comp_idx in methods_results.get("muscle", []):
                score += 2

            # Drift detection (βάρος 1)
            if comp_idx in methods_results.get("drift", []):
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

        return final_artifacts, methods_results

    def get_artifact_explanation(
        self, component_idx: int, methods_results: Dict[str, List[int]]
    ) -> str:
        """
        Επεξήγηση γιατί μια συνιστώσα θεωρείται artifact

        Args:
            component_idx: Δείκτης συνιστώσας
            methods_results: Αποτελέσματα των μεθόδων εντοπισμού

        Returns:
            Κείμενο επεξήγησης
        """
        reasons = []

        # ICA-specific
        if component_idx in methods_results.get("eog", []):
            reasons.append("EOG (κίνηση ματιών)")

        # PCA-specific
        if component_idx in methods_results.get("variance", []):
            reasons.append("Υψηλή διακύμανση (PCA)")

        if component_idx in methods_results.get("spatial", []):
            reasons.append("Χωρικό pattern (frontal)")

        # Common methods
        if component_idx in methods_results.get("statistical", []):
            reasons.append("Στατιστικά outlier")

        if component_idx in methods_results.get("muscle", []):
            reasons.append("Μυϊκή δραστηριότητα")

        if component_idx in methods_results.get("drift", []):
            reasons.append("Drift σήματος")

        if not reasons:
            return "Καθαρό εγκεφαλικό σήμα"

        return f"Πιθανό artifact: {', '.join(reasons)}"
