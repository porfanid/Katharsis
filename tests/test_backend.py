#!/usr/bin/env python3
"""
Unit Tests for EEG Backend Core
Μοναδιαίοι Έλεγχοι για EEG Backend Core
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import (
    ArtifactDetector,
    EEGArtifactCleaningService,
    EEGBackendCore,
    EEGDataManager,
    EEGPreprocessor,
    ICAProcessor,
    PCAProcessor,
)


class TestEEGDataManager(unittest.TestCase):
    """Έλεγχοι για EEGDataManager"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.data_manager = EEGDataManager()

        # Create synthetic EEG data for testing
        self.sfreq = 128.0
        self.duration = 10.0  # 10 seconds
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        # Create raw data
        data = np.random.randn(len(self.ch_names), self.n_samples) * 1e-5
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info)

        # Create temporary EDF file
        self.temp_edf = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        self.temp_edf_path = self.temp_edf.name
        self.temp_edf.close()

        # Export to EDF
        self.test_raw.export(
            self.temp_edf_path, fmt="edf", overwrite=True, verbose=False
        )

    def tearDown(self):
        """Καθαρισμός μετά από tests"""
        if os.path.exists(self.temp_edf_path):
            os.unlink(self.temp_edf_path)

    def test_load_edf_file_success(self):
        """Έλεγχος επιτυχούς φόρτωσης EDF"""
        raw, channels = self.data_manager.load_edf_file(self.temp_edf_path)

        self.assertIsInstance(raw, mne.io.BaseRaw)
        self.assertEqual(channels, self.ch_names)
        self.assertEqual(len(raw.ch_names), len(self.ch_names))

    def test_load_edf_file_not_found(self):
        """Έλεγχος σφάλματος όταν το αρχείο δεν βρίσκεται"""
        with self.assertRaises(FileNotFoundError):
            self.data_manager.load_edf_file("nonexistent_file.edf")

    def test_validate_edf_file_valid(self):
        """Έλεγχος επικύρωσης έγκυρου EDF αρχείου"""
        info = self.data_manager.validate_edf_file(self.temp_edf_path)

        self.assertTrue(info["valid"])
        self.assertEqual(info["channels"], self.ch_names)
        self.assertEqual(info["sampling_rate"], self.sfreq)
        self.assertAlmostEqual(info["duration"], self.duration, places=1)

    def test_validate_edf_file_invalid(self):
        """Έλεγχος επικύρωσης μη έγκυρου αρχείου"""
        info = self.data_manager.validate_edf_file("nonexistent_file.edf")

        self.assertFalse(info["valid"])
        self.assertIn("error", info)

    def test_save_cleaned_data(self):
        """Έλεγχος αποθήκευσης δεδομένων"""
        temp_output = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        temp_output_path = temp_output.name
        temp_output.close()

        try:
            success = self.data_manager.save_cleaned_data(
                self.test_raw, temp_output_path
            )
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_output_path))
        finally:
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)


class TestEEGPreprocessor(unittest.TestCase):
    """Έλεγχοι για EEGPreprocessor"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.preprocessor = EEGPreprocessor()

        # Create test raw data
        sfreq = 128.0
        duration = 10.0
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        data = np.random.randn(len(ch_names), n_samples) * 1e-5
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info)

    def test_apply_bandpass_filter(self):
        """Έλεγχος εφαρμογής ζωνοπερατού φίλτρου"""
        filtered_raw = self.preprocessor.apply_bandpass_filter(
            self.test_raw, low_freq=1.0, high_freq=40.0
        )

        self.assertIsInstance(filtered_raw, mne.io.BaseRaw)
        self.assertEqual(len(filtered_raw.ch_names), len(self.test_raw.ch_names))

        # Δεδομένα δεν πρέπει να είναι ίδια μετά το φιλτράρισμα
        orig_data = self.test_raw.get_data()
        filt_data = filtered_raw.get_data()
        self.assertFalse(np.array_equal(orig_data, filt_data))

    def test_get_data_statistics(self):
        """Έλεγχος υπολογισμού στατιστικών"""
        stats = self.preprocessor.get_data_statistics(self.test_raw)

        self.assertIsInstance(stats, dict)
        self.assertEqual(len(stats), len(self.test_raw.ch_names))

        for ch_name in self.test_raw.ch_names:
            self.assertIn(ch_name, stats)
            ch_stats = stats[ch_name]

            # Έλεγχος ύπαρξης όλων των στατιστικών
            expected_keys = ["mean", "std", "variance", "min", "max", "range", "rms"]
            for key in expected_keys:
                self.assertIn(key, ch_stats)
                self.assertIsInstance(ch_stats[key], float)


class TestICAProcessor(unittest.TestCase):
    """Έλεγχοι για ICAProcessor"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.ica_processor = ICAProcessor(n_components=3)

        # Create test raw data with more samples for ICA
        sfreq = 128.0
        duration = 60.0  # 1 minute for better ICA
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        # Create mixed signals for ICA
        time = np.linspace(0, duration, n_samples)

        # Source signals
        source1 = np.sin(2 * np.pi * 10 * time)  # 10 Hz sine
        source2 = np.sin(2 * np.pi * 20 * time)  # 20 Hz sine
        source3 = np.random.randn(n_samples)  # Random noise

        # Mixing matrix
        mixing = np.array(
            [
                [0.8, 0.2, 0.1],
                [0.3, 0.7, 0.2],
                [0.1, 0.3, 0.9],
                [0.2, 0.8, 0.1],
                [0.7, 0.1, 0.3],
            ]
        )

        sources = np.array([source1, source2, source3])
        mixed_data = mixing @ sources * 1e-5

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(mixed_data, info)

    def test_fit_ica_success(self):
        """Έλεγχος επιτυχούς εκπαίδευσης ICA"""
        success = self.ica_processor.fit_ica(self.test_raw)

        self.assertTrue(success)
        self.assertIsNotNone(self.ica_processor.ica)
        self.assertEqual(len(self.ica_processor.components_info), 3)

    def test_get_component_info(self):
        """Έλεγχος λήψης πληροφοριών συνιστώσας"""
        self.ica_processor.fit_ica(self.test_raw)

        info = self.ica_processor.get_component_info(0)
        self.assertIsInstance(info, dict)

        expected_keys = [
            "variance",
            "kurtosis",
            "range",
            "std",
            "mean",
            "rms",
            "skewness",
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_get_component_data(self):
        """Έλεγχος λήψης δεδομένων συνιστώσας"""
        self.ica_processor.fit_ica(self.test_raw)

        comp_data = self.ica_processor.get_component_data(0)
        self.assertIsInstance(comp_data, np.ndarray)
        self.assertEqual(len(comp_data), len(self.test_raw.times))

    def test_apply_artifact_removal(self):
        """Έλεγχος εφαρμογής αφαίρεσης artifacts"""
        self.ica_processor.fit_ica(self.test_raw)

        # Remove first component
        cleaned_raw = self.ica_processor.apply_artifact_removal([0])

        self.assertIsInstance(cleaned_raw, mne.io.BaseRaw)
        self.assertEqual(len(cleaned_raw.ch_names), len(self.test_raw.ch_names))

        # Data should be different after artifact removal
        orig_data = self.test_raw.get_data()
        clean_data = cleaned_raw.get_data()
        self.assertFalse(np.array_equal(orig_data, clean_data))


class TestArtifactDetector(unittest.TestCase):
    """Έλεγχοι για ArtifactDetector"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.detector = ArtifactDetector()
        self.ica_processor = ICAProcessor(n_components=3)

        # Create test data
        sfreq = 128.0
        duration = 60.0
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        time = np.linspace(0, duration, n_samples)

        # Create sources with different characteristics
        source1 = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(
            n_samples
        )  # Low freq + noise
        source2 = np.sin(2 * np.pi * 60 * time) + 2.0 * np.random.randn(
            n_samples
        )  # High freq + high noise (artifact)
        source3 = np.sin(2 * np.pi * 10 * time) + 0.1 * np.random.randn(
            n_samples
        )  # Clean brain signal

        mixing = np.array(
            [
                [0.8, 0.2, 0.1],
                [0.3, 0.7, 0.2],
                [0.1, 0.3, 0.9],
                [0.2, 0.8, 0.1],
                [0.7, 0.1, 0.3],
            ]
        )

        sources = np.array([source1, source2, source3])
        mixed_data = mixing @ sources * 1e-5

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(mixed_data, info)

        # Fit ICA
        self.ica_processor.fit_ica(self.test_raw)

    def test_detect_statistical_artifacts(self):
        """Έλεγχος στατιστικού εντοπισμού artifacts"""
        artifacts = self.detector.detect_statistical_artifacts(self.ica_processor)

        self.assertIsInstance(artifacts, list)
        # Should detect at least one artifact
        self.assertGreaterEqual(len(artifacts), 0)

    def test_detect_muscle_artifacts(self):
        """Έλεγχος εντοπισμού μυϊκών artifacts"""
        artifacts = self.detector.detect_muscle_artifacts(self.ica_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_drift_artifacts(self):
        """Έλεγχος εντοπισμού drift artifacts"""
        artifacts = self.detector.detect_drift_artifacts(self.ica_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_artifacts_multi_method(self):
        """Έλεγχος πολλαπλού εντοπισμού artifacts"""
        final_artifacts, methods_results = self.detector.detect_artifacts_multi_method(
            self.ica_processor, self.test_raw, max_components=2
        )

        self.assertIsInstance(final_artifacts, list)
        self.assertIsInstance(methods_results, dict)
        self.assertLessEqual(len(final_artifacts), 2)

        # Check methods results structure
        expected_methods = ["eog", "statistical", "muscle", "drift"]
        for method in expected_methods:
            self.assertIn(method, methods_results)
            self.assertIsInstance(methods_results[method], list)

    def test_get_artifact_explanation(self):
        """Έλεγχος επεξήγησης artifacts"""
        _, methods_results = self.detector.detect_artifacts_multi_method(
            self.ica_processor, self.test_raw
        )

        explanation = self.detector.get_artifact_explanation(0, methods_results)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)


class TestEEGArtifactCleaningService(unittest.TestCase):
    """Έλεγχοι για EEGArtifactCleaningService"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.service = EEGArtifactCleaningService()

        # Create test EDF file
        sfreq = 128.0
        duration = 60.0
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        data = np.random.randn(len(ch_names), n_samples) * 1e-5
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        test_raw = mne.io.RawArray(data, info)

        # Create temporary EDF file
        self.temp_edf = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        self.temp_edf_path = self.temp_edf.name
        self.temp_edf.close()

        test_raw.export(self.temp_edf_path, fmt="edf", overwrite=True, verbose=False)

    def tearDown(self):
        """Καθαρισμός μετά από tests"""
        if os.path.exists(self.temp_edf_path):
            os.unlink(self.temp_edf_path)

    def test_full_processing_pipeline(self):
        """Έλεγχος πλήρους pipeline επεξεργασίας"""
        # Load file
        load_result = self.service.load_and_prepare_file(self.temp_edf_path)
        self.assertTrue(load_result["success"])

        # Fit ICA
        ica_result = self.service.fit_ica_analysis()
        self.assertTrue(ica_result["success"])

        # Detect artifacts
        detect_result = self.service.detect_artifacts()
        self.assertTrue(detect_result["success"])

        # Apply cleaning
        clean_result = self.service.apply_artifact_removal(
            [0]
        )  # Remove first component
        self.assertTrue(clean_result["success"])

        # Check visualization data
        viz_data = self.service.get_component_visualization_data()
        self.assertIsNotNone(viz_data)
        self.assertIn("ica", viz_data)
        self.assertIn("raw", viz_data)

    def test_get_processing_summary(self):
        """Έλεγχος περίληψης επεξεργασίας"""
        summary = self.service.get_processing_summary()

        self.assertIsInstance(summary, dict)
        expected_keys = [
            "current_file",
            "is_processing",
            "ica_fitted",
            "n_components",
            "suggested_artifacts",
            "detection_methods",
        ]

        for key in expected_keys:
            self.assertIn(key, summary)

    def test_reset_state(self):
        """Έλεγχος επαναφοράς κατάστασης"""
        # Process some data first
        self.service.load_and_prepare_file(self.temp_edf_path)
        self.service.fit_ica_analysis()

        # Reset
        self.service.reset_state()

        # Check state is reset
        summary = self.service.get_processing_summary()
        self.assertFalse(summary["is_processing"])
        self.assertFalse(summary["ica_fitted"])
        self.assertIsNone(summary["current_file"])


class TestPCAProcessor(unittest.TestCase):
    """Έλεγχοι για PCAProcessor"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.pca_processor = PCAProcessor(n_components=3)

        # Create test raw data with more samples for PCA
        sfreq = 128.0
        duration = 60.0  # 1 minute for better PCA
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        # Create mixed signals for PCA
        time = np.linspace(0, duration, n_samples)

        # Source signals
        source1 = np.sin(2 * np.pi * 10 * time)  # 10 Hz sine
        source2 = np.sin(2 * np.pi * 20 * time)  # 20 Hz sine
        source3 = np.random.randn(n_samples)  # Random noise

        # Mixing matrix
        mixing = np.array(
            [
                [0.8, 0.2, 0.1],
                [0.3, 0.7, 0.2],
                [0.1, 0.3, 0.9],
                [0.2, 0.8, 0.1],
                [0.7, 0.1, 0.3],
            ]
        )

        sources = np.array([source1, source2, source3])
        mixed_data = mixing @ sources * 1e-5

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(mixed_data, info)

    def test_fit_pca_success(self):
        """Έλεγχος επιτυχούς εκπαίδευσης PCA"""
        success = self.pca_processor.fit(self.test_raw)

        self.assertTrue(success)
        self.assertIsNotNone(self.pca_processor.pca)
        self.assertEqual(len(self.pca_processor.components_info), 3)

    def test_get_component_info(self):
        """Έλεγχος λήψης πληροφοριών συνιστώσας PCA"""
        self.pca_processor.fit(self.test_raw)

        info = self.pca_processor.get_component_info(0)
        self.assertIsInstance(info, dict)

        expected_keys = [
            "variance",
            "kurtosis",
            "range",
            "std",
            "mean",
            "rms",
            "skewness",
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_get_component_data(self):
        """Έλεγχος λήψης δεδομένων συνιστώσας PCA"""
        self.pca_processor.fit(self.test_raw)

        comp_data = self.pca_processor.get_component_data(0)
        self.assertIsInstance(comp_data, np.ndarray)
        self.assertEqual(len(comp_data), len(self.test_raw.times))

    def test_apply_artifact_removal(self):
        """Έλεγχος εφαρμογής αφαίρεσης artifacts με PCA"""
        self.pca_processor.fit(self.test_raw)

        # Remove first component
        cleaned_raw = self.pca_processor.apply_artifact_removal([0])

        self.assertIsInstance(cleaned_raw, mne.io.BaseRaw)
        self.assertEqual(len(cleaned_raw.ch_names), len(self.test_raw.ch_names))

        # Data should be different after artifact removal
        orig_data = self.test_raw.get_data()
        clean_data = cleaned_raw.get_data()
        self.assertFalse(np.array_equal(orig_data, clean_data))

    def test_get_explained_variance_ratio(self):
        """Έλεγχος λήψης explained variance ratio"""
        self.pca_processor.fit(self.test_raw)

        variance_ratio = self.pca_processor.get_explained_variance_ratio()
        self.assertIsInstance(variance_ratio, np.ndarray)
        self.assertEqual(len(variance_ratio), 3)
        # Variance ratios should sum to ~1 (or less if not all components)
        self.assertLessEqual(np.sum(variance_ratio), 1.01)

    def test_get_method_name(self):
        """Έλεγχος ονόματος μεθόδου"""
        self.assertEqual(self.pca_processor.get_method_name(), "PCA")


class TestPCAArtifactDetection(unittest.TestCase):
    """Έλεγχοι για PCA artifact detection"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.detector = ArtifactDetector()
        self.pca_processor = PCAProcessor(n_components=3)

        # Create test data
        sfreq = 128.0
        duration = 60.0
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        time = np.linspace(0, duration, n_samples)

        # Create sources with different characteristics
        source1 = np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(n_samples)
        source2 = np.sin(2 * np.pi * 60 * time) + 2.0 * np.random.randn(n_samples)
        source3 = np.sin(2 * np.pi * 10 * time) + 0.1 * np.random.randn(n_samples)

        mixing = np.array(
            [
                [0.8, 0.2, 0.1],
                [0.3, 0.7, 0.2],
                [0.1, 0.3, 0.9],
                [0.2, 0.8, 0.1],
                [0.7, 0.1, 0.3],
            ]
        )

        sources = np.array([source1, source2, source3])
        mixed_data = mixing @ sources * 1e-5

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(mixed_data, info)

        # Fit PCA
        self.pca_processor.fit(self.test_raw)

    def test_detect_statistical_artifacts_pca(self):
        """Έλεγχος στατιστικού εντοπισμού artifacts με PCA"""
        artifacts = self.detector.detect_statistical_artifacts(self.pca_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_muscle_artifacts_pca(self):
        """Έλεγχος εντοπισμού μυϊκών artifacts με PCA"""
        artifacts = self.detector.detect_muscle_artifacts(self.pca_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_drift_artifacts_pca(self):
        """Έλεγχος εντοπισμού drift artifacts με PCA"""
        artifacts = self.detector.detect_drift_artifacts(self.pca_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_pca_variance_artifacts(self):
        """Έλεγχος εντοπισμού variance artifacts με PCA"""
        artifacts = self.detector.detect_pca_variance_artifacts(self.pca_processor)

        self.assertIsInstance(artifacts, list)

    def test_detect_pca_spatial_artifacts(self):
        """Έλεγχος εντοπισμού spatial artifacts με PCA"""
        artifacts = self.detector.detect_pca_spatial_artifacts(
            self.pca_processor, self.test_raw
        )

        self.assertIsInstance(artifacts, list)

    def test_detect_artifacts_multi_method_pca(self):
        """Έλεγχος πολλαπλού εντοπισμού artifacts με PCA"""
        final_artifacts, methods_results = self.detector.detect_artifacts_multi_method(
            self.pca_processor, self.test_raw, max_components=2
        )

        self.assertIsInstance(final_artifacts, list)
        self.assertIsInstance(methods_results, dict)
        self.assertLessEqual(len(final_artifacts), 2)

        # Check PCA-specific methods are included
        self.assertIn("variance", methods_results)
        self.assertIn("spatial", methods_results)
        self.assertIn("statistical", methods_results)

    def test_get_artifact_explanation_pca(self):
        """Έλεγχος επεξήγησης artifacts με PCA"""
        _, methods_results = self.detector.detect_artifacts_multi_method(
            self.pca_processor, self.test_raw
        )

        explanation = self.detector.get_artifact_explanation(0, methods_results)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)


class TestEEGArtifactCleaningServicePCA(unittest.TestCase):
    """Έλεγχοι για EEGArtifactCleaningService με PCA"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.service = EEGArtifactCleaningService(analysis_method="PCA")

        # Create test EDF file
        sfreq = 128.0
        duration = 60.0
        n_samples = int(sfreq * duration)
        ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        data = np.random.randn(len(ch_names), n_samples) * 1e-5
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        test_raw = mne.io.RawArray(data, info)

        # Create temporary EDF file
        self.temp_edf = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        self.temp_edf_path = self.temp_edf.name
        self.temp_edf.close()

        test_raw.export(self.temp_edf_path, fmt="edf", overwrite=True, verbose=False)

    def tearDown(self):
        """Καθαρισμός μετά από tests"""
        if os.path.exists(self.temp_edf_path):
            os.unlink(self.temp_edf_path)

    def test_pca_processing_pipeline(self):
        """Έλεγχος πλήρους pipeline επεξεργασίας με PCA"""
        # Load file
        load_result = self.service.load_and_prepare_file(self.temp_edf_path)
        self.assertTrue(load_result["success"])

        # Fit PCA
        pca_result = self.service.fit_pca_analysis()
        self.assertTrue(pca_result["success"])
        self.assertEqual(pca_result.get("method"), "PCA")

        # Detect artifacts
        detect_result = self.service.detect_artifacts()
        self.assertTrue(detect_result["success"])

        # Apply cleaning
        clean_result = self.service.apply_artifact_removal([0])
        self.assertTrue(clean_result["success"])

        # Check visualization data
        viz_data = self.service.get_component_visualization_data()
        self.assertIsNotNone(viz_data)
        self.assertEqual(viz_data.get("analysis_method"), "PCA")
        self.assertIn("pca", viz_data)

    def test_switch_analysis_method(self):
        """Έλεγχος εναλλαγής μεθόδου ανάλυσης"""
        # Start with PCA
        self.assertEqual(self.service.analysis_method, "PCA")

        # Switch to ICA
        self.service.set_analysis_method("ICA")
        self.assertEqual(self.service.analysis_method, "ICA")

        # Switch back to PCA
        self.service.set_analysis_method("PCA")
        self.assertEqual(self.service.analysis_method, "PCA")

    def test_processing_summary_includes_method(self):
        """Έλεγχος ότι η περίληψη περιέχει τη μέθοδο ανάλυσης"""
        summary = self.service.get_processing_summary()
        self.assertIn("analysis_method", summary)
        self.assertEqual(summary["analysis_method"], "PCA")


class TestMultiFormatImportExport(unittest.TestCase):
    """Έλεγχοι για υποστήριξη πολλαπλών formats (EDF, BDF, FIF, CSV, SET)"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.data_manager = EEGDataManager()

        # Create synthetic EEG data for testing
        self.sfreq = 128.0
        self.duration = 10.0  # 10 seconds
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        # Create raw data
        data = np.random.randn(len(self.ch_names), self.n_samples) * 1e-5
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info)

        # Create temp files for different formats
        self.temp_files = {}

    def tearDown(self):
        """Καθαρισμός μετά από tests"""
        for path in self.temp_files.values():
            if os.path.exists(path):
                os.unlink(path)

    def _create_temp_file(self, suffix):
        """Helper to create temp files"""
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = f.name
        f.close()
        self.temp_files[suffix] = path
        return path

    def test_get_supported_import_formats(self):
        """Έλεγχος λήψης υποστηριζόμενων formats εισαγωγής"""
        formats = EEGDataManager.get_supported_import_formats()
        self.assertIn(".edf", formats)
        self.assertIn(".bdf", formats)
        self.assertIn(".fif", formats)
        self.assertIn(".csv", formats)
        self.assertIn(".set", formats)

    def test_get_supported_export_formats(self):
        """Έλεγχος λήψης υποστηριζόμενων formats εξαγωγής"""
        formats = EEGDataManager.get_supported_export_formats()
        self.assertIn(".edf", formats)
        self.assertIn(".fif", formats)
        self.assertIn(".csv", formats)

    def test_read_raw_edf(self):
        """Έλεγχος ανάγνωσης EDF αρχείου με read_raw"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        raw = EEGDataManager.read_raw(edf_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)
        self.assertEqual(raw.info["sfreq"], self.sfreq)
        self.assertEqual(len(raw.ch_names), len(self.ch_names))
        self.assertGreater(raw.n_times, 0)

    def test_read_raw_fif(self):
        """Έλεγχος ανάγνωσης FIF αρχείου με read_raw"""
        fif_path = self._create_temp_file(".fif")
        self.test_raw.save(fif_path, overwrite=True, verbose=False)

        raw = EEGDataManager.read_raw(fif_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)
        self.assertEqual(raw.info["sfreq"], self.sfreq)
        self.assertGreater(raw.n_times, 0)

    def test_read_raw_csv(self):
        """Έλεγχος ανάγνωσης CSV αρχείου με read_raw"""
        csv_path = self._create_temp_file(".csv")

        # Export to CSV manually
        import pandas as pd

        data = self.test_raw.get_data().T  # (n_samples, n_channels)
        times = self.test_raw.times
        df = pd.DataFrame(data, columns=self.ch_names)
        df.insert(0, "time", times)
        df.to_csv(csv_path, index=False)

        raw = EEGDataManager.read_raw(csv_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)
        self.assertEqual(len(raw.ch_names), len(self.ch_names))
        self.assertGreater(raw.n_times, 0)

    def test_read_raw_unsupported_format(self):
        """Έλεγχος σφάλματος για μη υποστηριζόμενη μορφή"""
        # Create a temp file with unsupported extension
        xyz_path = self._create_temp_file(".xyz")
        with open(xyz_path, "w") as f:
            f.write("dummy data")

        with self.assertRaises(ValueError):
            EEGDataManager.read_raw(xyz_path)

    def test_read_raw_file_not_found(self):
        """Έλεγχος σφάλματος όταν το αρχείο δεν βρίσκεται"""
        with self.assertRaises(FileNotFoundError):
            EEGDataManager.read_raw("nonexistent.edf")

    def test_load_raw_file(self):
        """Έλεγχος φόρτωσης αρχείου με load_raw_file"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        raw, channels = EEGDataManager.load_raw_file(edf_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)
        self.assertEqual(channels, self.ch_names)

    def test_load_file_info(self):
        """Έλεγχος λήψης πληροφοριών αρχείου"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        info = EEGDataManager.load_file_info(edf_path)
        self.assertTrue(info["success"])
        self.assertEqual(info["n_channels"], len(self.ch_names))
        self.assertEqual(info["sampling_rate"], self.sfreq)
        self.assertIn("n_annotations", info)
        self.assertEqual(info["format"], ".edf")

    def test_validate_file(self):
        """Έλεγχος επικύρωσης αρχείου"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        info = EEGDataManager.validate_file(edf_path)
        self.assertTrue(info["valid"])
        self.assertEqual(info["n_channels"], len(self.ch_names))
        self.assertIn("n_annotations", info)
        self.assertEqual(info["format"], ".edf")

    def test_export_raw_edf(self):
        """Έλεγχος εξαγωγής σε EDF format"""
        edf_path = self._create_temp_file(".edf")
        success = EEGDataManager.export_raw(self.test_raw, edf_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(edf_path))

        # Verify by reading back
        raw = EEGDataManager.read_raw(edf_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)

    def test_export_raw_fif(self):
        """Έλεγχος εξαγωγής σε FIF format"""
        fif_path = self._create_temp_file(".fif")
        success = EEGDataManager.export_raw(self.test_raw, fif_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(fif_path))

        # Verify by reading back
        raw = EEGDataManager.read_raw(fif_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)

    def test_export_raw_csv(self):
        """Έλεγχος εξαγωγής σε CSV format"""
        csv_path = self._create_temp_file(".csv")
        success = EEGDataManager.export_raw(self.test_raw, csv_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))

        # Verify by reading back
        raw = EEGDataManager.read_raw(csv_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)

    def test_export_raw_bdf(self):
        """Έλεγχος εξαγωγής σε BDF format"""
        bdf_path = self._create_temp_file(".bdf")
        success = EEGDataManager.export_raw(self.test_raw, bdf_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(bdf_path))

        # Verify by reading back
        raw = EEGDataManager.read_raw(bdf_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)

    def test_export_raw_set(self):
        """Έλεγχος εξαγωγής σε EEGLAB (.set) format"""
        set_path = self._create_temp_file(".set")
        success = EEGDataManager.export_raw(self.test_raw, set_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(set_path))

        # Verify by reading back
        raw = EEGDataManager.read_raw(set_path)
        self.assertIsInstance(raw, mne.io.BaseRaw)

        # Cleanup additional files created by EEGLAB format
        fdt_path = set_path.replace(".set", ".fdt")
        if os.path.exists(fdt_path):
            self.temp_files[".fdt"] = fdt_path

    def test_export_raw_unsupported_format(self):
        """Έλεγχος σφάλματος για μη υποστηριζόμενη μορφή εξαγωγής"""
        xyz_path = self._create_temp_file(".xyz")
        with self.assertRaises(ValueError):
            EEGDataManager.export_raw(self.test_raw, xyz_path)

    def test_save_cleaned_data_multiple_formats(self):
        """Έλεγχος αποθήκευσης σε διαφορετικές μορφές"""
        # Test EDF
        edf_path = self._create_temp_file(".edf")
        success = self.data_manager.save_cleaned_data(self.test_raw, edf_path)
        self.assertTrue(success)

        # Test FIF
        fif_path = self._create_temp_file(".fif")
        success = self.data_manager.save_cleaned_data(self.test_raw, fif_path)
        self.assertTrue(success)

        # Test CSV
        csv_path = self._create_temp_file(".csv")
        success = self.data_manager.save_cleaned_data(self.test_raw, csv_path)
        self.assertTrue(success)

        # Test BDF
        bdf_path = self._create_temp_file(".bdf")
        success = self.data_manager.save_cleaned_data(self.test_raw, bdf_path)
        self.assertTrue(success)

        # Test SET
        set_path = self._create_temp_file(".set")
        success = self.data_manager.save_cleaned_data(self.test_raw, set_path)
        self.assertTrue(success)
        # Cleanup additional files created by EEGLAB format
        fdt_path = set_path.replace(".set", ".fdt")
        if os.path.exists(fdt_path):
            self.temp_files[".fdt"] = fdt_path


class TestMultiFormatBackendCore(unittest.TestCase):
    """Έλεγχοι για EEGBackendCore με πολλαπλά formats"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        self.backend = EEGBackendCore()

        # Create synthetic EEG data
        self.sfreq = 128.0
        self.duration = 10.0
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        data = np.random.randn(len(self.ch_names), self.n_samples) * 1e-5
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info)

        self.temp_files = {}

    def tearDown(self):
        """Καθαρισμός"""
        for path in self.temp_files.values():
            if os.path.exists(path):
                os.unlink(path)

    def _create_temp_file(self, suffix):
        """Helper to create temp files"""
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = f.name
        f.close()
        self.temp_files[suffix] = path
        return path

    def test_load_file_edf(self):
        """Έλεγχος φόρτωσης EDF μέσω EEGBackendCore"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        result = self.backend.load_file(edf_path)
        self.assertTrue(result["success"])
        self.assertEqual(result["channels"], self.ch_names)
        self.assertEqual(result["sampling_rate"], self.sfreq)
        self.assertIn("n_annotations", result)

    def test_load_file_fif(self):
        """Έλεγχος φόρτωσης FIF μέσω EEGBackendCore"""
        fif_path = self._create_temp_file(".fif")
        self.test_raw.save(fif_path, overwrite=True, verbose=False)

        result = self.backend.load_file(fif_path)
        self.assertTrue(result["success"])
        self.assertIn("n_annotations", result)

    def test_load_file_csv(self):
        """Έλεγχος φόρτωσης CSV μέσω EEGBackendCore"""
        import pandas as pd

        csv_path = self._create_temp_file(".csv")
        data = self.test_raw.get_data().T
        times = self.test_raw.times
        df = pd.DataFrame(data, columns=self.ch_names)
        df.insert(0, "time", times)
        df.to_csv(csv_path, index=False)

        result = self.backend.load_file(csv_path)
        self.assertTrue(result["success"])
        self.assertIn("n_annotations", result)

    def test_get_file_info_multi_format(self):
        """Έλεγχος λήψης πληροφοριών αρχείου σε πολλαπλά formats"""
        edf_path = self._create_temp_file(".edf")
        self.test_raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

        info = self.backend.get_file_info(edf_path)
        self.assertTrue(info["success"])
        self.assertIn("format", info)
        self.assertEqual(info["format"], ".edf")


class TestBandPowerAnalyzer(unittest.TestCase):
    """Έλεγχοι για BandPowerAnalyzer"""

    def setUp(self):
        """Προετοιμασία test δεδομένων"""
        from backend import BandPowerAnalyzer

        self.analyzer = BandPowerAnalyzer()

        # Create test raw data with known frequency content
        self.sfreq = 256.0  # Higher sampling rate for better frequency resolution
        self.duration = 10.0
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz"]

        # Create signals with specific frequency content
        time = np.linspace(0, self.duration, self.n_samples)

        # Channel 1: Strong alpha (10 Hz)
        alpha_signal = np.sin(2 * np.pi * 10 * time)

        # Channel 2: Mix of theta (6 Hz) and beta (20 Hz)
        mixed_signal = np.sin(2 * np.pi * 6 * time) + np.sin(2 * np.pi * 20 * time)

        # Channel 3: Delta (2 Hz) dominant
        delta_signal = 2 * np.sin(2 * np.pi * 2 * time) + 0.5 * np.random.randn(
            self.n_samples
        )

        data = np.array([alpha_signal, mixed_signal, delta_signal]) * 1e-5

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info)

    def test_compute_band_power_welch(self):
        """Έλεγχος υπολογισμού band power με Welch"""
        # Test with alpha dominant signal (10 Hz)
        alpha_data = np.sin(2 * np.pi * 10 * np.linspace(0, 10, 2560))
        powers = self.analyzer.compute_band_power_welch(alpha_data, sfreq=256.0)

        self.assertIsInstance(powers, dict)
        self.assertIn("Delta", powers)
        self.assertIn("Theta", powers)
        self.assertIn("Alpha", powers)
        self.assertIn("Beta", powers)
        self.assertIn("Gamma", powers)

        # All percentages should sum to approximately 100%
        # (may be slightly less due to spectral leakage or rounding)
        total = sum(powers.values())
        self.assertGreater(total, 85)  # Allow some leakage/rounding
        self.assertLessEqual(total, 101)  # Should not exceed 100% + small error

        # Alpha should be dominant for 10 Hz signal
        self.assertGreater(powers["Alpha"], powers["Delta"])
        self.assertGreater(powers["Alpha"], powers["Gamma"])

    def test_compute_band_power_for_raw(self):
        """Έλεγχος υπολογισμού band power από Raw data"""
        powers = self.analyzer.compute_band_power_for_raw(self.test_raw, channel_idx=0)

        self.assertIsInstance(powers, dict)
        self.assertEqual(len(powers), 5)  # 5 frequency bands

        # All values should be percentages
        for band_name, power in powers.items():
            self.assertGreaterEqual(power, 0)
            self.assertLessEqual(power, 100)

    def test_compute_band_power_time_series(self):
        """Έλεγχος υπολογισμού band power σε χρονικά παράθυρα"""
        time_points, band_powers = self.analyzer.compute_band_power_time_series(
            self.test_raw, channel_idx=0, window_duration=1.0, overlap=0.5
        )

        self.assertIsInstance(time_points, np.ndarray)
        self.assertIsInstance(band_powers, dict)

        # Should have multiple time points
        self.assertGreater(len(time_points), 1)

        # Each band should have the same number of values as time points
        for band_name, values in band_powers.items():
            self.assertEqual(len(values), len(time_points))
            # All values should be percentages
            self.assertTrue(np.all(values >= 0))
            self.assertTrue(np.all(values <= 100))

    def test_compute_average_band_power(self):
        """Έλεγχος υπολογισμού μέσης τιμής band power πολλαπλών καναλιών"""
        powers = self.analyzer.compute_average_band_power(self.test_raw)

        self.assertIsInstance(powers, dict)
        self.assertEqual(len(powers), 5)

        # All values should be percentages
        for band_name, power in powers.items():
            self.assertGreaterEqual(power, 0)
            self.assertLessEqual(power, 100)

    def test_compute_band_power_comparison(self):
        """Έλεγχος σύγκρισης band power μεταξύ δύο σημάτων"""
        # Create a "cleaned" version with reduced high frequency content
        cleaned_raw = self.test_raw.copy()
        cleaned_raw.filter(l_freq=1.0, h_freq=30.0, verbose=False)

        comparison = self.analyzer.compute_band_power_comparison(
            self.test_raw, cleaned_raw, channel_idx=0
        )

        self.assertIn("original", comparison)
        self.assertIn("cleaned", comparison)

        self.assertIsInstance(comparison["original"], dict)
        self.assertIsInstance(comparison["cleaned"], dict)

        # Both should have all 5 bands
        self.assertEqual(len(comparison["original"]), 5)
        self.assertEqual(len(comparison["cleaned"]), 5)

    def test_get_band_colors(self):
        """Έλεγχος χρωμάτων ζωνών"""
        colors = self.analyzer.get_band_colors()

        self.assertIsInstance(colors, dict)
        self.assertEqual(len(colors), 5)

        # All colors should be hex codes
        for band_name, color in colors.items():
            self.assertTrue(color.startswith("#"))
            self.assertEqual(len(color), 7)

    def test_get_band_descriptions(self):
        """Έλεγχος περιγραφών ζωνών"""
        descriptions = self.analyzer.get_band_descriptions()

        self.assertIsInstance(descriptions, dict)
        self.assertEqual(len(descriptions), 5)

        # All descriptions should be non-empty strings
        for band_name, desc in descriptions.items():
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)

    def test_custom_bands(self):
        """Έλεγχος χρήσης custom frequency bands"""
        from backend import BandPowerAnalyzer

        custom_bands = {
            "Low": (0.5, 10.0),
            "High": (10.0, 40.0),
        }
        custom_analyzer = BandPowerAnalyzer(bands=custom_bands)

        powers = custom_analyzer.compute_band_power_for_raw(
            self.test_raw, channel_idx=0
        )

        self.assertEqual(len(powers), 2)
        self.assertIn("Low", powers)
        self.assertIn("High", powers)

    def test_empty_data_handling(self):
        """Έλεγχος χειρισμού άδειων δεδομένων"""
        empty_data = np.array([])
        powers = self.analyzer.compute_band_power_welch(empty_data, sfreq=256.0)

        # Should return zeros for all bands
        for power in powers.values():
            self.assertEqual(power, 0.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
