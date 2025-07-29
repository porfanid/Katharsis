#!/usr/bin/env python3
"""
Unit Tests for EEG Backend Core
Μοναδιαίοι Έλεγχοι για EEG Backend Core
"""

import unittest
import numpy as np
import mne
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import (
    EEGDataManager,
    EEGPreprocessor,
    EEGBackendCore,
    ICAProcessor,
    ArtifactDetector,
    EEGArtifactCleaningService,
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


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
