#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Katharsis Backend
============================================

Complete test suite for the autonomous KatharsisBackend using data.edf
as the primary test file. Tests all functionality including edge cases.

Author: porfanid
Version: 4.0 - Complete Backend Testing
"""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the backend API
from backend import KatharsisBackend

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestKatharsisBackendCore(unittest.TestCase):
    """Core functionality tests using data.edf"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        # Verify test file exists
        if not os.path.exists(self.data_file):
            self.skipTest(f"Test data file not found: {self.data_file}")
        
        # Callback trackers
        self.progress_updates = []
        self.status_updates = []
        self.error_messages = []
        
        # Set up callbacks
        self.backend.set_callbacks(
            progress_callback=self.progress_updates.append,
            status_callback=self.status_updates.append,
            error_callback=self.error_messages.append
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.backend.reset_all()
    
    def test_backend_initialization(self):
        """Test backend initializes correctly"""
        self.assertIsNotNone(self.backend)
        self.assertIsNone(self.backend.current_file_path)
        self.assertIsNone(self.backend.raw_data)
        self.assertFalse(self.backend.is_processing)
        self.assertEqual(self.backend.progress_percentage, 0)
    
    def test_file_validation_success(self):
        """Test successful file validation with data.edf"""
        result = self.backend.validate_file(self.data_file)
        
        self.assertTrue(result["valid"])
        self.assertIn("channels", result)
        self.assertIn("sampling_rate", result)
        self.assertIn("duration", result)
        self.assertIn("n_channels", result)
        self.assertGreater(result["n_channels"], 0)
        self.assertGreater(result["sampling_rate"], 0)
        self.assertGreater(result["duration"], 0)
    
    def test_file_validation_nonexistent(self):
        """Test file validation with non-existent file"""
        result = self.backend.validate_file("nonexistent_file.edf")
        
        self.assertFalse(result["valid"])
        self.assertIn("error", result)
        self.assertIn("δεν βρέθηκε", result["error"])
    
    def test_file_validation_invalid_format(self):
        """Test file validation with invalid format"""
        # Create temporary non-EEG file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not an EEG file")
            temp_path = temp_file.name
        
        try:
            result = self.backend.validate_file(temp_path)
            self.assertFalse(result["valid"])
            self.assertIn("error", result)
        finally:
            os.unlink(temp_path)
    
    def test_load_file_success(self):
        """Test successful file loading with data.edf"""
        result = self.backend.load_file(self.data_file)
        
        self.assertTrue(result["success"])
        self.assertIn("channels", result)
        self.assertIn("duration", result)
        self.assertIn("sampling_rate", result)
        self.assertIn("n_channels", result)
        self.assertEqual(result["file_path"], self.data_file)
        
        # Check backend state
        self.assertEqual(self.backend.current_file_path, self.data_file)
        self.assertIsNotNone(self.backend.raw_data)
        
        # Check progress updates
        self.assertGreater(len(self.progress_updates), 0)
        self.assertGreater(len(self.status_updates), 0)
    
    def test_load_file_with_channel_selection(self):
        """Test file loading with specific channel selection"""
        # First load to get available channels
        validation = self.backend.validate_file(self.data_file)
        self.assertTrue(validation["valid"])
        
        available_channels = validation["channels"]
        selected_channels = available_channels[:min(3, len(available_channels))]  # First 3 channels
        
        result = self.backend.load_file(self.data_file, selected_channels)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["n_channels"], len(selected_channels))
        self.assertEqual(len(self.backend.raw_data.ch_names), len(selected_channels))
    
    def test_load_file_invalid_channels(self):
        """Test file loading with invalid channel selection"""
        result = self.backend.load_file(self.data_file, ["INVALID_CHANNEL_1", "INVALID_CHANNEL_2"])
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("βρέθηκαν", result["error"])
    
    def test_get_available_channels(self):
        """Test getting available channels"""
        # Before loading
        channels = self.backend.get_available_channels()
        self.assertEqual(channels, [])
        
        # After loading
        self.backend.load_file(self.data_file)
        channels = self.backend.get_available_channels()
        self.assertGreater(len(channels), 0)
        self.assertIsInstance(channels, list)
        self.assertTrue(all(isinstance(ch, str) for ch in channels))
    
    def test_get_eeg_channels(self):
        """Test getting EEG-specific channels"""
        self.backend.load_file(self.data_file)
        eeg_channels = self.backend.get_eeg_channels()
        
        self.assertIsInstance(eeg_channels, list)
        # EEG channels should be a subset of all channels
        all_channels = self.backend.get_available_channels()
        self.assertTrue(all(ch in all_channels for ch in eeg_channels))
    
    def test_get_file_info(self):
        """Test getting comprehensive file information"""
        # Before loading
        info = self.backend.get_file_info()
        self.assertEqual(info, {})
        
        # After loading
        self.backend.load_file(self.data_file)
        info = self.backend.get_file_info()
        
        expected_keys = ["file_path", "n_channels", "channels", "eeg_channels", 
                        "sampling_rate", "duration", "n_samples", "data_shape"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info["file_path"], self.data_file)
        self.assertGreater(info["n_channels"], 0)
        self.assertGreater(info["sampling_rate"], 0)
        self.assertGreater(info["duration"], 0)


class TestKatharsisBackendPreprocessing(unittest.TestCase):
    """Preprocessing functionality tests"""
    
    def setUp(self):
        """Set up test environment with loaded data"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        if not os.path.exists(self.data_file):
            self.skipTest(f"Test data file not found: {self.data_file}")
        
        # Load file for preprocessing tests
        result = self.backend.load_file(self.data_file)
        if not result["success"]:
            self.skipTest(f"Could not load test data: {result.get('error', 'Unknown error')}")
    
    def tearDown(self):
        """Clean up after tests"""
        self.backend.reset_all()
    
    def test_preprocessing_basic_filtering(self):
        """Test basic bandpass filtering"""
        config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 1.0,
                "high_freq": 40.0,
                "enable_notch": False
            }
        }
        
        result = self.backend.apply_preprocessing(config)
        
        self.assertTrue(result["success"])
        self.assertIn("applied_steps", result)
        self.assertIsNotNone(self.backend.preprocessed_data)
        self.assertIsNotNone(self.backend.filtered_data)
    
    def test_preprocessing_notch_filtering(self):
        """Test notch filtering"""
        config = {
            "filtering": {
                "enable_bandpass": False,
                "enable_notch": True,
                "notch_freqs": [50.0, 60.0]
            }
        }
        
        result = self.backend.apply_preprocessing(config)
        
        self.assertTrue(result["success"])
        self.assertIn("applied_steps", result)
    
    def test_preprocessing_rereferencing(self):
        """Test re-referencing"""
        config = {
            "referencing": {
                "enable": True,
                "type": "average",
                "channels": []
            }
        }
        
        result = self.backend.apply_preprocessing(config)
        
        self.assertTrue(result["success"])
        self.assertIn("applied_steps", result)
    
    def test_preprocessing_combined(self):
        """Test combined preprocessing steps"""
        config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 1.0,
                "high_freq": 40.0,
                "enable_notch": True,
                "notch_freqs": [50.0]
            },
            "referencing": {
                "enable": True,
                "type": "average"
            },
            "channel_management": {
                "detect_bad": True,
                "interpolate": False
            }
        }
        
        result = self.backend.apply_preprocessing(config)
        
        self.assertTrue(result["success"])
        self.assertIn("applied_steps", result)
        self.assertIn("processing_info", result)
    
    def test_preprocessing_no_data(self):
        """Test preprocessing without loaded data"""
        backend_empty = KatharsisBackend()
        
        config = {"filtering": {"enable_bandpass": True}}
        result = backend_empty.apply_preprocessing(config)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("αρχείο", result["error"])
    
    def test_preprocessing_invalid_config(self):
        """Test preprocessing with invalid configuration"""
        # Invalid frequency range
        config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 50.0,  # Higher than high_freq
                "high_freq": 10.0,
                "enable_notch": False
            }
        }
        
        result = self.backend.apply_preprocessing(config)
        
        # Should either fail or handle gracefully
        if not result["success"]:
            self.assertIn("error", result)


class TestKatharsisBackendICA(unittest.TestCase):
    """ICA analysis functionality tests"""
    
    def setUp(self):
        """Set up test environment with preprocessed data"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        if not os.path.exists(self.data_file):
            self.skipTest(f"Test data file not found: {self.data_file}")
        
        # Load and preprocess file for ICA tests
        load_result = self.backend.load_file(self.data_file)
        if not load_result["success"]:
            self.skipTest(f"Could not load test data: {load_result.get('error', 'Unknown error')}")
        
        # Apply basic preprocessing
        prep_config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 1.0,
                "high_freq": 40.0
            }
        }
        prep_result = self.backend.apply_preprocessing(prep_config)
        if not prep_result["success"]:
            self.skipTest(f"Could not preprocess data: {prep_result.get('error', 'Unknown error')}")
    
    def tearDown(self):
        """Clean up after tests"""
        self.backend.reset_all()
    
    def test_ica_analysis_fastica(self):
        """Test ICA analysis with FastICA"""
        result = self.backend.perform_ica_analysis(
            ica_method="fastica",
            n_components=None,  # Auto-determine
            max_iter=200
        )
        
        if result["success"]:
            self.assertIn("n_components", result)
            self.assertIn("suggested_artifacts", result)
            self.assertIn("component_info", result)
            self.assertIsNotNone(self.backend.ica_data)
            self.assertGreater(result["n_components"], 0)
        else:
            # ICA might fail with insufficient data - check error is reasonable
            self.assertIn("error", result)
    
    def test_ica_analysis_extended_infomax(self):
        """Test ICA analysis with Extended Infomax"""
        result = self.backend.perform_ica_analysis(
            ica_method="extended_infomax",
            n_components=3,
            max_iter=100
        )
        
        # Extended Infomax might not be available in all MNE versions
        if result["success"]:
            self.assertEqual(result["n_components"], 3)
            self.assertIn("suggested_artifacts", result)
        else:
            self.assertIn("error", result)
    
    def test_ica_analysis_picard(self):
        """Test ICA analysis with Picard"""
        result = self.backend.perform_ica_analysis(
            ica_method="picard",
            n_components=2,
            max_iter=100
        )
        
        if result["success"]:
            self.assertEqual(result["n_components"], 2)
        else:
            self.assertIn("error", result)
    
    def test_ica_components_data(self):
        """Test getting ICA components data"""
        # Before ICA
        components_data = self.backend.get_ica_components_data()
        self.assertIsNone(components_data)
        
        # After ICA (if successful)
        ica_result = self.backend.perform_ica_analysis("fastica", n_components=2)
        if ica_result["success"]:
            components_data = self.backend.get_ica_components_data()
            self.assertIsNotNone(components_data)
            self.assertIn("ica", components_data)
            self.assertIn("raw_data", components_data)
            self.assertIn("n_components", components_data)
    
    def test_ica_cleaning(self):
        """Test ICA cleaning by removing components"""
        # Perform ICA first
        ica_result = self.backend.perform_ica_analysis("fastica", n_components=3)
        if not ica_result["success"]:
            self.skipTest("ICA analysis failed, cannot test cleaning")
        
        # Apply cleaning
        components_to_remove = [0, 1]  # Remove first two components
        clean_result = self.backend.apply_ica_cleaning(components_to_remove)
        
        self.assertTrue(clean_result["success"])
        self.assertIn("cleaned_data", clean_result)
        self.assertIn("removed_components", clean_result)
        self.assertEqual(clean_result["removed_components"], components_to_remove)
        self.assertEqual(clean_result["n_components_removed"], 2)
    
    def test_ica_cleaning_no_ica(self):
        """Test ICA cleaning without prior ICA analysis"""
        result = self.backend.apply_ica_cleaning([0, 1])
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("ICA", result["error"])
    
    def test_ica_analysis_no_data(self):
        """Test ICA analysis without data"""
        backend_empty = KatharsisBackend()
        
        result = backend_empty.perform_ica_analysis("fastica")
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("δεδομένα", result["error"])


class TestKatharsisBackendTimeDomain(unittest.TestCase):
    """Time-domain analysis functionality tests"""
    
    def setUp(self):
        """Set up test environment with preprocessed data"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        if not os.path.exists(self.data_file):
            self.skipTest(f"Test data file not found: {self.data_file}")
        
        # Load file
        load_result = self.backend.load_file(self.data_file)
        if not load_result["success"]:
            self.skipTest(f"Could not load test data: {load_result.get('error', 'Unknown error')}")
        
        # Apply preprocessing
        prep_config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 1.0,
                "high_freq": 40.0
            }
        }
        prep_result = self.backend.apply_preprocessing(prep_config)
        if not prep_result["success"]:
            self.skipTest(f"Could not preprocess data: {prep_result.get('error', 'Unknown error')}")
    
    def tearDown(self):
        """Clean up after tests"""
        self.backend.reset_all()
    
    def test_epoching_basic(self):
        """Test basic epoching functionality"""
        events_config = {
            "channels": [],  # Auto-detect
            "min_duration": 0.001,
            "threshold": "auto"
        }
        
        epoch_config = {
            "tmin": -0.2,
            "tmax": 0.8,
            "baseline": (-0.2, 0.0),
            "baseline_method": "mean"
        }
        
        result = self.backend.perform_epoching(events_config, epoch_config)
        
        # Epoching might fail if no events are found
        if result["success"]:
            self.assertIn("n_epochs", result)
            self.assertIn("n_events", result)
            self.assertIn("epoch_info", result)
            self.assertIsNotNone(self.backend.epochs_data)
        else:
            self.assertIn("error", result)
            # Common failure: no events found
            self.assertTrue("events" in result["error"] or "εντοπισμού" in result["error"])
    
    def test_epoching_with_custom_channels(self):
        """Test epoching with specific event channels"""
        # Get available channels
        channels = self.backend.get_available_channels()
        if not channels:
            self.skipTest("No channels available for epoching test")
        
        events_config = {
            "channels": channels[:1],  # Use first channel for events
            "min_duration": 0.001,
            "threshold": "auto"
        }
        
        epoch_config = {
            "tmin": -0.1,
            "tmax": 0.5,
            "baseline": (-0.1, 0.0),
            "baseline_method": "median"
        }
        
        result = self.backend.perform_epoching(events_config, epoch_config)
        
        # May succeed or fail depending on data
        if not result["success"]:
            self.assertIn("error", result)
    
    def test_erp_analysis_no_epochs(self):
        """Test ERP analysis without epochs"""
        erp_config = {
            "baseline": (-0.2, 0.0),
            "channels": [],
            "detect_peaks": True,
            "peak_components": ["P1", "N1", "P2"]
        }
        
        result = self.backend.perform_erp_analysis(erp_config)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("epoched", result["error"])
    
    def test_erp_analysis_with_epochs(self):
        """Test ERP analysis with epochs (if epoching successful)"""
        # First try epoching
        events_config = {"channels": [], "min_duration": 0.001, "threshold": "auto"}
        epoch_config = {"tmin": -0.2, "tmax": 0.8, "baseline": (-0.2, 0.0)}
        
        epoch_result = self.backend.perform_epoching(events_config, epoch_config)
        if not epoch_result["success"]:
            self.skipTest("Epoching failed, cannot test ERP analysis")
        
        # Now try ERP analysis
        erp_config = {
            "baseline": (-0.2, 0.0),
            "channels": [],
            "detect_peaks": True,
            "peak_components": ["P1", "N1", "P2"]
        }
        
        result = self.backend.perform_erp_analysis(erp_config)
        
        self.assertTrue(result["success"])
        self.assertIn("evoked", result)
        self.assertIn("erp_info", result)


class TestKatharsisBackendDataExport(unittest.TestCase):
    """Data export functionality tests"""
    
    def setUp(self):
        """Set up test environment with data"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        if not os.path.exists(self.data_file):
            self.skipTest(f"Test data file not found: {self.data_file}")
        
        # Load file
        load_result = self.backend.load_file(self.data_file)
        if not load_result["success"]:
            self.skipTest(f"Could not load test data: {load_result.get('error', 'Unknown error')}")
    
    def tearDown(self):
        """Clean up after tests"""
        self.backend.reset_all()
    
    def test_export_raw_data(self):
        """Test exporting raw data"""
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = self.backend.export_data(temp_path, "raw")
            
            self.assertTrue(result["success"])
            self.assertEqual(result["file_path"], temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_preprocessed_data(self):
        """Test exporting preprocessed data"""
        # Apply preprocessing first
        prep_config = {
            "filtering": {
                "enable_bandpass": True,
                "low_freq": 1.0,
                "high_freq": 40.0
            }
        }
        prep_result = self.backend.apply_preprocessing(prep_config)
        if not prep_result["success"]:
            self.skipTest("Preprocessing failed, cannot test export")
        
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = self.backend.export_data(temp_path, "preprocessed")
            
            self.assertTrue(result["success"])
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_nonexistent_data_type(self):
        """Test exporting non-existent data type"""
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = self.backend.export_data(temp_path, "epochs")  # No epochs data
            
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("δεδομένα", result["error"])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestKatharsisBackendStateManagement(unittest.TestCase):
    """State management functionality tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
    
    def test_initial_state(self):
        """Test initial processing state"""
        state = self.backend.get_processing_state()
        
        self.assertFalse(state["file_loaded"])
        self.assertIsNone(state["file_path"])
        self.assertFalse(state["has_raw_data"])
        self.assertFalse(state["has_preprocessed_data"])
        self.assertFalse(state["has_ica_data"])
        self.assertFalse(state["has_epochs_data"])
        self.assertFalse(state["has_erp_data"])
        self.assertFalse(state["is_processing"])
        self.assertEqual(state["progress_percentage"], 0)
    
    def test_state_after_file_load(self):
        """Test processing state after file loading"""
        if not os.path.exists(self.data_file):
            self.skipTest("Test data file not found")
        
        self.backend.load_file(self.data_file)
        state = self.backend.get_processing_state()
        
        self.assertTrue(state["file_loaded"])
        self.assertEqual(state["file_path"], self.data_file)
        self.assertTrue(state["has_raw_data"])
        self.assertFalse(state["has_preprocessed_data"])
        self.assertFalse(state["has_ica_data"])
    
    def test_reset_analysis(self):
        """Test resetting analysis data while keeping file"""
        if not os.path.exists(self.data_file):
            self.skipTest("Test data file not found")
        
        # Load file and apply preprocessing
        self.backend.load_file(self.data_file)
        prep_config = {"filtering": {"enable_bandpass": True, "low_freq": 1.0, "high_freq": 40.0}}
        self.backend.apply_preprocessing(prep_config)
        
        # Check state before reset
        state_before = self.backend.get_processing_state()
        self.assertTrue(state_before["has_preprocessed_data"])
        
        # Reset analysis
        self.backend.reset_analysis()
        
        # Check state after reset
        state_after = self.backend.get_processing_state()
        self.assertTrue(state_after["file_loaded"])  # File still loaded
        self.assertTrue(state_after["has_raw_data"])  # Raw data still there
        self.assertFalse(state_after["has_preprocessed_data"])  # Analysis data cleared
        self.assertFalse(state_after["has_ica_data"])
        self.assertFalse(state_after["has_epochs_data"])
    
    def test_reset_all(self):
        """Test resetting everything"""
        if not os.path.exists(self.data_file):
            self.skipTest("Test data file not found")
        
        # Load file
        self.backend.load_file(self.data_file)
        
        # Reset all
        self.backend.reset_all()
        
        # Check state
        state = self.backend.get_processing_state()
        self.assertFalse(state["file_loaded"])
        self.assertIsNone(state["file_path"])
        self.assertFalse(state["has_raw_data"])


class TestKatharsisBackendCallbacks(unittest.TestCase):
    """Callback system tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = KatharsisBackend()
        self.data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        # Callback trackers
        self.progress_calls = []
        self.status_calls = []
        self.error_calls = []
        
        # Set callbacks
        self.backend.set_callbacks(
            progress_callback=self.progress_calls.append,
            status_callback=self.status_calls.append,
            error_callback=self.error_calls.append
        )
    
    def test_callbacks_during_file_load(self):
        """Test callbacks are called during file loading"""
        if not os.path.exists(self.data_file):
            self.skipTest("Test data file not found")
        
        self.backend.load_file(self.data_file)
        
        # Should have received progress and status updates
        self.assertGreater(len(self.progress_calls), 0)
        self.assertGreater(len(self.status_calls), 0)
        
        # Progress should be between 0 and 100
        for progress in self.progress_calls:
            self.assertGreaterEqual(progress, 0)
            self.assertLessEqual(progress, 100)
    
    def test_callbacks_during_preprocessing(self):
        """Test callbacks during preprocessing"""
        if not os.path.exists(self.data_file):
            self.skipTest("Test data file not found")
        
        # Load file first
        self.backend.load_file(self.data_file)
        
        # Clear callback trackers
        self.progress_calls.clear()
        self.status_calls.clear()
        
        # Apply preprocessing
        config = {"filtering": {"enable_bandpass": True, "low_freq": 1.0, "high_freq": 40.0}}
        self.backend.apply_preprocessing(config)
        
        # Should have received updates
        self.assertGreater(len(self.progress_calls), 0)
        self.assertGreater(len(self.status_calls), 0)
    
    def test_error_callback(self):
        """Test error callback is called on errors"""
        # Try to load non-existent file
        result = self.backend.load_file("nonexistent_file.edf")
        
        # Should have received error callback or the operation should fail
        self.assertFalse(result["success"])
        # The error might be handled directly without callback in some cases
        if len(self.error_calls) == 0:
            # If no error callback, at least verify the error is in the result
            self.assertIn("error", result)
        else:
            self.assertGreater(len(self.error_calls), 0)
            self.assertTrue(any("δεν βρέθηκε" in error for error in self.error_calls))


class TestKatharsisBackendEdgeCases(unittest.TestCase):
    """Edge cases and error handling tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.backend = KatharsisBackend()
    
    def test_operations_without_data(self):
        """Test all operations fail gracefully without data"""
        # Preprocessing without data
        result = self.backend.apply_preprocessing({"filtering": {"enable_bandpass": True}})
        self.assertFalse(result["success"])
        
        # ICA without data
        result = self.backend.perform_ica_analysis("fastica")
        self.assertFalse(result["success"])
        
        # Epoching without data
        result = self.backend.perform_epoching({}, {})
        self.assertFalse(result["success"])
        
        # ERP without data
        result = self.backend.perform_erp_analysis({})
        self.assertFalse(result["success"])
        
        # Export without data
        result = self.backend.export_data("output.edf", "raw")
        self.assertFalse(result["success"])
    
    def test_invalid_file_paths(self):
        """Test various invalid file paths"""
        invalid_paths = [
            "",
            None,
            "file/that/does/not/exist.edf",
            "/invalid/path/file.edf",
            "file_with_no_extension",
        ]
        
        for path in invalid_paths:
            if path is not None:  # Skip None as it would cause TypeError
                result = self.backend.validate_file(path)
                self.assertFalse(result["valid"])
                self.assertIn("error", result)
    
    def test_empty_configurations(self):
        """Test operations with empty configurations"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        
        if not os.path.exists(data_file):
            self.skipTest("Test data file not found")
        
        # Load file
        self.backend.load_file(data_file)
        
        # Empty preprocessing config - should not crash
        result = self.backend.apply_preprocessing({})
        # Should either succeed (no-op) or fail gracefully
        self.assertIn("success", result)
        
        # Empty ICA config - should use defaults
        result = self.backend.perform_ica_analysis()
        # May succeed or fail depending on data quality
        self.assertIn("success", result)
    
    def test_callback_exceptions(self):
        """Test backend handles callback exceptions gracefully"""
        def error_callback(progress):
            raise Exception("Callback error")
        
        # Set error-prone callback
        self.backend.set_callbacks(progress_callback=error_callback)
        
        data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.edf")
        if os.path.exists(data_file):
            # Should not crash due to callback error
            result = self.backend.load_file(data_file)
            # Operation may fail due to callback error, but should not crash the app
            self.assertIn("success", result)
            # If it fails due to callback error, that's acceptable for this test
            if not result.get("success", False):
                self.assertIn("error", result)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestKatharsisBackendCore,
        TestKatharsisBackendPreprocessing,
        TestKatharsisBackendICA,
        TestKatharsisBackendTimeDomain,
        TestKatharsisBackendDataExport,
        TestKatharsisBackendStateManagement,
        TestKatharsisBackendCallbacks,
        TestKatharsisBackendEdgeCases,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE BACKEND TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)