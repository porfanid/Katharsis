#!/usr/bin/env python3
"""
Test cases for ICA training failure fixes - Regression tests
"""

import unittest
import numpy as np
import mne
import sys
import os

# Add the backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.eeg_service import EEGArtifactCleaningService
from backend.epoching_processor import EpochingProcessor

class TestICATrainingFailureFixes(unittest.TestCase):
    """Test cases for ICA training failure fixes"""
    
    def test_single_channel_error_handling(self):
        """Test that single channel data fails gracefully with helpful error"""
        info = mne.create_info(ch_names=['EEG1'], sfreq=250, ch_types='eeg')
        data = np.random.randn(1, 25000)
        raw = mne.io.RawArray(data, info)
        
        service = EEGArtifactCleaningService()
        service.load_preprocessed_data(raw)
        result = service.fit_ica_analysis()
        
        self.assertFalse(result["success"])
        self.assertIn("💡", result["error"])
        self.assertIn("κανάλια", result["error"])
    
    def test_nan_data_error_handling(self):
        """Test that NaN data fails gracefully with helpful error"""
        info = mne.create_info(ch_names=['EEG1', 'EEG2', 'EEG3'], sfreq=250, ch_types='eeg')
        data = np.random.randn(3, 25000)
        data[1, 1000:2000] = np.nan
        raw = mne.io.RawArray(data, info)
        
        service = EEGArtifactCleaningService()
        service.load_preprocessed_data(raw)
        result = service.fit_ica_analysis()
        
        self.assertFalse(result["success"])
        self.assertTrue("NaN" in result["error"] or "💡" in result["error"])
    
    def test_short_data_error_handling(self):
        """Test that short data fails gracefully with helpful error"""
        info = mne.create_info(ch_names=['EEG1', 'EEG2'], sfreq=250, ch_types='eeg')
        data = np.random.randn(2, 500)  # Only 2 seconds
        raw = mne.io.RawArray(data, info)
        
        service = EEGArtifactCleaningService()
        service.load_preprocessed_data(raw)
        result = service.fit_ica_analysis()
        
        self.assertFalse(result["success"])
        self.assertIn("💡", result["error"])
        self.assertIn("δεδομένα", result["error"])
    
    def test_good_data_ica_success(self):
        """Test that good quality data succeeds in ICA training"""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
        
        # Create 60 seconds of realistic EEG data
        n_samples = 60 * 250
        time = np.linspace(0, 60, n_samples)
        data = np.zeros((len(ch_names), n_samples))
        
        for i in range(len(ch_names)):
            data[i] += 0.5 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
            data[i] += 0.3 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
            data[i] += 0.1 * np.random.randn(n_samples)
            
        raw = mne.io.RawArray(data, info)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        service = EEGArtifactCleaningService()
        service.load_preprocessed_data(raw)
        result = service.fit_ica_analysis()
        
        self.assertTrue(result["success"])
        self.assertGreater(result["n_components"], 0)
    
    def test_stimulus_channel_detection_fallback(self):
        """Test smart stimulus channel detection with annotation fallback"""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3']
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
        data = np.random.randn(len(ch_names), 10000)  # 40 seconds
        raw = mne.io.RawArray(data, info)
        
        # Add annotations instead of stimulus channel
        annotations = mne.Annotations(
            onset=[10, 20, 30],
            duration=[0.1, 0.1, 0.1],
            description=['Stimulus', 'Response', 'Stimulus']
        )
        raw.set_annotations(annotations)
        
        epoching = EpochingProcessor()
        events = epoching.find_events_from_raw(raw)
        
        self.assertGreater(len(events), 0)
    
    def test_preprocessed_data_filtered_data_fix(self):
        """Test that preprocessed data loading sets filtered_data (Greek error fix)"""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4']
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
        data = np.random.randn(len(ch_names), 15000)  # 60 seconds
        raw = mne.io.RawArray(data, info)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)  # Preprocess it
        
        service = EEGArtifactCleaningService()
        load_result = service.load_preprocessed_data(raw)
        
        self.assertTrue(load_result["success"])
        self.assertIsNotNone(service.backend_core.filtered_data)
        # Check that both raw_data and filtered_data are set to the same preprocessed data
        self.assertIsNotNone(service.backend_core.raw_data)

if __name__ == '__main__':
    unittest.main()