#!/usr/bin/env python3
"""
Test for Greek error fix - Preprocessed data ICA integration
"""

import pytest
import numpy as np
import mne
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend import EEGArtifactCleaningService


def create_test_raw_data(n_channels=8, n_times=1000, sfreq=256.0):
    """Create synthetic EEG data for testing"""
    # Generate sample EEG data
    times = np.arange(n_times) / sfreq
    data = np.random.randn(n_channels, n_times) * 1e-6  # Convert to Volts
    
    # Add some structured signals
    for i in range(n_channels):
        data[i] += 0.1e-6 * np.sin(2 * np.pi * 10 * times)  # 10 Hz signal
    
    # Create channel names
    ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # Create MNE info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    return raw


class TestPreprocessedDataICA:
    """Test suite for preprocessed data ICA integration"""
    
    def test_load_preprocessed_data_sets_filtered_data(self):
        """Test that load_preprocessed_data properly sets filtered_data"""
        # Create test data
        raw_data = create_test_raw_data()
        
        # Apply preprocessing (filtering)
        preprocessed_data = raw_data.copy()
        preprocessed_data.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # Initialize service
        service = EEGArtifactCleaningService()
        
        # Load preprocessed data
        result = service.load_preprocessed_data(preprocessed_data)
        
        # Check that loading succeeded
        assert result["success"], f"Failed to load preprocessed data: {result.get('error')}"
        
        # Check that filtered data is available
        filtered_data = service.backend_core.get_filtered_data()
        assert filtered_data is not None, "Filtered data should not be None after loading preprocessed data"
        
        # Check that the filtered data is the same as the preprocessed data
        assert filtered_data.ch_names == preprocessed_data.ch_names
        assert filtered_data.n_times == preprocessed_data.n_times
        assert filtered_data.info['sfreq'] == preprocessed_data.info['sfreq']
    
    def test_ica_analysis_with_preprocessed_data(self):
        """Test that ICA analysis works with preprocessed data (fixes Greek error)"""
        # Create test data
        raw_data = create_test_raw_data()
        
        # Apply preprocessing
        preprocessed_data = raw_data.copy()
        preprocessed_data.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # Initialize service
        service = EEGArtifactCleaningService()
        
        # Load preprocessed data
        load_result = service.load_preprocessed_data(preprocessed_data)
        assert load_result["success"], f"Failed to load preprocessed data: {load_result.get('error')}"
        
        # Try ICA analysis - this should work now (not give Greek error)
        ica_result = service.fit_ica_analysis()
        
        # Check that ICA analysis succeeded
        assert ica_result["success"], f"ICA analysis failed: {ica_result.get('error')}"
        assert "n_components" in ica_result
        assert ica_result["n_components"] > 0
        
        # Verify the error message is NOT the Greek error
        if not ica_result["success"]:
            error_msg = ica_result.get("error", "")
            assert "Δεν υπάρχουν φιλτραρισμένα δεδομένα" not in error_msg, \
                   "Should not get the Greek 'no filtered data' error"
    
    def test_backward_compatibility_file_loading(self):
        """Test that the old file loading path still works"""
        # Create test data
        raw_data = create_test_raw_data()
        
        # Initialize service
        service = EEGArtifactCleaningService()
        
        # Simulate old loading path
        service.backend_core.raw_data = raw_data
        service.backend_core.filtered_data = raw_data.copy().filter(1.0, 40.0, verbose=False)
        service.is_processing = True
        
        # Test that filtered data is available
        filtered_data = service.backend_core.get_filtered_data()
        assert filtered_data is not None, "Filtered data should be available in old loading path"
        
        # Test that ICA analysis works
        ica_result = service.fit_ica_analysis()
        assert ica_result["success"], f"ICA analysis failed in old loading path: {ica_result.get('error')}"
    
    def test_preprocessed_data_attributes(self):
        """Test that all required attributes are set when loading preprocessed data"""
        # Create test data
        raw_data = create_test_raw_data(n_channels=6, n_times=2000)
        preprocessed_data = raw_data.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        # Initialize service
        service = EEGArtifactCleaningService()
        
        # Load preprocessed data
        result = service.load_preprocessed_data(preprocessed_data)
        assert result["success"]
        
        # Check all required attributes are set
        assert service.backend_core.raw_data is not None
        assert service.backend_core.filtered_data is not None
        assert hasattr(service.backend_core, 'data')
        assert hasattr(service.backend_core, 'info')
        assert hasattr(service.backend_core, 'sfreq')
        assert hasattr(service.backend_core, 'channels')
        
        # Check values are correct
        assert service.backend_core.sfreq == preprocessed_data.info['sfreq']
        assert service.backend_core.channels == preprocessed_data.ch_names
        assert service.is_processing == True
        assert service.ica_fitted == False


if __name__ == "__main__":
    # Run tests when executed directly
    test_suite = TestPreprocessedDataICA()
    
    print("Running Greek error fix tests...")
    print("=" * 50)
    
    try:
        test_suite.test_load_preprocessed_data_sets_filtered_data()
        print("✓ Test 1: Load preprocessed data sets filtered data")
        
        test_suite.test_ica_analysis_with_preprocessed_data()
        print("✓ Test 2: ICA analysis with preprocessed data")
        
        test_suite.test_backward_compatibility_file_loading()
        print("✓ Test 3: Backward compatibility file loading")
        
        test_suite.test_preprocessed_data_attributes()
        print("✓ Test 4: Preprocessed data attributes")
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED - Greek error fix is working!")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ TEST ERROR: {str(e)}")
        sys.exit(1)