#!/usr/bin/env python3
"""
Test to reproduce the channel mismatch error
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import mne
from backend.eeg_service import EEGArtifactCleaningService

def create_test_data_with_mismatch():
    """Create test data that might cause the channel mismatch error"""
    # Create synthetic EEG data
    n_channels = 5
    n_samples = 2000
    sfreq = 250
    
    # Create channel names
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create synthetic data
    data = np.random.randn(n_channels, n_samples) * 1e-6  # Simulate EEG data in Volts
    
    # Create raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    print(f"Original: {len(raw.ch_names)} channels, data shape: {raw.get_data().shape}")
    
    # Simulate what might happen during preprocessing that causes the mismatch
    # Option 1: Manually drop a channel from data but not from info (this would cause the error)
    # Let's simulate this by creating a scenario similar to what happens in our pipeline
    
    return raw

def test_channel_mismatch_detection():
    """Test the channel mismatch issue"""
    print("Testing channel mismatch detection...")
    
    # Create test data
    raw = create_test_data_with_mismatch()
    
    # Try to process with our service
    service = EEGArtifactCleaningService()
    
    try:
        # Load the data
        result = service.load_preprocessed_data(raw)
        print(f"Load result: {result}")
        
        # Try to fit ICA - this is where the error might occur
        ica_result = service.fit_ica_analysis()
        print(f"ICA result: {ica_result}")
        
    except Exception as e:
        print(f"Error encountered: {e}")
        print(f"Error type: {type(e)}")
        return str(e)
    
    return "No error"

def simulate_channel_mismatch():
    """Simulate the exact channel mismatch scenario"""
    print("\nSimulating channel mismatch scenario...")
    
    # Create data with 5 channels
    n_channels = 5
    n_samples = 2000
    sfreq = 250
    
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.randn(n_channels, n_samples) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    
    print(f"Before: {len(raw.ch_names)} channels in info, data shape: {raw.get_data().shape}")
    
    # Simulate what happens during preprocessing that might cause mismatch
    # Let's say we remove one channel from the data but don't update info properly
    
    # Method 1: Manually create mismatch (this would cause the error)
    # raw._data = raw._data[:-1, :]  # Remove last channel from data
    # This would leave info with 5 channels but data with 4 channels
    
    # Method 2: Let's see what our preprocessing does
    from backend.preprocessing_pipeline import PreprocessingPipeline, PreprocessingPresets
    
    pipeline = PreprocessingPipeline()
    config = PreprocessingPresets.get_minimal_preset()
    
    try:
        processed_raw, results = pipeline.run_pipeline(raw, config)
        print(f"After preprocessing: {len(processed_raw.ch_names)} channels in info, data shape: {processed_raw.get_data().shape}")
        
        # Check for mismatch
        if len(processed_raw.ch_names) != processed_raw.get_data().shape[0]:
            print(f"MISMATCH DETECTED: {len(processed_raw.ch_names)} channels in info vs {processed_raw.get_data().shape[0]} in data")
            return processed_raw
        else:
            print("No mismatch detected in preprocessing")
            
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None
    
    return processed_raw

if __name__ == "__main__":
    print("=== Testing Channel Mismatch Issue ===")
    
    # Test 1: Basic channel mismatch detection
    error = test_channel_mismatch_detection()
    print(f"Test 1 result: {error}")
    
    # Test 2: Simulate preprocessing scenarios
    result = simulate_channel_mismatch()
    
    print("\n=== Test completed ===")