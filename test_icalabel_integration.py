#!/usr/bin/env python3
"""
Test script for ICLabel integration
"""

import numpy as np
import mne
from backend.artifact_detector import ArtifactDetector

def create_dummy_data():
    """Create dummy EEG data for testing"""
    # Create a simple dummy dataset
    n_channels = 10
    n_times = 1000
    sfreq = 250
    
    # Generate dummy EEG data
    data = np.random.randn(n_channels, n_times) * 1e-6
    
    # Create standard 10-20 channel names for ICLabel compatibility
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels
    
    # Create info
    info = mne.create_info(ch_names, sfreq, ch_types)
    
    # Create raw object
    raw = mne.io.RawArray(data, info)
    
    # Set standard montage for ICLabel
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, verbose=False)
        print("✓ Set standard montage")
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")
    
    return raw

def test_icalabel_integration():
    """Test ICLabel integration"""
    print("Testing ICLabel integration...")
    
    try:
        # Create dummy data
        raw = create_dummy_data()
        print("✓ Created dummy EEG data")
        
        # Filter data
        raw_filtered = raw.copy().filter(1, 40, verbose=False)
        print("✓ Filtered data")
        
        # Fit ICA (with fewer components for speed)
        ica = mne.preprocessing.ICA(n_components=5, random_state=42, verbose=False)
        ica.fit(raw_filtered, verbose=False)
        print("✓ Fitted ICA")
        
        # Test ArtifactDetector
        detector = ArtifactDetector()
        print("✓ Created ArtifactDetector")
        
        # Test ICLabel detection
        artifacts, icalabel_info = detector.detect_with_icalabel(ica, raw_filtered)
        print(f"✓ ICLabel detection completed: {len(artifacts)} artifacts found")
        
        if icalabel_info:
            print("ICLabel categories detected:")
            for comp_idx, info in icalabel_info.items():
                category = info['icalabel_category']
                probability = info['icalabel_probability']
                emoji = info['icalabel_emoji']
                print(f"  IC {comp_idx}: {emoji} {category} ({probability:.1%})")
        else:
            print("No ICLabel info (might be expected with dummy data)")
            
        print("✓ ICLabel integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_icalabel_integration()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")