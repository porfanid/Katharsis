#!/usr/bin/env python3
"""
Integration test for ICLabel in the full service pipeline
"""

import numpy as np
import mne
from backend.eeg_service import EEGArtifactCleaningService
from backend.ica_processor import ICAProcessor

def create_test_data():
    """Create test EEG data"""
    n_channels = 10
    n_times = 2000  # More samples for ICA
    sfreq = 250
    
    # Generate dummy EEG data
    data = np.random.randn(n_channels, n_times) * 1e-6
    
    # Add some artifact-like signals
    # Eye blink artifact
    data[0, 100:150] += 50e-6 * np.ones(50)  # Strong frontal activity
    
    # Muscle artifact  
    data[5:7, 500:600] += 10e-6 * np.random.randn(2, 100)  # High frequency
    
    # Standard 10-20 channel names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels
    
    # Create info and raw
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Set montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, verbose=False)
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")
    
    return raw

def test_full_pipeline():
    """Test the full ICLabel integration pipeline"""
    print("Testing full ICLabel integration pipeline...")
    
    try:
        # Create test data
        raw = create_test_data()
        print("✓ Created test EEG data")
        
        # Save as temporary EDF for testing
        temp_file = "/tmp/test_data.edf"
        raw.export(temp_file, fmt='edf', overwrite=True, verbose=False)
        print("✓ Saved test data")
        
        # Initialize service
        service = EEGArtifactCleaningService()
        print("✓ Created EEGArtifactCleaningService")
        
        # Load and prepare file
        result = service.load_and_prepare_file(temp_file)
        if not result['success']:
            print(f"✗ Failed to load file: {result.get('error', 'Unknown error')}")
            return False
        print("✓ Loaded and prepared file")
        
        # Fit ICA
        result = service.fit_ica_analysis()
        if not result['success']:
            print(f"✗ Failed to fit ICA: {result.get('error', 'Unknown error')}")
            return False
        print("✓ Fitted ICA analysis")
        
        # Detect artifacts with ICLabel
        result = service.detect_artifacts()
        if not result['success']:
            print(f"✗ Failed to detect artifacts: {result.get('error', 'Unknown error')}")
            return False
        print("✓ Detected artifacts with ICLabel")
        
        # Check results
        suggested_artifacts = result['suggested_artifacts']
        icalabel_info = result.get('icalabel_info', {})
        explanations = result['explanations']
        
        print(f"  Found {len(suggested_artifacts)} suggested artifacts")
        
        if icalabel_info:
            print("  ICLabel categories:")
            for comp_idx, info in icalabel_info.items():
                category = info['icalabel_category']
                probability = info['icalabel_probability']
                emoji = info['icalabel_emoji']
                is_artifact = info['is_artifact']
                print(f"    IC {comp_idx}: {emoji} {category} ({probability:.1%}) {'[ARTIFACT]' if is_artifact else ''}")
        
        print("  Component explanations:")
        for i, explanation in explanations.items():
            print(f"    IC {i}: {explanation}")
        
        # Test visualization data
        viz_data = service.get_component_visualization_data()
        if viz_data and 'icalabel_info' in viz_data:
            print("✓ Visualization data includes ICLabel info")
        else:
            print("! Visualization data missing ICLabel info")
        
        print("✓ Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n🎉 Full integration test passed!")
    else:
        print("\n❌ Integration test failed!")