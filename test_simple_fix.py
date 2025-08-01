#!/usr/bin/env python3
"""
Simple test for channel mismatch fix - focused on real user scenario
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import mne
from backend.eeg_service import EEGArtifactCleaningService

def create_realistic_eeg_data():
    """Create realistic EEG data that works well with ICA"""
    n_channels = 4
    n_samples = 5000  # 20 seconds at 250 Hz
    sfreq = 250
    
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create realistic EEG-like data
    t = np.linspace(0, n_samples/sfreq, n_samples)
    data = np.zeros((n_channels, n_samples))
    
    # Add realistic EEG components for each channel
    for ch in range(n_channels):
        # Base EEG activity
        alpha = 0.8 * np.sin(2 * np.pi * 10 * t + ch * np.pi/4)  # 10 Hz alpha
        beta = 0.4 * np.sin(2 * np.pi * 20 * t + ch * np.pi/3)   # 20 Hz beta
        theta = 0.6 * np.sin(2 * np.pi * 6 * t + ch * np.pi/6)   # 6 Hz theta
        
        # Add some artifacts for ICA to find
        if ch == 0:  # Eye blink artifact in frontal channel
            blink_times = [1.0, 3.5, 8.2, 12.1, 16.8]  # seconds
            for blink_t in blink_times:
                blink_idx = int(blink_t * sfreq)
                if blink_idx < n_samples - 100:
                    blink = 3.0 * np.exp(-((t[blink_idx:blink_idx+100] - blink_t)**2) / 0.01)
                    data[ch, blink_idx:blink_idx+100] += blink
        
        # Combine components
        data[ch, :] = alpha + beta + theta + 0.1 * np.random.randn(n_samples)
        
        # Scale to realistic EEG amplitudes (microvolts)
        data[ch, :] *= 10e-6
    
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

def test_user_scenario():
    """Test the user's specific scenario"""
    print("=== Testing User's Channel Mismatch Scenario ===")
    
    # Create realistic test data
    raw = create_realistic_eeg_data()
    print(f"Created test data: {len(raw.ch_names)} channels, {raw.n_times} samples")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    
    # Test with EEG service (this is where the user encountered the error)
    service = EEGArtifactCleaningService()
    
    print("\n1. Loading preprocessed data...")
    try:
        load_result = service.load_preprocessed_data(raw)
        
        if load_result['success']:
            print(f"✅ Data loaded successfully")
            print(f"   Channels: {len(load_result['channels'])}")
            print(f"   Sampling rate: {load_result['sampling_rate']} Hz")
            print(f"   Samples: {load_result['n_samples']}")
            
            if 'fixes_applied' in load_result:
                print(f"   Fixes applied: {load_result['fixes_applied']}")
        else:
            print(f"❌ Data loading failed: {load_result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during data loading: {str(e)}")
        return False
    
    print("\n2. Attempting ICA analysis...")
    try:
        ica_result = service.fit_ica_analysis()
        
        if ica_result['success']:
            print(f"✅ ICA analysis completed successfully!")
            print(f"   Method: {ica_result.get('method', 'Unknown')}")
            print(f"   Components: {ica_result['n_components']}")
            print(f"   Explained variance: {ica_result.get('explained_variance', 0):.1%}")
            
            if 'auto_classifications' in ica_result:
                print(f"   Auto classifications: {ica_result['auto_classifications']}")
            if 'auto_reject_count' in ica_result:
                print(f"   Auto reject suggestions: {ica_result['auto_reject_count']}")
                
            return True
        else:
            print(f"❌ ICA analysis failed: {ica_result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during ICA analysis: {str(e)}")
        return False

def test_edge_cases():
    """Test various edge cases that could cause channel mismatch"""
    print("\n=== Testing Edge Cases ===")
    
    test_cases = []
    
    # Case 1: Very few channels
    print("\nCase 1: Few channels (2 channels)")
    raw = create_realistic_eeg_data()
    raw_2ch = raw.copy().pick(['EEG1', 'EEG2'])
    test_cases.append(('2_channels', raw_2ch))
    
    # Case 2: Short data
    print("Case 2: Short data (2 seconds)")
    raw = create_realistic_eeg_data()
    raw_short = raw.copy().crop(0, 2.0)  # Only 2 seconds
    test_cases.append(('short_data', raw_short))
    
    # Case 3: Normal case
    print("Case 3: Normal data")
    raw_normal = create_realistic_eeg_data()
    test_cases.append(('normal', raw_normal))
    
    success_count = 0
    
    for case_name, raw_data in test_cases:
        print(f"\nTesting {case_name}:")
        print(f"  Data shape: {raw_data.get_data().shape}")
        print(f"  Info channels: {len(raw_data.ch_names)}")
        
        try:
            service = EEGArtifactCleaningService()
            load_result = service.load_preprocessed_data(raw_data)
            
            if load_result['success']:
                ica_result = service.fit_ica_analysis()
                if ica_result['success']:
                    print(f"  ✅ SUCCESS - Components: {ica_result['n_components']}")
                    success_count += 1
                else:
                    print(f"  ⚠️  ICA failed: {ica_result['error'][:80]}...")
            else:
                print(f"  ❌ Load failed: {load_result['error'][:80]}...")
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:80]}...")
    
    print(f"\nEdge case results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

if __name__ == "__main__":
    print("Testing Channel Mismatch Fix for User-Reported Issue")
    print("=" * 60)
    
    # Test the main user scenario
    main_success = test_user_scenario()
    
    # Test edge cases
    edge_success = test_edge_cases()
    
    print("\n" + "=" * 60)
    if main_success and edge_success:
        print("🎉 ALL TESTS PASSED! Channel mismatch issue should be resolved.")
    elif main_success:
        print("✅ Main scenario works, some edge cases may still have issues.")
    else:
        print("❌ Main scenario still has issues that need to be addressed.")
    print("=" * 60)