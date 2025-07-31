#!/usr/bin/env python3
"""
Test Real Issues - Reproduce User-Reported Problems
==================================================

Test script to reproduce and fix the specific issues reported by the user:
1. Time-domain analysis failing due to missing 'STI 014' channel
2. ICA analysis not working properly

Author: porfanid
Version: 1.0
"""

import sys
import numpy as np
import mne
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

def test_stimulus_channel_issue():
    """Test the 'STI 014' channel issue that the user reported"""
    print("🔍 Testing stimulus channel detection issue...")
    
    try:
        from backend import EpochingProcessor, EpochingConfig
        
        # Create test data WITHOUT STI 014 channel (simulates real EEG files)
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types='eeg')
        data = np.random.randn(len(ch_names), 10000)  # 40 seconds of data
        raw = mne.io.RawArray(data, info, verbose=False)
        
        print(f"   Created test data with channels: {raw.ch_names}")
        print(f"   No stimulus channels present")
        
        # Try to use the epoching processor (this should fail with current code)
        processor = EpochingProcessor()
        config = EpochingConfig()
        
        try:
            # This is what fails in the GUI - looking for 'STI 014'
            events = processor.find_events_from_raw(raw)
            print(f"   ❌ Unexpected success: found {len(events)} events")
            return False
        except Exception as e:
            if "no stimulus channels found and no annotations available" in str(e).lower():
                print(f"   ✅ Fixed issue: Smart channel detection working - {str(e)}")
                return True
            elif "STI 014" in str(e):
                print(f"   ❌ Still old issue: {str(e)}")
                return False
            else:
                print(f"   ❌ Different error: {str(e)}")
                return False
                
    except Exception as e:
        print(f"   ❌ Test setup failed: {str(e)}")
        return False

def test_stimulus_channel_with_annotations():
    """Test creating events from annotations when no stimulus channel exists"""
    print("\n🔍 Testing annotation-based event creation...")
    
    try:
        from backend import EpochingProcessor, EpochingConfig
        
        # Create test data with annotations instead of stimulus channel
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types='eeg')
        data = np.random.randn(len(ch_names), 10000)  # 40 seconds of data
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Add some annotations (this is how many EEG files store events)
        annotations = mne.Annotations(
            onset=[5.0, 10.0, 15.0, 20.0, 25.0],  # Event times
            duration=[0.0, 0.0, 0.0, 0.0, 0.0],   # Duration
            description=['Stimulus', 'Stimulus', 'Response', 'Stimulus', 'Response']
        )
        raw.set_annotations(annotations)
        
        print(f"   Created test data with annotations: {len(annotations)} events")
        
        processor = EpochingProcessor()
        
        try:
            events, event_id = processor.create_events_from_annotations(raw)
            print(f"   ✅ Successfully created events from annotations: {len(events)} events")
            print(f"   Event IDs: {event_id}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to create events from annotations: {str(e)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test setup failed: {str(e)}")
        return False

def test_ica_issue():
    """Test ICA processing to identify potential issues"""
    print("\n🔍 Testing ICA processing...")
    
    try:
        from backend import EnhancedICAProcessor, ICAConfig, ICAMethod
        
        # Create realistic test data
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types='eeg')
        
        # Create realistic test data with actual signal structures
        n_samples = 25000  # 100 seconds
        data = np.zeros((len(ch_names), n_samples))
        
        # Create base EEG-like signals with different frequency components
        time = np.arange(n_samples) / 250.0
        for i, ch in enumerate(ch_names):
            # Base brain activity (mix of alpha, beta, gamma)
            alpha = np.sin(2 * np.pi * 10 * time) * np.exp(-time/50)  # 10 Hz alpha
            beta = np.sin(2 * np.pi * 20 * time) * 0.5 * np.exp(-time/100)  # 20 Hz beta
            gamma = np.random.normal(0, 0.3, n_samples)  # Random gamma activity
            
            # Add channel-specific components
            if 'Fp' in ch:  # Frontal channels - add eye artifacts
                eye_blinks = np.zeros(n_samples)
                blink_times = [5000, 15000, 20000]
                for bt in blink_times:
                    if bt < n_samples - 500:
                        blink_artifact = np.exp(-(np.arange(500) - 250)**2 / 10000) * 100
                        eye_blinks[bt:bt+500] = blink_artifact
                data[i] = (alpha + beta + gamma + eye_blinks) * 1e-6
            else:
                data[i] = (alpha + beta + gamma) * 1e-6
        
        raw = mne.io.RawArray(data, info, verbose=False)
        
        print(f"   Created test data: {len(ch_names)} channels, {n_samples/250:.1f} seconds")
        
        # Test ICA processing
        processor = EnhancedICAProcessor()
        config = ICAConfig(n_components=6, random_state=42, method=ICAMethod.MNE_DEFAULT, max_iter=1000)
        processor.config = config  # Set the config properly
        
        try:
            results = processor.run_ica_analysis(raw)
            
            if 'error' in results:
                print(f"   ❌ ICA processing failed: {results['error']}")
                return False
            elif results and 'ica' in results and results['ica'] is not None:
                ica = results['ica']
                print(f"   ✅ ICA analysis successful: {ica.n_components_} components")
                
                if 'component_classification' in results:
                    classifications = results['component_classification']
                    print(f"   ✅ Component classification successful: {len(classifications)} components classified")
                    return True
                else:
                    print(f"   ⚠️  ICA computed but no component classification")
                    return False
            else:
                print(f"   ❌ ICA analysis failed: no results returned or None ICA object")
                return False
                
        except Exception as e:
            print(f"   ❌ ICA processing failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test setup failed: {str(e)}")
        return False

def test_data_loading():
    """Test loading the actual data.edf file if it exists"""
    print("\n🔍 Testing actual data file loading...")
    
    data_file = Path("data.edf")
    if not data_file.exists():
        print("   ⚠️  No data.edf file found, skipping real data test")
        return True
    
    try:
        print(f"   Loading {data_file}...")
        raw = mne.io.read_raw_edf(str(data_file), preload=True, verbose=False)
        
        print(f"   ✅ File loaded successfully:")
        print(f"      Channels: {len(raw.ch_names)} ({raw.ch_names[:5]}...)")
        print(f"      Duration: {raw.times[-1]:.1f} seconds")
        print(f"      Sampling rate: {raw.info['sfreq']} Hz")
        
        # Check for stimulus channels
        stim_channels = [ch for ch in raw.ch_names if 'STI' in ch.upper() or 'TRIG' in ch.upper()]
        print(f"      Stimulus channels: {stim_channels if stim_channels else 'None found'}")
        
        # Check for annotations
        if raw.annotations:
            print(f"      Annotations: {len(raw.annotations)} found")
        else:
            print(f"      Annotations: None found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to load data file: {str(e)}")
        return False

def main():
    """Run all issue reproduction tests"""
    print("=" * 60)
    print("🚨 REAL ISSUES TESTING - Reproducing User Problems")
    print("=" * 60)
    
    tests = [
        ("Stimulus Channel Issue", test_stimulus_channel_issue),
        ("Annotation-based Events", test_stimulus_channel_with_annotations),
        ("ICA Processing", test_ica_issue),
        ("Real Data Loading", test_data_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test crashed: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"🧪 TEST RESULTS: {passed}/{total} tests identified issues correctly")
    
    if passed == total:
        print("🎯 All issues successfully reproduced - ready to implement fixes!")
    else:
        print("⚠️  Some tests failed - need to investigate further")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)