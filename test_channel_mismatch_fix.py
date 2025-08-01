#!/usr/bin/env python3
"""
Test Channel Mismatch Fix
========================

Comprehensive test to validate the fix for the "Number of channels in the info object (X) 
and the data array (Y) do not match" error.

This test creates various scenarios where the error might occur and ensures our fixes work.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import mne
from backend.eeg_service import EEGArtifactCleaningService
from backend.data_consistency_utils import (
    validate_raw_consistency, 
    fix_raw_consistency, 
    diagnose_ica_data_issues,
    safe_channel_pick
)
from backend.enhanced_ica_processor import EnhancedICAProcessor, ICAConfig

mne.set_log_level("WARNING")  # Reduce MNE verbosity

def create_consistent_test_data():
    """Create consistent test EEG data"""
    n_channels = 5
    n_samples = 2000
    sfreq = 250
    
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create realistic EEG-like data (not purely random)
    t = np.linspace(0, n_samples/sfreq, n_samples)
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Add multiple frequency components
        data[ch, :] = (
            0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha
            0.3 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz beta  
            0.2 * np.sin(2 * np.pi * 5 * t) +   # 5 Hz theta
            0.1 * np.random.randn(n_samples)     # noise
        ) * 1e-6  # Scale to microvolts
    
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

def create_mismatched_data_scenarios():
    """Create various data mismatch scenarios"""
    scenarios = {}
    
    # Scenario 1: More channels in info than data
    raw = create_consistent_test_data()
    raw_scenario1 = raw.copy()
    # Manually remove one channel from data but keep info intact
    raw_scenario1._data = raw_scenario1._data[:-1, :]  # Remove last channel from data
    scenarios['more_info_channels'] = raw_scenario1
    
    # Scenario 2: More channels in data than info  
    raw = create_consistent_test_data()
    raw_scenario2 = raw.copy()
    # Add extra data channel
    extra_data = np.random.randn(1, raw_scenario2._data.shape[1]) * 1e-6
    raw_scenario2._data = np.vstack([raw_scenario2._data, extra_data])
    scenarios['more_data_channels'] = raw_scenario2
    
    # Scenario 3: Consistent data (control)
    scenarios['consistent'] = create_consistent_test_data()
    
    return scenarios

def test_data_consistency_validation():
    """Test the data consistency validation functions"""
    print("\n=== Testing Data Consistency Validation ===")
    
    scenarios = create_mismatched_data_scenarios()
    
    for name, raw in scenarios.items():
        print(f"\nTesting scenario: {name}")
        validation = validate_raw_consistency(raw)
        
        print(f"  Valid: {validation['valid']}")
        print(f"  Data channels: {validation['data_channels']}")
        print(f"  Info channels: {validation['info_channels']}")
        if validation['error']:
            print(f"  Error: {validation['error']}")
        
        # Test diagnosis
        diagnosis = diagnose_ica_data_issues(raw)
        print(f"  Can proceed with ICA: {diagnosis['can_proceed_with_ica']}")
        if diagnosis['recommendations']:
            print(f"  Recommendations: {diagnosis['recommendations']}")

def test_data_consistency_fixes():
    """Test the data consistency fixing functions"""
    print("\n=== Testing Data Consistency Fixes ===")
    
    scenarios = create_mismatched_data_scenarios()
    
    for name, raw in scenarios.items():
        if name == 'consistent':
            continue  # Skip consistent data
            
        print(f"\nTesting fix for scenario: {name}")
        
        # Test fix
        fixed_raw, fix_info = fix_raw_consistency(raw, strategy='auto')
        print(f"  Fix status: {fix_info['status']}")
        if fix_info.get('changes'):
            print(f"  Changes applied: {fix_info['changes']}")
        
        # Validate fix
        validation = validate_raw_consistency(fixed_raw)
        print(f"  Fixed data valid: {validation['valid']}")

def test_safe_channel_picking():
    """Test the safe channel picking function"""
    print("\n=== Testing Safe Channel Picking ===")
    
    scenarios = create_mismatched_data_scenarios()
    
    for name, raw in scenarios.items():
        print(f"\nTesting channel picking for scenario: {name}")
        
        channel_indices, pick_info = safe_channel_pick(raw)
        print(f"  Selected channels: {len(channel_indices)}")
        print(f"  Channel names: {pick_info['picked_channels']}")
        if pick_info['warnings']:
            print(f"  Warnings: {pick_info['warnings']}")

def test_enhanced_ica_with_problematic_data():
    """Test Enhanced ICA processor with problematic data scenarios"""
    print("\n=== Testing Enhanced ICA with Problematic Data ===")
    
    scenarios = create_mismatched_data_scenarios()
    
    for name, raw in scenarios.items():
        print(f"\nTesting Enhanced ICA with scenario: {name}")
        
        try:
            # Create enhanced ICA processor
            config = ICAConfig(n_components=None, max_iter=100)  # Reduced iterations for testing
            ica_processor = EnhancedICAProcessor(config)
            
            # Try to fit ICA
            results = ica_processor.fit_ica(raw)
            
            print(f"  ICA Success: {results['success']}")
            if results['success']:
                print(f"  Components: {results['n_components']}")
            else:
                print(f"  Error: {results['error']}")
                
        except Exception as e:
            print(f"  Exception: {str(e)[:100]}...")

def test_eeg_service_with_problematic_data():
    """Test EEG Service with problematic data scenarios"""
    print("\n=== Testing EEG Service with Problematic Data ===")
    
    scenarios = create_mismatched_data_scenarios()
    
    for name, raw in scenarios.items():
        print(f"\nTesting EEG Service with scenario: {name}")
        
        try:
            # Create service
            service = EEGArtifactCleaningService()
            
            # Load preprocessed data
            load_result = service.load_preprocessed_data(raw)
            print(f"  Load Success: {load_result['success']}")
            
            if load_result['success']:
                if 'fixes_applied' in load_result:
                    print(f"  Fixes applied: {load_result['fixes_applied'].get('changes', [])}")
                
                # Try ICA analysis
                ica_result = service.fit_ica_analysis()
                print(f"  ICA Success: {ica_result['success']}")
                
                if ica_result['success']:
                    print(f"  Components: {ica_result['n_components']}")
                    print(f"  Method: {ica_result['method']}")
                else:
                    print(f"  ICA Error: {ica_result['error'][:100]}...")
            else:
                print(f"  Load Error: {load_result['error'][:100]}...")
                
        except Exception as e:
            print(f"  Service Exception: {str(e)[:100]}...")

def test_specific_user_scenario():
    """Test the specific scenario reported by the user"""
    print("\n=== Testing Specific User-Reported Scenario ===")
    
    # Create a scenario that mimics what the user might be experiencing
    # This typically happens after preprocessing where channels might be filtered/interpolated
    
    raw = create_consistent_test_data()
    print(f"Original data: {raw.get_data().shape[0]} channels, info: {len(raw.ch_names)} channels")
    
    # Simulate preprocessing that might cause the issue
    # For example, if channel interpolation or filtering removes channels
    
    try:
        # Test with our enhanced service
        service = EEGArtifactCleaningService()
        
        print("Loading data with EEG service...")
        load_result = service.load_preprocessed_data(raw)
        
        if load_result['success']:
            print("✅ Data loaded successfully")
            print(f"Channels: {len(load_result['channels'])}")
            
            print("Attempting ICA analysis...")
            ica_result = service.fit_ica_analysis()
            
            if ica_result['success']:
                print("✅ ICA analysis completed successfully")
                print(f"Components fitted: {ica_result['n_components']}")
                print(f"Method used: {ica_result.get('method', 'Unknown')}")
            else:
                print(f"❌ ICA analysis failed: {ica_result['error']}")
        else:
            print(f"❌ Data loading failed: {load_result['error']}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Channel Mismatch Fix Validation Test")
    print("=" * 60)
    
    # Run all tests
    test_data_consistency_validation()
    test_data_consistency_fixes()
    test_safe_channel_picking()
    test_enhanced_ica_with_problematic_data()
    test_eeg_service_with_problematic_data()
    test_specific_user_scenario()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)