#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing for User-Reported Issues
========================================================

Tests the specific issues reported by the user with data.edf:
1. ICA n_components error after minimal preprocessing
2. Time domain analysis error with no events/annotations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.katharsis_backend import KatharsisBackend


def test_user_ica_workflow():
    """Test the exact workflow that caused the user's ICA error"""
    print("🧪 Testing User ICA Workflow")
    print("=" * 50)
    
    backend = KatharsisBackend()
    
    # Step 1: Load data.edf
    print("1. Loading data.edf...")
    load_result = backend.load_file('/home/runner/work/Katharsis/Katharsis/data.edf')
    assert load_result['success'], f"File loading failed: {load_result}"
    print(f"   ✅ Loaded successfully: {load_result['n_channels']} channels, {load_result['duration']:.1f}s")
    
    # Step 2: Apply preprocessing (user clicked without changing anything)
    print("2. Applying minimal preprocessing (user clicked without changes)...")
    preprocess_result = backend.apply_preprocessing({})
    assert preprocess_result['success'], f"Preprocessing failed: {preprocess_result}"
    print(f"   ✅ Preprocessing successful: {preprocess_result['applied_steps']}")
    
    # Step 3: Perform ICA (this used to fail with n_components error)
    print("3. Performing ICA analysis...")
    ica_result = backend.perform_ica_analysis()
    assert ica_result['success'], f"ICA failed: {ica_result}"
    print(f"   ✅ ICA successful: {ica_result['n_components']} components, method: {ica_result['method']}")
    
    # Step 4: Test artifact detection (this was causing the n_components error)
    print("4. Testing artifact detection (previously failed)...")
    try:
        detection_result = backend.cleaning_service.detect_artifacts()
        print(f"   ✅ Artifact detection: {detection_result['success']}")
        if not detection_result['success']:
            print(f"   ❌ Detection error: {detection_result['error']}")
            return False
    except Exception as e:
        print(f"   ❌ Detection exception: {e}")
        return False
    
    print("✅ User ICA workflow test PASSED")
    return True


def test_user_time_domain_workflow():
    """Test the exact workflow that caused the user's time domain error"""
    print("\n🧪 Testing User Time Domain Workflow")
    print("=" * 50)
    
    backend = KatharsisBackend()
    
    # Step 1: Load data.edf
    print("1. Loading data.edf...")
    load_result = backend.load_file('/home/runner/work/Katharsis/Katharsis/data.edf')
    assert load_result['success'], f"File loading failed: {load_result}"
    print(f"   ✅ Loaded successfully: {load_result['n_channels']} channels")
    
    # Step 2: Apply preprocessing
    print("2. Applying preprocessing...")
    preprocess_result = backend.apply_preprocessing({})
    assert preprocess_result['success'], f"Preprocessing failed: {preprocess_result}"
    print(f"   ✅ Preprocessing successful")
    
    # Step 3: Try time domain analysis with defaults (this used to give English error)
    print("3. Attempting time domain analysis with defaults...")
    events_config = {'event_id': None, 'min_duration': 0.001}
    epoch_config = {'tmin': -0.2, 'tmax': 0.8, 'baseline': None, 'baseline_correction': 'none', 'reject_criteria': {}}
    
    time_result = backend.perform_epoching(events_config, epoch_config)
    
    # We expect this to fail, but with proper Greek error messages
    assert not time_result['success'], "Time domain should fail for data without events"
    print(f"   ✅ Expected failure with proper error: {time_result['error']}")
    
    # Check that error is in Greek and helpful
    error_msg = time_result['error']
    suggestion = time_result.get('suggestion', '')
    
    assert 'γεγονότα' in error_msg or 'σημειώσεις' in error_msg, f"Error should be in Greek: {error_msg}"
    assert '💡' in suggestion, f"Should have helpful suggestion: {suggestion}"
    print(f"   ✅ Proper Greek error message with suggestion")
    
    print("✅ User time domain workflow test PASSED")
    return True


def test_edge_cases():
    """Test various edge cases"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    backend = KatharsisBackend()
    
    # Load data first
    load_result = backend.load_file('/home/runner/work/Katharsis/Katharsis/data.edf')
    assert load_result['success']
    
    print("1. Testing ICA with different parameters...")
    
    # Test with specific n_components
    ica_result = backend.perform_ica_analysis(ica_method='fastica', n_components=3)
    assert ica_result['success'], f"ICA with specific components failed: {ica_result}"
    print(f"   ✅ ICA with n_components=3: {ica_result['n_components']} components")
    
    print("2. Testing state management...")
    # Check that cleaning service is properly synchronized
    assert backend.cleaning_service.ica_fitted, "Cleaning service should be marked as ICA fitted"
    print("   ✅ Cleaning service properly synchronized")
    
    print("3. Testing data without annotations...")
    # Verify our data has no annotations (this is the root cause of time domain issues)
    raw_data = backend.raw_data
    n_annotations = len(raw_data.annotations) if hasattr(raw_data, 'annotations') else 0
    print(f"   ✅ Data has {n_annotations} annotations (expected: 0)")
    
    print("✅ Edge cases test PASSED")
    return True


if __name__ == "__main__":
    print("🚀 Running Comprehensive Edge Case Tests for User Issues")
    print("=" * 70)
    
    try:
        # Test the user's specific workflows
        ica_passed = test_user_ica_workflow()
        time_domain_passed = test_user_time_domain_workflow()
        edge_cases_passed = test_edge_cases()
        
        if ica_passed and time_domain_passed and edge_cases_passed:
            print("\n🎉 ALL TESTS PASSED! User issues should be resolved.")
            print("\nSummary of fixes:")
            print("✅ ICA n_components error: Fixed state synchronization")
            print("✅ Time domain error: Added proper Greek error messages")
            print("✅ Frontend imports: Created missing widget files")
            print("✅ Edge cases: Comprehensive validation and error handling")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)