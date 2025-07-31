#!/usr/bin/env python3
"""
Test Phase 3 Backend Components
===============================

Test script for Phase 3 time-domain analysis backend components
without requiring GUI libraries.
"""

import sys
import numpy as np
import mne

def test_phase3_imports():
    """Test that Phase 3 modules can be imported"""
    print("Testing Phase 3 backend imports...")
    
    try:
        from backend import (
            EpochingProcessor, EpochingConfig, SegmentationConfig,
            ERPAnalyzer, ERPConfig, TimeDomainVisualizer, PlotConfig
        )
        print("✅ Phase 3 imports successful")
        return True
    except Exception as e:
        print(f"❌ Phase 3 import failed: {e}")
        return False

def test_epoching_processor():
    """Test basic epoching functionality"""
    print("\nTesting EpochingProcessor...")
    
    try:
        from backend import EpochingProcessor, EpochingConfig
        
        # Create test data
        info = mne.create_info(ch_names=['EEG1', 'EEG2'], sfreq=250.0, ch_types='eeg')
        data = np.random.randn(2, 1000)  # 2 channels, 1000 time points (4 seconds)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Create processor
        processor = EpochingProcessor()
        
        # Use less strict rejection criteria for test
        config = EpochingConfig(
            tmin=-0.2, 
            tmax=0.8,
            rejection_criteria={}  # No rejection for test
        )
        
        # Create simple events (stimulus at time points)
        events = np.array([[125, 0, 1], [375, 0, 1], [625, 0, 1], [875, 0, 1]])  # 4 events
        
        # Test epoching
        epochs = processor.create_epochs_from_events(raw, events, config)
        
        print(f"✅ Epoching successful: {len(epochs)} epochs created")
        print(f"   - Time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} s")
        print(f"   - Channels: {len(epochs.ch_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Epoching test failed: {e}")
        return False

def test_erp_analyzer():
    """Test basic ERP analysis functionality"""
    print("\nTesting ERPAnalyzer...")
    
    try:
        from backend import ERPAnalyzer, ERPConfig, EpochingProcessor, EpochingConfig
        
        # Create test data (similar to above)
        info = mne.create_info(ch_names=['EEG1', 'EEG2'], sfreq=250.0, ch_types='eeg')
        
        # Create data with simulated ERP (small positive deflection at 100ms)
        times = np.arange(1000) / 250.0  # 4 seconds
        data = np.random.randn(2, 1000) * 10e-6  # 10 µV noise
        
        # Add simulated P1 component at ~100ms after events
        for event_time in [0.5, 1.5, 2.5, 3.5]:  # Events at these times
            start_idx = int((event_time + 0.08) * 250)  # P1 at +80ms
            end_idx = int((event_time + 0.12) * 250)    # P1 ends at +120ms
            if start_idx < 1000 and end_idx < 1000:
                data[0, start_idx:end_idx] += 5e-6  # 5 µV P1 component
        
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Create epochs
        processor = EpochingProcessor()
        config = EpochingConfig(
            tmin=-0.2, 
            tmax=0.8,
            rejection_criteria={}  # No rejection for test
        )
        events = np.array([[125, 0, 1], [375, 0, 1], [625, 0, 1], [875, 0, 1]])
        epochs = processor.create_epochs_from_events(raw, events, config)
        
        # Analyze ERP
        analyzer = ERPAnalyzer()
        erp = analyzer.compute_erp(epochs, "test_condition")
        
        print(f"✅ ERP analysis successful")
        print(f"   - ERP channels: {len(erp.ch_names)}")
        print(f"   - ERP time points: {len(erp.times)}")
        print(f"   - Mean amplitude: {np.mean(erp.data)*1e6:.2f} µV")
        
        # Test peak detection
        peaks = analyzer.detect_peaks(erp)
        print(f"   - Peaks detected: {len(peaks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERP analysis test failed: {e}")
        return False

def test_visualization_config():
    """Test visualization configuration (without actual plotting)"""
    print("\nTesting TimeDomainVisualizer configuration...")
    
    try:
        from backend import TimeDomainVisualizer, PlotConfig
        
        # Test configuration
        config = PlotConfig(
            time_unit='ms',
            color_palette='viridis',
            show_confidence_intervals=True
        )
        
        # Create visualizer (but don't plot anything)
        visualizer = TimeDomainVisualizer(config)
        
        print("✅ Visualization configuration successful")
        print(f"   - Time unit: {config.time_unit}")
        print(f"   - Color palette: {config.color_palette}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Phase 3 Backend Testing ===\n")
    
    tests = [
        test_phase3_imports,
        test_epoching_processor,
        test_erp_analyzer,
        test_visualization_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())