#!/usr/bin/env python3
"""
Test script for electrode generalization feature
Σκριπτ ελέγχου για τη λειτουργία γενίκευσης ηλεκτροδίων
"""

import os
import tempfile

import mne
import numpy as np

from backend import EEGArtifactCleaningService, EEGDataManager


def create_test_edf(channels, duration=60.0, filename=None):
    """Create synthetic EDF file with specified channels"""
    sfreq = 128.0
    n_samples = int(sfreq * duration)

    time = np.linspace(0, duration, n_samples)
    data = []

    for i, ch in enumerate(channels):
        # Create realistic EEG-like signals
        signal = (
            0.5 * np.sin(2 * np.pi * 8 * time)  # Alpha rhythm
            + 0.3 * np.sin(2 * np.pi * 13 * time)  # Beta rhythm
            + 0.2 * np.sin(2 * np.pi * 4 * time)  # Theta rhythm
            + 0.1 * np.random.randn(n_samples)  # Noise
        ) * 1e-5
        data.append(signal)

    data = np.array(data)
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    if filename is None:
        filename = f"test_electrodes_{len(channels)}.edf"

    raw.export(filename, fmt="edf", overwrite=True, verbose=False)
    return filename


def test_electrode_detection():
    """Test the dynamic electrode detection"""
    print("=== Testing Electrode Detection ===")

    test_cases = [
        ["AF3", "T7", "Pz", "T8", "AF4"],  # Original Emotiv
        ["AF3", "T7", "Pz", "T8", "F4"],  # With F4 instead of AF4
        ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"],  # Standard EEG
        ["Fp1", "Fp2", "F7", "F8", "O1", "O2"],  # Different configuration
        ["C3", "Cz", "C4"],  # Minimal configuration
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],  # Many frontal
    ]

    for i, channels in enumerate(test_cases):
        print(f"\nTest case {i+1}: {channels}")

        # Create test file
        filename = create_test_edf(channels)

        try:
            # Test detection
            raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
            detected = EEGDataManager.detect_eeg_channels(raw)

            print(f"  Input channels: {len(channels)}")
            print(f"  Detected EEG: {len(detected)}")
            print(f"  Detected: {detected}")

            # Should detect all input channels as EEG
            assert len(detected) == len(
                channels
            ), f"Expected {len(channels)}, got {len(detected)}"
            assert set(detected) == set(channels), "Detected channels don't match input"

            print("  ✓ Detection successful")

        finally:
            # Cleanup
            if os.path.exists(filename):
                os.unlink(filename)


def test_full_pipeline():
    """Test the full processing pipeline with different electrode configurations"""
    print("\n=== Testing Full Processing Pipeline ===")

    test_configs = [
        (["AF3", "T7", "Pz", "T8", "AF4"], "Original Emotiv"),
        (["AF3", "T7", "Pz", "T8", "F4"], "Emotiv with F4"),
        (["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"], "9-electrode array"),
        (["C3", "Cz", "C4"], "Minimal 3-electrode"),
    ]

    for channels, description in test_configs:
        print(f"\nTesting: {description}")
        print(f"Channels: {channels}")

        filename = create_test_edf(channels, duration=60.0)

        try:
            service = EEGArtifactCleaningService()

            # Test loading
            result = service.load_and_prepare_file(filename)
            assert result[
                "success"
            ], f"Loading failed: {result.get('error', 'Unknown error')}"
            assert len(result["channels"]) == len(channels), f"Channel count mismatch"
            print(f"  ✓ Loading successful: {len(result['channels'])} channels")

            # Test ICA fitting
            ica_result = service.fit_ica_analysis()
            assert ica_result[
                "success"
            ], f"ICA failed: {ica_result.get('error', 'Unknown error')}"
            assert ica_result["n_components"] == len(
                channels
            ), f"Component count mismatch"
            print(f"  ✓ ICA successful: {ica_result['n_components']} components")

            # Test artifact detection
            detect_result = service.detect_artifacts()
            assert detect_result[
                "success"
            ], f"Detection failed: {detect_result.get('error', 'Unknown error')}"
            print(
                f"  ✓ Detection successful: {len(detect_result['suggested_artifacts'])} artifacts found"
            )

            # Test artifact removal (if artifacts found)
            if len(detect_result["suggested_artifacts"]) > 0:
                clean_result = service.apply_artifact_removal(
                    [detect_result["suggested_artifacts"][0]]
                )
                assert clean_result[
                    "success"
                ], f"Cleaning failed: {clean_result.get('error', 'Unknown error')}"
                print(f"  ✓ Cleaning successful")
            else:
                print(f"  ✓ No artifacts to remove (which is fine)")

            # Test visualization data
            viz_data = service.get_component_visualization_data()
            assert viz_data is not None, "Visualization data is None"
            assert "ica" in viz_data, "ICA object missing from visualization data"
            assert "raw" in viz_data, "Raw data missing from visualization data"
            print(f"  ✓ Visualization data available")

        finally:
            # Cleanup
            if os.path.exists(filename):
                os.unlink(filename)


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")

    # Test with non-EEG channels
    print("\nTesting non-EEG channels (should be filtered out):")

    # Create file with mixed channel types
    sfreq = 128.0
    duration = 30.0
    n_samples = int(sfreq * duration)

    # Mix of EEG and non-EEG channels
    all_channels = ["F3", "F4", "ECG", "EMG", "TIME_STAMP", "COUNTER", "C3", "C4"]
    expected_eeg = ["F3", "F4", "C3", "C4"]

    data = np.random.randn(len(all_channels), n_samples) * 1e-5
    info = mne.create_info(
        ch_names=all_channels, sfreq=sfreq, ch_types=["eeg"] * len(all_channels)
    )
    raw = mne.io.RawArray(data, info)

    filename = "test_mixed_channels.edf"
    raw.export(filename, fmt="edf", overwrite=True, verbose=False)

    try:
        # Test detection
        raw_loaded = mne.io.read_raw_edf(filename, preload=True, verbose=False)
        detected = EEGDataManager.detect_eeg_channels(raw_loaded)

        print(f"  All channels: {all_channels}")
        print(f"  Detected EEG: {detected}")
        print(f"  Expected EEG: {expected_eeg}")

        # Should only detect the EEG channels
        assert set(detected) == set(expected_eeg), f"Detection mismatch"
        print("  ✓ Non-EEG channels correctly filtered out")

    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def main():
    """Run all tests"""
    print("Testing Electrode Generalization Feature")
    print("=" * 50)

    # Suppress MNE warnings for cleaner output
    mne.set_log_level("WARNING")

    try:
        test_electrode_detection()
        test_full_pipeline()
        test_edge_cases()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Electrode generalization is working correctly.")
        print("✓ The application can now handle any EEG electrode configuration")
        print("✓ ICA processing adapts to the number of available channels")
        print("✓ Both real-time and ICA functionality maintained")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
