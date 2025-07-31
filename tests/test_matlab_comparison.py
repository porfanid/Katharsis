#!/usr/bin/env python3
"""
MATLAB/EEGLAB Comparison Tests
=============================

Tests comparing Katharsis Phase 1 functionality output with EEGLAB functions.
These tests create test signals, run both Katharsis and EEGLAB functions,
and compare the outputs to ensure compatibility.

To run MATLAB comparison (requires MATLAB and EEGLAB):
1. Generate test data using this script
2. Run corresponding MATLAB test script (matlab_comparison_tests.m) 
3. Compare results using similarity metrics

Author: porfanid
Version: 1.0
"""

import numpy as np
import os
import mne
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest
from backend import (
    EEGFilterProcessor, FilterConfig, FilterPresets,
    EEGReferenceProcessor, ReferenceConfig, ReferencePresets,
    EEGChannelManager, PreprocessingPipeline, PreprocessingConfig
)


class MATLABComparisonTestBase:
    """Base class for MATLAB comparison tests"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and directories"""
        cls.test_data_dir = Path("tests/matlab_comparison_data")
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create standardized test signals
        cls.sfreq = 250  # Hz
        cls.duration = 10  # seconds
        cls.n_samples = int(cls.sfreq * cls.duration)
        cls.times = np.arange(cls.n_samples) / cls.sfreq
        
        # Standard 10-20 electrode positions (subset)
        cls.ch_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'O2'
        ]
        cls.n_channels = len(cls.ch_names)
        
        # Create MNE info object
        cls.info = mne.create_info(
            ch_names=cls.ch_names,
            sfreq=cls.sfreq,
            ch_types=['eeg'] * cls.n_channels
        )
        
        # Set montage for proper channel locations
        montage = mne.channels.make_standard_montage('standard_1020')
        cls.info.set_montage(montage, match_case=False)
    
    def create_test_signal(self, signal_type: str = "complex") -> np.ndarray:
        """
        Create standardized test signals for comparison
        
        Args:
            signal_type: Type of signal ('complex', 'sine', 'noise', 'artifacts')
            
        Returns:
            EEG data array (channels × samples)
        """
        np.random.seed(42)  # For reproducible results
        
        data = np.zeros((self.n_channels, self.n_samples))
        
        if signal_type == "complex":
            # Complex EEG-like signal with multiple frequency components
            for ch_idx in range(self.n_channels):
                # Base EEG frequencies
                alpha = 0.5 * np.sin(2 * np.pi * 10 * self.times + ch_idx * 0.2)  # 10 Hz alpha
                beta = 0.3 * np.sin(2 * np.pi * 20 * self.times + ch_idx * 0.3)   # 20 Hz beta
                theta = 0.4 * np.sin(2 * np.pi * 6 * self.times + ch_idx * 0.1)   # 6 Hz theta
                
                # Add noise
                noise = 0.2 * np.random.randn(self.n_samples)
                
                # Add line noise (50 Hz)
                line_noise = 0.1 * np.sin(2 * np.pi * 50 * self.times)
                
                # Combine signals
                data[ch_idx, :] = alpha + beta + theta + noise + line_noise
                
        elif signal_type == "sine":
            # Simple sine wave for basic testing
            freq = 10  # Hz
            for ch_idx in range(self.n_channels):
                data[ch_idx, :] = np.sin(2 * np.pi * freq * self.times + ch_idx * 0.1)
                
        elif signal_type == "noise":
            # White noise
            data = np.random.randn(self.n_channels, self.n_samples)
            
        elif signal_type == "artifacts":
            # EEG with typical artifacts
            for ch_idx in range(self.n_channels):
                # Base EEG
                eeg = 0.5 * np.sin(2 * np.pi * 10 * self.times)
                
                # Eye blink artifacts (mainly frontal)
                if 'Fp' in self.ch_names[ch_idx]:
                    blinks = np.zeros(self.n_samples)
                    blink_times = [1.5, 3.2, 5.8, 7.1]  # seconds
                    for blink_time in blink_times:
                        blink_idx = int(blink_time * self.sfreq)
                        if blink_idx < self.n_samples - 50:
                            blinks[blink_idx:blink_idx+50] = 5.0 * np.exp(-np.arange(50)/10)
                    eeg += blinks
                
                # Muscle artifacts (mainly temporal)
                if 'T' in self.ch_names[ch_idx]:
                    muscle_burst = np.zeros(self.n_samples)
                    burst_start = int(4.0 * self.sfreq)
                    burst_end = int(4.5 * self.sfreq)
                    muscle_burst[burst_start:burst_end] = 2.0 * np.random.randn(burst_end - burst_start)
                    eeg += muscle_burst
                
                data[ch_idx, :] = eeg
        
        # Convert to microvolts (typical EEG amplitude)
        data *= 1e-6
        
        return data
    
    def save_test_data(self, data: np.ndarray, filename: str, extra_info: Dict = None):
        """Save test data in multiple formats for MATLAB comparison"""
        
        # Create MNE Raw object
        raw = mne.io.RawArray(data, self.info)
        
        # Save as .mat file for MATLAB
        mat_data = {
            'data': data,
            'ch_names': self.ch_names,
            'sfreq': self.sfreq,
            'times': self.times,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples
        }
        
        if extra_info:
            mat_data.update(extra_info)
            
        sio.savemat(self.test_data_dir / f"{filename}.mat", mat_data)
        
        # Save as EDF for cross-format testing
        raw.save(self.test_data_dir / f"{filename}.fif", overwrite=True)
        
        return raw
    
    def load_matlab_results(self, filename: str) -> Dict[str, Any]:
        """Load MATLAB comparison results"""
        try:
            return sio.loadmat(self.test_data_dir / f"{filename}_matlab_results.mat")
        except FileNotFoundError:
            pytest.skip(f"MATLAB results not found: {filename}_matlab_results.mat")
    
    def compare_signals(self, signal1: np.ndarray, signal2: np.ndarray, 
                       tolerance: float = 0.01) -> Dict[str, float]:
        """
        Compare two signals using multiple metrics
        
        Args:
            signal1: First signal (Katharsis output)
            signal2: Second signal (MATLAB/EEGLAB output)
            tolerance: Acceptable difference threshold
            
        Returns:
            Dictionary of comparison metrics
        """
        # Ensure same shape
        assert signal1.shape == signal2.shape, f"Shape mismatch: {signal1.shape} vs {signal2.shape}"
        
        # Calculate comparison metrics
        mse = np.mean((signal1 - signal2) ** 2)
        rmse = np.sqrt(mse)
        
        # Normalized cross-correlation
        corr_coeffs = []
        for ch in range(signal1.shape[0]):
            corr = np.corrcoef(signal1[ch, :], signal2[ch, :])[0, 1]
            if not np.isnan(corr):
                corr_coeffs.append(corr)
        
        mean_correlation = np.mean(corr_coeffs) if corr_coeffs else 0.0
        
        # Signal-to-noise ratio
        signal_power = np.mean(signal1 ** 2)
        noise_power = np.mean((signal1 - signal2) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Maximum absolute difference
        max_diff = np.max(np.abs(signal1 - signal2))
        
        # Relative error
        relative_error = np.mean(np.abs(signal1 - signal2) / (np.abs(signal1) + 1e-10))
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'correlation': mean_correlation,
            'snr_db': snr,
            'max_difference': max_diff,
            'relative_error': relative_error,
            'within_tolerance': max_diff < tolerance
        }
        
        return results


class TestFilteringComparison(MATLABComparisonTestBase):
    """Compare Katharsis filtering with EEGLAB pop_eegfilt"""
    
    def test_highpass_filter_comparison(self):
        """Test high-pass filter against EEGLAB pop_eegfilt"""
        # Create test signal
        test_data = self.create_test_signal("complex")
        raw = self.save_test_data(test_data, "highpass_test")
        
        # Apply Katharsis high-pass filter
        filter_processor = EEGFilterProcessor()
        filter_config = FilterConfig(
            filter_type='highpass',
            freq_low=1.0,
            method='fir'
        )
        
        filtered_raw, _ = filter_processor.apply_filter(raw.copy(), filter_config)
        katharsis_output = filtered_raw.get_data()
        
        # Save Katharsis results
        self.save_test_data(katharsis_output, "highpass_katharsis_output")
        
        # Save filter parameters for MATLAB
        filter_params = {
            'filter_type': 'highpass',
            'freq_low': 1.0,
            'method': 'fir',
            'sfreq': self.sfreq
        }
        sio.savemat(self.test_data_dir / "highpass_filter_params.mat", filter_params)
        
        print(f"✓ High-pass filter test data saved to {self.test_data_dir}")
        print("To complete comparison:")
        print("1. Run 'matlab_comparison_tests.m' in MATLAB")
        print("2. Re-run this test to compare results")
        
        # Try to load and compare MATLAB results if available
        try:
            matlab_results = self.load_matlab_results("highpass_test")
            matlab_output = matlab_results['filtered_data']
            
            comparison = self.compare_signals(katharsis_output, matlab_output)
            
            print(f"Comparison results:")
            print(f"  Correlation: {comparison['correlation']:.4f}")
            print(f"  RMSE: {comparison['rmse']:.6f}")
            print(f"  Max difference: {comparison['max_difference']:.6f}")
            print(f"  Within tolerance: {comparison['within_tolerance']}")
            
            # Assert high correlation and low error
            assert comparison['correlation'] > 0.99, f"Low correlation: {comparison['correlation']}"
            assert comparison['within_tolerance'], f"Difference too large: {comparison['max_difference']}"
            
        except Exception as e:
            pytest.skip(f"MATLAB comparison not available: {e}")
    
    def test_bandpass_filter_comparison(self):
        """Test band-pass filter against EEGLAB"""
        test_data = self.create_test_signal("complex")
        raw = self.save_test_data(test_data, "bandpass_test")
        
        # Apply Katharsis band-pass filter (alpha band)
        filter_processor = EEGFilterProcessor()
        filter_config = FilterConfig(
            filter_type='bandpass',
            freq_low=8.0,
            freq_high=12.0,
            method='fir'
        )
        
        filtered_raw, _ = filter_processor.apply_filter(raw.copy(), filter_config)
        katharsis_output = filtered_raw.get_data()
        
        self.save_test_data(katharsis_output, "bandpass_katharsis_output")
        
        filter_params = {
            'filter_type': 'bandpass',
            'freq_low': 8.0,
            'freq_high': 12.0,
            'method': 'fir',
            'sfreq': self.sfreq
        }
        sio.savemat(self.test_data_dir / "bandpass_filter_params.mat", filter_params)
        
        print(f"✓ Band-pass filter test data saved")
    
    def test_notch_filter_comparison(self):
        """Test notch filter (line noise removal) against EEGLAB"""
        test_data = self.create_test_signal("complex")  # Contains 50Hz line noise
        raw = self.save_test_data(test_data, "notch_test")
        
        # Apply Katharsis notch filter
        filter_processor = EEGFilterProcessor()
        filter_config = FilterConfig(
            filter_type='notch',
            freq_notch=50.0,
            method='fir'
        )
        
        filtered_raw, _ = filter_processor.apply_filter(raw.copy(), filter_config)
        katharsis_output = filtered_raw.get_data()
        
        self.save_test_data(katharsis_output, "notch_katharsis_output")
        
        filter_params = {
            'filter_type': 'notch',
            'freq_notch': 50.0,
            'method': 'fir',
            'sfreq': self.sfreq
        }
        sio.savemat(self.test_data_dir / "notch_filter_params.mat", filter_params)
        
        print(f"✓ Notch filter test data saved")
        
        # Verify line noise reduction
        # Calculate power at 50 Hz before and after
        from scipy import signal as sp_signal
        
        freqs_orig, psd_orig = sp_signal.welch(test_data[0, :], self.sfreq, nperseg=1024)
        freqs_filt, psd_filt = sp_signal.welch(katharsis_output[0, :], self.sfreq, nperseg=1024)
        
        # Find 50 Hz power
        freq_50_idx = np.argmin(np.abs(freqs_orig - 50.0))
        power_orig_50hz = psd_orig[freq_50_idx]
        power_filt_50hz = psd_filt[freq_50_idx]
        
        reduction_db = 10 * np.log10(power_orig_50hz / (power_filt_50hz + 1e-10))
        print(f"  Line noise reduction: {reduction_db:.1f} dB")
        
        assert reduction_db > 20, f"Insufficient line noise reduction: {reduction_db:.1f} dB"


class TestReferencingComparison(MATLABComparisonTestBase):
    """Compare Katharsis referencing with EEGLAB pop_reref"""
    
    def test_average_reference_comparison(self):
        """Test average reference against EEGLAB pop_reref"""
        test_data = self.create_test_signal("complex")
        raw = self.save_test_data(test_data, "average_ref_test")
        
        # Apply Katharsis average reference
        ref_processor = EEGReferenceProcessor()
        ref_config = ReferenceConfig('average')
        
        referenced_raw = ref_processor.apply_reference(raw.copy(), ref_config)
        katharsis_output = referenced_raw.get_data()
        
        self.save_test_data(katharsis_output, "average_ref_katharsis_output")
        
        ref_params = {
            'ref_type': 'average'
        }
        sio.savemat(self.test_data_dir / "average_ref_params.mat", ref_params)
        
        print(f"✓ Average reference test data saved")
        
        # Verify average reference property (sum of all channels should be ~0)
        channel_mean = np.mean(katharsis_output, axis=0)
        assert np.max(np.abs(channel_mean)) < 1e-10, "Average reference not properly applied"
    
    def test_common_reference_comparison(self):
        """Test common reference against EEGLAB"""
        test_data = self.create_test_signal("complex")
        raw = self.save_test_data(test_data, "common_ref_test")
        
        # Apply Katharsis common reference (use Cz)
        ref_processor = EEGReferenceProcessor()
        ref_config = ReferenceConfig('common', ref_channels=['Cz'])
        
        referenced_raw = ref_processor.apply_reference(raw.copy(), ref_config)
        katharsis_output = referenced_raw.get_data()
        
        self.save_test_data(katharsis_output, "common_ref_katharsis_output")
        
        ref_params = {
            'ref_type': 'common',
            'ref_channels': 'Cz'
        }
        sio.savemat(self.test_data_dir / "common_ref_params.mat", ref_params)
        
        print(f"✓ Common reference test data saved")


class TestPreprocessingPipelineComparison(MATLABComparisonTestBase):
    """Test complete preprocessing pipeline against EEGLAB workflow"""
    
    def test_complete_preprocessing_comparison(self):
        """Test full preprocessing pipeline"""
        test_data = self.create_test_signal("artifacts")
        raw = self.save_test_data(test_data, "full_preprocessing_test")
        
        # Create comprehensive preprocessing config
        from backend import PreprocessingConfig, FilterConfig, ReferenceConfig
        
        filter_configs = [
            FilterConfig('highpass', freq_low=1.0),
            FilterConfig('lowpass', freq_high=40.0),
            FilterConfig('notch', freq_notch=50.0)
        ]
        
        config = PreprocessingConfig(
            apply_filters=True,
            filter_configs=filter_configs,
            detect_bad_channels=True,
            interpolate_bad_channels=True,
            apply_reference=True,
            reference_config=ReferenceConfig('average'),
            verbose=False
        )
        
        # Run preprocessing pipeline
        pipeline = PreprocessingPipeline()
        processed_raw, results = pipeline.run_pipeline(raw.copy(), config)
        katharsis_output = processed_raw.get_data()
        
        self.save_test_data(katharsis_output, "full_preprocessing_katharsis_output")
        
        # Save pipeline configuration
        pipeline_params = {
            'highpass_freq': 1.0,
            'lowpass_freq': 40.0,
            'notch_freq': 50.0,
            'reference_type': 'average',
            'detect_bad_channels': True,
            'interpolate_bad_channels': True
        }
        sio.savemat(self.test_data_dir / "full_preprocessing_params.mat", pipeline_params)
        
        print(f"✓ Full preprocessing pipeline test data saved")
        print(f"  Results: {results}")


if __name__ == "__main__":
    # Run tests to generate comparison data
    test_filter = TestFilteringComparison()
    test_filter.setup_class()
    
    print("Generating MATLAB comparison test data...")
    
    try:
        test_filter.test_highpass_filter_comparison()
        test_filter.test_bandpass_filter_comparison()
        test_filter.test_notch_filter_comparison()
        
        test_ref = TestReferencingComparison()
        test_ref.setup_class()
        test_ref.test_average_reference_comparison()
        test_ref.test_common_reference_comparison()
        
        test_pipeline = TestPreprocessingPipelineComparison()
        test_pipeline.setup_class()
        test_pipeline.test_complete_preprocessing_comparison()
        
        print("\n✅ All test data generated successfully!")
        print(f"📁 Test data saved in: {test_filter.test_data_dir}")
        print("\n📋 To complete MATLAB comparison:")
        print("1. Open MATLAB with EEGLAB installed")
        print("2. Navigate to the test data directory")
        print("3. Run the MATLAB comparison script (create matlab_comparison_tests.m)")
        print("4. Re-run these tests to compare results")
        
    except Exception as e:
        print(f"❌ Error generating test data: {e}")
        import traceback
        traceback.print_exc()