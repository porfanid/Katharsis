#!/usr/bin/env python3
"""
Tests for Phase 1 Advanced Preprocessing Features
================================================

Comprehensive tests for the new preprocessing capabilities:
- Advanced filtering
- Re-referencing
- Channel management
- Preprocessing pipeline

Author: porfanid
Version: 1.0
"""

import pytest
import numpy as np
import mne
from pathlib import Path
import tempfile

from backend.filters import (
    EEGFilterProcessor, FilterConfig, FilterPresets
)
from backend.referencing import (
    EEGReferenceProcessor, ReferenceConfig, ReferencePresets
)
from backend.channel_management import (
    EEGChannelManager, BadChannelDetector, ChannelInterpolator
)
from backend.preprocessing_pipeline import (
    PreprocessingPipeline, PreprocessingConfig, PreprocessingPresets
)


class TestFilterProcessor:
    """Tests for EEG filtering functionality"""
    
    @pytest.fixture
    def sample_raw(self):
        """Create sample raw data for testing"""
        info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'],
            sfreq=256.0,
            ch_types='eeg'
        )
        # Create sample data with some noise
        n_samples = 256 * 10  # 10 seconds
        data = np.random.randn(6, n_samples) * 1e-6
        # Add some 50Hz noise
        times = np.arange(n_samples) / 256.0
        noise_50hz = np.sin(2 * np.pi * 50 * times) * 5e-6
        data += noise_50hz
        
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    
    def test_filter_config_validation(self):
        """Test filter configuration validation"""
        # Valid configurations
        config = FilterConfig('highpass', freq_low=1.0)
        assert config.filter_type == 'highpass'
        assert config.freq_low == 1.0
        
        # Invalid filter type
        with pytest.raises(ValueError):
            FilterConfig('invalid_type')
        
        # Missing frequency for highpass
        with pytest.raises(ValueError):
            FilterConfig('highpass')
        
        # Invalid frequency range for bandpass
        with pytest.raises(ValueError):
            FilterConfig('bandpass', freq_low=10.0, freq_high=5.0)
    
    def test_highpass_filter(self, sample_raw):
        """Test high-pass filtering"""
        processor = EEGFilterProcessor()
        config = FilterConfig('highpass', freq_low=1.0)
        
        filtered_raw, filter_info = processor.apply_filter(sample_raw, config)
        
        assert filter_info['success']
        assert filter_info['type'] == 'highpass'
        assert filtered_raw.info['sfreq'] == sample_raw.info['sfreq']
        assert len(filtered_raw.ch_names) == len(sample_raw.ch_names)
    
    def test_lowpass_filter(self, sample_raw):
        """Test low-pass filtering"""
        processor = EEGFilterProcessor()
        config = FilterConfig('lowpass', freq_low=40.0)
        
        filtered_raw, filter_info = processor.apply_filter(sample_raw, config)
        
        assert filter_info['success']
        assert filter_info['type'] == 'lowpass'
    
    def test_bandpass_filter(self, sample_raw):
        """Test band-pass filtering"""
        processor = EEGFilterProcessor()
        config = FilterConfig('bandpass', freq_low=1.0, freq_high=40.0)
        
        filtered_raw, filter_info = processor.apply_filter(sample_raw, config)
        
        assert filter_info['success']
        assert filter_info['type'] == 'bandpass'
    
    def test_notch_filter(self, sample_raw):
        """Test notch filtering"""
        processor = EEGFilterProcessor()
        config = FilterConfig('notch', freq_notch=50.0)
        
        filtered_raw, filter_info = processor.apply_filter(sample_raw, config)
        
        assert filter_info['success']
        assert filter_info['type'] == 'notch'
    
    def test_filter_presets(self):
        """Test filter presets"""
        preprocessing = FilterPresets.get_preprocessing_preset()
        assert len(preprocessing) >= 2  # Should have at least high-pass and low-pass
        
        erp = FilterPresets.get_erp_preset()
        assert len(erp) >= 2
        
        alpha = FilterPresets.get_alpha_analysis_preset()
        assert any(f.filter_type == 'bandpass' for f in alpha)
    
    def test_filter_history(self, sample_raw):
        """Test filter history tracking"""
        processor = EEGFilterProcessor()
        
        # Apply multiple filters
        config1 = FilterConfig('highpass', freq_low=1.0)
        config2 = FilterConfig('lowpass', freq_low=40.0)
        
        processor.apply_filter(sample_raw, config1)
        processor.apply_filter(sample_raw, config2)
        
        history = processor.get_filter_history()
        assert len(history) == 2
        assert history[0]['type'] == 'highpass'
        assert history[1]['type'] == 'lowpass'


class TestReferenceProcessor:
    """Tests for EEG re-referencing functionality"""
    
    @pytest.fixture
    def sample_raw_with_ref(self):
        """Create sample raw data with reference channels"""
        info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2', 'A1', 'A2'],
            sfreq=256.0,
            ch_types='eeg'
        )
        n_samples = 256 * 5  # 5 seconds
        data = np.random.randn(8, n_samples) * 1e-6
        
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    
    def test_reference_config_validation(self):
        """Test reference configuration validation"""
        # Valid configurations
        config = ReferenceConfig('average')
        assert config.ref_type == 'average'
        
        config = ReferenceConfig('common', ref_channels='Cz')
        assert config.ref_type == 'common'
        assert config.ref_channels == 'Cz'
        
        # Invalid reference type
        with pytest.raises(ValueError):
            ReferenceConfig('invalid_type')
        
        # Missing reference channels for common reference
        with pytest.raises(ValueError):
            ReferenceConfig('common')
    
    def test_average_reference(self, sample_raw_with_ref):
        """Test average reference"""
        processor = EEGReferenceProcessor()
        config = ReferenceConfig('average')
        
        ref_raw, ref_info = processor.apply_reference(sample_raw_with_ref, config)
        
        assert ref_info['success']
        assert ref_info['type'] == 'average'
    
    def test_common_reference(self, sample_raw_with_ref):
        """Test common reference"""
        processor = EEGReferenceProcessor()
        config = ReferenceConfig('common', ref_channels='C3')
        
        ref_raw, ref_info = processor.apply_reference(sample_raw_with_ref, config)
        
        assert ref_info['success']
        assert ref_info['type'] == 'common'
        assert ref_info['ref_channels'] == 'C3'
    
    def test_linked_ears_reference(self, sample_raw_with_ref):
        """Test linked ears reference"""
        processor = EEGReferenceProcessor()
        config = ReferenceConfig('linked_ears')
        
        ref_raw, ref_info = processor.apply_reference(sample_raw_with_ref, config)
        
        assert ref_info['success']
        assert ref_info['type'] == 'linked_ears'
    
    def test_bipolar_reference(self, sample_raw_with_ref):
        """Test bipolar reference"""
        processor = EEGReferenceProcessor()
        config = ReferenceConfig('bipolar', ref_channels=['Fp1', 'Fp2', 'C3', 'C4'])
        
        ref_raw, ref_info = processor.apply_reference(sample_raw_with_ref, config)
        
        assert ref_info['success']
        assert ref_info['type'] == 'bipolar'
        # Should have 2 bipolar channels (Fp1-Fp2, C3-C4)
        assert len(ref_raw.ch_names) == 2
    
    def test_reference_presets(self):
        """Test reference presets"""
        clinical = ReferencePresets.get_clinical_preset()
        assert clinical.ref_type == 'average'
        
        research = ReferencePresets.get_research_preset()
        assert research.ref_type == 'average'
        
        sleep = ReferencePresets.get_sleep_preset()
        assert sleep.ref_type == 'linked_ears'


class TestChannelManagement:
    """Tests for channel management functionality"""
    
    @pytest.fixture
    def sample_raw_with_bad_channels(self):
        """Create sample raw data with simulated bad channels"""
        info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'],
            sfreq=256.0,
            ch_types='eeg'
        )
        n_samples = 256 * 10  # 10 seconds
        data = np.random.randn(6, n_samples) * 1e-6
        
        # Make Fp2 a flat channel
        data[1, :] = 0.0
        
        # Make C4 a noisy channel
        data[3, :] *= 10
        
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    
    def test_bad_channel_detection(self, sample_raw_with_bad_channels):
        """Test automatic bad channel detection"""
        detector = BadChannelDetector()
        
        bad_channels = detector.detect_bad_channels(sample_raw_with_bad_channels)
        
        assert 'flat' in bad_channels
        assert 'noisy' in bad_channels
        assert 'uncorrelated' in bad_channels
        assert 'high_freq_noise' in bad_channels
        
        # Should detect Fp2 as flat
        assert 'Fp2' in bad_channels['flat']
        
        # Should detect C4 as noisy
        assert 'C4' in bad_channels['noisy']
    
    def test_channel_interpolation(self, sample_raw_with_bad_channels):
        """Test channel interpolation"""
        # Add montage for interpolation
        montage = mne.channels.make_standard_montage('standard_1020')
        sample_raw_with_bad_channels.set_montage(montage, match_case=False, verbose=False)
        
        interpolator = ChannelInterpolator()
        bad_channels = ['Fp2']
        
        interp_raw, interp_info = interpolator.interpolate_channels(
            sample_raw_with_bad_channels, bad_channels
        )
        
        assert interp_info['success']
        assert interp_info['n_interpolated'] == 1
        assert 'Fp2' in interp_info['interpolated_channels']
    
    def test_channel_manager_analysis(self, sample_raw_with_bad_channels):
        """Test comprehensive channel analysis"""
        manager = EEGChannelManager()
        
        analysis = manager.analyze_channels(sample_raw_with_bad_channels)
        
        assert 'n_channels' in analysis
        assert 'channel_types' in analysis
        assert 'bad_channels' in analysis
        assert 'groups' in analysis
        
        assert analysis['n_channels'] == 6
        assert analysis['channel_types']['eeg'] == 6
    
    def test_channel_recommendations(self, sample_raw_with_bad_channels):
        """Test channel processing recommendations"""
        manager = EEGChannelManager()
        
        recommendations = manager.get_channel_recommendations(sample_raw_with_bad_channels)
        
        assert 'bad_channels_to_remove' in recommendations
        assert 'bad_channels_to_interpolate' in recommendations
        assert 'montage_suggestions' in recommendations


class TestPreprocessingPipeline:
    """Tests for complete preprocessing pipeline"""
    
    @pytest.fixture
    def sample_raw_full(self):
        """Create comprehensive sample raw data"""
        info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'],
            sfreq=256.0,
            ch_types='eeg'
        )
        n_samples = 256 * 30  # 30 seconds
        data = np.random.randn(10, n_samples) * 1e-6
        
        # Add some realistic EEG features
        times = np.arange(n_samples) / 256.0
        
        # Add alpha rhythm (10 Hz) to posterior channels
        alpha = np.sin(2 * np.pi * 10 * times) * 2e-6
        data[6:, :] += alpha  # P3, P4, O1, O2
        
        # Add line noise
        line_noise = np.sin(2 * np.pi * 50 * times) * 1e-6
        data += line_noise
        
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    
    def test_preprocessing_config(self):
        """Test preprocessing configuration"""
        config = PreprocessingConfig()
        
        assert config.apply_filters is True
        assert config.detect_bad_channels is True
        assert config.interpolate_bad_channels is True
        assert config.load_montage is True
        assert config.apply_reference is True
    
    def test_complete_pipeline(self, sample_raw_full):
        """Test complete preprocessing pipeline"""
        pipeline = PreprocessingPipeline()
        config = PreprocessingPresets.get_research_preset()
        
        # Reduce verbosity for testing
        config.verbose = False
        
        processed_raw, results = pipeline.run_pipeline(sample_raw_full, config)
        
        assert results['success']
        assert len(results['steps_completed']) > 0
        assert 'channel_analysis' in results['steps_completed']
        
        # Check that data was processed
        assert processed_raw.info['sfreq'] == sample_raw_full.info['sfreq']
        assert len(processed_raw.ch_names) == len(sample_raw_full.ch_names)
    
    def test_custom_pipeline_creation(self, sample_raw_full):
        """Test custom pipeline configuration creation"""
        pipeline = PreprocessingPipeline()
        
        steps = ['channel_analysis', 'filtering', 'referencing']
        config = pipeline.create_custom_pipeline(sample_raw_full, steps)
        
        assert config.apply_filters is True
        assert config.apply_reference is True
        # Should disable steps not in the list
        assert config.detect_bad_channels is False
    
    def test_pipeline_presets(self):
        """Test preprocessing presets"""
        clinical = PreprocessingPresets.get_clinical_preset()
        assert clinical.apply_filters is True
        assert clinical.reference_config.ref_type == 'average'
        
        research = PreprocessingPresets.get_research_preset()
        assert research.apply_filters is True
        
        erp = PreprocessingPresets.get_erp_preset()
        assert erp.apply_filters is True
        
        minimal = PreprocessingPresets.get_minimal_preset()
        assert minimal.apply_filters is True
        assert minimal.detect_bad_channels is False
    
    def test_pipeline_summary(self, sample_raw_full):
        """Test pipeline summary generation"""
        pipeline = PreprocessingPipeline()
        config = PreprocessingPresets.get_minimal_preset()
        config.verbose = False
        
        processed_raw, results = pipeline.run_pipeline(sample_raw_full, config)
        summary = pipeline.get_pipeline_summary(results)
        
        assert "Pipeline Summary" in summary
        assert "completed successfully" in summary or "failed" in summary
        assert "Steps completed:" in summary


class TestIntegration:
    """Integration tests for all Phase 1 features"""
    
    @pytest.fixture
    def real_eeg_file(self):
        """Use the sample EDF file if available"""
        edf_file = Path(__file__).parent.parent / "data.edf"
        if edf_file.exists():
            return str(edf_file)
        else:
            pytest.skip("Sample EDF file not available")
    
    def test_full_integration_with_real_data(self, real_eeg_file):
        """Test complete integration with real EEG data"""
        # Load real data
        raw = mne.io.read_raw_edf(real_eeg_file, verbose=False)
        
        # Create and run preprocessing pipeline
        pipeline = PreprocessingPipeline()
        config = PreprocessingPresets.get_research_preset()
        config.verbose = False
        
        # Disable montage loading if channels don't match standard montages
        config.load_montage = False
        
        processed_raw, results = pipeline.run_pipeline(raw, config)
        
        assert results['success']
        assert processed_raw is not None
        
        # Verify data integrity
        assert processed_raw.info['sfreq'] == raw.info['sfreq']
        assert processed_raw.n_times == raw.n_times
    
    def test_filter_chain_integration(self, real_eeg_file):
        """Test applying multiple filters in sequence"""
        raw = mne.io.read_raw_edf(real_eeg_file, verbose=False)
        
        processor = EEGFilterProcessor()
        
        # Apply filter chain
        filters = FilterPresets.get_preprocessing_preset()
        filtered_raw = raw.copy()
        
        for filter_config in filters:
            filtered_raw, filter_info = processor.apply_filter(
                filtered_raw, filter_config, copy=False
            )
            assert filter_info['success']
        
        # Verify final result
        assert filtered_raw.info['sfreq'] == raw.info['sfreq']
        assert len(filtered_raw.ch_names) == len(raw.ch_names)
    
    def test_backwards_compatibility(self, real_eeg_file):
        """Test that new features don't break existing functionality"""
        # Test that existing EEGArtifactCleaningService still works
        from backend import EEGArtifactCleaningService
        
        service = EEGArtifactCleaningService()
        
        # Load file using existing method
        load_result = service.load_and_prepare_file(real_eeg_file)
        
        assert load_result['success']
        assert service.backend_core.raw_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])