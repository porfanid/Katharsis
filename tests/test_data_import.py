#!/usr/bin/env python3
"""
Test cases for multi-format EEG data import functionality
"""

import pytest
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import tempfile
import os

from backend.eeg_backend import EEGDataManager


class TestDataImport:
    """Test class for multi-format data import functionality"""
    
    @pytest.fixture
    def sample_edf_path(self):
        """Path to sample EDF file"""
        return '/home/runner/work/Katharsis/Katharsis/data.edf'
    
    @pytest.fixture
    def test_csv_file(self):
        """Create a temporary CSV file for testing"""
        # Generate synthetic EEG data
        sfreq = 256
        n_channels = 5
        duration = 2  # seconds
        n_times = int(sfreq * duration)
        
        ch_names = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        
        # Generate some synthetic EEG-like data
        np.random.seed(42)
        data = np.random.randn(n_channels, n_times) * 10  # µV scale
        
        # Create DataFrame and save as temporary CSV
        df = pd.DataFrame(data.T, columns=ch_names)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        os.unlink(f.name)
    
    def test_format_detection(self):
        """Test file format detection"""
        test_cases = [
            ('test.edf', 'edf'),
            ('test.bdf', 'bdf'),
            ('test.fif', 'fif'),
            ('test.csv', 'csv'),
            ('test.set', 'set'),
            ('TEST.EDF', 'edf'),  # Case insensitive
            ('data.FIF', 'fif'),
        ]
        
        for filename, expected_format in test_cases:
            result = EEGDataManager.detect_file_format(filename)
            assert result == expected_format, f"Expected {expected_format} for {filename}, got {result}"
    
    def test_unsupported_format(self):
        """Test error handling for unsupported formats"""
        with pytest.raises(ValueError, match="Μη υποστηριζόμενος τύπος αρχείου"):
            EEGDataManager.detect_file_format('test.txt')
    
    def test_edf_file_info(self, sample_edf_path):
        """Test EDF file info extraction"""
        if not Path(sample_edf_path).exists():
            pytest.skip("Sample EDF file not found")
            
        info = EEGDataManager.get_file_info(sample_edf_path)
        
        assert info['success'] == True
        assert info['format'] == 'edf'
        assert 'channels' in info
        assert 'sampling_rate' in info
        assert 'n_channels' in info
        assert 'n_times' in info
        assert 'detected_eeg' in info
        assert info['n_channels'] > 0
        assert info['n_times'] > 0
        assert len(info['detected_eeg']) > 0
    
    def test_edf_data_loading(self, sample_edf_path):
        """Test EDF data loading"""
        if not Path(sample_edf_path).exists():
            pytest.skip("Sample EDF file not found")
            
        raw, channels = EEGDataManager.load_raw_file(sample_edf_path)
        
        assert isinstance(raw, mne.io.BaseRaw)
        assert raw.info['nchan'] > 0
        assert raw.n_times > 0
        assert len(channels) >= 3  # Minimum required channels
        assert all(ch in raw.ch_names for ch in channels)
    
    def test_csv_file_info(self, test_csv_file):
        """Test CSV file info extraction"""
        info = EEGDataManager.get_file_info(test_csv_file)
        
        assert info['success'] == True
        assert info['format'] == 'csv'
        assert info['n_channels'] == 5
        assert info['sampling_rate'] == 256.0
        assert len(info['detected_eeg']) == 5
        assert set(info['channels']) == {'AF3', 'T7', 'Pz', 'T8', 'AF4'}
    
    def test_csv_data_loading(self, test_csv_file):
        """Test CSV data loading"""
        raw, channels = EEGDataManager.load_raw_file(test_csv_file)
        
        assert isinstance(raw, mne.io.BaseRaw)
        assert raw.info['nchan'] == 5
        assert raw.info['sfreq'] == 256.0
        assert len(channels) == 5
        assert set(channels) == {'AF3', 'T7', 'Pz', 'T8', 'AF4'}
    
    def test_nonexistent_file(self):
        """Test error handling for nonexistent files"""
        info = EEGDataManager.get_file_info('/nonexistent/file.edf')
        assert info['success'] == False
        assert 'error' in info
        
        with pytest.raises(FileNotFoundError):
            EEGDataManager.load_raw_file('/nonexistent/file.edf')
    
    def test_channel_selection(self, sample_edf_path):
        """Test custom channel selection"""
        if not Path(sample_edf_path).exists():
            pytest.skip("Sample EDF file not found")
            
        # First get available channels
        info = EEGDataManager.get_file_info(sample_edf_path)
        available_eeg = info['detected_eeg']
        
        if len(available_eeg) >= 3:
            # Select subset of channels
            selected = available_eeg[:3]
            raw, channels = EEGDataManager.load_raw_file(sample_edf_path, selected)
            
            assert len(channels) == 3
            assert set(channels) == set(selected)
            assert raw.info['nchan'] == 3
    
    def test_insufficient_channels(self, test_csv_file):
        """Test error handling for insufficient channels"""
        # Try to select too few channels
        with pytest.raises(ValueError, match="Χρειάζονται τουλάχιστον 3 κανάλια"):
            EEGDataManager.load_raw_file(test_csv_file, ['AF3', 'T7'])
    
    def test_invalid_channel_selection(self, test_csv_file):
        """Test error handling for invalid channel selection"""
        with pytest.raises(ValueError, match="δεν υπάρχει στο αρχείο"):
            EEGDataManager.load_raw_file(test_csv_file, ['INVALID', 'AF3', 'T7'])
    
    def test_backward_compatibility(self, sample_edf_path):
        """Test that old EDF-specific methods still work"""
        if not Path(sample_edf_path).exists():
            pytest.skip("Sample EDF file not found")
            
        # Test old method
        old_info = EEGDataManager.load_edf_file_info(sample_edf_path)
        
        # Test new method
        new_info = EEGDataManager.get_file_info(sample_edf_path)
        
        # Compare essential fields
        assert old_info['success'] == new_info['success']
        assert old_info['channels'] == new_info['channels']
        assert old_info['sampling_rate'] == new_info['sampling_rate']
        assert old_info['detected_eeg'] == new_info['detected_eeg']
    

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])