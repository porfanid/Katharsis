#!/usr/bin/env python3
"""
Backend Core - EEG Data Processing Core Logic
Κεντρικό Backend - Κύρια Λογική Επεξεργασίας EEG Δεδομένων
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy import stats
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')


class EEGDataManager:
    """Διαχείριση δεδομένων EEG - φόρτωση, αποθήκευση, επικύρωση"""
    
    @staticmethod
    def load_edf_file(file_path: str) -> Tuple[mne.io.Raw, List[str]]:
        """
        Φόρτωση EDF αρχείου και εξαγωγή EEG καναλιών
        
        Args:
            file_path: Διαδρομή του EDF αρχείου
            
        Returns:
            Tuple[mne.io.Raw, List[str]]: Raw δεδομένα και λίστα EEG καναλιών
            
        Raises:
            FileNotFoundError: Εάν το αρχείο δεν βρεθεί
            ValueError: Εάν το αρχείο δεν είναι έγκυρο EDF
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε")
            
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            raise ValueError(f"Σφάλμα φόρτωσης EDF αρχείου: {str(e)}")
        
        # Εξαγωγή EEG καναλιών
        eeg_channels = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        available_channels = [ch for ch in eeg_channels if ch in raw.ch_names]
        
        if not available_channels:
            raise ValueError("Δεν βρέθηκαν έγκυρα EEG κανάλια στο αρχείο")
        
        # Κρατάμε μόνο τα EEG κανάλια
        raw.pick_channels(available_channels)
        
        # Ορισμός montage για τοπογραφική απεικόνιση
        raw.set_montage('standard_1020', on_missing='warn')
        
        return raw, available_channels
    
    @staticmethod
    def save_cleaned_data(raw: mne.io.Raw, output_path: str) -> bool:
        """
        Αποθήκευση καθαρισμένων δεδομένων σε EDF format
        
        Args:
            raw: Καθαρισμένα Raw δεδομένα
            output_path: Διαδρομή εξόδου
            
        Returns:
            bool: True εάν η αποθήκευση ήταν επιτυχής
        """
        try:
            raw.export(output_path, fmt='edf', overwrite=True, verbose=False)
            return True
        except Exception as e:
            print(f"Σφάλμα αποθήκευσης: {str(e)}")
            return False
    
    @staticmethod
    def validate_edf_file(file_path: str) -> Dict[str, Any]:
        """
        Επικύρωση και πληροφορίες EDF αρχείου
        
        Args:
            file_path: Διαδρομή αρχείου
            
        Returns:
            Dict με πληροφορίες αρχείου
        """
        try:
            raw, channels = EEGDataManager.load_edf_file(file_path)
            
            info = {
                'valid': True,
                'channels': channels,
                'sampling_rate': raw.info['sfreq'],
                'duration': raw.times[-1],
                'n_samples': len(raw.times),
                'n_channels': len(channels)
            }
            
            return info
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


class EEGPreprocessor:
    """Προεπεξεργασία EEG δεδομένων - φιλτράρισμα, τυποποίηση"""
    
    @staticmethod
    def apply_bandpass_filter(raw: mne.io.Raw, 
                            low_freq: float = 1.0, 
                            high_freq: float = 40.0) -> mne.io.Raw:
        """
        Εφαρμογή ζωνοπερατού φίλτρου
        
        Args:
            raw: Raw EEG δεδομένα
            low_freq: Κάτω συχνότητα (Hz)
            high_freq: Άνω συχνότητα (Hz)
            
        Returns:
            Φιλτραρισμένα Raw δεδομένα
        """
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=low_freq, 
            h_freq=high_freq, 
            verbose=False,
            fir_design='firwin'
        )
        return raw_filtered
    
    @staticmethod
    def get_data_statistics(raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
        """
        Υπολογισμός στατιστικών δεδομένων ανά κανάλι
        
        Args:
            raw: Raw EEG δεδομένα
            
        Returns:
            Dictionary με στατιστικά ανά κανάλι
        """
        data = raw.get_data() * 1e6  # Μετατροπή σε μV
        stats_dict = {}
        
        for i, ch_name in enumerate(raw.ch_names):
            channel_data = data[i]
            stats_dict[ch_name] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'variance': float(np.var(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'range': float(np.ptp(channel_data)),
                'rms': float(np.sqrt(np.mean(channel_data**2)))
            }
        
        return stats_dict


class EEGBackendCore:
    """Κεντρικό Backend για EEG επεξεργασία"""
    
    def __init__(self):
        self.data_manager = EEGDataManager()
        self.preprocessor = EEGPreprocessor()
        self.raw_data = None
        self.filtered_data = None
        self.current_file = None
        
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Φόρτωση και αρχική επεξεργασία αρχείου
        
        Args:
            file_path: Διαδρομή αρχείου
            
        Returns:
            Dictionary με πληροφορίες φόρτωσης
        """
        try:
            # Φόρτωση δεδομένων
            self.raw_data, channels = self.data_manager.load_edf_file(file_path)
            self.current_file = file_path
            
            # Εφαρμογή φίλτρου
            self.filtered_data = self.preprocessor.apply_bandpass_filter(self.raw_data)
            
            # Επιστροφή πληροφοριών
            return {
                'success': True,
                'channels': channels,
                'sampling_rate': self.raw_data.info['sfreq'],
                'duration': self.raw_data.times[-1],
                'n_samples': len(self.raw_data.times),
                'stats_original': self.preprocessor.get_data_statistics(self.raw_data),
                'stats_filtered': self.preprocessor.get_data_statistics(self.filtered_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_cleaned_data(self, cleaned_raw: mne.io.Raw, output_path: str) -> bool:
        """Αποθήκευση καθαρισμένων δεδομένων"""
        return self.data_manager.save_cleaned_data(cleaned_raw, output_path)
    
    def get_filtered_data(self) -> Optional[mne.io.Raw]:
        """Επιστροφή φιλτραρισμένων δεδομένων"""
        return self.filtered_data
    
    def get_original_data(self) -> Optional[mne.io.Raw]:
        """Επιστροφή αρχικών δεδομένων"""
        return self.raw_data