#!/usr/bin/env python3
"""
EEG Backend Core - Κεντρικό σύστημα επεξεργασίας EEG δεδομένων
===========================================================

Περιέχει τις βασικές κλάσεις και συναρτήσεις για την επεξεργασία δεδομένων EEG:
- Διαχείριση φόρτωσης και αποθήκευσης αρχείων EDF
- Φιλτράρισμα και προεπεξεργασία σημάτων
- Αυτόματος εντοπισμός EEG καναλιών
- Στατιστική ανάλυση δεδομένων

Author: porfanid
Version: 1.0
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy import stats
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class EEGDataManager:
    """
    Διαχείριση δεδομένων EEG - φόρτωση, αποθήκευση, επικύρωση

    Παρέχει στατικές μεθόδους για:
    - Αυτόματο εντοπισμό EEG καναλιών
    - Φόρτωση και αποθήκευση EDF αρχείων
    - Εξαγωγή πληροφοριών αρχείου
    """

    @staticmethod
    def detect_eeg_channels(raw: mne.io.Raw) -> List[str]:
        """
        Αυτόματος εντοπισμός EEG καναλιών από τα διαθέσιμα κανάλια

        Args:
            raw: Raw EEG δεδομένα

        Returns:
            List[str]: Λίστα με τα εντοπισμένα EEG κανάλια
        """
        # Κοινά EEG κανάλια βάσει του 10-20 συστήματος
        common_eeg_channels = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "A1",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "A2",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "PO9",
            "O1",
            "Oz",
            "O2",
            "PO10",
            "AF3",
            "AF4",
            "F1",
            "F2",
            "F5",
            "F6",
            "FT7",
            "FC3",
            "FC4",
            "FT8",
            "C1",
            "C2",
            "C5",
            "C6",
            "TP7",
            "CP3",
            "CPz",
            "CP4",
            "TP8",
            "P1",
            "P2",
            "P5",
            "P6",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
        ]

        # Βρες διαθέσιμα EEG κανάλια
        available_eeg_channels = []

        for ch_name in raw.ch_names:
            # Έλεγχος για κοινά EEG κανάλια
            if ch_name in common_eeg_channels:
                available_eeg_channels.append(ch_name)
            # Έλεγχος για κανάλια που μοιάζουν με EEG (π.χ. F4, P3, etc.)
            elif (
                len(ch_name) >= 2
                and ch_name[0].upper() in ["F", "C", "P", "O", "T", "A"]
                and ch_name[1:].replace("z", "").replace("Z", "").isdigit()
            ):
                available_eeg_channels.append(ch_name)
            # Έλεγχος για κανάλια με πρόθεμα AF, FP, PO, etc.
            elif (
                len(ch_name) >= 3
                and ch_name[:2].upper() in ["AF", "FP", "PO", "FC", "CP", "FT", "TP"]
                and ch_name[2:].replace("z", "").replace("Z", "").isdigit()
            ):
                available_eeg_channels.append(ch_name)

        return available_eeg_channels

    @staticmethod
    def load_edf_file(
        file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, List[str]]:
        """
        Φόρτωση EDF αρχείου και εξαγωγή καναλιών

        Args:
            file_path: Διαδρομή του EDF αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)

        Returns:
            Tuple[mne.io.Raw, List[str]]: Raw δεδομένα και λίστα καναλιών

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

        if selected_channels is None:
            # Αυτόματος εντοπισμός EEG καναλιών (backward compatibility)
            available_channels = EEGDataManager.detect_eeg_channels(raw)

            if not available_channels:
                raise ValueError("Δεν βρέθηκαν έγκυρα EEG κανάλια στο αρχείο")
        else:
            # Χρήση επιλεγμένων καναλιών
            available_channels = []
            for ch in selected_channels:
                if ch in raw.ch_names:
                    available_channels.append(ch)
                else:
                    raise ValueError(f"Το κανάλι '{ch}' δεν υπάρχει στο αρχείο")

            if len(available_channels) < 3:
                raise ValueError("Χρειάζονται τουλάχιστον 3 κανάλια για την ανάλυση")

        # Κρατάμε μόνο τα επιλεγμένα κανάλια
        raw.pick_channels(available_channels)

        # Ορισμός montage για τοπογραφική απεικόνιση
        try:
            raw.set_montage("standard_1020", on_missing="warn")
        except Exception:
            # Αν αποτύχει το montage, συνεχίζουμε χωρίς αυτό
            pass

        return raw, available_channels

    @staticmethod
    def load_edf_file_info(file_path: str) -> Dict[str, Any]:
        """
        Φόρτωση πληροφοριών EDF αρχείου χωρίς επεξεργασία

        Args:
            file_path: Διαδρομή του EDF αρχείου

        Returns:
            Dict με πληροφορίες αρχείου
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            return {
                "success": True,
                "channels": raw.ch_names,
                "sampling_rate": raw.info["sfreq"],
                "n_channels": len(raw.ch_names),
                "detected_eeg": EEGDataManager.detect_eeg_channels(raw),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
            raw.export(output_path, fmt="edf", overwrite=True, verbose=False)
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
                "valid": True,
                "channels": channels,
                "sampling_rate": raw.info["sfreq"],
                "duration": raw.times[-1],
                "n_samples": len(raw.times),
                "n_channels": len(channels),
            }

            return info

        except Exception as e:
            return {"valid": False, "error": str(e)}


class EEGPreprocessor:
    """
    Προεπεξεργασία EEG δεδομένων - φιλτράρισμα, τυποποίηση

    Παρέχει στατικές μεθόδους για:
    - Εφαρμογή ζωνοπερατών φίλτρων
    - Υπολογισμό στατιστικών δεδομένων
    - Προεπεξεργασία σημάτων
    """

    @staticmethod
    def apply_bandpass_filter(
        raw: mne.io.Raw, low_freq: float = 1.0, high_freq: float = 40.0
    ) -> mne.io.Raw:
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
            l_freq=low_freq, h_freq=high_freq, verbose=False, fir_design="firwin"
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
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "variance": float(np.var(channel_data)),
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
                "range": float(np.ptp(channel_data)),
                "rms": float(np.sqrt(np.mean(channel_data**2))),
            }

        return stats_dict


class EEGBackendCore:
    """
    Κεντρικό Backend για EEG επεξεργασία

    Συνδυάζει τη διαχείριση δεδομένων και την προεπεξεργασία για να παρέχει
    μια ενιαία διεπαφή για φόρτωση, επεξεργασία και αποθήκευση EEG δεδομένων.

    Attributes:
        data_manager (EEGDataManager): Διαχειριστής δεδομένων
        preprocessor (EEGPreprocessor): Προεπεξεργαστής δεδομένων
        raw_data (mne.io.Raw): Αρχικά δεδομένα
        filtered_data (mne.io.Raw): Φιλτραρισμένα δεδομένα
        current_file (str): Τρέχον αρχείο που επεξεργάζεται
    """

    def __init__(self):
        """
        Αρχικοποίηση του κεντρικού backend

        Δημιουργεί instances των data manager και preprocessor και
        αρχικοποιεί τις μεταβλητές κατάστασης.
        """
        self.data_manager = EEGDataManager()
        self.preprocessor = EEGPreprocessor()
        self.raw_data = None
        self.filtered_data = None
        self.current_file = None

    def load_file(
        self, file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Φόρτωση και αρχική επεξεργασία αρχείου

        Args:
            file_path: Διαδρομή αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)

        Returns:
            Dictionary με πληροφορίες φόρτωσης
        """
        try:
            # Φόρτωση δεδομένων
            self.raw_data, channels = self.data_manager.load_edf_file(
                file_path, selected_channels
            )
            self.current_file = file_path

            # Εφαρμογή φίλτρου
            self.filtered_data = self.preprocessor.apply_bandpass_filter(self.raw_data)

            # Επιστροφή πληροφοριών
            return {
                "success": True,
                "channels": channels,
                "sampling_rate": self.raw_data.info["sfreq"],
                "duration": self.raw_data.times[-1],
                "n_samples": len(self.raw_data.times),
                "stats_original": self.preprocessor.get_data_statistics(self.raw_data),
                "stats_filtered": self.preprocessor.get_data_statistics(
                    self.filtered_data
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Λήψη πληροφοριών αρχείου χωρίς φόρτωση δεδομένων

        Args:
            file_path: Διαδρομή αρχείου

        Returns:
            Dictionary με πληροφορίες αρχείου
        """
        try:
            return self.data_manager.load_edf_file_info(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_cleaned_data(self, cleaned_raw: mne.io.Raw, output_path: str) -> bool:
        """
        Αποθήκευση καθαρισμένων δεδομένων

        Αποθηκεύει τα καθαρισμένα EEG δεδομένα σε EDF format.

        Args:
            cleaned_raw (mne.io.Raw): Τα καθαρισμένα δεδομένα
            output_path (str): Διαδρομή αρχείου εξόδου

        Returns:
            bool: True εάν η αποθήκευση ήταν επιτυχής
        """
        return self.data_manager.save_cleaned_data(cleaned_raw, output_path)

    def get_filtered_data(self) -> Optional[mne.io.Raw]:
        """
        Επιστροφή φιλτραρισμένων δεδομένων

        Returns:
            Optional[mne.io.Raw]: Τα φιλτραρισμένα δεδομένα ή None αν δεν υπάρχουν
        """
        return self.filtered_data

    def get_original_data(self) -> Optional[mne.io.Raw]:
        """
        Επιστροφή αρχικών δεδομένων

        Returns:
            Optional[mne.io.Raw]: Τα αρχικά δεδομένα ή None αν δεν υπάρχουν
        """
        return self.raw_data
