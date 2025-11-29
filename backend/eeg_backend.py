#!/usr/bin/env python3
"""
EEG Backend Core - Κεντρικό σύστημα επεξεργασίας EEG δεδομένων
===========================================================

Περιέχει τις βασικές κλάσεις και συναρτήσεις για την επεξεργασία δεδομένων EEG:
- Διαχείριση φόρτωσης και αποθήκευσης αρχείων (.edf, .bdf, .fif, .csv, .set)
- Φιλτράρισμα και προεπεξεργασία σημάτων
- Αυτόματος εντοπισμός EEG καναλιών
- Στατιστική ανάλυση δεδομένων

Author: porfanid
Version: 1.1
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

# Supported file formats for import and export
SUPPORTED_IMPORT_FORMATS = [".edf", ".bdf", ".fif", ".csv", ".set"]
SUPPORTED_EXPORT_FORMATS = [".edf", ".bdf", ".fif", ".csv", ".set"]

# Default sampling rate for CSV files without time column
DEFAULT_SAMPLING_RATE = 256.0

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class EEGDataManager:
    """
    Διαχείριση δεδομένων EEG - φόρτωση, αποθήκευση, επικύρωση

    Παρέχει στατικές μεθόδους για:
    - Αυτόματο εντοπισμό EEG καναλιών
    - Φόρτωση αρχείων σε διάφορες μορφές (.edf, .bdf, .fif, .csv, .set)
    - Αποθήκευση αρχείων σε διάφορες μορφές
    - Εξαγωγή πληροφοριών αρχείου
    """

    @staticmethod
    def get_supported_import_formats() -> List[str]:
        """
        Επιστρέφει τη λίστα με τις υποστηριζόμενες μορφές εισαγωγής.

        Returns:
            List[str]: Λίστα με τις υποστηριζόμενες επεκτάσεις εισαγωγής
        """
        return SUPPORTED_IMPORT_FORMATS.copy()

    @staticmethod
    def get_supported_export_formats() -> List[str]:
        """
        Επιστρέφει τη λίστα με τις υποστηριζόμενες μορφές εξαγωγής.

        Returns:
            List[str]: Λίστα με τις υποστηριζόμενες επεκτάσεις εξαγωγής
        """
        return SUPPORTED_EXPORT_FORMATS.copy()

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
    def read_raw(file_path: str, preload: bool = True) -> mne.io.Raw:
        """
        Φόρτωση raw EEG δεδομένων από διάφορες μορφές αρχείων.

        Υποστηρίζει τις ακόλουθες μορφές:
        - .edf (European Data Format)
        - .bdf (BioSemi Data Format)
        - .fif (MNE-Python native format)
        - .csv (Comma-Separated Values)
        - .set (EEGLAB format)

        Args:
            file_path: Διαδρομή του αρχείου
            preload: Αν True, φορτώνει τα δεδομένα στη μνήμη

        Returns:
            mne.io.Raw: Raw EEG δεδομένα

        Raises:
            FileNotFoundError: Εάν το αρχείο δεν βρεθεί
            ValueError: Εάν η μορφή αρχείου δεν υποστηρίζεται
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε")

        ext = path.suffix.lower()

        if ext not in SUPPORTED_IMPORT_FORMATS:
            raise ValueError(
                f"Μη υποστηριζόμενη μορφή αρχείου: {ext}. "
                f"Υποστηριζόμενες μορφές: {SUPPORTED_IMPORT_FORMATS}"
            )

        try:
            if ext == ".edf":
                raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
            elif ext == ".bdf":
                raw = mne.io.read_raw_bdf(file_path, preload=preload, verbose=False)
            elif ext == ".fif":
                raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=False)
            elif ext == ".set":
                raw = mne.io.read_raw_eeglab(file_path, preload=preload, verbose=False)
            elif ext == ".csv":
                raw = EEGDataManager._read_raw_csv(file_path)
            else:
                raise ValueError(f"Μη υποστηριζόμενη μορφή αρχείου: {ext}")

            return raw

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Σφάλμα φόρτωσης αρχείου {ext}: {str(e)}")

    @staticmethod
    def _read_raw_csv(
        file_path: str,
        sfreq: Optional[float] = None,
        ch_types: str = "eeg",
    ) -> mne.io.Raw:
        """
        Φόρτωση EEG δεδομένων από CSV αρχείο.

        Το CSV αρχείο πρέπει να έχει:
        - Κάθε στήλη είναι ένα κανάλι (εκτός από στήλη 'time' αν υπάρχει)
        - Κάθε γραμμή είναι ένα δείγμα
        - Η πρώτη γραμμή περιέχει τα ονόματα των καναλιών

        Args:
            file_path: Διαδρομή του CSV αρχείου
            sfreq: Sampling frequency σε Hz. Αν None, υπολογίζεται από τα δεδομένα.
            ch_types: Τύπος καναλιών (default: 'eeg')

        Returns:
            mne.io.Raw: Raw EEG δεδομένα

        Raises:
            ValueError: Εάν το αρχείο δεν είναι έγκυρο CSV
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Σφάλμα ανάγνωσης CSV: {str(e)}")

        # Check for time column and remove it
        time_columns = ["time", "Time", "TIME", "timestamp", "Timestamp"]
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break

        if time_col is not None:
            time_data = df[time_col].values
            df = df.drop(columns=[time_col])
            # Calculate sampling frequency from time column
            if sfreq is None and len(time_data) > 1:
                time_diff = np.diff(time_data)
                median_diff = np.median(time_diff)
                # Validate time difference is positive and non-zero
                if median_diff > 0:
                    sfreq = float(1.0 / median_diff)
                else:
                    sfreq = DEFAULT_SAMPLING_RATE
                    warnings.warn(
                        f"Invalid time differences in data, using default "
                        f"sfreq={sfreq} Hz",
                        UserWarning,
                    )
        elif sfreq is None:
            # Default sampling frequency if not provided
            sfreq = DEFAULT_SAMPLING_RATE
            warnings.warn(
                f"Δεν βρέθηκε στήλη χρόνου, χρήση default sfreq={sfreq} Hz",
                UserWarning,
            )

        ch_names = list(df.columns)
        # Data should be (n_channels, n_samples)
        data = df.values.T

        # Convert to volts if data appears to be in microvolts
        # EEG data typically ranges from -100 to +100 microvolts
        # If max absolute value is greater than 1, assume data is in microvolts
        if np.max(np.abs(data)) > 1:
            data = data * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)

        return raw

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
        except (ValueError, KeyError, RuntimeError) as e:
            # Αν αποτύχει το montage, συνεχίζουμε χωρίς αυτό
            import warnings

            warnings.warn(f"Αδυναμία ορισμού montage: {str(e)}", UserWarning)

        return raw, available_channels

    @staticmethod
    def load_raw_file(
        file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, List[str]]:
        """
        Φόρτωση αρχείου EEG από διάφορες μορφές και εξαγωγή καναλιών.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Διαδρομή του αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)

        Returns:
            Tuple[mne.io.Raw, List[str]]: Raw δεδομένα και λίστα καναλιών

        Raises:
            FileNotFoundError: Εάν το αρχείο δεν βρεθεί
            ValueError: Εάν το αρχείο δεν είναι έγκυρο ή δεν υποστηρίζεται
        """
        raw = EEGDataManager.read_raw(file_path, preload=True)

        if selected_channels is None:
            # Αυτόματος εντοπισμός EEG καναλιών
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
        except (ValueError, KeyError, RuntimeError) as e:
            warnings.warn(f"Αδυναμία ορισμού montage: {str(e)}", UserWarning)

        return raw, available_channels

    @staticmethod
    def load_file_info(file_path: str) -> Dict[str, Any]:
        """
        Φόρτωση πληροφοριών αρχείου EEG χωρίς επεξεργασία.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Διαδρομή του αρχείου

        Returns:
            Dict με πληροφορίες αρχείου:
                - success (bool): Επιτυχία φόρτωσης
                - channels (List[str]): Ονόματα καναλιών
                - sampling_rate (float): Ρυθμός δειγματοληψίας
                - n_channels (int): Αριθμός καναλιών
                - detected_eeg (List[str]): Εντοπισμένα EEG κανάλια
                - n_annotations (int): Αριθμός annotations/markers
                - format (str): Μορφή αρχείου
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε")

        ext = Path(file_path).suffix.lower()

        try:
            raw = EEGDataManager.read_raw(file_path, preload=False)
            return {
                "success": True,
                "channels": list(raw.ch_names),
                "sampling_rate": raw.info["sfreq"],
                "n_channels": len(raw.ch_names),
                "detected_eeg": EEGDataManager.detect_eeg_channels(raw),
                "n_annotations": len(raw.annotations),
                "format": ext,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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
                "channels": list(raw.ch_names),
                "sampling_rate": raw.info["sfreq"],
                "n_channels": len(raw.ch_names),
                "detected_eeg": EEGDataManager.detect_eeg_channels(raw),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def export_raw(raw: mne.io.Raw, output_path: str) -> bool:
        """
        Εξαγωγή raw δεδομένων σε διάφορες μορφές.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            raw: Raw EEG δεδομένα
            output_path: Διαδρομή εξόδου

        Returns:
            bool: True εάν η εξαγωγή ήταν επιτυχής

        Raises:
            ValueError: Εάν η μορφή εξόδου δεν υποστηρίζεται
        """
        ext = Path(output_path).suffix.lower()

        if ext not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Μη υποστηριζόμενη μορφή εξαγωγής: {ext}. "
                f"Υποστηριζόμενες μορφές: {SUPPORTED_EXPORT_FORMATS}"
            )

        try:
            if ext == ".edf":
                raw.export(output_path, fmt="edf", overwrite=True, verbose=False)
            elif ext == ".bdf":
                # BDF uses the same export mechanism as EDF with auto format
                raw.export(output_path, fmt="auto", overwrite=True, verbose=False)
            elif ext == ".fif":
                raw.save(output_path, overwrite=True, verbose=False)
            elif ext == ".csv":
                EEGDataManager._export_raw_csv(raw, output_path)
            elif ext == ".set":
                # EEGLAB format requires eeglabio package
                raw.export(output_path, fmt="eeglab", overwrite=True, verbose=False)
            return True
        except Exception as e:
            print(f"Σφάλμα εξαγωγής: {str(e)}")
            return False

    @staticmethod
    def _export_raw_csv(raw: mne.io.Raw, output_path: str) -> None:
        """
        Εξαγωγή raw δεδομένων σε CSV μορφή.

        Args:
            raw: Raw EEG δεδομένα
            output_path: Διαδρομή εξόδου
        """
        data = raw.get_data().T  # (n_samples, n_channels)
        times = raw.times

        df = pd.DataFrame(data, columns=raw.ch_names)
        df.insert(0, "time", times)
        df.to_csv(output_path, index=False)

    @staticmethod
    def save_cleaned_data(raw: mne.io.Raw, output_path: str) -> bool:
        """
        Αποθήκευση καθαρισμένων δεδομένων.

        Υποστηρίζει αυτόματη ανίχνευση μορφής από την επέκταση αρχείου.
        Για backward compatibility, αν η επέκταση δεν υποστηρίζεται,
        χρησιμοποιείται EDF format.

        Args:
            raw: Καθαρισμένα Raw δεδομένα
            output_path: Διαδρομή εξόδου

        Returns:
            bool: True εάν η αποθήκευση ήταν επιτυχής
        """
        ext = Path(output_path).suffix.lower()

        # Use export_raw for supported formats
        if ext in SUPPORTED_EXPORT_FORMATS:
            return EEGDataManager.export_raw(raw, output_path)

        # Fallback to EDF for unsupported extensions (backward compatibility)
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

    @staticmethod
    def validate_file(file_path: str) -> Dict[str, Any]:
        """
        Επικύρωση και πληροφορίες αρχείου EEG.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Διαδρομή αρχείου

        Returns:
            Dict με πληροφορίες αρχείου:
                - valid (bool): Αν το αρχείο είναι έγκυρο
                - channels (List[str]): Ονόματα καναλιών
                - sampling_rate (float): Ρυθμός δειγματοληψίας
                - duration (float): Διάρκεια σε δευτερόλεπτα
                - n_samples (int): Αριθμός δειγμάτων
                - n_channels (int): Αριθμός καναλιών
                - n_annotations (int): Αριθμός annotations/markers
                - format (str): Μορφή αρχείου
        """
        try:
            raw, channels = EEGDataManager.load_raw_file(file_path)
            ext = Path(file_path).suffix.lower()

            info = {
                "valid": True,
                "channels": channels,
                "sampling_rate": raw.info["sfreq"],
                "duration": raw.times[-1],
                "n_samples": len(raw.times),
                "n_channels": len(channels),
                "n_annotations": len(raw.annotations),
                "format": ext,
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
        Φόρτωση και αρχική επεξεργασία αρχείου.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Διαδρομή αρχείου
            selected_channels: Λίστα επιλεγμένων καναλιών (None για αυτόματη ανίχνευση)

        Returns:
            Dictionary με πληροφορίες φόρτωσης
        """
        try:
            # Φόρτωση δεδομένων (υποστηρίζει πολλαπλές μορφές)
            self.raw_data, channels = self.data_manager.load_raw_file(
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
                "n_annotations": len(self.raw_data.annotations),
                "stats_original": self.preprocessor.get_data_statistics(self.raw_data),
                "stats_filtered": self.preprocessor.get_data_statistics(
                    self.filtered_data
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Λήψη πληροφοριών αρχείου χωρίς φόρτωση δεδομένων.

        Υποστηρίζει: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Διαδρομή αρχείου

        Returns:
            Dictionary με πληροφορίες αρχείου
        """
        try:
            return self.data_manager.load_file_info(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_cleaned_data(self, cleaned_raw: mne.io.Raw, output_path: str) -> bool:
        """
        Αποθήκευση καθαρισμένων δεδομένων.

        Υποστηρίζει: .edf, .fif, .csv

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
