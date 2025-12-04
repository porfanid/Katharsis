#!/usr/bin/env python3
"""
EEG Backend Core - Central EEG Data Processing System
======================================================

Contains the core classes and functions for EEG data processing:
- File loading and saving management (.edf, .bdf, .fif, .csv, .set)
- Signal filtering and preprocessing
- Automatic EEG channel detection
- Statistical data analysis

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
# Note: BDF export is not supported by MNE's export function
SUPPORTED_EXPORT_FORMATS = [".edf", ".fif", ".csv", ".set"]

# Default sampling rate for CSV files without time column
DEFAULT_SAMPLING_RATE = 256.0

# Threshold for detecting if data is in microvolts vs volts
# EEG data typically ranges from -100 to +100 microvolts
# If max absolute value is greater than this threshold, assume data is in microvolts
MICROVOLT_DETECTION_THRESHOLD = 1.0

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class EEGDataManager:
    """
    EEG Data Management - loading, saving, validation.

    Provides static methods for:
    - Automatic EEG channel detection
    - Loading files in various formats (.edf, .bdf, .fif, .csv, .set)
    - Saving files in various formats
    - Extracting file information
    """

    @staticmethod
    def get_supported_import_formats() -> List[str]:
        """
        Returns the list of supported import formats.

        Returns:
            List[str]: List of supported import file extensions
        """
        return SUPPORTED_IMPORT_FORMATS.copy()

    @staticmethod
    def get_supported_export_formats() -> List[str]:
        """
        Returns the list of supported export formats.

        Returns:
            List[str]: List of supported export file extensions
        """
        return SUPPORTED_EXPORT_FORMATS.copy()

    @staticmethod
    def detect_eeg_channels(raw: mne.io.Raw) -> List[str]:
        """
        Automatic detection of EEG channels from available channels.

        Args:
            raw: Raw EEG data

        Returns:
            List[str]: List of detected EEG channels
        """
        # Common EEG channels based on the 10-20 system
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

        # Find available EEG channels
        available_eeg_channels = []

        for ch_name in raw.ch_names:
            # Check for common EEG channels
            if ch_name in common_eeg_channels:
                available_eeg_channels.append(ch_name)
            # Check for channels that look like EEG (e.g., F4, P3, etc.)
            elif (
                len(ch_name) >= 2
                and ch_name[0].upper() in ["F", "C", "P", "O", "T", "A"]
                and ch_name[1:].replace("z", "").replace("Z", "").isdigit()
            ):
                available_eeg_channels.append(ch_name)
            # Check for channels with prefix AF, FP, PO, etc.
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
        Load raw EEG data from various file formats.

        Supports the following formats:
        - .edf (European Data Format)
        - .bdf (BioSemi Data Format)
        - .fif (MNE-Python native format)
        - .csv (Comma-Separated Values)
        - .set (EEGLAB format)

        Args:
            file_path: Path to the file
            preload: If True, loads data into memory

        Returns:
            mne.io.Raw: Raw EEG data

        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        ext = path.suffix.lower()

        if ext not in SUPPORTED_IMPORT_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {SUPPORTED_IMPORT_FORMATS}"
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
                raise ValueError(f"Unsupported file format: {ext}")

            return raw

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading {ext} file: {str(e)}")

    @staticmethod
    def _read_raw_csv(
        file_path: str,
        sfreq: Optional[float] = None,
        ch_types: str = "eeg",
    ) -> mne.io.Raw:
        """
        Load EEG data from CSV file.

        The CSV file should have:
        - Each column is a channel (except 'time' column if present)
        - Each row is a sample
        - The first row contains channel names

        Args:
            file_path: Path to the CSV file
            sfreq: Sampling frequency in Hz. If None, calculated from data.
            ch_types: Type of channels (default: 'eeg')

        Returns:
            mne.io.Raw: Raw EEG data

        Raises:
            ValueError: If the file is not a valid CSV
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {str(e)}")

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
                f"No time column found, using default sfreq={sfreq} Hz",
                UserWarning,
            )

        ch_names = list(df.columns)
        # Data should be (n_channels, n_samples)
        data = df.values.T

        # Convert to volts if data appears to be in microvolts
        if np.max(np.abs(data)) > MICROVOLT_DETECTION_THRESHOLD:
            data = data * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)

        return raw

    @staticmethod
    def load_edf_file(
        file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, List[str]]:
        """
        Load EDF file and extract channels.

        Args:
            file_path: Path to the EDF file
            selected_channels: List of selected channels (None for auto detection)

        Returns:
            Tuple[mne.io.Raw, List[str]]: Raw data and list of channels

        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file is not a valid EDF
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            raise ValueError(f"Error loading EDF file: {str(e)}")

        if selected_channels is None:
            # Automatic EEG channel detection (backward compatibility)
            available_channels = EEGDataManager.detect_eeg_channels(raw)

            if not available_channels:
                raise ValueError("No valid EEG channels found in the file")
        else:
            # Use selected channels
            available_channels = []
            for ch in selected_channels:
                if ch in raw.ch_names:
                    available_channels.append(ch)
                else:
                    raise ValueError(f"Channel '{ch}' does not exist in the file")

            if len(available_channels) < 3:
                raise ValueError("At least 3 channels are required for analysis")

        # Keep only selected channels
        raw.pick_channels(available_channels)

        # Set montage for topographic visualization
        try:
            raw.set_montage("standard_1020", on_missing="warn")
        except (ValueError, KeyError, RuntimeError) as e:
            # If montage fails, continue without it
            import warnings

            warnings.warn(f"Unable to set montage: {str(e)}", UserWarning)

        return raw, available_channels

    @staticmethod
    def load_raw_file(
        file_path: str, selected_channels: Optional[List[str]] = None
    ) -> Tuple[mne.io.Raw, List[str]]:
        """
        Load EEG file from various formats and extract channels.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Path to the file
            selected_channels: List of selected channels (None for auto detection)

        Returns:
            Tuple[mne.io.Raw, List[str]]: Raw data and list of channels

        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file is invalid or not supported
        """
        raw = EEGDataManager.read_raw(file_path, preload=True)

        if selected_channels is None:
            # Automatic EEG channel detection
            available_channels = EEGDataManager.detect_eeg_channels(raw)

            if not available_channels:
                raise ValueError("No valid EEG channels found in the file")
        else:
            # Use selected channels
            available_channels = []
            for ch in selected_channels:
                if ch in raw.ch_names:
                    available_channels.append(ch)
                else:
                    raise ValueError(f"Channel '{ch}' does not exist in the file")

            if len(available_channels) < 3:
                raise ValueError("At least 3 channels are required for analysis")

        # Keep only selected channels
        raw.pick_channels(available_channels)

        # Set montage for topographic visualization
        try:
            raw.set_montage("standard_1020", on_missing="warn")
        except (ValueError, KeyError, RuntimeError) as e:
            warnings.warn(f"Unable to set montage: {str(e)}", UserWarning)

        return raw, available_channels

    @staticmethod
    def load_file_info(file_path: str) -> Dict[str, Any]:
        """
        Load EEG file information without processing.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: Path to the file

        Returns:
            Dict with file information:
                - success (bool): Loading success
                - channels (List[str]): Channel names
                - sampling_rate (float): Sampling rate
                - n_channels (int): Number of channels
                - detected_eeg (List[str]): Detected EEG channels
                - n_annotations (int): Number of annotations/markers
                - format (str): File format
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")

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
        Load EDF file information without processing.

        Args:
            file_path: Path to the EDF file

        Returns:
            Dict with file information
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")

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
        Export raw data to various formats.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            raw: Raw EEG data
            output_path: Output path

        Returns:
            bool: True if export was successful

        Raises:
            ValueError: If output format is not supported
        """
        ext = Path(output_path).suffix.lower()

        if ext not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export format: {ext}. "
                f"Supported formats: {SUPPORTED_EXPORT_FORMATS}"
            )

        try:
            if ext == ".edf":
                raw.export(output_path, fmt="edf", overwrite=True, verbose=False)
            elif ext == ".fif":
                raw.save(output_path, overwrite=True, verbose=False)
            elif ext == ".csv":
                EEGDataManager._export_raw_csv(raw, output_path)
            elif ext == ".set":
                # EEGLAB format requires eeglabio package
                raw.export(output_path, fmt="eeglab", overwrite=True, verbose=False)
            return True
        except Exception as e:
            warnings.warn(f"Export error: {str(e)}", UserWarning)
            return False

    @staticmethod
    def _export_raw_csv(raw: mne.io.Raw, output_path: str) -> None:
        """
        Export raw data to CSV format.

        Args:
            raw: Raw EEG data
            output_path: Output path
        """
        data = raw.get_data().T  # (n_samples, n_channels)
        times = raw.times

        df = pd.DataFrame(data, columns=raw.ch_names)
        df.insert(0, "time", times)
        df.to_csv(output_path, index=False)

    @staticmethod
    def save_cleaned_data(raw: mne.io.Raw, output_path: str) -> bool:
        """
        Save cleaned data.

        Supports automatic format detection from file extension.
        For backward compatibility, if extension is not supported,
        uses EDF format.

        Args:
            raw: Cleaned Raw data
            output_path: Output path

        Returns:
            bool: True if saving was successful
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
            warnings.warn(f"Save error: {str(e)}", UserWarning)
            return False

    @staticmethod
    def validate_edf_file(file_path: str) -> Dict[str, Any]:
        """
        Validate and get EDF file information.

        Args:
            file_path: File path

        Returns:
            Dict with file information
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
        Validate and get EEG file information.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: File path

        Returns:
            Dict with file information:
                - valid (bool): If file is valid
                - channels (List[str]): Channel names
                - sampling_rate (float): Sampling rate
                - duration (float): Duration in seconds
                - n_samples (int): Number of samples
                - n_channels (int): Number of channels
                - n_annotations (int): Number of annotations/markers
                - format (str): File format
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
    EEG Data Preprocessing - filtering, standardization.

    Provides static methods for:
    - Applying band-pass filters
    - Computing data statistics
    - Preprocessing signals
    """

    @staticmethod
    def apply_bandpass_filter(
        raw: mne.io.Raw, low_freq: float = 1.0, high_freq: float = 40.0
    ) -> mne.io.Raw:
        """
        Apply band-pass filter.

        Args:
            raw: Raw EEG data
            low_freq: Low frequency cutoff (Hz)
            high_freq: High frequency cutoff (Hz)

        Returns:
            Filtered Raw data
        """
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=low_freq, h_freq=high_freq, verbose=False, fir_design="firwin"
        )
        return raw_filtered

    @staticmethod
    def get_data_statistics(raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
        """
        Compute data statistics per channel.

        Args:
            raw: Raw EEG data

        Returns:
            Dictionary with statistics per channel
        """
        data = raw.get_data() * 1e6  # Convert to μV
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


class SignalEditor:
    """
    Signal Editing Utilities - cutting, joining, and annotation detection.

    Provides static methods for:
    - Detecting resting phase annotations (eyes open/closed)
    - Cutting signal regions
    - Joining signal segments
    """

    # Common resting phase annotation patterns
    EYES_OPEN_PATTERNS = [
        "eyes open",
        "eyes_open",
        "eyesopen",
        "eo",
        "e/o",
        "rest_open",
        "resting_open",
    ]

    EYES_CLOSED_PATTERNS = [
        "eyes closed",
        "eyes_closed",
        "eyesclosed",
        "ec",
        "e/c",
        "rest_closed",
        "resting_closed",
    ]

    # Default duration for annotations without specified duration
    DEFAULT_PHASE_DURATION = 60.0

    @staticmethod
    def detect_resting_phases(raw: mne.io.Raw) -> List[Dict[str, Any]]:
        """
        Detect resting phase annotations in EEG data.

        Looks for annotations indicating "eyes open" or "eyes closed"
        states commonly used in resting state EEG paradigms.
        Also checks for marker channels (MarkerValueInt, MarkerIndex) if
        standard annotations are not present.

        Args:
            raw: Raw EEG data with annotations

        Returns:
            List of phase dictionaries with keys:
                - label: Phase label (e.g., "Eyes Open", "Eyes Closed")
                - start: Start time in seconds
                - end: End time in seconds
                - duration: Duration in seconds
        """
        phases = []

        # First try standard annotations
        if raw.annotations and len(raw.annotations) > 0:
            default_duration = SignalEditor.DEFAULT_PHASE_DURATION

            for annot in raw.annotations:
                description = annot["description"].lower().strip()
                onset = annot["onset"]
                duration = annot["duration"]

                # Check for eyes open patterns
                is_eyes_open = any(
                    pattern in description
                    for pattern in SignalEditor.EYES_OPEN_PATTERNS
                )

                # Check for eyes closed patterns
                is_eyes_closed = any(
                    pattern in description
                    for pattern in SignalEditor.EYES_CLOSED_PATTERNS
                )

                if is_eyes_open:
                    actual_duration = duration if duration > 0 else default_duration
                    phases.append(
                        {
                            "label": "Eyes Open",
                            "start": onset,
                            "end": onset + actual_duration,
                            "duration": actual_duration,
                            "original_description": annot["description"],
                        }
                    )
                elif is_eyes_closed:
                    actual_duration = duration if duration > 0 else default_duration
                    phases.append(
                        {
                            "label": "Eyes Closed",
                            "start": onset,
                            "end": onset + actual_duration,
                            "duration": actual_duration,
                            "original_description": annot["description"],
                        }
                    )

            if phases:
                return phases

        # If no standard annotations, check for marker channels
        # Common in Emotiv and other EEG systems
        phases = SignalEditor._detect_phases_from_marker_channels(raw)

        return phases

    @staticmethod
    def _detect_phases_from_marker_channels(raw: mne.io.Raw) -> List[Dict[str, Any]]:
        """
        Detect resting phases from marker channels in EEG data.

        Some EEG systems (like Emotiv) store markers in dedicated channels
        rather than as annotations. This method checks for common marker
        channel patterns.

        Common marker values:
        - MarkerValueInt: 1 = Eyes Open condition, 3 = Eyes Closed condition
        - MarkerIndex: positive = start, negative = end of a phase

        Args:
            raw: Raw EEG data

        Returns:
            List of phase dictionaries
        """
        import numpy as np

        phases = []

        # Check for marker channels
        marker_channels = ["MarkerValueInt", "MarkerIndex", "Marker", "MARKER"]
        available_markers = [ch for ch in marker_channels if ch in raw.ch_names]

        if not available_markers:
            return phases

        # Try MarkerValueInt + MarkerIndex combination (Emotiv style)
        if "MarkerValueInt" in raw.ch_names and "MarkerIndex" in raw.ch_names:
            try:
                marker_val_data = raw.get_data(picks=["MarkerValueInt"])[0]
                marker_idx_data = raw.get_data(picks=["MarkerIndex"])[0]

                # Scale values (they may be stored as microvolts)
                # Detect scaling by checking max absolute value
                max_val = np.max(np.abs(marker_val_data))
                if max_val < 0.001:
                    scale_factor = 1e6
                else:
                    scale_factor = 1

                marker_val_scaled = np.round(marker_val_data * scale_factor).astype(int)
                marker_idx_scaled = np.round(marker_idx_data * scale_factor).astype(int)

                # Find marker events (non-zero values)
                non_zero_idx = np.where(marker_val_scaled != 0)[0]

                # Build event list: (time, marker_value, marker_index)
                events = []
                for idx in non_zero_idx:
                    time = raw.times[idx]
                    val = marker_val_scaled[idx]
                    idx_val = marker_idx_scaled[idx]
                    events.append((time, val, idx_val))

                # Define marker value to label mapping
                # Common convention: 1 = Eyes Open, 3 = Eyes Closed
                # But this can vary, so we'll use a flexible approach
                marker_labels = {}
                unique_vals = set(e[1] for e in events)

                # Assign labels based on marker values
                for val in sorted(unique_vals):
                    if val == 1:
                        marker_labels[val] = "Eyes Open"
                    elif val == 3:
                        marker_labels[val] = "Eyes Closed"
                    elif val == 2:
                        marker_labels[val] = "Eyes Closed"
                    else:
                        marker_labels[val] = f"Condition {val}"

                # Parse events to find phase start/end
                # Positive marker_index = start, negative = end
                active_phases = {}  # marker_value -> start_time

                for time, marker_val, marker_idx in events:
                    if marker_idx > 0:
                        # Start of a phase
                        active_phases[marker_val] = time
                    elif marker_idx < 0:
                        # End of a phase
                        if marker_val in active_phases:
                            start_time = active_phases.pop(marker_val)
                            duration = time - start_time
                            label = marker_labels.get(
                                marker_val, f"Condition {marker_val}"
                            )
                            phases.append(
                                {
                                    "label": label,
                                    "start": start_time,
                                    "end": time,
                                    "duration": duration,
                                    "original_description": f"Marker {marker_val}",
                                }
                            )

                # Handle any phases that started but didn't end
                for marker_val, start_time in active_phases.items():
                    end_time = raw.times[-1]
                    duration = end_time - start_time
                    label = marker_labels.get(marker_val, f"Condition {marker_val}")
                    phases.append(
                        {
                            "label": label,
                            "start": start_time,
                            "end": end_time,
                            "duration": duration,
                            "original_description": f"Marker {marker_val} (unclosed)",
                        }
                    )

            except (ValueError, IndexError, RuntimeError):
                pass

        # Sort phases by start time
        phases.sort(key=lambda x: x["start"])

        return phases

    @staticmethod
    def cut_signal_regions(
        raw: mne.io.Raw,
        regions_to_cut: List[Tuple[float, float]],
    ) -> mne.io.Raw:
        """
        Cut specified regions from the signal and join remaining segments.

        Annotations are preserved and their positions are adjusted to account
        for the removed regions. Annotations that fall entirely within cut
        regions are removed, and annotations that span cut boundaries are
        truncated or split as appropriate.

        Args:
            raw: Raw EEG data
            regions_to_cut: List of (start_time, end_time) tuples in seconds

        Returns:
            New Raw object with cut regions removed and segments joined

        Raises:
            ValueError: If regions are invalid or result in empty signal
        """
        if not regions_to_cut:
            return raw.copy()

        # Sort regions by start time
        sorted_regions = sorted(regions_to_cut, key=lambda x: x[0])

        # Validate regions
        signal_duration = raw.times[-1]
        for start, end in sorted_regions:
            if start < 0 or end > signal_duration:
                raise ValueError(
                    f"Region ({start}, {end}) is outside signal range "
                    f"(0, {signal_duration})"
                )
            if start >= end:
                raise ValueError(
                    f"Invalid region: start ({start}) must be less than end ({end})"
                )

        # Check for overlapping regions
        for i in range(len(sorted_regions) - 1):
            if sorted_regions[i][1] > sorted_regions[i + 1][0]:
                raise ValueError(
                    f"Overlapping regions: {sorted_regions[i]} and "
                    f"{sorted_regions[i + 1]}"
                )

        # Calculate segments to keep
        segments_to_keep = []
        current_start = 0.0

        for cut_start, cut_end in sorted_regions:
            if current_start < cut_start:
                segments_to_keep.append((current_start, cut_start))
            current_start = cut_end

        # Add final segment if needed
        if current_start < signal_duration:
            segments_to_keep.append((current_start, signal_duration))

        if not segments_to_keep:
            raise ValueError("Cannot cut all regions - would result in empty signal")

        # Extract and concatenate segments
        sfreq = raw.info["sfreq"]
        segments_data = []

        for seg_start, seg_end in segments_to_keep:
            start_sample = int(seg_start * sfreq)
            end_sample = int(seg_end * sfreq)
            segment = raw.get_data(start=start_sample, stop=end_sample)
            segments_data.append(segment)

        # Concatenate all segments
        concatenated_data = np.concatenate(segments_data, axis=1)

        # Create new Raw object
        info = raw.info.copy()
        new_raw = mne.io.RawArray(concatenated_data, info, verbose=False)

        # Adjust annotations to account for cut regions
        if raw.annotations and len(raw.annotations) > 0:
            adjusted_annotations = SignalEditor._adjust_annotations_after_cuts(
                raw.annotations, sorted_regions, segments_to_keep
            )
            if adjusted_annotations is not None:
                new_raw.set_annotations(adjusted_annotations)

        return new_raw

    @staticmethod
    def _adjust_annotations_after_cuts(
        annotations: mne.Annotations,
        cut_regions: List[Tuple[float, float]],
        kept_segments: List[Tuple[float, float]],
    ) -> Optional[mne.Annotations]:
        """
        Adjust annotation positions after cutting signal regions.

        Annotations are shifted to account for removed time. Annotations
        that fall entirely within cut regions are removed.

        Args:
            annotations: Original annotations
            cut_regions: List of (start, end) tuples that were cut
            kept_segments: List of (start, end) tuples that were kept

        Returns:
            New Annotations object with adjusted positions, or None if empty
        """
        if not annotations or len(annotations) == 0:
            return None

        new_onsets = []
        new_durations = []
        new_descriptions = []

        # Calculate cumulative time removed before each kept segment
        cumulative_removed = []
        total_removed = 0.0
        for i, (seg_start, seg_end) in enumerate(kept_segments):
            cumulative_removed.append(total_removed)
            if i < len(kept_segments) - 1:
                # Time removed is the gap before next kept segment
                next_seg_start = kept_segments[i + 1][0]
                total_removed += next_seg_start - seg_end

        for annot in annotations:
            onset = annot["onset"]
            duration = annot["duration"]
            description = annot["description"]
            annot_end = onset + duration

            # Find which kept segment(s) this annotation falls into
            new_onset = None
            new_duration = 0.0

            for i, (seg_start, seg_end) in enumerate(kept_segments):
                # Check if annotation overlaps with this segment
                if annot_end <= seg_start:
                    # Annotation is before this segment
                    continue
                if onset >= seg_end:
                    # Annotation is after this segment
                    continue

                # Annotation overlaps with this segment
                # Calculate the portion within this segment
                overlap_start = max(onset, seg_start)
                overlap_end = min(annot_end, seg_end)

                if overlap_start < overlap_end:
                    # Calculate new onset in the concatenated signal
                    time_removed_before = cumulative_removed[i]
                    segment_offset = overlap_start - seg_start

                    # New position = kept segments before this one + offset within this segment
                    kept_time_before = sum(
                        end - start for start, end in kept_segments[:i]
                    )
                    adjusted_onset = kept_time_before + segment_offset

                    if new_onset is None:
                        new_onset = adjusted_onset
                    new_duration += overlap_end - overlap_start

            # Only keep annotation if it has some remaining duration
            if new_onset is not None and new_duration > 0:
                new_onsets.append(new_onset)
                new_durations.append(new_duration)
                new_descriptions.append(description)

        if not new_onsets:
            return None

        return mne.Annotations(
            onset=new_onsets,
            duration=new_durations,
            description=new_descriptions,
        )

    @staticmethod
    def get_time_segment(
        raw: mne.io.Raw,
        start_time: float,
        end_time: float,
    ) -> mne.io.Raw:
        """
        Extract a time segment from the signal.

        Args:
            raw: Raw EEG data
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            New Raw object containing only the specified time segment
        """
        raw_cropped = raw.copy()
        raw_cropped.crop(tmin=start_time, tmax=end_time, include_tmax=True)
        return raw_cropped

    @staticmethod
    def get_annotations_in_range(
        raw: mne.io.Raw,
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Get annotations within a specified time range.

        Args:
            raw: Raw EEG data with annotations
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of annotation dictionaries within the time range
        """
        annotations = []

        if not raw.annotations or len(raw.annotations) == 0:
            return annotations

        for annot in raw.annotations:
            onset = annot["onset"]
            duration = annot["duration"]
            annot_end = onset + duration if duration > 0 else onset

            # Check if annotation overlaps with the time range
            if onset <= end_time and annot_end >= start_time:
                annotations.append(
                    {
                        "description": annot["description"],
                        "onset": onset,
                        "duration": duration,
                    }
                )

        return annotations


class EEGBackendCore:
    """
    Central Backend for EEG Processing.

    Combines data management and preprocessing to provide a unified
    interface for loading, processing, and saving EEG data.

    Attributes:
        data_manager (EEGDataManager): Data manager
        preprocessor (EEGPreprocessor): Data preprocessor
        raw_data (mne.io.Raw): Original data
        filtered_data (mne.io.Raw): Filtered data
        current_file (str): Current file being processed
    """

    def __init__(self):
        """
        Initialize the central backend.

        Creates instances of data manager and preprocessor and
        initializes state variables.
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
        Load and initially process file.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: File path
            selected_channels: List of selected channels (None for auto detection)

        Returns:
            Dictionary with loading information
        """
        try:
            # Load data (supports multiple formats)
            self.raw_data, channels = self.data_manager.load_raw_file(
                file_path, selected_channels
            )
            self.current_file = file_path

            # Apply filter
            self.filtered_data = self.preprocessor.apply_bandpass_filter(self.raw_data)

            # Return information
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

    def load_from_raw(
        self, raw: mne.io.Raw, already_filtered: bool = False
    ) -> Dict[str, Any]:
        """
        Load and process from pre-loaded Raw object.

        Used when signal has been modified before processing (e.g., from
        the Signal Preview screen where filtering and manual cleaning
        have already been applied).

        Args:
            raw: Pre-loaded MNE Raw object
            already_filtered: If True, skip band-pass filtering (data already filtered)

        Returns:
            Dictionary with loading information
        """
        try:
            import warnings

            # Store the raw data
            self.raw_data = raw.copy()
            self.current_file = "modified_signal"

            # Ensure data is loaded
            if not self.raw_data.preload:
                self.raw_data.load_data()

            # Set montage for topographic visualization (same as load_raw_file)
            try:
                self.raw_data.set_montage("standard_1020", on_missing="warn")
            except (ValueError, KeyError, RuntimeError) as e:
                # If montage fails, continue without it
                warnings.warn(f"Unable to set montage: {str(e)}", UserWarning)

            # Apply filter only if not already filtered
            if already_filtered:
                # Data is already preprocessed (filtered), use it directly
                self.filtered_data = self.raw_data.copy()
            else:
                # Apply band-pass filter
                self.filtered_data = self.preprocessor.apply_bandpass_filter(
                    self.raw_data
                )

            # Return information
            return {
                "success": True,
                "channels": list(self.raw_data.ch_names),
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
        Get file information without loading data.

        Supports: .edf, .bdf, .fif, .csv, .set

        Args:
            file_path: File path

        Returns:
            Dictionary with file information
        """
        try:
            return self.data_manager.load_file_info(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_cleaned_data(self, cleaned_raw: mne.io.Raw, output_path: str) -> bool:
        """
        Save cleaned data.

        Supports: .edf, .fif, .csv

        Args:
            cleaned_raw (mne.io.Raw): The cleaned data
            output_path (str): Output file path

        Returns:
            bool: True if saving was successful
        """
        return self.data_manager.save_cleaned_data(cleaned_raw, output_path)

    def get_filtered_data(self) -> Optional[mne.io.Raw]:
        """
        Return filtered data.

        Returns:
            Optional[mne.io.Raw]: The filtered data or None if not available
        """
        return self.filtered_data

    def get_original_data(self) -> Optional[mne.io.Raw]:
        """
        Return original data.

        Returns:
            Optional[mne.io.Raw]: The original data or None if not available
        """
        return self.raw_data
