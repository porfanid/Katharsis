#!/usr/bin/env python3
"""
Katharsis Backend API - Autonomous EEG Analysis Package
====================================================

Complete backend API for EEG analysis that operates independently of any UI framework.
This package provides all EEG processing functionality through a clean, unified interface.

Features:
- Multi-format EEG data loading (EDF, BDF, FIF, CSV, SET)
- Advanced preprocessing with filtering, re-referencing, and channel management
- ICA analysis with multiple algorithms and automatic artifact detection
- Time-domain analysis and ERP computation
- Comprehensive error handling and validation
- Complete independence from PyQt6 or any GUI framework

Author: porfanid
Version: 4.0 - Complete Backend Separation
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

import mne
import numpy as np

# Import all backend modules
from .eeg_service import EEGArtifactCleaningService
from .eeg_backend import EEGBackendCore, EEGDataManager, EEGPreprocessor
from .ica_processor import ICAProcessor
from .artifact_detector import ArtifactDetector
from .enhanced_ica_processor import EnhancedICAProcessor, ICAMethod
from .enhanced_artifact_detector import EnhancedArtifactDetector, ArtifactType
from .preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig
from .epoching_processor import EpochingProcessor, EpochingConfig, BaselineCorrectionMethod
from .erp_analyzer import ERPAnalyzer, ERPConfig
from .time_domain_visualizer import TimeDomainVisualizer
from .data_consistency_utils import validate_raw_consistency, fix_raw_consistency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress MNE warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class KatharsisBackend:
    """
    Unified Backend API for Katharsis EEG Analysis
    
    This class provides a complete, autonomous backend for EEG analysis that operates
    independently of any frontend framework. All business logic is contained here,
    and the frontend only needs to call methods and display results.
    
    Key Features:
    - Complete data processing pipeline
    - Multiple analysis modes (ICA, time-domain, frequency-domain)
    - Comprehensive error handling with Greek localization
    - Progress tracking and status updates
    - Cross-platform compatibility
    - Extensive validation and consistency checking
    """
    
    def __init__(self):
        """Initialize the backend with all processing components"""
        # Core components
        self.data_manager = EEGDataManager()
        self.preprocessor = EEGPreprocessor()
        self.cleaning_service = EEGArtifactCleaningService()
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.epoching_processor = EpochingProcessor()
        self.erp_analyzer = ERPAnalyzer()
        self.time_domain_visualizer = TimeDomainVisualizer()
        
        # Enhanced processors
        self.enhanced_ica = EnhancedICAProcessor()
        self.enhanced_artifact_detector = EnhancedArtifactDetector()
        
        # State management
        self.current_file_path: Optional[str] = None
        self.raw_data: Optional[mne.io.Raw] = None
        self.filtered_data: Optional[mne.io.Raw] = None
        self.preprocessed_data: Optional[mne.io.Raw] = None
        self.ica_data: Optional[Dict] = None
        self.epochs_data: Optional[mne.Epochs] = None
        self.erp_data: Optional[Dict] = None
        
        # Processing state
        self.is_processing = False
        self.processing_stage = ""
        self.progress_percentage = 0
        
        # Callbacks for progress/status updates
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        
        # File info cache
        self.file_info: Dict[str, Any] = {}
        
    def set_callbacks(self, 
                     progress_callback: Optional[Callable[[int], None]] = None,
                     status_callback: Optional[Callable[[str], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """Set callback functions for progress and status updates"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.error_callback = error_callback
        
        # Set callbacks on service components
        self.cleaning_service.set_progress_callback(progress_callback)
        self.cleaning_service.set_status_callback(status_callback)
    
    def _update_progress(self, percentage: int):
        """Update progress and notify callbacks"""
        self.progress_percentage = percentage
        if self.progress_callback:
            self.progress_callback(percentage)
    
    def _update_status(self, status: str):
        """Update status and notify callbacks"""
        self.processing_stage = status
        logger.info(f"Status: {status}")
        if self.status_callback:
            self.status_callback(status)
    
    def _handle_error(self, error: str):
        """Handle errors and notify callbacks"""
        logger.error(f"Error: {error}")
        if self.error_callback:
            self.error_callback(error)
    
    # === File Operations ===
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an EEG file and return comprehensive information
        
        Args:
            file_path: Path to the EEG file
            
        Returns:
            Dict containing validation results and file information
        """
        try:
            file_path = str(Path(file_path).resolve())
            
            # Check file exists
            if not Path(file_path).exists():
                return {
                    "valid": False,
                    "error": f"Το αρχείο δεν βρέθηκε: {file_path}",
                    "details": "File not found"
                }
            
            # Validate based on extension
            ext = Path(file_path).suffix.lower()
            
            if ext == '.edf':
                info = self.data_manager.validate_edf_file(file_path)
            elif ext in ['.fif', '.fiff']:
                info = self._validate_fif_file(file_path)
            elif ext == '.csv':
                info = self._validate_csv_file(file_path)
            elif ext == '.set':
                info = self._validate_set_file(file_path)
            else:
                return {
                    "valid": False,
                    "error": f"Μη υποστηριζόμενος τύπος αρχείου: {ext}",
                    "details": f"Unsupported file type: {ext}"
                }
            
            # Cache file info if valid
            if info.get("valid", False):
                self.file_info[file_path] = info
            
            return info
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Σφάλμα επικύρωσης αρχείου: {str(e)}",
                "details": str(e)
            }
    
    def load_file(self, file_path: str, selected_channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load EEG file and prepare for analysis
        
        Args:
            file_path: Path to the EEG file
            selected_channels: Optional list of channels to load
            
        Returns:
            Dict containing load results and data information
        """
        try:
            self._update_status("Φόρτωση αρχείου...")
            self._update_progress(10)
            
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "details": validation.get("details", "")
                }
            
            # Load based on file type
            ext = Path(file_path).suffix.lower()
            
            if ext == '.edf':
                raw, available_channels = self.data_manager.load_edf_file(file_path)
            elif ext in ['.fif', '.fiff']:
                raw, available_channels = self._load_fif_file(file_path)
            elif ext == '.csv':
                raw, available_channels = self._load_csv_file(file_path)
            elif ext == '.set':
                raw, available_channels = self._load_set_file(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Μη υποστηριζόμενος τύπος αρχείου: {ext}"
                }
            
            # Apply channel selection if specified
            if selected_channels:
                if not all(ch in available_channels for ch in selected_channels):
                    missing = [ch for ch in selected_channels if ch not in available_channels]
                    return {
                        "success": False,
                        "error": f"Κανάλια δεν βρέθηκαν: {missing}"
                    }
                raw.pick_channels(selected_channels)
                available_channels = selected_channels
            
            # Validate data consistency
            consistency = validate_raw_consistency(raw)
            if not consistency["valid"]:
                self._update_status("Επιδιόρθωση συνέπειας δεδομένων...")
                raw, fix_info = fix_raw_consistency(raw, strategy='auto')
                if fix_info["fixed"]:
                    self._handle_error(f"Προειδοποίηση: {fix_info['message']}")
            
            # Store data
            self.current_file_path = file_path
            self.raw_data = raw
            self.filtered_data = None
            self.preprocessed_data = None
            
            self._update_progress(30)
            self._update_status("Αρχείο φορτώθηκε επιτυχώς")
            
            return {
                "success": True,
                "channels": available_channels,
                "duration": raw.n_times / raw.info['sfreq'],
                "sampling_rate": raw.info['sfreq'],
                "n_channels": len(raw.ch_names),
                "file_path": file_path
            }
            
        except Exception as e:
            error_msg = f"Σφάλμα φόρτωσης αρχείου: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    def get_available_channels(self) -> List[str]:
        """Get list of available channels from loaded data"""
        if self.raw_data is None:
            return []
        return self.raw_data.ch_names.copy()
    
    def get_eeg_channels(self) -> List[str]:
        """Get list of EEG channels from loaded data"""
        if self.raw_data is None:
            return []
        return self.data_manager.detect_eeg_channels(self.raw_data)
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded file"""
        if self.raw_data is None:
            return {}
        
        return {
            "file_path": self.current_file_path,
            "n_channels": len(self.raw_data.ch_names),
            "channels": self.raw_data.ch_names.copy(),
            "eeg_channels": self.get_eeg_channels(),
            "sampling_rate": self.raw_data.info['sfreq'],
            "duration": self.raw_data.n_times / self.raw_data.info['sfreq'],
            "n_samples": self.raw_data.n_times,
            "data_shape": self.raw_data.get_data().shape
        }
    
    # === Preprocessing ===
    
    def apply_preprocessing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply advanced preprocessing pipeline
        
        Args:
            config: Preprocessing configuration dictionary
            
        Returns:
            Dict containing preprocessing results
        """
        try:
            if self.raw_data is None:
                return {
                    "success": False,
                    "error": "Δεν έχει φορτωθεί αρχείο"
                }
            
            self._update_status("Εφαρμογή προεπεξεργασίας...")
            self._update_progress(40)
            
            # Apply basic preprocessing using the existing preprocessor
            result_data = self.raw_data.copy()
            applied_steps = []
            processing_info = {}
            
            # Apply filtering if requested
            if "filtering" in config:
                filter_config = config["filtering"]
                
                if filter_config.get("enable_bandpass", False):
                    low_freq = filter_config.get("low_freq", 1.0)
                    high_freq = filter_config.get("high_freq", 40.0)
                    
                    result_data.filter(
                        low_freq, high_freq,
                        picks='eeg',
                        verbose=False
                    )
                    applied_steps.append(f"Bandpass filter: {low_freq}-{high_freq} Hz")
                
                if filter_config.get("enable_notch", False):
                    notch_freqs = filter_config.get("notch_freqs", [50.0])
                    result_data.notch_filter(
                        notch_freqs,
                        picks='eeg',
                        verbose=False
                    )
                    applied_steps.append(f"Notch filter: {notch_freqs} Hz")
            
            # Apply re-referencing if requested
            if "referencing" in config:
                ref_config = config["referencing"]
                
                if ref_config.get("enable", False):
                    ref_type = ref_config.get("type", "average")
                    
                    if ref_type == "average":
                        result_data.set_eeg_reference('average', projection=False, verbose=False)
                        applied_steps.append("Average reference")
                    elif ref_type == "common":
                        ref_channels = ref_config.get("channels", [])
                        if ref_channels:
                            result_data.set_eeg_reference(ref_channels, verbose=False)
                            applied_steps.append(f"Common reference: {ref_channels}")
            
            # Store results
            self.preprocessed_data = result_data
            self.filtered_data = self.preprocessed_data  # For compatibility
            
            self._update_progress(60)
            self._update_status("Προεπεξεργασία ολοκληρώθηκε")
            
            return {
                "success": True,
                "applied_steps": applied_steps,
                "processing_info": processing_info,
                "data_shape": self.preprocessed_data.get_data().shape
            }
                
        except Exception as e:
            error_msg = f"Σφάλμα προεπεξεργασίας: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    # === ICA Analysis ===
    
    def perform_ica_analysis(self, 
                           ica_method: str = "fastica",
                           n_components: Optional[int] = None,
                           max_iter: int = 200) -> Dict[str, Any]:
        """
        Perform ICA analysis with enhanced algorithms
        
        Args:
            ica_method: ICA algorithm to use
            n_components: Number of components (auto if None)
            max_iter: Maximum iterations
            
        Returns:
            Dict containing ICA analysis results
        """
        try:
            # Use preprocessed data if available, otherwise filtered, otherwise raw
            data_to_use = self.preprocessed_data or self.filtered_data or self.raw_data
            
            if data_to_use is None:
                return {
                    "success": False,
                    "error": "Δεν υπάρχουν δεδομένα για ανάλυση"
                }
            
            self._update_status("Εκπαίδευση μοντέλου ICA...")
            self._update_progress(70)
            
            # Update enhanced ICA processor configuration
            if hasattr(self.enhanced_ica, 'config'):
                # Map string to enum manually
                method_map = {
                    "fastica": ICAMethod.FASTICA,
                    "extended_infomax": ICAMethod.EXTENDED_INFOMAX,
                    "picard": ICAMethod.PICARD,
                    "mne_default": ICAMethod.MNE_DEFAULT
                }
                self.enhanced_ica.config.method = method_map.get(ica_method, ICAMethod.FASTICA)
                if n_components is not None:
                    self.enhanced_ica.config.n_components = n_components
                self.enhanced_ica.config.max_iter = max_iter
            
            # Perform ICA analysis
            result = self.enhanced_ica.fit_ica(data_to_use)
            
            if result["success"]:
                self.ica_data = {
                    "ica": self.enhanced_ica.ica,
                    "n_components": result.get("n_components", 0),
                    "component_info": result.get("component_info", {}),
                }
                
                # Perform automatic artifact detection if we have the enhanced detector
                try:
                    artifact_result = self.enhanced_artifact_detector.detect_artifacts(
                        self.enhanced_ica.ica, data_to_use
                    )
                    self.ica_data["artifact_detection"] = artifact_result
                except Exception as e:
                    # Fallback to basic artifact detection
                    self.ica_data["artifact_detection"] = {
                        "suggested_components": [],
                        "artifact_types": {},
                        "error": f"Artifact detection failed: {str(e)}"
                    }
                
                self._update_progress(90)
                self._update_status("ICA ανάλυση ολοκληρώθηκε")
                
                return {
                    "success": True,
                    "n_components": result.get("n_components", 0),
                    "method": ica_method,
                    "suggested_artifacts": self.ica_data["artifact_detection"].get("suggested_components", []),
                    "artifact_types": self.ica_data["artifact_detection"].get("artifact_types", {}),
                    "component_info": result.get("component_info", {}),
                    "mixing_matrix": getattr(self.enhanced_ica.ica, 'mixing_', None),
                    "unmixing_matrix": getattr(self.enhanced_ica.ica, 'unmixing_', None)
                }
            else:
                self._handle_error(f"Σφάλμα ICA: {result['error']}")
                return result
                
        except Exception as e:
            error_msg = f"Σφάλμα ICA ανάλυσης: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    def get_ica_components_data(self) -> Optional[Dict[str, Any]]:
        """Get ICA components data for visualization"""
        if self.ica_data is None:
            return None
        
        return {
            "ica": self.ica_data["ica"],
            "raw_data": self.preprocessed_data or self.filtered_data or self.raw_data,
            "n_components": self.ica_data["n_components"],
            "component_info": self.ica_data.get("component_info", {}),
            "suggested_artifacts": self.ica_data.get("artifact_detection", {}).get("suggested_components", []),
            "artifact_types": self.ica_data.get("artifact_detection", {}).get("artifact_types", {})
        }
    
    def apply_ica_cleaning(self, components_to_remove: List[int]) -> Dict[str, Any]:
        """
        Apply ICA cleaning by removing specified components
        
        Args:
            components_to_remove: List of component indices to remove
            
        Returns:
            Dict containing cleaning results
        """
        try:
            if self.ica_data is None:
                return {
                    "success": False,
                    "error": "Δεν έχει εκτελεστεί ICA ανάλυση"
                }
            
            self._update_status("Εφαρμογή καθαρισμού ICA...")
            
            # Get the data that was used for ICA
            data_to_clean = self.preprocessed_data or self.filtered_data or self.raw_data
            
            # Apply cleaning
            ica = self.ica_data["ica"]
            ica.exclude = components_to_remove
            cleaned_data = ica.apply(data_to_clean.copy())
            
            self._update_status("Καθαρισμός ολοκληρώθηκε")
            
            return {
                "success": True,
                "cleaned_data": cleaned_data,
                "removed_components": components_to_remove,
                "n_components_removed": len(components_to_remove)
            }
            
        except Exception as e:
            error_msg = f"Σφάλμα καθαρισμού ICA: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    # === Time-Domain Analysis ===
    
    def perform_epoching(self, 
                        events_config: Dict[str, Any],
                        epoch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform epoching for time-domain analysis
        
        Args:
            events_config: Configuration for event detection
            epoch_config: Configuration for epoching
            
        Returns:
            Dict containing epoching results
        """
        try:
            data_to_use = self.preprocessed_data or self.filtered_data or self.raw_data
            
            if data_to_use is None:
                return {
                    "success": False,
                    "error": "Δεν υπάρχουν δεδομένα για epoching"
                }
            
            self._update_status("Εκτέλεση epoching...")
            self._update_progress(70)
            
            # Try to find events
            try:
                events = self.epoching_processor.find_events_from_raw(
                    data_to_use,
                    stim_channel=events_config.get("channels", [None])[0] if events_config.get("channels") else None,
                    min_duration=events_config.get("min_duration", 0.001),
                    threshold=events_config.get("threshold", "auto")
                )
                
                if len(events) == 0:
                    # Fallback to creating events from annotations
                    events, event_dict = self.epoching_processor.create_events_from_annotations(data_to_use)
                    
                    if len(events) == 0:
                        return {
                            "success": False,
                            "error": "Δεν βρέθηκαν events για epoching. Δοκιμάστε διαφορετικές παραμέτρους."
                        }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Σφάλμα εντοπισμού events: {str(e)}"
                }
            
            # Create epochs configuration
            config = EpochingConfig()
            config.tmin = epoch_config.get("tmin", -0.2)
            config.tmax = epoch_config.get("tmax", 0.8)
            config.baseline = epoch_config.get("baseline", (-0.2, 0.0))
            config.preload = True
            
            # Create epochs
            epochs = self.epoching_processor.create_epochs_from_events(
                data_to_use, events, config
            )
            
            if epochs is None or len(epochs) == 0:
                return {
                    "success": False,
                    "error": "Δεν δημιουργήθηκαν epochs. Ελέγξτε τις παραμέτρους."
                }
            
            self.epochs_data = epochs
            
            self._update_progress(90)
            self._update_status("Epoching ολοκληρώθηκε")
            
            return {
                "success": True,
                "n_epochs": len(epochs),
                "n_events": len(events),
                "epoch_info": {
                    "tmin": epochs.tmin,
                    "tmax": epochs.tmax,
                    "baseline": epochs.baseline,
                    "n_channels": len(epochs.ch_names),
                    "sampling_rate": epochs.info['sfreq']
                },
                "events": events,
                "event_info": {"n_events": len(events)}
            }
                
        except Exception as e:
            error_msg = f"Σφάλμα epoching: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    def perform_erp_analysis(self, erp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ERP analysis on epoched data
        
        Args:
            erp_config: Configuration for ERP analysis
            
        Returns:
            Dict containing ERP analysis results
        """
        try:
            if self.epochs_data is None:
                return {
                    "success": False,
                    "error": "Δεν υπάρχουν epoched δεδομένα για ERP ανάλυση"
                }
            
            self._update_status("Εκτέλεση ERP ανάλυσης...")
            self._update_progress(80)
            
            # Create ERP configuration
            config = ERPConfig()
            config.baseline = erp_config.get("baseline", (-0.2, 0.0))
            config.channels = erp_config.get("channels", [])
            config.reject_criteria = erp_config.get("reject_criteria", {})
            
            # Compute ERP using the analyzer
            result = self.erp_analyzer.compute_erp(self.epochs_data, config)
            
            if result["success"]:
                self.erp_data = result
                
                self._update_progress(95)
                self._update_status("ERP ανάλυση ολοκληρώθηκε")
                
                return {
                    "success": True,
                    "evoked": result["evoked"],
                    "grand_average": result.get("grand_average", None),
                    "peak_analysis": result.get("peak_analysis", {}),
                    "statistical_analysis": result.get("statistical_analysis", {}),
                    "erp_info": {
                        "n_averages": result["evoked"].nave,
                        "channels": result["evoked"].ch_names,
                        "times": result["evoked"].times,
                        "baseline": result["evoked"].baseline
                    }
                }
            else:
                self._handle_error(f"Σφάλμα ERP ανάλυσης: {result['error']}")
                return result
                
        except Exception as e:
            error_msg = f"Σφάλμα ERP ανάλυσης: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    # === Data Export ===
    
    def export_data(self, output_path: str, data_type: str = "cleaned") -> Dict[str, Any]:
        """
        Export processed data to file
        
        Args:
            output_path: Path to save the data
            data_type: Type of data to export ("raw", "preprocessed", "cleaned", "epochs", "erp")
            
        Returns:
            Dict containing export results
        """
        try:
            self._update_status(f"Εξαγωγή {data_type} δεδομένων...")
            
            if data_type == "raw" and self.raw_data is not None:
                data_to_save = self.raw_data
            elif data_type == "preprocessed" and self.preprocessed_data is not None:
                data_to_save = self.preprocessed_data
            elif data_type == "epochs" and self.epochs_data is not None:
                # Export epochs
                self.epochs_data.save(output_path, overwrite=True)
                return {"success": True, "file_path": output_path}
            elif data_type == "erp" and self.erp_data is not None:
                # Export ERP (evoked data)
                self.erp_data["evoked"].save(output_path, overwrite=True)
                return {"success": True, "file_path": output_path}
            else:
                return {
                    "success": False,
                    "error": f"Δεν υπάρχουν δεδομένα τύπου: {data_type}"
                }
            
            # Export raw-like data
            success = self.data_manager.save_cleaned_data(data_to_save, output_path)
            
            if success:
                self._update_status("Εξαγωγή ολοκληρώθηκε")
                return {
                    "success": True,
                    "file_path": output_path
                }
            else:
                return {
                    "success": False,
                    "error": "Αποτυχία εξαγωγής δεδομένων"
                }
                
        except Exception as e:
            error_msg = f"Σφάλμα εξαγωγής: {str(e)}"
            self._handle_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "details": str(e)
            }
    
    # === State Management ===
    
    def get_processing_state(self) -> Dict[str, Any]:
        """Get comprehensive processing state information"""
        return {
            "file_loaded": self.current_file_path is not None,
            "file_path": self.current_file_path,
            "has_raw_data": self.raw_data is not None,
            "has_preprocessed_data": self.preprocessed_data is not None,
            "has_ica_data": self.ica_data is not None,
            "has_epochs_data": self.epochs_data is not None,
            "has_erp_data": self.erp_data is not None,
            "is_processing": self.is_processing,
            "processing_stage": self.processing_stage,
            "progress_percentage": self.progress_percentage,
            "file_info": self.get_file_info()
        }
    
    def reset_analysis(self):
        """Reset all analysis data while keeping the loaded file"""
        self.preprocessed_data = None
        self.ica_data = None
        self.epochs_data = None
        self.erp_data = None
        self.is_processing = False
        self.processing_stage = ""
        self.progress_percentage = 0
    
    def reset_all(self):
        """Reset everything including loaded file"""
        self.current_file_path = None
        self.raw_data = None
        self.filtered_data = None
        self.reset_analysis()
        self.file_info.clear()
    
    # === Helper Methods for Multi-format Support ===
    
    def _validate_fif_file(self, file_path: str) -> Dict[str, Any]:
        """Validate FIF file"""
        try:
            raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
            return {
                "valid": True,
                "channels": raw.ch_names,
                "sampling_rate": raw.info['sfreq'],
                "duration": raw.n_times / raw.info['sfreq'],
                "n_channels": len(raw.ch_names)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Σφάλμα επικύρωσης FIF: {str(e)}"
            }
    
    def _load_fif_file(self, file_path: str) -> Tuple[mne.io.Raw, List[str]]:
        """Load FIF file"""
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        return raw, raw.ch_names
    
    def _validate_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=10)  # Check first 10 rows
            return {
                "valid": True,
                "channels": list(df.columns),
                "n_channels": len(df.columns),
                "estimated_samples": len(df) if len(df) < 1000000 else "Large file"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Σφάλμα επικύρωσης CSV: {str(e)}"
            }
    
    def _load_csv_file(self, file_path: str) -> Tuple[mne.io.Raw, List[str]]:
        """Load CSV file"""
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Assume first column is time or index, rest are channels
        if 'time' in df.columns.str.lower():
            ch_names = [col for col in df.columns if col.lower() != 'time']
            data = df[ch_names].values.T
        else:
            ch_names = list(df.columns)
            data = df.values.T
        
        # Create info structure
        sfreq = 250.0  # Default sampling rate for CSV
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        return raw, ch_names
    
    def _validate_set_file(self, file_path: str) -> Dict[str, Any]:
        """Validate EEGLAB SET file"""
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
            return {
                "valid": True,
                "channels": raw.ch_names,
                "sampling_rate": raw.info['sfreq'],
                "duration": raw.n_times / raw.info['sfreq'],
                "n_channels": len(raw.ch_names)
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Σφάλμα επικύρωσης SET: {str(e)}"
            }
    
    def _load_set_file(self, file_path: str) -> Tuple[mne.io.Raw, List[str]]:
        """Load EEGLAB SET file"""
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        return raw, raw.ch_names