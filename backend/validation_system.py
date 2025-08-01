#!/usr/bin/env python3
"""
Comprehensive Validation System - Katharsis Backend
==================================================

Advanced validation system to prevent analysis failures by checking prerequisites
and data quality before performing any EEG analysis operations.

Features:
- Data quality validation (NaN, infinite values, signal integrity)
- Analysis prerequisites checking (sufficient channels, duration, etc.)
- ICA validation (proper training, component availability)
- Time-domain analysis validation (events, epochs compatibility)
- Greek error messages with actionable guidance

Author: porfanid
Version: 1.0 - Comprehensive Validation Framework
"""

import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

import mne
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    level: ValidationLevel
    message_en: str
    message_gr: str
    suggestion_gr: str
    details: Optional[Dict[str, Any]] = None


class ComprehensiveValidator:
    """
    Comprehensive validation system for EEG analysis operations
    
    Performs thorough checks before any analysis to prevent failures
    and provide actionable feedback to users.
    """
    
    def __init__(self):
        self.validation_history = []
        
    def validate_file_loading(self, file_path: str) -> ValidationResult:
        """
        Validate file can be loaded properly
        
        Args:
            file_path: Path to EEG file
            
        Returns:
            ValidationResult with detailed feedback
        """
        try:
            from pathlib import Path
            path = Path(file_path)
            
            # Check file exists
            if not path.exists():
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"File not found: {file_path}",
                    message_gr=f"Το αρχείο δεν βρέθηκε: {file_path}",
                    suggestion_gr="💡 Ελέγξτε τη διαδρομή του αρχείου και βεβαιωθείτε ότι υπάρχει"
                )
            
            # Check file size
            file_size = path.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="File is empty",
                    message_gr="Το αρχείο είναι κενό",
                    suggestion_gr="💡 Χρησιμοποιήστε ένα έγκυρο αρχείο EEG με δεδομένα"
                )
            
            # Check file extension
            supported_extensions = {'.edf', '.bdf', '.fif', '.csv', '.set'}
            if path.suffix.lower() not in supported_extensions:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message_en=f"Unsupported file extension: {path.suffix}",
                    message_gr=f"Μη υποστηριζόμενη επέκταση αρχείου: {path.suffix}",
                    suggestion_gr=f"💡 Υποστηριζόμενες επεκτάσεις: {', '.join(supported_extensions)}"
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message_en="File validation passed",
                message_gr="Επιτυχής επικύρωση αρχείου",
                suggestion_gr="✅ Το αρχείο είναι έτοιμο για φόρτωση",
                details={"file_size": file_size, "extension": path.suffix}
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"File validation failed: {str(e)}",
                message_gr=f"Αποτυχία επικύρωσης αρχείου: {str(e)}",
                suggestion_gr="💡 Ελέγξτε τη διαδρομή και τα δικαιώματα πρόσβασης στο αρχείο"
            )
    
    def validate_raw_data_quality(self, raw: mne.io.Raw) -> ValidationResult:
        """
        Validate loaded raw data quality
        
        Args:
            raw: MNE Raw object
            
        Returns:
            ValidationResult with data quality assessment
        """
        try:
            data = raw.get_data()
            
            # Check for NaN values
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                nan_percentage = (nan_count / data.size) * 100
                if nan_percentage > 10:
                    return ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message_en=f"Too many NaN values: {nan_percentage:.1f}%",
                        message_gr=f"Πάρα πολλές τιμές NaN: {nan_percentage:.1f}%",
                        suggestion_gr="💡 Δοκιμάστε διαφορετικό αρχείο ή επεξεργαστείτε τα δεδομένα",
                        details={"nan_count": int(nan_count), "nan_percentage": nan_percentage}
                    )
                else:
                    return ValidationResult(
                        passed=True,
                        level=ValidationLevel.WARNING,
                        message_en=f"Some NaN values found: {nan_percentage:.1f}%",
                        message_gr=f"Βρέθηκαν κάποιες τιμές NaN: {nan_percentage:.1f}%",
                        suggestion_gr="⚠️ Παρακολουθήστε τα αποτελέσματα - μπορεί να επηρεαστεί η ανάλυση"
                    )
            
            # Check for infinite values
            inf_count = np.isinf(data).sum()
            if inf_count > 0:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Infinite values found: {inf_count}",
                    message_gr=f"Βρέθηκαν άπειρες τιμές: {inf_count}",
                    suggestion_gr="💡 Τα δεδομένα χρειάζονται προεπεξεργασία για αφαίρεση άπειρων τιμών"
                )
            
            # Check signal amplitude range
            signal_range = np.ptp(data)
            if signal_range < 1e-12:  # Very small signal
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="Signal amplitude too small",
                    message_gr="Το πλάτος σήματος είναι πολύ μικρό",
                    suggestion_gr="💡 Ελέγξτε αν τα δεδομένα έχουν φορτωθεί σωστά"
                )
            
            # Check channel count
            n_channels = len(raw.ch_names)
            if n_channels < 2:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Insufficient channels: {n_channels} (minimum 2 required)",
                    message_gr=f"Ανεπαρκή κανάλια: {n_channels} (απαιτούνται τουλάχιστον 2)",
                    suggestion_gr="💡 Χρησιμοποιήστε αρχείο με περισσότερα κανάλια EEG"
                )
            
            # Check duration
            duration = raw.times[-1] - raw.times[0]
            if duration < 1.0:  # Less than 1 second
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Recording too short: {duration:.2f}s (minimum 1s required)",
                    message_gr=f"Εγγραφή πολύ σύντομη: {duration:.2f}s (απαιτείται τουλάχιστον 1s)",
                    suggestion_gr="💡 Χρησιμοποιήστε εγγραφή με μεγαλύτερη διάρκεια"
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message_en="Data quality validation passed",
                message_gr="Επιτυχής επικύρωση ποιότητας δεδομένων",
                suggestion_gr="✅ Τα δεδομένα είναι κατάλληλα για ανάλυση",
                details={
                    "n_channels": n_channels,
                    "duration": duration,
                    "signal_range": signal_range,
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"Data validation failed: {str(e)}",
                message_gr=f"Αποτυχία επικύρωσης δεδομένων: {str(e)}",
                suggestion_gr="💡 Ελέγξτε αν τα δεδομένα έχουν φορτωθεί σωστά"
            )
    
    def validate_ica_prerequisites(self, raw: mne.io.Raw, n_components: Optional[int] = None) -> ValidationResult:
        """
        Validate prerequisites for ICA analysis
        
        Args:
            raw: MNE Raw object
            n_components: Requested number of components
            
        Returns:
            ValidationResult for ICA feasibility
        """
        try:
            n_channels = len(raw.ch_names)
            duration = raw.times[-1] - raw.times[0]
            
            # Check minimum channels for ICA
            if n_channels < 2:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Insufficient channels for ICA: {n_channels} (minimum 2 required)",
                    message_gr=f"Ανεπαρκή κανάλια για ICA: {n_channels} (απαιτούνται ≥2)",
                    suggestion_gr="💡 Επιλέξτε περισσότερα κανάλια EEG"
                )
            
            # Check component count validity
            if n_components is not None:
                if n_components <= 0:
                    return ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message_en=f"Invalid component count: {n_components}",
                        message_gr=f"Μη έγκυρος αριθμός συνιστωσών: {n_components}",
                        suggestion_gr="💡 Ο αριθμός συνιστωσών πρέπει να είναι θετικός"
                    )
                
                if n_components > n_channels:
                    return ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message_en=f"Too many components: {n_components} > {n_channels} channels",
                        message_gr=f"Πάρα πολλές συνιστώσες: {n_components} > {n_channels} κανάλια",
                        suggestion_gr=f"💡 Μέγιστες συνιστώσες: {n_channels}"
                    )
            
            # Check minimum duration for stable ICA
            min_duration = 4.0  # seconds
            if duration < min_duration:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Insufficient data for ICA: {duration:.1f}s < {min_duration}s",
                    message_gr=f"Ανεπαρκή δεδομένα για ICA: {duration:.1f}s < {min_duration}s",
                    suggestion_gr="💡 Χρησιμοποιήστε εγγραφή με μεγαλύτερη διάρκεια"
                )
            
            # Check data quality for ICA
            data = raw.get_data()
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="Data contains NaN or infinite values",
                    message_gr="Τα δεδομένα περιέχουν τιμές NaN ή άπειρες",
                    suggestion_gr="💡 Εφαρμόστε προεπεξεργασία για καθαρισμό δεδομένων"
                )
            
            # Check signal variance (too low variance can cause ICA issues)
            signal_var = np.var(data, axis=1)
            low_var_channels = np.sum(signal_var < 1e-20)
            if low_var_channels > 0:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message_en=f"Low variance channels detected: {low_var_channels}",
                    message_gr=f"Κανάλια με χαμηλή διακύμανση: {low_var_channels}",
                    suggestion_gr="⚠️ Κάποια κανάλια μπορεί να προκαλέσουν προβλήματα στο ICA"
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message_en="ICA prerequisites validated successfully",
                message_gr="Επιτυχής επικύρωση προαπαιτούμενων ICA",
                suggestion_gr="✅ Τα δεδομένα είναι κατάλληλα για ανάλυση ICA",
                details={
                    "n_channels": n_channels,
                    "duration": duration,
                    "requested_components": n_components,
                    "low_variance_channels": int(low_var_channels)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"ICA validation failed: {str(e)}",
                message_gr=f"Αποτυχία επικύρωσης ICA: {str(e)}",
                suggestion_gr="💡 Ελέγξτε τα δεδομένα και δοκιμάστε ξανά"
            )
    
    def validate_ica_results(self, ica_processor) -> ValidationResult:
        """
        Validate ICA training results
        
        Args:
            ica_processor: Trained ICA processor object
            
        Returns:
            ValidationResult for ICA training success
        """
        try:
            # Check if ICA is fitted
            if not hasattr(ica_processor, 'ica') or ica_processor.ica is None:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="ICA object not found",
                    message_gr="Δεν βρέθηκε αντικείμενο ICA",
                    suggestion_gr="💡 Εκτελέστε πρώτα την εκπαίδευση ICA"
                )
            
            # Check n_components - handle both enhanced and regular ICA processors
            n_components = None
            
            # Try to get n_components from the MNE ICA object (enhanced processor)
            if hasattr(ica_processor.ica, 'n_components_'):
                n_components = ica_processor.ica.n_components_
            # Fallback to processor attribute (regular processor)
            elif hasattr(ica_processor, 'n_components'):
                n_components = ica_processor.n_components
            
            if n_components is None:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="ICA n_components is None",
                    message_gr="Το ICA δεν έχει εκπαιδευτεί σωστά - n_components είναι None",
                    suggestion_gr="💡 Δοκιμάστε να επαναλάβετε την εκπαίδευση ICA με διαφορετικές παραμέτρους"
                )
            
            if n_components <= 0:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Invalid n_components: {n_components}",
                    message_gr=f"Μη έγκυρος αριθμός συνιστωσών: {n_components}",
                    suggestion_gr="💡 Το ICA χρειάζεται επανεκπαίδευση"
                )
            
            # Check if mixing matrix exists and has proper shape
            try:
                mixing_matrix = None
                
                # Try enhanced processor method first
                if hasattr(ica_processor, 'get_mixing_matrix'):
                    mixing_matrix = ica_processor.get_mixing_matrix()
                # Fallback to direct access
                elif hasattr(ica_processor, 'mixing_matrix_'):
                    mixing_matrix = ica_processor.mixing_matrix_
                elif hasattr(ica_processor.ica, 'mixing_'):
                    mixing_matrix = ica_processor.ica.mixing_
                
                if mixing_matrix is None:
                    return ValidationResult(
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message_en="Mixing matrix not available",
                        message_gr="Μη διαθέσιμος πίνακας ανάμιξης",
                        suggestion_gr="⚠️ Ορισμένες λειτουργίες μπορεί να μην είναι διαθέσιμες"
                    )
                
                # Validate mixing matrix shape (should be n_channels x n_components or n_components x n_components)
                if mixing_matrix.shape[1] != n_components and mixing_matrix.shape[0] != n_components:
                    return ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message_en=f"Mixing matrix shape mismatch: {mixing_matrix.shape}",
                        message_gr=f"Ασυμβατότητα σχήματος πίνακα ανάμιξης: {mixing_matrix.shape}",
                        suggestion_gr="💡 Το ICA χρειάζεται επανεκπαίδευση"
                    )
                    
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message_en=f"Cannot validate mixing matrix: {str(e)}",
                    message_gr=f"Αδυναμία επικύρωσης πίνακα ανάμιξης: {str(e)}",
                    suggestion_gr="⚠️ Ορισμένες λειτουργίες μπορεί να μην λειτουργούν σωστά"
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message_en="ICA results validation passed",
                message_gr="Επιτυχής επικύρωση αποτελεσμάτων ICA",
                suggestion_gr="✅ Το ICA έχει εκπαιδευτεί σωστά και είναι έτοιμο για χρήση",
                details={
                    "n_components": n_components,
                    "has_mixing_matrix": mixing_matrix is not None if 'mixing_matrix' in locals() else False
                }
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"ICA results validation failed: {str(e)}",
                message_gr=f"Αποτυχία επικύρωσης αποτελεσμάτων ICA: {str(e)}",
                suggestion_gr="💡 Επανεκπαιδεύστε το ICA και δοκιμάστε ξανά"
            )
    
    def validate_time_domain_prerequisites(self, raw: mne.io.Raw) -> ValidationResult:
        """
        Validate prerequisites for time-domain analysis
        
        Args:
            raw: MNE Raw object
            
        Returns:
            ValidationResult for time-domain analysis feasibility
        """
        try:
            # Check if events or annotations exist
            events = None
            try:
                events = mne.find_events(raw, verbose=False)
            except:
                pass
            
            annotations = raw.annotations
            has_events = events is not None and len(events) > 0
            has_annotations = annotations is not None and len(annotations) > 0
            
            if not has_events and not has_annotations:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en="No events or annotations found",
                    message_gr="Δεν βρέθηκαν γεγονότα ή σημειώσεις",
                    suggestion_gr="💡 Για ανάλυση χρονικού τομέα χρειάζονται γεγονότα ή σημειώσεις"
                )
            
            # Check sufficient data for epoching
            duration = raw.times[-1] - raw.times[0]
            if duration < 2.0:  # Minimum for meaningful epochs
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message_en=f"Insufficient data for epoching: {duration:.1f}s",
                    message_gr=f"Ανεπαρκή δεδομένα για εποχές: {duration:.1f}s",
                    suggestion_gr="💡 Χρησιμοποιήστε εγγραφή με μεγαλύτερη διάρκεια (≥2s)"
                )
            
            # Validate event information
            details = {
                "has_events": has_events,
                "has_annotations": has_annotations,
                "duration": duration
            }
            
            if has_events:
                details["n_events"] = len(events)
                details["unique_event_ids"] = list(np.unique(events[:, 2]))
            
            if has_annotations:
                details["n_annotations"] = len(annotations)
                details["annotation_descriptions"] = list(set(annotations.description))
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message_en="Time-domain prerequisites validated",
                message_gr="Επιτυχής επικύρωση προαπαιτούμενων ανάλυσης χρονικού τομέα",
                suggestion_gr="✅ Τα δεδομένα είναι κατάλληλα για ανάλυση χρονικού τομέα",
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"Time-domain validation failed: {str(e)}",
                message_gr=f"Αποτυχία επικύρωσης χρονικού τομέα: {str(e)}",
                suggestion_gr="💡 Ελέγξτε τα δεδομένα και δοκιμάστε ξανά"
            )
    
    def validate_all_prerequisites(self, 
                                 raw: mne.io.Raw = None, 
                                 ica_processor=None,
                                 analysis_type: str = "general") -> List[ValidationResult]:
        """
        Run comprehensive validation for specified analysis type
        
        Args:
            raw: MNE Raw object (if available)
            ica_processor: ICA processor (if available)
            analysis_type: Type of analysis ("general", "ica", "time_domain")
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        try:
            # Always validate raw data quality if available
            if raw is not None:
                results.append(self.validate_raw_data_quality(raw))
            
            # Analysis-specific validations
            if analysis_type in ["ica", "general"] and raw is not None:
                results.append(self.validate_ica_prerequisites(raw))
                
                # If ICA processor is available, validate its results
                if ica_processor is not None:
                    results.append(self.validate_ica_results(ica_processor))
            
            if analysis_type in ["time_domain", "general"] and raw is not None:
                results.append(self.validate_time_domain_prerequisites(raw))
            
            # Store in history
            self.validation_history.extend(results)
            
            return results
            
        except Exception as e:
            error_result = ValidationResult(
                passed=False,
                level=ValidationLevel.CRITICAL,
                message_en=f"Comprehensive validation failed: {str(e)}",
                message_gr=f"Αποτυχία συνολικής επικύρωσης: {str(e)}",
                suggestion_gr="💡 Επικοινωνήστε με την υποστήριξη για βοήθεια"
            )
            results.append(error_result)
            return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get summary of validation results
        
        Args:
            results: List of validation results
            
        Returns:
            Summary dictionary
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        levels = {}
        for level in ValidationLevel:
            levels[level.value] = sum(1 for r in results if r.level == level)
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "levels": levels,
            "ready_for_analysis": failed == 0,
            "critical_issues": levels.get("critical", 0),
            "error_issues": levels.get("error", 0)
        }