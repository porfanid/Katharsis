#!/usr/bin/env python3
"""
Backend Package Initialization
Αρχικοποίηση Backend Package
"""

from .artifact_detector import ArtifactDetector
from .eeg_backend import EEGBackendCore, EEGDataManager, EEGPreprocessor
from .eeg_service import EEGArtifactCleaningService
from .ica_processor import ICAProcessor

# Phase 1 - Advanced Preprocessing
from .filters import EEGFilterProcessor, FilterConfig, FilterPresets
from .referencing import EEGReferenceProcessor, ReferenceConfig, ReferencePresets
from .channel_management import EEGChannelManager, BadChannelDetector, ChannelInterpolator, MontageManager
from .preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig, PreprocessingPresets

__all__ = [
    "EEGBackendCore",
    "EEGDataManager", 
    "EEGPreprocessor",
    "ICAProcessor",
    "ArtifactDetector",
    "EEGArtifactCleaningService",
    # Phase 1 - Advanced Preprocessing
    "EEGFilterProcessor",
    "FilterConfig",
    "FilterPresets",
    "EEGReferenceProcessor", 
    "ReferenceConfig",
    "ReferencePresets",
    "EEGChannelManager",
    "BadChannelDetector",
    "ChannelInterpolator",
    "MontageManager",
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "PreprocessingPresets",
]
