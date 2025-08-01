#!/usr/bin/env python3
"""
Backend Package Initialization
Αρχικοποίηση Backend Package
"""

# Primary unified backend API
from .katharsis_backend import KatharsisBackend

# Legacy individual components (for compatibility)
from .artifact_detector import ArtifactDetector
from .eeg_backend import EEGBackendCore, EEGDataManager, EEGPreprocessor
from .eeg_service import EEGArtifactCleaningService
from .ica_processor import ICAProcessor

# Phase 1 - Advanced Preprocessing
from .filters import EEGFilterProcessor, FilterConfig, FilterPresets
from .referencing import EEGReferenceProcessor, ReferenceConfig, ReferencePresets
from .channel_management import EEGChannelManager, BadChannelDetector, ChannelInterpolator, MontageManager
from .preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig, PreprocessingPresets

# Phase 2 - Enhanced ICA and Artifact Detection
from .enhanced_ica_processor import EnhancedICAProcessor, ICAConfig, ICAMethod
from .enhanced_artifact_detector import EnhancedArtifactDetector, ArtifactType, DetectionConfig

# Phase 3 - Time-Domain Analysis & ERPs
from .epoching_processor import EpochingProcessor, EpochingConfig, SegmentationConfig, BaselineCorrectionMethod, EpochRejectionCriteria
from .erp_analyzer import ERPAnalyzer, ERPConfig, PeakDetectionMethod, ERPComponent, StatisticalTest
from .time_domain_visualizer import TimeDomainVisualizer, PlotConfig, PlotType, LayoutType

# Data consistency utilities
from .data_consistency_utils import validate_raw_consistency, fix_raw_consistency

__all__ = [
    # Primary unified API
    "KatharsisBackend",
    
    # Legacy components (for compatibility)
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
    
    # Phase 2 - Enhanced ICA and Artifact Detection
    "EnhancedICAProcessor",
    "ICAConfig", 
    "ICAMethod",
    "EnhancedArtifactDetector",
    "ArtifactType",
    "DetectionConfig",
    
    # Phase 3 - Time-Domain Analysis & ERPs
    "EpochingProcessor",
    "EpochingConfig",
    "SegmentationConfig", 
    "BaselineCorrectionMethod",
    "EpochRejectionCriteria",
    "ERPAnalyzer",
    "ERPConfig",
    "PeakDetectionMethod",
    "ERPComponent", 
    "StatisticalTest",
    "TimeDomainVisualizer",
    "PlotConfig",
    "PlotType",
    "LayoutType",
    
    # Data consistency utilities
    "validate_raw_consistency",
    "fix_raw_consistency",
]
