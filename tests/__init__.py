#!/usr/bin/env python3
"""
Tests Package Initialization
Αρχικοποίηση Package Tests
"""

# Test imports for easy access
from .test_backend import *
from .test_components import *

__all__ = [
    # Backend tests
    'TestEEGDataManager',
    'TestEEGPreprocessor', 
    'TestICAProcessor',
    'TestArtifactDetector',
    'TestEEGArtifactCleaningService',
    
    # GUI tests
    'TestICAComponentSelector',
    'TestResultsDisplayWidget',
    'TestStatisticsTableWidget',
    'TestComponentIntegration'
]