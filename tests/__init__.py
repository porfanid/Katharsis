#!/usr/bin/env python3
"""
Tests Package Initialization
Αρχικοποίηση Package Tests
"""

# Backend tests - always available
from .test_backend import *

# GUI tests - only import if Qt is available
__all__ = [
    # Backend tests
    'TestEEGDataManager',
    'TestEEGPreprocessor', 
    'TestICAProcessor',
    'TestArtifactDetector',
    'TestEEGArtifactCleaningService',
]

# Try to import GUI tests - skip if Qt is not available
try:
    from .test_components import *
    __all__.extend([
        # GUI tests
        'TestICAComponentSelector',
        'TestResultsDisplayWidget',
        'TestStatisticsTableWidget',
        'TestComponentIntegration'
    ])
except ImportError:
    # Qt not available - skip GUI tests
    pass