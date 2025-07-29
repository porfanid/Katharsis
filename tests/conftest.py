#!/usr/bin/env python3
"""
pytest configuration and fixtures
Διαμόρφωση και fixtures για pytest
"""

import os
import sys
import pytest


def pytest_configure(config):
    """Configure pytest environment for cross-platform compatibility."""
    # Ensure Qt runs in offscreen mode for headless testing
    if 'QT_QPA_PLATFORM' not in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Suppress Qt debug output
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.xcb.warning=false'
    
    # Configure MNE for testing
    os.environ['MNE_LOGGING_LEVEL'] = 'WARNING'
    os.environ['MNE_USE_CUDA'] = 'false'
    
    # Ensure reproducible results
    os.environ['PYTHONHASHSEED'] = '0'


@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Set up the test environment before running tests."""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Configure matplotlib for headless testing
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    yield
    
    # Cleanup after all tests
    pass


@pytest.fixture(scope='session')
def qapp():
    """Create a QApplication instance for GUI testing."""
    try:
        from PyQt6.QtWidgets import QApplication
        
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
            
        yield app
        
        # Don't quit the app as it might be shared
    except ImportError:
        pytest.skip("PyQt6 not available")


@pytest.fixture
def temp_dir(tmpdir):
    """Provide a temporary directory for test files."""
    return tmpdir