#!/usr/bin/env python3
"""
Frontend Package - Pure PyQt6 User Interface
==========================================

This package contains only UI components that interact with the backend API.
No business logic is contained here - all analysis and processing is handled
by the autonomous backend package.

Author: porfanid
Version: 4.0 - Complete Frontend/Backend Separation
"""

from .main_window import KatharsisMainWindow
from .file_selection_widget import FileSelectionWidget  
from .channel_selection_widget import ChannelSelectionWidget
from .preprocessing_widget import PreprocessingWidget
from .analysis_selection_widget import AnalysisSelectionWidget
from .ica_analysis_widget import ICAAnalysisWidget
from .time_domain_widget import TimeDomainWidget
from .results_widget import ResultsWidget

__all__ = [
    "KatharsisMainWindow",
    "FileSelectionWidget",
    "ChannelSelectionWidget", 
    "PreprocessingWidget",
    "AnalysisSelectionWidget",
    "ICAAnalysisWidget",
    "TimeDomainWidget",
    "ResultsWidget",
]