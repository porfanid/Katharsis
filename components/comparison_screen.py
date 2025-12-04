#!/usr/bin/env python3
"""
Comparison Screen Widget - "Before & After" Visual Comparison
"""

from typing import Any, Dict, List, Optional

import mne
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .band_power_display import BandPowerAnalysisWidget
from .results_display import ResultsDisplayWidget
from .signal_editor import SignalCutter


class ComparisonScreen(QWidget):
    """
    Full screen widget for Before & After comparison

    Includes:
    - Results display with statistics and signal comparison
    - Band power analysis with time range selection
    - Resting phase (eyes open/closed) analysis
    - Signal region cutter for manual cleaning
    """

    # Signal for return to home screen
    return_to_home = pyqtSignal()
    # Signal when user wants to apply signal cuts
    apply_signal_cuts = pyqtSignal(list)

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self._original_data = None
        self._cleaned_data = None
        self.setup_ui()

    def setup_ui(self):
        """Create UI for the comparison screen"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Create tab widget for different analysis views
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 10))

        # Tab 1: Results Display (existing)
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(5, 5, 5, 5)

        self.results_widget = ResultsDisplayWidget()
        results_layout.addWidget(self.results_widget)

        self.tab_widget.addTab(results_tab, "📈 Signal Comparison")

        # Tab 2: Band Power Analysis with Time Range Selection
        band_power_tab = QWidget()
        band_power_layout = QVBoxLayout(band_power_tab)
        band_power_layout.setContentsMargins(5, 5, 5, 5)

        # Scroll area for band power analysis
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.band_power_widget = BandPowerAnalysisWidget(
            theme=self.theme,
            parent=self,
        )
        scroll_area.setWidget(self.band_power_widget)
        band_power_layout.addWidget(scroll_area)

        self.tab_widget.addTab(band_power_tab, "📊 Frequency Analysis")

        # Tab 3: Signal Cutter
        cutter_tab = QWidget()
        cutter_layout = QVBoxLayout(cutter_tab)
        cutter_layout.setContentsMargins(5, 5, 5, 5)

        self.signal_cutter = SignalCutter(
            theme=self.theme,
            parent=self,
        )
        self.signal_cutter.apply_cuts.connect(self._on_apply_cuts)
        cutter_layout.addWidget(self.signal_cutter)
        cutter_layout.addStretch()

        self.tab_widget.addTab(cutter_tab, "✂️ Signal Editor")

        layout.addWidget(self.tab_widget)

        # Button section at bottom
        button_layout = QHBoxLayout()

        # Spacer to push button to center
        button_layout.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        # Return to home button
        self.return_button = QPushButton("🏠 Return to Home / Process New File")
        self.return_button.setMinimumHeight(50)
        self.return_button.setMinimumWidth(400)
        self.return_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.return_button.clicked.connect(self.return_to_home.emit)

        # Apply theme styling
        self.return_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.theme['primary']};
            }}
        """
        )

        button_layout.addWidget(self.return_button)

        # Spacer to keep button centered
        button_layout.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        layout.addLayout(button_layout)

    def update_comparison(
        self,
        original_data: mne.io.Raw,
        cleaned_data: mne.io.Raw,
        original_stats: Dict[str, Dict[str, float]],
        cleaned_stats: Dict[str, Dict[str, float]],
        components_removed: List[int],
        input_file: str = "",
        output_file: str = "",
    ):
        """
        Update comparison screen with cleaning data

        Args:
            original_data: Original EEG data
            cleaned_data: Cleaned EEG data
            original_stats: Statistics of original data
            cleaned_stats: Statistics of cleaned data
            components_removed: List of removed components
            input_file: Input file
            output_file: Output file
        """
        # Store data references
        self._original_data = original_data
        self._cleaned_data = cleaned_data

        # Update results display
        self.results_widget.update_results(
            original_data=original_data,
            cleaned_data=cleaned_data,
            original_stats=original_stats,
            cleaned_stats=cleaned_stats,
            components_removed=components_removed,
            input_file=input_file,
            output_file=output_file,
        )

        # Update band power analysis widget
        self.band_power_widget.set_data(original_data, cleaned_data)

        # Update signal cutter with max time
        if original_data is not None:
            max_time = original_data.times[-1]
            self.signal_cutter.set_max_time(max_time)

    def _on_apply_cuts(self, regions: List):
        """Handle signal cut request."""
        self.apply_signal_cuts.emit(regions)

    def clear_comparison(self):
        """Clear comparison screen"""
        self._original_data = None
        self._cleaned_data = None
        self.results_widget.clear_results()
        self.band_power_widget.clear()
        self.signal_cutter.clear_regions()
