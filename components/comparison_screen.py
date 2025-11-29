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
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from .results_display import ResultsDisplayWidget


class ComparisonScreen(QWidget):
    """
    Full screen widget for Before & After comparison
    """

    # Signal for return to home screen
    return_to_home = pyqtSignal()

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        """Create UI for the comparison screen"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Results display widget (existing component)
        self.results_widget = ResultsDisplayWidget()
        layout.addWidget(self.results_widget)

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
        self.results_widget.update_results(
            original_data=original_data,
            cleaned_data=cleaned_data,
            original_stats=original_stats,
            cleaned_stats=cleaned_stats,
            components_removed=components_removed,
            input_file=input_file,
            output_file=output_file,
        )

    def clear_comparison(self):
        """Clear comparison screen"""
        self.results_widget.clear_results()
