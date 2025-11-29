#!/usr/bin/env python3
"""
Results Display Widget - Display cleaning results and statistics
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StatisticsTableWidget(QWidget):
    """Widget for displaying statistics tables"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Create UI"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("📊 Cleaning Results Statistics")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)

        # Statistics table
        self.table = QTableWidget()
        self.table.setStyleSheet(
            """
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                gridline-color: #ecf0f1;
            }
            QTableWidget::item {
                padding: 8px;
                color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """
        )

        layout.addWidget(self.table)

    def update_statistics(
        self,
        original_stats: Dict[str, Dict[str, float]],
        cleaned_stats: Dict[str, Dict[str, float]],
    ):
        """
        Update statistics table

        Args:
            original_stats: Statistics of original data
            cleaned_stats: Statistics of cleaned data
        """
        channels = list(original_stats.keys())

        # Define columns
        headers = [
            "Channel",
            "Original RMS (μV)",
            "Clean RMS (μV)",
            "Reduction (%)",
            "Original Range",
            "Clean Range",
        ]

        self.table.setRowCount(len(channels))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Fill data
        for row, channel in enumerate(channels):
            orig_stats = original_stats[channel]
            clean_stats = cleaned_stats[channel]

            # Calculate noise reduction
            orig_rms = orig_stats["rms"]
            clean_rms = clean_stats["rms"]
            reduction = ((orig_rms - clean_rms) / orig_rms) * 100 if orig_rms > 0 else 0

            # Data for each column
            row_data = [
                channel,
                f"{orig_rms:.1f}",
                f"{clean_rms:.1f}",
                f"{reduction:.1f}%",
                f"{orig_stats['range']:.1f}",
                f"{clean_stats['range']:.1f}",
            ]

            for col, data in enumerate(row_data):
                item = QTableWidgetItem(str(data))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color based on reduction
                if col == 3:  # Reduction column
                    if reduction > 50:
                        item.setBackground(QColor("#d5f4e6"))  # Green for good reduction
                    elif reduction > 25:
                        item.setBackground(
                            QColor("#fff3cd")
                        )  # Yellow for moderate reduction
                    else:
                        item.setBackground(
                            QColor("#f8d7da")
                        )  # Red for low reduction

                self.table.setItem(row, col, item)

        # Adjust columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class ComparisonPlotWidget(QWidget):
    """Widget for before/after comparison visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Create UI"""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("📈 Signal Comparison: Before vs After Cleaning")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_comparison(
        self,
        original_data: mne.io.Raw,
        cleaned_data: mne.io.Raw,
        time_window: float = 10.0,
    ):
        """
        Visualize data comparison

        Args:
            original_data: Original data
            cleaned_data: Cleaned data
            time_window: Time window for visualization (seconds)
        """
        self.figure.clear()

        try:
            # Get data
            orig_data = original_data.get_data() * 1e6  # Convert to μV
            clean_data = cleaned_data.get_data() * 1e6

            # Time axis
            times = original_data.times
            max_samples = int(time_window * original_data.info["sfreq"])
            display_times = times[:max_samples]

            # Channels
            channels = original_data.ch_names
            n_channels = len(channels)

            # Create subplots
            for i, channel in enumerate(channels):
                ax = self.figure.add_subplot(n_channels, 1, i + 1)

                # Data for visualization
                orig_display = orig_data[i, :max_samples]
                clean_display = clean_data[i, :max_samples]

                # Plots
                ax.plot(
                    display_times,
                    orig_display,
                    color="#e74c3c",
                    alpha=0.7,
                    linewidth=1.5,
                    label="Original signal",
                )
                ax.plot(
                    display_times,
                    clean_display,
                    color="#27ae60",
                    alpha=0.8,
                    linewidth=1.5,
                    label="Clean signal",
                )

                # Style
                ax.set_title(
                    f"Channel {channel}", fontsize=10, color="#2c3e50", fontweight="bold"
                )
                ax.set_xlabel(
                    "Time (s)" if i == n_channels - 1 else "",
                    fontsize=9,
                    color="#2c3e50",
                )
                ax.set_ylabel("Amplitude (μV)", fontsize=9, color="#2c3e50")
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8, colors="#2c3e50")

                # Legend only on first plot
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            self.figure.tight_layout()

        except Exception as e:
            # Error plot
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Visualization error: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
            )
            ax.set_title("Visualization Error")

        self.canvas.draw()


class ResultsDisplayWidget(QWidget):
    """Central widget for displaying results"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Create UI"""
        layout = QVBoxLayout(self)

        # Header
        header_label = QLabel("🎯 EEG Artifact Cleaning Results")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_label.setStyleSheet(
            """
            QLabel {
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                margin: 10px;
            }
        """
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Statistics table
        self.statistics_widget = StatisticsTableWidget()
        layout.addWidget(self.statistics_widget)

        # Comparison plot
        self.comparison_widget = ComparisonPlotWidget()
        layout.addWidget(self.comparison_widget)

        # Summary section
        self.summary_widget = self.create_summary_widget()
        layout.addWidget(self.summary_widget)

    def create_summary_widget(self) -> QWidget:
        """Create summary widget"""
        group_box = QGroupBox("📋 Processing Summary")
        group_box.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        group_box.setStyleSheet(
            """
            QGroupBox {
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """
        )

        layout = QGridLayout(group_box)

        # Labels for summary
        self.file_label = QLabel("File: -")
        self.components_label = QLabel("Components removed: -")
        self.avg_reduction_label = QLabel("Average noise reduction: -")
        self.status_label = QLabel("Status: -")

        labels = [
            self.file_label,
            self.components_label,
            self.avg_reduction_label,
            self.status_label,
        ]

        for i, label in enumerate(labels):
            label.setFont(QFont("Arial", 10))
            label.setStyleSheet("color: #34495e; margin: 5px;")
            layout.addWidget(label, i, 0)

        return group_box

    def update_results(
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
        Update results display

        Args:
            original_data: Original data
            cleaned_data: Cleaned data
            original_stats: Statistics of original data
            cleaned_stats: Statistics of cleaned data
            components_removed: Components that were removed
            input_file: Input file
            output_file: Output file
        """
        try:
            # Update statistics
            self.statistics_widget.update_statistics(original_stats, cleaned_stats)

            # Update visualization
            self.comparison_widget.plot_comparison(original_data, cleaned_data)

            # Calculate average reduction
            total_reduction = 0
            channels = list(original_stats.keys())

            for channel in channels:
                orig_rms = original_stats[channel]["rms"]
                clean_rms = cleaned_stats[channel]["rms"]
                if orig_rms > 0:
                    reduction = ((orig_rms - clean_rms) / orig_rms) * 100
                    total_reduction += reduction

            avg_reduction = total_reduction / len(channels) if channels else 0

            # Update summary
            import os

            filename = os.path.basename(input_file) if input_file else "Unknown"

            self.file_label.setText(f"File: {filename}")
            self.components_label.setText(
                f"Components removed: {components_removed}"
            )
            self.avg_reduction_label.setText(
                f"Average noise reduction: {avg_reduction:.1f}%"
            )
            self.status_label.setText("Status: ✅ Cleaning completed successfully")

            # Color status based on result
            if avg_reduction > 50:
                color = "#27ae60"  # Green for excellent result
            elif avg_reduction > 25:
                color = "#f39c12"  # Orange for good result
            else:
                color = "#e74c3c"  # Red for low result

            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        except Exception as e:
            # Display error
            self.status_label.setText(f"Status: ❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def clear_results(self):
        """Clear results display"""
        self.statistics_widget.table.setRowCount(0)
        self.comparison_widget.figure.clear()
        self.comparison_widget.canvas.draw()

        self.file_label.setText("File: -")
        self.components_label.setText("Components removed: -")
        self.avg_reduction_label.setText("Average noise reduction: -")
        self.status_label.setText("Status: -")
        self.status_label.setStyleSheet("color: #34495e;")
