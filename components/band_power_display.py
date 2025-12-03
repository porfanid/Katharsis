#!/usr/bin/env python3
"""
Band Power Display Widget - Display EEG frequency band power percentages
=========================================================================

Widget for displaying the percentage power distribution in each EEG
frequency band (Delta, Theta, Alpha, Beta, Gamma).

Author: porfanid
Version: 1.0
"""

from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class BandPowerBarWidget(QWidget):
    """
    Widget for displaying a single band's power as a progress bar with label.
    """

    def __init__(
        self,
        band_name: str,
        color: str,
        description: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.band_name = band_name
        self.color = color
        self.description = description
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        # Band name label with fixed width
        self.name_label = QLabel(f"{self.band_name}:")
        self.name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.name_label.setFixedWidth(60)
        self.name_label.setStyleSheet(f"color: {self.color};")
        layout.addWidget(self.name_label)

        # Progress bar for power percentage
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
                text-align: center;
                font-weight: bold;
                font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {self.color};
                border-radius: 3px;
            }}
        """
        )
        layout.addWidget(self.progress_bar, 1)

        # Value label with percentage
        self.value_label = QLabel("0.0%")
        self.value_label.setFont(QFont("Arial", 9))
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.value_label)

    def set_value(self, percentage: float):
        """Set the power percentage value (0-100)."""
        percentage = max(0, min(100, percentage))
        self.progress_bar.setValue(int(percentage))
        self.value_label.setText(f"{percentage:.1f}%")


class BandPowerDisplayWidget(QWidget):
    """
    Widget for displaying all EEG band power percentages.

    Shows real-time band power distribution using progress bars
    for each frequency band (Delta, Theta, Alpha, Beta, Gamma).
    """

    # Default colors for each band
    BAND_COLORS = {
        "Delta": "#2E86AB",  # Blue - Deep sleep
        "Theta": "#A23B72",  # Magenta - Light sleep
        "Alpha": "#F18F01",  # Orange - Relaxation
        "Beta": "#C73E1D",  # Red - Focus
        "Gamma": "#6B2737",  # Dark red - Cognition
    }

    # Descriptions for each band
    BAND_DESCRIPTIONS = {
        "Delta": "Deep sleep (0.5-4 Hz)",
        "Theta": "Light sleep (4-8 Hz)",
        "Alpha": "Relaxation (8-12 Hz)",
        "Beta": "Focus (12-30 Hz)",
        "Gamma": "Cognition (30-40 Hz)",
    }

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.theme = theme or {}
        self.band_widgets: Dict[str, BandPowerBarWidget] = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(4)

        # Title
        title_label = QLabel("📊 EEG Frequency Bands")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')}; margin-bottom: 5px;"
        )
        layout.addWidget(title_label)

        # Create bar widgets for each band
        band_order = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        for band_name in band_order:
            color = self.BAND_COLORS.get(band_name, "#666666")
            description = self.BAND_DESCRIPTIONS.get(band_name, "")
            band_widget = BandPowerBarWidget(band_name, color, description)
            band_widget.setToolTip(description)
            self.band_widgets[band_name] = band_widget
            layout.addWidget(band_widget)

    def update_band_powers(self, band_powers: Dict[str, float]):
        """
        Update the display with new band power values.

        Args:
            band_powers: Dictionary mapping band names to percentages (0-100)
        """
        for band_name, widget in self.band_widgets.items():
            power = band_powers.get(band_name, 0.0)
            widget.set_value(power)

    def clear(self):
        """Reset all band power displays to zero."""
        for widget in self.band_widgets.values():
            widget.set_value(0.0)


class BandPowerComparisonWidget(QWidget):
    """
    Widget for comparing band power between original and cleaned signals.

    Shows side-by-side comparison using pie charts or bar plots.
    """

    BAND_COLORS = {
        "Delta": "#2E86AB",
        "Theta": "#A23B72",
        "Alpha": "#F18F01",
        "Beta": "#C73E1D",
        "Gamma": "#6B2737",
    }

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.theme = theme or {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Title
        title_label = QLabel("📊 Frequency Band Comparison")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        layout.addWidget(title_label)

        # Figure for comparison plots
        self.figure = Figure(figsize=(8, 3), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(150)
        layout.addWidget(self.canvas)

        # Initialize with empty plot
        self.show_empty_plot()

    def show_empty_plot(self):
        """Show an empty plot with instructions."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Comparison will appear after selecting components",
            ha="center",
            va="center",
            fontsize=10,
            color=self.theme.get("text_light", "#6c757d"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def update_comparison(
        self,
        original_powers: Dict[str, float],
        cleaned_powers: Dict[str, float],
    ):
        """
        Update the comparison display with original and cleaned band powers.

        Args:
            original_powers: Band power percentages for original signal
            cleaned_powers: Band power percentages for cleaned signal
        """
        self.figure.clear()

        # Create side-by-side bar chart
        ax = self.figure.add_subplot(111)

        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        x = np.arange(len(bands))
        width = 0.35

        original_vals = [original_powers.get(b, 0) for b in bands]
        cleaned_vals = [cleaned_powers.get(b, 0) for b in bands]

        # Create bars
        bars1 = ax.bar(
            x - width / 2,
            original_vals,
            width,
            label="Original",
            color=[self.BAND_COLORS[b] for b in bands],
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            cleaned_vals,
            width,
            label="Cleaned",
            color=[self.BAND_COLORS[b] for b in bands],
            alpha=1.0,
            edgecolor="black",
            linewidth=0.5,
        )

        # Labels and formatting
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, fontsize=9)
        ax.set_ylim(0, max(max(original_vals), max(cleaned_vals)) * 1.2 + 5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars1, original_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.7,
                )

        for bar, val in zip(bars2, cleaned_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()

    def clear(self):
        """Clear the comparison display."""
        self.show_empty_plot()


class BandPowerAnalysisWidget(QWidget):
    """
    Enhanced widget for band power analysis with time range selection.

    Combines time range selection with band power display to allow
    users to analyze frequency content in specific time windows.

    Signals:
        time_range_changed: Emitted when time range changes (start, end)
    """

    time_range_changed = pyqtSignal(float, float)

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the band power analysis widget.

        Args:
            theme: Theme dictionary for styling
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = theme or {}
        self._raw_data = None
        self._cleaned_data = None
        self._max_time = 100.0

        # Import here to avoid circular imports
        from .signal_editor import RestingPhaseDisplay, TimeRangeSelector

        self._time_range_selector_class = TimeRangeSelector
        self._resting_phase_display_class = RestingPhaseDisplay

        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Time range selector
        self.time_range_selector = self._time_range_selector_class(
            theme=self.theme,
            parent=self,
        )
        self.time_range_selector.range_changed.connect(self._on_range_changed)
        layout.addWidget(self.time_range_selector)

        # Band power comparison
        self.band_power_comparison = BandPowerComparisonWidget(
            theme=self.theme,
            parent=self,
        )
        layout.addWidget(self.band_power_comparison)

        # Resting phase display
        self.resting_phase_display = self._resting_phase_display_class(
            theme=self.theme,
            parent=self,
        )
        layout.addWidget(self.resting_phase_display)

    def set_data(
        self,
        original_raw,
        cleaned_raw=None,
    ):
        """
        Set the EEG data for analysis.

        Args:
            original_raw: Original MNE Raw object
            cleaned_raw: Cleaned MNE Raw object (optional)
        """
        self._raw_data = original_raw
        self._cleaned_data = cleaned_raw

        if original_raw is not None:
            self._max_time = original_raw.times[-1]
            self.time_range_selector.set_time_range(0.0, self._max_time)

            # Detect and display resting phases
            self._update_resting_phases()

        self._update_band_powers()

    def _on_range_changed(self, start_time: float, end_time: float):
        """Handle time range change."""
        self._update_band_powers(start_time, end_time)
        self.time_range_changed.emit(start_time, end_time)

    def _update_band_powers(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Update band power display for the selected time range."""
        if self._raw_data is None:
            self.band_power_comparison.clear()
            return

        from backend import BandPowerAnalyzer

        analyzer = BandPowerAnalyzer()

        # Get time range
        if start_time is None or end_time is None:
            start_time, end_time = self.time_range_selector.get_range()

        # Calculate band powers for original signal
        original_powers = analyzer.compute_band_power_for_raw(
            self._raw_data,
            channel_idx=0,
            tmin=start_time,
            tmax=end_time,
        )

        # Calculate band powers for cleaned signal if available
        if self._cleaned_data is not None:
            cleaned_powers = analyzer.compute_band_power_for_raw(
                self._cleaned_data,
                channel_idx=0,
                tmin=start_time,
                tmax=end_time,
            )
        else:
            cleaned_powers = original_powers

        self.band_power_comparison.update_comparison(original_powers, cleaned_powers)

    def _update_resting_phases(self):
        """Update resting phase display."""
        if self._raw_data is None:
            self.resting_phase_display.clear()
            return

        from backend import BandPowerAnalyzer, SignalEditor

        # Detect resting phases
        phases = SignalEditor.detect_resting_phases(self._raw_data)

        if not phases:
            self.resting_phase_display.update_phases([])
            return

        # Calculate band powers for each phase
        analyzer = BandPowerAnalyzer()
        original_powers = {}
        cleaned_powers = {}

        for phase in phases:
            phase_label = phase["label"]
            start = phase["start"]
            end = phase["end"]

            # Original signal powers
            try:
                orig_power = analyzer.compute_band_power_for_raw(
                    self._raw_data,
                    channel_idx=0,
                    tmin=start,
                    tmax=min(end, self._raw_data.times[-1]),
                )
                original_powers[phase_label] = orig_power
            except Exception:
                original_powers[phase_label] = None

            # Cleaned signal powers
            if self._cleaned_data is not None:
                try:
                    clean_power = analyzer.compute_band_power_for_raw(
                        self._cleaned_data,
                        channel_idx=0,
                        tmin=start,
                        tmax=min(end, self._cleaned_data.times[-1]),
                    )
                    cleaned_powers[phase_label] = clean_power
                except Exception:
                    cleaned_powers[phase_label] = None

        self.resting_phase_display.update_phases(
            phases, original_powers, cleaned_powers
        )

    def clear(self):
        """Clear all displays."""
        self._raw_data = None
        self._cleaned_data = None
        self.band_power_comparison.clear()
        self.resting_phase_display.clear()
