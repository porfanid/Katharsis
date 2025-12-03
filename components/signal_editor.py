#!/usr/bin/env python3
"""
Signal Editor Widget - Time range selection and signal cutting tools
====================================================================

Provides widgets for:
- Time range selection with dual sliders for frequency analysis
- Resting phase (eyes open/closed) detection and display
- Manual signal region cutting and joining

Author: porfanid
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TimeRangeSelector(QWidget):
    """
    Widget for selecting a time range with dual sliders.

    Allows users to select a start and end time for frequency band analysis.

    Signals:
        range_changed: Emitted when the time range changes (start_time, end_time)
    """

    range_changed = pyqtSignal(float, float)

    # Minimum interval between start and end times
    MIN_INTERVAL = 0.1
    # Default slider resolution (steps per second)
    DEFAULT_RESOLUTION = 100

    def __init__(
        self,
        min_time: float = 0.0,
        max_time: float = 100.0,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the time range selector.

        Args:
            min_time: Minimum time value in seconds
            max_time: Maximum time value in seconds
            theme: Theme dictionary for styling
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = theme or {}
        self._min_time = min_time
        self._max_time = max_time
        # Adjust resolution for longer signals to avoid excessive slider steps
        if max_time > 1000:
            self._resolution = 10  # 10 steps per second for long signals
        elif max_time > 100:
            self._resolution = 50  # 50 steps per second for medium signals
        else:
            self._resolution = self.DEFAULT_RESOLUTION

        # Current selection
        self._start_time = min_time
        self._end_time = max_time

        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("📊 Time Range Selection")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')};"
        )
        layout.addWidget(title_label)

        # Start time controls
        start_layout = QHBoxLayout()
        start_label = QLabel("Start:")
        start_label.setFont(QFont("Arial", 10))
        start_label.setFixedWidth(50)
        start_layout.addWidget(start_label)

        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_slider.setMinimum(0)
        self.start_slider.setMaximum(int(self._max_time * self._resolution))
        self.start_slider.setValue(0)
        self.start_slider.valueChanged.connect(self._on_start_slider_changed)
        start_layout.addWidget(self.start_slider)

        self.start_spinbox = QSpinBox()
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(int(self._max_time))
        self.start_spinbox.setValue(0)
        self.start_spinbox.setSuffix(" s")
        self.start_spinbox.setFixedWidth(80)
        self.start_spinbox.valueChanged.connect(self._on_start_spinbox_changed)
        start_layout.addWidget(self.start_spinbox)

        layout.addLayout(start_layout)

        # End time controls
        end_layout = QHBoxLayout()
        end_label = QLabel("End:")
        end_label.setFont(QFont("Arial", 10))
        end_label.setFixedWidth(50)
        end_layout.addWidget(end_label)

        self.end_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_slider.setMinimum(0)
        self.end_slider.setMaximum(int(self._max_time * self._resolution))
        self.end_slider.setValue(int(self._max_time * self._resolution))
        self.end_slider.valueChanged.connect(self._on_end_slider_changed)
        end_layout.addWidget(self.end_slider)

        self.end_spinbox = QSpinBox()
        self.end_spinbox.setMinimum(0)
        self.end_spinbox.setMaximum(int(self._max_time))
        self.end_spinbox.setValue(int(self._max_time))
        self.end_spinbox.setSuffix(" s")
        self.end_spinbox.setFixedWidth(80)
        self.end_spinbox.valueChanged.connect(self._on_end_spinbox_changed)
        end_layout.addWidget(self.end_spinbox)

        layout.addLayout(end_layout)

        # Duration label
        self.duration_label = QLabel(f"Duration: {self._max_time:.1f} s")
        self.duration_label.setFont(QFont("Arial", 9))
        self.duration_label.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')};"
        )
        layout.addWidget(self.duration_label)

        # Reset button
        reset_btn = QPushButton("Reset to Full Range")
        reset_btn.clicked.connect(self.reset_range)
        reset_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """
        )
        layout.addWidget(reset_btn)

    def set_time_range(self, min_time: float, max_time: float):
        """
        Set the available time range.

        Args:
            min_time: Minimum time in seconds
            max_time: Maximum time in seconds
        """
        self._min_time = min_time
        self._max_time = max_time

        # Update slider ranges
        max_val = int(max_time * self._resolution)
        self.start_slider.setMaximum(max_val)
        self.end_slider.setMaximum(max_val)

        # Update spinbox ranges
        self.start_spinbox.setMaximum(int(max_time))
        self.end_spinbox.setMaximum(int(max_time))

        # Reset selection to full range
        self.reset_range()

    def reset_range(self):
        """Reset selection to full range."""
        self._start_time = self._min_time
        self._end_time = self._max_time

        # Update UI without triggering signals
        self.start_slider.blockSignals(True)
        self.end_slider.blockSignals(True)
        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)

        self.start_slider.setValue(0)
        self.end_slider.setValue(int(self._max_time * self._resolution))
        self.start_spinbox.setValue(0)
        self.end_spinbox.setValue(int(self._max_time))

        self.start_slider.blockSignals(False)
        self.end_slider.blockSignals(False)
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)

        self._update_duration_label()
        self.range_changed.emit(self._start_time, self._end_time)

    def get_range(self) -> Tuple[float, float]:
        """
        Get the currently selected time range.

        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        return (self._start_time, self._end_time)

    def _on_start_slider_changed(self, value: int):
        """Handle start slider value change."""
        self._start_time = value / self._resolution

        # Ensure start < end
        if self._start_time >= self._end_time:
            self._start_time = self._end_time - self.MIN_INTERVAL
            self.start_slider.blockSignals(True)
            self.start_slider.setValue(int(self._start_time * self._resolution))
            self.start_slider.blockSignals(False)

        # Update spinbox
        self.start_spinbox.blockSignals(True)
        self.start_spinbox.setValue(int(self._start_time))
        self.start_spinbox.blockSignals(False)

        self._update_duration_label()
        self.range_changed.emit(self._start_time, self._end_time)

    def _on_end_slider_changed(self, value: int):
        """Handle end slider value change."""
        self._end_time = value / self._resolution

        # Ensure end > start
        if self._end_time <= self._start_time:
            self._end_time = self._start_time + self.MIN_INTERVAL
            self.end_slider.blockSignals(True)
            self.end_slider.setValue(int(self._end_time * self._resolution))
            self.end_slider.blockSignals(False)

        # Update spinbox
        self.end_spinbox.blockSignals(True)
        self.end_spinbox.setValue(int(self._end_time))
        self.end_spinbox.blockSignals(False)

        self._update_duration_label()
        self.range_changed.emit(self._start_time, self._end_time)

    def _on_start_spinbox_changed(self, value: int):
        """Handle start spinbox value change."""
        self._start_time = float(value)

        # Ensure start < end
        if self._start_time >= self._end_time:
            self._start_time = self._end_time - 1.0
            self.start_spinbox.blockSignals(True)
            self.start_spinbox.setValue(int(self._start_time))
            self.start_spinbox.blockSignals(False)

        # Update slider
        self.start_slider.blockSignals(True)
        self.start_slider.setValue(int(self._start_time * self._resolution))
        self.start_slider.blockSignals(False)

        self._update_duration_label()
        self.range_changed.emit(self._start_time, self._end_time)

    def _on_end_spinbox_changed(self, value: int):
        """Handle end spinbox value change."""
        self._end_time = float(value)

        # Ensure end > start
        if self._end_time <= self._start_time:
            self._end_time = self._start_time + 1.0
            self.end_spinbox.blockSignals(True)
            self.end_spinbox.setValue(int(self._end_time))
            self.end_spinbox.blockSignals(False)

        # Update slider
        self.end_slider.blockSignals(True)
        self.end_slider.setValue(int(self._end_time * self._resolution))
        self.end_slider.blockSignals(False)

        self._update_duration_label()
        self.range_changed.emit(self._start_time, self._end_time)

    def _update_duration_label(self):
        """Update the duration display label."""
        duration = self._end_time - self._start_time
        self.duration_label.setText(f"Duration: {duration:.1f} s")


class RestingPhaseDisplay(QWidget):
    """
    Widget for displaying resting phase (eyes open/closed) analysis.

    Shows frequency band percentages for detected resting phases
    in both original and cleaned signals.
    """

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the resting phase display.

        Args:
            theme: Theme dictionary for styling
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = theme or {}
        self.phases: List[Dict] = []
        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("👁️ Resting Phase Analysis")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')};"
        )
        layout.addWidget(title_label)

        # Container for phase displays
        self.phases_container = QVBoxLayout()
        layout.addLayout(self.phases_container)

        # No phases message
        self.no_phases_label = QLabel(
            "No resting phase annotations detected.\n"
            "Look for 'eyes open' or 'eyes closed' markers."
        )
        self.no_phases_label.setFont(QFont("Arial", 9))
        self.no_phases_label.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')};"
            "padding: 10px;"
        )
        self.no_phases_label.setWordWrap(True)
        self.phases_container.addWidget(self.no_phases_label)

    def update_phases(
        self,
        phases: List[Dict],
        original_powers: Optional[Dict[str, Dict[str, float]]] = None,
        cleaned_powers: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Update the display with detected resting phases.

        Args:
            phases: List of phase dictionaries with keys:
                - label: Phase label (e.g., "Eyes Open", "Eyes Closed")
                - start: Start time in seconds
                - end: End time in seconds
            original_powers: Band power for each phase in original signal
            cleaned_powers: Band power for each phase in cleaned signal
        """
        self.phases = phases

        # Clear existing displays
        while self.phases_container.count():
            item = self.phases_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not phases:
            self.no_phases_label = QLabel(
                "No resting phase annotations detected.\n"
                "Look for 'eyes open' or 'eyes closed' markers."
            )
            self.no_phases_label.setFont(QFont("Arial", 9))
            self.no_phases_label.setStyleSheet(
                f"color: {self.theme.get('text_light', '#6c757d')};"
                "padding: 10px;"
            )
            self.no_phases_label.setWordWrap(True)
            self.phases_container.addWidget(self.no_phases_label)
            return

        # Create display for each phase
        for i, phase in enumerate(phases):
            phase_widget = self._create_phase_widget(
                phase,
                original_powers.get(phase["label"]) if original_powers else None,
                cleaned_powers.get(phase["label"]) if cleaned_powers else None,
            )
            self.phases_container.addWidget(phase_widget)

    def _create_phase_widget(
        self,
        phase: Dict,
        original_power: Optional[Dict[str, float]],
        cleaned_power: Optional[Dict[str, float]],
    ) -> QWidget:
        """Create a widget for displaying a single phase's analysis."""
        # Determine icon based on phase label
        label_lower = phase["label"].lower()
        if "open" in label_lower:
            icon = "👁️"
        elif "close" in label_lower or "closed" in label_lower:
            icon = "😌"
        else:
            icon = "📊"

        group = QGroupBox(f"{icon} {phase['label']}")
        group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        group.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: {self.theme.get('primary', '#007AFF')};
            }}
        """
        )

        layout = QVBoxLayout(group)

        # Time range
        time_label = QLabel(
            f"Time: {phase['start']:.1f}s - {phase['end']:.1f}s "
            f"(Duration: {phase['end'] - phase['start']:.1f}s)"
        )
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')};"
        )
        layout.addWidget(time_label)

        # Band powers
        if original_power or cleaned_power:
            bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
            band_colors = {
                "Delta": "#2E86AB",
                "Theta": "#A23B72",
                "Alpha": "#F18F01",
                "Beta": "#C73E1D",
                "Gamma": "#6B2737",
            }

            for band in bands:
                band_layout = QHBoxLayout()

                band_label = QLabel(f"{band}:")
                band_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                band_label.setStyleSheet(f"color: {band_colors.get(band, '#333')};")
                band_label.setFixedWidth(50)
                band_layout.addWidget(band_label)

                if original_power:
                    orig_val = original_power.get(band, 0)
                    orig_label = QLabel(f"Original: {orig_val:.1f}%")
                    orig_label.setFont(QFont("Arial", 9))
                    orig_label.setFixedWidth(100)
                    band_layout.addWidget(orig_label)

                if cleaned_power:
                    clean_val = cleaned_power.get(band, 0)
                    clean_label = QLabel(f"Cleaned: {clean_val:.1f}%")
                    clean_label.setFont(QFont("Arial", 9))
                    clean_label.setStyleSheet(
                        f"color: {self.theme.get('success', '#28a745')};"
                    )
                    clean_label.setFixedWidth(100)
                    band_layout.addWidget(clean_label)

                band_layout.addStretch()
                layout.addLayout(band_layout)
        else:
            no_data_label = QLabel("No band power data available")
            no_data_label.setFont(QFont("Arial", 9))
            no_data_label.setStyleSheet(
                f"color: {self.theme.get('text_light', '#6c757d')};"
            )
            layout.addWidget(no_data_label)

        return group

    def clear(self):
        """Clear all phase displays."""
        self.update_phases([])


class SignalCutter(QWidget):
    """
    Widget for manually cutting signal regions.

    Allows users to select regions to remove and join the remaining signal.

    Signals:
        regions_changed: Emitted when cut regions change
        apply_cuts: Emitted when user requests to apply the cuts
    """

    regions_changed = pyqtSignal(list)  # List of (start, end) tuples
    apply_cuts = pyqtSignal(list)

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the signal cutter.

        Args:
            theme: Theme dictionary for styling
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = theme or {}
        self._max_time = 100.0
        self._cut_regions: List[Tuple[float, float]] = []

        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Title
        title_label = QLabel("✂️ Signal Region Cutter")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')};"
        )
        layout.addWidget(title_label)

        # Instructions
        instructions = QLabel(
            "Select regions to remove from the signal.\n"
            "The remaining segments will be joined together."
        )
        instructions.setFont(QFont("Arial", 9))
        instructions.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')};"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Add region controls
        add_region_group = QGroupBox("Add Cut Region")
        add_region_group.setFont(QFont("Arial", 10))
        add_layout = QVBoxLayout(add_region_group)

        # Start time
        start_layout = QHBoxLayout()
        start_label = QLabel("Start:")
        start_label.setFixedWidth(40)
        start_layout.addWidget(start_label)

        self.cut_start_spinbox = QSpinBox()
        self.cut_start_spinbox.setMinimum(0)
        self.cut_start_spinbox.setMaximum(int(self._max_time))
        self.cut_start_spinbox.setSuffix(" s")
        start_layout.addWidget(self.cut_start_spinbox)

        add_layout.addLayout(start_layout)

        # End time
        end_layout = QHBoxLayout()
        end_label = QLabel("End:")
        end_label.setFixedWidth(40)
        end_layout.addWidget(end_label)

        self.cut_end_spinbox = QSpinBox()
        self.cut_end_spinbox.setMinimum(0)
        self.cut_end_spinbox.setMaximum(int(self._max_time))
        self.cut_end_spinbox.setValue(10)
        self.cut_end_spinbox.setSuffix(" s")
        end_layout.addWidget(self.cut_end_spinbox)

        add_layout.addLayout(end_layout)

        # Add button
        add_btn = QPushButton("➕ Add Cut Region")
        add_btn.clicked.connect(self._add_cut_region)
        add_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('danger', '#dc3545')};
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """
        )
        add_layout.addWidget(add_btn)

        layout.addWidget(add_region_group)

        # Current regions display
        self.regions_label = QLabel("Cut regions: None")
        self.regions_label.setFont(QFont("Arial", 9))
        self.regions_label.setWordWrap(True)
        layout.addWidget(self.regions_label)

        # Action buttons
        buttons_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_regions)
        clear_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('text_light', '#6c757d')};
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #5a6268;
            }}
        """
        )
        buttons_layout.addWidget(clear_btn)

        apply_btn = QPushButton("✂️ Apply Cuts")
        apply_btn.clicked.connect(self._apply_cuts)
        apply_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """
        )
        buttons_layout.addWidget(apply_btn)

        layout.addLayout(buttons_layout)

    def set_max_time(self, max_time: float):
        """
        Set the maximum time for the signal.

        Args:
            max_time: Maximum time in seconds
        """
        self._max_time = max_time
        self.cut_start_spinbox.setMaximum(int(max_time))
        self.cut_end_spinbox.setMaximum(int(max_time))
        self.cut_end_spinbox.setValue(min(10, int(max_time)))

    def get_cut_regions(self) -> List[Tuple[float, float]]:
        """
        Get the list of cut regions.

        Returns:
            List of (start, end) tuples in seconds
        """
        return self._cut_regions.copy()

    def _add_cut_region(self):
        """Add a new cut region."""
        start = float(self.cut_start_spinbox.value())
        end = float(self.cut_end_spinbox.value())

        if start >= end:
            QMessageBox.warning(
                self,
                "Invalid Region",
                "Start time must be less than end time.",
            )
            return

        # Check for overlapping regions
        for existing_start, existing_end in self._cut_regions:
            if not (end <= existing_start or start >= existing_end):
                QMessageBox.warning(
                    self,
                    "Overlapping Region",
                    f"This region overlaps with an existing cut "
                    f"({existing_start:.1f}s - {existing_end:.1f}s).",
                )
                return

        self._cut_regions.append((start, end))
        self._cut_regions.sort(key=lambda x: x[0])  # Sort by start time
        self._update_regions_display()
        self.regions_changed.emit(self._cut_regions)

    def clear_regions(self):
        """Clear all cut regions."""
        self._cut_regions = []
        self._update_regions_display()
        self.regions_changed.emit(self._cut_regions)

    def _apply_cuts(self):
        """Emit signal to apply cuts."""
        if not self._cut_regions:
            QMessageBox.information(
                self,
                "No Regions",
                "No cut regions have been defined.",
            )
            return

        self.apply_cuts.emit(self._cut_regions)

    def _update_regions_display(self):
        """Update the regions display label."""
        if not self._cut_regions:
            self.regions_label.setText("Cut regions: None")
        else:
            regions_str = ", ".join(
                f"[{start:.1f}s - {end:.1f}s]"
                for start, end in self._cut_regions
            )
            total_cut = sum(end - start for start, end in self._cut_regions)
            self.regions_label.setText(
                f"Cut regions: {regions_str}\n"
                f"Total time to remove: {total_cut:.1f}s"
            )
