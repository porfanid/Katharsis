#!/usr/bin/env python3
"""
Signal Editor Widget - Time range selection and signal cutting tools
====================================================================

Provides widgets for:
- Time range selection with dual sliders for frequency analysis
- Resting phase (eyes open/closed) detection and display
- Manual signal region cutting and joining with visual timeline

Author: porfanid
Version: 1.1
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
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
        title_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
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
        title_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
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
            f"color: {self.theme.get('text_light', '#6c757d')};" "padding: 10px;"
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
                f"color: {self.theme.get('text_light', '#6c757d')};" "padding: 10px;"
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
        time_label.setStyleSheet(f"color: {self.theme.get('text_light', '#6c757d')};")
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


class SignalCutterTimeline(QWidget):
    """
    Visual timeline widget for selecting and displaying multiple cut regions.

    Shows a timeline bar with:
    - Draggable left/right markers for current selection
    - Visual display of all added cut regions
    - Ability to click on existing regions to remove them
    - Display of EEG annotations/labels on the timeline
    """

    # Marker positions changed signal (left_pos, right_pos in seconds)
    markers_changed = pyqtSignal(float, float)
    # Signal when user clicks on an existing region to remove it
    region_clicked = pyqtSignal(int)  # Index of clicked region

    # Color mapping for common annotation types
    ANNOTATION_COLORS = {
        "eyes open": "#28a745",  # Green
        "eyes_open": "#28a745",
        "eyesopen": "#28a745",
        "eo": "#28a745",
        "eyes closed": "#6f42c1",  # Purple
        "eyes_closed": "#6f42c1",
        "eyesclosed": "#6f42c1",
        "ec": "#6f42c1",
    }
    DEFAULT_ANNOTATION_COLOR = "#ffc107"  # Yellow for unknown types

    # Maximum length for annotation labels on timeline
    MAX_LABEL_LENGTH = 15

    # Colors for frequency analysis ranges
    FREQ_RANGE1_COLOR = "#007AFF"  # Blue for Range 1
    FREQ_RANGE2_COLOR = "#fd7e14"  # Orange for Range 2

    def __init__(
        self,
        theme: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
        show_markers: bool = True,
    ):
        super().__init__(parent)
        self.theme = theme or {}
        self._max_time = 100.0
        self._left_marker = 0.0
        self._right_marker = 10.0
        self._cut_regions: List[Tuple[float, float]] = []  # List of (start, end) tuples
        self._annotations: List[Dict] = []  # List of annotation dicts
        self._freq_ranges: List[Tuple[float, float, str]] = (
            []
        )  # List of (start, end, label) tuples
        self._dragging = None  # None, 'left', 'right', or 'region'
        self._drag_offset = 0
        self._show_markers = show_markers  # Whether to show draggable markers

        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        if show_markers:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def set_annotations(self, annotations: List[Dict]):
        """
        Set the list of annotations to display on the timeline.

        Args:
            annotations: List of annotation dicts with keys:
                - onset: Start time in seconds
                - duration: Duration in seconds
                - description: Annotation description/label
        """
        self._annotations = annotations.copy() if annotations else []
        self.update()

    def _get_annotation_color(self, description: str) -> str:
        """Get the color for an annotation based on its description."""
        desc_lower = description.lower().strip()
        return self.ANNOTATION_COLORS.get(desc_lower, self.DEFAULT_ANNOTATION_COLOR)

    def set_frequency_ranges(
        self,
        range1: Optional[Tuple[float, float, str]] = None,
        range2: Optional[Tuple[float, float, str]] = None,
    ):
        """
        Set the frequency analysis ranges to display on the timeline.

        Args:
            range1: Tuple of (start, end, label) for Range 1 (Blue)
            range2: Tuple of (start, end, label) for Range 2 (Orange)
        """
        self._freq_ranges = []
        if range1 is not None:
            self._freq_ranges.append(
                (range1[0], range1[1], range1[2], self.FREQ_RANGE1_COLOR)
            )
        if range2 is not None:
            self._freq_ranges.append(
                (range2[0], range2[1], range2[2], self.FREQ_RANGE2_COLOR)
            )
        self.update()

    def set_cut_regions(self, regions: List[Tuple[float, float]]):
        """Set the list of cut regions to display."""
        self._cut_regions = regions.copy()
        self.update()

    def set_max_time(self, max_time: float):
        """Set the maximum time for the timeline."""
        self._max_time = max(max_time, 1.0)
        if self._right_marker > self._max_time:
            self._right_marker = self._max_time
        if self._left_marker > self._right_marker:
            self._left_marker = max(0, self._right_marker - 1)
        self.update()

    def set_markers(self, left: float, right: float):
        """Set marker positions."""
        self._left_marker = max(0, min(left, self._max_time))
        self._right_marker = max(0, min(right, self._max_time))
        if self._left_marker > self._right_marker:
            self._left_marker, self._right_marker = (
                self._right_marker,
                self._left_marker,
            )
        self.update()
        self.markers_changed.emit(self._left_marker, self._right_marker)

    def get_markers(self) -> Tuple[float, float]:
        """Get current marker positions."""
        return (self._left_marker, self._right_marker)

    def _time_to_x(self, time: float) -> int:
        """Convert time to x position."""
        margin = 20
        usable_width = self.width() - 2 * margin
        return int(margin + (time / self._max_time) * usable_width)

    def _x_to_time(self, x: int) -> float:
        """Convert x position to time."""
        margin = 20
        usable_width = self.width() - 2 * margin
        time = ((x - margin) / usable_width) * self._max_time
        return max(0, min(time, self._max_time))

    def _get_region_at_pos(self, x: int, y: int) -> int:
        """Get index of cut region at position, or -1 if none."""
        timeline_top = 30
        timeline_height = 35

        # Check if y is in the timeline area
        if not (timeline_top <= y <= timeline_top + timeline_height):
            return -1

        for i, (start, end) in enumerate(self._cut_regions):
            left_x = self._time_to_x(start)
            right_x = self._time_to_x(end)
            if left_x <= x <= right_x:
                return i
        return -1

    def paintEvent(self, event):
        """Paint the timeline with all cut regions."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors from theme
        bg_color = QColor(self.theme.get("background", "#FFFFFF"))
        primary = QColor(self.theme.get("primary", "#007AFF"))
        danger = QColor(self.theme.get("danger", "#dc3545"))
        success = QColor(self.theme.get("success", "#28a745"))
        border = QColor(self.theme.get("border", "#dee2e6"))
        text_color = QColor(self.theme.get("text", "#212529"))

        # Background
        painter.fillRect(self.rect(), bg_color)

        margin = 20
        timeline_top = 30
        timeline_height = 35

        # Title
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(
            margin,
            18,
            "Drag markers to select region, then click 'Add Region'. Click existing regions to remove them.",
        )

        # Timeline track background
        track_rect = QRect(
            margin, timeline_top, self.width() - 2 * margin, timeline_height
        )
        painter.setPen(QPen(border, 1))
        painter.setBrush(QBrush(QColor("#f8f9fa")))
        painter.drawRoundedRect(track_rect, 4, 4)

        # Draw annotations/labels on the timeline (similar style to cut regions)
        for annot in self._annotations:
            onset = annot.get("onset", 0)
            duration = annot.get("duration", 0)
            description = annot.get("description", "")

            # Calculate annotation end time
            annot_end = onset + duration if duration > 0 else onset + 1.0

            # Skip annotations completely outside visible range
            # (annotation ends before timeline starts OR starts after timeline ends)
            if annot_end < 0 or onset > self._max_time:
                continue

            # Calculate positions (clip to visible range)
            annot_left_x = self._time_to_x(max(0, onset))
            annot_right_x = self._time_to_x(min(self._max_time, annot_end))

            # Ensure minimum width for visibility
            region_width = annot_right_x - annot_left_x
            if region_width < 4:
                annot_right_x = annot_left_x + 4
                region_width = 4

            # Get color for this annotation type
            base_color = QColor(self._get_annotation_color(description))

            # Draw annotation region with strong visibility (like cut regions)
            annot_rect = QRect(
                annot_left_x,
                timeline_top + 2,
                annot_right_x - annot_left_x,
                timeline_height - 4,
            )

            # Fill with semi-transparent color
            fill_color = QColor(base_color)
            fill_color.setAlpha(150)  # More opaque for better visibility
            painter.setBrush(QBrush(fill_color))
            painter.setPen(
                QPen(base_color.darker(110), 2)
            )  # Solid border like cut regions
            painter.drawRect(annot_rect)

            # Always draw label text - truncate based on available width
            # Truncate description if too long for the region
            if len(description) > self.MAX_LABEL_LENGTH:
                label = description[: self.MAX_LABEL_LENGTH] + "..."
            else:
                label = description

            # Draw label with contrasting color (white text like cut region numbers)
            painter.setPen(QPen(QColor("white")))
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            painter.drawText(annot_rect, Qt.AlignmentFlag.AlignCenter, label)

        # Draw frequency analysis ranges (before cut regions so they appear behind)
        for freq_range in self._freq_ranges:
            start, end, label, color = freq_range
            if start >= end or end < 0 or start > self._max_time:
                continue

            freq_left_x = self._time_to_x(max(0, start))
            freq_right_x = self._time_to_x(min(self._max_time, end))

            freq_rect = QRect(
                freq_left_x,
                timeline_top + 2,
                freq_right_x - freq_left_x,
                timeline_height - 4,
            )

            # Semi-transparent fill
            freq_color = QColor(color)
            freq_color.setAlpha(120)
            painter.setBrush(QBrush(freq_color))
            painter.setPen(QPen(QColor(color).darker(110), 2))
            painter.drawRect(freq_rect)

            # Label with white text
            painter.setPen(QPen(QColor("white")))
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            painter.drawText(freq_rect, Qt.AlignmentFlag.AlignCenter, label)

        # Draw existing cut regions (darker red, already added)
        for i, (start, end) in enumerate(self._cut_regions):
            left_x = self._time_to_x(start)
            right_x = self._time_to_x(end)

            region_rect = QRect(
                left_x, timeline_top + 2, right_x - left_x, timeline_height - 4
            )

            # Darker red for committed regions
            committed_color = QColor(danger)
            committed_color.setAlpha(180)
            painter.setBrush(QBrush(committed_color))
            painter.setPen(QPen(danger.darker(110), 2))
            painter.drawRect(region_rect)

            # Region number label
            painter.setPen(QPen(QColor("white")))
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            label_text = f"#{i+1}"
            painter.drawText(region_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        # Draw current selection and markers only if enabled
        if self._show_markers:
            left_x = self._time_to_x(self._left_marker)
            right_x = self._time_to_x(self._right_marker)

            selection_rect = QRect(
                left_x, timeline_top + 2, right_x - left_x, timeline_height - 4
            )
            selection_color = QColor(primary)
            selection_color.setAlpha(80)
            painter.setBrush(QBrush(selection_color))
            painter.setPen(QPen(primary, 2, Qt.PenStyle.DashLine))
            painter.drawRect(selection_rect)

            # Left marker handle
            painter.setBrush(QBrush(primary))
            painter.setPen(QPen(primary.darker(110), 2))

            # Left triangle marker
            left_points = [
                QPoint(left_x, timeline_top - 5),
                QPoint(left_x - 10, timeline_top - 18),
                QPoint(left_x + 10, timeline_top - 18),
            ]
            painter.drawPolygon(left_points)
            painter.drawLine(
                left_x, timeline_top, left_x, timeline_top + timeline_height
            )

            # Right triangle marker
            right_points = [
                QPoint(right_x, timeline_top - 5),
                QPoint(right_x - 10, timeline_top - 18),
                QPoint(right_x + 10, timeline_top - 18),
            ]
            painter.drawPolygon(right_points)
            painter.drawLine(
                right_x, timeline_top, right_x, timeline_top + timeline_height
            )

        # Time labels at bottom
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Arial", 8))

        # Start (0s)
        painter.drawText(margin, self.height() - 5, "0s")

        # End
        end_text = f"{self._max_time:.0f}s"
        painter.drawText(self.width() - margin - 30, self.height() - 5, end_text)

        # Middle markers
        for frac in [0.25, 0.5, 0.75]:
            x = self._time_to_x(self._max_time * frac)
            painter.drawLine(
                x, timeline_top + timeline_height, x, timeline_top + timeline_height + 5
            )
            time_label = f"{self._max_time * frac:.0f}s"
            painter.drawText(x - 15, self.height() - 5, time_label)

        # Current selection time labels (above markers) - only if markers enabled
        if self._show_markers:
            left_x = self._time_to_x(self._left_marker)
            right_x = self._time_to_x(self._right_marker)

            painter.setPen(QPen(primary))
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))

            left_text = f"{self._left_marker:.1f}s"
            painter.drawText(left_x - 20, timeline_top - 22, left_text)

            right_text = f"{self._right_marker:.1f}s"
            painter.drawText(right_x - 20, timeline_top - 22, right_text)

            # Selection duration in the middle
            duration = self._right_marker - self._left_marker
            duration_text = f"Selection: {duration:.1f}s"
            center_x = (left_x + right_x) // 2
            painter.drawText(
                center_x - 40, timeline_top + timeline_height // 2 + 4, duration_text
            )

    def mousePressEvent(self, event):
        """Handle mouse press for dragging markers or clicking regions."""
        if not self._show_markers:
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        x = event.position().x()
        y = event.position().y()

        # Check if clicking on an existing region
        region_idx = self._get_region_at_pos(int(x), int(y))
        if region_idx >= 0:
            # Emit signal to remove this region
            self.region_clicked.emit(region_idx)
            return

        left_x = self._time_to_x(self._left_marker)
        right_x = self._time_to_x(self._right_marker)

        # Check if clicking on left marker (within 15px)
        if abs(x - left_x) < 15:
            self._dragging = "left"
        # Check if clicking on right marker
        elif abs(x - right_x) < 15:
            self._dragging = "right"
        # Check if clicking in the selection region (to drag both)
        elif left_x < x < right_x:
            self._dragging = "region"
            self._drag_offset = x - left_x

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if not self._show_markers:
            return

        x = event.position().x()
        y = event.position().y()

        if self._dragging == "left":
            new_time = self._x_to_time(x)
            if new_time < self._right_marker - 0.5:  # Minimum 0.5s region
                self._left_marker = new_time
                self.update()
                self.markers_changed.emit(self._left_marker, self._right_marker)

        elif self._dragging == "right":
            new_time = self._x_to_time(x)
            if new_time > self._left_marker + 0.5:  # Minimum 0.5s region
                self._right_marker = new_time
                self.update()
                self.markers_changed.emit(self._left_marker, self._right_marker)

        elif self._dragging == "region":
            new_left_x = x - self._drag_offset
            new_left = self._x_to_time(new_left_x)
            duration = self._right_marker - self._left_marker

            # Keep within bounds
            if new_left < 0:
                new_left = 0
            if new_left + duration > self._max_time:
                new_left = self._max_time - duration

            self._left_marker = new_left
            self._right_marker = new_left + duration
            self.update()
            self.markers_changed.emit(self._left_marker, self._right_marker)
        else:
            # Update cursor based on position
            left_x = self._time_to_x(self._left_marker)
            right_x = self._time_to_x(self._right_marker)

            # Check if over an existing region
            region_idx = self._get_region_at_pos(int(x), int(y))
            if region_idx >= 0:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.setToolTip(f"Click to remove cut region #{region_idx + 1}")
            elif abs(x - left_x) < 15 or abs(x - right_x) < 15:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.setToolTip("Drag to adjust selection")
            elif left_x < x < right_x:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.setToolTip("Drag to move selection")
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.setToolTip("")

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if not self._show_markers:
            return
        self._dragging = None


class SignalCutter(QWidget):
    """
    Widget for manually cutting signal regions with visual timeline.

    Features a video-editor style timeline with draggable markers
    to select regions to remove from the signal.

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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Apply white theme background
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """
        )

        # Title
        title_label = QLabel("✂️ Signal Region Cutter")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('primary', '#007AFF')}; background: transparent;"
        )
        layout.addWidget(title_label)

        # Instructions
        instructions = QLabel(
            "Drag the markers on the timeline below to select a region to cut. "
            "The red area shows what will be removed."
        )
        instructions.setFont(QFont("Arial", 10))
        instructions.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Visual Timeline
        timeline_group = QGroupBox("📍 Cut Region Timeline")
        timeline_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        timeline_group.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.get('primary', '#007AFF')};
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """
        )
        timeline_layout = QVBoxLayout(timeline_group)

        # Timeline widget
        self.timeline = SignalCutterTimeline(theme=self.theme)
        self.timeline.markers_changed.connect(self._on_markers_changed)
        self.timeline.region_clicked.connect(self._on_region_clicked)
        timeline_layout.addWidget(self.timeline)

        # Fine-tune controls
        fine_tune_layout = QHBoxLayout()

        # Start time spinbox
        start_label = QLabel("Start:")
        start_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')}; background: transparent;"
        )
        fine_tune_layout.addWidget(start_label)

        self.start_spinbox = QSpinBox()
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(int(self._max_time))
        self.start_spinbox.setSuffix(" s")
        self.start_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.start_spinbox.setStyleSheet(
            f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 5px;
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        fine_tune_layout.addWidget(self.start_spinbox)

        fine_tune_layout.addSpacing(20)

        # End time spinbox
        end_label = QLabel("End:")
        end_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')}; background: transparent;"
        )
        fine_tune_layout.addWidget(end_label)

        self.end_spinbox = QSpinBox()
        self.end_spinbox.setMinimum(0)
        self.end_spinbox.setMaximum(int(self._max_time))
        self.end_spinbox.setValue(10)
        self.end_spinbox.setSuffix(" s")
        self.end_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.end_spinbox.setStyleSheet(
            f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 5px;
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        fine_tune_layout.addWidget(self.end_spinbox)

        fine_tune_layout.addStretch()
        timeline_layout.addLayout(fine_tune_layout)

        layout.addWidget(timeline_group)

        # Add to cut list button
        add_btn = QPushButton("➕ Add This Region to Cut List")
        add_btn.clicked.connect(self._add_cut_region)
        add_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('danger', '#dc3545')};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """
        )
        layout.addWidget(add_btn)

        # Current regions display
        regions_group = QGroupBox("📋 Regions to Cut")
        regions_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        regions_group.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.get('text', '#212529')};
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """
        )
        regions_layout = QVBoxLayout(regions_group)

        self.regions_label = QLabel(
            "No regions selected yet.\nUse the timeline above to select regions to cut."
        )
        self.regions_label.setFont(QFont("Arial", 10))
        self.regions_label.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;"
        )
        self.regions_label.setWordWrap(True)
        regions_layout.addWidget(self.regions_label)

        layout.addWidget(regions_group)

        # Action buttons
        buttons_layout = QHBoxLayout()

        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_regions)
        clear_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('text_light', '#6c757d')};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
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
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """
        )
        buttons_layout.addWidget(apply_btn)

        layout.addLayout(buttons_layout)

    def _on_markers_changed(self, left: float, right: float):
        """Handle timeline marker changes."""
        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)
        self.start_spinbox.setValue(int(left))
        self.end_spinbox.setValue(int(right))
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)

    def _on_spinbox_changed(self):
        """Handle spinbox value changes."""
        left = float(self.start_spinbox.value())
        right = float(self.end_spinbox.value())
        if left < right:
            self.timeline.set_markers(left, right)

    def set_max_time(self, max_time: float):
        """
        Set the maximum time for the signal.

        Args:
            max_time: Maximum time in seconds
        """
        self._max_time = max_time
        self.timeline.set_max_time(max_time)
        self.start_spinbox.setMaximum(int(max_time))
        self.end_spinbox.setMaximum(int(max_time))
        self.end_spinbox.setValue(min(10, int(max_time)))
        self.timeline.set_markers(0, min(10, max_time))

    def get_cut_regions(self) -> List[Tuple[float, float]]:
        """
        Get the list of cut regions.

        Returns:
            List of (start, end) tuples in seconds
        """
        return self._cut_regions.copy()

    def _add_cut_region(self):
        """Add the current timeline selection as a cut region."""
        start, end = self.timeline.get_markers()

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
        self._update_timeline()
        self.regions_changed.emit(self._cut_regions)

    def _on_region_clicked(self, index: int):
        """Handle click on existing region to remove it."""
        if 0 <= index < len(self._cut_regions):
            removed = self._cut_regions.pop(index)
            self._update_regions_display()
            self._update_timeline()
            self.regions_changed.emit(self._cut_regions)

            # Show brief confirmation
            QMessageBox.information(
                self,
                "Region Removed",
                f"Cut region #{index + 1} ({removed[0]:.1f}s - {removed[1]:.1f}s) has been removed.",
            )

    def clear_regions(self):
        """Clear all cut regions."""
        self._cut_regions = []
        self._update_regions_display()
        self._update_timeline()
        self.regions_changed.emit(self._cut_regions)

    def _update_timeline(self):
        """Update the timeline widget with current cut regions."""
        self.timeline.set_cut_regions(self._cut_regions)

    def _apply_cuts(self):
        """Emit signal to apply cuts."""
        if not self._cut_regions:
            QMessageBox.information(
                self,
                "No Regions",
                "No cut regions have been defined.\n"
                "Use the timeline to select regions, then click 'Add This Region'.",
            )
            return

        self.apply_cuts.emit(self._cut_regions)

    def _update_regions_display(self):
        """Update the regions display label."""
        if not self._cut_regions:
            self.regions_label.setText(
                "No regions selected yet.\n"
                "Use the timeline above to select regions to cut."
            )
            self.regions_label.setStyleSheet(
                f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;"
            )
        else:
            regions_str = "\n".join(
                f"  • Region {i+1}: {start:.1f}s → {end:.1f}s ({end-start:.1f}s)"
                for i, (start, end) in enumerate(self._cut_regions)
            )
            total_cut = sum(end - start for start, end in self._cut_regions)
            self.regions_label.setText(
                f"{regions_str}\n\n" f"📊 Total time to remove: {total_cut:.1f}s"
            )
            self.regions_label.setStyleSheet(
                f"color: {self.theme.get('text', '#212529')}; background: transparent;"
            )
