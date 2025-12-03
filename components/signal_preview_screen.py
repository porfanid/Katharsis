#!/usr/bin/env python3
"""
Signal Preview Screen - Preview and edit signal before processing
==================================================================

Provides a screen where users can:
- Preview the loaded EEG signal per electrode
- Select time ranges for frequency analysis per electrode
- Detect and view resting phases (eyes open/closed)
- Cut signal regions manually before artifact removal (per electrode view)

Author: porfanid
Version: 2.0
"""

from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy.ndimage import binary_dilation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .band_power_display import BandPowerComparisonWidget
from backend import BandPowerAnalyzer, SignalEditor


class SignalEditingHelpDialog(QDialog):
    """
    Help dialog explaining how to use signal editing features.
    """

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("📖 Signal Editing Help")
        self.setMinimumSize(600, 500)
        self.setup_ui()

    def setup_ui(self):
        """Create the help dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("📖 How to Edit Your EEG Signal")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.theme.get('primary', '#007AFF')};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Help content
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(False)
        help_text.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 8px;
                padding: 15px;
                font-size: 12px;
            }}
        """
        )

        help_content = """
        <h2>🎯 Overview</h2>
        <p>Each electrode has its own tab where you can:</p>
        <ul>
            <li><b>View the signal</b> with navigation and zoom controls</li>
            <li><b>See resting phases</b> (eyes open/closed) highlighted on the plot</li>
            <li><b>Cut regions</b> using draggable markers on the timeline</li>
            <li><b>Analyze frequency bands</b> for selected time ranges</li>
        </ul>

        <h2>✂️ How to Cut Signal Regions</h2>
        <ol>
            <li><b>Navigate to the electrode tab</b> you want to edit</li>
            <li><b>Use the timeline markers</b> to select the region to cut:
                <ul>
                    <li>Drag the <b>left marker</b> to set the start time</li>
                    <li>Drag the <b>right marker</b> to set the end time</li>
                    <li>Or use the spinboxes for precise control</li>
                </ul>
            </li>
            <li><b>Click 'Add Region'</b> to mark the region for cutting</li>
            <li><b>Repeat</b> for any additional regions</li>
            <li><b>Click 'Apply Cuts'</b> to remove all marked regions</li>
        </ol>

        <h2>📊 Frequency Analysis</h2>
        <p>Use the time range selector to analyze frequency bands for specific portions:</p>
        <ul>
            <li><b>Delta (1-4 Hz)</b>: Deep sleep, unconscious states</li>
            <li><b>Theta (4-8 Hz)</b>: Light sleep, relaxation, meditation</li>
            <li><b>Alpha (8-13 Hz)</b>: Relaxed wakefulness, eyes closed</li>
            <li><b>Beta (13-30 Hz)</b>: Active thinking, concentration</li>
            <li><b>Gamma (30-100 Hz)</b>: Higher cognitive functions</li>
        </ul>

        <h2>💡 Tips</h2>
        <ul>
            <li>Cuts apply to <b>all electrodes</b> simultaneously</li>
            <li>Use 'Reset to Original' to undo all cuts</li>
            <li>Check annotations for eyes open/closed markers</li>
            <li>Review each electrode before cutting</li>
        </ul>
        """

        help_text.setHtml(help_content)
        layout.addWidget(help_text)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """
        )
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)


class ElectrodeSignalWidget(QWidget):
    """
    Combined signal visualization and cutting widget for a single electrode.
    
    Shows:
    - Signal plot with navigation
    - Annotations (eyes open/closed) on the plot
    - Cut region timeline overlaid on signal
    - Frequency analysis for the electrode
    """
    
    # Signals
    cut_region_added = pyqtSignal(float, float)  # start, end
    cut_region_removed = pyqtSignal(int)  # index
    
    def __init__(
        self,
        channel_name: str,
        channel_idx: int,
        theme: Dict[str, str],
        parent=None
    ):
        super().__init__(parent)
        self.theme = theme
        self.channel_name = channel_name
        self.channel_idx = channel_idx
        self._raw_data = None
        self._view_start = 0.0
        self._view_window = 10.0
        self._max_time = 100.0
        self._cut_regions: List[Tuple[float, float]] = []
        self._current_selection = (0.0, 10.0)  # Current marker positions
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Apply white theme background
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme.get('background', '#FFFFFF')};
                color: {self.theme.get('text', '#212529')};
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet(f"""
            QSplitter {{
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QSplitter::handle {{
                background-color: {self.theme.get('border', '#dee2e6')};
                height: 4px;
            }}
        """)
        
        # === Top section: Signal plot with cutting ===
        signal_widget = QWidget()
        signal_widget.setStyleSheet(f"background-color: {self.theme.get('background', '#FFFFFF')};")
        signal_layout = QVBoxLayout(signal_widget)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        signal_layout.setSpacing(5)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel(f"📈 {self.channel_name} Signal")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.theme.get('primary', '#007AFF')}; background: transparent;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # View window selector
        view_label = QLabel("View:")
        view_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')}; background: transparent;")
        header_layout.addWidget(view_label)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["5s", "10s", "30s", "60s", "Full"])
        self.view_combo.setCurrentText("10s")
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        self.view_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 60px;
                color: {self.theme.get('text', '#212529')};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: white;
                color: {self.theme.get('text', '#212529')};
                selection-background-color: {self.theme.get('primary', '#007AFF')};
                selection-color: white;
            }}
        """)
        header_layout.addWidget(self.view_combo)
        
        signal_layout.addLayout(header_layout)
        
        # Signal plot with matplotlib - set white background
        self.figure = Figure(figsize=(12, 3), dpi=80, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(180)
        signal_layout.addWidget(self.canvas)
        
        # Navigation slider
        nav_layout = QHBoxLayout()
        
        self.nav_label = QLabel("Position: 0.0s")
        self.nav_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')}; background: transparent;")
        self.nav_label.setFixedWidth(100)
        nav_layout.addWidget(self.nav_label)
        
        self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        self.nav_slider.setMinimum(0)
        self.nav_slider.setMaximum(1000)
        self.nav_slider.setValue(0)
        self.nav_slider.valueChanged.connect(self._on_nav_changed)
        self.nav_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                height: 6px;
                background: #f8f9fa;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self.theme.get('primary', '#007AFF')};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
        """)
        nav_layout.addWidget(self.nav_slider)
        
        self.duration_label = QLabel("/ 0.0s")
        self.duration_label.setStyleSheet(f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;")
        self.duration_label.setFixedWidth(70)
        nav_layout.addWidget(self.duration_label)
        
        signal_layout.addLayout(nav_layout)
        
        # === Cut region selection with visual timeline ===
        cut_group = QGroupBox("✂️ Cut Region Selection - Drag markers to select region")
        cut_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        cut_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('danger', '#dc3545')};
                border-radius: 6px;
                margin-top: 8px;
                padding: 10px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.get('danger', '#dc3545')};
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QLabel {{
                color: {self.theme.get('text', '#212529')};
                background: transparent;
            }}
        """)
        cut_layout = QVBoxLayout(cut_group)
        
        # Visual timeline with markers
        from .signal_editor import SignalCutterTimeline
        self.cut_timeline = SignalCutterTimeline(theme=self.theme)
        self.cut_timeline.markers_changed.connect(self._on_markers_changed)
        self.cut_timeline.region_clicked.connect(self._on_timeline_region_clicked)
        cut_layout.addWidget(self.cut_timeline)
        
        # Button row
        button_layout = QHBoxLayout()
        
        # Selection info label
        self.selection_info = QLabel("Selection: 0.0s - 10.0s (10.0s)")
        self.selection_info.setStyleSheet(f"color: {self.theme.get('text', '#212529')}; background: transparent;")
        button_layout.addWidget(self.selection_info)
        
        button_layout.addStretch()
        
        # Add region button
        add_btn = QPushButton("➕ Add Selected Region")
        add_btn.clicked.connect(self._add_cut_region)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('danger', '#dc3545')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """)
        button_layout.addWidget(add_btn)
        
        cut_layout.addLayout(button_layout)
        
        # Current cut regions display
        self.regions_label = QLabel("No cut regions defined. Drag the markers above to select a region.")
        self.regions_label.setStyleSheet(f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;")
        self.regions_label.setWordWrap(True)
        cut_layout.addWidget(self.regions_label)
        
        signal_layout.addWidget(cut_group)
        
        splitter.addWidget(signal_widget)
        
        # === Bottom section: Frequency analysis ===
        freq_widget = QWidget()
        freq_widget.setStyleSheet(f"background-color: {self.theme.get('background', '#FFFFFF')};")
        freq_layout = QVBoxLayout(freq_widget)
        freq_layout.setContentsMargins(5, 5, 5, 5)
        
        freq_header = QLabel("📊 Frequency Band Analysis")
        freq_header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        freq_header.setStyleSheet(f"color: {self.theme.get('primary', '#007AFF')}; background: transparent;")
        freq_layout.addWidget(freq_header)
        
        # === Two time range selectors for frequency comparison ===
        comparison_layout = QHBoxLayout()
        
        # Range 1 (Original/Left)
        range1_group = QGroupBox("📊 Range 1 (Blue)")
        range1_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('primary', '#007AFF')};
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.get('primary', '#007AFF')};
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """)
        range1_layout = QHBoxLayout(range1_group)
        
        range1_layout.addWidget(QLabel("From:"))
        self.freq_start1_spin = QSpinBox()
        self.freq_start1_spin.setMinimum(0)
        self.freq_start1_spin.setMaximum(int(self._max_time))
        self.freq_start1_spin.setSuffix(" s")
        self.freq_start1_spin.valueChanged.connect(self._update_frequency_analysis)
        self.freq_start1_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px;
                color: {self.theme.get('text', '#212529')};
            }}
        """)
        range1_layout.addWidget(self.freq_start1_spin)
        
        range1_layout.addWidget(QLabel("to"))
        self.freq_end1_spin = QSpinBox()
        self.freq_end1_spin.setMinimum(0)
        self.freq_end1_spin.setMaximum(int(self._max_time))
        self.freq_end1_spin.setValue(int(self._max_time / 2))
        self.freq_end1_spin.setSuffix(" s")
        self.freq_end1_spin.valueChanged.connect(self._update_frequency_analysis)
        self.freq_end1_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px;
                color: {self.theme.get('text', '#212529')};
            }}
        """)
        range1_layout.addWidget(self.freq_end1_spin)
        
        comparison_layout.addWidget(range1_group)
        
        # Range 2 (Comparison/Right)
        range2_group = QGroupBox("📊 Range 2 (Orange)")
        range2_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid #fd7e14;
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #fd7e14;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """)
        range2_layout = QHBoxLayout(range2_group)
        
        range2_layout.addWidget(QLabel("From:"))
        self.freq_start2_spin = QSpinBox()
        self.freq_start2_spin.setMinimum(0)
        self.freq_start2_spin.setMaximum(int(self._max_time))
        self.freq_start2_spin.setValue(int(self._max_time / 2))
        self.freq_start2_spin.setSuffix(" s")
        self.freq_start2_spin.valueChanged.connect(self._update_frequency_analysis)
        self.freq_start2_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px;
                color: {self.theme.get('text', '#212529')};
            }}
        """)
        range2_layout.addWidget(self.freq_start2_spin)
        
        range2_layout.addWidget(QLabel("to"))
        self.freq_end2_spin = QSpinBox()
        self.freq_end2_spin.setMinimum(0)
        self.freq_end2_spin.setMaximum(int(self._max_time))
        self.freq_end2_spin.setValue(int(self._max_time))
        self.freq_end2_spin.setSuffix(" s")
        self.freq_end2_spin.valueChanged.connect(self._update_frequency_analysis)
        self.freq_end2_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px;
                color: {self.theme.get('text', '#212529')};
            }}
        """)
        range2_layout.addWidget(self.freq_end2_spin)
        
        comparison_layout.addWidget(range2_group)
        
        freq_layout.addLayout(comparison_layout)
        
        # Voltage threshold input with auto-detect button
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("⚡ Voltage Threshold:")
        threshold_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')}; background: transparent;")
        threshold_layout.addWidget(threshold_label)
        
        self.voltage_threshold_spin = QSpinBox()
        self.voltage_threshold_spin.setMinimum(1)
        self.voltage_threshold_spin.setMaximum(500)
        self.voltage_threshold_spin.setValue(100)
        self.voltage_threshold_spin.setSuffix(" μV")
        self.voltage_threshold_spin.setToolTip("Signals exceeding this threshold will be marked for removal")
        self.voltage_threshold_spin.valueChanged.connect(self._update_plot)
        self.voltage_threshold_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 4px;
                min-width: 80px;
                color: {self.theme.get('text', '#212529')};
            }}
        """)
        threshold_layout.addWidget(self.voltage_threshold_spin)
        
        # Auto-detect artifacts button
        auto_detect_btn = QPushButton("🔍 Auto-detect Artifacts")
        auto_detect_btn.clicked.connect(self._auto_detect_artifacts)
        auto_detect_btn.setToolTip("Automatically mark regions exceeding the voltage threshold for cutting")
        auto_detect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('danger', '#dc3545')};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """)
        threshold_layout.addWidget(auto_detect_btn)
        
        # Threshold violations count
        self.threshold_violations_label = QLabel("")
        self.threshold_violations_label.setStyleSheet(f"color: {self.theme.get('danger', '#dc3545')}; background: transparent; font-weight: bold;")
        threshold_layout.addWidget(self.threshold_violations_label)
        
        threshold_layout.addStretch()
        freq_layout.addLayout(threshold_layout)
        
        # Band power display
        self.band_power_widget = BandPowerComparisonWidget(
            theme=self.theme,
            parent=self,
        )
        freq_layout.addWidget(self.band_power_widget)
        
        splitter.addWidget(freq_widget)
        
        # Set splitter sizes
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)
    
    def set_data(self, raw: mne.io.Raw):
        """Set the raw data for this electrode."""
        self._raw_data = raw
        if raw is not None:
            self._max_time = raw.times[-1]
            self._view_start = 0.0
            self.duration_label.setText(f"/ {self._max_time:.1f}s")
            
            # Update visual timeline
            self.cut_timeline.set_max_time(self._max_time)
            initial_end = min(10.0, self._max_time)
            self.cut_timeline.set_markers(0.0, initial_end)
            self._current_selection = (0.0, initial_end)
            self.selection_info.setText(f"Selection: 0.0s - {initial_end:.1f}s ({initial_end:.1f}s)")
            
            # Update frequency analysis ranges (two ranges for comparison)
            half_time = int(self._max_time / 2)
            self.freq_start1_spin.setMaximum(int(self._max_time))
            self.freq_end1_spin.setMaximum(int(self._max_time))
            self.freq_end1_spin.setValue(half_time)
            
            self.freq_start2_spin.setMaximum(int(self._max_time))
            self.freq_start2_spin.setValue(half_time)
            self.freq_end2_spin.setMaximum(int(self._max_time))
            self.freq_end2_spin.setValue(int(self._max_time))
            
            if self.view_combo.currentText() == "Full":
                self._view_window = self._max_time
            
            self._update_plot()
            self._update_frequency_analysis()
    
    def set_cut_regions(self, regions: List[Tuple[float, float]]):
        """Set the cut regions (shared across electrodes)."""
        self._cut_regions = regions.copy()
        self.cut_timeline.set_cut_regions(regions)
        self._update_regions_display()
        self._update_plot()
    
    def _on_view_changed(self, text: str):
        """Handle view window change."""
        if text == "Full":
            self._view_window = self._max_time
        else:
            self._view_window = float(text.replace("s", ""))
        self._update_plot()
    
    def _on_nav_changed(self, value: int):
        """Handle navigation slider change."""
        max_start = max(0, self._max_time - self._view_window)
        self._view_start = (value / 1000.0) * max_start
        self.nav_label.setText(f"Position: {self._view_start:.1f}s")
        self._update_plot()
    
    def _on_markers_changed(self, left: float, right: float):
        """Handle timeline marker changes."""
        self._current_selection = (left, right)
        duration = right - left
        self.selection_info.setText(f"Selection: {left:.1f}s - {right:.1f}s ({duration:.1f}s)")
        self._update_plot()
    
    def _on_timeline_region_clicked(self, index: int):
        """Handle click on existing region in timeline to remove it."""
        if 0 <= index < len(self._cut_regions):
            removed = self._cut_regions.pop(index)
            self._update_regions_display()
            self._update_plot()
            # Update timeline
            self.cut_timeline.set_cut_regions(self._cut_regions)
            # Notify parent
            self.cut_region_removed.emit(index)
    
    def _add_cut_region(self):
        """Add the current selection as a cut region."""
        start, end = self._current_selection
        
        if start >= end:
            QMessageBox.warning(
                self, "Invalid Region",
                "Start time must be less than end time."
            )
            return
        
        # Check for overlaps
        for existing_start, existing_end in self._cut_regions:
            if not (end <= existing_start or start >= existing_end):
                QMessageBox.warning(
                    self, "Overlapping Region",
                    f"This region overlaps with existing cut "
                    f"({existing_start:.1f}s - {existing_end:.1f}s)."
                )
                return
        
        self.cut_region_added.emit(start, end)
    
    def _update_regions_display(self):
        """Update the cut regions display label."""
        if not self._cut_regions:
            self.regions_label.setText("No cut regions defined. Drag the markers above to select a region.")
            self.regions_label.setStyleSheet(
                f"color: {self.theme.get('text_light', '#6c757d')}; background: transparent;"
            )
        else:
            regions_str = ", ".join(
                f"[{s:.1f}s - {e:.1f}s]" for s, e in self._cut_regions
            )
            total = sum(e - s for s, e in self._cut_regions)
            self.regions_label.setText(
                f"Cut regions: {regions_str} (Total: {total:.1f}s) - Click on timeline to remove"
            )
            self.regions_label.setStyleSheet(
                f"color: {self.theme.get('danger', '#dc3545')}; font-weight: bold; background: transparent;"
            )
    
    def _update_plot(self):
        """Update the signal plot with annotations and cut regions."""
        if self._raw_data is None:
            self._show_empty_message()
            return
        
        self.figure.clear()
        self.figure.set_facecolor('white')
        
        try:
            sfreq = self._raw_data.info["sfreq"]
            
            # Get data for this channel only
            data = self._raw_data.get_data(picks=[self.channel_idx]) * 1e6  # μV
            times = self._raw_data.times
            
            # Calculate view range
            view_end = min(self._view_start + self._view_window, self._max_time)
            start_idx = int(self._view_start * sfreq)
            end_idx = min(int(view_end * sfreq), data.shape[1])
            
            display_times = times[start_idx:end_idx]
            display_data = data[0, start_idx:end_idx]
            
            ax = self.figure.add_subplot(111, facecolor='white')
            
            # Get voltage threshold
            threshold = self.voltage_threshold_spin.value()
            
            # Plot signal - color code based on threshold
            # First plot all in blue
            ax.plot(
                display_times, display_data,
                color=self.theme.get('primary', '#007AFF'),
                linewidth=0.8, alpha=0.9
            )
            
            # Highlight regions exceeding threshold in red
            exceeds_threshold = np.abs(display_data) > threshold
            if np.any(exceeds_threshold):
                # Create masked array for exceeding regions
                masked_data = np.ma.masked_where(~exceeds_threshold, display_data)
                ax.plot(display_times, masked_data, color='red', linewidth=1.5, alpha=0.9)
                
                # Count violations in current view
                violation_count = np.sum(np.diff(exceeds_threshold.astype(int)) == 1)
                self.threshold_violations_label.setText(f"⚠️ {violation_count} violation(s) in view")
            else:
                self.threshold_violations_label.setText("")
            
            # Draw threshold lines
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'+{threshold} μV')
            ax.axhline(y=-threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'-{threshold} μV')
            
            # Add annotations (eyes open/closed)
            if self._raw_data.annotations is not None:
                for annot in self._raw_data.annotations:
                    onset = annot["onset"]
                    duration = annot["duration"] if annot["duration"] > 0 else 5.0
                    description = annot["description"].lower()
                    annot_end = onset + duration
                    
                    if annot_end >= self._view_start and onset <= view_end:
                        if "open" in description:
                            color = "#28a745"
                            label = "Eyes Open"
                        elif "close" in description:
                            color = "#6f42c1"
                            label = "Eyes Closed"
                        else:
                            color = "#ffc107"
                            label = annot["description"]
                        
                        draw_start = max(onset, self._view_start)
                        draw_end = min(annot_end, view_end)
                        ax.axvspan(draw_start, draw_end, alpha=0.2, color=color)
                        
                        mid = (draw_start + draw_end) / 2
                        if self._view_start <= mid <= view_end:
                            ax.text(
                                mid, ax.get_ylim()[1] * 0.95, label,
                                ha='center', va='top', fontsize=8,
                                color=color, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white', alpha=0.8)
                            )
            
            # Highlight cut regions
            for start, end in self._cut_regions:
                if end >= self._view_start and start <= view_end:
                    draw_start = max(start, self._view_start)
                    draw_end = min(end, view_end)
                    ax.axvspan(draw_start, draw_end, alpha=0.4,
                               color=self.theme.get('danger', '#dc3545'),
                               hatch='///', edgecolor='darkred')
            
            # Highlight current selection (dashed outline)
            sel_start, sel_end = self._current_selection
            if sel_end >= self._view_start and sel_start <= view_end:
                draw_start = max(sel_start, self._view_start)
                draw_end = min(sel_end, view_end)
                ax.axvspan(draw_start, draw_end, alpha=0.15,
                           color=self.theme.get('primary', '#007AFF'),
                           linestyle='--', linewidth=2,
                           edgecolor=self.theme.get('primary', '#007AFF'))
            
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Amplitude (μV)", fontsize=9)
            ax.set_xlim(self._view_start, view_end)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            self.figure.tight_layout()
            
        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}",
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax.transAxes)
        
        self.canvas.draw()
    
    def _show_empty_message(self):
        """Show empty message when no data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "No data loaded",
                ha='center', va='center', fontsize=12,
                color=self.theme.get('text_light', '#6c757d'),
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
    
    def _update_frequency_analysis(self):
        """Update band power analysis for this electrode with two ranges for comparison."""
        if self._raw_data is None:
            self.band_power_widget.clear()
            return
        
        try:
            analyzer = BandPowerAnalyzer()
            
            # Range 1
            start1 = self.freq_start1_spin.value()
            end1 = self.freq_end1_spin.value()
            
            # Range 2
            start2 = self.freq_start2_spin.value()
            end2 = self.freq_end2_spin.value()
            
            powers1 = None
            powers2 = None
            
            if start1 < end1:
                powers1 = analyzer.compute_band_power_for_raw(
                    self._raw_data,
                    channel_idx=self.channel_idx,
                    tmin=float(start1),
                    tmax=float(end1),
                )
            
            if start2 < end2:
                powers2 = analyzer.compute_band_power_for_raw(
                    self._raw_data,
                    channel_idx=self.channel_idx,
                    tmin=float(start2),
                    tmax=float(end2),
                )
            
            if powers1 is not None and powers2 is not None:
                self.band_power_widget.update_comparison(powers1, powers2)
            elif powers1 is not None:
                self.band_power_widget.update_comparison(powers1, powers1)
            elif powers2 is not None:
                self.band_power_widget.update_comparison(powers2, powers2)
            else:
                self.band_power_widget.clear()
            
        except Exception:
            self.band_power_widget.clear()
    
    def _auto_detect_artifacts(self):
        """Automatically detect and mark regions exceeding voltage threshold for cutting."""
        if self._raw_data is None:
            return
        
        threshold = self.voltage_threshold_spin.value()
        
        try:
            # Get data for this channel
            data = self._raw_data.get_data(picks=[self.channel_idx]) * 1e6  # Convert to μV
            times = self._raw_data.times
            sfreq = self._raw_data.info["sfreq"]
            
            # Find samples exceeding threshold
            exceeds = np.abs(data[0]) > threshold
            
            if not np.any(exceeds):
                QMessageBox.information(
                    self, "No Artifacts Found",
                    f"No signal segments exceed the {threshold} μV threshold."
                )
                return
            
            # Find continuous regions exceeding threshold
            # Add padding to merge nearby regions (0.5s)
            padding_samples = int(0.5 * sfreq)
            
            # Dilate the exceeds array to merge nearby regions
            structure = np.ones(padding_samples)
            dilated = binary_dilation(exceeds, structure=structure)
            
            # Find start/end of each region
            diff = np.diff(dilated.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            
            # Handle edge cases
            if dilated[0]:
                starts = np.insert(starts, 0, 0)
            if dilated[-1]:
                ends = np.append(ends, len(dilated))
            
            # Convert to time and create regions
            new_regions = []
            for start_idx, end_idx in zip(starts, ends):
                start_time = times[start_idx]
                end_time = times[min(end_idx, len(times) - 1)]
                
                # Check if this region overlaps with existing cuts
                overlaps = False
                for existing_start, existing_end in self._cut_regions:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    new_regions.append((start_time, end_time))
            
            if not new_regions:
                QMessageBox.information(
                    self, "No New Artifacts",
                    "All detected artifacts are already marked for cutting."
                )
                return
            
            # Ask user to confirm
            total_time = sum(end - start for start, end in new_regions)
            reply = QMessageBox.question(
                self, "Confirm Auto-Detection",
                f"Found {len(new_regions)} region(s) exceeding {threshold} μV.\n"
                f"Total time to remove: {total_time:.1f}s\n\n"
                f"Add these regions to the cut list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                for start, end in new_regions:
                    self.cut_region_added.emit(start, end)
                
                QMessageBox.information(
                    self, "Regions Added",
                    f"Added {len(new_regions)} artifact region(s) to the cut list."
                )
        
        except Exception as e:
            QMessageBox.warning(
                self, "Detection Error",
                f"Failed to detect artifacts: {str(e)}"
            )
    
    def clear(self):
        """Clear the widget."""
        self._raw_data = None
        self._cut_regions = []
        self._show_empty_message()
        self._update_regions_display()
        self.band_power_widget.clear()


class SignalPreviewScreen(QWidget):
    """
    Screen for previewing and editing EEG signal before processing.
    
    Organizes by electrode - each electrode has its own tab with:
    - Signal preview with navigation
    - Cut region selection on the signal
    - Frequency analysis

    Signals:
        proceed_to_processing: Emitted when user wants to continue to ICA/PCA
        signal_modified: Emitted when signal has been modified (e.g., regions cut)
        return_to_channels: Emitted when user wants to go back to channel selection
    """

    proceed_to_processing = pyqtSignal(object)  # Emits modified raw data
    signal_modified = pyqtSignal(object)  # Emits modified raw data
    return_to_channels = pyqtSignal()

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self._raw_data = None
        self._original_raw_data = None
        self._file_path = ""
        self._cut_regions: List[Tuple[float, float]] = []
        self._electrode_widgets: Dict[str, ElectrodeSignalWidget] = {}
        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Apply white theme background to entire screen
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
        """)

        # Title
        title_label = QLabel("🔬 Signal Preview & Editing")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            f"color: {self.theme.get('primary', '#007AFF')}; margin: 5px; background: transparent;"
        )
        layout.addWidget(title_label)

        # Description
        description = QLabel(
            "Each electrode has its own tab. View the signal, select cut regions, "
            "and analyze frequency bands. Cuts apply to all electrodes."
        )
        description.setFont(QFont("Arial", 11))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')}; margin-bottom: 10px; background: transparent;"
        )
        layout.addWidget(description)

        # Header with file info and controls
        header_layout = QHBoxLayout()
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setFont(QFont("Arial", 10))
        self.file_info_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')}; background: transparent;"
        )
        header_layout.addWidget(self.file_info_label)
        
        header_layout.addStretch()
        
        # Help button
        help_btn = QPushButton("📖 How to Use")
        help_btn.clicked.connect(self._show_help_dialog)
        help_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """)
        header_layout.addWidget(help_btn)
        
        layout.addLayout(header_layout)

        # Tab widget for electrodes
        self.electrode_tabs = QTabWidget()
        self.electrode_tabs.setFont(QFont("Arial", 10))
        self.electrode_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                background-color: {self.theme.get('background', '#FFFFFF')};
            }}
            QTabBar::tab {{
                background-color: #f8f9fa;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: {self.theme.get('text', '#212529')};
            }}
            QTabBar::tab:selected {{
                background-color: {self.theme.get('background', '#FFFFFF')};
                border-bottom-color: {self.theme.get('background', '#FFFFFF')};
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.electrode_tabs)

        # Action bar
        action_layout = QHBoxLayout()
        
        # Regions summary
        self.regions_summary = QLabel("No cut regions")
        self.regions_summary.setFont(QFont("Arial", 10))
        self.regions_summary.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')};"
        )
        action_layout.addWidget(self.regions_summary)
        
        action_layout.addStretch()
        
        # Clear cuts button
        clear_btn = QPushButton("🗑️ Clear All Cuts")
        clear_btn.clicked.connect(self._clear_all_cuts)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('text_light', '#6c757d')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #5a6268;
            }}
        """)
        action_layout.addWidget(clear_btn)
        
        # Apply cuts button
        apply_btn = QPushButton("✂️ Apply Cuts")
        apply_btn.clicked.connect(self._apply_cuts)
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('danger', '#dc3545')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """)
        action_layout.addWidget(apply_btn)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset to Original")
        reset_btn.clicked.connect(self._reset_signal)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('text_light', '#6c757d')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #5a6268;
            }}
        """)
        action_layout.addWidget(reset_btn)
        
        layout.addLayout(action_layout)

        # Bottom button bar
        button_layout = QHBoxLayout()

        # Back button
        back_btn = QPushButton("⬅️ Back to Channel Selection")
        back_btn.setMinimumHeight(45)
        back_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        back_btn.clicked.connect(self.return_to_channels.emit)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('text_light', '#6c757d')};
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #5a6268;
            }}
        """)
        button_layout.addWidget(back_btn)

        button_layout.addStretch()

        # Continue button
        self.continue_btn = QPushButton("▶️ Continue to Artifact Removal")
        self.continue_btn.setMinimumHeight(45)
        self.continue_btn.setMinimumWidth(280)
        self.continue_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.continue_btn.clicked.connect(self._on_continue)
        self.continue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('primary_hover', '#0056b3')};
            }}
        """)
        button_layout.addWidget(self.continue_btn)

        layout.addLayout(button_layout)

    def set_data(self, raw: mne.io.Raw, file_path: str = ""):
        """
        Set the EEG data for preview and editing.

        Args:
            raw: MNE Raw object (should be preloaded)
            file_path: Path to the source file
        """
        self._raw_data = raw.copy() if raw is not None else None
        self._original_raw_data = raw.copy() if raw is not None else None
        self._file_path = file_path
        self._cut_regions = []
        
        # Clear existing tabs
        self.electrode_tabs.clear()
        self._electrode_widgets.clear()

        if raw is not None:
            # Update file info
            duration = raw.times[-1]
            n_channels = len(raw.ch_names)
            sfreq = raw.info["sfreq"]
            n_annotations = len(raw.annotations)

            self.file_info_label.setText(
                f"📁 {file_path.split('/')[-1] if file_path else 'Unknown'} | "
                f"🧠 {n_channels} channels | "
                f"⏱️ {duration:.1f}s | "
                f"⚡ {sfreq:.0f} Hz | "
                f"📌 {n_annotations} annotations"
            )

            # Create tab for each electrode
            for idx, ch_name in enumerate(raw.ch_names):
                electrode_widget = ElectrodeSignalWidget(
                    channel_name=ch_name,
                    channel_idx=idx,
                    theme=self.theme,
                    parent=self
                )
                electrode_widget.set_data(raw)
                electrode_widget.cut_region_added.connect(self._on_cut_region_added)
                
                self._electrode_widgets[ch_name] = electrode_widget
                self.electrode_tabs.addTab(electrode_widget, f"🧠 {ch_name}")

        self._update_regions_summary()

    def _on_cut_region_added(self, start: float, end: float):
        """Handle cut region added from any electrode widget."""
        self._cut_regions.append((start, end))
        self._cut_regions.sort(key=lambda x: x[0])
        
        # Update all electrode widgets
        for widget in self._electrode_widgets.values():
            widget.set_cut_regions(self._cut_regions)
        
        self._update_regions_summary()

    def _clear_all_cuts(self):
        """Clear all cut regions."""
        self._cut_regions = []
        for widget in self._electrode_widgets.values():
            widget.set_cut_regions([])
        self._update_regions_summary()

    def _apply_cuts(self):
        """Apply the cut regions to the signal."""
        if self._raw_data is None or not self._cut_regions:
            QMessageBox.information(
                self, "No Cuts",
                "No cut regions have been defined."
            )
            return

        try:
            # Apply cuts
            cut_raw = SignalEditor.cut_signal_regions(
                self._raw_data, self._cut_regions
            )
            self._raw_data = cut_raw
            self._cut_regions = []

            # Update all electrode widgets with new data
            for idx, (ch_name, widget) in enumerate(self._electrode_widgets.items()):
                widget.set_data(cut_raw)
                widget.set_cut_regions([])

            # Update file info
            duration = cut_raw.times[-1]
            n_channels = len(cut_raw.ch_names)
            sfreq = cut_raw.info["sfreq"]

            self.file_info_label.setText(
                f"📁 {self._file_path.split('/')[-1] if self._file_path else 'Unknown'} | "
                f"🧠 {n_channels} channels | "
                f"⏱️ {duration:.1f}s (modified) | "
                f"⚡ {sfreq:.0f} Hz"
            )

            self._update_regions_summary()
            self.signal_modified.emit(cut_raw)

            QMessageBox.information(
                self, "Signal Modified",
                f"Signal regions have been cut successfully.\n"
                f"New duration: {duration:.1f} seconds"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to cut signal regions:\n{str(e)}"
            )

    def _reset_signal(self):
        """Reset signal to original state."""
        if self._original_raw_data is None:
            return

        reply = QMessageBox.question(
            self, "Reset Signal",
            "Are you sure you want to reset to the original signal?\n"
            "All modifications will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._raw_data = self._original_raw_data.copy()
            self._cut_regions = []
            self.set_data(self._raw_data, self._file_path)

    def _update_regions_summary(self):
        """Update the regions summary label."""
        if not self._cut_regions:
            self.regions_summary.setText("No cut regions")
            self.regions_summary.setStyleSheet(
                f"color: {self.theme.get('text_light', '#6c757d')};"
            )
        else:
            total = sum(e - s for s, e in self._cut_regions)
            self.regions_summary.setText(
                f"✂️ {len(self._cut_regions)} cut region(s) | "
                f"Total: {total:.1f}s to remove"
            )
            self.regions_summary.setStyleSheet(
                f"color: {self.theme.get('danger', '#dc3545')}; font-weight: bold;"
            )

    def _show_help_dialog(self):
        """Show the help dialog."""
        dialog = SignalEditingHelpDialog(self.theme, self)
        dialog.exec()

    def _on_continue(self):
        """Continue to processing with current signal."""
        if self._raw_data is not None:
            self.proceed_to_processing.emit(self._raw_data)

    def get_current_raw(self) -> Optional[mne.io.Raw]:
        """Get the current (possibly modified) raw data."""
        return self._raw_data

    def clear(self):
        """Clear all data and reset UI."""
        self._raw_data = None
        self._original_raw_data = None
        self._file_path = ""
        self._cut_regions = []
        self.electrode_tabs.clear()
        self._electrode_widgets.clear()
        self.file_info_label.setText("No file loaded")
        self._update_regions_summary()
