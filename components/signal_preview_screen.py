#!/usr/bin/env python3
"""
Signal Preview Screen - Preview and edit signal before processing
==================================================================

Provides a screen where users can:
- Preview the loaded EEG signal
- Select time ranges for frequency analysis
- Detect and view resting phases (eyes open/closed)
- Cut signal regions manually before artifact removal

Author: porfanid
Version: 1.0
"""

from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .band_power_display import BandPowerComparisonWidget
from .signal_editor import RestingPhaseDisplay, SignalCutter, TimeRangeSelector


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
        <p>The Signal Editor allows you to <b>manually remove unwanted portions</b> of your EEG recording 
        before automatic artifact removal. This is useful for:</p>
        <ul>
            <li>Removing sections with excessive noise or movement artifacts</li>
            <li>Cutting out recording start/end artifacts</li>
            <li>Removing specific time periods that you don't want to analyze</li>
        </ul>

        <h2>✂️ How to Cut Signal Regions</h2>
        <ol>
            <li><b>Go to the "Signal Editor" tab</b> in this screen</li>
            <li><b>Define a region to cut:</b>
                <ul>
                    <li>Enter the <b>Start time</b> (in seconds) where the unwanted region begins</li>
                    <li>Enter the <b>End time</b> (in seconds) where the unwanted region ends</li>
                </ul>
            </li>
            <li><b>Click "Add Cut Region"</b> to mark this region for removal</li>
            <li><b>Repeat</b> steps 2-3 for any additional regions you want to remove</li>
            <li><b>Click "Apply Cuts"</b> to remove all marked regions</li>
        </ol>

        <h2>⚠️ Important Notes</h2>
        <ul>
            <li><b>Regions cannot overlap</b> - each cut region must be separate</li>
            <li><b>The signal will be joined</b> - after cutting, the remaining segments 
            are automatically concatenated</li>
            <li><b>You can reset</b> - use the "Reset to Original Signal" button to undo all cuts</li>
            <li><b>Preview first</b> - check the Signal Preview tab to identify problem areas</li>
        </ul>

        <h2>📊 Using Frequency Analysis</h2>
        <p>The "Frequency Analysis" tab helps you understand your signal:</p>
        <ul>
            <li><b>Time Range Selection:</b> Use the sliders to analyze specific portions of your recording</li>
            <li><b>Band Power:</b> See the distribution of Delta, Theta, Alpha, Beta, and Gamma waves</li>
            <li><b>Resting Phases:</b> If your recording has "eyes open"/"eyes closed" markers, 
            they will be automatically detected and analyzed</li>
        </ul>

        <h2>▶️ When You're Ready</h2>
        <p>Click <b>"Continue to Artifact Removal"</b> to proceed with ICA/PCA analysis. 
        Any signal modifications you've made will be preserved.</p>

        <h2>💡 Tips</h2>
        <ul>
            <li>Use the Signal Preview to identify noisy sections before cutting</li>
            <li>Cut conservatively - you can always process more data later</li>
            <li>Check the frequency analysis after cutting to verify signal quality</li>
        </ul>
        """

        help_text.setHtml(help_content)
        layout.addWidget(help_text)

        # Close button
        close_btn = QPushButton("✓ Got it!")
        close_btn.setMinimumHeight(40)
        close_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
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


class SignalPlotWidget(QWidget):
    """Widget for displaying EEG signal preview."""

    def __init__(self, theme: Optional[Dict[str, str]] = None, parent=None):
        super().__init__(parent)
        self.theme = theme or {}
        self._raw_data = None
        self.setup_ui()

    def setup_ui(self):
        """Create the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("📈 Signal Preview")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')};"
        )
        layout.addWidget(title_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 4), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)
        layout.addWidget(self.canvas)

        # Show empty message initially
        self._show_empty_message()

    def _show_empty_message(self):
        """Show message when no data is loaded."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Load a file to preview the signal",
            ha="center", va="center",
            fontsize=12,
            color=self.theme.get("text_light", "#6c757d"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()

    def set_data(self, raw: mne.io.Raw, time_window: float = 10.0):
        """
        Set the EEG data for preview.

        Args:
            raw: MNE Raw object
            time_window: Time window to display in seconds
        """
        self._raw_data = raw
        self._update_plot(time_window)

    def _update_plot(self, time_window: float = 10.0):
        """Update the signal plot."""
        if self._raw_data is None:
            self._show_empty_message()
            return

        self.figure.clear()

        try:
            # Get data
            data = self._raw_data.get_data() * 1e6  # Convert to μV
            times = self._raw_data.times
            channels = self._raw_data.ch_names

            # Limit to time window
            max_samples = min(
                int(time_window * self._raw_data.info["sfreq"]),
                data.shape[1]
            )
            display_times = times[:max_samples]
            display_data = data[:, :max_samples]

            # Create subplot for each channel (max 5 channels for readability)
            n_channels = min(len(channels), 5)
            
            for i in range(n_channels):
                ax = self.figure.add_subplot(n_channels, 1, i + 1)
                ax.plot(
                    display_times,
                    display_data[i],
                    color=self.theme.get("primary", "#007AFF"),
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.set_ylabel(channels[i], fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                
                if i == n_channels - 1:
                    ax.set_xlabel("Time (s)", fontsize=9)
                else:
                    ax.set_xticklabels([])

            self.figure.tight_layout()

        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Error displaying signal: {str(e)}",
                ha="center", va="center",
                fontsize=10,
                color="red",
                transform=ax.transAxes,
            )

        self.canvas.draw()

    def clear(self):
        """Clear the plot."""
        self._raw_data = None
        self._show_empty_message()


class SignalPreviewScreen(QWidget):
    """
    Screen for previewing and editing EEG signal before processing.

    Provides:
    - Signal preview visualization
    - Time range selection for frequency analysis
    - Resting phase detection and display
    - Signal cutting tools

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
        self._original_raw_data = None  # Keep original for reset
        self._file_path = ""
        self.setup_ui()

    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("🔬 Signal Preview & Editing")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            f"color: {self.theme['primary']}; margin: 5px;"
        )
        layout.addWidget(title_label)

        # Description
        description = QLabel(
            "Preview your signal, analyze frequency bands, detect resting phases, "
            "and optionally cut unwanted regions before artifact removal."
        )
        description.setFont(QFont("Arial", 11))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet(
            f"color: {self.theme['text_light']}; margin-bottom: 10px;"
        )
        layout.addWidget(description)

        # Main content area with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 10))

        # Tab 1: Signal Preview
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(5, 5, 5, 5)

        self.signal_plot = SignalPlotWidget(theme=self.theme)
        preview_layout.addWidget(self.signal_plot)

        # File info label
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setFont(QFont("Arial", 10))
        self.file_info_label.setStyleSheet(
            f"color: {self.theme['text_light']}; padding: 5px;"
        )
        preview_layout.addWidget(self.file_info_label)

        self.tab_widget.addTab(preview_tab, "📈 Signal Preview")

        # Tab 2: Frequency Analysis
        freq_tab = QWidget()
        freq_layout = QVBoxLayout(freq_tab)
        freq_layout.setContentsMargins(5, 5, 5, 5)

        # Scroll area for frequency analysis
        freq_scroll = QScrollArea()
        freq_scroll.setWidgetResizable(True)
        freq_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        freq_content = QWidget()
        freq_content_layout = QVBoxLayout(freq_content)

        # Time range selector
        self.time_range_selector = TimeRangeSelector(
            theme=self.theme,
            parent=self,
        )
        self.time_range_selector.range_changed.connect(self._on_time_range_changed)
        freq_content_layout.addWidget(self.time_range_selector)

        # Band power display
        self.band_power_widget = BandPowerComparisonWidget(
            theme=self.theme,
            parent=self,
        )
        freq_content_layout.addWidget(self.band_power_widget)

        # Resting phase display
        self.resting_phase_display = RestingPhaseDisplay(
            theme=self.theme,
            parent=self,
        )
        freq_content_layout.addWidget(self.resting_phase_display)

        freq_content_layout.addStretch()
        freq_scroll.setWidget(freq_content)
        freq_layout.addWidget(freq_scroll)

        self.tab_widget.addTab(freq_tab, "📊 Frequency Analysis")

        # Tab 3: Signal Editor (Cutting)
        editor_tab = QWidget()
        editor_layout = QVBoxLayout(editor_tab)
        editor_layout.setContentsMargins(10, 10, 10, 10)
        editor_layout.setSpacing(10)

        # Header with help button
        header_layout = QHBoxLayout()

        editor_title = QLabel("✂️ Manual Signal Cutting")
        editor_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        editor_title.setStyleSheet(f"color: {self.theme['primary']};")
        header_layout.addWidget(editor_title)

        header_layout.addStretch()

        # Help button
        help_btn = QPushButton("📖 How to Use")
        help_btn.setFont(QFont("Arial", 10))
        help_btn.clicked.connect(self._show_help_dialog)
        help_btn.setStyleSheet(
            f"""
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
        """
        )
        header_layout.addWidget(help_btn)

        editor_layout.addLayout(header_layout)

        # Quick instructions box
        instructions_box = QGroupBox("📋 Quick Instructions")
        instructions_box.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        instructions_box.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background-color: #f8f9fa;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {self.theme.get('text', '#212529')};
                background-color: #f8f9fa;
            }}
        """
        )
        instructions_layout = QVBoxLayout(instructions_box)

        quick_instructions = QLabel(
            "<b>1.</b> Enter the <b>Start</b> and <b>End</b> time (in seconds) of the region to remove<br>"
            "<b>2.</b> Click <b>'Add Cut Region'</b> to mark the region<br>"
            "<b>3.</b> Repeat for any additional regions<br>"
            "<b>4.</b> Click <b>'Apply Cuts'</b> to remove all marked regions<br><br>"
            "<i>💡 Tip: Check the Signal Preview tab first to identify noisy sections</i>"
        )
        quick_instructions.setFont(QFont("Arial", 10))
        quick_instructions.setWordWrap(True)
        quick_instructions.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        instructions_layout.addWidget(quick_instructions)

        editor_layout.addWidget(instructions_box)

        # Signal Cutter widget
        self.signal_cutter = SignalCutter(
            theme=self.theme,
            parent=self,
        )
        self.signal_cutter.apply_cuts.connect(self._on_apply_cuts)
        editor_layout.addWidget(self.signal_cutter)

        # Buttons row
        buttons_layout = QHBoxLayout()

        # Reset button
        reset_btn = QPushButton("🔄 Reset to Original Signal")
        reset_btn.clicked.connect(self._reset_signal)
        reset_btn.setStyleSheet(
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
        buttons_layout.addWidget(reset_btn)

        buttons_layout.addStretch()

        editor_layout.addLayout(buttons_layout)
        editor_layout.addStretch()

        self.tab_widget.addTab(editor_tab, "✂️ Signal Editor")

        layout.addWidget(self.tab_widget)

        # Bottom button bar
        button_layout = QHBoxLayout()

        # Back button
        back_btn = QPushButton("⬅️ Back to Channel Selection")
        back_btn.setMinimumHeight(45)
        back_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        back_btn.clicked.connect(self.return_to_channels.emit)
        back_btn.setStyleSheet(
            f"""
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
        """
        )
        button_layout.addWidget(back_btn)

        button_layout.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )

        # Continue button
        self.continue_btn = QPushButton("▶️ Continue to Artifact Removal")
        self.continue_btn.setMinimumHeight(45)
        self.continue_btn.setMinimumWidth(280)
        self.continue_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.continue_btn.clicked.connect(self._on_continue)
        self.continue_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['primary']};
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme['primary_hover']};
            }}
        """
        )
        button_layout.addWidget(self.continue_btn)

        layout.addLayout(button_layout)

    def set_data(self, raw: mne.io.Raw, file_path: str = ""):
        """
        Set the EEG data for preview and editing.

        Args:
            raw: MNE Raw object (should be preloaded)
            file_path: Path to the source file
        """
        # Store both current and original
        self._raw_data = raw.copy() if raw is not None else None
        self._original_raw_data = raw.copy() if raw is not None else None
        self._file_path = file_path

        if raw is not None:
            # Update signal preview
            self.signal_plot.set_data(raw)

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

            # Update time range selector
            self.time_range_selector.set_time_range(0.0, duration)

            # Update signal cutter
            self.signal_cutter.set_max_time(duration)

            # Update frequency analysis
            self._update_frequency_analysis()

            # Detect resting phases
            self._detect_resting_phases()

    def _update_frequency_analysis(self):
        """Update band power analysis for current time range."""
        if self._raw_data is None:
            return

        try:
            from backend import BandPowerAnalyzer

            analyzer = BandPowerAnalyzer()
            start_time, end_time = self.time_range_selector.get_range()

            # Calculate band powers
            powers = analyzer.compute_band_power_for_raw(
                self._raw_data,
                channel_idx=0,
                tmin=start_time,
                tmax=end_time,
            )

            # Update display (show same data for both since no cleaned version yet)
            self.band_power_widget.update_comparison(powers, powers)

        except Exception as e:
            print(f"Error updating frequency analysis: {e}")

    def _detect_resting_phases(self):
        """Detect and display resting phases from annotations."""
        if self._raw_data is None:
            self.resting_phase_display.update_phases([])
            return

        try:
            from backend import BandPowerAnalyzer, SignalEditor

            # Detect phases
            phases = SignalEditor.detect_resting_phases(self._raw_data)

            if not phases:
                self.resting_phase_display.update_phases([])
                return

            # Calculate band powers for each phase
            analyzer = BandPowerAnalyzer()
            original_powers = {}

            for phase in phases:
                phase_label = phase["label"]
                start = phase["start"]
                end = min(phase["end"], self._raw_data.times[-1])

                try:
                    power = analyzer.compute_band_power_for_raw(
                        self._raw_data,
                        channel_idx=0,
                        tmin=start,
                        tmax=end,
                    )
                    original_powers[phase_label] = power
                except Exception:
                    original_powers[phase_label] = None

            self.resting_phase_display.update_phases(phases, original_powers)

        except Exception as e:
            print(f"Error detecting resting phases: {e}")
            self.resting_phase_display.update_phases([])

    def _on_time_range_changed(self, start: float, end: float):
        """Handle time range change."""
        self._update_frequency_analysis()

    def _on_apply_cuts(self, regions: List[Tuple[float, float]]):
        """Apply signal cuts."""
        if self._raw_data is None or not regions:
            return

        try:
            from backend import SignalEditor

            # Apply cuts
            cut_raw = SignalEditor.cut_signal_regions(self._raw_data, regions)
            self._raw_data = cut_raw

            # Update UI
            self.signal_plot.set_data(cut_raw)
            duration = cut_raw.times[-1]
            self.time_range_selector.set_time_range(0.0, duration)
            self.signal_cutter.set_max_time(duration)
            self.signal_cutter.clear_regions()

            # Update file info
            n_channels = len(cut_raw.ch_names)
            sfreq = cut_raw.info["sfreq"]

            self.file_info_label.setText(
                f"📁 {self._file_path.split('/')[-1] if self._file_path else 'Unknown'} | "
                f"🧠 {n_channels} channels | "
                f"⏱️ {duration:.1f}s (modified) | "
                f"⚡ {sfreq:.0f} Hz"
            )

            # Update frequency analysis
            self._update_frequency_analysis()
            self._detect_resting_phases()

            # Emit signal
            self.signal_modified.emit(cut_raw)

            QMessageBox.information(
                self,
                "Signal Modified",
                f"Signal regions have been cut successfully.\n"
                f"New duration: {duration:.1f} seconds",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to cut signal regions:\n{str(e)}",
            )

    def _reset_signal(self):
        """Reset signal to original state."""
        if self._original_raw_data is None:
            return

        reply = QMessageBox.question(
            self,
            "Reset Signal",
            "Are you sure you want to reset to the original signal?\n"
            "All modifications will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._raw_data = self._original_raw_data.copy()
            self.set_data(self._raw_data, self._file_path)
            self.signal_cutter.clear_regions()

    def _show_help_dialog(self):
        """Show the help dialog for signal editing."""
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
        self.signal_plot.clear()
        self.file_info_label.setText("No file loaded")
        self.signal_cutter.clear_regions()
        self.band_power_widget.clear()
        self.resting_phase_display.clear()
