#!/usr/bin/env python3
"""
ICA Component Selector Widget - v4.0 - Correct Event Bubbling for Scrolling
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# --- 1. CREATE A CUSTOM CANVAS ---
# This class inherits all properties from FigureCanvas,
# but changes the wheelEvent behavior.
class CustomCanvas(FigureCanvas):
    def wheelEvent(self, event: QWheelEvent):
        """
        Instead of consuming the event, ignore it.
        When an event is ignored, Qt automatically forwards it to the parent widget.
        """
        event.ignore()


# --- 2. BACKGROUND THREAD FOR PREVIEW UPDATE ---
class PreviewUpdateThread(QThread):
    """
    Background thread for calculating the cleaned signal preview.

    This thread handles the computationally intensive task of applying
    artifact removal to generate a preview of the cleaned EEG signal,
    keeping the GUI responsive during processing.

    Attributes:
        preview_ready: Signal emitted when preview calculation is complete.
            Emits (original_raw, cleaned_raw) tuple.

    Args:
        ica: ICA object for backward compatibility with MNE ICA.
        raw: The raw EEG data (mne.io.Raw object).
        components_to_remove: List of component indices to exclude.
        processor: Component processor (ICA, PCA, or Wavelet).
        analysis_method: Analysis method string ("ICA", "PCA", or "WAVELETS").
    """

    preview_ready = pyqtSignal(object, object)  # (original_raw, cleaned_raw)

    def __init__(
        self,
        ica,
        raw,
        components_to_remove: List[int],
        processor=None,
        analysis_method="ICA",
    ):
        """Initialize the preview update thread with processing parameters."""
        super().__init__()
        self.ica = ica
        self.raw = raw
        self.components_to_remove = components_to_remove
        self.processor = processor
        self.analysis_method = analysis_method

    def run(self):
        """
        Execute the preview calculation in the background.

        Applies artifact removal using the configured processor or ICA object
        and emits the preview_ready signal with the results.
        """
        try:
            # If no components to remove, return original signal as both
            if not self.components_to_remove:
                self.preview_ready.emit(self.raw, self.raw)
                return

            # Use processor if available (for PCA and generic ICA)
            if self.processor is not None:
                cleaned_raw = self.processor.apply_artifact_removal(
                    self.components_to_remove
                )
                self.preview_ready.emit(self.raw, cleaned_raw)
                return

            # Fallback to ICA-specific handling (for backward compatibility)
            if self.ica is not None:
                # Create copy for cleaning
                cleaned_raw = self.raw.copy()

                # Set components to remove
                ica_copy = self.ica.copy()
                ica_copy.exclude = self.components_to_remove

                # Apply cleaning
                cleaned_raw = ica_copy.apply(cleaned_raw, verbose=False)

                # Emit results
                self.preview_ready.emit(self.raw, cleaned_raw)
            else:
                self.preview_ready.emit(self.raw, None)

        except Exception as e:
            print(f"Error in preview thread: {str(e)}")
            # Emit only original signal in case of error
            self.preview_ready.emit(self.raw, None)


# --- 3. PREVIEW WIDGET ---
class PreviewWidget(QWidget):
    """
    Widget for displaying real-time preview of the cleaned EEG signal.

    Shows original and cleaned signal comparison with navigation controls,
    band power analysis displays, and timeline visualization. Supports
    ICA, PCA, and Wavelet analysis methods with FFT comparison for Wavelets.

    Attributes:
        theme: Dictionary containing UI theme colors.
        selected_channel_idx: Index of currently displayed channel.
        channel_names: List of available channel names.
        range1: Time range tuple (start, end) for Eyes Closed analysis.
        range2: Time range tuple (start, end) for Eyes Open analysis.

    Args:
        theme: Dictionary containing UI color scheme.
        parent: Optional parent widget.
    """

    def __init__(self, theme: Dict[str, str], parent=None):
        """Initialize the preview widget with theme and layout."""
        super().__init__(parent)
        self.theme = theme
        self.selected_channel_idx = 0
        self.channel_names = []
        self.update_callback = None  # Callback for preview update
        self.band_power_analyzer = None  # Will be set on first use
        # Range 1 is Eyes Closed (displayed first)
        # Range 2 is Eyes Open (displayed second)
        self.range1 = None  # (start, end) tuple for Range 1 (Eyes Closed)
        self.range2 = None  # (start, end) tuple for Range 2 (Eyes Open)
        self._custom_ranges_set = False  # Track if custom ranges were set
        self._max_time = 100.0
        self._view_window = 10.0  # View window in seconds
        self._view_start = 0.0  # Current view start position
        self._original_raw = None
        self._cleaned_raw = None
        self._analysis_method = "ICA"  # Track analysis method for FFT display
        self.setup_ui()

    def set_analysis_method(self, method: str):
        """
        Set the analysis method for FFT display selection.

        Args:
            method: Analysis method ("ICA", "PCA", or "WAVELETS").
        """
        self._analysis_method = method

    def setup_ui(self):
        """Set up the user interface components and layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header layout with title and controls
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)

        # Title
        title_label = QLabel("📊 Live Preview of Cleaning Result")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.theme['text']};")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # View window selector
        view_label = QLabel("View:")
        view_label.setStyleSheet(f"color: {self.theme['text']}; font-size: 11px;")
        header_layout.addWidget(view_label)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["5s", "10s", "30s", "60s", "Full"])
        self.view_combo.setCurrentText("10s")
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        self.view_combo.setMinimumWidth(60)
        self.view_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 3px 6px;
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        header_layout.addWidget(self.view_combo)

        # Channel selection
        channel_label = QLabel("Channel:")
        channel_label.setStyleSheet(f"color: {self.theme['text']}; font-size: 11px;")
        header_layout.addWidget(channel_label)

        self.channel_dropdown = QComboBox()
        self.channel_dropdown.setMinimumWidth(100)
        self.channel_dropdown.setStyleSheet(
            f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                padding: 3px 6px;
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        self.channel_dropdown.currentIndexChanged.connect(self._on_channel_changed)
        header_layout.addWidget(self.channel_dropdown)

        layout.addLayout(header_layout)

        # Main content: signal plot + band power widgets
        content_layout = QHBoxLayout()
        content_layout.setSpacing(5)

        # Left side: Signal plots with timeline
        signal_layout = QVBoxLayout()
        signal_layout.setSpacing(3)

        # Canvas for signal plots
        self.figure = Figure(figsize=(8, 4), dpi=80)
        self.canvas = CustomCanvas(self.figure)
        self.canvas.setMinimumHeight(180)
        signal_layout.addWidget(self.canvas)

        # Navigation slider
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(5)

        self.nav_label = QLabel("Pos: 0.0s")
        self.nav_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        self.nav_label.setMinimumWidth(60)
        nav_layout.addWidget(self.nav_label)

        self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        self.nav_slider.setMinimum(0)
        self.nav_slider.setMaximum(1000)
        self.nav_slider.setValue(0)
        self.nav_slider.valueChanged.connect(self._on_nav_changed)
        self.nav_slider.setStyleSheet(
            f"""
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
        """
        )
        nav_layout.addWidget(self.nav_slider)

        self.duration_label = QLabel("/ 0.0s")
        self.duration_label.setStyleSheet(
            f"color: {self.theme.get('text', '#212529')};"
        )
        nav_layout.addWidget(self.duration_label)

        signal_layout.addLayout(nav_layout)

        # Timeline with annotations (no drag markers - display only)
        from .signal_editor import SignalCutterTimeline

        self.timeline = SignalCutterTimeline(theme=self.theme, show_markers=False)
        self.timeline.setMinimumHeight(60)
        self.timeline.setMaximumHeight(80)
        signal_layout.addWidget(self.timeline)

        content_layout.addLayout(signal_layout, stretch=3)

        # Right side: Band power displays (stacked vertically)
        from .band_power_display import BandPowerComparisonWidget

        band_power_layout = QVBoxLayout()
        band_power_layout.setSpacing(5)

        # Range 1 band power widget (typically Eyes Closed - displayed first)
        self.band_power_widget_range1 = BandPowerComparisonWidget(self.theme)
        self.band_power_widget_range1.setMinimumWidth(250)
        self.band_power_widget_range1.setMaximumWidth(320)
        band_power_layout.addWidget(self.band_power_widget_range1)

        # Range 2 band power widget (typically Eyes Open - displayed second)
        self.band_power_widget_range2 = BandPowerComparisonWidget(self.theme)
        self.band_power_widget_range2.setMinimumWidth(250)
        self.band_power_widget_range2.setMaximumWidth(320)
        band_power_layout.addWidget(self.band_power_widget_range2)

        content_layout.addLayout(band_power_layout, stretch=1)

        layout.addLayout(content_layout)

        # Initial empty plot
        self.show_empty_plot()

    def _on_view_changed(self, text: str):
        """Handle view window change."""
        if text == "Full":
            self._view_window = self._max_time
        else:
            self._view_window = float(text.replace("s", ""))
        self._update_signal_plot()

    def _on_nav_changed(self, value: int):
        """Handle navigation slider change."""
        max_start = max(0, self._max_time - self._view_window)
        self._view_start = (value / 1000.0) * max_start
        self.nav_label.setText(f"Pos: {self._view_start:.1f}s")
        self._update_signal_plot()

    def set_update_callback(self, callback):
        """Set callback for preview update"""
        self.update_callback = callback

    def set_channel_data(self, raw):
        """Update dropdown with available channels and extract annotation time ranges"""
        self.channel_names = raw.ch_names
        self.channel_dropdown.clear()
        self.channel_dropdown.addItems(self.channel_names)
        self.selected_channel_idx = 0

        # Store max time and update duration label
        self._max_time = raw.times[-1]
        self.duration_label.setText(f"/ {self._max_time:.1f}s")
        self.timeline.set_max_time(self._max_time)

        # Only extract from annotations if custom ranges weren't set
        if not self._custom_ranges_set:
            # Extract Eyes Closed and Eyes Open annotation time ranges
            # Range 1 = Eyes Closed (first), Range 2 = Eyes Open (second)
            self.range1 = None  # Eyes Closed
            self.range2 = None  # Eyes Open
            annotations_list = []

            if raw.annotations and len(raw.annotations) > 0:
                for annot in raw.annotations:
                    desc_lower = annot["description"].lower()
                    onset = float(annot["onset"])
                    duration = float(annot["duration"])
                    end_time = onset + duration

                    # Add to annotations list for timeline display
                    annotations_list.append(
                        {
                            "onset": onset,
                            "duration": duration,
                            "description": annot["description"],
                        }
                    )

                    # Eyes Closed goes to Range 1 (displayed first)
                    if "eyes closed" in desc_lower or "closed" in desc_lower:
                        if self.range1 is None:
                            self.range1 = (onset, end_time)
                    # Eyes Open goes to Range 2 (displayed second)
                    elif "eyes open" in desc_lower or "open" in desc_lower:
                        if self.range2 is None:
                            self.range2 = (onset, end_time)

            # Set annotations on timeline
            self.timeline.set_annotations(annotations_list)
        else:
            # Custom ranges are set, just update timeline with annotations
            annotations_list = []
            if raw.annotations and len(raw.annotations) > 0:
                for annot in raw.annotations:
                    annotations_list.append(
                        {
                            "onset": float(annot["onset"]),
                            "duration": float(annot["duration"]),
                            "description": annot["description"],
                        }
                    )
            self.timeline.set_annotations(annotations_list)

        # Update view window if needed
        if self.view_combo.currentText() == "Full":
            self._view_window = self._max_time

    def set_frequency_ranges(self, frequency_ranges: Dict[str, Tuple[float, float]]):
        """
        Set custom frequency analysis ranges from the preview screen.

        Args:
            frequency_ranges: Dictionary with 'range1' and 'range2' keys,
                             each containing (start, end) tuples.
                             Range 1 = Eyes Closed (displayed first)
                             Range 2 = Eyes Open (displayed second)
        """
        if not frequency_ranges:
            return

        if "range1" in frequency_ranges:
            self.range1 = frequency_ranges["range1"]
        if "range2" in frequency_ranges:
            self.range2 = frequency_ranges["range2"]

        self._custom_ranges_set = True

        # If we have data, update the displays
        if self._original_raw is not None:
            self._update_band_power_displays()

    def _on_channel_changed(self, index):
        """Called when channel selection changes"""
        self.selected_channel_idx = index
        # Trigger preview update if we have a callback
        if self.update_callback:
            self.update_callback()

    def show_empty_plot(self):
        """Display empty plot with instructions"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Select components to see preview of cleaned signal",
            ha="center",
            va="center",
            fontsize=12,
            color=self.theme.get("text_light", "#6c757d"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        self.canvas.draw()

    def update_preview(self, original_raw, cleaned_raw):
        """Update preview with new data - stores raw data and updates plot"""
        # Store raw data for navigation
        self._original_raw = original_raw
        self._cleaned_raw = cleaned_raw

        # Initialize band power analyzer if needed
        if self.band_power_analyzer is None:
            from backend.band_power_analyzer import BandPowerAnalyzer

            self.band_power_analyzer = BandPowerAnalyzer()

        # Update signal plot
        self._update_signal_plot()

        # Update band power comparisons (these use full annotation time ranges)
        self._update_band_power_displays()

    def _update_signal_plot(self):
        """Update the signal plot with current view window and position"""
        if self._original_raw is None:
            self.show_empty_plot()
            return

        try:
            self.figure.clear()

            sfreq = self._original_raw.info["sfreq"]
            channel_idx = self.selected_channel_idx

            # Calculate sample indices for current view
            start_sample = int(self._view_start * sfreq)
            end_sample = int((self._view_start + self._view_window) * sfreq)

            # Get data for current view window
            original_data = self._original_raw.get_data()[:, start_sample:end_sample]
            time_points = np.arange(original_data.shape[1]) / sfreq + self._view_start

            channel_name = (
                self.channel_names[channel_idx]
                if channel_idx < len(self.channel_names)
                else f"Channel {channel_idx}"
            )

            if self._cleaned_raw is not None:
                cleaned_data = self._cleaned_raw.get_data()[:, start_sample:end_sample]

                # Ensure data matches
                min_samples = min(original_data.shape[1], cleaned_data.shape[1])
                original_data = original_data[:, :min_samples]
                cleaned_data = cleaned_data[:, :min_samples]
                time_points = time_points[:min_samples]

                # For Wavelet mode, show 3 plots: original, cleaned, and FFT comparison
                if self._analysis_method == "WAVELETS":
                    ax1 = self.figure.add_subplot(3, 1, 1)
                    ax2 = self.figure.add_subplot(3, 1, 2)
                    ax3 = self.figure.add_subplot(3, 1, 3)
                else:
                    # Two subplots - original and cleaned
                    ax1 = self.figure.add_subplot(2, 1, 1)
                    ax2 = self.figure.add_subplot(2, 1, 2)
                    ax3 = None

                # Original signal
                ax1.plot(
                    time_points,
                    original_data[channel_idx, :],
                    color=self.theme.get("danger", "#e74c3c"),
                    linewidth=0.8,
                    alpha=0.9,
                )
                ax1.set_title(
                    f"Original Signal - {channel_name}",
                    fontsize=9,
                    color=self.theme["text"],
                )
                ax1.set_ylabel("Amp (μV)", fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis="both", labelsize=7)

                # Cleaned signal
                ax2.plot(
                    time_points,
                    cleaned_data[channel_idx, :],
                    color=self.theme.get("success", "#27ae60"),
                    linewidth=0.8,
                    alpha=0.9,
                )
                ax2.set_title(
                    f"Cleaned Signal - {channel_name}",
                    fontsize=9,
                    color=self.theme["text"],
                )
                if ax3 is None:
                    ax2.set_xlabel("Time (s)", fontsize=8)
                ax2.set_ylabel("Amp (μV)", fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis="both", labelsize=7)

                # FFT comparison for Wavelet mode
                if ax3 is not None:
                    self._plot_fft_comparison_preview(
                        ax3,
                        original_data[channel_idx, :],
                        cleaned_data[channel_idx, :],
                        sfreq,
                        channel_name,
                    )

            else:
                # Only original signal
                ax = self.figure.add_subplot(111)
                ax.plot(
                    time_points,
                    original_data[channel_idx, :],
                    color=self.theme.get("primary", "#007AFF"),
                    linewidth=0.8,
                )
                ax.set_title(
                    f"Original Signal - {channel_name}",
                    fontsize=10,
                    color=self.theme["text"],
                )
                ax.set_xlabel("Time (s)", fontsize=9)
                ax.set_ylabel("Amplitude (μV)", fontsize=9)
                ax.grid(True, alpha=0.3)

            self.figure.tight_layout(pad=0.5)
            self.canvas.draw()

        except Exception as e:
            print(f"Error updating signal plot: {str(e)}")
            self.show_error_plot(str(e))

    def _plot_fft_comparison_preview(
        self, ax, original_data, cleaned_data, sfreq, channel_name
    ):
        """
        Plot FFT comparison of original vs cleaned signal for Wavelet preview.

        Args:
            ax: Matplotlib axis to plot on
            original_data: Original signal data (1D array)
            cleaned_data: Cleaned signal data (1D array)
            sfreq: Sampling frequency
            channel_name: Name of the channel
        """
        from scipy import signal as scipy_signal

        # Use Welch's method for smoother PSD estimate
        n = len(original_data)
        nperseg = min(1024, n // 4) if n > 4 else n

        if nperseg < 4:
            ax.text(0.5, 0.5, "Not enough data for FFT", ha="center", va="center")
            return

        freq_orig, psd_orig = scipy_signal.welch(
            original_data, fs=sfreq, nperseg=nperseg
        )
        freq_clean, psd_clean = scipy_signal.welch(
            cleaned_data, fs=sfreq, nperseg=nperseg
        )

        # Limit frequency range to 0-50 Hz (typical EEG range)
        max_freq = min(50, sfreq / 2)
        freq_mask = freq_orig <= max_freq

        # Plot both spectra
        ax.semilogy(
            freq_orig[freq_mask],
            psd_orig[freq_mask],
            color=self.theme.get("danger", "#e74c3c"),
            linewidth=1,
            alpha=0.8,
            label="Original",
        )
        ax.semilogy(
            freq_clean[freq_mask],
            psd_clean[freq_mask],
            color=self.theme.get("success", "#27ae60"),
            linewidth=1,
            alpha=0.8,
            label="Cleaned",
        )

        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.set_title(
            f"FFT Comparison - {channel_name}", fontsize=9, color=self.theme["text"]
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(0, max_freq)

    def _update_band_power_displays(self):
        """Update the band power comparison displays for Range 1 and Range 2.

        Range 1 = Eyes Closed (displayed first)
        Range 2 = Eyes Open (displayed second)
        """
        if self._original_raw is None:
            self.band_power_widget_range1.clear()
            self.band_power_widget_range2.clear()
            return

        channel_idx = self.selected_channel_idx

        # Compute and display band power comparison for Range 1 (Eyes Closed)
        try:
            if self.range1 is not None:
                tmin_r1, tmax_r1 = self.range1
                original_powers_r1 = (
                    self.band_power_analyzer.compute_band_power_for_raw(
                        self._original_raw,
                        channel_idx=channel_idx,
                        tmin=tmin_r1,
                        tmax=tmax_r1,
                    )
                )
                if self._cleaned_raw is not None:
                    cleaned_powers_r1 = (
                        self.band_power_analyzer.compute_band_power_for_raw(
                            self._cleaned_raw,
                            channel_idx=channel_idx,
                            tmin=tmin_r1,
                            tmax=tmax_r1,
                        )
                    )
                else:
                    cleaned_powers_r1 = original_powers_r1
                self.band_power_widget_range1.update_comparison(
                    original_powers_r1, cleaned_powers_r1, title="😌 Eyes Closed"
                )
            else:
                self.band_power_widget_range1.clear()
        except Exception as bp_error:
            print(f"Error computing Range 1 (Eyes Closed) band power: {bp_error}")
            self.band_power_widget_range1.clear()

        # Compute and display band power comparison for Range 2 (Eyes Open)
        try:
            if self.range2 is not None:
                tmin_r2, tmax_r2 = self.range2
                original_powers_r2 = (
                    self.band_power_analyzer.compute_band_power_for_raw(
                        self._original_raw,
                        channel_idx=channel_idx,
                        tmin=tmin_r2,
                        tmax=tmax_r2,
                    )
                )
                if self._cleaned_raw is not None:
                    cleaned_powers_r2 = (
                        self.band_power_analyzer.compute_band_power_for_raw(
                            self._cleaned_raw,
                            channel_idx=channel_idx,
                            tmin=tmin_r2,
                            tmax=tmax_r2,
                        )
                    )
                else:
                    cleaned_powers_r2 = original_powers_r2
                self.band_power_widget_range2.update_comparison(
                    original_powers_r2, cleaned_powers_r2, title="👁️ Eyes Open"
                )
            else:
                self.band_power_widget_range2.clear()
        except Exception as bp_error:
            print(f"Error computing Range 2 (Eyes Open) band power: {bp_error}")
            self.band_power_widget_range2.clear()

    def show_error_plot(self, error_msg: str):
        """Display error message"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Preview error:\n{error_msg}",
            ha="center",
            va="center",
            fontsize=10,
            color=self.theme.get("danger", "#e74c3c"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()


class ComponentDisplayWidget(QWidget):
    def __init__(self, component_idx: int, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.component_idx = component_idx
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        # Horizontal layout for time-series and topomap
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Figure for time-series (left)
        self.timeseries_figure = Figure(figsize=(4, 2.5), dpi=90)
        self.timeseries_canvas = CustomCanvas(self.timeseries_figure)
        layout.addWidget(self.timeseries_canvas, 2)  # 2/3 of space

        # Figure for topomap (right)
        self.topomap_figure = Figure(figsize=(2.5, 2.5), dpi=90)
        self.topomap_canvas = CustomCanvas(self.topomap_figure)
        layout.addWidget(self.topomap_canvas, 1)  # 1/3 of space

    def plot_component(self, ica, raw, is_artifact, component_info):
        try:
            # 1. Time-series plot (left)
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)

            sources = ica.get_sources(raw).get_data()
            comp_data = sources[self.component_idx]
            times = raw.times[: len(comp_data)]
            color = (
                self.theme.get("danger", "#e74c3c")
                if is_artifact
                else self.theme.get("success", "#27ae60")
            )

            ax_time.plot(times, comp_data, color=color, linewidth=1)
            ax_time.set_title(
                f"IC {self.component_idx} - Time Series",
                fontsize=9,
                color=self.theme["text"],
            )
            ax_time.grid(True, linestyle="--", alpha=0.5)
            ax_time.set_xlabel("Time (s)", fontsize=8)
            ax_time.set_ylabel("Amplitude", fontsize=8)
            self.timeseries_figure.tight_layout(pad=0.3)

            # 2. Topographic map (right)
            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)

            # Get spatial patterns of the component
            component_weights = ica.get_components()[:, self.component_idx]

            # Topographic display with MNE
            import mne.viz

            mne.viz.plot_topomap(
                component_weights,
                raw.info,
                axes=ax_topo,
                show=False,
                cmap="RdBu_r",
                sensors=True,
            )
            ax_topo.set_title(
                f"IC {self.component_idx} - Topomap",
                fontsize=9,
                color=self.theme["text"],
            )
            self.topomap_figure.tight_layout(pad=0.3)

        except Exception as e:
            # In case of error, display message in both plots
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)
            ax_time.text(
                0.5,
                0.5,
                f"Time series error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)
            ax_topo.text(
                0.5,
                0.5,
                f"Topomap error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

        # Refresh both canvases
        self.timeseries_canvas.draw()
        self.topomap_canvas.draw()

    def plot_component_generic(self, processor, raw, is_artifact, method="PCA"):
        """
        Plot component using generic processor (works for PCA and Wavelets)

        Args:
            processor: Component processor (PCAProcessor, WaveletProcessor, or similar)
            raw: Raw EEG data
            is_artifact: Whether this component is suggested as artifact
            method: Analysis method name ("ICA", "PCA", or "WAVELETS")
        """
        try:
            # 1. Time-series plot (left)
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)

            sources = processor.get_sources_data()
            comp_data = sources[self.component_idx]
            times = raw.times[: len(comp_data)]
            color = (
                self.theme.get("danger", "#e74c3c")
                if is_artifact
                else self.theme.get("success", "#27ae60")
            )

            # Determine label based on method
            if method == "ICA":
                comp_label = f"IC {self.component_idx}"
            elif method == "WAVELETS":
                # Use actual channel name for Wavelet
                if raw is not None and self.component_idx < len(raw.ch_names):
                    comp_label = raw.ch_names[self.component_idx]
                else:
                    comp_label = f"CH {self.component_idx}"
            else:
                comp_label = f"PC {self.component_idx}"

            ax_time.plot(times, comp_data, color=color, linewidth=1)
            ax_time.set_title(
                f"{comp_label} - Time Series",
                fontsize=9,
                color=self.theme["text"],
            )
            ax_time.grid(True, linestyle="--", alpha=0.5)
            ax_time.set_xlabel("Time (s)", fontsize=8)
            ax_time.set_ylabel("Amplitude", fontsize=8)
            self.timeseries_figure.tight_layout(pad=0.3)

            # 2. Topographic map (right) or info panel for Wavelet
            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)

            # Get spatial patterns from processor
            components = processor.get_components()
            if components is not None:
                component_weights = components[:, self.component_idx]

                # Topographic display with MNE
                import mne.viz

                mne.viz.plot_topomap(
                    component_weights,
                    raw.info,
                    axes=ax_topo,
                    show=False,
                    cmap="RdBu_r",
                    sensors=True,
                )
                ax_topo.set_title(
                    f"{comp_label} - Topomap",
                    fontsize=9,
                    color=self.theme["text"],
                )
            else:
                # No spatial data - show FFT comparison for Wavelet, text for others
                if method == "WAVELETS":
                    # Show FFT/Frequency spectrum comparison for Wavelet mode
                    # Get denoised data from processor
                    denoised_data = processor.get_denoised_data()
                    if denoised_data is not None:
                        denoised_comp = denoised_data[self.component_idx]
                        self._plot_fft_comparison(
                            ax_topo,
                            comp_data,
                            denoised_comp,
                            raw.info["sfreq"],
                            comp_label,
                        )
                    else:
                        # Fallback to single FFT if no denoised data
                        self._plot_fft_single(
                            ax_topo, comp_data, raw.info["sfreq"], comp_label
                        )
                else:
                    ax_topo.text(
                        0.5,
                        0.5,
                        "No spatial data",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    ax_topo.set_title(
                        f"{comp_label}",
                        fontsize=9,
                        color=self.theme["text"],
                    )
                    ax_topo.set_xticks([])
                    ax_topo.set_yticks([])

            self.topomap_figure.tight_layout(pad=0.3)

        except Exception as e:
            # Error handling
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)
            ax_time.text(
                0.5,
                0.5,
                f"Time series error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)
            ax_topo.text(
                0.5,
                0.5,
                f"Topomap error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

        # Refresh canvases
        self.timeseries_canvas.draw()
        self.topomap_canvas.draw()

    def _plot_fft_comparison(self, ax, original_data, denoised_data, sfreq, label):
        """
        Plot FFT comparison of original vs denoised signal.

        Args:
            ax: Matplotlib axis to plot on
            original_data: Original signal data (1D array)
            denoised_data: Denoised signal data (1D array)
            sfreq: Sampling frequency
            label: Channel/component label
        """
        from scipy import signal as scipy_signal

        # Compute FFT for both signals
        n = len(original_data)
        freq = np.fft.rfftfreq(n, 1 / sfreq)

        # Use Welch's method for smoother PSD estimate
        nperseg = min(1024, n // 4)
        freq_orig, psd_orig = scipy_signal.welch(
            original_data, fs=sfreq, nperseg=nperseg
        )
        freq_clean, psd_clean = scipy_signal.welch(
            denoised_data, fs=sfreq, nperseg=nperseg
        )

        # Limit frequency range to 0-50 Hz (typical EEG range)
        max_freq = min(50, sfreq / 2)
        freq_mask = freq_orig <= max_freq

        # Plot both spectra
        ax.semilogy(
            freq_orig[freq_mask],
            psd_orig[freq_mask],
            color=self.theme.get("danger", "#e74c3c"),
            linewidth=1,
            alpha=0.8,
            label="Original",
        )
        ax.semilogy(
            freq_clean[freq_mask],
            psd_clean[freq_mask],
            color=self.theme.get("success", "#27ae60"),
            linewidth=1,
            alpha=0.8,
            label="Cleaned",
        )

        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.set_ylabel("PSD", fontsize=7)
        ax.set_title(f"{label} - FFT", fontsize=9, color=self.theme["text"])
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xlim(0, max_freq)

    def _plot_fft_single(self, ax, data, sfreq, label):
        """
        Plot single FFT spectrum (fallback when no denoised data).

        Args:
            ax: Matplotlib axis to plot on
            data: Signal data (1D array)
            sfreq: Sampling frequency
            label: Channel/component label
        """
        from scipy import signal as scipy_signal

        n = len(data)
        nperseg = min(1024, n // 4)
        freq, psd = scipy_signal.welch(data, fs=sfreq, nperseg=nperseg)

        # Limit frequency range to 0-50 Hz
        max_freq = min(50, sfreq / 2)
        freq_mask = freq <= max_freq

        ax.semilogy(
            freq[freq_mask],
            psd[freq_mask],
            color=self.theme.get("primary", "#007AFF"),
            linewidth=1,
        )

        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.set_ylabel("PSD", fontsize=7)
        ax.set_title(f"{label} - FFT", fontsize=9, color=self.theme["text"])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xlim(0, max_freq)


class ICAComponentSelector(QWidget):
    """
    Component selector widget for ICA, PCA, and Wavelet analysis.

    Provides an interactive UI for selecting components to remove during
    artifact cleaning. Supports multiple analysis methods and displays
    real-time preview of cleaning results.

    Attributes:
        components_selected: Signal emitted when component selection is confirmed.
            Emits list of selected component indices.
        back_requested: Signal emitted when user wants to return to previous screen.

    Args:
        theme: Dictionary containing UI color scheme.
        parent: Optional parent widget.

    Example:
        >>> selector = ICAComponentSelector(theme)
        >>> selector.set_ica_data(ica, raw, suggested, info, explanations)
        >>> selector.components_selected.connect(on_components_selected)
    """

    components_selected = pyqtSignal(list)
    back_requested = pyqtSignal()  # Signal emitted when user wants to go back

    def __init__(self, theme: Dict[str, str], parent=None):
        """Initialize the component selector with theme and default state."""
        super().__init__(parent)
        self.theme = theme
        self.ica = None  # Can be ICA object or None for PCA
        self.pca = None  # PCA processor for PCA analysis
        self.processor = None  # Generic processor reference
        self.raw = None
        self.suggested_artifacts = []
        self.checkboxes = {}
        self.component_widgets = {}
        self.components_info = {}
        self.explanations = {}
        self.analysis_method = "ICA"  # Track current method
        self.n_components = 0  # Track number of components

        # Wavelet-specific state
        self.wavelet_info = {}
        self.noise_reduction_stats = {}

        # Preview functionality
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)  # Μόνο μία φορά όταν λήξει
        self.preview_timer.timeout.connect(self._start_preview_update)
        self.preview_thread = None

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface components and layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        header_layout = QHBoxLayout()

        # Back button
        self.back_btn = QPushButton("⬅️ Back to Preview")
        self.back_btn.setMinimumHeight(40)
        self.back_btn.clicked.connect(self._on_back_clicked)
        self.back_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme.get('secondary', '#6c757d')};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('text_light', '#5a6268')};
            }}
        """
        )
        header_layout.addWidget(self.back_btn)

        header_layout.addStretch()

        self.title_label = QLabel("🔍 Select Components for Removal")
        self.title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        controls_layout = QHBoxLayout()
        self.select_suggested_btn = QPushButton("Select Suggested")
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Clear Selection")
        controls_layout.addWidget(self.select_suggested_btn)
        controls_layout.addWidget(self.select_all_btn)
        controls_layout.addWidget(self.select_none_btn)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Wavelet configuration panel (visible only for WAVELETS method)
        self._create_wavelet_config_panel()
        main_layout.addWidget(self.wavelet_config_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ border: none; background: {self.theme['background']}; width: 12px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: #bdc3c7; min-height: 20px; border-radius: 6px; }}
            QScrollBar::handle:vertical:hover {{ background: #95a5a6; }}
        """
        )

        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(0, 0, 5, 0)
        self.components_layout.setSpacing(10)
        self.scroll_area.setWidget(self.components_widget)
        main_layout.addWidget(self.scroll_area, 1)  # Μικρότερο stretch factor

        # Add Preview Widget
        self.preview_widget = PreviewWidget(self.theme)
        self.preview_widget.setMinimumHeight(300)  # Minimum height for preview
        main_layout.addWidget(self.preview_widget, 1)  # Equal space with scroll area

        self.apply_btn = QPushButton("✅ Apply Cleaning and Save")
        self.apply_btn.setMinimumHeight(50)
        self.apply_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        main_layout.addWidget(self.apply_btn)

        self.apply_styling()

    def _create_wavelet_config_panel(self):
        """Create the wavelet configuration panel (visible only for WAVELETS method)."""
        self.wavelet_config_panel = QGroupBox("🌊 Wavelet Configuration")
        self.wavelet_config_panel.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.wavelet_config_panel.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme.get('primary', '#007AFF')};
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                background-color: #f8f9fa;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {self.theme.get('primary', '#007AFF')};
                background-color: #f8f9fa;
            }}
        """
        )
        panel_layout = QVBoxLayout(self.wavelet_config_panel)
        panel_layout.setSpacing(10)

        # Row 1: Wavelet family and Level
        row1 = QHBoxLayout()

        # Wavelet family
        row1.addWidget(QLabel("Wavelet:"))
        self.wavelet_combo = QComboBox()
        wavelet_families = {
            "db4": "Daubechies 4",
            "db8": "Daubechies 8",
            "sym4": "Symlet 4",
            "sym8": "Symlet 8",
            "coif3": "Coiflet 3",
            "bior3.5": "Biorthogonal 3.5",
        }
        for key, name in wavelet_families.items():
            self.wavelet_combo.addItem(name, key)
        self.wavelet_combo.currentIndexChanged.connect(self._on_wavelet_param_changed)
        row1.addWidget(self.wavelet_combo)

        row1.addSpacing(20)

        # Level
        row1.addWidget(QLabel("Level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 10)
        self.level_spin.setValue(5)
        self.level_spin.valueChanged.connect(self._on_wavelet_param_changed)
        row1.addWidget(self.level_spin)

        row1.addStretch()
        panel_layout.addLayout(row1)

        # Row 2: Threshold mode and method
        row2 = QHBoxLayout()

        # Threshold mode
        row2.addWidget(QLabel("Threshold Mode:"))
        self.threshold_mode_combo = QComboBox()
        self.threshold_mode_combo.addItem("Soft (smoother)", "soft")
        self.threshold_mode_combo.addItem("Hard (sharper)", "hard")
        self.threshold_mode_combo.currentIndexChanged.connect(
            self._on_wavelet_param_changed
        )
        row2.addWidget(self.threshold_mode_combo)

        row2.addSpacing(20)

        # Threshold method
        row2.addWidget(QLabel("Method:"))
        self.threshold_method_combo = QComboBox()
        threshold_methods = {
            "visushrink": "VisuShrink (Universal)",
            "bayeshrink": "BayesShrink (Adaptive)",
            "sureshrink": "SUREShrink (Optimal)",
            "manual": "Manual (Custom)",
        }
        for key, name in threshold_methods.items():
            self.threshold_method_combo.addItem(name, key)
        self.threshold_method_combo.currentIndexChanged.connect(
            self._on_threshold_method_changed
        )
        row2.addWidget(self.threshold_method_combo)

        row2.addStretch()
        panel_layout.addLayout(row2)

        # Row 3: Threshold scale (for automatic methods) or manual threshold
        row3 = QHBoxLayout()

        # Threshold scale slider (for automatic methods)
        self.threshold_scale_widget = QWidget()
        scale_layout = QHBoxLayout(self.threshold_scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(QLabel("Threshold Scale:"))
        self.threshold_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_scale_slider.setMinimum(25)  # 0.25
        self.threshold_scale_slider.setMaximum(300)  # 3.0
        self.threshold_scale_slider.setValue(100)  # 1.0
        self.threshold_scale_slider.setMinimumWidth(150)
        self.threshold_scale_slider.valueChanged.connect(self._on_wavelet_param_changed)
        scale_layout.addWidget(self.threshold_scale_slider)
        self.threshold_scale_label = QLabel("1.00×")
        self.threshold_scale_label.setMinimumWidth(50)
        self.threshold_scale_label.setStyleSheet(
            f"color: {self.theme.get('primary', '#007AFF')}; font-weight: bold;"
        )
        scale_layout.addWidget(self.threshold_scale_label)
        scale_layout.addWidget(QLabel("(<1=more denoising, >1=less denoising)"))
        scale_layout.addStretch()
        row3.addWidget(self.threshold_scale_widget)

        # Manual threshold slider (for manual mode)
        self.manual_threshold_widget = QWidget()
        manual_layout = QHBoxLayout(self.manual_threshold_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addWidget(QLabel("Manual Threshold:"))
        self.manual_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.manual_threshold_slider.setMinimum(0)
        self.manual_threshold_slider.setMaximum(200)  # 0 to 2.0
        self.manual_threshold_slider.setValue(10)  # 0.1
        self.manual_threshold_slider.setMinimumWidth(150)
        self.manual_threshold_slider.valueChanged.connect(
            self._on_wavelet_param_changed
        )
        manual_layout.addWidget(self.manual_threshold_slider)
        self.manual_threshold_label = QLabel("0.10")
        self.manual_threshold_label.setMinimumWidth(50)
        self.manual_threshold_label.setStyleSheet(
            f"color: {self.theme.get('primary', '#007AFF')}; font-weight: bold;"
        )
        manual_layout.addWidget(self.manual_threshold_label)
        manual_layout.addWidget(QLabel("(lower=more denoising)"))
        manual_layout.addStretch()
        row3.addWidget(self.manual_threshold_widget)

        # Initially show scale widget, hide manual widget
        self.manual_threshold_widget.setVisible(False)

        panel_layout.addLayout(row3)

        # Initially hide the entire panel (will be shown when WAVELETS method is selected)
        self.wavelet_config_panel.setVisible(False)

    def _on_threshold_method_changed(self):
        """Handle threshold method change - show/hide appropriate controls."""
        method = self.threshold_method_combo.currentData()
        is_manual = method == "manual"
        self.threshold_scale_widget.setVisible(not is_manual)
        self.manual_threshold_widget.setVisible(is_manual)
        self._on_wavelet_param_changed()

    def _on_wavelet_param_changed(self):
        """Handle wavelet parameter change - trigger recalculation."""
        # Update labels
        scale_value = self.threshold_scale_slider.value() / 100.0
        self.threshold_scale_label.setText(f"{scale_value:.2f}×")

        manual_value = self.manual_threshold_slider.value() / 100.0
        self.manual_threshold_label.setText(f"{manual_value:.2f}")

        # If processor is available and it's a wavelet processor, update parameters
        if self.processor is not None and hasattr(self.processor, "set_wavelet_params"):
            try:
                self.processor.set_wavelet_params(
                    wavelet=self.wavelet_combo.currentData(),
                    level=self.level_spin.value(),
                    threshold_mode=self.threshold_mode_combo.currentData(),
                    threshold_method=self.threshold_method_combo.currentData(),
                    threshold_scale=scale_value,
                    manual_threshold=manual_value,
                )
                # Trigger preview update
                self._start_preview_update()
            except Exception as e:
                print(f"Error updating wavelet parameters: {e}")

    def _on_back_clicked(self):
        """Handle back button click"""
        self.back_requested.emit()

        self.select_all_btn.clicked.connect(lambda: self.set_all_checkboxes(True))
        self.select_none_btn.clicked.connect(lambda: self.set_all_checkboxes(False))
        self.select_suggested_btn.clicked.connect(self.select_suggested)
        self.apply_btn.clicked.connect(self.emit_selected_components)

        # --- 3. REMOVED OLD EVENT FILTER ---
        # No longer needed, since we solved the problem at the source.
        # self.installEventFilter(self) <-- REMOVED

    def apply_styling(self):
        # Style the buttons
        btn_style = """
            QPushButton {
                background-color: #5D6D7E; color: white; padding: 10px;
                border: none; font-size: 12px; border-radius: 6px;
            }
            QPushButton:hover { background-color: #85929E; }
        """
        self.select_all_btn.setStyleSheet(btn_style)
        self.select_none_btn.setStyleSheet(btn_style)
        self.select_suggested_btn.setStyleSheet(btn_style)

        self.apply_btn.setStyleSheet(
            f"""
            QPushButton {{ background-color: {self.theme['success']}; color: white; border-radius: 8px; }}
            QPushButton:hover {{ background-color: {self.theme.get('success_hover', self.theme['success'])}; }}
        """
        )

    def create_single_component_widget(self, i):
        is_artifact = i in self.suggested_artifacts
        comp_container = QWidget()
        comp_container.setMinimumHeight(200)
        comp_layout = QHBoxLayout(comp_container)
        comp_layout.setContentsMargins(10, 5, 10, 5)

        # Create vertical layout for checkbox and new button
        controls_layout = QVBoxLayout()

        # Use appropriate label based on analysis method
        if self.analysis_method == "ICA":
            comp_label = f"IC {i}"
        elif self.analysis_method == "WAVELETS":
            # For Wavelet, use actual channel names
            if self.raw is not None and i < len(self.raw.ch_names):
                comp_label = f"🌊 {self.raw.ch_names[i]}"
            else:
                comp_label = f"CH {i}"
        else:
            comp_label = f"PC {i}"
        checkbox = QCheckBox(f" {comp_label}")
        checkbox.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        checkbox.setChecked(is_artifact)
        checkbox.setStyleSheet(f"color: {self.theme['text_light']}; border: none;")
        checkbox.toggled.connect(
            lambda state, widget=comp_container: self.update_selection_style(
                widget, state
            )
        )
        checkbox.toggled.connect(self._on_checkbox_toggled)  # Add for preview
        self.checkboxes[i] = checkbox

        # The new "Analyze" button
        details_btn = QPushButton("🔎 Analyze")
        details_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #e9ecef; /* Απαλό γκρι φόντο */
                color: {self.theme.get('text_light', '#6c757d')}; /* Πιο σκούρο κείμενο */
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                font-size: 11px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #d4e6f1; /* Απαλό μπλε στο hover */
                border-color: {self.theme.get('primary', '#007AFF')};
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        details_btn.clicked.connect(
            lambda state, idx=i: self.show_component_properties(idx)
        )  # Connect to new function

        controls_layout.addWidget(checkbox)
        controls_layout.addWidget(details_btn)
        controls_layout.addStretch()

        plot_widget = ComponentDisplayWidget(i, self.theme)
        # Pass the appropriate data for plotting
        if self.ica is not None:
            plot_widget.plot_component(self.ica, self.raw, is_artifact, {})
        elif self.processor is not None:
            plot_widget.plot_component_generic(
                self.processor, self.raw, is_artifact, self.analysis_method
            )

        comp_layout.addLayout(controls_layout)  # Add layout with controls
        comp_layout.addWidget(plot_widget, 1)
        self.components_layout.addWidget(comp_container)

        self.update_selection_style(comp_container, checkbox.isChecked())

    def update_selection_style(self, widget: QWidget, is_selected: bool):
        # ... (Η συνάρτηση παραμένει ίδια)
        if is_selected:
            bg_color = "#fadbd8"
            border_color = self.theme["danger"]
        else:
            bg_color = "#e8f8f5"
            border_color = self.theme["success"]
        widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 5px;
            }}
        """
        )

    def set_all_checkboxes(self, state: bool):
        # ... (Η συνάρτηση παραμένει ίδια)
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)
        # Trigger preview update after setting all checkboxes
        self._on_checkbox_toggled()

    def select_suggested(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        for i, checkbox in self.checkboxes.items():
            checkbox.setChecked(i in self.suggested_artifacts)
        # Trigger preview update after selecting suggested
        self._on_checkbox_toggled()

    def emit_selected_components(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        selected = [i for i, cb in self.checkboxes.items() if cb.isChecked()]
        self.components_selected.emit(selected)

    def get_selected_components(self):
        """Return list of selected component indices"""
        return [i for i, cb in self.checkboxes.items() if cb.isChecked()]

    def select_all_components(self):
        """Select all components"""
        self.set_all_checkboxes(True)

    def select_no_components(self):
        """Deselect all components"""
        self.set_all_checkboxes(False)

    def select_suggested_components(self):
        """Select only the suggested artifact components"""
        self.select_suggested()

    def _on_checkbox_toggled(self):
        """Called when any checkbox is toggled - starts the preview update timer"""
        # Check if we have data to work with (ICA or processor)
        if (self.ica or self.processor) and self.raw:
            # Restart the timer - if the user makes quick changes,
            # wait 500ms from the last change
            self.preview_timer.stop()
            self.preview_timer.start(500)

    def _start_preview_update(self):
        """Starts the background thread to compute the cleaned signal"""
        if not self.raw:
            return

        if not self.ica and not self.processor:
            return

        # Cancel any previous thread if still running
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.quit()
            self.preview_thread.wait()

        # Get currently selected components
        selected_components = [i for i, cb in self.checkboxes.items() if cb.isChecked()]

        # Create and start new thread
        self.preview_thread = PreviewUpdateThread(
            self.ica,
            self.raw,
            selected_components,
            processor=self.processor,
            analysis_method=self.analysis_method,
        )
        self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
        self.preview_thread.start()

    def set_ica_data(
        self,
        ica=None,
        pca=None,
        processor=None,
        raw=None,
        suggested_artifacts=None,
        components_info=None,
        explanations=None,
        analysis_method="ICA",
        wavelet_info=None,
        noise_reduction_stats=None,
        **kwargs,
    ):
        """
        Set component data for visualization

        Supports both ICA and PCA analysis methods.

        Args:
            ica: MNE ICA object (for ICA analysis)
            pca: sklearn PCA object (for PCA analysis)
            processor: Generic component processor
            raw: Raw EEG data
            suggested_artifacts: List of suggested artifact component indices
            components_info: Dictionary with component statistics
            explanations: Dictionary with artifact explanations
            analysis_method: "ICA", "PCA", or "WAVELETS"
            wavelet_info: Wavelet configuration info (for WAVELETS method)
            noise_reduction_stats: Noise reduction statistics (for WAVELETS method)
        """
        self.ica = ica
        self.pca = pca
        self.processor = processor
        self.raw = raw
        self.suggested_artifacts = suggested_artifacts or []
        self.components_info = components_info or {}
        self.explanations = explanations or {}
        self.analysis_method = analysis_method
        self.wavelet_info = wavelet_info or {}
        self.noise_reduction_stats = noise_reduction_stats or {}

        # Determine number of components based on analysis method
        if ica is not None:
            self.n_components = ica.n_components_
        elif processor is not None:
            self.n_components = processor.n_components
        else:
            self.n_components = 0

        # Update title based on method
        if analysis_method == "WAVELETS":
            # Show wavelet configuration panel
            self.wavelet_config_panel.setVisible(True)

            # Populate wavelet controls with current settings
            wavelet_name = self.wavelet_info.get("wavelet", "db4")
            level = self.wavelet_info.get("level", 5)
            threshold_mode = self.wavelet_info.get("threshold_mode", "soft")
            threshold_method = self.wavelet_info.get("threshold_method", "visushrink")
            threshold_scale = self.wavelet_info.get("threshold_scale", 1.0)
            manual_threshold = self.wavelet_info.get("manual_threshold", 0.1)

            # Set combo boxes and sliders
            wavelet_idx = self.wavelet_combo.findData(wavelet_name)
            if wavelet_idx >= 0:
                self.wavelet_combo.setCurrentIndex(wavelet_idx)

            self.level_spin.setValue(level)

            mode_idx = self.threshold_mode_combo.findData(threshold_mode)
            if mode_idx >= 0:
                self.threshold_mode_combo.setCurrentIndex(mode_idx)

            method_idx = self.threshold_method_combo.findData(threshold_method)
            if method_idx >= 0:
                self.threshold_method_combo.setCurrentIndex(method_idx)

            self.threshold_scale_slider.setValue(int(threshold_scale * 100))
            self.manual_threshold_slider.setValue(int(manual_threshold * 100))

            # Update title
            method = threshold_method
            self.title_label.setText(f"🌊 Wavelet Denoising - Real-Time Configuration")
        else:
            # Hide wavelet configuration panel for ICA/PCA
            self.wavelet_config_panel.setVisible(False)
            self.title_label.setText(
                f"🔍 Select {analysis_method} Components for Removal"
            )

        # Clear existing components
        while self.components_layout.count():
            item = self.components_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.checkboxes.clear()
        self.component_widgets.clear()

        # Create component widgets
        for i in range(self.n_components):
            self.create_single_component_widget(i)
        self.components_layout.addStretch(1)

        # Update preview widget with channel data, method, and callback
        self.preview_widget.set_channel_data(raw)
        self.preview_widget.set_analysis_method(analysis_method)
        self.preview_widget.set_update_callback(self._start_preview_update)

        # Update initial preview with suggested components
        if self.suggested_artifacts:
            self._start_preview_update()

    def set_frequency_ranges(self, frequency_ranges: Dict[str, Tuple[float, float]]):
        """
        Set custom frequency analysis ranges from the preview screen.

        This method delegates to the preview widget to preserve user's
        custom frequency band analysis ranges when transitioning from
        the signal preview screen to the ICA/PCA component selector.

        Args:
            frequency_ranges: Dictionary with 'range1' and 'range2' keys,
                             each containing (start, end) tuples.
                             Range 1 = Eyes Closed (displayed first)
                             Range 2 = Eyes Open (displayed second)
        """
        if frequency_ranges and self.preview_widget:
            self.preview_widget.set_frequency_ranges(frequency_ranges)

    def _create_spectrogram_plot(self, component_idx):
        """
        Creates a spectrogram plot for the specific component.
        The spectrogram is ideal for detecting muscle artifacts that
        appear as short bursts of energy across a wide frequency range.
        Works for ICA, PCA, and Wavelets.
        """
        try:
            from scipy import signal

            # Get component data - use processor for both ICA and PCA
            if self.processor is not None:
                sources = self.processor.get_sources_data()
            elif self.ica is not None:
                sources = self.ica.get_sources(self.raw).get_data()
            else:
                return None

            component_data = sources[component_idx]
            if self.analysis_method == "ICA":
                comp_label = f"IC {component_idx}"
            elif self.analysis_method == "WAVELETS":
                # Use actual channel name for Wavelet
                if self.raw is not None and component_idx < len(self.raw.ch_names):
                    comp_label = self.raw.ch_names[component_idx]
                else:
                    comp_label = f"CH {component_idx}"
            else:
                comp_label = f"PC {component_idx}"

            # Parameters for spectrogram
            fs = self.raw.info["sfreq"]  # Sampling frequency

            # Calculate spectrogram
            # Using window that gives good time-frequency resolution
            nperseg = min(1024, len(component_data) // 8)  # Window size
            noverlap = nperseg // 2  # Window overlap

            frequencies, times, Sxx = signal.spectrogram(
                component_data,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
            )

            # Create figure
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)

            # Display spectrogram in dB scale for better visualization
            Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Add small value to avoid log(0)

            # Create the spectrogram plot
            im = ax.pcolormesh(
                times, frequencies, Sxx_db, shading="gouraud", cmap="viridis"
            )

            # Set axes and labels
            ax.set_ylabel("Frequency (Hz)", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_title(
                f"Spectrogram - {comp_label}\n(Time-Frequency Analysis for Muscle Artifact Detection)",
                fontsize=11,
                color=self.theme.get("text", "#000000"),
            )

            # Limit frequencies to range of interest (0-100 Hz typical for EEG)
            ax.set_ylim(0, min(100, fs / 2))

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
            cbar.ax.tick_params(labelsize=8)

            # Grid for better readability
            ax.grid(True, alpha=0.3)

            # Final layout
            fig.tight_layout(pad=2.0)

            return fig

        except Exception as e:
            print(f"Error creating spectrogram: {str(e)}")

            # In case of error, create a figure with error message
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error creating Spectrogram:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="red",
            )
            ax.set_title(
                f"Spectrogram - IC {component_idx} (Error)",
                color=self.theme.get("text", "#000000"),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            return fig

    def show_component_properties(self, component_idx):
        """
        Creates and displays a new window with the component properties.
        Includes topography, PSD and Spectrogram for full analysis.
        Works for ICA, PCA, and Wavelets analysis.
        """
        # Check we have data to work with
        if not self.raw:
            return
        if not self.ica and not self.processor:
            return

        if self.analysis_method == "ICA":
            comp_label = f"IC {component_idx}"
        elif self.analysis_method == "WAVELETS":
            # Use actual channel name for Wavelet
            if self.raw is not None and component_idx < len(self.raw.ch_names):
                comp_label = self.raw.ch_names[component_idx]
            else:
                comp_label = f"CH {component_idx}"
        else:
            comp_label = f"PC {component_idx}"

        # For ICA, use MNE's built-in plot_properties
        # For PCA and Wavelets, create custom plots
        if self.ica is not None:
            # MNE creates the plots. show=False is critical
            # to get the figures instead of displaying them directly.
            figures = self.ica.plot_properties(
                self.raw, picks=component_idx, show=False
            )

        else:
            # For PCA and Wavelets, create custom property plots
            figures = self._create_generic_property_plots(component_idx)

        # Create spectrogram plot
        spectrogram_fig = self._create_spectrogram_plot(component_idx)

        # Create new dialog window (pop-up)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Detailed Analysis of {comp_label}")
        dialog.setMinimumSize(1000, 800)  # Larger window for extra plot
        dialog_layout = QVBoxLayout(dialog)

        # Add title
        title_label = QLabel(f"🔬 {comp_label} Analysis")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.theme['text']}; margin: 10px; text-align: center;"
        )
        dialog_layout.addWidget(title_label)

        # For each figure created by MNE or custom, create a canvas
        for fig in figures:
            canvas = FigureCanvas(fig)
            dialog_layout.addWidget(canvas)

        # Add spectrogram at the end
        if spectrogram_fig:
            spectrogram_canvas = FigureCanvas(spectrogram_fig)
            dialog_layout.addWidget(spectrogram_canvas)

        # Show the window
        dialog.exec()

    def _create_generic_property_plots(self, component_idx):
        """
        Create property plots for PCA and Wavelet components.
        """
        figures = []

        try:
            from scipy import signal

            # Get component data from processor
            sources = self.processor.get_sources_data()
            comp_data = sources[component_idx]
            sfreq = self.raw.info["sfreq"]

            # Determine label based on method
            if self.analysis_method == "WAVELETS":
                comp_label = "CH"
                ch_name = (
                    self.raw.ch_names[component_idx]
                    if component_idx < len(self.raw.ch_names)
                    else f"Channel {component_idx}"
                )
            else:
                comp_label = "PC"
                ch_name = None

            # Figure 1: Time series and topomap (or info for Wavelet)
            fig1 = Figure(figsize=(10, 4), dpi=100)

            # Time series
            ax1 = fig1.add_subplot(121)
            times = self.raw.times[: len(comp_data)]
            ax1.plot(times, comp_data, linewidth=0.5)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"{comp_label} {component_idx} - Time Series")
            ax1.grid(True, alpha=0.3)

            # Topomap (or info panel for Wavelet)
            ax2 = fig1.add_subplot(122)
            components = self.processor.get_components()
            if components is not None:
                import mne.viz

                mne.viz.plot_topomap(
                    components[:, component_idx],
                    self.raw.info,
                    axes=ax2,
                    show=False,
                    cmap="RdBu_r",
                )
                ax2.set_title(f"{comp_label} {component_idx} - Topomap")
            else:
                # For Wavelet - show info panel instead
                if self.analysis_method == "WAVELETS":
                    # Get wavelet info if available
                    wavelet_info = getattr(
                        self.processor, "get_wavelet_info", lambda: {}
                    )()
                    ax2.text(
                        0.5,
                        0.7,
                        f"🌊 Wavelet Denoising",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax2.text(
                        0.5,
                        0.5,
                        f"Channel: {ch_name}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    if wavelet_info:
                        ax2.text(
                            0.5,
                            0.35,
                            f"Wavelet: {wavelet_info.get('wavelet', 'N/A')}",
                            ha="center",
                            va="center",
                            fontsize=10,
                        )
                        ax2.text(
                            0.5,
                            0.22,
                            f"Level: {wavelet_info.get('level', 'N/A')}",
                            ha="center",
                            va="center",
                            fontsize=10,
                        )
                    ax2.set_title(f"{comp_label} {component_idx} - Info")
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No spatial data",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    ax2.set_title(f"{comp_label} {component_idx}")
                ax2.set_xticks([])
                ax2.set_yticks([])

            fig1.tight_layout()
            figures.append(fig1)

            # Figure 2: PSD (Power Spectral Density)
            fig2 = Figure(figsize=(10, 3), dpi=100)
            ax3 = fig2.add_subplot(111)

            freqs, psd = signal.welch(
                comp_data, fs=sfreq, nperseg=min(1024, len(comp_data))
            )
            ax3.semilogy(freqs, psd)
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Power Spectral Density")
            ax3.set_title(f"{comp_label} {component_idx} - Power Spectrum")
            ax3.set_xlim(0, min(50, sfreq / 2))
            ax3.grid(True, alpha=0.3)

            fig2.tight_layout()
            figures.append(fig2)

        except Exception as e:
            # Create error figure
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating plots: {e}", ha="center", va="center")
            figures.append(fig)

        return figures
