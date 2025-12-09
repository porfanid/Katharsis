#!/usr/bin/env python3
"""
Channel Selector Component - Interactive channel selection interface
"""

from typing import Any, Dict, List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AnalysisMethodSelector(QWidget):
    """
    Analysis method selector widget for ICA/PCA/Wavelet selection.
    Displays three buttons for method selection with descriptions.
    Includes wavelet-specific settings (level, family, threshold mode) when Wavelet is selected.
    """

    method_changed = pyqtSignal(str)  # Emits "ICA", "PCA", or "WAVELETS"
    wavelet_level_changed = pyqtSignal(int)  # Emits wavelet level (1-10)
    wavelet_family_changed = pyqtSignal(str)  # Emits wavelet family (e.g., 'db4')
    wavelet_threshold_mode_changed = pyqtSignal(
        str
    )  # Emits threshold mode ('soft' or 'hard')
    wavelet_threshold_method_changed = pyqtSignal(
        str
    )  # Emits threshold method ('visushrink', 'bayeshrink', 'sureshrink')

    # Method definitions with icons, names, and descriptions
    METHODS = {
        "ICA": {
            "icon": "🧠",
            "name": "ICA",
            "description": "Best for detecting eye blinks and muscle artifacts",
        },
        "PCA": {
            "icon": "📊",
            "name": "PCA",
            "description": "Faster, ideal for quick preliminary analysis",
        },
        "WAVELETS": {
            "icon": "🌊",
            "name": "Wavelet",
            "description": "Best for low-channel systems (≤8 channels). Denoises all channels automatically.",
        },
    }

    # Available wavelet families with display names
    WAVELET_FAMILIES = {
        "db4": "Daubechies 4 (db4)",
        "db8": "Daubechies 8 (db8)",
        "sym4": "Symlet 4 (sym4)",
        "sym8": "Symlet 8 (sym8)",
        "coif3": "Coiflet 3 (coif3)",
        "bior3.5": "Biorthogonal 3.5 (bior3.5)",
    }

    # Threshold modes
    THRESHOLD_MODES = {
        "soft": "Soft (smoother)",
        "hard": "Hard (sharper)",
    }

    # Threshold methods with descriptions
    # Note: These descriptions should match WaveletProcessor.get_available_threshold_methods()
    THRESHOLD_METHODS = {
        "visushrink": "VisuShrink - Universal (conservative, good for general use)",
        "bayeshrink": "BayesShrink - Adaptive (data-driven, better for non-stationary noise)",
        "sureshrink": "SUREShrink - Optimal (MSE-minimizing, best for complex signals)",
    }

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self._selected_method = "ICA"  # Default method
        self._buttons: Dict[str, QPushButton] = {}
        self._wavelet_level = 5  # Default wavelet level
        self._wavelet_family = "db4"  # Default wavelet family
        self._threshold_mode = "soft"  # Default threshold mode
        self._threshold_method = "visushrink"  # Default threshold method
        self._setup_ui()

    def _setup_ui(self):
        """Create the method selector UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        for method_key, method_info in self.METHODS.items():
            btn = QPushButton(f"{method_info['icon']} {method_info['name']}")
            btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            btn.setFixedHeight(44)
            btn.setMinimumWidth(100)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda checked, m=method_key: self._on_method_clicked(m)
            )
            self._buttons[method_key] = btn
            buttons_layout.addWidget(btn)

        main_layout.addLayout(buttons_layout)

        # Wavelet settings (visible only when Wavelet is selected)
        self._wavelet_settings = QWidget()
        wavelet_main_layout = QVBoxLayout(self._wavelet_settings)
        wavelet_main_layout.setContentsMargins(0, 5, 0, 0)
        wavelet_main_layout.setSpacing(8)

        # First row: Level and Wavelet Family
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(15)

        # Level control
        level_label = QLabel("🔧 Level:")
        level_label.setFont(QFont("Arial", 10))
        level_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        row1_layout.addWidget(level_label)

        self._level_spinbox = QSpinBox()
        self._level_spinbox.setRange(1, 10)
        self._level_spinbox.setValue(self._wavelet_level)
        self._level_spinbox.setFont(QFont("Arial", 10))
        self._level_spinbox.setFixedWidth(60)
        self._level_spinbox.setStyleSheet(
            f"""
            QSpinBox {{
                padding: 5px;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                background-color: white;
            }}
            QSpinBox:focus {{
                border-color: {self.theme.get('primary', '#007AFF')};
            }}
            """
        )
        self._level_spinbox.valueChanged.connect(self._on_level_changed)
        row1_layout.addWidget(self._level_spinbox)

        level_hint = QLabel("(1-10)")
        level_hint.setFont(QFont("Arial", 9))
        level_hint.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')}; font-style: italic;"
        )
        row1_layout.addWidget(level_hint)

        row1_layout.addSpacing(20)

        # Wavelet Family selector
        family_label = QLabel("🌊 Wavelet:")
        family_label.setFont(QFont("Arial", 10))
        family_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        row1_layout.addWidget(family_label)

        self._family_combo = QComboBox()
        self._family_combo.setFont(QFont("Arial", 10))
        self._family_combo.setMinimumWidth(150)
        for key, display_name in self.WAVELET_FAMILIES.items():
            self._family_combo.addItem(display_name, key)
        self._family_combo.setCurrentIndex(0)  # Default to db4
        self._family_combo.setStyleSheet(
            f"""
            QComboBox {{
                padding: 5px 10px;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                background-color: white;
            }}
            QComboBox:focus {{
                border-color: {self.theme.get('primary', '#007AFF')};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}
            """
        )
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        row1_layout.addWidget(self._family_combo)

        row1_layout.addStretch()
        wavelet_main_layout.addLayout(row1_layout)

        # Second row: Threshold Mode
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)

        threshold_label = QLabel("⚙️ Threshold:")
        threshold_label.setFont(QFont("Arial", 10))
        threshold_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        row2_layout.addWidget(threshold_label)

        self._threshold_combo = QComboBox()
        self._threshold_combo.setFont(QFont("Arial", 10))
        self._threshold_combo.setMinimumWidth(130)
        for key, display_name in self.THRESHOLD_MODES.items():
            self._threshold_combo.addItem(display_name, key)
        self._threshold_combo.setCurrentIndex(0)  # Default to soft
        self._threshold_combo.setStyleSheet(
            f"""
            QComboBox {{
                padding: 5px 10px;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                background-color: white;
            }}
            QComboBox:focus {{
                border-color: {self.theme.get('primary', '#007AFF')};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}
            """
        )
        self._threshold_combo.currentIndexChanged.connect(self._on_threshold_changed)
        row2_layout.addWidget(self._threshold_combo)

        threshold_hint = QLabel("(soft=smoother denoising, hard=sharper cutoff)")
        threshold_hint.setFont(QFont("Arial", 9))
        threshold_hint.setStyleSheet(
            f"color: {self.theme.get('text_light', '#6c757d')}; font-style: italic;"
        )
        row2_layout.addWidget(threshold_hint)

        row2_layout.addStretch()
        wavelet_main_layout.addLayout(row2_layout)

        # Third row: Threshold Method (Adaptive Thresholding)
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(15)

        method_label = QLabel("🎯 Method:")
        method_label.setFont(QFont("Arial", 10))
        method_label.setStyleSheet(f"color: {self.theme.get('text', '#212529')};")
        row3_layout.addWidget(method_label)

        self._threshold_method_combo = QComboBox()
        self._threshold_method_combo.setFont(QFont("Arial", 10))
        self._threshold_method_combo.setMinimumWidth(350)
        for key, display_name in self.THRESHOLD_METHODS.items():
            self._threshold_method_combo.addItem(display_name, key)
        self._threshold_method_combo.setCurrentIndex(0)  # Default to visushrink
        self._threshold_method_combo.setStyleSheet(
            f"""
            QComboBox {{
                padding: 5px 10px;
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 4px;
                background-color: white;
            }}
            QComboBox:focus {{
                border-color: {self.theme.get('primary', '#007AFF')};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}
            """
        )
        self._threshold_method_combo.currentIndexChanged.connect(
            self._on_threshold_method_changed
        )
        row3_layout.addWidget(self._threshold_method_combo)

        row3_layout.addStretch()
        wavelet_main_layout.addLayout(row3_layout)

        main_layout.addWidget(self._wavelet_settings)

        # Initially hide wavelet settings
        self._wavelet_settings.setVisible(False)

        # Update button styles for initial selection
        self._update_button_styles()

    def _update_button_styles(self):
        """Update button styles based on current selection."""
        primary_color = self.theme.get("primary", "#007AFF")

        for method_key, btn in self._buttons.items():
            if method_key == self._selected_method:
                # Selected style
                btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {primary_color};
                        color: white;
                        border: 2px solid {primary_color};
                        border-radius: 8px;
                        padding: 8px 16px;
                    }}
                    QPushButton:hover {{
                        background-color: {primary_color};
                    }}
                    """
                )
            else:
                # Unselected style
                btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: #e9ecef;
                        color: #6c757d;
                        border: 2px solid #dee2e6;
                        border-radius: 8px;
                        padding: 8px 16px;
                    }}
                    QPushButton:hover {{
                        background-color: #d4e6f1;
                        border-color: {primary_color};
                        color: {self.theme.get('text', '#212529')};
                    }}
                    """
                )

        # Show/hide wavelet settings
        self._wavelet_settings.setVisible(self._selected_method == "WAVELETS")

    def _on_method_clicked(self, method: str):
        """Handle method button click."""
        if method != self._selected_method:
            self._selected_method = method
            self._update_button_styles()
            self.method_changed.emit(method)

    def _on_level_changed(self, value: int):
        """Handle wavelet level change."""
        self._wavelet_level = value
        self.wavelet_level_changed.emit(value)

    def _on_family_changed(self, index: int):
        """Handle wavelet family change."""
        self._wavelet_family = self._family_combo.currentData()
        self.wavelet_family_changed.emit(self._wavelet_family)

    def _on_threshold_changed(self, index: int):
        """Handle threshold mode change."""
        self._threshold_mode = self._threshold_combo.currentData()
        self.wavelet_threshold_mode_changed.emit(self._threshold_mode)

    def _on_threshold_method_changed(self, index: int):
        """Handle threshold method change."""
        self._threshold_method = self._threshold_method_combo.currentData()
        self.wavelet_threshold_method_changed.emit(self._threshold_method)

    def get_selected_method(self) -> str:
        """Get the currently selected method."""
        return self._selected_method

    def get_wavelet_level(self) -> int:
        """Get the current wavelet decomposition level."""
        return self._wavelet_level

    def get_wavelet_family(self) -> str:
        """Get the current wavelet family."""
        return self._wavelet_family

    def get_threshold_mode(self) -> str:
        """Get the current threshold mode."""
        return self._threshold_mode

    def get_threshold_method(self) -> str:
        """Get the current threshold method."""
        return self._threshold_method

    def set_selected_method(self, method: str):
        """Set the selected method programmatically."""
        if method in self.METHODS and method != self._selected_method:
            self._selected_method = method
            self._update_button_styles()
            self.method_changed.emit(method)

    def get_method_description(self, method: str = None) -> str:
        """Get the description for a method."""
        if method is None:
            method = self._selected_method
        return self.METHODS.get(method, {}).get("description", "")


class ChannelCheckBox(QCheckBox):
    """Custom checkbox for channel selection with additional info"""

    def __init__(self, channel_name: str, channel_info: Dict[str, Any]):
        super().__init__(channel_name)
        self.channel_name = channel_name
        self.channel_info = channel_info
        self.setFont(QFont("Arial", 11))

        # Style the checkbox
        self.setStyleSheet(
            """
            QCheckBox {
                padding: 8px;
                border-radius: 4px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #212529;
            }
            QCheckBox:hover {
                background-color: #e9ecef;
                border-color: #007AFF;
            }
            QCheckBox:checked {
                background-color: #d4e6f1;
                border-color: #007AFF;
                font-weight: bold;
                color: #212529;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #6c757d;
                background-color: white;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #007AFF;
                background-color: #007AFF;
                border-radius: 3px;
            }
        """
        )


class FileInfoWidget(QFrame):
    """Widget to display file information"""

    def __init__(self, theme: Dict[str, str]):
        super().__init__()
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.theme['background']};
                border: 2px solid {self.theme['border']};
                border-radius: 8px;
                padding: 10px;
            }}
        """
        )

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📄 File Information")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {self.theme['primary']};")
        layout.addWidget(title)

        # Info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: white;
                border: 1px solid {self.theme['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 10px;
                color: {self.theme['text']};
            }}
        """
        )
        layout.addWidget(self.info_text)

    def update_info(
        self,
        file_path: str,
        channel_count: int,
        sampling_rate: float,
        duration: float,
        all_channels: List[str],
    ):
        """Update file information display"""
        info_text = f"""
📁 File: {file_path.split('/')[-1]}
📊 Total Channels: {len(all_channels)}
⚡ Sampling Rate: {sampling_rate} Hz
⏱️ Duration: {duration:.1f} seconds
🧠 Selected EEG: {channel_count}

All available channels:
{', '.join(all_channels)}
        """.strip()
        self.info_text.setPlainText(info_text)


class ChannelSelectorWidget(QWidget):
    """Main channel selection widget with analysis method selection"""

    # Emits tuple of (selected channel names, analysis method, wavelet_params dict)
    channels_selected = pyqtSignal(list, str, dict)

    def __init__(self, theme: Dict[str, str]):
        super().__init__()
        self.theme = theme
        self.all_channels = []
        self.eeg_channels = []
        self.channel_checkboxes = {}
        self.current_file = ""
        self.raw_data = None
        self.analysis_method = "ICA"  # Default to ICA
        self.wavelet_level = 5  # Default wavelet level
        self.wavelet_family = "db4"  # Default wavelet family
        self.threshold_mode = "soft"  # Default threshold mode
        self.threshold_method = "visushrink"  # Default threshold method

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # Title
        title = QLabel("🧠 EEG Channel Selection")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {self.theme['primary']}; margin: 10px;")
        main_layout.addWidget(title)

        # Description
        self.description = QLabel(
            "Select the EEG channels you want to include in the analysis.\n"
            "At least 3 channels are recommended for optimal results."
        )
        self.description.setFont(QFont("Arial", 12))
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setStyleSheet(
            f"color: {self.theme['text_light']}; margin-bottom: 15px;"
        )
        main_layout.addWidget(self.description)

        # Analysis Method Selector (ICA/PCA/Wavelet)
        method_group = QGroupBox("🔬 Analysis Method")
        method_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        method_group.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.theme['primary']};
                border-radius: 8px;
                margin: 5px 0px;
                padding: 15px;
                background-color: white;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {self.theme['primary']};
                background-color: white;
            }}
        """
        )
        method_layout = QVBoxLayout(method_group)
        method_layout.setSpacing(10)

        # Method selector buttons (ICA, PCA, Wavelet)
        self.method_selector = AnalysisMethodSelector(self.theme)
        self.method_selector.method_changed.connect(self._on_method_changed)
        self.method_selector.wavelet_level_changed.connect(
            self._on_wavelet_level_changed
        )
        self.method_selector.wavelet_family_changed.connect(
            self._on_wavelet_family_changed
        )
        self.method_selector.wavelet_threshold_mode_changed.connect(
            self._on_threshold_mode_changed
        )
        self.method_selector.wavelet_threshold_method_changed.connect(
            self._on_threshold_method_changed
        )
        method_layout.addWidget(self.method_selector)

        # Method description label
        self.method_info_label = QLabel(self.method_selector.get_method_description())
        self.method_info_label.setFont(QFont("Arial", 10))
        self.method_info_label.setStyleSheet(
            f"color: {self.theme['text_light']}; font-style: italic;"
        )
        method_layout.addWidget(self.method_info_label)

        main_layout.addWidget(method_group)

        # Create splitter for layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: File info
        self.file_info_widget = FileInfoWidget(self.theme)
        self.file_info_widget.setMaximumWidth(350)
        splitter.addWidget(self.file_info_widget)

        # Right side: Channel selection
        channel_widget = QWidget()
        channel_layout = QVBoxLayout(channel_widget)

        # Search and filter controls
        filter_layout = QHBoxLayout()

        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search channels...")
        self.search_box.setFont(QFont("Arial", 11))
        self.search_box.textChanged.connect(self.filter_channels)
        self.search_box.setStyleSheet(
            f"""
            QLineEdit {{
                padding: 8px 12px;
                border: 2px solid {self.theme['border']};
                border-radius: 6px;
                font-size: 11px;
                background-color: white;
                color: {self.theme['text']};
            }}
            QLineEdit:focus {{
                border-color: {self.theme['primary']};
                background-color: #f8f9fa;
            }}
        """
        )
        filter_layout.addWidget(self.search_box)

        # Quick selection buttons
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_eeg)
        select_all_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['success']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme['success_hover']};
            }}
        """
        )
        filter_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        clear_all_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['danger']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
        """
        )
        filter_layout.addWidget(clear_all_btn)

        channel_layout.addLayout(filter_layout)

        # Channels scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                border: 2px solid {self.theme['border']};
                border-radius: 8px;
                background-color: white;
            }}
            QScrollArea QWidget {{
                background-color: white;
            }}
        """
        )

        self.channels_widget = QWidget()
        self.channels_widget.setStyleSheet("background-color: white;")
        self.channels_layout = QGridLayout(self.channels_widget)
        self.channels_layout.setSpacing(5)

        scroll_area.setWidget(self.channels_widget)
        channel_layout.addWidget(scroll_area)

        splitter.addWidget(channel_widget)
        splitter.setSizes([350, 650])  # Set relative sizes

        # Bottom controls
        bottom_layout = QHBoxLayout()

        # Selection counter
        self.selection_label = QLabel("📊 Selected: 0 channels")
        self.selection_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.selection_label.setStyleSheet(f"color: {self.theme['text']};")
        bottom_layout.addWidget(self.selection_label)

        bottom_layout.addStretch()

        # Action buttons
        self.continue_btn = QPushButton("✅ Continue with Selected Channels")
        self.continue_btn.setMinimumHeight(45)
        self.continue_btn.setMinimumWidth(250)
        self.continue_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.continue_btn.clicked.connect(self.confirm_selection)
        self.continue_btn.setEnabled(False)
        self.continue_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['primary']};
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme['primary_hover']};
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
            }}
        """
        )
        bottom_layout.addWidget(self.continue_btn)

        main_layout.addLayout(bottom_layout)

    def set_edf_file(self, file_path: str):
        """
        Load and analyze EEG file for channel selection.

        Supports multiple formats: EDF, BDF, FIF, CSV, SET (EEGLAB)
        """
        try:
            self.current_file = file_path

            # Load file using the generic reader for multi-format support
            from backend.eeg_backend import EEGDataManager

            raw = EEGDataManager.read_raw(file_path, preload=False)
            self.raw_data = raw

            # Get all channels
            self.all_channels = list(raw.ch_names)

            # Detect potential EEG channels using existing logic
            potential_eeg = EEGDataManager.detect_eeg_channels(raw)

            # Categorize channels
            self.eeg_channels = []
            self.other_channels = []

            for ch in self.all_channels:
                if ch in potential_eeg:
                    self.eeg_channels.append(ch)
                else:
                    self.other_channels.append(ch)

            # Update file info
            self.file_info_widget.update_info(
                file_path,
                len(self.eeg_channels),
                raw.info["sfreq"],
                len(raw.annotations),
                self.all_channels,
            )

            # Create channel checkboxes
            self.create_channel_checkboxes()

            # Pre-select detected EEG channels
            self.select_detected_eeg()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to load file:\n{str(e)}")

    def create_channel_checkboxes(self):
        """Create checkboxes for all channels"""
        # Clear existing checkboxes
        for checkbox in self.channel_checkboxes.values():
            checkbox.setParent(None)
        self.channel_checkboxes.clear()

        row = 0
        col = 0
        max_cols = 3

        # EEG channels first (recommended)
        if self.eeg_channels:
            eeg_group = QGroupBox("🧠 Recommended EEG Channels")
            eeg_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            eeg_group.setStyleSheet(
                f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid {self.theme['success']};
                    border-radius: 8px;
                    margin: 10px 0px;
                    padding-top: 15px;
                    background-color: white;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {self.theme['success']};
                    background-color: white;
                }}
            """
            )

            eeg_layout = QGridLayout(eeg_group)
            eeg_row, eeg_col = 0, 0

            for channel in self.eeg_channels:
                checkbox = ChannelCheckBox(channel, {"type": "eeg"})
                checkbox.stateChanged.connect(self.update_selection_count)
                self.channel_checkboxes[channel] = checkbox

                eeg_layout.addWidget(checkbox, eeg_row, eeg_col)
                eeg_col += 1
                if eeg_col >= max_cols:
                    eeg_col = 0
                    eeg_row += 1

            self.channels_layout.addWidget(eeg_group, row, 0, 1, max_cols)
            row += 1

        # Other channels
        if self.other_channels:
            other_group = QGroupBox("📊 Other Available Channels")
            other_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            other_group.setStyleSheet(
                f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid {self.theme['border']};
                    border-radius: 8px;
                    margin: 10px 0px;
                    padding-top: 15px;
                    background-color: white;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {self.theme['text_light']};
                    background-color: white;
                }}
            """
            )

            other_layout = QGridLayout(other_group)
            other_row, other_col = 0, 0

            for channel in self.other_channels:
                checkbox = ChannelCheckBox(channel, {"type": "other"})
                checkbox.stateChanged.connect(self.update_selection_count)
                self.channel_checkboxes[channel] = checkbox

                other_layout.addWidget(checkbox, other_row, other_col)
                other_col += 1
                if other_col >= max_cols:
                    other_col = 0
                    other_row += 1

            self.channels_layout.addWidget(other_group, row, 0, 1, max_cols)

    def select_detected_eeg(self):
        """Pre-select the automatically detected EEG channels"""
        for channel in self.eeg_channels:
            if channel in self.channel_checkboxes:
                self.channel_checkboxes[channel].setChecked(True)
        self.update_selection_count()

    def filter_channels(self, text: str):
        """Filter channels based on search text"""
        text = text.lower()
        for channel, checkbox in self.channel_checkboxes.items():
            if text in channel.lower():
                checkbox.show()
            else:
                checkbox.hide()

    def select_all_eeg(self):
        """Select all detected EEG channels"""
        for channel in self.eeg_channels:
            if channel in self.channel_checkboxes:
                self.channel_checkboxes[channel].setChecked(True)
        self.update_selection_count()

    def clear_all(self):
        """Clear all selections"""
        for checkbox in self.channel_checkboxes.values():
            checkbox.setChecked(False)
        self.update_selection_count()

    def update_selection_count(self):
        """Update selection counter and enable/disable continue button"""
        selected_count = len(self.get_selected_channels())
        self.selection_label.setText(f"📊 Selected: {selected_count} channels")

        # Enable continue button only if at least 3 channels are selected
        self.continue_btn.setEnabled(selected_count >= 3)

        if selected_count < 3:
            self.selection_label.setStyleSheet(f"color: {self.theme['danger']};")
        elif selected_count >= 3:
            self.selection_label.setStyleSheet(f"color: {self.theme['success']};")

    def get_selected_channels(self) -> List[str]:
        """Get list of selected channel names"""
        selected = []
        for channel, checkbox in self.channel_checkboxes.items():
            if checkbox.isChecked():
                selected.append(channel)
        return selected

    def confirm_selection(self):
        """Confirm channel selection and proceed"""
        selected_channels = self.get_selected_channels()

        if len(selected_channels) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Channels",
                f"Please select at least 3 channels for reliable {self.analysis_method} analysis.",
            )
            return

        # Show confirmation
        eeg_count = len([ch for ch in selected_channels if ch in self.eeg_channels])
        other_count = len(selected_channels) - eeg_count

        msg = f"""
        Channel Selection Confirmation:

        🧠 EEG Channels: {eeg_count}
        📊 Other Channels: {other_count}
        📈 Total: {len(selected_channels)}
        🔬 Analysis Method: {self.analysis_method}

        Selected channels:
        {', '.join(selected_channels)}

        Do you want to continue with these channels?
        """

        reply = QMessageBox.question(
            self,
            "Confirm Selection",
            msg.strip(),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            wavelet_params = {
                "level": self.wavelet_level,
                "wavelet": self.wavelet_family,
                "threshold_mode": self.threshold_mode,
                "threshold_method": self.threshold_method,
            }
            self.channels_selected.emit(
                selected_channels, self.analysis_method, wavelet_params
            )

    def _on_method_changed(self, method: str):
        """Handle analysis method change"""
        self.analysis_method = method
        self.method_info_label.setText(
            self.method_selector.get_method_description(method)
        )

    def _on_wavelet_level_changed(self, level: int):
        """Handle wavelet level change"""
        self.wavelet_level = level

    def _on_wavelet_family_changed(self, family: str):
        """Handle wavelet family change"""
        self.wavelet_family = family

    def _on_threshold_mode_changed(self, mode: str):
        """Handle threshold mode change"""
        self.threshold_mode = mode

    def _on_threshold_method_changed(self, method: str):
        """Handle threshold method change"""
        self.threshold_method = method

    def get_analysis_method(self) -> str:
        """Get the selected analysis method"""
        return self.analysis_method

    def get_wavelet_level(self) -> int:
        """Get the selected wavelet decomposition level"""
        return self.wavelet_level

    def get_wavelet_family(self) -> str:
        """Get the selected wavelet family"""
        return self.wavelet_family

    def get_threshold_mode(self) -> str:
        """Get the selected threshold mode"""
        return self.threshold_mode
