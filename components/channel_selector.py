#!/usr/bin/env python3
"""
Channel Selector Component - Interactive channel selection interface
"""

from typing import Any, Dict, List, Optional

import mne
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ToggleSwitch(QWidget):
    """
    Custom toggle switch widget for ICA/PCA selection
    A modern-looking animated toggle switch
    """

    toggled = pyqtSignal(bool)  # True = PCA, False = ICA

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self._is_checked = False  # False = ICA (left), True = PCA (right)
        self._handle_position = 3  # Starting position

        # Size settings
        self.setFixedSize(180, 44)

        # Animation for smooth toggle
        self._animation = QPropertyAnimation(self, b"handle_position")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @property
    def handle_position(self):
        return self._handle_position

    @handle_position.setter
    def handle_position(self, pos):
        self._handle_position = pos
        self.update()

    def isChecked(self):
        return self._is_checked

    def setChecked(self, checked: bool):
        if self._is_checked != checked:
            self._is_checked = checked
            self._animate_toggle()
            self.toggled.emit(checked)

    def _animate_toggle(self):
        self._animation.stop()
        if self._is_checked:
            # Move to PCA (right)
            self._animation.setStartValue(self._handle_position)
            self._animation.setEndValue(self.width() // 2 + 3)
        else:
            # Move to ICA (left)
            self._animation.setStartValue(self._handle_position)
            self._animation.setEndValue(3)
        self._animation.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._is_checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colors
        primary_color = QColor(self.theme.get("primary", "#007AFF"))
        secondary_color = QColor("#6c757d")
        bg_color = QColor("#e9ecef")
        handle_color = QColor("white")

        # Draw background track
        track_rect = QRect(0, 0, self.width(), self.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(track_rect, 22, 22)

        # Draw left side (ICA) background
        left_rect = QRect(0, 0, self.width() // 2, self.height())
        if not self._is_checked:
            painter.setBrush(QBrush(primary_color))
        else:
            painter.setBrush(QBrush(secondary_color.lighter(130)))
        painter.drawRoundedRect(left_rect, 22, 22)
        # Fix the right edge of left side
        painter.drawRect(QRect(self.width() // 2 - 22, 0, 22, self.height()))

        # Draw right side (PCA) background
        right_rect = QRect(self.width() // 2, 0, self.width() // 2, self.height())
        if self._is_checked:
            painter.setBrush(QBrush(primary_color))
        else:
            painter.setBrush(QBrush(secondary_color.lighter(130)))
        painter.drawRoundedRect(right_rect, 22, 22)
        # Fix the left edge of right side
        painter.drawRect(QRect(self.width() // 2, 0, 22, self.height()))

        # Draw labels
        painter.setPen(
            QPen(QColor("white") if not self._is_checked else QColor("#6c757d"))
        )
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        painter.drawText(
            QRect(5, 0, self.width() // 2 - 5, self.height()),
            Qt.AlignmentFlag.AlignCenter,
            "ICA",
        )

        painter.setPen(QPen(QColor("white") if self._is_checked else QColor("#6c757d")))
        painter.drawText(
            QRect(self.width() // 2, 0, self.width() // 2 - 5, self.height()),
            Qt.AlignmentFlag.AlignCenter,
            "PCA",
        )

        # Draw handle (sliding circle)
        handle_width = self.width() // 2 - 6
        handle_height = self.height() - 6
        handle_rect = QRect(int(self._handle_position), 3, handle_width, handle_height)

        # Handle shadow
        shadow_rect = QRect(
            int(self._handle_position) + 2, 5, handle_width, handle_height
        )
        painter.setBrush(QBrush(QColor(0, 0, 0, 30)))
        painter.drawRoundedRect(shadow_rect, 19, 19)

        # Handle
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(QColor("#dee2e6"), 1))
        painter.drawRoundedRect(handle_rect, 19, 19)

        # Handle label
        handle_text = "🧠 ICA" if not self._is_checked else "📊 PCA"
        painter.setPen(QPen(primary_color))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(handle_rect, Qt.AlignmentFlag.AlignCenter, handle_text)


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

    # Emits tuple of (selected channel names, analysis method)
    channels_selected = pyqtSignal(list, str)

    def __init__(self, theme: Dict[str, str]):
        super().__init__()
        self.theme = theme
        self.all_channels = []
        self.eeg_channels = []
        self.channel_checkboxes = {}
        self.current_file = ""
        self.raw_data = None
        self.analysis_method = "ICA"  # Default to ICA

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

        # Analysis Method Selector (ICA/PCA toggle)
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
        method_layout = QHBoxLayout(method_group)
        method_layout.setSpacing(20)

        # ICA label
        ica_label = QLabel("🧠 ICA")
        ica_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        ica_label.setStyleSheet(f"color: {self.theme['text']};")
        method_layout.addWidget(ica_label)

        # Custom toggle switch for ICA/PCA selection
        self.method_toggle = ToggleSwitch(self.theme)
        self.method_toggle.toggled.connect(self._on_toggle_changed)
        method_layout.addWidget(self.method_toggle)

        # PCA label
        pca_label = QLabel("📊 PCA")
        pca_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        pca_label.setStyleSheet(f"color: {self.theme['text']};")
        method_layout.addWidget(pca_label)

        method_layout.addStretch()

        # Method description label
        self.method_info_label = QLabel(
            "ICA: Best for detecting eye blinks and muscle artifacts"
        )
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
            QMessageBox.critical(
                self, "Error", f"Unable to load file:\n{str(e)}"
            )

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
            self.channels_selected.emit(selected_channels, self.analysis_method)

    def _on_toggle_changed(self, is_pca: bool):
        """Handle analysis method toggle change"""
        if is_pca:
            self.analysis_method = "PCA"
            self.method_info_label.setText(
                "PCA: Faster, ideal for quick preliminary analysis"
            )
        else:
            self.analysis_method = "ICA"
            self.method_info_label.setText(
                "ICA: Best for detecting eye blinks and muscle artifacts"
            )

    def get_analysis_method(self) -> str:
        """Get the selected analysis method"""
        return self.analysis_method
