#!/usr/bin/env python3
"""
Channel Selector Component - Interactive channel selection interface
Στοιχείο Επιλογής Καναλιών - Διαδραστική διεπαφή επιλογής καναλιών
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QScrollArea,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QTextEdit,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QPalette
from typing import List, Dict, Any, Optional
import mne


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
        title = QLabel("📄 Πληροφορίες Αρχείου")
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
📁 Αρχείο: {file_path.split('/')[-1]}
📊 Συνολικά Κανάλια: {len(all_channels)}
⚡ Συχνότητα Δειγματοληψίας: {sampling_rate} Hz
⏱️ Διάρκεια: {duration:.1f} δευτερόλεπτα
🧠 Επιλεγμένα EEG: {channel_count}

Όλα τα διαθέσιμα κανάλια:
{', '.join(all_channels)}
        """.strip()
        self.info_text.setPlainText(info_text)


class ChannelSelectorWidget(QWidget):
    """Main channel selection widget"""

    channels_selected = pyqtSignal(list)  # Emits list of selected channel names

    def __init__(self, theme: Dict[str, str]):
        super().__init__()
        self.theme = theme
        self.all_channels = []
        self.eeg_channels = []
        self.channel_checkboxes = {}
        self.current_file = ""
        self.raw_data = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # Title
        title = QLabel("🧠 Επιλογή Καναλιών EEG")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {self.theme['primary']}; margin: 10px;")
        main_layout.addWidget(title)

        # Description
        description = QLabel(
            "Επιλέξτε τα κανάλια EEG που θέλετε να συμπεριλάβετε στην ανάλυση.\n"
            "Συνιστώνται τουλάχιστον 3 κανάλια για βέλτιστα αποτελέσματα ICA."
        )
        description.setFont(QFont("Arial", 12))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet(
            f"color: {self.theme['text_light']}; margin-bottom: 15px;"
        )
        main_layout.addWidget(description)

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
        self.search_box.setPlaceholderText("🔍 Αναζήτηση καναλιών...")
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
        select_all_btn = QPushButton("Επιλογή Όλων")
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

        clear_all_btn = QPushButton("Καθαρισμός")
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
        self.selection_label = QLabel("📊 Επιλεγμένα: 0 κανάλια")
        self.selection_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.selection_label.setStyleSheet(f"color: {self.theme['text']};")
        bottom_layout.addWidget(self.selection_label)

        bottom_layout.addStretch()

        # Action buttons
        self.continue_btn = QPushButton("✅ Συνέχεια με Επιλεγμένα Κανάλια")
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
        """Load and analyze EDF file for channel selection"""
        try:
            self.current_file = file_path

            # Load file to get channel information
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            self.raw_data = raw

            # Get all channels
            self.all_channels = raw.ch_names

            # Detect potential EEG channels using existing logic
            from backend.eeg_backend import EEGDataManager

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
                0,
                self.all_channels,
            )

            # Create channel checkboxes
            self.create_channel_checkboxes()

            # Pre-select detected EEG channels
            self.select_detected_eeg()

        except Exception as e:
            QMessageBox.critical(
                self, "Σφάλμα", f"Αδυναμία φόρτωσης αρχείου:\n{str(e)}"
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
            eeg_group = QGroupBox("🧠 Προτεινόμενα EEG Κανάλια")
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
            other_group = QGroupBox("📊 Άλλα Διαθέσιμα Κανάλια")
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
        self.selection_label.setText(f"📊 Επιλεγμένα: {selected_count} κανάλια")

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
                "Ανεπαρκή Κανάλια",
                "Παρακαλώ επιλέξτε τουλάχιστον 3 κανάλια για αξιόπιστη ανάλυση ICA.",
            )
            return

        # Show confirmation
        eeg_count = len([ch for ch in selected_channels if ch in self.eeg_channels])
        other_count = len(selected_channels) - eeg_count

        msg = f"""
        Επιβεβαίωση Επιλογής Καναλιών:
        
        🧠 EEG Κανάλια: {eeg_count}
        📊 Άλλα Κανάλια: {other_count}
        📈 Συνολικά: {len(selected_channels)}
        
        Επιλεγμένα κανάλια:
        {', '.join(selected_channels)}
        
        Θέλετε να συνεχίσετε με αυτά τα κανάλια;
        """

        reply = QMessageBox.question(
            self,
            "Επιβεβαίωση Επιλογής",
            msg.strip(),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.channels_selected.emit(selected_channels)
