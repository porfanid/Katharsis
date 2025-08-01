#!/usr/bin/env python3
"""
Channel Selection Widget - Pure UI component for channel selection
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class ChannelSelectionWidget(QWidget):
    """Widget for selecting EEG channels"""
    
    channels_selected = pyqtSignal(list)  # Signal emitted when channels are selected
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Επιλογή Καναλιών")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Επιλέξτε τα κανάλια EEG που θέλετε να αναλύσετε.\n"
            "Μπορείτε να επιλέξετε πολλαπλά κανάλια κρατώντας Ctrl."
        )
        description.setFont(QFont("Arial", 12))
        description.setStyleSheet("color: #34495e; margin-bottom: 30px;")
        layout.addWidget(description)
        
        # Channel list
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.channel_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #ecf0f1;
            }
        """)
        layout.addWidget(self.channel_list)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.select_all_button = QPushButton("Επιλογή Όλων")
        self.select_all_button.clicked.connect(self.select_all_channels)
        button_layout.addWidget(self.select_all_button)
        
        self.clear_selection_button = QPushButton("Καθαρισμός Επιλογής")
        self.clear_selection_button.clicked.connect(self.clear_selection)
        button_layout.addWidget(self.clear_selection_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_channels(self, channels):
        """Load available channels into the list"""
        self.channel_list.clear()
        for channel in channels:
            item = QListWidgetItem(channel)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked)  # Default to selected
            self.channel_list.addItem(item)
    
    def select_all_channels(self):
        """Select all channels"""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setSelected(True)
    
    def clear_selection(self):
        """Clear all channel selections"""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setSelected(False)
    
    def get_selected_channels(self):
        """Get list of selected channels"""
        selected = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.isSelected():
                selected.append(item.text())
        return selected