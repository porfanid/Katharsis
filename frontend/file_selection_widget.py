#!/usr/bin/env python3
"""
File Selection Widget - Pure UI component for file selection
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class FileSelectionWidget(QWidget):
    """Widget for selecting EEG files"""
    
    file_selected = pyqtSignal(str)  # Signal emitted when file is selected
    
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
        title = QLabel("Επιλογή Αρχείου EEG")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Επιλέξτε ένα αρχείο EEG για ανάλυση.\n"
            "Υποστηριζόμενοι τύποι: .edf, .fif, .csv, .set"
        )
        description.setFont(QFont("Arial", 12))
        description.setStyleSheet("color: #34495e; margin-bottom: 30px;")
        layout.addWidget(description)
        
        # File selection button
        self.select_button = QPushButton("Επιλογή Αρχείου")
        self.select_button.setFont(QFont("Arial", 14))
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        # Selected file label
        self.file_label = QLabel("Δεν έχει επιλεγεί αρχείο")
        self.file_label.setFont(QFont("Arial", 10))
        self.file_label.setStyleSheet("color: #7f8c8d; margin-top: 20px;")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_file(self):
        """Open file dialog and select EEG file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Επιλογή Αρχείου EEG",
            "",
            "EEG Files (*.edf *.fif *.fiff *.csv *.set);;All Files (*)"
        )
        
        if file_path:
            self.file_label.setText(f"Επιλεγμένο: {file_path}")
            self.file_selected.emit(file_path)