#!/usr/bin/env python3
"""
Results Widget - Pure UI component for displaying analysis results
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class ResultsWidget(QWidget):
    """Widget for displaying analysis results"""
    
    export_requested = pyqtSignal()  # Signal emitted when export is requested
    
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
        title = QLabel("Αποτελέσματα Ανάλυσης")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 30px;")
        layout.addWidget(title)
        
        # Results display area
        self.results_display = QTextEdit()
        self.results_display.setStyleSheet("""
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 15px;
                background-color: #f8f9fa;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)
        
        # Export button
        self.export_button = QPushButton("Εξαγωγή Αποτελεσμάτων")
        self.export_button.setFont(QFont("Arial", 14))
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.export_button.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_button)
        
        self.setLayout(layout)
    
    def show_results(self, results):
        """Display analysis results"""
        self.results_display.setText(str(results))