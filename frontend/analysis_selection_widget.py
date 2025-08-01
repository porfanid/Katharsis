#!/usr/bin/env python3
"""
Analysis Selection Widget - Pure UI component for choosing analysis method
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class AnalysisSelectionWidget(QWidget):
    """Widget for selecting analysis method"""
    
    analysis_selected = pyqtSignal(str)  # Signal emitted when analysis method is selected
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Title
        title = QLabel("Επιλογή Μεθόδου Ανάλυσης")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Επιλέξτε τη μέθοδο ανάλυσης που θέλετε να εφαρμόσετε στα δεδομένα σας."
        )
        description.setFont(QFont("Arial", 12))
        description.setStyleSheet("color: #34495e; margin-bottom: 40px;")
        layout.addWidget(description)
        
        # Analysis options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(20)
        
        # ICA Analysis button
        self.ica_button = QPushButton("🧠 ICA Analysis")
        self.ica_button.setFont(QFont("Arial", 16))
        self.ica_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 20px 40px;
                border-radius: 12px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.ica_button.clicked.connect(lambda: self.select_analysis("ica"))
        options_layout.addWidget(self.ica_button)
        
        # ICA description
        ica_desc = QLabel("Ανάλυση Ανεξάρτητων Συνιστωσών για εντοπισμό και αφαίρεση artifacts")
        ica_desc.setFont(QFont("Arial", 10))
        ica_desc.setStyleSheet("color: #7f8c8d; margin-left: 20px; margin-bottom: 20px;")
        options_layout.addWidget(ica_desc)
        
        # Time Domain Analysis button
        self.time_domain_button = QPushButton("📊 Time Domain Analysis")
        self.time_domain_button.setFont(QFont("Arial", 16))
        self.time_domain_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 20px 40px;
                border-radius: 12px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.time_domain_button.clicked.connect(lambda: self.select_analysis("time_domain"))
        options_layout.addWidget(self.time_domain_button)
        
        # Time domain description
        td_desc = QLabel("Ανάλυση χρονικού τομέα, epoching, και ERPs")
        td_desc.setFont(QFont("Arial", 10))
        td_desc.setStyleSheet("color: #7f8c8d; margin-left: 20px;")
        options_layout.addWidget(td_desc)
        
        layout.addLayout(options_layout)
        layout.addStretch()
        self.setLayout(layout)
    
    def select_analysis(self, analysis_type):
        """Handle analysis selection"""
        self.analysis_selected.emit(analysis_type)