#!/usr/bin/env python3
"""
ICA Analysis Widget - Pure UI component for ICA analysis configuration
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QSpinBox, QPushButton, QGroupBox,
                           QTextEdit, QProgressBar)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class ICAAnalysisWidget(QWidget):
    """Widget for ICA analysis configuration and execution"""
    
    ica_configured = pyqtSignal(dict)  # Signal emitted when ICA is configured
    
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
        title = QLabel("ICA Analysis")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Configuration group
        config_group = QGroupBox("Παράμετροι ICA")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        config_layout = QVBoxLayout()
        
        # ICA Method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Μέθοδος ICA:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "fastica",
            "extended_infomax", 
            "picard",
            "mne_default"
        ])
        method_layout.addWidget(self.method_combo)
        config_layout.addLayout(method_layout)
        
        # Number of components
        components_layout = QHBoxLayout()
        components_layout.addWidget(QLabel("Αριθμός συνιστωσών:"))
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 50)
        self.components_spin.setValue(0)  # 0 means auto
        self.components_spin.setSpecialValueText("Auto")
        components_layout.addWidget(self.components_spin)
        config_layout.addLayout(components_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Run ICA button
        self.run_button = QPushButton("Εκτέλεση ICA Analysis")
        self.run_button.setFont(QFont("Arial", 14))
        self.run_button.setStyleSheet("""
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
        self.run_button.clicked.connect(self.run_ica)
        layout.addWidget(self.run_button)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
                background-color: #f8f9fa;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.results_text.setPlaceholderText("Τα αποτελέσματα της ICA ανάλυσης θα εμφανιστούν εδώ...")
        layout.addWidget(self.results_text)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def run_ica(self):
        """Run ICA analysis with current configuration"""
        config = {
            'method': self.method_combo.currentText(),
            'n_components': self.components_spin.value() if self.components_spin.value() > 0 else None,
            'random_state': 42
        }
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        
        self.ica_configured.emit(config)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        """Display ICA results"""
        if results.get('success', False):
            text = f"""
✅ ICA Analysis επιτυχής!

Παράμετροι:
- Μέθοδος: {results.get('method', 'N/A')}
- Συνιστώσες: {results.get('n_components', 'N/A')}

Αποτελέσματα:
- Εντοπισμένα artifacts: {len(results.get('suggested_artifacts', []))}
- Τύποι artifacts: {list(results.get('artifact_types', {}).keys())}
"""
        else:
            text = f"❌ Σφάλμα ICA: {results.get('error', 'Άγνωστο σφάλμα')}"
        
        self.results_text.setText(text)
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)