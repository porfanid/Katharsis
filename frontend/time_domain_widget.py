#!/usr/bin/env python3
"""
Time Domain Widget - Pure UI component for time domain analysis
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QDoubleSpinBox, QPushButton, QGroupBox,
                           QTextEdit, QProgressBar, QComboBox)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class TimeDomainWidget(QWidget):
    """Widget for time domain analysis configuration and execution"""
    
    time_domain_configured = pyqtSignal(dict)  # Signal emitted when time domain is configured
    
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
        title = QLabel("Time Domain Analysis")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Configuration group
        config_group = QGroupBox("Παράμετροι Epoching")
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
        
        # Time window
        tmin_layout = QHBoxLayout()
        tmin_layout.addWidget(QLabel("Tmin (s):"))
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(-5.0, 0.0)
        self.tmin_spin.setValue(-0.2)
        self.tmin_spin.setSingleStep(0.1)
        tmin_layout.addWidget(self.tmin_spin)
        config_layout.addLayout(tmin_layout)
        
        tmax_layout = QHBoxLayout()
        tmax_layout.addWidget(QLabel("Tmax (s):"))
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0.0, 5.0)
        self.tmax_spin.setValue(0.8)
        self.tmax_spin.setSingleStep(0.1)
        tmax_layout.addWidget(self.tmax_spin)
        config_layout.addLayout(tmax_layout)
        
        # Baseline correction
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel("Baseline correction:"))
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItems([
            "None",
            "Mean",
            "Median",
            "Loess",
            "Rescale"
        ])
        baseline_layout.addWidget(self.baseline_combo)
        config_layout.addLayout(baseline_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Εκτέλεση Time Domain Analysis")
        self.run_button.setFont(QFont("Arial", 14))
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.run_button.clicked.connect(self.run_time_domain)
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
        self.results_text.setPlaceholderText("Τα αποτελέσματα της ανάλυσης χρονικού τομέα θα εμφανιστούν εδώ...")
        layout.addWidget(self.results_text)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def run_time_domain(self):
        """Run time domain analysis with current configuration"""
        events_config = {
            'event_id': None,
            'min_duration': 0.001
        }
        
        epoch_config = {
            'tmin': self.tmin_spin.value(),
            'tmax': self.tmax_spin.value(),
            'baseline': None,
            'baseline_correction': self.baseline_combo.currentText().lower(),
            'reject_criteria': {}
        }
        
        config = {
            'events_config': events_config,
            'epoch_config': epoch_config
        }
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        
        self.time_domain_configured.emit(config)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        """Display time domain results"""
        if results.get('success', False):
            text = f"""
✅ Time Domain Analysis επιτυχής!

Αποτελέσματα:
- Epochs: {results.get('n_epochs', 'N/A')}
- Events: {results.get('n_events', 'N/A')}
- Time window: {results.get('epoch_info', {}).get('tmin', 'N/A')} - {results.get('epoch_info', {}).get('tmax', 'N/A')} s
- Sampling rate: {results.get('epoch_info', {}).get('sampling_rate', 'N/A')} Hz
"""
        else:
            text = f"❌ Σφάλμα Time Domain: {results.get('error', 'Άγνωστο σφάλμα')}"
            if results.get('suggestion'):
                text += f"\n{results.get('suggestion')}"
        
        self.results_text.setText(text)
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)