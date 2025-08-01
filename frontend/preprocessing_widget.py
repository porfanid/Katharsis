#!/usr/bin/env python3
"""
Preprocessing Widget - Pure UI component for preprocessing configuration
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton,
                           QGroupBox, QComboBox)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class PreprocessingWidget(QWidget):
    """Widget for configuring preprocessing options"""
    
    preprocessing_configured = pyqtSignal(dict)  # Signal emitted when preprocessing is configured
    
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
        title = QLabel("Προεπεξεργασία Δεδομένων")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Filtering group
        filter_group = QGroupBox("Φιλτράρισμα")
        filter_group.setStyleSheet("""
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
        filter_layout = QVBoxLayout()
        
        self.apply_filter_check = QCheckBox("Εφαρμογή φιλτραρίσματος")
        filter_layout.addWidget(self.apply_filter_check)
        
        # High-pass filter
        hp_layout = QHBoxLayout()
        hp_layout.addWidget(QLabel("High-pass (Hz):"))
        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(0.0, 50.0)
        self.highpass_spin.setValue(1.0)
        self.highpass_spin.setSingleStep(0.1)
        hp_layout.addWidget(self.highpass_spin)
        filter_layout.addLayout(hp_layout)
        
        # Low-pass filter
        lp_layout = QHBoxLayout()
        lp_layout.addWidget(QLabel("Low-pass (Hz):"))
        self.lowpass_spin = QDoubleSpinBox()
        self.lowpass_spin.setRange(1.0, 200.0)
        self.lowpass_spin.setValue(40.0)
        self.lowpass_spin.setSingleStep(1.0)
        lp_layout.addWidget(self.lowpass_spin)
        filter_layout.addLayout(lp_layout)
        
        # Notch filter
        self.apply_notch_check = QCheckBox("Notch filter (50/60 Hz)")
        filter_layout.addWidget(self.apply_notch_check)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Re-referencing group
        reref_group = QGroupBox("Re-referencing")
        reref_layout = QVBoxLayout()
        
        self.apply_reref_check = QCheckBox("Εφαρμογή re-referencing")
        reref_layout.addWidget(self.apply_reref_check)
        
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference:"))
        self.reference_combo = QComboBox()
        self.reference_combo.addItems([
            "Average reference",
            "Common reference", 
            "Bipolar referencing"
        ])
        ref_layout.addWidget(self.reference_combo)
        reref_layout.addLayout(ref_layout)
        
        reref_group.setLayout(reref_layout)
        layout.addWidget(reref_group)
        
        # Apply button
        self.apply_button = QPushButton("Εφαρμογή Προεπεξεργασίας")
        self.apply_button.setFont(QFont("Arial", 14))
        self.apply_button.setStyleSheet("""
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
        self.apply_button.clicked.connect(self.apply_preprocessing)
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def apply_preprocessing(self):
        """Apply preprocessing with current configuration"""
        config = {
            'apply_filter': self.apply_filter_check.isChecked(),
            'filter_low': self.highpass_spin.value() if self.apply_filter_check.isChecked() else None,
            'filter_high': self.lowpass_spin.value() if self.apply_filter_check.isChecked() else None,
            'apply_notch': self.apply_notch_check.isChecked(),
            'notch_freq': 50.0,
            'apply_reref': self.apply_reref_check.isChecked(),
            'ref_channels': self.reference_combo.currentText().lower().replace(' ', '_')
        }
        self.preprocessing_configured.emit(config)