#!/usr/bin/env python3
"""
Advanced Preprocessing GUI Widget
================================

GUI widget for Phase 1 advanced preprocessing features:
- Advanced filtering interface
- Re-referencing options
- Channel management and analysis
- Complete preprocessing pipeline

Author: porfanid
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QTextEdit,
    QGroupBox, QTabWidget, QProgressBar, QMessageBox,
    QSplitter, QScrollArea, QFrame
)
from PyQt6.QtGui import QFont

import mne
import numpy as np

from backend import (
    PreprocessingPipeline, PreprocessingConfig, PreprocessingPresets,
    FilterConfig, FilterPresets, ReferenceConfig, ReferencePresets,
    EEGChannelManager
)


class PreprocessingWorkerThread(QThread):
    """Worker thread for preprocessing operations"""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    preprocessing_complete = pyqtSignal(bool, str, dict, object)  # success, message, results, processed_raw
    
    def __init__(self, raw, config):
        super().__init__()
        self.raw = raw
        self.config = config
        self.pipeline = PreprocessingPipeline()
    
    def run(self):
        try:
            self.status_update.emit("Starting preprocessing pipeline...")
            self.progress_update.emit(10)
            
            # Run preprocessing
            processed_raw, results = self.pipeline.run_pipeline(self.raw, self.config)
            
            self.progress_update.emit(100)
            self.preprocessing_complete.emit(True, "Preprocessing completed successfully!", results, processed_raw)
            
        except Exception as e:
            self.preprocessing_complete.emit(False, f"Preprocessing failed: {str(e)}", {}, None)


class FilterConfigWidget(QWidget):
    """Widget for configuring individual filters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Filter type
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(['highpass', 'lowpass', 'bandpass', 'bandstop', 'notch'])
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.filter_type_combo)
        
        # Frequency parameters
        self.freq_low_spin = QDoubleSpinBox()
        self.freq_low_spin.setRange(0.01, 1000.0)
        self.freq_low_spin.setValue(1.0)
        self.freq_low_spin.setSuffix(" Hz")
        layout.addWidget(QLabel("Low:"))
        layout.addWidget(self.freq_low_spin)
        
        self.freq_high_spin = QDoubleSpinBox()
        self.freq_high_spin.setRange(0.01, 1000.0)
        self.freq_high_spin.setValue(40.0)
        self.freq_high_spin.setSuffix(" Hz")
        layout.addWidget(QLabel("High:"))
        layout.addWidget(self.freq_high_spin)
        
        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItems(['fir', 'iir'])
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_combo)
        
        # Connect signals
        self.filter_type_combo.currentTextChanged.connect(self.update_frequency_widgets)
        self.update_frequency_widgets()
    
    def update_frequency_widgets(self):
        """Update frequency widget visibility based on filter type"""
        filter_type = self.filter_type_combo.currentText()
        
        # Show/hide frequency controls based on filter type
        if filter_type in ['highpass', 'lowpass']:
            self.freq_high_spin.setVisible(filter_type == 'lowpass')
        elif filter_type in ['bandpass', 'bandstop']:
            self.freq_high_spin.setVisible(True)
        elif filter_type == 'notch':
            self.freq_high_spin.setVisible(False)
    
    def get_filter_config(self) -> FilterConfig:
        """Get filter configuration from widget settings"""
        filter_type = self.filter_type_combo.currentText()
        method = self.method_combo.currentText()
        
        if filter_type == 'highpass':
            return FilterConfig(filter_type, freq_low=self.freq_low_spin.value(), method=method)
        elif filter_type == 'lowpass':
            return FilterConfig(filter_type, freq_low=self.freq_low_spin.value(), method=method)
        elif filter_type in ['bandpass', 'bandstop']:
            return FilterConfig(
                filter_type, 
                freq_low=self.freq_low_spin.value(),
                freq_high=self.freq_high_spin.value(),
                method=method
            )
        elif filter_type == 'notch':
            return FilterConfig(filter_type, freq_notch=self.freq_low_spin.value(), method=method)


class AdvancedPreprocessingWidget(QWidget):
    """Main widget for advanced preprocessing interface"""
    
    # Signals
    preprocessing_complete = pyqtSignal(object, dict)  # processed_raw, results
    
    def __init__(self, theme=None, parent=None):
        super().__init__(parent)
        self.theme = theme or {}
        self.raw_data = None
        self.channel_manager = EEGChannelManager()
        self.current_analysis = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔧 Advanced EEG Preprocessing")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Main content in tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Tab 1: Channel Analysis
        self.create_channel_analysis_tab()
        
        # Tab 2: Filtering
        self.create_filtering_tab()
        
        # Tab 3: Re-referencing
        self.create_referencing_tab()
        
        # Tab 4: Pipeline Configuration
        self.create_pipeline_tab()
        
        # Bottom section: Progress and controls
        self.create_bottom_section(layout)
    
    def create_channel_analysis_tab(self):
        """Create channel analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis controls
        controls_group = QGroupBox("Channel Analysis")
        controls_layout = QHBoxLayout(controls_group)
        
        self.analyze_btn = QPushButton("🔍 Analyze Channels")
        self.analyze_btn.setEnabled(False)
        controls_layout.addWidget(self.analyze_btn)
        
        self.detect_bad_btn = QPushButton("⚠️ Detect Bad Channels")
        self.detect_bad_btn.setEnabled(False)
        controls_layout.addWidget(self.detect_bad_btn)
        
        layout.addWidget(controls_group)
        
        # Analysis results
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        results_layout.addWidget(self.analysis_text)
        
        layout.addWidget(results_group)
        
        # Bad channels management
        bad_channels_group = QGroupBox("Bad Channels Management")
        bad_layout = QHBoxLayout(bad_channels_group)
        
        # Bad channels list
        bad_list_layout = QVBoxLayout()
        bad_list_layout.addWidget(QLabel("Detected Bad Channels:"))
        self.bad_channels_list = QListWidget()
        bad_list_layout.addWidget(self.bad_channels_list)
        bad_layout.addLayout(bad_list_layout)
        
        # Bad channels controls
        bad_controls_layout = QVBoxLayout()
        self.interpolate_bad_check = QCheckBox("Interpolate bad channels")
        self.interpolate_bad_check.setChecked(True)
        bad_controls_layout.addWidget(self.interpolate_bad_check)
        
        self.remove_bad_btn = QPushButton("Remove Selected")
        bad_controls_layout.addWidget(self.remove_bad_btn)
        
        self.add_bad_btn = QPushButton("Mark as Bad")
        bad_controls_layout.addWidget(self.add_bad_btn)
        
        bad_controls_layout.addStretch()
        bad_layout.addLayout(bad_controls_layout)
        
        layout.addWidget(bad_channels_group)
        
        self.tab_widget.addTab(tab, "📊 Channel Analysis")
    
    def create_filtering_tab(self):
        """Create filtering configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Filter presets
        presets_group = QGroupBox("Filter Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom", "Preprocessing", "ERP Analysis", 
            "Alpha Analysis", "Beta Analysis", "Gamma Analysis"
        ])
        presets_layout.addWidget(QLabel("Preset:"))
        presets_layout.addWidget(self.preset_combo)
        
        self.load_preset_btn = QPushButton("Load Preset")
        presets_layout.addWidget(self.load_preset_btn)
        
        layout.addWidget(presets_group)
        
        # Filter configuration
        filters_group = QGroupBox("Filter Configuration")
        filters_layout = QVBoxLayout(filters_group)
        
        # Add filter button
        add_filter_layout = QHBoxLayout()
        self.add_filter_btn = QPushButton("➕ Add Filter")
        add_filter_layout.addWidget(self.add_filter_btn)
        add_filter_layout.addStretch()
        filters_layout.addLayout(add_filter_layout)
        
        # Filters list
        self.filters_scroll = QScrollArea()
        self.filters_widget = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_widget)
        self.filters_scroll.setWidget(self.filters_widget)
        self.filters_scroll.setWidgetResizable(True)
        filters_layout.addWidget(self.filters_scroll)
        
        layout.addWidget(filters_group)
        
        self.tab_widget.addTab(tab, "🔄 Filtering")
    
    def create_referencing_tab(self):
        """Create re-referencing configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Reference type selection
        ref_type_group = QGroupBox("Reference Type")
        ref_type_layout = QGridLayout(ref_type_group)
        
        self.ref_type_combo = QComboBox()
        self.ref_type_combo.addItems([
            "average", "common", "bipolar", "linked_ears", "custom"
        ])
        ref_type_layout.addWidget(QLabel("Reference Type:"), 0, 0)
        ref_type_layout.addWidget(self.ref_type_combo, 0, 1)
        
        # Reference channels
        self.ref_channels_combo = QComboBox()
        self.ref_channels_combo.setEnabled(False)
        ref_type_layout.addWidget(QLabel("Reference Channel:"), 1, 0)
        ref_type_layout.addWidget(self.ref_channels_combo, 1, 1)
        
        layout.addWidget(ref_type_group)
        
        # Reference presets
        presets_group = QGroupBox("Reference Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        self.ref_preset_combo = QComboBox()
        self.ref_preset_combo.addItems([
            "Clinical", "Research", "ERP", "Sleep Study"
        ])
        presets_layout.addWidget(QLabel("Preset:"))
        presets_layout.addWidget(self.ref_preset_combo)
        
        self.load_ref_preset_btn = QPushButton("Load Preset")
        presets_layout.addWidget(self.load_ref_preset_btn)
        
        layout.addWidget(presets_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🔗 Re-referencing")
    
    def create_pipeline_tab(self):
        """Create preprocessing pipeline configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Pipeline presets
        presets_group = QGroupBox("Pipeline Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        self.pipeline_preset_combo = QComboBox()
        self.pipeline_preset_combo.addItems([
            "Custom", "Clinical", "Research", "ERP Analysis", "Sleep Study", "Minimal"
        ])
        presets_layout.addWidget(QLabel("Preset:"))
        presets_layout.addWidget(self.pipeline_preset_combo)
        
        self.load_pipeline_preset_btn = QPushButton("Load Preset")
        presets_layout.addWidget(self.load_pipeline_preset_btn)
        
        layout.addWidget(presets_group)
        
        # Pipeline steps
        steps_group = QGroupBox("Pipeline Steps")
        steps_layout = QVBoxLayout(steps_group)
        
        self.enable_filtering_check = QCheckBox("Enable Filtering")
        self.enable_filtering_check.setChecked(True)
        steps_layout.addWidget(self.enable_filtering_check)
        
        self.enable_bad_detection_check = QCheckBox("Detect Bad Channels")
        self.enable_bad_detection_check.setChecked(True)
        steps_layout.addWidget(self.enable_bad_detection_check)
        
        self.enable_interpolation_check = QCheckBox("Interpolate Bad Channels")
        self.enable_interpolation_check.setChecked(True)
        steps_layout.addWidget(self.enable_interpolation_check)
        
        self.enable_montage_check = QCheckBox("Load Standard Montage")
        self.enable_montage_check.setChecked(True)
        steps_layout.addWidget(self.enable_montage_check)
        
        self.enable_referencing_check = QCheckBox("Apply Re-referencing")
        self.enable_referencing_check.setChecked(True)
        steps_layout.addWidget(self.enable_referencing_check)
        
        layout.addWidget(steps_group)
        
        # Montage selection
        montage_group = QGroupBox("Montage Selection")
        montage_layout = QHBoxLayout(montage_group)
        
        self.montage_combo = QComboBox()
        self.montage_combo.addItems([
            'standard_1020', 'standard_1005', 'biosemi64', 'biosemi128'
        ])
        montage_layout.addWidget(QLabel("Montage:"))
        montage_layout.addWidget(self.montage_combo)
        
        layout.addWidget(montage_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "⚙️ Pipeline")
    
    def create_bottom_section(self, parent_layout):
        """Create bottom section with progress and controls"""
        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(progress_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("👁️ Preview Settings")
        self.preview_btn.setEnabled(False)
        controls_layout.addWidget(self.preview_btn)
        
        controls_layout.addStretch()
        
        self.reset_btn = QPushButton("🔄 Reset")
        controls_layout.addWidget(self.reset_btn)
        
        self.run_preprocessing_btn = QPushButton("▶️ Run Preprocessing")
        self.run_preprocessing_btn.setEnabled(False)
        controls_layout.addWidget(self.run_preprocessing_btn)
        
        parent_layout.addLayout(controls_layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Channel analysis
        self.analyze_btn.clicked.connect(self.analyze_channels)
        self.detect_bad_btn.clicked.connect(self.detect_bad_channels)
        
        # Filtering
        self.add_filter_btn.clicked.connect(self.add_filter_widget)
        self.load_preset_btn.clicked.connect(self.load_filter_preset)
        
        # Referencing
        self.ref_type_combo.currentTextChanged.connect(self.update_reference_widgets)
        self.load_ref_preset_btn.clicked.connect(self.load_reference_preset)
        
        # Pipeline
        self.load_pipeline_preset_btn.clicked.connect(self.load_pipeline_preset)
        
        # Controls
        self.preview_btn.clicked.connect(self.preview_settings)
        self.reset_btn.clicked.connect(self.reset_settings)
        self.run_preprocessing_btn.clicked.connect(self.run_preprocessing)
    
    def load_data(self, file_path: str, selected_channels: List[str] = None):
        """Load EEG data from file path with optional channel selection"""
        try:
            self.status_label.setText("Loading EEG data...")
            
            # Load raw data using mne
            if file_path.endswith('.edf'):
                raw = mne.io.read_raw_edf(file_path, preload=True)
            elif file_path.endswith('.bdf'):
                raw = mne.io.read_raw_bdf(file_path, preload=True)
            elif file_path.endswith('.fif'):
                raw = mne.io.read_raw_fif(file_path, preload=True)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Select specific channels if provided
            if selected_channels:
                raw = raw.pick_channels(selected_channels)
            
            self.set_raw_data(raw)
            self.status_label.setText(f"Loaded {len(raw.ch_names)} channels from {Path(file_path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"Failed to load EEG data:\n{str(e)}")
            self.status_label.setText("Loading failed")

    def set_raw_data(self, raw: mne.io.Raw):
        """Set the raw EEG data for preprocessing"""
        self.raw_data = raw
        
        # Enable controls
        self.analyze_btn.setEnabled(True)
        self.detect_bad_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.run_preprocessing_btn.setEnabled(True)
        
        # Update channel lists
        self.update_channel_lists()
        
        # Auto-analyze channels
        self.analyze_channels()
    
    def update_channel_lists(self):
        """Update channel combo boxes with current channels"""
        if self.raw_data is None:
            return
        
        channels = self.raw_data.ch_names
        self.ref_channels_combo.clear()
        self.ref_channels_combo.addItems(channels)
    
    def analyze_channels(self):
        """Analyze current channels"""
        if self.raw_data is None:
            return
        
        try:
            self.status_label.setText("Analyzing channels...")
            
            analysis = self.channel_manager.analyze_channels(self.raw_data)
            self.current_analysis = analysis
            
            # Display analysis results
            self.display_channel_analysis(analysis)
            
            self.status_label.setText("Channel analysis complete")
            
        except Exception as e:
            QMessageBox.warning(self, "Analysis Error", f"Channel analysis failed:\n{str(e)}")
            self.status_label.setText("Analysis failed")
    
    def display_channel_analysis(self, analysis: Dict):
        """Display channel analysis results"""
        text_parts = []
        
        text_parts.append(f"📊 Channel Analysis Results")
        text_parts.append(f"=" * 30)
        text_parts.append(f"Total channels: {analysis['n_channels']}")
        text_parts.append(f"Duration: {analysis.get('duration', 'N/A')}")
        
        # Channel types
        text_parts.append(f"\n📋 Channel Types:")
        for ch_type, count in analysis['channel_types'].items():
            text_parts.append(f"  {ch_type}: {count}")
        
        # Bad channels
        bad_channels = analysis.get('bad_channels', {})
        text_parts.append(f"\n⚠️ Bad Channels:")
        total_bad = 0
        for bad_type, channels in bad_channels.items():
            if channels:
                text_parts.append(f"  {bad_type}: {len(channels)} channels")
                total_bad += len(channels)
        if total_bad == 0:
            text_parts.append("  No bad channels detected")
        
        # Montage info
        montage_info = analysis.get('montage_info', {})
        text_parts.append(f"\n🗺️ Montage:")
        if montage_info.get('has_montage', False):
            text_parts.append(f"  ✅ Montage present")
        else:
            text_parts.append(f"  ❌ No montage loaded")
        
        self.analysis_text.setPlainText("\n".join(text_parts))
    
    def detect_bad_channels(self):
        """Detect and display bad channels"""
        if self.raw_data is None or self.current_analysis is None:
            return
        
        try:
            bad_channels = self.current_analysis.get('bad_channels', {})
            
            # Clear and populate bad channels list
            self.bad_channels_list.clear()
            
            all_bad = set()
            for bad_type, channels in bad_channels.items():
                for ch in channels:
                    all_bad.add(ch)
                    item = QListWidgetItem(f"{ch} ({bad_type})")
                    self.bad_channels_list.addItem(item)
            
            if not all_bad:
                item = QListWidgetItem("No bad channels detected")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.bad_channels_list.addItem(item)
            
            self.status_label.setText(f"Found {len(all_bad)} bad channels")
            
        except Exception as e:
            QMessageBox.warning(self, "Detection Error", f"Bad channel detection failed:\n{str(e)}")
    
    def add_filter_widget(self):
        """Add a new filter configuration widget"""
        filter_widget = FilterConfigWidget()
        
        # Create container with remove button
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.addWidget(filter_widget)
        
        remove_btn = QPushButton("❌")
        remove_btn.setMaximumWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_filter_widget(container))
        container_layout.addWidget(remove_btn)
        
        self.filters_layout.addWidget(container)
    
    def remove_filter_widget(self, widget):
        """Remove a filter configuration widget"""
        self.filters_layout.removeWidget(widget)
        widget.deleteLater()
    
    def load_filter_preset(self):
        """Load filter preset"""
        preset = self.preset_combo.currentText()
        
        # Clear existing filters
        for i in reversed(range(self.filters_layout.count())):
            child = self.filters_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add preset filters
        if preset == "Preprocessing":
            configs = FilterPresets.get_preprocessing_preset()
        elif preset == "ERP Analysis":
            configs = FilterPresets.get_erp_preset()
        elif preset == "Alpha Analysis":
            configs = FilterPresets.get_alpha_analysis_preset()
        elif preset == "Beta Analysis":
            configs = FilterPresets.get_beta_analysis_preset()
        elif preset == "Gamma Analysis":
            configs = FilterPresets.get_gamma_analysis_preset()
        else:
            return
        
        for config in configs:
            self.add_filter_widget()
            # Set the last added widget's configuration
            # (This would need more implementation to set specific values)
    
    def update_reference_widgets(self):
        """Update reference widgets based on selected type"""
        ref_type = self.ref_type_combo.currentText()
        self.ref_channels_combo.setEnabled(ref_type in ['common', 'custom'])
    
    def load_reference_preset(self):
        """Load reference preset"""
        preset = self.ref_preset_combo.currentText()
        
        if preset == "Clinical":
            self.ref_type_combo.setCurrentText("average")
        elif preset == "Research":
            self.ref_type_combo.setCurrentText("average")
        elif preset == "ERP":
            self.ref_type_combo.setCurrentText("average")
        elif preset == "Sleep Study":
            self.ref_type_combo.setCurrentText("linked_ears")
    
    def load_pipeline_preset(self):
        """Load pipeline preset"""
        preset = self.pipeline_preset_combo.currentText()
        
        if preset == "Clinical":
            self.enable_all_steps(True)
        elif preset == "Research":
            self.enable_all_steps(True)
        elif preset == "ERP Analysis":
            self.enable_all_steps(True)
        elif preset == "Sleep Study":
            self.enable_all_steps(True)
            self.enable_interpolation_check.setChecked(False)
        elif preset == "Minimal":
            self.enable_all_steps(False)
            self.enable_filtering_check.setChecked(True)
    
    def enable_all_steps(self, enabled: bool):
        """Enable/disable all pipeline steps"""
        self.enable_filtering_check.setChecked(enabled)
        self.enable_bad_detection_check.setChecked(enabled)
        self.enable_interpolation_check.setChecked(enabled)
        self.enable_montage_check.setChecked(enabled)
        self.enable_referencing_check.setChecked(enabled)
    
    def preview_settings(self):
        """Preview current preprocessing settings"""
        if self.raw_data is None:
            return
        
        try:
            config = self.get_preprocessing_config()
            
            # Create preview text
            preview_text = []
            preview_text.append("🔧 Preprocessing Configuration Preview")
            preview_text.append("=" * 40)
            
            if config.apply_filters:
                preview_text.append(f"✅ Filtering: {len(config.filter_configs)} filters")
            else:
                preview_text.append("❌ Filtering: Disabled")
            
            if config.detect_bad_channels:
                preview_text.append("✅ Bad Channel Detection: Enabled")
            else:
                preview_text.append("❌ Bad Channel Detection: Disabled")
            
            if config.interpolate_bad_channels:
                preview_text.append("✅ Channel Interpolation: Enabled")
            else:
                preview_text.append("❌ Channel Interpolation: Disabled")
            
            if config.load_montage:
                preview_text.append(f"✅ Montage: {config.montage_name}")
            else:
                preview_text.append("❌ Montage: Disabled")
            
            if config.apply_reference:
                preview_text.append(f"✅ Re-referencing: {config.reference_config.ref_type}")
            else:
                preview_text.append("❌ Re-referencing: Disabled")
            
            QMessageBox.information(self, "Preview", "\n".join(preview_text))
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Cannot preview settings:\n{str(e)}")
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get current preprocessing configuration"""
        # Get filter configurations
        filter_configs = []
        if self.enable_filtering_check.isChecked():
            # Collect all filter widgets (simplified)
            filter_configs = FilterPresets.get_preprocessing_preset()  # Default for now
        
        # Get reference configuration
        ref_config = None
        if self.enable_referencing_check.isChecked():
            ref_type = self.ref_type_combo.currentText()
            if ref_type == 'common':
                ref_channels = self.ref_channels_combo.currentText()
                ref_config = ReferenceConfig(ref_type, ref_channels=ref_channels)
            else:
                ref_config = ReferenceConfig(ref_type)
        
        return PreprocessingConfig(
            apply_filters=self.enable_filtering_check.isChecked(),
            filter_configs=filter_configs,
            detect_bad_channels=self.enable_bad_detection_check.isChecked(),
            interpolate_bad_channels=self.enable_interpolation_check.isChecked(),
            load_montage=self.enable_montage_check.isChecked(),
            montage_name=self.montage_combo.currentText(),
            apply_reference=self.enable_referencing_check.isChecked(),
            reference_config=ref_config,
            verbose=False
        )
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        # Reset pipeline steps
        self.enable_all_steps(True)
        
        # Reset reference
        self.ref_type_combo.setCurrentText("average")
        
        # Reset montage
        self.montage_combo.setCurrentText("standard_1020")
        
        # Clear filters
        for i in reversed(range(self.filters_layout.count())):
            child = self.filters_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Clear analysis
        self.analysis_text.clear()
        self.bad_channels_list.clear()
        
        self.status_label.setText("Settings reset")
    
    def run_preprocessing(self):
        """Run the preprocessing pipeline"""
        if self.raw_data is None:
            return
        
        try:
            config = self.get_preprocessing_config()
            
            # Start preprocessing in worker thread
            self.worker_thread = PreprocessingWorkerThread(self.raw_data, config)
            self.worker_thread.progress_update.connect(self.progress_bar.setValue)
            self.worker_thread.status_update.connect(self.status_label.setText)
            self.worker_thread.preprocessing_complete.connect(self.on_preprocessing_complete)
            
            # Disable controls during processing
            self.run_preprocessing_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", f"Failed to start preprocessing:\n{str(e)}")
    
    def on_preprocessing_complete(self, success: bool, message: str, results: Dict, processed_raw):
        """Handle preprocessing completion"""
        # Re-enable controls
        self.run_preprocessing_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Emit the processed data to parent
            self.preprocessing_complete.emit(processed_raw, results)
            
        else:
            QMessageBox.critical(self, "Error", message)
        
        self.status_label.setText("Ready")