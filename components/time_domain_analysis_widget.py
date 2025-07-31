#!/usr/bin/env python3
"""
Time-Domain Analysis GUI Widget
==============================

Phase 3 GUI component for time-domain analysis including:
- Epoching and segmentation controls
- ERP analysis interface
- Time-domain visualization
- Statistical comparisons

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
    QSplitter, QScrollArea, QFrame, QSlider
)
from PyQt6.QtGui import QFont

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backend import (
    EpochingProcessor, EpochingConfig, SegmentationConfig, 
    BaselineCorrectionMethod, EpochRejectionCriteria,
    ERPAnalyzer, ERPConfig, PeakDetectionMethod, ERPComponent, StatisticalTest,
    TimeDomainVisualizer, PlotConfig, PlotType, LayoutType
)


class TimeDomainWorkerThread(QThread):
    """Worker thread for time-domain analysis operations"""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    analysis_complete = pyqtSignal(bool, str, dict)  # success, message, results
    
    def __init__(self, raw, analysis_config):
        super().__init__()
        self.raw = raw
        self.analysis_config = analysis_config
        
    def run(self):
        try:
            self.status_update.emit("Starting time-domain analysis...")
            self.progress_update.emit(10)
            
            results = {}
            
            # Epoching
            if self.analysis_config.get('create_epochs', False):
                self.status_update.emit("Creating epochs...")
                epoching_processor = EpochingProcessor()
                
                if self.analysis_config.get('use_events', True):
                    # Find events
                    events = epoching_processor.find_events_from_raw(self.raw)
                    epochs = epoching_processor.create_epochs_from_events(
                        self.raw, events, self.analysis_config['epoching_config']
                    )
                else:
                    # Fixed-length epochs
                    epochs = epoching_processor.create_fixed_length_epochs(
                        self.raw, self.analysis_config['segmentation_config']
                    )
                
                results['epochs'] = epochs
                results['epoching_metrics'] = epoching_processor.quality_metrics_
                self.progress_update.emit(40)
            
            # ERP Analysis
            if self.analysis_config.get('compute_erp', False) and 'epochs' in results:
                self.status_update.emit("Computing ERPs...")
                erp_analyzer = ERPAnalyzer(self.analysis_config['erp_config'])
                
                erp = erp_analyzer.compute_erp(results['epochs'])
                results['erp'] = erp
                results['erp_statistics'] = erp_analyzer.statistics_
                
                # Peak detection
                peaks = erp_analyzer.detect_peaks(erp)
                results['peaks'] = peaks
                
                self.progress_update.emit(70)
            
            # Visualization
            if self.analysis_config.get('create_plots', False):
                self.status_update.emit("Creating visualizations...")
                visualizer = TimeDomainVisualizer(self.analysis_config['plot_config'])
                
                plots = {}
                
                # Time series plot
                if 'time_series' in self.analysis_config.get('plot_types', []):
                    fig = visualizer.plot_timeseries(self.raw)
                    plots['timeseries'] = fig
                
                # ERP plot
                if 'erp' in self.analysis_config.get('plot_types', []) and 'erp' in results:
                    fig = visualizer.plot_erp(results['erp'])
                    plots['erp'] = fig
                
                # Butterfly plot
                if 'butterfly' in self.analysis_config.get('plot_types', []) and 'erp' in results:
                    fig = visualizer.plot_butterfly(results['erp'])
                    plots['butterfly'] = fig
                
                results['plots'] = plots
                self.progress_update.emit(90)
            
            self.progress_update.emit(100)
            self.analysis_complete.emit(True, "Time-domain analysis completed successfully!", results)
            
        except Exception as e:
            self.analysis_complete.emit(False, f"Analysis failed: {str(e)}", {})


class EpochingControlsWidget(QWidget):
    """Widget for epoching configuration controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Epoching method
        method_group = QGroupBox("Epoching Method")
        method_layout = QVBoxLayout(method_group)
        
        self.event_based_radio = QCheckBox("Event-based epoching")
        self.event_based_radio.setChecked(True)
        method_layout.addWidget(self.event_based_radio)
        
        self.fixed_length_radio = QCheckBox("Fixed-length epoching")
        method_layout.addWidget(self.fixed_length_radio)
        
        layout.addWidget(method_group)
        
        # Event-based parameters
        event_group = QGroupBox("Event-based Parameters")
        event_layout = QGridLayout(event_group)
        
        event_layout.addWidget(QLabel("Pre-stimulus (s):"), 0, 0)
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(-10.0, 0.0)
        self.tmin_spin.setValue(-0.2)
        self.tmin_spin.setDecimals(3)
        self.tmin_spin.setSuffix(" s")
        event_layout.addWidget(self.tmin_spin, 0, 1)
        
        event_layout.addWidget(QLabel("Post-stimulus (s):"), 0, 2)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0.0, 10.0)
        self.tmax_spin.setValue(0.8)
        self.tmax_spin.setDecimals(3)
        self.tmax_spin.setSuffix(" s")
        event_layout.addWidget(self.tmax_spin, 0, 3)
        
        event_layout.addWidget(QLabel("Baseline method:"), 1, 0)
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItems(['mean', 'median', 'rescale', 'zscore', 'logratio', 'none'])
        event_layout.addWidget(self.baseline_combo, 1, 1)
        
        event_layout.addWidget(QLabel("Baseline start (s):"), 1, 2)
        self.baseline_start_spin = QDoubleSpinBox()
        self.baseline_start_spin.setRange(-10.0, 0.0)
        self.baseline_start_spin.setValue(-0.1)
        self.baseline_start_spin.setDecimals(3)
        self.baseline_start_spin.setSuffix(" s")
        event_layout.addWidget(self.baseline_start_spin, 1, 3)
        
        event_layout.addWidget(QLabel("Baseline end (s):"), 2, 0)
        self.baseline_end_spin = QDoubleSpinBox()
        self.baseline_end_spin.setRange(-10.0, 1.0)
        self.baseline_end_spin.setValue(0.0)
        self.baseline_end_spin.setDecimals(3)
        self.baseline_end_spin.setSuffix(" s")
        event_layout.addWidget(self.baseline_end_spin, 2, 1)
        
        layout.addWidget(event_group)
        
        # Fixed-length parameters
        segment_group = QGroupBox("Fixed-length Parameters")
        segment_layout = QGridLayout(segment_group)
        
        segment_layout.addWidget(QLabel("Segment length (s):"), 0, 0)
        self.segment_length_spin = QDoubleSpinBox()
        self.segment_length_spin.setRange(0.1, 60.0)
        self.segment_length_spin.setValue(2.0)
        self.segment_length_spin.setDecimals(1)
        self.segment_length_spin.setSuffix(" s")
        segment_layout.addWidget(self.segment_length_spin, 0, 1)
        
        segment_layout.addWidget(QLabel("Overlap (%):"), 0, 2)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setValue(0)
        self.overlap_spin.setSuffix(" %")
        segment_layout.addWidget(self.overlap_spin, 0, 3)
        
        layout.addWidget(segment_group)
        
        # Rejection criteria
        rejection_group = QGroupBox("Epoch Rejection Criteria")
        rejection_layout = QGridLayout(rejection_group)
        
        rejection_layout.addWidget(QLabel("Amplitude threshold (µV):"), 0, 0)
        self.amp_threshold_spin = QSpinBox()
        self.amp_threshold_spin.setRange(50, 500)
        self.amp_threshold_spin.setValue(150)
        self.amp_threshold_spin.setSuffix(" µV")
        rejection_layout.addWidget(self.amp_threshold_spin, 0, 1)
        
        rejection_layout.addWidget(QLabel("Gradient threshold (µV):"), 0, 2)
        self.grad_threshold_spin = QSpinBox()
        self.grad_threshold_spin.setRange(25, 200)
        self.grad_threshold_spin.setValue(75)
        self.grad_threshold_spin.setSuffix(" µV")
        rejection_layout.addWidget(self.grad_threshold_spin, 0, 3)
        
        self.enable_rejection_checkbox = QCheckBox("Enable automatic rejection")
        self.enable_rejection_checkbox.setChecked(True)
        rejection_layout.addWidget(self.enable_rejection_checkbox, 1, 0, 1, 2)
        
        layout.addWidget(rejection_group)
    
    def get_epoching_config(self) -> EpochingConfig:
        """Get epoching configuration from UI controls"""
        
        baseline_method = BaselineCorrectionMethod(self.baseline_combo.currentText())
        
        rejection_criteria = {}
        if self.enable_rejection_checkbox.isChecked():
            rejection_criteria[EpochRejectionCriteria.AMPLITUDE] = self.amp_threshold_spin.value() * 1e-6
            rejection_criteria[EpochRejectionCriteria.GRADIENT] = self.grad_threshold_spin.value() * 1e-6
        
        return EpochingConfig(
            tmin=self.tmin_spin.value(),
            tmax=self.tmax_spin.value(),
            baseline_method=baseline_method,
            baseline_tmin=self.baseline_start_spin.value(),
            baseline_tmax=self.baseline_end_spin.value(),
            rejection_criteria=rejection_criteria
        )
    
    def get_segmentation_config(self) -> SegmentationConfig:
        """Get segmentation configuration from UI controls"""
        
        return SegmentationConfig(
            segment_length=self.segment_length_spin.value(),
            overlap=self.overlap_spin.value() / 100.0,
            max_amplitude=self.amp_threshold_spin.value() * 1e-6,
            max_gradient=self.grad_threshold_spin.value() * 1e-6
        )


class ERPAnalysisWidget(QWidget):
    """Widget for ERP analysis controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # ERP computation
        computation_group = QGroupBox("ERP Computation")
        computation_layout = QGridLayout(computation_group)
        
        computation_layout.addWidget(QLabel("Averaging method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['mean', 'median', 'trimmed_mean'])
        computation_layout.addWidget(self.method_combo, 0, 1)
        
        computation_layout.addWidget(QLabel("Trim percentage:"), 0, 2)
        self.trim_spin = QSpinBox()
        self.trim_spin.setRange(0, 25)
        self.trim_spin.setValue(10)
        self.trim_spin.setSuffix(" %")
        computation_layout.addWidget(self.trim_spin, 0, 3)
        
        self.smoothing_checkbox = QCheckBox("Apply smoothing")
        computation_layout.addWidget(self.smoothing_checkbox, 1, 0)
        
        computation_layout.addWidget(QLabel("Smoothing window (ms):"), 1, 1)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 50)
        self.smoothing_spin.setValue(10)
        self.smoothing_spin.setSuffix(" ms")
        computation_layout.addWidget(self.smoothing_spin, 1, 2)
        
        layout.addWidget(computation_group)
        
        # Peak detection
        peaks_group = QGroupBox("Peak Detection")
        peaks_layout = QGridLayout(peaks_group)
        
        peaks_layout.addWidget(QLabel("Detection method:"), 0, 0)
        self.peak_method_combo = QComboBox()
        self.peak_method_combo.addItems(['automatic', 'adaptive_threshold', 'manual'])
        peaks_layout.addWidget(self.peak_method_combo, 0, 1)
        
        self.detect_components_checkbox = QCheckBox("Detect standard ERP components")
        self.detect_components_checkbox.setChecked(True)
        peaks_layout.addWidget(self.detect_components_checkbox, 1, 0, 1, 2)
        
        # Component selection
        components_layout = QHBoxLayout()
        self.p1_checkbox = QCheckBox("P1")
        self.n1_checkbox = QCheckBox("N1")
        self.p2_checkbox = QCheckBox("P2")
        self.n2_checkbox = QCheckBox("N2")
        self.p3_checkbox = QCheckBox("P3")
        
        for checkbox in [self.p1_checkbox, self.n1_checkbox, self.p2_checkbox, self.n2_checkbox, self.p3_checkbox]:
            checkbox.setChecked(True)
            components_layout.addWidget(checkbox)
        
        peaks_layout.addLayout(components_layout, 2, 0, 1, 4)
        
        layout.addWidget(peaks_group)
        
        # Statistical analysis
        stats_group = QGroupBox("Statistical Analysis")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("Confidence level:"), 0, 0)
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(90, 99)
        self.confidence_spin.setValue(95)
        self.confidence_spin.setSuffix(" %")
        stats_layout.addWidget(self.confidence_spin, 0, 1)
        
        self.compute_auc_checkbox = QCheckBox("Compute area under curve")
        self.compute_auc_checkbox.setChecked(True)
        stats_layout.addWidget(self.compute_auc_checkbox, 1, 0, 1, 2)
        
        layout.addWidget(stats_group)
    
    def get_erp_config(self) -> ERPConfig:
        """Get ERP configuration from UI controls"""
        
        # Get selected components
        components = {}
        if self.detect_components_checkbox.isChecked():
            if self.p1_checkbox.isChecked():
                components[ERPComponent.P1] = (0.08, 0.12)
            if self.n1_checkbox.isChecked():
                components[ERPComponent.N1] = (0.12, 0.20)
            if self.p2_checkbox.isChecked():
                components[ERPComponent.P2] = (0.15, 0.25)
            if self.n2_checkbox.isChecked():
                components[ERPComponent.N2] = (0.20, 0.35)
            if self.p3_checkbox.isChecked():
                components[ERPComponent.P3] = (0.30, 0.60)
        
        return ERPConfig(
            method=self.method_combo.currentText(),
            trim_percent=self.trim_spin.value() / 100.0,
            confidence_level=self.confidence_spin.value() / 100.0,
            peak_detection_method=PeakDetectionMethod(self.peak_method_combo.currentText()),
            component_windows=components if components else None,
            apply_smoothing=self.smoothing_checkbox.isChecked(),
            smoothing_window=self.smoothing_spin.value() / 1000.0  # Convert to seconds
        )


class VisualizationControlsWidget(QWidget):
    """Widget for visualization controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Plot types
        plots_group = QGroupBox("Plot Types")
        plots_layout = QVBoxLayout(plots_group)
        
        self.timeseries_checkbox = QCheckBox("Time series plot")
        self.timeseries_checkbox.setChecked(True)
        plots_layout.addWidget(self.timeseries_checkbox)
        
        self.erp_checkbox = QCheckBox("ERP waveforms")
        self.erp_checkbox.setChecked(True)
        plots_layout.addWidget(self.erp_checkbox)
        
        self.butterfly_checkbox = QCheckBox("Butterfly plot")
        self.butterfly_checkbox.setChecked(True)
        plots_layout.addWidget(self.butterfly_checkbox)
        
        self.channel_array_checkbox = QCheckBox("Channel array")
        plots_layout.addWidget(self.channel_array_checkbox)
        
        layout.addWidget(plots_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout(display_group)
        
        display_layout.addWidget(QLabel("Time unit:"), 0, 0)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(['s', 'ms'])
        display_layout.addWidget(self.time_unit_combo, 0, 1)
        
        display_layout.addWidget(QLabel("Color palette:"), 0, 2)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(['husl', 'Set1', 'viridis', 'plasma', 'tab10'])
        display_layout.addWidget(self.palette_combo, 0, 3)
        
        self.show_ci_checkbox = QCheckBox("Show confidence intervals")
        self.show_ci_checkbox.setChecked(True)
        display_layout.addWidget(self.show_ci_checkbox, 1, 0, 1, 2)
        
        self.show_zero_checkbox = QCheckBox("Show zero lines")
        self.show_zero_checkbox.setChecked(True)
        display_layout.addWidget(self.show_zero_checkbox, 1, 2, 1, 2)
        
        layout.addWidget(display_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        
        export_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['png', 'pdf', 'svg', 'jpg'])
        export_layout.addWidget(self.format_combo, 0, 1)
        
        export_layout.addWidget(QLabel("DPI:"), 0, 2)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        export_layout.addWidget(self.dpi_spin, 0, 3)
        
        layout.addWidget(export_group)
    
    def get_plot_config(self) -> PlotConfig:
        """Get plot configuration from UI controls"""
        
        return PlotConfig(
            time_unit=self.time_unit_combo.currentText(),
            color_palette=self.palette_combo.currentText(),
            show_confidence_intervals=self.show_ci_checkbox.isChecked(),
            show_zero_line=self.show_zero_checkbox.isChecked(),
            export_format=self.format_combo.currentText(),
            export_dpi=self.dpi_spin.value()
        )
    
    def get_selected_plot_types(self) -> List[str]:
        """Get selected plot types"""
        
        plot_types = []
        if self.timeseries_checkbox.isChecked():
            plot_types.append('time_series')
        if self.erp_checkbox.isChecked():
            plot_types.append('erp')
        if self.butterfly_checkbox.isChecked():
            plot_types.append('butterfly')
        if self.channel_array_checkbox.isChecked():
            plot_types.append('channel_array')
        
        return plot_types


class TimeDomainAnalysisWidget(QWidget):
    """Main widget for Phase 3 time-domain analysis"""
    
    analysis_complete = pyqtSignal(dict)  # Results dictionary
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw = None
        self.results = {}
        self.worker_thread = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Phase 3: Time-Domain Analysis & ERPs")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Main content in tabs
        self.tab_widget = QTabWidget()
        
        # Epoching tab
        self.epoching_controls = EpochingControlsWidget()
        self.tab_widget.addTab(self.epoching_controls, "📊 Epoching")
        
        # ERP Analysis tab
        self.erp_controls = ERPAnalysisWidget()
        self.tab_widget.addTab(self.erp_controls, "🧠 ERP Analysis")
        
        # Visualization tab
        self.viz_controls = VisualizationControlsWidget()
        self.tab_widget.addTab(self.viz_controls, "📈 Visualization")
        
        layout.addWidget(self.tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_analysis_btn = QPushButton("🚀 Run Time-Domain Analysis")
        self.run_analysis_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.run_analysis_btn)
        
        self.export_results_btn = QPushButton("💾 Export Results")
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_results_btn)
        
        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready for time-domain analysis")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setVisible(False)
        layout.addWidget(self.results_text)
    
    def set_data(self, raw: mne.io.Raw):
        """Set the EEG data for analysis"""
        self.raw = raw
        self.run_analysis_btn.setEnabled(True)
        self.status_label.setText(f"Data loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
    
    def run_analysis(self):
        """Run time-domain analysis"""
        if self.raw is None:
            QMessageBox.warning(self, "No Data", "Please load EEG data first.")
            return
        
        try:
            # Prepare analysis configuration
            analysis_config = {
                'create_epochs': True,
                'compute_erp': True,
                'create_plots': True,
                'use_events': self.epoching_controls.event_based_radio.isChecked(),
                'epoching_config': self.epoching_controls.get_epoching_config(),
                'segmentation_config': self.epoching_controls.get_segmentation_config(),
                'erp_config': self.erp_controls.get_erp_config(),
                'plot_config': self.viz_controls.get_plot_config(),
                'plot_types': self.viz_controls.get_selected_plot_types()
            }
            
            # Start worker thread
            self.worker_thread = TimeDomainWorkerThread(self.raw, analysis_config)
            self.worker_thread.progress_update.connect(self.progress_bar.setValue)
            self.worker_thread.status_update.connect(self.status_label.setText)
            self.worker_thread.analysis_complete.connect(self.on_analysis_complete)
            
            # Update UI
            self.run_analysis_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Start analysis
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to start analysis: {str(e)}")
            self.run_analysis_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def on_analysis_complete(self, success: bool, message: str, results: dict):
        """Handle analysis completion"""
        self.run_analysis_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.results = results
            self.export_results_btn.setEnabled(True)
            
            # Display results summary
            self.display_results_summary(results)
            self.results_text.setVisible(True)
            
            # Emit completion signal
            self.analysis_complete.emit(results)
            
            QMessageBox.information(self, "Analysis Complete", message)
        else:
            QMessageBox.critical(self, "Analysis Failed", message)
    
    def display_results_summary(self, results: dict):
        """Display analysis results summary"""
        summary = ["=== Time-Domain Analysis Results ===\n"]
        
        if 'epochs' in results:
            epochs = results['epochs']
            summary.append(f"📊 Epochs: {len(epochs)} epochs created")
            summary.append(f"   - Time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} s")
            summary.append(f"   - Channels: {len(epochs.ch_names)}")
            
            if 'epoching_metrics' in results and results['epoching_metrics']:
                metrics = results['epoching_metrics']
                summary.append(f"   - Rejection rate: {metrics.rejection_rate:.1%}")
                summary.append(f"   - Signal quality (SNR): {metrics.snr_estimate:.2f}")
        
        if 'erp' in results:
            erp = results['erp']
            summary.append(f"\n🧠 ERP Computed:")
            summary.append(f"   - Channels: {len(erp.ch_names)}")
            summary.append(f"   - Time points: {len(erp.times)}")
            
            if 'erp_statistics' in results:
                for condition, stats in results['erp_statistics'].items():
                    summary.append(f"   - {condition}: Mean amplitude = {stats.mean_amplitude*1e6:.2f} µV")
                    summary.append(f"     Reliability = {stats.reliability:.3f}")
        
        if 'peaks' in results:
            peaks = results['peaks']
            summary.append(f"\n🎯 Peak Detection: {len(peaks)} peaks found")
            for peak in peaks[:5]:  # Show first 5 peaks
                summary.append(f"   - {peak.component.value} at {peak.latency:.3f}s: {peak.amplitude*1e6:.1f}µV ({peak.channel})")
        
        if 'plots' in results:
            plots = results['plots']
            summary.append(f"\n📈 Visualizations: {len(plots)} plots created")
            for plot_name in plots.keys():
                summary.append(f"   - {plot_name}")
        
        self.results_text.setText('\n'.join(summary))
    
    def export_results(self):
        """Export analysis results"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        try:
            from pathlib import Path
            output_dir = Path("time_domain_results")
            output_dir.mkdir(exist_ok=True)
            
            # Export plots
            if 'plots' in self.results:
                for plot_name, fig in self.results['plots'].items():
                    fig.savefig(
                        output_dir / f"{plot_name}.{self.viz_controls.format_combo.currentText()}",
                        dpi=self.viz_controls.dpi_spin.value(),
                        bbox_inches='tight'
                    )
            
            # Export data summaries
            if 'epoching_metrics' in self.results and self.results['epoching_metrics']:
                metrics = self.results['epoching_metrics']
                with open(output_dir / "epoching_summary.txt", 'w') as f:
                    f.write(f"Epochs Summary\n")
                    f.write(f"==============\n")
                    f.write(f"Total epochs: {metrics.n_epochs_total}\n")
                    f.write(f"Good epochs: {metrics.n_epochs_good}\n")
                    f.write(f"Rejected epochs: {metrics.n_epochs_rejected}\n")
                    f.write(f"Rejection rate: {metrics.rejection_rate:.1%}\n")
                    f.write(f"SNR estimate: {metrics.snr_estimate:.2f}\n")
            
            QMessageBox.information(self, "Export Complete", f"Results exported to {output_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def get_analysis_results(self) -> dict:
        """Get the analysis results"""
        return self.results