#!/usr/bin/env python3
"""
Katharsis Main Window - Clean PyQt6 Frontend
==========================================

Pure UI component that interacts only with the KatharsisBackend API.
Contains no business logic - all processing is delegated to the backend.

Author: porfanid
Version: 4.0 - Complete Frontend/Backend Separation
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QPushButton, QProgressBar, QMessageBox, QStatusBar,
    QSplashScreen, QApplication
)

from backend import KatharsisBackend


class ProgressWindow(QWidget):
    """Progress window for long-running operations"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Katharsis - Επεξεργασία")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        self.title_label = QLabel("Επεξεργασία δεδομένων...")
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(self.title_label)
        
        # Status
        self.status_label = QLabel("Προετοιμασία...")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #34495e;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 6px;
                margin: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
    
    def update_progress(self, percentage: int):
        """Update progress bar"""
        self.progress_bar.setValue(percentage)
    
    def update_status(self, status: str):
        """Update status text"""
        self.status_label.setText(status)


class BackendWorkerThread(QThread):
    """Worker thread for backend operations"""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    operation_complete = pyqtSignal(dict)
    
    def __init__(self, backend: KatharsisBackend, operation: str, **kwargs):
        super().__init__()
        self.backend = backend
        self.operation = operation
        self.kwargs = kwargs
        
        # Set up backend callbacks
        self.backend.set_callbacks(
            progress_callback=self.progress_update.emit,
            status_callback=self.status_update.emit,
            error_callback=self.error_occurred.emit
        )
    
    def run(self):
        """Execute the backend operation"""
        try:
            if self.operation == "load_file":
                result = self.backend.load_file(
                    self.kwargs["file_path"],
                    self.kwargs.get("selected_channels")
                )
            elif self.operation == "preprocessing":
                result = self.backend.apply_preprocessing(self.kwargs["config"])
            elif self.operation == "ica_analysis":
                result = self.backend.perform_ica_analysis(
                    self.kwargs.get("ica_method", "fastica"),
                    self.kwargs.get("n_components"),
                    self.kwargs.get("max_iter", 200)
                )
            elif self.operation == "ica_cleaning":
                result = self.backend.apply_ica_cleaning(self.kwargs["components_to_remove"])
            elif self.operation == "epoching":
                result = self.backend.perform_epoching(
                    self.kwargs["events_config"],
                    self.kwargs["epoch_config"]
                )
            elif self.operation == "erp_analysis":
                result = self.backend.perform_erp_analysis(self.kwargs["erp_config"])
            elif self.operation == "export_data":
                result = self.backend.export_data(
                    self.kwargs["output_path"],
                    self.kwargs.get("data_type", "cleaned")
                )
            else:
                result = {"success": False, "error": f"Unknown operation: {self.operation}"}
            
            self.operation_complete.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(f"Σφάλμα εκτέλεσης: {str(e)}")
            self.operation_complete.emit({"success": False, "error": str(e)})


class KatharsisMainWindow(QMainWindow):
    """
    Main application window - Pure PyQt6 frontend
    
    This window contains only UI logic and delegates all business logic
    to the KatharsisBackend. It provides a clean interface for:
    - File selection and validation
    - Channel selection
    - Preprocessing configuration
    - Analysis type selection (ICA vs Time-domain)
    - Results visualization
    - Data export
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize backend
        self.backend = KatharsisBackend()
        
        # UI state
        self.current_step = 0
        self.processing_thread: Optional[BackendWorkerThread] = None
        self.progress_window: Optional[ProgressWindow] = None
        
        # Initialize UI
        self.setup_ui()
        self.setup_style()
        self.show_step(0)  # Start with file selection
    
    def setup_ui(self):
        """Set up the main UI structure"""
        self.setWindowTitle("Katharsis - EEG Artifact Cleaner v4.0")
        self.setMinimumSize(1200, 800)
        
        # Central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        self.setup_header(main_layout)
        
        # Stacked widget for different steps
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Import and add UI widgets
        from .file_selection_widget import FileSelectionWidget
        from .channel_selection_widget import ChannelSelectionWidget
        from .preprocessing_widget import PreprocessingWidget
        from .analysis_selection_widget import AnalysisSelectionWidget
        from .ica_analysis_widget import ICAAnalysisWidget
        from .time_domain_widget import TimeDomainWidget
        from .results_widget import ResultsWidget
        
        # Add widgets to stack
        self.file_widget = FileSelectionWidget(self.backend)
        self.channel_widget = ChannelSelectionWidget(self.backend)
        self.preprocessing_widget = PreprocessingWidget(self.backend)
        self.analysis_selection_widget = AnalysisSelectionWidget(self.backend)
        self.ica_widget = ICAAnalysisWidget(self.backend)
        self.time_domain_widget = TimeDomainWidget(self.backend)
        self.results_widget = ResultsWidget(self.backend)
        
        self.stacked_widget.addWidget(self.file_widget)         # 0
        self.stacked_widget.addWidget(self.channel_widget)      # 1
        self.stacked_widget.addWidget(self.preprocessing_widget) # 2
        self.stacked_widget.addWidget(self.analysis_selection_widget) # 3
        self.stacked_widget.addWidget(self.ica_widget)          # 4
        self.stacked_widget.addWidget(self.time_domain_widget)  # 5
        self.stacked_widget.addWidget(self.results_widget)      # 6
        
        # Connect signals
        self.connect_signals()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Έτοιμο - Επιλέξτε αρχείο EEG")
        
        # Navigation buttons
        self.setup_navigation(main_layout)
    
    def setup_header(self, layout):
        """Set up the header section"""
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-bottom: 2px solid #2c3e50;
            }
        """)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 10, 20, 10)
        
        # Title
        title = QLabel("Katharsis EEG Artifact Cleaner")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white; border: none;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Version
        version = QLabel("v4.0")
        version.setFont(QFont("Arial", 12))
        version.setStyleSheet("color: #ecf0f1; border: none;")
        header_layout.addWidget(version)
        
        layout.addWidget(header)
    
    def setup_navigation(self, layout):
        """Set up navigation buttons"""
        nav_widget = QWidget()
        nav_widget.setFixedHeight(60)
        nav_widget.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                border-top: 1px solid #bdc3c7;
            }
        """)
        
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(20, 10, 20, 10)
        nav_layout.setSpacing(10)
        
        # Back button
        self.back_button = QPushButton("← Πίσω")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.clicked.connect(self.go_back)
        
        # Next button
        self.next_button = QPushButton("Επόμενο →")
        self.next_button.setFont(QFont("Arial", 12))
        self.next_button.clicked.connect(self.go_next)
        
        # Reset button
        self.reset_button = QPushButton("Επαναφορά")
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.clicked.connect(self.reset_application)
        
        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.reset_button)
        nav_layout.addWidget(self.next_button)
        
        layout.addWidget(nav_widget)
    
    def setup_style(self):
        """Apply consistent styling across the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                min-width: 120px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #21618c);
            }
            
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
            
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
            
            QStatusBar {
                background-color: #34495e;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border: none;
            }
        """)
    
    def connect_signals(self):
        """Connect signals between widgets"""
        # File selection completed
        self.file_widget.file_selected.connect(self.on_file_selected)
        
        # Channel selection completed  
        self.channel_widget.channels_selected.connect(self.on_channels_selected)
        
        # Preprocessing completed
        self.preprocessing_widget.preprocessing_completed.connect(self.on_preprocessing_completed)
        
        # Analysis type selected
        self.analysis_selection_widget.analysis_selected.connect(self.on_analysis_selected)
        
        # ICA analysis completed
        self.ica_widget.ica_completed.connect(self.on_ica_completed)
        
        # Time-domain analysis completed
        self.time_domain_widget.analysis_completed.connect(self.on_time_domain_completed)
    
    def show_step(self, step: int):
        """Show a specific step in the workflow"""
        self.current_step = step
        self.stacked_widget.setCurrentIndex(step)
        
        # Update navigation buttons
        self.back_button.setEnabled(step > 0)
        
        # Update next button based on step
        if step == 0:  # File selection
            self.next_button.setText("Επόμενο →")
            self.next_button.setEnabled(self.backend.get_processing_state()["file_loaded"])
        elif step == 1:  # Channel selection
            self.next_button.setText("Επόμενο →")
            self.next_button.setEnabled(len(self.channel_widget.get_selected_channels()) > 0)
        elif step == 2:  # Preprocessing
            self.next_button.setText("Επόμενο →")
            self.next_button.setEnabled(True)
        elif step == 3:  # Analysis selection
            self.next_button.setText("Επόμενο →")
            self.next_button.setEnabled(True)
        elif step in [4, 5]:  # ICA or Time-domain analysis
            self.next_button.setText("Ανάλυση")
            self.next_button.setEnabled(True)
        elif step == 6:  # Results
            self.next_button.setText("Εξαγωγή")
            self.next_button.setEnabled(True)
        
        # Update status bar
        step_names = [
            "Επιλογή αρχείου EEG",
            "Επιλογή καναλιών", 
            "Προεπεξεργασία",
            "Επιλογή ανάλυσης",
            "ICA Ανάλυση",
            "Ανάλυση χρονικού πεδίου",
            "Αποτελέσματα"
        ]
        self.status_bar.showMessage(f"Βήμα {step + 1}/7: {step_names[step]}")
    
    def go_back(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
    
    def go_next(self):
        """Go to next step or perform action"""
        if self.current_step == 0:  # File selection
            if self.backend.get_processing_state()["file_loaded"]:
                self.show_step(1)
        elif self.current_step == 1:  # Channel selection
            selected_channels = self.channel_widget.get_selected_channels()
            if selected_channels:
                # Apply channel selection to backend
                self.apply_channel_selection(selected_channels)
                self.show_step(2)
        elif self.current_step == 2:  # Preprocessing
            # Get preprocessing config and apply
            config = self.preprocessing_widget.get_config()
            self.apply_preprocessing(config)
        elif self.current_step == 3:  # Analysis selection
            analysis_type = self.analysis_selection_widget.get_selected_analysis()
            if analysis_type == "ica":
                self.show_step(4)
            elif analysis_type == "time_domain":
                self.show_step(5)
        elif self.current_step == 4:  # ICA analysis
            # Start ICA analysis
            config = self.ica_widget.get_config()
            self.perform_ica_analysis(config)
        elif self.current_step == 5:  # Time-domain analysis
            # Start time-domain analysis
            config = self.time_domain_widget.get_config()
            self.perform_time_domain_analysis(config)
        elif self.current_step == 6:  # Results - Export
            self.export_results()
    
    def reset_application(self):
        """Reset the entire application"""
        reply = QMessageBox.question(
            self, "Επαναφορά", 
            "Είστε βέβαιοι ότι θέλετε να επαναφέρετε την εφαρμογή;",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.backend.reset_all()
            self.show_step(0)
            
            # Reset all widgets
            self.file_widget.reset()
            self.channel_widget.reset()
            self.preprocessing_widget.reset()
            self.analysis_selection_widget.reset()
            self.ica_widget.reset()
            self.time_domain_widget.reset()
            self.results_widget.reset()
    
    # === Backend Operation Methods ===
    
    def apply_channel_selection(self, selected_channels: List[str]):
        """Apply channel selection to backend data"""
        # This would typically involve reloading with selected channels
        # For now, we just store the selection
        pass
    
    def apply_preprocessing(self, config: Dict[str, Any]):
        """Apply preprocessing with progress tracking"""
        self.run_backend_operation("preprocessing", config=config)
    
    def perform_ica_analysis(self, config: Dict[str, Any]):
        """Perform ICA analysis with progress tracking"""
        self.run_backend_operation("ica_analysis", **config)
    
    def perform_time_domain_analysis(self, config: Dict[str, Any]):
        """Perform time-domain analysis with progress tracking"""
        # First epoching
        self.run_backend_operation("epoching", 
                                 events_config=config["events"],
                                 epoch_config=config["epochs"])
    
    def export_results(self):
        """Export analysis results"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Αποθήκευση αποτελεσμάτων",
            "", "EDF Files (*.edf);;All Files (*)"
        )
        
        if file_path:
            self.run_backend_operation("export_data", 
                                     output_path=file_path,
                                     data_type="cleaned")
    
    def run_backend_operation(self, operation: str, **kwargs):
        """Run a backend operation with progress tracking"""
        # Show progress window
        self.progress_window = ProgressWindow()
        self.progress_window.show()
        
        # Create and start worker thread
        self.processing_thread = BackendWorkerThread(self.backend, operation, **kwargs)
        self.processing_thread.progress_update.connect(self.progress_window.update_progress)
        self.processing_thread.status_update.connect(self.progress_window.update_status)
        self.processing_thread.error_occurred.connect(self.on_backend_error)
        self.processing_thread.operation_complete.connect(self.on_backend_operation_complete)
        self.processing_thread.start()
    
    # === Signal Handlers ===
    
    def on_file_selected(self, file_path: str):
        """Handle file selection"""
        self.status_bar.showMessage(f"Αρχείο επιλέχθηκε: {file_path}")
        self.next_button.setEnabled(True)
    
    def on_channels_selected(self, channels: List[str]):
        """Handle channel selection"""
        self.status_bar.showMessage(f"Επιλέχθηκαν {len(channels)} κανάλια")
        self.next_button.setEnabled(len(channels) > 0)
    
    def on_preprocessing_completed(self, result: Dict[str, Any]):
        """Handle preprocessing completion"""
        if result["success"]:
            self.show_step(3)  # Go to analysis selection
        else:
            QMessageBox.warning(self, "Σφάλμα", f"Προεπεξεργασία απέτυχε: {result['error']}")
    
    def on_analysis_selected(self, analysis_type: str):
        """Handle analysis type selection"""
        self.next_button.setEnabled(True)
    
    def on_ica_completed(self, result: Dict[str, Any]):
        """Handle ICA analysis completion"""
        if result["success"]:
            self.show_step(6)  # Go to results
        else:
            QMessageBox.warning(self, "Σφάλμα", f"ICA ανάλυση απέτυχε: {result['error']}")
    
    def on_time_domain_completed(self, result: Dict[str, Any]):
        """Handle time-domain analysis completion"""
        if result["success"]:
            self.show_step(6)  # Go to results
        else:
            QMessageBox.warning(self, "Σφάλμα", f"Ανάλυση χρονικού πεδίου απέτυχε: {result['error']}")
    
    def on_backend_error(self, error: str):
        """Handle backend errors"""
        if self.progress_window:
            self.progress_window.hide()
        QMessageBox.critical(self, "Σφάλμα", error)
    
    def on_backend_operation_complete(self, result: Dict[str, Any]):
        """Handle backend operation completion"""
        if self.progress_window:
            self.progress_window.hide()
        
        if result["success"]:
            # Move to next step based on current operation
            if self.current_step == 2:  # Preprocessing completed
                self.show_step(3)
            elif self.current_step == 4:  # ICA completed
                self.show_step(6)
            elif self.current_step == 5:  # Time-domain completed
                self.show_step(6)
            elif self.current_step == 6:  # Export completed
                QMessageBox.information(self, "Επιτυχία", "Τα δεδομένα εξήχθησαν επιτυχώς!")
        else:
            QMessageBox.warning(self, "Σφάλμα", result.get("error", "Άγνωστο σφάλμα"))
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "Έξοδος", 
                "Υπάρχει επεξεργασία σε εξέλιξη. Είστε βέβαιοι ότι θέλετε να κλείσετε;",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.processing_thread.terminate()
                self.processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()