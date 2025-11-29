#!/usr/bin/env python3
"""
Katharsis - EEG Artifact Cleaner GUI Application
=================================================

Katharsis is an application for automatic EEG artifact cleaning.
It uses Independent Component Analysis (ICA) or Principal Component
Analysis (PCA) techniques for detecting and removing artifacts from
eye blinks and other muscle movements.

Features:
- Graphical user interface with PyQt6
- Support for multiple EEG file formats (EDF, BDF, FIF, CSV, SET/EEGLAB)
- Automatic channel detection and selection
- ICA/PCA analysis with component visualization
- Before/after cleaning comparison
- Export cleaned data in multiple formats

Author: porfanid
Version: 3.3
License: MIT
"""

import sys
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplashScreen,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class BackendInitializationThread(QThread):
    """
    Thread for backend component initialization

    Loads the required libraries and initializes the EEG cleaning service
    in a separate thread to avoid blocking the GUI.

    Signals:
        progress_update (int): Progress update (0-100)
        status_update (str): Status update
        initialization_complete (object): Completion with the service
    """

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    initialization_complete = pyqtSignal(object)  # service only

    def __init__(self):
        """Initialize the thread"""
        super().__init__()

    def run(self):
        """
        Execute backend initialization

        Loads libraries and creates the EEG cleaning service.
        Shows progress updates on the splash screen.
        """
        try:
            self.status_update.emit("Loading libraries...")
            self.progress_update.emit(20)

            # Import heavy libraries
            from backend import EEGArtifactCleaningService

            self.progress_update.emit(50)

            self.status_update.emit("Initializing services...")
            # Initialize backend service
            service = EEGArtifactCleaningService()
            self.progress_update.emit(80)

            self.status_update.emit("Completing...")
            self.progress_update.emit(100)
            self.initialization_complete.emit(service)

        except Exception as e:
            self.status_update.emit(f"Initialization error: {str(e)}")


def create_splash_pixmap():
    """
    Creates the pixmap for the application splash screen

    Creates a graphical splash screen with the application title
    and subtitle on a blue background.

    Returns:
        QPixmap: The pixmap for the splash screen
    """
    pixmap = QPixmap(700, 400)
    pixmap.fill(QColor("#007AFF"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Background gradient effect
    painter.fillRect(pixmap.rect(), QColor("#007AFF"))

    # Title
    painter.setPen(QColor("white"))
    title_font = QFont("Arial", 24, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.drawText(
        pixmap.rect().adjusted(20, 50, -20, 0),
        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
        "🧠 Katharsis - EEG Artifact Cleaner Pro",
    )

    # Subtitle
    subtitle_font = QFont("Arial", 14)
    painter.setFont(subtitle_font)
    painter.drawText(
        pixmap.rect().adjusted(20, 120, -20, 0),
        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
        "Professional EEG Data Cleaning",
    )

    painter.end()
    return pixmap


class LoadingSplashScreen(QSplashScreen):
    """
    Custom splash screen with progress bar

    Shows loading progress with a progress bar
    and status messages during initialization.

    Attributes:
        progress (int): Current progress (0-100)
        status_text (str): Current status message
    """

    def __init__(self):
        """Initialize the splash screen"""
        pixmap = create_splash_pixmap()
        super().__init__(pixmap)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )

        # Progress bar
        self.progress = 0
        self.status_text = "Starting application..."

    def set_progress(self, value):
        """
        Set progress and repaint

        Args:
            value (int): New progress value (0-100)
        """
        self.progress = value
        self.repaint()

    def set_status(self, text):
        """
        Set status text and repaint

        Args:
            text (str): New status message
        """
        self.status_text = text
        self.repaint()

    def drawContents(self, painter):
        """
        Draw splash screen contents

        Draws the progress bar and status text on top of the base pixmap.

        Args:
            painter (QPainter): Painter for drawing
        """
        super().drawContents(painter)

        # Draw progress bar
        progress_rect = self.rect().adjusted(100, 280, -100, -80)
        painter.setPen(QColor("white"))
        painter.drawRect(progress_rect)

        # Fill progress
        if self.progress > 0:
            fill_width = int(progress_rect.width() * self.progress / 100)
            fill_rect = progress_rect.adjusted(
                2, 2, -progress_rect.width() + fill_width - 2, -2
            )
            painter.fillRect(fill_rect, QColor("#28a745"))

        # Draw progress text
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 12))
        text_rect = self.rect().adjusted(0, 320, 0, 0)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
            f"{self.status_text} ({self.progress}%)",
        )


# Threads for EEG data processing
class EEGProcessingThread(QThread):
    """
    Thread for processing EEG data in background

    Performs loading, filtering, ICA analysis and artifact detection
    without blocking the GUI.

    Signals:
        progress_update (int): Progress update (0-100)
        status_update (str): Status update
        processing_complete (bool, str): Processing completion (success, message)
        ica_ready (dict): ICA data ready for visualization
    """

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(bool, str)
    ica_ready = pyqtSignal(dict)

    def __init__(self, service, input_file, selected_channels=None):
        """
        Initialize the processing thread

        Args:
            service: EEG cleaning service
            input_file (str): Input file path
            selected_channels (List[str], optional): Selected channels
        """
        super().__init__()
        self.service = service
        self.input_file = input_file
        self.selected_channels = selected_channels

    def run(self):
        """
        Execute EEG data processing

        Loads the file, trains the ICA model, detects artifacts
        and prepares data for visualization.
        """
        try:
            self.status_update.emit("Loading and preparing file...")
            load_result = self.service.load_and_prepare_file(
                self.input_file, self.selected_channels
            )
            if not load_result["success"]:
                self.processing_complete.emit(
                    False,
                    f"Loading error: {load_result.get('error', 'Unknown error')}",
                )
                return
            self.progress_update.emit(30)

            # Use generic fit_analysis() which respects the set analysis method
            method_name = self.service.analysis_method
            self.status_update.emit(f"Training {method_name} model...")
            analysis_result = self.service.fit_analysis()
            if not analysis_result["success"]:
                self.processing_complete.emit(
                    False,
                    f"{method_name} error: {analysis_result.get('error', 'Unknown error')}",
                )
                return
            self.progress_update.emit(70)

            self.status_update.emit("Automatic artifact detection...")
            detection_result = self.service.detect_artifacts()
            if not detection_result["success"]:
                self.processing_complete.emit(
                    False,
                    f"Detection error: {detection_result.get('error', 'Unknown error')}",
                )
                return
            self.progress_update.emit(90)

            viz_data = self.service.get_component_visualization_data()
            if not viz_data:
                self.processing_complete.emit(
                    False, "Failed to create visualization data."
                )
                return
            self.ica_ready.emit(viz_data)
            self.progress_update.emit(100)
            self.processing_complete.emit(True, "Ready for selection.")
        except Exception as e:
            self.processing_complete.emit(False, f"Critical error: {str(e)}")


class CleaningThread(QThread):
    """
    Thread for artifact cleaning in background

    Applies removal of selected artifacts and saves
    the cleaned data.

    Signals:
        cleaning_complete (bool, str, dict): Cleaning completion
                                           (success, message, results)
    """

    cleaning_complete = pyqtSignal(bool, str, dict)

    def __init__(self, service, components, output_file):
        """
        Initialize the cleaning thread

        Args:
            service: EEG cleaning service
            components (List[int]): List of components to remove
            output_file (str): Output file path
        """
        super().__init__()
        self.service = service
        self.components_to_remove = components
        self.output_file = output_file

    def run(self):
        """
        Execute artifact cleaning

        Applies removal of selected components and saves
        the cleaned data to an EDF file.
        """
        try:
            clean_result = self.service.apply_artifact_removal(
                self.components_to_remove
            )
            if not clean_result["success"]:
                self.cleaning_complete.emit(
                    False, clean_result.get("error", "Unknown error"), {}
                )
                return
            cleaned_data = clean_result["cleaned_data"]
            if not self.service.save_cleaned_data(cleaned_data, self.output_file):
                self.cleaning_complete.emit(False, "File save error.", {})
                return

            # Get original data for comparison
            original_data = self.service.backend_core.get_filtered_data()

            results = {
                **clean_result,
                "input_file": self.service.current_file,
                "output_file": self.output_file,
                "original_data": original_data,
            }
            self.cleaning_complete.emit(True, "Cleaning completed!", results)
        except Exception as e:
            self.cleaning_complete.emit(False, f"Critical error: {str(e)}", {})


class EEGArtifactCleanerGUI(QMainWindow):
    """
    Main GUI class for the EEG artifact cleaning application

    Manages all application screens and user interaction:
    - Welcome screen for file selection
    - Channel selection for channel selection
    - ICA component selector for artifact selection and removal
    - Comparison screen for result comparison

    Attributes:
        service: Backend service for EEG cleaning
        ica_selector_screen: ICA component selection screen
        current_input_file (str): Current input file
        splash: Splash screen during startup
    """

    def __init__(self):
        """
        Initialize the main GUI application

        Creates the splash screen and starts backend initialization
        in a separate thread.
        """
        super().__init__()
        self.service = None
        self.ica_selector_screen = None
        self.current_input_file = ""

        # Show loading screen
        self.splash = LoadingSplashScreen()
        self.splash.show()

        # Initialize backend in separate thread
        self.init_thread = BackendInitializationThread()
        self.init_thread.progress_update.connect(self.splash.set_progress)
        self.init_thread.status_update.connect(self.splash.set_status)
        self.init_thread.initialization_complete.connect(
            self.on_initialization_complete
        )
        self.init_thread.start()

    def on_initialization_complete(self, service):
        """
        Called when backend initialization is complete

        Creates GUI elements and displays the main window.

        Args:
            service: The initialized EEG cleaning service
        """
        try:
            self.service = service

            # Create GUI components in main thread
            self.status_update_timer = QTimer()
            self.splash.set_status("Creating GUI elements...")
            self.splash.set_progress(90)

            # Import and create component selector in main thread
            from components import (
                ChannelSelectorWidget,
                ComparisonScreen,
                ICAComponentSelector,
            )

            theme = {
                "background": "#FFFFFF",
                "primary": "#007AFF",
                "primary_hover": "#0056b3",
                "success": "#28a745",
                "success_hover": "#218838",
                "danger": "#dc3545",
                "text": "#212529",
                "text_light": "#6c757d",
                "statusbar_bg": "#343a40",
                "statusbar_text": "#FFFFFF",
                "border": "#dee2e6",
            }
            self.channel_selector_screen = ChannelSelectorWidget(theme=theme)
            self.ica_selector_screen = ICAComponentSelector(theme=theme)
            self.comparison_screen = ComparisonScreen(theme=theme)

            # Setup UI now that components are ready
            self.setup_ui()
            self.setup_connections()

            # Hide splash and show main window
            QTimer.singleShot(500, self.finish_loading)  # Small delay to show 100%
        except Exception as e:
            print(f"GUI setup error: {str(e)}")
            # Fallback: show error message and exit gracefully
            QMessageBox.critical(
                None,
                "Initialization Error",
                f"Could not initialize the application:\n{str(e)}",
            )
            if hasattr(self, "splash"):
                self.splash.hide()
            sys.exit(1)

    def finish_loading(self):
        """
        Complete the loading process

        Hides the splash screen and displays the main window.
        """
        self.splash.hide()
        self.show()

    def setup_ui(self):
        """
        Create and layout GUI elements

        Creates the stacked widget for different screens and configures
        the general application style.
        """
        self.setWindowTitle("Katharsis - EEG Artifact Cleaner Pro")
        self.setGeometry(100, 100, 1100, 850)
        self.setMinimumSize(800, 600)

        # Theme is created here to be available to child widgets
        self.theme = {
            "background": "#FFFFFF",
            "primary": "#007AFF",
            "primary_hover": "#0056b3",
            "success": "#28a745",
            "success_hover": "#218838",
            "danger": "#dc3545",
            "text": "#212529",
            "text_light": "#6c757d",
            "statusbar_bg": "#343a40",
            "statusbar_text": "#FFFFFF",
            "border": "#dee2e6",
        }

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_screen = self.create_welcome_screen()

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.channel_selector_screen)
        self.stacked_widget.addWidget(self.ica_selector_screen)
        self.stacked_widget.addWidget(self.comparison_screen)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_welcome_screen(self):
        """
        Create the welcome screen

        Creates the initial screen with the application title and
        file selection button.

        Returns:
            QWidget: The welcome screen
        """
        screen = QWidget()
        layout = QVBoxLayout(screen)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(25)
        title = QLabel("🧠 Katharsis - EEG Artifact Cleaner")
        title.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        layout.addWidget(title, 0, Qt.AlignmentFlag.AlignCenter)
        self.select_input_btn = QPushButton("🔍 Select EEG File for Analysis")
        self.select_input_btn.setMinimumHeight(60)
        self.select_input_btn.setMinimumWidth(400)
        self.select_input_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(self.select_input_btn, 0, Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(400)
        layout.addWidget(self.progress_bar, 0, Qt.AlignmentFlag.AlignCenter)
        return screen

    def setup_connections(self):
        """
        Setup signal connections between widgets

        Connects signals from different screens to their corresponding methods
        for communication between components.
        """
        self.select_input_btn.clicked.connect(self.select_input_file)
        self.channel_selector_screen.channels_selected.connect(
            self.on_channels_selected
        )
        self.ica_selector_screen.components_selected.connect(self.apply_cleaning)
        self.comparison_screen.return_to_home.connect(self.reset_ui)

    def show_message_box(self, icon, title, text):
        """
        Helper function for displaying QMessageBox with proper style

        Creates and displays a message box with the application theme.

        Args:
            icon: The message box icon (QMessageBox.Icon)
            title (str): The window title
            text (str): The message text
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(text)
        msg_box.setWindowTitle(title)
        # Apply global stylesheet to messagebox before showing it
        msg_box.setStyleSheet(QApplication.instance().styleSheet())
        msg_box.exec()

    def select_input_file(self):
        """
        Select EEG file for processing

        Opens file dialog for EEG file selection (supports multiple formats)
        and navigates to the channel selection screen.

        Supported formats: EDF, BDF, FIF, CSV, SET (EEGLAB)
        """
        # Define supported formats for import
        file_filter = (
            "EEG Files (*.edf *.bdf *.fif *.csv *.set);;"
            "EDF Files (*.edf);;"
            "BDF Files (*.bdf);;"
            "FIF Files (*.fif);;"
            "CSV Files (*.csv);;"
            "EEGLAB Files (*.set);;"
            "All Files (*.*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File",
            str(Path.home()),
            file_filter,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if file_path:
            self.current_input_file = file_path
            # Go to channel selection instead of directly processing
            self.show_channel_selection()

    def show_channel_selection(self):
        """
        Display the channel selection screen

        Loads the selected file into the channel selection screen and
        navigates to that screen.
        """
        try:
            self.channel_selector_screen.set_edf_file(self.current_input_file)
            # Navigate to channel selection screen (index 1)
            self.stacked_widget.setCurrentIndex(1)
            self.status_bar.showMessage("Select channels for analysis")
        except Exception as e:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Unable to load file for channel selection:\n{str(e)}",
            )

    def on_channels_selected(self, selected_channels, analysis_method="ICA"):
        """
        Handle channel selection and start processing

        Stores the selected channels and analysis method,
        and starts data processing.

        Args:
            selected_channels (List[str]): List of selected channels
            analysis_method (str): Analysis method ("ICA" or "PCA")
        """
        self.selected_channels = selected_channels
        self.analysis_method = analysis_method
        # Set the analysis method in the service
        self.service.set_analysis_method(analysis_method)
        self.start_processing()

    def start_processing(self):
        """
        Start EEG data processing

        Creates and starts the processing thread for file loading,
        ICA analysis and artifact detection.
        """
        self.select_input_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        # Use selected channels if available
        channels_to_use = getattr(self, "selected_channels", None)

        self.processing_thread = EEGProcessingThread(
            self.service, self.current_input_file, channels_to_use
        )
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.status_update.connect(self.status_bar.showMessage)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.ica_ready.connect(self.on_ica_ready)
        self.processing_thread.start()

    def on_ica_ready(self, viz_data):
        """
        Handle ICA data readiness

        Loads visualization data into the component selection screen
        and navigates to that screen.

        Args:
            viz_data (dict): Data for ICA component visualization
        """
        self.ica_selector_screen.set_ica_data(**viz_data)
        # Navigate to ICA selector screen (index 2)
        self.stacked_widget.setCurrentIndex(2)

    def apply_cleaning(self, selected_components):
        """
        Apply artifact cleaning

        Asks the user to select an output file and starts
        cleaning of selected artifacts.

        Supported output formats: EDF, BDF, FIF, CSV, SET (EEGLAB)

        Args:
            selected_components (List[int]): List of components to remove
        """
        # Get the input file extension to suggest the same format by default
        input_ext = Path(self.current_input_file).suffix.lower()
        input_stem = Path(self.current_input_file).stem

        # Use the same format as input by default
        default_path = str(
            Path(self.current_input_file).parent / f"{input_stem}_clean{input_ext}"
        )

        # Define supported formats for export
        file_filter = (
            "EEG Files (*.edf *.bdf *.fif *.csv *.set);;"
            "EDF Files (*.edf);;"
            "BDF Files (*.bdf);;"
            "FIF Files (*.fif);;"
            "CSV Files (*.csv);;"
            "EEGLAB Files (*.set);;"
            "All Files (*.*)"
        )

        output_file, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Cleaned File",
            default_path,
            file_filter,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not output_file:
            self.status_bar.showMessage("Cleaning process cancelled.", 3000)
            return

        # Ensure the file has a valid extension
        output_path = Path(output_file)
        if not output_path.suffix:
            # Add default extension based on selected filter or input format
            output_file = str(output_path) + input_ext

        self.cleaning_thread = CleaningThread(
            self.service, selected_components, output_file
        )
        self.cleaning_thread.cleaning_complete.connect(self.on_cleaning_complete)
        self.cleaning_thread.start()
        self.status_bar.showMessage("Applying cleaning...")

    def on_processing_complete(self, success, message):
        """
        Handle processing completion

        Displays error message if processing failed and resets the UI.

        Args:
            success (bool): Whether processing was successful
            message (str): Status message
        """
        if not success:
            self.show_message_box(QMessageBox.Icon.Critical, "Error", message)
            self.reset_ui()

    def on_cleaning_complete(self, success, message, results):
        """
        Handle cleaning completion

        Displays the comparison screen if cleaning was successful,
        or an error message if it failed.

        Args:
            success (bool): Whether cleaning was successful
            message (str): Status message
            results (dict): Cleaning results for comparison
        """
        if success:
            # Navigate to comparison screen instead of showing QMessageBox
            try:
                self.comparison_screen.update_comparison(
                    original_data=results["original_data"],
                    cleaned_data=results["cleaned_data"],
                    original_stats=results["original_stats"],
                    cleaned_stats=results["cleaned_stats"],
                    components_removed=results["components_removed"],
                    input_file=results["input_file"],
                    output_file=results["output_file"],
                )
                # Navigate to comparison screen (index 3)
                self.stacked_widget.setCurrentIndex(3)
                self.status_bar.showMessage(
                    "Result comparison - Cleaning successful!"
                )
            except Exception as e:
                # Fallback to original message box if comparison screen fails
                full_message = f"{message}\n\nSaved to:\n{results['output_file']}\n\nComparison screen error: {str(e)}"
                self.show_message_box(
                    QMessageBox.Icon.Information, "Success", full_message
                )
                self.reset_ui()
        else:
            self.show_message_box(QMessageBox.Icon.Critical, "Error", message)
            self.reset_ui()

    def reset_ui(self):
        """
        Reset UI to initial state

        Returns to the welcome screen and resets the state
        of control elements.
        """
        self.stacked_widget.setCurrentIndex(0)
        self.select_input_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ready")


def get_global_stylesheet(theme):
    """
    Creates the central stylesheet for the entire application

    Creates a comprehensive CSS stylesheet that covers all widgets
    of the application, using the theme colors.

    Args:
        theme (dict): Dictionary with theme colors
                     (background, primary, success, text, etc.)

    Returns:
        str: The CSS stylesheet for the application
    """
    return f"""
        /* --- General Style --- */
        QWidget {{
            font-family: Arial;
            color: {theme['text']};
        }}
        QMainWindow, QDialog {{
            background-color: {theme['background']};
        }}
        /* ... (rest of general style remains the same) ... */
        QStatusBar {{
            background-color: {theme['statusbar_bg']};
            color: {theme['statusbar_text']};
            font-weight: bold;
        }}
        QPushButton {{
            font-weight: bold;
            border-radius: 8px;
            padding: 12px;
            color: white;
            background-color: {theme['primary']};
            border: none;
        }}
        QPushButton:hover {{
            background-color: {theme['primary_hover']};
        }}
        QProgressBar {{
            border: 1px solid {theme['border']};
            border-radius: 4px;
            background-color: #e9ecef;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {theme['primary']};
            border-radius: 4px;
        }}
        QScrollBar:vertical {{
            border: none;
            background: #e9ecef;
            width: 14px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: #bdc3c7;
            min-height: 25px;
            border-radius: 7px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: #95a5a6;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}

        /* --- Styling for File Selection Window --- */
        QFileDialog {{
            background-color: {theme['background']};
        }}
        QFileDialog QListView, 
        QFileDialog QTreeView {{
            background-color: white;
            border: 1px solid {theme['border']};
            border-radius: 4px;
        }}
        QFileDialog QTreeView::item:selected, 
        QFileDialog QListView::item:selected {{
            background-color: {theme['primary']};
            color: white;
        }}
        QFileDialog QHeaderView::section {{
            background-color: {theme['background']};
            padding: 5px;
            border-top: 0px;
            border-left: 0px;
            border-right: 1px solid {theme['border']};
            border-bottom: 2px solid {theme['border']};
            color: {theme['text']};
            font-weight: bold;
        }}
        QFileDialog QLineEdit, 
        QFileDialog QComboBox {{
            padding: 8px;
            border: 1px solid {theme['border']};
            border-radius: 4px;
            background-color: white;
        }}
        QFileDialog QPushButton {{
            min-width: 80px;
        }}
        QFileDialog QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            padding: 5px;
            border-radius: 4px;
        }}
        QFileDialog QToolButton:hover, QFileDialog QToolButton:pressed {{
            background-color: #e9ecef;
            border: 1px solid {theme['border']};
        }}
        QFileDialog QToolButton:checked {{
            background-color: #d4e6f1;
            border: 1px solid {theme['primary']};
        }}

        /* --- *** Η ΤΕΛΙΚΗ ΠΡΟΣΘΗΚΗ ΕΙΝΑΙ ΕΔΩ *** --- */
        /* Στοχεύουμε ΜΟΝΟ τα ToolButtons που είναι παιδιά ενός HeaderView */
        QHeaderView QToolButton {{
            background-color: #e9ecef;
            border: 1px solid {theme['border']};
            padding: 4px;
            margin: 2px;
        }}
        QHeaderView QToolButton:hover, QHeaderView QToolButton:pressed {{
            background-color: #d4e6f1;
            border-color: {theme['primary']};
        }}
        
        /* Styling για το μενού που ανοίγει */
        QMenu {{
            background-color: white;
            border: 1px solid {theme['border']};
            padding: 5px;
        }}
        QMenu::item {{
            padding: 8px 20px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background-color: {theme['primary']};
            color: white;
        }}
        
        /* Styling για τα παράθυρα διαλόγου */
        QMessageBox {{
             background-color: {theme['background']};
        }}
        QMessageBox QLabel {{
            color: {theme['text']};
        }}
    """


def main():
    """
    Main application startup function

    Initializes the PyQt6 application, applies the global stylesheet
    and starts the main application window.
    """
    app = QApplication(sys.argv)

    # Default theme for initial styling
    default_theme = {
        "background": "#FFFFFF",
        "primary": "#007AFF",
        "primary_hover": "#0056b3",
        "success": "#28a745",
        "success_hover": "#218838",
        "danger": "#dc3545",
        "text": "#212529",
        "text_light": "#6c757d",
        "statusbar_bg": "#343a40",
        "statusbar_text": "#FFFFFF",
        "border": "#dee2e6",
    }

    # Apply style to entire application
    app.setStyleSheet(get_global_stylesheet(default_theme))

    window = EEGArtifactCleanerGUI()
    # Don't show window immediately - it will be shown after loading completes
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
