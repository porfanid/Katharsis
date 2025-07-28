#!/usr/bin/env python3
"""
EEG Artifact Cleaner - v3.2 - Loading Screen Implementation
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QStatusBar,
    QStackedWidget, QMessageBox, QSplashScreen
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor
from PyQt6.QtCore import QSize

class BackendInitializationThread(QThread):
    """Thread για φόρτωση των heavy backend components"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    initialization_complete = pyqtSignal(object)  # service only
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        try:
            self.status_update.emit("Φόρτωση βιβλιοθηκών...")
            self.progress_update.emit(20)
            
            # Import heavy libraries
            from backend import EEGArtifactCleaningService  
            self.progress_update.emit(50)
            
            self.status_update.emit("Αρχικοποίηση υπηρεσιών...")
            # Initialize backend service
            service = EEGArtifactCleaningService()
            self.progress_update.emit(80)
            
            self.status_update.emit("Ολοκλήρωση...")
            self.progress_update.emit(100)
            self.initialization_complete.emit(service)
            
        except Exception as e:
            self.status_update.emit(f"Σφάλμα αρχικοποίησης: {str(e)}")

def create_splash_pixmap():
    """Δημιουργεί ένα pixmap για το splash screen"""
    pixmap = QPixmap(500, 300)
    pixmap.fill(QColor("#007AFF"))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Background gradient effect
    painter.fillRect(pixmap.rect(), QColor("#007AFF"))
    
    # Title
    painter.setPen(QColor("white"))
    title_font = QFont("Arial", 24, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop, 
                    "🧠 EEG Artifact Cleaner Pro")
    
    # Subtitle
    subtitle_font = QFont("Arial", 12)
    painter.setFont(subtitle_font)
    painter.drawText(pixmap.rect().adjusted(0, 80, 0, 0), 
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                    "Επαγγελματικός Καθαρισμός EEG Δεδομένων")
    
    painter.end()
    return pixmap

class LoadingSplashScreen(QSplashScreen):
    """Προσαρμοσμένο splash screen με progress bar"""
    
    def __init__(self):
        pixmap = create_splash_pixmap()
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
        # Progress bar
        self.progress = 0
        self.status_text = "Εκκίνηση εφαρμογής..."
        
    def set_progress(self, value):
        self.progress = value
        self.repaint()
        
    def set_status(self, text):
        self.status_text = text
        self.repaint()
        
    def drawContents(self, painter):
        super().drawContents(painter)
        
        # Draw progress bar
        progress_rect = self.rect().adjusted(50, 200, -50, -80)
        painter.setPen(QColor("white"))
        painter.drawRect(progress_rect)
        
        # Fill progress
        if self.progress > 0:
            fill_width = int(progress_rect.width() * self.progress / 100)
            fill_rect = progress_rect.adjusted(2, 2, -progress_rect.width() + fill_width - 2, -2)
            painter.fillRect(fill_rect, QColor("#28a745"))
        
        # Draw progress text
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 10))
        text_rect = self.rect().adjusted(0, 230, 0, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                        f"{self.status_text} ({self.progress}%)")

# Οι κλάσεις EEGProcessingThread και CleaningThread παραμένουν ίδιες
class EEGProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(bool, str)
    ica_ready = pyqtSignal(dict)
    def __init__(self, service, input_file):
        super().__init__()
        self.service = service
        self.input_file = input_file
    def run(self):
        try:
            self.status_update.emit("Φόρτωση και προετοιμασία αρχείου...")
            load_result = self.service.load_and_prepare_file(self.input_file)
            if not load_result['success']:
                self.processing_complete.emit(False, f"Σφάλμα φόρτωσης: {load_result.get('error', 'Άγνωστο σφάλμα')}")
                return
            self.progress_update.emit(30)
            self.status_update.emit("Εκπαίδευση μοντέλου ICA...")
            ica_result = self.service.fit_ica_analysis()
            if not ica_result['success']:
                self.processing_complete.emit(False, f"Σφάλμα ICA: {ica_result.get('error', 'Άγνωστο σφάλμα')}")
                return
            self.progress_update.emit(70)
            self.status_update.emit("Αυτόματος εντοπισμός artifacts...")
            detection_result = self.service.detect_artifacts()
            if not detection_result['success']:
                self.processing_complete.emit(False, f"Σφάλμα εντοπισμού: {detection_result.get('error', 'Άγνωστο σφάλμα')}")
                return
            self.progress_update.emit(90)
            viz_data = self.service.get_component_visualization_data()
            if not viz_data:
                self.processing_complete.emit(False, "Αποτυχία δημιουργίας δεδομένων οπτικοποίησης.")
                return
            self.ica_ready.emit(viz_data)
            self.progress_update.emit(100)
            self.processing_complete.emit(True, "Έτοιμο για επιλογή.")
        except Exception as e:
            self.processing_complete.emit(False, f"Κρίσιμο σφάλμα: {str(e)}")

class CleaningThread(QThread):
    cleaning_complete = pyqtSignal(bool, str, dict)
    def __init__(self, service, components, output_file):
        super().__init__()
        self.service = service
        self.components_to_remove = components
        self.output_file = output_file
    def run(self):
        try:
            clean_result = self.service.apply_artifact_removal(self.components_to_remove)
            if not clean_result['success']:
                self.cleaning_complete.emit(False, clean_result.get('error', 'Άγνωστο σφάλμα'), {})
                return
            cleaned_data = clean_result['cleaned_data']
            if not self.service.save_cleaned_data(cleaned_data, self.output_file):
                self.cleaning_complete.emit(False, "Σφάλμα αποθήκευσης αρχείου.", {})
                return
            
            # Get original data for comparison
            original_data = self.service.backend_core.get_filtered_data()
            
            results = {
                **clean_result, 
                'input_file': self.service.current_file, 
                'output_file': self.output_file,
                'original_data': original_data
            }
            self.cleaning_complete.emit(True, "Ο καθαρισμός ολοκληρώθηκε!", results)
        except Exception as e:
            self.cleaning_complete.emit(False, f"Κρίσιμο σφάλμα: {str(e)}", {})


class EEGArtifactCleanerGUI(QMainWindow):
    def __init__(self):
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
        self.init_thread.initialization_complete.connect(self.on_initialization_complete)
        self.init_thread.start()

    def on_initialization_complete(self, service):
        """Called when backend initialization is complete"""
        try:
            self.service = service
            
            # Create GUI components in main thread
            self.status_update_timer = QTimer()
            self.splash.set_status("Δημιουργία στοιχείων GUI...")
            self.splash.set_progress(90)
            
            # Import and create component selector in main thread
            from components import ICAComponentSelector, ComparisonScreen
            theme = {
                "background": "#FFFFFF", "primary": "#007AFF", "primary_hover": "#0056b3",
                "success": "#28a745", "success_hover": "#218838", "danger": "#dc3545",
                "text": "#212529", "text_light": "#6c757d", "statusbar_bg": "#343a40",
                "statusbar_text": "#FFFFFF", "border": "#dee2e6"
            }
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
            QMessageBox.critical(None, "Σφάλμα Αρχικοποίησης", 
                               f"Δεν ήταν δυνατή η αρχικοποίηση της εφαρμογής:\n{str(e)}")
            if hasattr(self, 'splash'):
                self.splash.hide()
            sys.exit(1)
        
    def finish_loading(self):
        self.splash.hide()
        self.show()

    def setup_ui(self):
        self.setWindowTitle("EEG Artifact Cleaner Pro")
        self.setGeometry(100, 100, 1100, 850)
        self.setMinimumSize(800, 600)

        # Η δημιουργία του theme μεταφέρεται εδώ για να είναι διαθέσιμο στα child widgets
        self.theme = {
            "background": "#FFFFFF", "primary": "#007AFF", "primary_hover": "#0056b3",
            "success": "#28a745", "success_hover": "#218838", "danger": "#dc3545",
            "text": "#212529", "text_light": "#6c757d", "statusbar_bg": "#343a40",
            "statusbar_text": "#FFFFFF", "border": "#dee2e6"
        }

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_screen = self.create_welcome_screen()

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.ica_selector_screen)
        self.stacked_widget.addWidget(self.comparison_screen)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Έτοιμο")
        
    def create_welcome_screen(self):
        # ... (Η συνάρτηση παραμένει ίδια με πριν)
        screen = QWidget()
        layout = QVBoxLayout(screen)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(25)
        title = QLabel("🧠 EEG Artifact Cleaner")
        title.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        layout.addWidget(title, 0, Qt.AlignmentFlag.AlignCenter)
        self.select_input_btn = QPushButton("🔍 Επιλογή Αρχείου EDF για Ανάλυση")
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
        # ... (Η συνάρτηση παραμένει ίδια με πριν)
        self.select_input_btn.clicked.connect(self.select_input_file)
        self.ica_selector_screen.components_selected.connect(self.apply_cleaning)
        self.comparison_screen.return_to_home.connect(self.reset_ui)
        
    def show_message_box(self, icon, title, text):
        """Βοηθητική συνάρτηση για εμφάνιση QMessageBox με το σωστό στυλ."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(text)
        msg_box.setWindowTitle(title)
        # Εφαρμόζουμε το global stylesheet στο messagebox πριν το δείξουμε
        msg_box.setStyleSheet(QApplication.instance().styleSheet())
        msg_box.exec()

    # --- Οι υπόλοιπες συναρτήσεις παραμένουν ίδιες με πριν ---
    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Επιλογή EDF", str(Path.home()), "*.edf", options=QFileDialog.Option.DontUseNativeDialog)
        if file_path:
            self.current_input_file = file_path
            self.start_processing()
    def start_processing(self):
        self.select_input_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.processing_thread = EEGProcessingThread(self.service, self.current_input_file)
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.status_update.connect(self.status_bar.showMessage)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.ica_ready.connect(self.on_ica_ready)
        self.processing_thread.start()
    def on_ica_ready(self, viz_data):
        self.ica_selector_screen.set_ica_data(**viz_data)
        self.stacked_widget.setCurrentIndex(1)
    def apply_cleaning(self, selected_components):
        default_path = self.current_input_file.replace('.edf', '_clean.edf')
        output_file, _ = QFileDialog.getSaveFileName(self, "Αποθήκευση Καθαρού Αρχείου", default_path, "*.edf", options=QFileDialog.Option.DontUseNativeDialog)
        if not output_file:
            self.status_bar.showMessage("Η διαδικασία καθαρισμού ακυρώθηκε.", 3000)
            return
        self.cleaning_thread = CleaningThread(self.service, selected_components, output_file)
        self.cleaning_thread.cleaning_complete.connect(self.on_cleaning_complete)
        self.cleaning_thread.start()
        self.status_bar.showMessage("Εφαρμογή καθαρισμού...")
    def on_processing_complete(self, success, message):
        if not success:
            self.show_message_box(QMessageBox.Icon.Critical, "Σφάλμα", message)
            self.reset_ui()
    def on_cleaning_complete(self, success, message, results):
        if success:
            # Navigate to comparison screen instead of showing QMessageBox
            try:
                self.comparison_screen.update_comparison(
                    original_data=results['original_data'],
                    cleaned_data=results['cleaned_data'],
                    original_stats=results['original_stats'],
                    cleaned_stats=results['cleaned_stats'],
                    components_removed=results['components_removed'],
                    input_file=results['input_file'],
                    output_file=results['output_file']
                )
                # Navigate to comparison screen (index 2)
                self.stacked_widget.setCurrentIndex(2)
                self.status_bar.showMessage("Σύγκριση αποτελεσμάτων - Επιτυχής καθαρισμός!")
            except Exception as e:
                # Fallback to original message box if comparison screen fails
                full_message = f"{message}\n\nΑποθηκεύτηκε στο:\n{results['output_file']}\n\nΣφάλμα οθόνης σύγκρισης: {str(e)}"
                self.show_message_box(QMessageBox.Icon.Information, "Επιτυχία", full_message)
                self.reset_ui()
        else:
            self.show_message_box(QMessageBox.Icon.Critical, "Σφάλμα", message)
            self.reset_ui()
    def reset_ui(self):
        self.stacked_widget.setCurrentIndex(0)
        self.select_input_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Έτοιμο")

def get_global_stylesheet(theme):
    """Δημιουργεί το κεντρικό, ολοκληρωμένο stylesheet για ΟΛΗ την εφαρμογή."""
    return f"""
        /* --- Γενικό Στυλ --- */
        QWidget {{
            font-family: Arial;
            color: {theme['text']};
        }}
        QMainWindow, QDialog {{
            background-color: {theme['background']};
        }}
        /* ... (το υπόλοιπο γενικό στυλ παραμένει ίδιο) ... */
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

        /* --- Styling για το Παράθυρο Επιλογής Αρχείου --- */
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
    app = QApplication(sys.argv)
    
    # Default theme for initial styling
    default_theme = {
        "background": "#FFFFFF", "primary": "#007AFF", "primary_hover": "#0056b3",
        "success": "#28a745", "success_hover": "#218838", "danger": "#dc3545",
        "text": "#212529", "text_light": "#6c757d", "statusbar_bg": "#343a40",
        "statusbar_text": "#FFFFFF", "border": "#dee2e6"
    }
    
    # Εφαρμόζουμε το στυλ σε ολόκληρη την εφαρμογή
    app.setStyleSheet(get_global_stylesheet(default_theme))
    
    window = EEGArtifactCleanerGUI()
    # Don't show window immediately - it will be shown after loading completes
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
