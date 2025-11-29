#!/usr/bin/env python3
"""
Katharsis - EEG Artifact Cleaner GUI Application
=====================================

Το Katharsis είναι μια εφαρμογή για τον αυτόματο καθαρισμό artifacts από δεδομένα EEG.
Χρησιμοποιεί τεχνικές Independent Component Analysis (ICA) για τον εντοπισμό και την
αφαίρεση artifacts που προέρχονται από βλεφαρισμούς και άλλες μυικές κινήσεις.

Χαρακτηριστικά:
- Γραφικό περιβάλλον χρήστη με PyQt6
- Υποστήριξη αρχείων EDF από συσκευές EEG
- Αυτόματος εντοπισμός και επιλογή καναλιών
- ICA ανάλυση με οπτικοποίηση συνιστωσών
- Σύγκριση πριν/μετά τον καθαρισμό
- Εξαγωγή καθαρών δεδομένων

Author: porfanid
Version: 3.2
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
    Thread για αρχικοποίηση των backend components

    Φορτώνει τις απαιτούμενες βιβλιοθήκες και αρχικοποιεί την υπηρεσία
    καθαρισμού EEG σε ξεχωριστό thread για να μην μπλοκάρει το GUI.

    Signals:
        progress_update (int): Ενημέρωση προόδου (0-100)
        status_update (str): Ενημέρωση κατάστασης
        initialization_complete (object): Ολοκλήρωση με την υπηρεσία
    """

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    initialization_complete = pyqtSignal(object)  # service only

    def __init__(self):
        """Αρχικοποίηση του thread"""
        super().__init__()

    def run(self):
        """
        Εκτέλεση της αρχικοποίησης του backend

        Φορτώνει τις βιβλιοθήκες και δημιουργεί την υπηρεσία καθαρισμού EEG.
        Εμφανίζει ενημερώσεις προόδου στο splash screen.
        """
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
    """
    Δημιουργεί το pixmap για το splash screen της εφαρμογής

    Δημιουργεί ένα γραφικό splash screen με τον τίτλο της εφαρμογής
    και υπότιτλο σε μπλε background.

    Returns:
        QPixmap: Το pixmap για το splash screen
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
        "Επαγγελματικός Καθαρισμός EEG Δεδομένων",
    )

    painter.end()
    return pixmap


class LoadingSplashScreen(QSplashScreen):
    """
    Προσαρμοσμένο splash screen με progress bar

    Εμφανίζει την πρόοδο φόρτωσης της εφαρμογής με progress bar
    και status messages κατά την αρχικοποίηση.

    Attributes:
        progress (int): Η τρέχουσα πρόοδος (0-100)
        status_text (str): Το τρέχον μήνυμα κατάστασης
    """

    def __init__(self):
        """Αρχικοποίηση του splash screen"""
        pixmap = create_splash_pixmap()
        super().__init__(pixmap)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )

        # Progress bar
        self.progress = 0
        self.status_text = "Εκκίνηση εφαρμογής..."

    def set_progress(self, value):
        """
        Ορισμός της προόδου και επανασχεδίαση

        Args:
            value (int): Η νέα τιμή προόδου (0-100)
        """
        self.progress = value
        self.repaint()

    def set_status(self, text):
        """
        Ορισμός του status text και επανασχεδίαση

        Args:
            text (str): Το νέο μήνυμα κατάστασης
        """
        self.status_text = text
        self.repaint()

    def drawContents(self, painter):
        """
        Σχεδίαση των περιεχομένων του splash screen

        Σχεδιάζει το progress bar και το status text πάνω στο βασικό pixmap.

        Args:
            painter (QPainter): Ο painter για σχεδίαση
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


# Threads για επεξεργασία EEG δεδομένων
class EEGProcessingThread(QThread):
    """
    Thread για επεξεργασία EEG δεδομένων σε background

    Εκτελεί τη φόρτωση, φιλτράρισμα, ICA ανάλυση και εντοπισμό artifacts
    χωρίς να μπλοκάρει το GUI.

    Signals:
        progress_update (int): Ενημέρωση προόδου (0-100)
        status_update (str): Ενημέρωση κατάστασης
        processing_complete (bool, str): Ολοκλήρωση επεξεργασίας (επιτυχία, μήνυμα)
        ica_ready (dict): ICA δεδομένα έτοιμα για οπτικοποίηση
    """

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(bool, str)
    ica_ready = pyqtSignal(dict)

    def __init__(self, service, input_file, selected_channels=None):
        """
        Αρχικοποίηση του thread επεξεργασίας

        Args:
            service: Η υπηρεσία καθαρισμού EEG
            input_file (str): Διαδρομή του αρχείου εισόδου
            selected_channels (List[str], optional): Επιλεγμένα κανάλια
        """
        super().__init__()
        self.service = service
        self.input_file = input_file
        self.selected_channels = selected_channels

    def run(self):
        """
        Εκτέλεση της επεξεργασίας EEG δεδομένων

        Φορτώνει το αρχείο, εκπαιδεύει το ICA μοντέλο, εντοπίζει artifacts
        και προετοιμάζει τα δεδομένα για οπτικοποίηση.
        """
        try:
            self.status_update.emit("Φόρτωση και προετοιμασία αρχείου...")
            load_result = self.service.load_and_prepare_file(
                self.input_file, self.selected_channels
            )
            if not load_result["success"]:
                self.processing_complete.emit(
                    False,
                    f"Σφάλμα φόρτωσης: {load_result.get('error', 'Άγνωστο σφάλμα')}",
                )
                return
            self.progress_update.emit(30)

            self.status_update.emit("Εκπαίδευση μοντέλου ICA...")
            ica_result = self.service.fit_ica_analysis()
            if not ica_result["success"]:
                self.processing_complete.emit(
                    False, f"Σφάλμα ICA: {ica_result.get('error', 'Άγνωστο σφάλμα')}"
                )
                return
            self.progress_update.emit(70)

            self.status_update.emit("Αυτόματος εντοπισμός artifacts...")
            detection_result = self.service.detect_artifacts()
            if not detection_result["success"]:
                self.processing_complete.emit(
                    False,
                    f"Σφάλμα εντοπισμού: {detection_result.get('error', 'Άγνωστο σφάλμα')}",
                )
                return
            self.progress_update.emit(90)

            viz_data = self.service.get_component_visualization_data()
            if not viz_data:
                self.processing_complete.emit(
                    False, "Αποτυχία δημιουργίας δεδομένων οπτικοποίησης."
                )
                return
            self.ica_ready.emit(viz_data)
            self.progress_update.emit(100)
            self.processing_complete.emit(True, "Έτοιμο για επιλογή.")
        except Exception as e:
            self.processing_complete.emit(False, f"Κρίσιμο σφάλμα: {str(e)}")


class CleaningThread(QThread):
    """
    Thread για καθαρισμό artifacts σε background

    Εφαρμόζει την αφαίρεση των επιλεγμένων artifacts και αποθηκεύει
    τα καθαρά δεδομένα.

    Signals:
        cleaning_complete (bool, str, dict): Ολοκλήρωση καθαρισμού
                                           (επιτυχία, μήνυμα, αποτελέσματα)
    """

    cleaning_complete = pyqtSignal(bool, str, dict)

    def __init__(self, service, components, output_file):
        """
        Αρχικοποίηση του thread καθαρισμού

        Args:
            service: Η υπηρεσία καθαρισμού EEG
            components (List[int]): Λίστα συνιστωσών προς αφαίρεση
            output_file (str): Διαδρομή αρχείου εξόδου
        """
        super().__init__()
        self.service = service
        self.components_to_remove = components
        self.output_file = output_file

    def run(self):
        """
        Εκτέλεση του καθαρισμού artifacts

        Εφαρμόζει την αφαίρεση των επιλεγμένων συνιστωσών και αποθηκεύει
        τα καθαρά δεδομένα σε αρχείο EDF.
        """
        try:
            clean_result = self.service.apply_artifact_removal(
                self.components_to_remove
            )
            if not clean_result["success"]:
                self.cleaning_complete.emit(
                    False, clean_result.get("error", "Άγνωστο σφάλμα"), {}
                )
                return
            cleaned_data = clean_result["cleaned_data"]
            if not self.service.save_cleaned_data(cleaned_data, self.output_file):
                self.cleaning_complete.emit(False, "Σφάλμα αποθήκευσης αρχείου.", {})
                return

            # Get original data for comparison
            original_data = self.service.backend_core.get_filtered_data()

            results = {
                **clean_result,
                "input_file": self.service.current_file,
                "output_file": self.output_file,
                "original_data": original_data,
            }
            self.cleaning_complete.emit(True, "Ο καθαρισμός ολοκληρώθηκε!", results)
        except Exception as e:
            self.cleaning_complete.emit(False, f"Κρίσιμο σφάλμα: {str(e)}", {})


class EEGArtifactCleanerGUI(QMainWindow):
    """
    Κύρια κλάση GUI για την εφαρμογή καθαρισμού EEG artifacts

    Διαχειρίζεται όλες τις οθόνες της εφαρμογής και την αλληλεπίδραση με τον χρήστη:
    - Welcome screen για επιλογή αρχείου
    - Channel selection για επιλογή καναλιών
    - ICA component selector για επιλογή artifacts προς αφαίρεση
    - Comparison screen για σύγκριση αποτελεσμάτων

    Attributes:
        service: Η υπηρεσία backend για καθαρισμό EEG
        ica_selector_screen: Η οθόνη επιλογής ICA συνιστωσών
        current_input_file (str): Το τρέχον αρχείο εισόδου
        splash: Το splash screen κατά την εκκίνηση
    """

    def __init__(self):
        """
        Αρχικοποίηση της κύριας εφαρμογής GUI

        Δημιουργεί το splash screen και αρχίζει την αρχικοποίηση του backend
        σε ξεχωριστό thread.
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
        Καλείται όταν ολοκληρωθεί η αρχικοποίηση του backend

        Δημιουργεί τα στοιχεία του GUI και εμφανίζει το κύριο παράθυρο.

        Args:
            service: Η αρχικοποιημένη υπηρεσία καθαρισμού EEG
        """
        try:
            self.service = service

            # Create GUI components in main thread
            self.status_update_timer = QTimer()
            self.splash.set_status("Δημιουργία στοιχείων GUI...")
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
                "Σφάλμα Αρχικοποίησης",
                f"Δεν ήταν δυνατή η αρχικοποίηση της εφαρμογής:\n{str(e)}",
            )
            if hasattr(self, "splash"):
                self.splash.hide()
            sys.exit(1)

    def finish_loading(self):
        """
        Ολοκληρώνει τη διαδικασία φόρτωσης

        Κρύβει το splash screen και εμφανίζει το κύριο παράθυρο.
        """
        self.splash.hide()
        self.show()

    def setup_ui(self):
        """
        Δημιουργία και διάταξη των στοιχείων του GUI

        Δημιουργεί το stacked widget για τις διάφορες οθόνες και ρυθμίζει
        το γενικό στυλ της εφαρμογής.
        """
        self.setWindowTitle("Katharsis - EEG Artifact Cleaner Pro")
        self.setGeometry(100, 100, 1100, 850)
        self.setMinimumSize(800, 600)

        # Η δημιουργία του theme μεταφέρεται εδώ για να είναι διαθέσιμο στα child widgets
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
        self.status_bar.showMessage("Έτοιμο")

    def create_welcome_screen(self):
        """
        Δημιουργία της οθόνης καλωσορίσματος

        Δημιουργεί την αρχική οθόνη με τον τίτλο της εφαρμογής και
        το κουμπί επιλογής αρχείου.

        Returns:
            QWidget: Η οθόνη καλωσορίσματος
        """
        screen = QWidget()
        layout = QVBoxLayout(screen)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(25)
        title = QLabel("🧠 Katharsis - EEG Artifact Cleaner")
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
        """
        Ρύθμιση των συνδέσεων σημάτων μεταξύ των widgets

        Συνδέει τα σήματα των διάφορων οθονών με τις αντίστοιχες μεθόδους
        για την επικοινωνία μεταξύ των components.
        """
        self.select_input_btn.clicked.connect(self.select_input_file)
        self.channel_selector_screen.channels_selected.connect(
            self.on_channels_selected
        )
        self.ica_selector_screen.components_selected.connect(self.apply_cleaning)
        self.comparison_screen.return_to_home.connect(self.reset_ui)

    def show_message_box(self, icon, title, text):
        """
        Βοηθητική συνάρτηση για εμφάνιση QMessageBox με σωστό στυλ

        Δημιουργεί και εμφανίζει ένα message box με το theme της εφαρμογής.

        Args:
            icon: Το εικονίδιο του message box (QMessageBox.Icon)
            title (str): Ο τίτλος του παραθύρου
            text (str): Το κείμενο του μηνύματος
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(text)
        msg_box.setWindowTitle(title)
        # Εφαρμόζουμε το global stylesheet στο messagebox πριν το δείξουμε
        msg_box.setStyleSheet(QApplication.instance().styleSheet())
        msg_box.exec()

    def select_input_file(self):
        """
        Επιλογή αρχείου EDF για επεξεργασία

        Ανοίγει file dialog για επιλογή αρχείου EDF και μεταβαίνει στην
        οθόνη επιλογής καναλιών.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Επιλογή EDF",
            str(Path.home()),
            "*.edf",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if file_path:
            self.current_input_file = file_path
            # Go to channel selection instead of directly processing
            self.show_channel_selection()

    def show_channel_selection(self):
        """
        Εμφάνιση της οθόνης επιλογής καναλιών

        Φορτώνει το επιλεγμένο αρχείο στην οθόνη επιλογής καναλιών και
        μεταβαίνει σε αυτή την οθόνη.
        """
        try:
            self.channel_selector_screen.set_edf_file(self.current_input_file)
            # Navigate to channel selection screen (index 1)
            self.stacked_widget.setCurrentIndex(1)
            self.status_bar.showMessage("Επιλέξτε κανάλια για ανάλυση")
        except Exception as e:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Σφάλμα",
                f"Αδυναμία φόρτωσης αρχείου για επιλογή καναλιών:\n{str(e)}",
            )

    def on_channels_selected(self, selected_channels, analysis_method="ICA"):
        """
        Χειρισμός επιλογής καναλιών και έναρξη επεξεργασίας

        Αποθηκεύει τα επιλεγμένα κανάλια και τη μέθοδο ανάλυσης,
        και ξεκινά την επεξεργασία των δεδομένων.

        Args:
            selected_channels (List[str]): Λίστα επιλεγμένων καναλιών
            analysis_method (str): Μέθοδος ανάλυσης ("ICA" ή "PCA")
        """
        self.selected_channels = selected_channels
        self.analysis_method = analysis_method
        # Set the analysis method in the service
        self.service.set_analysis_method(analysis_method)
        self.start_processing()

    def start_processing(self):
        """
        Έναρξη της επεξεργασίας των EEG δεδομένων

        Δημιουργεί και ξεκινά το thread επεξεργασίας για φόρτωση αρχείου,
        ICA ανάλυση και εντοπισμό artifacts.
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
        Χειρισμός ετοιμότητας των ICA δεδομένων

        Φορτώνει τα δεδομένα οπτικοποίησης στην οθόνη επιλογής συνιστωσών
        και μεταβαίνει σε αυτή την οθόνη.

        Args:
            viz_data (dict): Δεδομένα για οπτικοποίηση των ICA συνιστωσών
        """
        self.ica_selector_screen.set_ica_data(**viz_data)
        # Navigate to ICA selector screen (index 2)
        self.stacked_widget.setCurrentIndex(2)

    def apply_cleaning(self, selected_components):
        """
        Εφαρμογή καθαρισμού artifacts

        Ζητά από τον χρήστη να επιλέξει αρχείο εξόδου και ξεκινά τον
        καθαρισμό των επιλεγμένων artifacts.

        Args:
            selected_components (List[int]): Λίστα συνιστωσών προς αφαίρεση
        """
        default_path = self.current_input_file.replace(".edf", "_clean.edf")
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Αποθήκευση Καθαρού Αρχείου",
            default_path,
            "*.edf",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not output_file:
            self.status_bar.showMessage("Η διαδικασία καθαρισμού ακυρώθηκε.", 3000)
            return
        self.cleaning_thread = CleaningThread(
            self.service, selected_components, output_file
        )
        self.cleaning_thread.cleaning_complete.connect(self.on_cleaning_complete)
        self.cleaning_thread.start()
        self.status_bar.showMessage("Εφαρμογή καθαρισμού...")

    def on_processing_complete(self, success, message):
        """
        Χειρισμός ολοκλήρωσης επεξεργασίας

        Εμφανίζει μήνυμα σφάλματος αν η επεξεργασία απέτυχε και επαναφέρει το UI.

        Args:
            success (bool): Αν η επεξεργασία ήταν επιτυχής
            message (str): Μήνυμα κατάστασης
        """
        if not success:
            self.show_message_box(QMessageBox.Icon.Critical, "Σφάλμα", message)
            self.reset_ui()

    def on_cleaning_complete(self, success, message, results):
        """
        Χειρισμός ολοκλήρωσης καθαρισμού

        Εμφανίζει την οθόνη σύγκρισης αποτελεσμάτων αν ο καθαρισμός ήταν επιτυχής,
        ή μήνυμα σφάλματος αν απέτυχε.

        Args:
            success (bool): Αν ο καθαρισμός ήταν επιτυχής
            message (str): Μήνυμα κατάστασης
            results (dict): Αποτελέσματα καθαρισμού για σύγκριση
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
                    "Σύγκριση αποτελεσμάτων - Επιτυχής καθαρισμός!"
                )
            except Exception as e:
                # Fallback to original message box if comparison screen fails
                full_message = f"{message}\n\nΑποθηκεύτηκε στο:\n{results['output_file']}\n\nΣφάλμα οθόνης σύγκρισης: {str(e)}"
                self.show_message_box(
                    QMessageBox.Icon.Information, "Επιτυχία", full_message
                )
                self.reset_ui()
        else:
            self.show_message_box(QMessageBox.Icon.Critical, "Σφάλμα", message)
            self.reset_ui()

    def reset_ui(self):
        """
        Επαναφορά του UI στην αρχική κατάσταση

        Επιστρέφει στην οθόνη καλωσορίσματος και επαναφέρει την κατάσταση
        των στοιχείων ελέγχου.
        """
        self.stacked_widget.setCurrentIndex(0)
        self.select_input_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Έτοιμο")


def get_global_stylesheet(theme):
    """
    Δημιουργεί το κεντρικό stylesheet για ολόκληρη την εφαρμογή

    Δημιουργεί ένα ολοκληρωμένο CSS stylesheet που καλύπτει όλα τα widgets
    της εφαρμογής, χρησιμοποιώντας τα χρώματα του theme.

    Args:
        theme (dict): Dictionary με τα χρώματα του theme
                     (background, primary, success, text, κλπ.)

    Returns:
        str: Το CSS stylesheet για την εφαρμογή
    """
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
    """
    Κύρια συνάρτηση εκκίνησης της εφαρμογής

    Αρχικοποιεί την εφαρμογή PyQt6, εφαρμόζει το global stylesheet
    και εκκινεί το κύριο παράθυρο της εφαρμογής.
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

    # Εφαρμόζουμε το στυλ σε ολόκληρη την εφαρμογή
    app.setStyleSheet(get_global_stylesheet(default_theme))

    window = EEGArtifactCleanerGUI()
    # Don't show window immediately - it will be shown after loading completes
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
