#!/usr/bin/env python3
"""
ICA Component Selector Widget - v4.0 - Correct Event Bubbling for Scrolling
"""
from typing import Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# --- 1. ΔΗΜΙΟΥΡΓΟΥΜΕ ΕΝΑ CUSTOM CANVAS ---
# Αυτή η κλάση κληρονομεί όλες τις ιδιότητες του FigureCanvas,
# αλλά αλλάζει τη συμπεριφορά του wheelEvent.
class CustomCanvas(FigureCanvas):
    def wheelEvent(self, event: QWheelEvent):
        """
        Αντί να "καταναλώσει" το event, το αγνοεί.
        Όταν ένα event αγνοείται, η Qt το προωθεί αυτόματα στο γονικό widget.
        """
        event.ignore()


# --- 2. BACKGROUND THREAD ΓΙΑ PREVIEW UPDATE ---
class PreviewUpdateThread(QThread):
    """Thread για υπολογισμό του καθαρισμένου σήματος στο background"""

    preview_ready = pyqtSignal(object, object)  # (original_raw, cleaned_raw)

    def __init__(self, ica, raw, components_to_remove: List[int]):
        super().__init__()
        self.ica = ica
        self.raw = raw
        self.components_to_remove = components_to_remove

    def run(self):
        try:
            # If no components to remove, return original signal as both
            if not self.components_to_remove:
                self.preview_ready.emit(self.raw, self.raw)
                return

            # Δημιουργία αντιγράφου για καθαρισμό
            cleaned_raw = self.raw.copy()

            # Ορισμός συνιστωσών προς αφαίρεση
            ica_copy = self.ica.copy()
            ica_copy.exclude = self.components_to_remove

            # Εφαρμογή καθαρισμού
            cleaned_raw = ica_copy.apply(cleaned_raw, verbose=False)

            # Εκπομπή των αποτελεσμάτων
            self.preview_ready.emit(self.raw, cleaned_raw)

        except Exception as e:
            print(f"Σφάλμα στο preview thread: {str(e)}")
            # Εκπομπή μόνο του αρχικού σήματος σε περίπτωση σφάλματος
            self.preview_ready.emit(self.raw, None)


# --- 3. PREVIEW WIDGET ---
class PreviewWidget(QWidget):
    """Widget για εμφάνιση preview του καθαρισμένου σήματος"""

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self.selected_channel_idx = 0
        self.channel_names = []
        self.update_callback = None  # Callback για ενημέρωση preview
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header layout με τίτλο και dropdown για επιλογή καναλιού
        header_layout = QHBoxLayout()

        # Τίτλος
        title_label = QLabel("📊 Ζωντανή Προεπισκόπηση Αποτελέσματος Καθαρισμού")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.theme['text']}; margin-bottom: 5px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Dropdown για επιλογή καναλιού
        channel_label = QLabel("Κανάλι / Channel:")
        channel_label.setStyleSheet(f"color: {self.theme['text']}; font-size: 12px;")
        header_layout.addWidget(channel_label)

        self.channel_dropdown = QComboBox()
        self.channel_dropdown.setMinimumWidth(180)
        self.channel_dropdown.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.theme.get('background', '#ffffff')};
                color: {self.theme['text']};
                border: 2px solid {self.theme.get('border', '#dee2e6')};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 500;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {self.theme.get('primary', '#007AFF')};
                background-color: {self.theme.get('background', '#ffffff')};
            }}
            QComboBox:focus {{
                border-color: {self.theme.get('primary', '#007AFF')};
                outline: none;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                background-color: transparent;
            }}
            QComboBox::drop-down:hover {{
                background-color: rgba(0, 122, 255, 0.1);
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {self.theme['text']};
                margin: 0px;
            }}
            QComboBox::down-arrow:hover {{
                border-top-color: {self.theme.get('primary', '#007AFF')};
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.theme.get('background', '#ffffff')};
                color: {self.theme['text']};
                border: 2px solid {self.theme.get('primary', '#007AFF')};
                border-radius: 8px;
                padding: 4px;
                outline: none;
                selection-background-color: {self.theme.get('primary', '#007AFF')};
                selection-color: white;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                margin: 2px;
                border-radius: 4px;
                background-color: transparent;
                color: {self.theme['text']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: rgba(0, 122, 255, 0.1);
                color: {self.theme.get('primary', '#007AFF')};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.theme.get('primary', '#007AFF')};
                color: white;
            }}
        """
        )
        self.channel_dropdown.currentIndexChanged.connect(self._on_channel_changed)
        header_layout.addWidget(self.channel_dropdown)

        layout.addLayout(header_layout)

        # Canvas για τα γραφήματα
        self.figure = Figure(figsize=(12, 6), dpi=80)
        self.canvas = CustomCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Αρχική εμφάνιση κενού γραφήματος
        self.show_empty_plot()

    def set_update_callback(self, callback):
        """Ορισμός callback για ενημέρωση preview"""
        self.update_callback = callback

    def set_channel_data(self, raw):
        """Ενημέρωση του dropdown με τα διαθέσιμα κανάλια"""
        self.channel_names = raw.ch_names
        self.channel_dropdown.clear()
        self.channel_dropdown.addItems(self.channel_names)
        self.selected_channel_idx = 0

    def _on_channel_changed(self, index):
        """Καλείται όταν αλλάζει η επιλογή καναλιού"""
        self.selected_channel_idx = index
        # Trigger preview update if we have a callback
        if self.update_callback:
            self.update_callback()

    def show_empty_plot(self):
        """Εμφάνιση κενού γραφήματος με οδηγίες"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Επιλέξτε συνιστώσες για να δείτε την προεπισκόπηση του καθαρισμένου σήματος\n"
            "Select components to see preview of cleaned signal",
            ha="center",
            va="center",
            fontsize=12,
            color=self.theme.get("text_light", "#6c757d"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        self.canvas.draw()

    def update_preview(self, original_raw, cleaned_raw):
        """Ενημέρωση του preview με τα νέα δεδομένα"""
        try:
            self.figure.clear()

            # Χρησιμοποιούμε τα πρώτα 10 δευτερόλεπτα για preview
            preview_duration = 10.0
            n_samples = int(preview_duration * original_raw.info["sfreq"])

            # Λήψη των δεδομένων
            original_data = original_raw.get_data()[:, :n_samples]
            time_points = np.arange(n_samples) / original_raw.info["sfreq"]

            if cleaned_raw is not None:
                cleaned_data = cleaned_raw.get_data()[:, :n_samples]

                # Ensure cleaned_data matches the expected length
                min_samples = min(
                    original_data.shape[1], cleaned_data.shape[1], len(time_points)
                )
                original_data = original_data[:, :min_samples]
                cleaned_data = cleaned_data[:, :min_samples]
                time_points = time_points[:min_samples]

                # Δύο subplots - αρχικό και καθαρισμένο
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2)

                # Εμφάνιση του επιλεγμένου καναλιού
                channel_idx = self.selected_channel_idx

                # Αρχικό σήμα
                ax1.plot(
                    time_points,
                    original_data[channel_idx, :],
                    color=self.theme.get("danger", "#e74c3c"),
                    linewidth=1,
                    alpha=0.8,
                )
                channel_name = (
                    self.channel_names[channel_idx]
                    if channel_idx < len(self.channel_names)
                    else f"Channel {channel_idx}"
                )
                ax1.set_title(
                    f"Αρχικό Σήμα - {channel_name} / Original Signal - {channel_name}",
                    fontsize=10,
                    color=self.theme["text"],
                )
                ax1.set_ylabel("Amplitude (μV)", fontsize=9)
                ax1.grid(True, alpha=0.3)

                # Καθαρισμένο σήμα
                ax2.plot(
                    time_points,
                    cleaned_data[channel_idx, :],
                    color=self.theme.get("success", "#27ae60"),
                    linewidth=1,
                    alpha=0.8,
                )
                ax2.set_title(
                    f"Καθαρισμένο Σήμα - {channel_name} / Cleaned Signal - {channel_name}",
                    fontsize=10,
                    color=self.theme["text"],
                )
                ax2.set_xlabel("Χρόνος (s) / Time (s)", fontsize=9)
                ax2.set_ylabel("Amplitude (μV)", fontsize=9)
                ax2.grid(True, alpha=0.3)

            else:
                # Μόνο το αρχικό σήμα αν υπάρχει σφάλμα
                ax = self.figure.add_subplot(111)
                channel_idx = self.selected_channel_idx
                channel_name = (
                    self.channel_names[channel_idx]
                    if channel_idx < len(self.channel_names)
                    else f"Channel {channel_idx}"
                )
                ax.plot(
                    time_points,
                    original_data[channel_idx, :],
                    color=self.theme.get("primary", "#007AFF"),
                    linewidth=1,
                )
                ax.set_title(
                    f"Αρχικό Σήμα - {channel_name} / Original Signal - {channel_name}",
                    fontsize=12,
                    color=self.theme["text"],
                )
                ax.set_xlabel("Χρόνος (s) / Time (s)", fontsize=10)
                ax.set_ylabel("Amplitude (μV)", fontsize=10)
                ax.grid(True, alpha=0.3)

            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()

        except Exception as e:
            print(f"Σφάλμα στην ενημέρωση preview: {str(e)}")
            self.show_error_plot(str(e))

    def show_error_plot(self, error_msg: str):
        """Εμφάνιση μηνύματος σφάλματος"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Σφάλμα στην προεπισκόπηση:\n{error_msg}",
            ha="center",
            va="center",
            fontsize=10,
            color=self.theme.get("danger", "#e74c3c"),
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()


class ComponentDisplayWidget(QWidget):
    def __init__(self, component_idx: int, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.component_idx = component_idx
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        # Οριζόντια διάταξη για time-series και topomap
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Figure για time-series (αριστερά)
        self.timeseries_figure = Figure(figsize=(4, 2.5), dpi=90)
        self.timeseries_canvas = CustomCanvas(self.timeseries_figure)
        layout.addWidget(self.timeseries_canvas, 2)  # 2/3 του χώρου

        # Figure για topomap (δεξιά)
        self.topomap_figure = Figure(figsize=(2.5, 2.5), dpi=90)
        self.topomap_canvas = CustomCanvas(self.topomap_figure)
        layout.addWidget(self.topomap_canvas, 1)  # 1/3 του χώρου

    def plot_component(self, ica, raw, is_artifact, component_info):
        try:
            # 1. Time-series plot (αριστερά)
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)

            sources = ica.get_sources(raw).get_data()
            comp_data = sources[self.component_idx]
            times = raw.times[: len(comp_data)]
            color = (
                self.theme.get("danger", "#e74c3c")
                if is_artifact
                else self.theme.get("success", "#27ae60")
            )

            ax_time.plot(times, comp_data, color=color, linewidth=1)
            ax_time.set_title(
                f"IC {self.component_idx} - Time Series",
                fontsize=9,
                color=self.theme["text"],
            )
            ax_time.grid(True, linestyle="--", alpha=0.5)
            ax_time.set_xlabel("Time (s)", fontsize=8)
            ax_time.set_ylabel("Amplitude", fontsize=8)
            self.timeseries_figure.tight_layout(pad=0.3)

            # 2. Topographic map (δεξιά)
            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)

            # Λήψη των spatial patterns της συνιστώσας
            component_weights = ica.get_components()[:, self.component_idx]

            # Τοπογραφική απεικόνιση με MNE
            import mne.viz

            mne.viz.plot_topomap(
                component_weights,
                raw.info,
                axes=ax_topo,
                show=False,
                cmap="RdBu_r",
                sensors=True,
            )
            ax_topo.set_title(
                f"IC {self.component_idx} - Topomap",
                fontsize=9,
                color=self.theme["text"],
            )
            self.topomap_figure.tight_layout(pad=0.3)

        except Exception as e:
            # Σε περίπτωση σφάλματος, εμφάνιση μηνύματος και στα δύο plots
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111)
            ax_time.text(
                0.5,
                0.5,
                f"Time series error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)
            ax_topo.text(
                0.5,
                0.5,
                f"Topomap error: {e}",
                ha="center",
                va="center",
                color="red",
                fontsize=8,
            )

        # Ανανέωση και των δύο canvas
        self.timeseries_canvas.draw()
        self.topomap_canvas.draw()


class ICAComponentSelector(QWidget):
    components_selected = pyqtSignal(list)

    def __init__(self, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.theme = theme
        self.ica = None
        self.raw = None
        self.suggested_artifacts = []
        self.checkboxes = {}
        self.component_widgets = {}
        self.components_info = {}
        self.explanations = {}

        # Preview functionality
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)  # Μόνο μία φορά όταν λήξει
        self.preview_timer.timeout.connect(self._start_preview_update)
        self.preview_thread = None

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        title_label = QLabel("🔍 Επιλογή ICA Συνιστωσών για Αφαίρεση")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        controls_layout = QHBoxLayout()
        self.select_suggested_btn = QPushButton("Επιλογή Προτεινόμενων")
        self.select_all_btn = QPushButton("Επιλογή Όλων")
        self.select_none_btn = QPushButton("Καμία Επιλογή")
        controls_layout.addWidget(self.select_suggested_btn)
        controls_layout.addWidget(self.select_all_btn)
        controls_layout.addWidget(self.select_none_btn)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ border: none; background: {self.theme['background']}; width: 12px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: #bdc3c7; min-height: 20px; border-radius: 6px; }}
            QScrollBar::handle:vertical:hover {{ background: #95a5a6; }}
        """
        )

        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(0, 0, 5, 0)
        self.components_layout.setSpacing(10)
        self.scroll_area.setWidget(self.components_widget)
        main_layout.addWidget(self.scroll_area, 1)  # Μικρότερο stretch factor

        # Προσθήκη του Preview Widget
        self.preview_widget = PreviewWidget(self.theme)
        self.preview_widget.setMinimumHeight(300)  # Ελάχιστο ύψος για το preview
        main_layout.addWidget(self.preview_widget, 1)  # Ίσος χώρος με το scroll area

        self.apply_btn = QPushButton("✅ Εφαρμογή Καθαρισμού και Αποθήκευση")
        self.apply_btn.setMinimumHeight(50)
        self.apply_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        main_layout.addWidget(self.apply_btn)

        self.apply_styling()

        self.select_all_btn.clicked.connect(lambda: self.set_all_checkboxes(True))
        self.select_none_btn.clicked.connect(lambda: self.set_all_checkboxes(False))
        self.select_suggested_btn.clicked.connect(self.select_suggested)
        self.apply_btn.clicked.connect(self.emit_selected_components)

        # --- 3. ΑΦΑΙΡΟΥΜΕ ΤΟ ΠΑΛΙΟ EVENT FILTER ---
        # Δεν χρειάζεται πλέον, αφού λύσαμε το πρόβλημα στην πηγή του.
        # self.installEventFilter(self) <-- ΑΦΑΙΡΕΘΗΚΕ

    def apply_styling(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        btn_style = f"""
            QPushButton {{
                background-color: #5D6D7E; color: white; padding: 10px; 
                border: none; font-size: 12px; border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: #85929E; }}
        """
        self.select_all_btn.setStyleSheet(btn_style)
        self.select_none_btn.setStyleSheet(btn_style)
        self.select_suggested_btn.setStyleSheet(btn_style)

        self.apply_btn.setStyleSheet(
            f"""
            QPushButton {{ background-color: {self.theme['success']}; color: white; border-radius: 8px; }}
            QPushButton:hover {{ background-color: {self.theme.get('success_hover', self.theme['success'])}; }}
        """
        )

    def create_single_component_widget(self, i):
        is_artifact = i in self.suggested_artifacts
        comp_container = QWidget()
        comp_container.setMinimumHeight(200)
        comp_layout = QHBoxLayout(comp_container)
        comp_layout.setContentsMargins(10, 5, 10, 5)

        # Δημιουργούμε ένα κάθετο layout για το checkbox και το νέο κουμπί
        controls_layout = QVBoxLayout()

        checkbox = QCheckBox(f" IC {i}")
        checkbox.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        checkbox.setChecked(is_artifact)
        checkbox.setStyleSheet(f"color: {self.theme['text_light']}; border: none;")
        checkbox.toggled.connect(
            lambda state, widget=comp_container: self.update_selection_style(
                widget, state
            )
        )
        checkbox.toggled.connect(self._on_checkbox_toggled)  # Προσθήκη για preview
        self.checkboxes[i] = checkbox

        # Το νέο κουμπί "Ανάλυση"
        details_btn = QPushButton("🔎 Ανάλυση")
        details_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #e9ecef; /* Απαλό γκρι φόντο */
                color: {self.theme.get('text_light', '#6c757d')}; /* Πιο σκούρο κείμενο */
                border: 1px solid {self.theme.get('border', '#dee2e6')};
                font-size: 11px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #d4e6f1; /* Απαλό μπλε στο hover */
                border-color: {self.theme.get('primary', '#007AFF')};
                color: {self.theme.get('text', '#212529')};
            }}
        """
        )
        details_btn.clicked.connect(
            lambda state, idx=i: self.show_component_properties(idx)
        )  # Σύνδεση με τη νέα συνάρτηση

        controls_layout.addWidget(checkbox)
        controls_layout.addWidget(details_btn)
        controls_layout.addStretch()

        plot_widget = ComponentDisplayWidget(i, self.theme)
        plot_widget.plot_component(self.ica, self.raw, is_artifact, {})

        comp_layout.addLayout(controls_layout)  # Προσθέτουμε το layout με τα controls
        comp_layout.addWidget(plot_widget, 1)
        self.components_layout.addWidget(comp_container)

        self.update_selection_style(comp_container, checkbox.isChecked())

    def update_selection_style(self, widget: QWidget, is_selected: bool):
        # ... (Η συνάρτηση παραμένει ίδια)
        if is_selected:
            bg_color = "#fadbd8"
            border_color = self.theme["danger"]
        else:
            bg_color = "#e8f8f5"
            border_color = self.theme["success"]
        widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 5px;
            }}
        """
        )

    def set_all_checkboxes(self, state: bool):
        # ... (Η συνάρτηση παραμένει ίδια)
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)
        # Trigger preview update after setting all checkboxes
        self._on_checkbox_toggled()

    def select_suggested(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        for i, checkbox in self.checkboxes.items():
            checkbox.setChecked(i in self.suggested_artifacts)
        # Trigger preview update after selecting suggested
        self._on_checkbox_toggled()

    def emit_selected_components(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        selected = [i for i, cb in self.checkboxes.items() if cb.isChecked()]
        self.components_selected.emit(selected)

    def get_selected_components(self):
        """Return list of selected component indices"""
        return [i for i, cb in self.checkboxes.items() if cb.isChecked()]

    def select_all_components(self):
        """Select all components"""
        self.set_all_checkboxes(True)

    def select_no_components(self):
        """Deselect all components"""
        self.set_all_checkboxes(False)

    def select_suggested_components(self):
        """Select only the suggested artifact components"""
        self.select_suggested()

    def _on_checkbox_toggled(self):
        """Called when any checkbox is toggled - starts the preview update timer"""
        if self.ica and self.raw:
            # Restart the timer - αν ο χρήστης κάνει γρήγορες αλλαγές,
            # περιμένουμε 500ms από την τελευταία αλλαγή
            self.preview_timer.stop()
            self.preview_timer.start(500)

    def _start_preview_update(self):
        """Starts the background thread to compute the cleaned signal"""
        if not self.ica or not self.raw:
            return

        # Ακύρωσε το τυχόν προηγούμενο thread αν τρέχει ακόμα
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.quit()
            self.preview_thread.wait()

        # Πάρε τις τρέχουσες επιλεγμένες συνιστώσες
        selected_components = [i for i, cb in self.checkboxes.items() if cb.isChecked()]

        # Δημιούργησε και ξεκίνησε το νέο thread
        self.preview_thread = PreviewUpdateThread(
            self.ica, self.raw, selected_components
        )
        self.preview_thread.preview_ready.connect(self.preview_widget.update_preview)
        self.preview_thread.start()

    def set_ica_data(
        self,
        ica,
        raw,
        suggested_artifacts,
        components_info=None,
        explanations=None,
        **kwargs,
    ):
        # ... (Η συνάρτηση παραμένει ίδια με προσθήκη αρχικού preview)
        self.ica = ica
        self.raw = raw
        self.suggested_artifacts = suggested_artifacts
        self.components_info = components_info or {}
        self.explanations = explanations or {}

        while self.components_layout.count():
            item = self.components_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.checkboxes.clear()
        self.component_widgets.clear()
        for i in range(self.ica.n_components_):
            self.create_single_component_widget(i)
        self.components_layout.addStretch(1)

        # Ενημέρωση του preview widget με τα δεδομένα καναλιών και callback
        self.preview_widget.set_channel_data(raw)
        self.preview_widget.set_update_callback(self._start_preview_update)

        # Ενημέρωση του αρχικού preview με τις προτεινόμενες συνιστώσες
        if suggested_artifacts:
            self._start_preview_update()

    def _create_spectrogram_plot(self, component_idx):
        """
        Δημιουργεί spectrogram γράφημα για τη συγκεκριμένη ICA συνιστώσα.
        Το spectrogram είναι ιδανικό για τον εντοπισμό μυϊκών artifacts που 
        εμφανίζονται ως σύντομες εκρήξεις ενέργειας σε ευρύ φάσμα συχνοτήτων.
        """
        try:
            from scipy import signal
            import numpy as np
            
            # Λήψη των ICA sources
            sources = self.ica.get_sources(self.raw).get_data()
            component_data = sources[component_idx]
            
            # Παράμετροι για το spectrogram
            fs = self.raw.info['sfreq']  # Συχνότητα δειγματολήψίας
            
            # Υπολογισμός spectrogram
            # Χρησιμοποιούμε παράθυρο που δίνει καλή ανάλυση χρόνου-συχνότητας
            nperseg = min(1024, len(component_data) // 8)  # Μέγεθος παραθύρου
            noverlap = nperseg // 2  # Επικάλυψη παραθύρων
            
            frequencies, times, Sxx = signal.spectrogram(
                component_data, 
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            
            # Δημιουργία figure
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Εμφάνιση spectrogram σε dB scale για καλύτερη οπτικοποίηση
            Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Προσθέτουμε μικρή τιμή για αποφυγή log(0)
            
            # Δημιουργία του spectrogram plot
            im = ax.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
            
            # Ρύθμιση αξόνων και ετικετών
            ax.set_ylabel('Συχνότητα (Hz) / Frequency (Hz)', fontsize=10)
            ax.set_xlabel('Χρόνος (s) / Time (s)', fontsize=10)
            ax.set_title(f'Spectrogram - IC {component_idx}\n(Ανάλυση Χρόνου-Συχνότητας για Εντοπισμό Μυϊκών Artifacts)', 
                        fontsize=11, color=self.theme.get('text', '#000000'))
            
            # Περιορισμός συχνοτήτων στο ενδιαφέρον εύρος (0-100 Hz τυπικά για EEG)
            ax.set_ylim(0, min(100, fs/2))
            
            # Προσθήκη colorbar
            cbar = fig.colorbar(im, ax=ax, label='Ισχύς (dB) / Power (dB)')
            cbar.ax.tick_params(labelsize=8)
            
            # Grid για καλύτερη αναγνωσιμότητα
            ax.grid(True, alpha=0.3)
            
            # Τελική διαμόρφωση
            fig.tight_layout(pad=2.0)
            
            return fig
            
        except Exception as e:
            print(f"Σφάλμα στη δημιουργία spectrogram: {str(e)}")
            
            # Σε περίπτωση σφάλματος, δημιουργούμε ένα figure με μήνυμα σφάλματος
            fig = Figure(figsize=(10, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Σφάλμα στη δημιουργία Spectrogram:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title(f'Spectrogram - IC {component_idx} (Σφάλμα)', 
                        color=self.theme.get('text', '#000000'))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            return fig

    def show_component_properties(self, component_idx):
        """
        Δημιουργεί και εμφανίζει ένα νέο παράθυρο με τις ιδιότητες της συνιστώσας.
        Περιλαμβάνει τοπογραφία, PSD και Spectrogram για πλήρη ανάλυση.
        """
        if not self.ica or not self.raw:
            return

        # Το MNE δημιουργεί τα γραφήματα. Το show=False είναι κρίσιμο
        # για να πάρουμε τα figures αντί να τα εμφανίσει μόνο του.
        figures = self.ica.plot_properties(self.raw, picks=component_idx, show=False)

        # Δημιουργούμε το spectrogram γράφημα
        spectrogram_fig = self._create_spectrogram_plot(component_idx)

        # Δημιουργούμε ένα νέο παράθυρο διαλόγου (pop-up)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Λεπτομερής Ανάλυση Συνιστώσας IC {component_idx} / Detailed Analysis of Component IC {component_idx}")
        dialog.setMinimumSize(1000, 800)  # Μεγαλύτερο παράθυρο για το επιπλέον γράφημα
        dialog_layout = QVBoxLayout(dialog)

        # Προσθήκη τίτλου
        title_label = QLabel(f"🔬 Ανάλυση Συνιστώσας IC {component_idx}")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {self.theme['text']}; margin: 10px; text-align: center;")
        dialog_layout.addWidget(title_label)

        # Για κάθε figure που έφτιαξε το MNE, δημιουργούμε έναν καμβά και τον
        # προσθέτουμε στο παράθυρο.
        for fig in figures:
            canvas = FigureCanvas(fig)
            dialog_layout.addWidget(canvas)

        # Προσθήκη του spectrogram στο τέλος
        if spectrogram_fig:
            spectrogram_canvas = FigureCanvas(spectrogram_fig)
            dialog_layout.addWidget(spectrogram_canvas)

        # Εμφανίζουμε το παράθυρο
        dialog.exec()
