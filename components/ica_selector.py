#!/usr/bin/env python3
"""
ICA Component Selector Widget - v4.0 - Correct Event Bubbling for Scrolling
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QCheckBox, QLabel, QPushButton, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QFont, QWheelEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List

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

class ComponentDisplayWidget(QWidget):
    def __init__(self, component_idx: int, theme: Dict[str, str], parent=None):
        super().__init__(parent)
        self.component_idx = component_idx
        self.theme = theme
        self.setup_ui()
    
    def setup_ui(self):
        # Οριζόντια διάταξη για time-series και topomap
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
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
            times = raw.times[:len(comp_data)]
            color = self.theme.get('danger', '#e74c3c') if is_artifact else self.theme.get('success', '#27ae60')
            
            ax_time.plot(times, comp_data, color=color, linewidth=1)
            ax_time.set_title(f"IC {self.component_idx} - Time Series", fontsize=9, color=self.theme['text'])
            ax_time.grid(True, linestyle='--', alpha=0.5)
            ax_time.set_xlabel('Time (s)', fontsize=8)
            ax_time.set_ylabel('Amplitude', fontsize=8)
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
                cmap='RdBu_r',
                sensors=True
            )
            ax_topo.set_title(f"IC {self.component_idx} - Topomap", fontsize=9, color=self.theme['text'])
            self.topomap_figure.tight_layout(pad=0.3)
            
        except Exception as e:
            # Σε περίπτωση σφάλματος, εμφάνιση μηνύματος και στα δύο plots
            self.timeseries_figure.clear()
            ax_time = self.timeseries_figure.add_subplot(111) 
            ax_time.text(0.5, 0.5, f'Time series error: {e}', ha='center', va='center', color='red', fontsize=8)
            
            self.topomap_figure.clear()
            ax_topo = self.topomap_figure.add_subplot(111)
            ax_topo.text(0.5, 0.5, f'Topomap error: {e}', ha='center', va='center', color='red', fontsize=8)
        
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
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ border: none; background: {self.theme['background']}; width: 12px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: #bdc3c7; min-height: 20px; border-radius: 6px; }}
            QScrollBar::handle:vertical:hover {{ background: #95a5a6; }}
        """)
        
        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(0, 0, 5, 0)
        self.components_layout.setSpacing(10)
        self.scroll_area.setWidget(self.components_widget)
        main_layout.addWidget(self.scroll_area, 1)

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
        
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {self.theme['success']}; color: white; border-radius: 8px; }}
            QPushButton:hover {{ background-color: {self.theme['success_hover']}; }}
        """)

    def set_ica_data(self, ica, raw, suggested_artifacts, **kwargs):
        # ... (Η συνάρτηση παραμένει ίδια)
        self.ica = ica
        self.raw = raw
        self.suggested_artifacts = suggested_artifacts
        while self.components_layout.count():
            item = self.components_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.checkboxes.clear()
        self.component_widgets.clear()
        for i in range(self.ica.n_components_):
            self.create_single_component_widget(i)
        self.components_layout.addStretch(1)

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
        checkbox.toggled.connect(lambda state, widget=comp_container: self.update_selection_style(widget, state))
        self.checkboxes[i] = checkbox

        # Το νέο κουμπί "Ανάλυση"
        details_btn = QPushButton("🔎 Ανάλυση")
        details_btn.setStyleSheet(f"""
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
        """)
        details_btn.clicked.connect(lambda state, idx=i: self.show_component_properties(idx))  # Σύνδεση με τη νέα συνάρτηση

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
            border_color = self.theme['danger']
        else:
            bg_color = "#e8f8f5"
            border_color = self.theme['success']
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 5px;
            }}
        """)
            
    def set_all_checkboxes(self, state: bool):
        # ... (Η συνάρτηση παραμένει ίδια)
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)

    def select_suggested(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        for i, checkbox in self.checkboxes.items():
            checkbox.setChecked(i in self.suggested_artifacts)
            
    def emit_selected_components(self):
        # ... (Η συνάρτηση παραμένει ίδια)
        selected = [i for i, cb in self.checkboxes.items() if cb.isChecked()]
        self.components_selected.emit(selected)

    def show_component_properties(self, component_idx):
        """
        Δημιουργεί και εμφανίζει ένα νέο παράθυρο με τις ιδιότητες της συνιστώσας.
        """
        if not self.ica or not self.raw:
            return

        # Το MNE δημιουργεί τα γραφήματα. Το show=False είναι κρίσιμο
        # για να πάρουμε τα figures αντί να τα εμφανίσει μόνο του.
        figures = self.ica.plot_properties(self.raw, picks=component_idx, show=False)
        
        # Δημιουργούμε ένα νέο παράθυρο διαλόγου (pop-up)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Detailed Analysis of Component IC {component_idx}")
        dialog.setMinimumSize(800, 600)
        dialog_layout = QVBoxLayout(dialog)
        
        # Για κάθε figure που έφτιαξε το MNE, δημιουργούμε έναν καμβά και τον
        # προσθέτουμε στο παράθυρο.
        for fig in figures:
            canvas = FigureCanvas(fig)
            dialog_layout.addWidget(canvas)
        
        # Εμφανίζουμε το παράθυρο
        dialog.exec()
