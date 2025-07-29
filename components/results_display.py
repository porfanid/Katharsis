#!/usr/bin/env python3
"""
Results Display Widget - Εμφάνιση αποτελεσμάτων καθαρισμού
Results Display Widget - Display cleaning results and statistics
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QGroupBox,
    QGridLayout,
    QHeaderView,
    QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Any, Optional
import mne


class StatisticsTableWidget(QWidget):
    """Widget για εμφάνιση στατιστικών πινάκων"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Δημιουργία UI"""
        layout = QVBoxLayout(self)

        # Τίτλος
        title_label = QLabel("📊 Στατιστικά Αποτελέσματα Καθαρισμού")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)

        # Πίνακας στατιστικών
        self.table = QTableWidget()
        self.table.setStyleSheet(
            """
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                gridline-color: #ecf0f1;
            }
            QTableWidget::item {
                padding: 8px;
                color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """
        )

        layout.addWidget(self.table)

    def update_statistics(
        self,
        original_stats: Dict[str, Dict[str, float]],
        cleaned_stats: Dict[str, Dict[str, float]],
    ):
        """
        Ενημέρωση πίνακα στατιστικών

        Args:
            original_stats: Στατιστικά αρχικών δεδομένων
            cleaned_stats: Στατιστικά καθαρισμένων δεδομένων
        """
        channels = list(original_stats.keys())

        # Ορισμός στηλών
        headers = [
            "Κανάλι",
            "Αρχικό RMS (μV)",
            "Καθαρό RMS (μV)",
            "Μείωση (%)",
            "Αρχικό Εύρος",
            "Καθαρό Εύρος",
        ]

        self.table.setRowCount(len(channels))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Συμπλήρωση δεδομένων
        for row, channel in enumerate(channels):
            orig_stats = original_stats[channel]
            clean_stats = cleaned_stats[channel]

            # Υπολογισμός μείωσης θορύβου
            orig_rms = orig_stats["rms"]
            clean_rms = clean_stats["rms"]
            reduction = ((orig_rms - clean_rms) / orig_rms) * 100 if orig_rms > 0 else 0

            # Δεδομένα για κάθε στήλη
            row_data = [
                channel,
                f"{orig_rms:.1f}",
                f"{clean_rms:.1f}",
                f"{reduction:.1f}%",
                f"{orig_stats['range']:.1f}",
                f"{clean_stats['range']:.1f}",
            ]

            for col, data in enumerate(row_data):
                item = QTableWidgetItem(str(data))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Χρωματισμός βάσει μείωσης
                if col == 3:  # Στήλη μείωσης
                    if reduction > 50:
                        item.setBackground(QColor("#d5f4e6"))  # Πράσινο για καλή μείωση
                    elif reduction > 25:
                        item.setBackground(
                            QColor("#fff3cd")
                        )  # Κίτρινο για μέτρια μείωση
                    else:
                        item.setBackground(
                            QColor("#f8d7da")
                        )  # Κόκκινο για χαμηλή μείωση

                self.table.setItem(row, col, item)

        # Προσαρμογή στηλών
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class ComparisonPlotWidget(QWidget):
    """Widget για οπτικοποίηση σύγκρισης πριν/μετά"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Δημιουργία UI"""
        layout = QVBoxLayout(self)

        # Τίτλος
        title_label = QLabel("📈 Σύγκριση Σημάτων: Πριν vs Μετά τον Καθαρισμό")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_comparison(
        self,
        original_data: mne.io.Raw,
        cleaned_data: mne.io.Raw,
        time_window: float = 10.0,
    ):
        """
        Οπτικοποίηση σύγκρισης δεδομένων

        Args:
            original_data: Αρχικά δεδομένα
            cleaned_data: Καθαρισμένα δεδομένα
            time_window: Παράθυρο χρόνου για οπτικοποίηση (δευτερόλεπτα)
        """
        self.figure.clear()

        try:
            # Λήψη δεδομένων
            orig_data = original_data.get_data() * 1e6  # Μετατροπή σε μV
            clean_data = cleaned_data.get_data() * 1e6

            # Χρονικός άξονας
            times = original_data.times
            max_samples = int(time_window * original_data.info["sfreq"])
            display_times = times[:max_samples]

            # Κανάλια
            channels = original_data.ch_names
            n_channels = len(channels)

            # Δημιουργία subplots
            for i, channel in enumerate(channels):
                ax = self.figure.add_subplot(n_channels, 1, i + 1)

                # Δεδομένα για οπτικοποίηση
                orig_display = orig_data[i, :max_samples]
                clean_display = clean_data[i, :max_samples]

                # Plots
                ax.plot(
                    display_times,
                    orig_display,
                    color="#e74c3c",
                    alpha=0.7,
                    linewidth=1.5,
                    label="Αρχικό σήμα",
                )
                ax.plot(
                    display_times,
                    clean_display,
                    color="#27ae60",
                    alpha=0.8,
                    linewidth=1.5,
                    label="Καθαρό σήμα",
                )

                # Στυλ
                ax.set_title(
                    f"Κανάλι {channel}", fontsize=10, color="#2c3e50", fontweight="bold"
                )
                ax.set_xlabel(
                    "Χρόνος (s)" if i == n_channels - 1 else "",
                    fontsize=9,
                    color="#2c3e50",
                )
                ax.set_ylabel("Amplitude (μV)", fontsize=9, color="#2c3e50")
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8, colors="#2c3e50")

                # Legend μόνο στο πρώτο plot
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            self.figure.tight_layout()

        except Exception as e:
            # Error plot
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Σφάλμα οπτικοποίησης: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
            )
            ax.set_title("Σφάλμα Οπτικοποίησης")

        self.canvas.draw()


class ResultsDisplayWidget(QWidget):
    """Κεντρικό widget για εμφάνιση αποτελεσμάτων"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Δημιουργία UI"""
        layout = QVBoxLayout(self)

        # Header
        header_label = QLabel("🎯 Αποτελέσματα Καθαρισμού EEG Artifacts")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_label.setStyleSheet(
            """
            QLabel {
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                margin: 10px;
            }
        """
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Statistics table
        self.statistics_widget = StatisticsTableWidget()
        layout.addWidget(self.statistics_widget)

        # Comparison plot
        self.comparison_widget = ComparisonPlotWidget()
        layout.addWidget(self.comparison_widget)

        # Summary section
        self.summary_widget = self.create_summary_widget()
        layout.addWidget(self.summary_widget)

    def create_summary_widget(self) -> QWidget:
        """Δημιουργία widget περίληψης"""
        group_box = QGroupBox("📋 Περίληψη Επεξεργασίας")
        group_box.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        group_box.setStyleSheet(
            """
            QGroupBox {
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """
        )

        layout = QGridLayout(group_box)

        # Labels για περίληψη
        self.file_label = QLabel("Αρχείο: -")
        self.components_label = QLabel("Συνιστώσες που αφαιρέθηκαν: -")
        self.avg_reduction_label = QLabel("Μέση μείωση θορύβου: -")
        self.status_label = QLabel("Κατάσταση: -")

        labels = [
            self.file_label,
            self.components_label,
            self.avg_reduction_label,
            self.status_label,
        ]

        for i, label in enumerate(labels):
            label.setFont(QFont("Arial", 10))
            label.setStyleSheet("color: #34495e; margin: 5px;")
            layout.addWidget(label, i, 0)

        return group_box

    def update_results(
        self,
        original_data: mne.io.Raw,
        cleaned_data: mne.io.Raw,
        original_stats: Dict[str, Dict[str, float]],
        cleaned_stats: Dict[str, Dict[str, float]],
        components_removed: List[int],
        input_file: str = "",
        output_file: str = "",
    ):
        """
        Ενημέρωση εμφάνισης αποτελεσμάτων

        Args:
            original_data: Αρχικά δεδομένα
            cleaned_data: Καθαρισμένα δεδομένα
            original_stats: Στατιστικά αρχικών δεδομένων
            cleaned_stats: Στατιστικά καθαρισμένων δεδομένων
            components_removed: Συνιστώσες που αφαιρέθηκαν
            input_file: Αρχείο εισόδου
            output_file: Αρχείο εξόδου
        """
        try:
            # Ενημέρωση στατιστικών
            self.statistics_widget.update_statistics(original_stats, cleaned_stats)

            # Ενημέρωση οπτικοποίησης
            self.comparison_widget.plot_comparison(original_data, cleaned_data)

            # Υπολογισμός μέσης μείωσης
            total_reduction = 0
            channels = list(original_stats.keys())

            for channel in channels:
                orig_rms = original_stats[channel]["rms"]
                clean_rms = cleaned_stats[channel]["rms"]
                if orig_rms > 0:
                    reduction = ((orig_rms - clean_rms) / orig_rms) * 100
                    total_reduction += reduction

            avg_reduction = total_reduction / len(channels) if channels else 0

            # Ενημέρωση περίληψης
            import os

            filename = os.path.basename(input_file) if input_file else "Άγνωστο"

            self.file_label.setText(f"Αρχείο: {filename}")
            self.components_label.setText(
                f"Συνιστώσες που αφαιρέθηκαν: {components_removed}"
            )
            self.avg_reduction_label.setText(
                f"Μέση μείωση θορύβου: {avg_reduction:.1f}%"
            )
            self.status_label.setText("Κατάσταση: ✅ Καθαρισμός ολοκληρώθηκε επιτυχώς")

            # Χρωματισμός status βάσει αποτελέσματος
            if avg_reduction > 50:
                color = "#27ae60"  # Πράσινο για εξαιρετικό αποτέλεσμα
            elif avg_reduction > 25:
                color = "#f39c12"  # Πορτοκαλί για καλό αποτέλεσμα
            else:
                color = "#e74c3c"  # Κόκκινο για χαμηλό αποτέλεσμα

            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        except Exception as e:
            # Εμφάνιση σφάλματος
            self.status_label.setText(f"Κατάσταση: ❌ Σφάλμα: {str(e)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def clear_results(self):
        """Καθαρισμός εμφάνισης αποτελεσμάτων"""
        self.statistics_widget.table.setRowCount(0)
        self.comparison_widget.figure.clear()
        self.comparison_widget.canvas.draw()

        self.file_label.setText("Αρχείο: -")
        self.components_label.setText("Συνιστώσες που αφαιρέθηκαν: -")
        self.avg_reduction_label.setText("Μέση μείωση θορύβου: -")
        self.status_label.setText("Κατάσταση: -")
        self.status_label.setStyleSheet("color: #34495e;")
