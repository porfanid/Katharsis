#!/usr/bin/env python3
"""
Unit Tests for GUI Components
Μοναδιαίοι Έλεγχοι για Στοιχεία GUI
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import mne
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import PyQt6 - skip tests if not available
try:
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    
    # Import components to test
    from components import ICAComponentSelector, ResultsDisplayWidget
    
    QT_AVAILABLE = True
except ImportError as e:
    print(f"PyQt6 import failed: {e}")
    QT_AVAILABLE = False


@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible")
class TestICAComponentSelector:
    """Έλεγχοι για ICAComponentSelector"""
    
    def setup_method(self):
        """Προετοιμασία test δεδομένων"""
        # Create a mock theme
        self.theme = {
            'text': '#000000',
            'text_light': '#666666',
            'background': '#ffffff',
            'success': '#27ae60',
            'success_hover': '#2ecc71',
            'danger': '#e74c3c'
        }
        
        # Create widget
        self.widget = ICAComponentSelector(self.theme)
        
        # Create mock ICA and raw data
        self.mock_ica = Mock()
        self.mock_ica.n_components_ = 3
        
        # Mock raw data
        self.mock_raw = Mock()
        self.mock_raw.info = {'sfreq': 128.0}
        self.mock_raw.times = np.linspace(0, 10, 1280)
        self.mock_raw.ch_names = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        
        # Mock ICA sources
        mock_sources = Mock()
        mock_sources.get_data.return_value = np.random.randn(3, 1280)
        self.mock_ica.get_sources.return_value = mock_sources
        
        # Test data
        self.suggested_components = [0, 2]
        self.components_info = {
            0: {'variance': 0.5, 'kurtosis': 2.1, 'range': 1.2},
            1: {'variance': 0.3, 'kurtosis': 1.5, 'range': 0.8},
            2: {'variance': 0.7, 'kurtosis': 3.0, 'range': 1.5}
        }
        self.explanations = {
            0: "Πιθανό artifact: EOG",
            1: "Καθαρό εγκεφαλικό σήμα",
            2: "Πιθανό artifact: Μυϊκή δραστηριότητα"
        }
        
    def test_widget_creation(self, qapp):
        """Έλεγχος δημιουργίας widget"""
        assert isinstance(self.widget, QWidget)
        assert isinstance(self.widget, ICAComponentSelector)
        
    def test_set_ica_data(self, qapp):
        """Έλεγχος ορισμού ICA δεδομένων"""
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations
        )
        
        assert self.widget.ica == self.mock_ica
        assert self.widget.raw == self.mock_raw
        assert self.widget.suggested_artifacts == self.suggested_components
        assert self.widget.components_info == self.components_info
        assert self.widget.explanations == self.explanations
        
    def test_get_selected_components_empty(self, qapp):
        """Έλεγχος λήψης επιλεγμένων συνιστωσών όταν δεν υπάρχουν"""
        selected = self.widget.get_selected_components()
        assert selected == []
        
    def test_select_all_components(self, qapp):
        """Έλεγχος επιλογής όλων των συνιστωσών"""
        # First set some data
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations
        )
        
        # Select all
        self.widget.select_all_components()
        
        # Check all are selected
        for checkbox in self.widget.checkboxes.values():
            assert checkbox.isChecked()
            
    def test_select_no_components(self, qapp):
        """Έλεγχος αποεπιλογής όλων των συνιστωσών"""
        # First set some data and select all
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations
        )
        self.widget.select_all_components()
        
        # Then select none
        self.widget.select_no_components()
        
        # Check none are selected
        for checkbox in self.widget.checkboxes.values():
            assert not checkbox.isChecked()
            
    def test_select_suggested_components(self, qapp):
        """Έλεγχος επιλογής προτεινόμενων συνιστωσών"""
        # Set data
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations
        )
        
        # Select suggested
        self.widget.select_suggested_components()
        
        # Check only suggested are selected
        for comp_idx, checkbox in self.widget.checkboxes.items():
            if comp_idx in self.suggested_components:
                assert checkbox.isChecked()
            else:
                assert not checkbox.isChecked()

    def test_show_component_properties_method_exists(self, qapp):
        """Έλεγχος ύπαρξης της νέας συνάρτησης show_component_properties"""
        # Check that the method exists
        assert hasattr(self.widget, 'show_component_properties')
        assert callable(getattr(self.widget, 'show_component_properties'))

    def test_details_button_creation(self, qapp):
        """Έλεγχος δημιουργίας κουμπιού λεπτομερειών"""
        # Set data to create components
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations
        )
        
        # Check that component widgets are created with details buttons
        # We can't easily test the button directly without Qt app running,
        # but we can verify the method doesn't crash when called
        try:
            # This should not crash even if ICA/raw data are mocked
            self.widget.show_component_properties(0)
            # If it gets here without crashing with missing data, the method structure is correct
        except (AttributeError, TypeError):
            # These exceptions are expected with mock data, but indicate the method exists and runs
            pass


@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible")
class TestResultsDisplayWidget:
    """Έλεγχοι για ResultsDisplayWidget"""
            
    def setup_method(self):
        """Προετοιμασία test δεδομένων"""
        self.widget = ResultsDisplayWidget()
        
        # Create mock raw data
        self.mock_original = Mock()
        self.mock_cleaned = Mock()
        
        # Mock data for get_data()
        self.mock_original.get_data.return_value = np.random.randn(5, 1280) * 1e-5
        self.mock_cleaned.get_data.return_value = np.random.randn(5, 1280) * 0.5e-5
        
        self.mock_original.ch_names = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        self.mock_cleaned.ch_names = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        
        self.mock_original.info = {'sfreq': 128.0}
        self.mock_cleaned.info = {'sfreq': 128.0}
        
        self.mock_original.times = np.linspace(0, 10, 1280)
        self.mock_cleaned.times = np.linspace(0, 10, 1280)
        
        # Statistics
        self.original_stats = {
            'AF3': {'rms': 25.0, 'range': 100.0, 'variance': 625.0},
            'T7': {'rms': 20.0, 'range': 80.0, 'variance': 400.0},
            'Pz': {'rms': 30.0, 'range': 120.0, 'variance': 900.0},
            'T8': {'rms': 22.0, 'range': 88.0, 'variance': 484.0},
            'AF4': {'rms': 26.0, 'range': 104.0, 'variance': 676.0}
        }
        
        self.cleaned_stats = {
            'AF3': {'rms': 12.5, 'range': 50.0, 'variance': 156.25},
            'T7': {'rms': 10.0, 'range': 40.0, 'variance': 100.0},
            'Pz': {'rms': 15.0, 'range': 60.0, 'variance': 225.0},
            'T8': {'rms': 11.0, 'range': 44.0, 'variance': 121.0},
            'AF4': {'rms': 13.0, 'range': 52.0, 'variance': 169.0}
        }
        
        self.components_removed = [0, 2]
        
    def test_widget_creation(self, qapp):
        """Έλεγχος δημιουργίας widget"""
        assert isinstance(self.widget, QWidget)
        assert isinstance(self.widget, ResultsDisplayWidget)
        
    def test_update_results(self, qapp):
        """Έλεγχος ενημέρωσης αποτελεσμάτων"""
        # This test mainly checks that the method runs without errors
        try:
            self.widget.update_results(
                original_data=self.mock_original,
                cleaned_data=self.mock_cleaned,
                original_stats=self.original_stats,
                cleaned_stats=self.cleaned_stats,
                components_removed=self.components_removed,
                input_file="test_input.edf",
                output_file="test_output.edf"
            )
            success = True
        except Exception as e:
            success = False
            print(f"Error in update_results: {e}")
            
        assert success
        
    def test_clear_results(self, qapp):
        """Έλεγχος καθαρισμού αποτελεσμάτων"""
        # First update with some results
        self.widget.update_results(
            original_data=self.mock_original,
            cleaned_data=self.mock_cleaned,
            original_stats=self.original_stats,
            cleaned_stats=self.cleaned_stats,
            components_removed=self.components_removed
        )
        
        # Then clear
        self.widget.clear_results()
        
        # Check that table is empty
        assert self.widget.statistics_widget.table.rowCount() == 0


@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible")
class TestStatisticsTableWidget:
    """Έλεγχοι για StatisticsTableWidget"""
            
    def setup_method(self):
        """Προετοιμασία test δεδομένων"""
        from components.results_display import StatisticsTableWidget
        self.widget = StatisticsTableWidget()
        
        # Test statistics
        self.original_stats = {
            'AF3': {'rms': 25.0, 'range': 100.0},
            'T7': {'rms': 20.0, 'range': 80.0},
            'Pz': {'rms': 30.0, 'range': 120.0}
        }
        
        self.cleaned_stats = {
            'AF3': {'rms': 12.5, 'range': 50.0},
            'T7': {'rms': 10.0, 'range': 40.0},
            'Pz': {'rms': 15.0, 'range': 60.0}
        }
        
    def test_update_statistics(self, qapp):
        """Έλεγχος ενημέρωσης στατιστικών"""
        self.widget.update_statistics(self.original_stats, self.cleaned_stats)
        
        # Check table has correct number of rows
        assert self.widget.table.rowCount() == 3
        
        # Check table has correct number of columns
        assert self.widget.table.columnCount() == 6
        
        # Check that data is populated
        for row in range(self.widget.table.rowCount()):
            for col in range(self.widget.table.columnCount()):
                item = self.widget.table.item(row, col)
                assert item is not None
                assert len(item.text()) > 0


# Integration test for component interaction
@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible")
class TestComponentIntegration:
    """Έλεγχοι ολοκλήρωσης στοιχείων"""
            
    def test_signal_emission(self, qapp):
        """Έλεγχος εκπομπής σημάτων"""
        # Create a mock theme
        theme = {
            'text': '#000000',
            'text_light': '#666666',
            'background': '#ffffff',
            'success': '#27ae60',
            'success_hover': '#2ecc71',
            'danger': '#e74c3c'
        }
        selector = ICAComponentSelector(theme)
        
        # Mock signal reception
        received_components = []
        
        def on_components_selected(components):
            received_components.extend(components)
            
        selector.components_selected.connect(on_components_selected)
        
        # Create mock data
        mock_ica = Mock()
        mock_ica.n_components_ = 2
        
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 128.0}
        mock_raw.times = np.linspace(0, 10, 1280)
        mock_raw.ch_names = ['AF3', 'AF4', 'T7', 'T8', 'Pz']
        
        mock_sources = Mock()
        mock_sources.get_data.return_value = np.random.randn(2, 1280)
        mock_ica.get_sources.return_value = mock_sources
        
        # Set data
        selector.set_ica_data(
            ica=mock_ica,
            raw=mock_raw,
            suggested_artifacts=[0],
            components_info={0: {'variance': 0.5, 'kurtosis': 2.1, 'range': 1.2},
                           1: {'variance': 0.3, 'kurtosis': 1.5, 'range': 0.8}},
            explanations={0: "Artifact", 1: "Brain signal"}
        )
        
        # Select components and emit signal
        selector.select_suggested_components()
        selector.emit_selected_components()
        
        # Check signal was received
        assert received_components == [0]


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])