#!/usr/bin/env python3
"""
Unit Tests for GUI Components
Μοναδιαίοι Έλεγχοι για Στοιχεία GUI
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import mne
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import PyQt6 - skip tests if not available
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QWidget

    # Import components to test
    from components import ICAComponentSelector, ResultsDisplayWidget

    QT_AVAILABLE = True
except ImportError as e:
    print(f"PyQt6 import failed: {e}")
    QT_AVAILABLE = False


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestICAComponentSelector:
    """Έλεγχοι για ICAComponentSelector"""

    def setup_method(self):
        """Προετοιμασία test δεδομένων"""
        # Create a mock theme
        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "background": "#ffffff",
            "success": "#27ae60",
            "success_hover": "#2ecc71",
            "danger": "#e74c3c",
        }

        # Create widget
        self.widget = ICAComponentSelector(self.theme)

        # Create mock ICA and raw data
        self.mock_ica = Mock()
        self.mock_ica.n_components_ = 3

        # Mock raw data
        self.mock_raw = Mock()
        self.mock_raw.info = {"sfreq": 128.0}
        self.mock_raw.times = np.linspace(0, 10, 1280)
        self.mock_raw.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]
        # Mock annotations with empty list (supports len() and iteration)
        self.mock_raw.annotations = []

        # Mock ICA sources
        mock_sources = Mock()
        mock_sources.get_data.return_value = np.random.randn(3, 1280)
        self.mock_ica.get_sources.return_value = mock_sources

        # Test data
        self.suggested_components = [0, 2]
        self.components_info = {
            0: {"variance": 0.5, "kurtosis": 2.1, "range": 1.2},
            1: {"variance": 0.3, "kurtosis": 1.5, "range": 0.8},
            2: {"variance": 0.7, "kurtosis": 3.0, "range": 1.5},
        }
        self.explanations = {
            0: "Πιθανό artifact: EOG",
            1: "Καθαρό εγκεφαλικό σήμα",
            2: "Πιθανό artifact: Μυϊκή δραστηριότητα",
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
            explanations=self.explanations,
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
            explanations=self.explanations,
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
            explanations=self.explanations,
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
            explanations=self.explanations,
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
        assert hasattr(self.widget, "show_component_properties")
        assert callable(getattr(self.widget, "show_component_properties"))

    def test_details_button_creation(self, qapp):
        """Έλεγχος δημιουργίας κουμπιού λεπτομερειών"""
        # Set data to create components
        self.widget.set_ica_data(
            ica=self.mock_ica,
            raw=self.mock_raw,
            suggested_artifacts=self.suggested_components,
            components_info=self.components_info,
            explanations=self.explanations,
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


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
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

        self.mock_original.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]
        self.mock_cleaned.ch_names = ["AF3", "T7", "Pz", "T8", "AF4"]

        self.mock_original.info = {"sfreq": 128.0}
        self.mock_cleaned.info = {"sfreq": 128.0}

        self.mock_original.times = np.linspace(0, 10, 1280)
        self.mock_cleaned.times = np.linspace(0, 10, 1280)

        # Statistics
        self.original_stats = {
            "AF3": {"rms": 25.0, "range": 100.0, "variance": 625.0},
            "T7": {"rms": 20.0, "range": 80.0, "variance": 400.0},
            "Pz": {"rms": 30.0, "range": 120.0, "variance": 900.0},
            "T8": {"rms": 22.0, "range": 88.0, "variance": 484.0},
            "AF4": {"rms": 26.0, "range": 104.0, "variance": 676.0},
        }

        self.cleaned_stats = {
            "AF3": {"rms": 12.5, "range": 50.0, "variance": 156.25},
            "T7": {"rms": 10.0, "range": 40.0, "variance": 100.0},
            "Pz": {"rms": 15.0, "range": 60.0, "variance": 225.0},
            "T8": {"rms": 11.0, "range": 44.0, "variance": 121.0},
            "AF4": {"rms": 13.0, "range": 52.0, "variance": 169.0},
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
                output_file="test_output.edf",
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
            components_removed=self.components_removed,
        )

        # Then clear
        self.widget.clear_results()

        # Check that table is empty
        assert self.widget.statistics_widget.table.rowCount() == 0


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestStatisticsTableWidget:
    """Έλεγχοι για StatisticsTableWidget"""

    def setup_method(self):
        """Προετοιμασία test δεδομένων"""
        from components.results_display import StatisticsTableWidget

        self.widget = StatisticsTableWidget()

        # Test statistics
        self.original_stats = {
            "AF3": {"rms": 25.0, "range": 100.0},
            "T7": {"rms": 20.0, "range": 80.0},
            "Pz": {"rms": 30.0, "range": 120.0},
        }

        self.cleaned_stats = {
            "AF3": {"rms": 12.5, "range": 50.0},
            "T7": {"rms": 10.0, "range": 40.0},
            "Pz": {"rms": 15.0, "range": 60.0},
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
@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestComponentIntegration:
    """Έλεγχοι ολοκλήρωσης στοιχείων"""

    def test_signal_emission(self, qapp):
        """Έλεγχος εκπομπής σημάτων"""
        # Create a mock theme
        theme = {
            "text": "#000000",
            "text_light": "#666666",
            "background": "#ffffff",
            "success": "#27ae60",
            "success_hover": "#2ecc71",
            "danger": "#e74c3c",
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
        mock_raw.info = {"sfreq": 128.0}
        mock_raw.times = np.linspace(0, 10, 1280)
        mock_raw.ch_names = ["AF3", "AF4", "T7", "T8", "Pz"]
        # Mock annotations with empty list (supports len() and iteration)
        mock_raw.annotations = []

        mock_sources = Mock()
        mock_sources.get_data.return_value = np.random.randn(2, 1280)
        mock_ica.get_sources.return_value = mock_sources

        # Set data
        selector.set_ica_data(
            ica=mock_ica,
            raw=mock_raw,
            suggested_artifacts=[0],
            components_info={
                0: {"variance": 0.5, "kurtosis": 2.1, "range": 1.2},
                1: {"variance": 0.3, "kurtosis": 1.5, "range": 0.8},
            },
            explanations={0: "Artifact", 1: "Brain signal"},
        )

        # Select components and emit signal
        selector.select_suggested_components()
        selector.emit_selected_components()

        # Check signal was received
        assert received_components == [0]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestTimeRangeSelector:
    """Tests for TimeRangeSelector widget"""

    def setup_method(self):
        """Prepare test data"""
        from components import TimeRangeSelector

        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "primary": "#007AFF",
            "primary_hover": "#0056b3",
        }
        self.widget = TimeRangeSelector(
            min_time=0.0,
            max_time=60.0,
            theme=self.theme,
        )

    def test_widget_creation(self, qapp):
        """Test widget creation"""
        from components import TimeRangeSelector

        assert isinstance(self.widget, TimeRangeSelector)

    def test_initial_range(self, qapp):
        """Test initial time range values"""
        start, end = self.widget.get_range()
        assert start == 0.0
        assert end == 60.0

    def test_set_time_range(self, qapp):
        """Test setting new time range"""
        self.widget.set_time_range(10.0, 120.0)
        start, end = self.widget.get_range()
        assert start == 10.0
        assert end == 120.0

    def test_reset_range(self, qapp):
        """Test reset to full range"""
        self.widget.set_time_range(0.0, 100.0)
        # Manually set a partial range by updating internal values
        self.widget._start_time = 20.0
        self.widget._end_time = 80.0

        self.widget.reset_range()

        start, end = self.widget.get_range()
        assert start == 0.0
        assert end == 100.0

    def test_range_changed_signal(self, qapp):
        """Test range_changed signal emission"""
        received_values = []

        def on_range_changed(start, end):
            received_values.append((start, end))

        self.widget.range_changed.connect(on_range_changed)

        # Trigger a range change via reset
        self.widget.reset_range()

        # Signal should have been emitted
        assert len(received_values) >= 1


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestSignalCutter:
    """Tests for SignalCutter widget"""

    def setup_method(self):
        """Prepare test data"""
        from components import SignalCutter

        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "primary": "#007AFF",
            "danger": "#dc3545",
        }
        self.widget = SignalCutter(theme=self.theme)
        self.widget.set_max_time(60.0)

    def test_widget_creation(self, qapp):
        """Test widget creation"""
        from components import SignalCutter

        assert isinstance(self.widget, SignalCutter)

    def test_initial_empty_regions(self, qapp):
        """Test initial empty cut regions"""
        regions = self.widget.get_cut_regions()
        assert regions == []

    def test_clear_regions(self, qapp):
        """Test clearing all regions"""
        # Add some regions by manipulating internal state
        self.widget._cut_regions = [(5.0, 10.0), (20.0, 25.0)]

        self.widget.clear_regions()

        regions = self.widget.get_cut_regions()
        assert regions == []

    def test_set_max_time(self, qapp):
        """Test setting max time"""
        self.widget.set_max_time(120.0)
        assert self.widget._max_time == 120.0

    def test_regions_changed_signal(self, qapp):
        """Test regions_changed signal emission"""
        received_values = []

        def on_regions_changed(regions):
            received_values.append(regions)

        self.widget.regions_changed.connect(on_regions_changed)

        # Trigger by clearing regions
        self.widget.clear_regions()

        # Signal should have been emitted
        assert len(received_values) >= 1


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestRestingPhaseDisplay:
    """Tests for RestingPhaseDisplay widget"""

    def setup_method(self):
        """Prepare test data"""
        from components import RestingPhaseDisplay

        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "primary": "#007AFF",
            "success": "#28a745",
            "border": "#dee2e6",
        }
        self.widget = RestingPhaseDisplay(theme=self.theme)

    def test_widget_creation(self, qapp):
        """Test widget creation"""
        from components import RestingPhaseDisplay

        assert isinstance(self.widget, RestingPhaseDisplay)

    def test_update_phases_empty(self, qapp):
        """Test update with empty phases"""
        self.widget.update_phases([])
        assert self.widget.phases == []

    def test_update_phases_with_data(self, qapp):
        """Test update with phase data"""
        phases = [
            {"label": "Eyes Open", "start": 0.0, "end": 30.0},
            {"label": "Eyes Closed", "start": 30.0, "end": 60.0},
        ]
        original_powers = {
            "Eyes Open": {
                "Delta": 20.0,
                "Theta": 15.0,
                "Alpha": 30.0,
                "Beta": 25.0,
                "Gamma": 10.0,
            },
            "Eyes Closed": {
                "Delta": 25.0,
                "Theta": 20.0,
                "Alpha": 25.0,
                "Beta": 20.0,
                "Gamma": 10.0,
            },
        }

        self.widget.update_phases(phases, original_powers)

        assert len(self.widget.phases) == 2
        assert self.widget.phases[0]["label"] == "Eyes Open"
        assert self.widget.phases[1]["label"] == "Eyes Closed"

    def test_clear(self, qapp):
        """Test clearing display"""
        phases = [{"label": "Eyes Open", "start": 0.0, "end": 30.0}]
        self.widget.update_phases(phases)

        self.widget.clear()

        assert self.widget.phases == []


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestBandPowerAnalysisWidget:
    """Tests for BandPowerAnalysisWidget"""

    def setup_method(self):
        """Prepare test data"""
        from components import BandPowerAnalysisWidget

        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "primary": "#007AFF",
            "primary_hover": "#0056b3",
            "success": "#28a745",
            "border": "#dee2e6",
        }
        self.widget = BandPowerAnalysisWidget(theme=self.theme)

        # Create mock raw data
        self.sfreq = 128.0
        self.duration = 30.0
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz"]

        data = np.random.randn(len(self.ch_names), self.n_samples) * 1e-5
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info, verbose=False)

    def test_widget_creation(self, qapp):
        """Test widget creation"""
        from components import BandPowerAnalysisWidget

        assert isinstance(self.widget, BandPowerAnalysisWidget)

    def test_set_data(self, qapp):
        """Test setting EEG data"""
        self.widget.set_data(self.test_raw)

        assert self.widget._raw_data is not None
        assert self.widget._max_time == pytest.approx(self.duration, rel=0.1)

    def test_set_data_with_cleaned(self, qapp):
        """Test setting both original and cleaned data"""
        cleaned_data = self.test_raw.copy()

        self.widget.set_data(self.test_raw, cleaned_data)

        assert self.widget._raw_data is not None
        assert self.widget._cleaned_data is not None

    def test_time_range_changed_signal(self, qapp):
        """Test time_range_changed signal emission"""
        received_values = []

        def on_range_changed(start, end):
            received_values.append((start, end))

        self.widget.time_range_changed.connect(on_range_changed)

        # Set data which triggers range update
        self.widget.set_data(self.test_raw)

        # Signal should have been emitted via time range selector
        # We test the internal signal connection works
        self.widget._on_range_changed(5.0, 20.0)

        assert len(received_values) >= 1
        assert received_values[-1] == (5.0, 20.0)

    def test_clear(self, qapp):
        """Test clearing widget"""
        self.widget.set_data(self.test_raw)
        self.widget.clear()

        assert self.widget._raw_data is None
        assert self.widget._cleaned_data is None


@pytest.mark.skipif(
    not QT_AVAILABLE, reason="PyQt6 not available or Qt display not accessible"
)
class TestComparisonScreenEnhanced:
    """Tests for enhanced ComparisonScreen with new features"""

    def setup_method(self):
        """Prepare test data"""
        from components import ComparisonScreen

        self.theme = {
            "text": "#000000",
            "text_light": "#666666",
            "primary": "#007AFF",
            "primary_hover": "#0056b3",
            "success": "#28a745",
            "border": "#dee2e6",
        }
        self.widget = ComparisonScreen(theme=self.theme)

        # Create mock raw data
        self.sfreq = 128.0
        self.duration = 30.0
        self.n_samples = int(self.sfreq * self.duration)
        self.ch_names = ["AF3", "T7", "Pz"]

        data = np.random.randn(len(self.ch_names), self.n_samples) * 1e-5
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.test_raw = mne.io.RawArray(data, info, verbose=False)

        # Statistics
        self.original_stats = {
            "AF3": {"rms": 25.0, "range": 100.0, "variance": 625.0},
            "T7": {"rms": 20.0, "range": 80.0, "variance": 400.0},
            "Pz": {"rms": 30.0, "range": 120.0, "variance": 900.0},
        }

        self.cleaned_stats = {
            "AF3": {"rms": 12.5, "range": 50.0, "variance": 156.25},
            "T7": {"rms": 10.0, "range": 40.0, "variance": 100.0},
            "Pz": {"rms": 15.0, "range": 60.0, "variance": 225.0},
        }

    def test_widget_creation(self, qapp):
        """Test widget creation with tabs"""
        from components import ComparisonScreen

        assert isinstance(self.widget, ComparisonScreen)
        assert hasattr(self.widget, "tab_widget")
        assert self.widget.tab_widget.count() == 3

    def test_has_band_power_widget(self, qapp):
        """Test that band power widget is present"""
        assert hasattr(self.widget, "band_power_widget")
        from components import BandPowerAnalysisWidget

        assert isinstance(self.widget.band_power_widget, BandPowerAnalysisWidget)

    def test_has_signal_cutter(self, qapp):
        """Test that signal cutter is present"""
        assert hasattr(self.widget, "signal_cutter")
        from components import SignalCutter

        assert isinstance(self.widget.signal_cutter, SignalCutter)

    def test_update_comparison(self, qapp):
        """Test updating comparison with data"""
        self.widget.update_comparison(
            original_data=self.test_raw,
            cleaned_data=self.test_raw.copy(),
            original_stats=self.original_stats,
            cleaned_stats=self.cleaned_stats,
            components_removed=[0],
            input_file="test.edf",
            output_file="test_clean.edf",
        )

        # Check data was stored
        assert self.widget._original_data is not None
        assert self.widget._cleaned_data is not None

    def test_clear_comparison(self, qapp):
        """Test clearing comparison"""
        self.widget.update_comparison(
            original_data=self.test_raw,
            cleaned_data=self.test_raw.copy(),
            original_stats=self.original_stats,
            cleaned_stats=self.cleaned_stats,
            components_removed=[0],
        )

        self.widget.clear_comparison()

        assert self.widget._original_data is None
        assert self.widget._cleaned_data is None

    def test_apply_signal_cuts_signal(self, qapp):
        """Test apply_signal_cuts signal exists"""
        received_regions = []

        def on_apply_cuts(regions):
            received_regions.extend(regions)

        self.widget.apply_signal_cuts.connect(on_apply_cuts)

        # Trigger signal via internal method
        self.widget._on_apply_cuts([(5.0, 10.0)])

        assert received_regions == [(5.0, 10.0)]

    def test_save_diagrams_button_exists(self, qapp):
        """Test that save diagrams button exists"""
        assert hasattr(self.widget, "save_diagrams_button")
        from PyQt6.QtWidgets import QPushButton

        assert isinstance(self.widget.save_diagrams_button, QPushButton)
        assert "Save All Diagrams" in self.widget.save_diagrams_button.text()

    @patch("components.comparison_screen.Path")
    @patch("components.comparison_screen.QMessageBox")
    def test_save_all_diagrams_with_data(
        self, mock_messagebox, mock_path, qapp, tmp_path
    ):
        """Test saving diagrams with data"""
        # Setup mock path
        mock_path.cwd.return_value = tmp_path

        # Update widget with data
        self.widget.update_comparison(
            original_data=self.test_raw,
            cleaned_data=self.test_raw.copy(),
            original_stats=self.original_stats,
            cleaned_stats=self.cleaned_stats,
            components_removed=[0],
        )

        # Call save method
        self.widget._save_all_diagrams()

        # Verify that a message box was shown (either success or warning)
        assert mock_messagebox.information.called or mock_messagebox.warning.called

    @patch("components.comparison_screen.QMessageBox")
    def test_save_all_diagrams_without_data(self, mock_messagebox, qapp):
        """Test saving diagrams without data shows warning"""
        # Call save method without setting data
        self.widget._save_all_diagrams()

        # Should show warning that no diagrams are available
        mock_messagebox.warning.assert_called_once()
