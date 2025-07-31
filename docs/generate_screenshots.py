#!/usr/bin/env python3
"""
Screenshot Generator for Katharsis User Guide
============================================

This script automatically generates screenshots of the Katharsis application
for the user guide documentation. It runs through the complete workflow
and captures images at each step.

Usage:
    python docs/generate_screenshots.py

Requirements:
    - PyQt6
    - All Katharsis dependencies
    - Test EEG data file
    - Display environment (for GUI screenshots)

Author: porfanid
Version: 1.0
"""

import sys
import os
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
import mne

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eeg_gui_app import EEGArtifactCleanerGUI
from backend import EEGArtifactCleaningService


class ScreenshotGenerator(QThread):
    """Thread for generating screenshots automatically"""
    
    screenshot_taken = pyqtSignal(str, str)  # filename, description
    
    def __init__(self, app, gui):
        super().__init__()
        self.app = app
        self.gui = gui
        self.screenshot_dir = project_root / "docs" / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Create test data if needed
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create or locate test EEG data"""
        test_data_path = project_root / "test_eeg_data.fif"
        
        if not test_data_path.exists():
            print("Creating test EEG data...")
            # Create synthetic EEG data for screenshots
            n_channels = 19
            sfreq = 250
            duration = 120  # 2 minutes
            
            ch_names = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T7', 'C3', 'Cz', 'C4', 'T8',
                'P7', 'P3', 'Pz', 'P4', 'P8',
                'O1', 'O2'
            ]
            
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            
            # Create realistic EEG data with artifacts
            import numpy as np
            np.random.seed(42)
            
            n_samples = int(sfreq * duration)
            data = np.zeros((n_channels, n_samples))
            times = np.arange(n_samples) / sfreq
            
            for ch_idx in range(n_channels):
                # Base EEG frequencies
                alpha = 0.5 * np.sin(2 * np.pi * 10 * times)
                beta = 0.3 * np.sin(2 * np.pi * 20 * times)
                theta = 0.4 * np.sin(2 * np.pi * 6 * times)
                noise = 0.2 * np.random.randn(n_samples)
                
                # Add artifacts for frontal channels (eye blinks)
                if 'Fp' in ch_names[ch_idx]:
                    blinks = np.zeros(n_samples)
                    blink_times = np.arange(10, duration-10, 3)  # Every 3 seconds
                    for blink_time in blink_times:
                        blink_idx = int(blink_time * sfreq)
                        if blink_idx < n_samples - 50:
                            blinks[blink_idx:blink_idx+50] = 5.0 * np.exp(-np.arange(50)/10)
                    data[ch_idx, :] = alpha + beta + theta + noise + blinks
                else:
                    data[ch_idx, :] = alpha + beta + theta + noise
            
            # Convert to microvolts
            data *= 1e-6
            
            raw = mne.io.RawArray(data, info)
            raw.save(test_data_path, overwrite=True)
            print(f"Test data created: {test_data_path}")
        
        self.test_data_path = test_data_path
    
    def take_screenshot(self, filename, description=""):
        """Take a screenshot of the current GUI state"""
        screenshot_path = self.screenshot_dir / filename
        
        # Take screenshot
        pixmap = self.gui.grab()
        success = pixmap.save(str(screenshot_path))
        
        if success:
            print(f"✓ Screenshot saved: {filename} - {description}")
            self.screenshot_taken.emit(filename, description)
        else:
            print(f"✗ Failed to save screenshot: {filename}")
    
    def run(self):
        """Generate all screenshots"""
        print("Starting screenshot generation...")
        
        # Wait for GUI to be fully loaded
        time.sleep(2)
        
        # 1. Welcome screen
        self.take_screenshot("02_welcome_screen.png", "Welcome screen with file selection")
        time.sleep(1)
        
        # Simulate file selection (this would normally open a dialog)
        # For automation, we'll directly load the test file
        if hasattr(self.gui, 'service') and self.gui.service:
            try:
                # Load test data
                self.gui.service.load_eeg_data(str(self.test_data_path))
                self.gui.current_input_file = str(self.test_data_path)
                
                # Move to channel selection
                QTimer.singleShot(1000, lambda: self.continue_screenshots())
                
            except Exception as e:
                print(f"Error loading test data: {e}")
    
    def continue_screenshots(self):
        """Continue with remaining screenshots"""
        try:
            # 2. Channel selection screen
            if self.gui.stacked_widget.currentIndex() != 1:
                self.gui.stacked_widget.setCurrentIndex(1)
            
            time.sleep(1)
            self.take_screenshot("03_channel_selection.png", "Channel selection interface")
            
            # Move to preprocessing
            if hasattr(self.gui, 'preprocessing_screen'):
                self.gui.stacked_widget.setCurrentIndex(2)
                time.sleep(1)
                
                # 3. Preprocessing overview
                self.take_screenshot("04_preprocessing_overview.png", "Advanced preprocessing interface")
                
                # Take screenshots of individual tabs
                if hasattr(self.gui.preprocessing_screen, 'widget') and hasattr(self.gui.preprocessing_screen.widget, 'tab_widget'):
                    tab_widget = self.gui.preprocessing_screen.widget.tab_widget
                    
                    # Tab 1: Channel Analysis
                    tab_widget.setCurrentIndex(0)
                    time.sleep(0.5)
                    self.take_screenshot("05_channel_analysis.png", "Channel analysis tab")
                    
                    # Tab 2: Filtering
                    tab_widget.setCurrentIndex(1)
                    time.sleep(0.5)
                    self.take_screenshot("06_filtering.png", "Filtering configuration tab")
                    
                    # Tab 3: Re-referencing
                    tab_widget.setCurrentIndex(2)
                    time.sleep(0.5)
                    self.take_screenshot("07_referencing.png", "Re-referencing options tab")
                    
                    # Tab 4: Pipeline
                    tab_widget.setCurrentIndex(3)
                    time.sleep(0.5)
                    self.take_screenshot("08_pipeline.png", "Pipeline configuration tab")
            
            print("✅ Screenshot generation completed!")
            print(f"📁 Screenshots saved to: {self.screenshot_dir}")
            
        except Exception as e:
            print(f"Error during screenshot generation: {e}")
            import traceback
            traceback.print_exc()


def create_placeholder_screenshots():
    """Create placeholder images for screenshots that require user interaction"""
    from PyQt6.QtGui import QPainter, QFont, QColor
    from PyQt6.QtCore import Qt
    
    screenshot_dir = project_root / "docs" / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    
    # Create placeholder images for interactive screenshots
    placeholders = [
        ("01_splash_screen.png", "Loading Splash Screen", "Katharsis initializing..."),
        ("09_processing.png", "Processing Progress", "Preprocessing pipeline running..."),
        ("10_enhanced_ica.png", "Enhanced ICA Analysis", "ICA component analysis interface"),
        ("11_artifact_classification.png", "Artifact Classification", "Automatic artifact detection results"),
        ("12_ica_components.png", "ICA Components", "Individual component visualization"),
        ("13_component_selection.png", "Component Selection", "User selection of artifacts to remove"),
        ("14_results_comparison.png", "Results Comparison", "Before/after comparison of cleaned data"),
        ("15_data_export.png", "Data Export", "Export options for processed data")
    ]
    
    for filename, title, description in placeholders:
        filepath = screenshot_dir / filename
        
        # Create a placeholder image
        pixmap = QPixmap(800, 600)
        pixmap.fill(QColor(240, 240, 240))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 100, 100))
        
        # Title
        font = QFont("Arial", 18, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                        f"\n\n{title}\n")
        
        # Description
        font = QFont("Arial", 12)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter,
                        f"{description}\n\n[Screenshot will be generated during actual usage]")
        
        painter.end()
        pixmap.save(str(filepath))
        print(f"✓ Placeholder created: {filename}")


def main():
    """Main function to generate screenshots"""
    print("Katharsis Screenshot Generator")
    print("==============================")
    
    # Check if running in headless environment
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        print("⚠️ Running in headless mode - creating placeholder screenshots only")
        create_placeholder_screenshots()
        return
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    try:
        # Create GUI instance
        gui = EEGArtifactCleanerGUI()
        gui.show()
        
        # Wait for initialization to complete
        def start_screenshots():
            if gui.service is not None:
                # Start screenshot generation
                screenshot_generator = ScreenshotGenerator(app, gui)
                screenshot_generator.start()
            else:
                # Wait a bit more for initialization
                QTimer.singleShot(1000, start_screenshots)
        
        QTimer.singleShot(3000, start_screenshots)  # Wait 3 seconds for full startup
        
        # Run for limited time (30 seconds max)
        QTimer.singleShot(30000, app.quit)
        
        # Run the application
        app.exec()
        
    except Exception as e:
        print(f"Error during screenshot generation: {e}")
        # Create placeholders instead
        print("Creating placeholder screenshots...")
        create_placeholder_screenshots()
    
    finally:
        print("Screenshot generation finished.")


if __name__ == "__main__":
    main()