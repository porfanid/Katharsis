#!/usr/bin/env python3
"""
Katharsis - EEG Artifact Cleaner v4.0
====================================

Complete separation between frontend and backend.
- Backend: Autonomous KatharsisBackend handles all business logic
- Frontend: Pure PyQt6 UI that only displays data from backend

Author: porfanid
Version: 4.0 - Complete Frontend/Backend Separation
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from backend import KatharsisBackend
from frontend import KatharsisMainWindow


def setup_application_style(app: QApplication):
    """Set up global application style"""
    app.setStyle('Fusion')  # Use Fusion style for consistent cross-platform appearance
    
    # Set application font
    font = QFont("Arial", 11)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    # Set application properties
    app.setApplicationName("Katharsis")
    app.setApplicationVersion("4.0")
    app.setOrganizationName("porfanid")
    app.setOrganizationDomain("github.com/porfanid")


def main():
    """Main application entry point"""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set up application style
    setup_application_style(app)
    
    try:
        # Test backend initialization
        print("🔧 Initializing Katharsis Backend...")
        backend = KatharsisBackend()
        print("✅ Backend initialized successfully")
        
        # Test backend with data.edf if available
        data_file = Path("data.edf")
        if data_file.exists():
            print("📊 Testing backend with data.edf...")
            validation = backend.validate_file(str(data_file))
            if validation["valid"]:
                print(f"   ✅ File validation: {validation['n_channels']} channels, "
                      f"{validation['duration']:.1f}s duration")
                
                # Test file loading
                load_result = backend.load_file(str(data_file))
                if load_result["success"]:
                    print(f"   ✅ File loading: {load_result['n_channels']} channels loaded")
                    
                    # Test basic preprocessing
                    prep_config = {
                        "filtering": {
                            "enable_bandpass": True,
                            "low_freq": 1.0,
                            "high_freq": 40.0
                        }
                    }
                    prep_result = backend.apply_preprocessing(prep_config)
                    if prep_result["success"]:
                        print(f"   ✅ Preprocessing: {len(prep_result['applied_steps'])} steps applied")
                    else:
                        print(f"   ⚠️ Preprocessing failed: {prep_result.get('error', 'Unknown error')}")
                    
                    # Test ICA analysis
                    ica_result = backend.perform_ica_analysis("fastica", n_components=3)
                    if ica_result["success"]:
                        print(f"   ✅ ICA analysis: {ica_result['n_components']} components, "
                              f"{len(ica_result['suggested_artifacts'])} artifacts detected")
                    else:
                        print(f"   ⚠️ ICA analysis failed: {ica_result.get('error', 'Unknown error')}")
                        
                else:
                    print(f"   ❌ File loading failed: {load_result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ File validation failed: {validation.get('error', 'Unknown error')}")
        else:
            print("   ℹ️ data.edf not found, skipping backend test")
        
        print("\n🎨 Creating GUI...")
        
        # Create main window (this will create its own backend instance)
        main_window = KatharsisMainWindow()
        
        # Show the main window
        main_window.show()
        
        print("✅ Katharsis started successfully!")
        print("   - Backend: Completely autonomous, handles all EEG processing")
        print("   - Frontend: Pure PyQt6 UI, only displays backend results")
        print("   - Architecture: Clean separation enables independent testing and development")
        
        # Run the application
        sys.exit(app.exec())
        
    except ImportError as e:
        error_msg = f"Missing required dependency: {str(e)}\n\nPlease install requirements:\npip install -r requirements.txt"
        print(f"❌ {error_msg}")
        
        # Show error dialog if possible
        try:
            QMessageBox.critical(None, "Dependency Error", error_msg)
        except:
            pass
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Failed to start Katharsis: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Show error dialog if possible
        try:
            QMessageBox.critical(None, "Startup Error", error_msg)
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()