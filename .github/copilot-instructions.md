# Copilot Instructions for Katharsis

This document provides guidelines for GitHub Copilot and the Copilot coding agent when working on the Katharsis EEG Artifact Cleaner project.

## Project Overview

Katharsis is a Python application for automatic EEG artifact cleaning using Independent Component Analysis (ICA). It features a PyQt6 GUI and processes EDF (European Data Format) files.

## Technology Stack

- **Python**: 3.9+ (CI tests: 3.9, 3.10, 3.11, 3.12, 3.13)
- **GUI Framework**: PyQt6
- **EEG Processing**: MNE-Python
- **Numerical Computing**: NumPy, SciPy
- **Machine Learning**: Scikit-learn (for ICA)
- **Testing**: pytest, pytest-cov, pytest-qt
- **Code Quality**: flake8, black, isort, mypy, pylint, bandit

## Project Structure

```
Katharsis/
├── backend/                 # Core processing logic
│   ├── eeg_backend.py       # Data management & I/O
│   ├── ica_processor.py     # ICA implementation
│   ├── artifact_detector.py # Artifact detection algorithms
│   └── eeg_service.py       # Main service orchestration
├── components/              # GUI components (PyQt6)
│   ├── channel_selector.py  # Channel selection widget
│   ├── ica_selector.py      # ICA component selector
│   ├── comparison_screen.py # Results comparison
│   └── results_display.py   # Results visualization
├── tests/                   # Test suite
│   ├── test_backend.py      # Backend tests
│   ├── test_components.py   # GUI tests
│   └── conftest.py          # pytest fixtures
├── docs/                    # Documentation & GitHub Pages
├── eeg_gui_app.py           # Main application entry point
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Project configuration
```

## Code Style and Conventions

### Python Style

- Follow **PEP 8** for code style
- Use **Black** for code formatting (line length: 88)
- Use **isort** for import sorting (Black-compatible profile)
- Maximum line length: 127 characters for flake8, 88 for Black
- Use **type hints** where applicable

### Naming Conventions

```python
# Classes: PascalCase
class EEGProcessor:
    pass

# Functions/Variables: snake_case
def process_eeg_data():
    file_path = "data.edf"

# Constants: UPPER_SNAKE_CASE
MAX_COMPONENTS = 10

# Private methods: leading underscore
def _internal_method():
    pass
```

### Documentation

- Use **Google-style docstrings** for all public functions and classes
- Include Args, Returns, Raises, and Example sections where appropriate

```python
def process_signal(data: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Process EEG signal with filtering.
    
    Args:
        data (np.ndarray): Raw EEG data
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        np.ndarray: Filtered data
        
    Raises:
        ValueError: If data is empty
    """
```

## Build and Test Commands

### Installation

```bash
# Create virtual environment
python -m venv katharsis_env
source katharsis_env/bin/activate  # Linux/Mac
katharsis_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python eeg_gui_app.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run backend tests only
python -m pytest tests/test_backend.py -v

# Run GUI tests (requires display or xvfb on Linux)
# On Linux headless: xvfb-run -a python -m pytest tests/test_components.py -v
# Or set environment: QT_QPA_PLATFORM=offscreen python -m pytest tests/test_components.py -v
python -m pytest tests/test_components.py -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov=components --cov-report=html
```

### Code Quality Checks

```bash
# Linting (critical errors only)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full linting with warnings
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Code formatting check
black --check --diff .

# Import sorting check
isort --check-only --diff .

# Type checking
mypy backend/ --ignore-missing-imports --no-strict-optional

# Security check
bandit -r backend/ -f txt
```

## Development Guidelines

### Backend Development

- Backend modules should be self-contained and testable without GUI
- Use `EEGBackendCore` for data management and I/O
- Use `ICAProcessor` for ICA analysis
- Use `ArtifactDetector` for artifact detection
- Use `EEGArtifactCleaningService` for orchestrating the full workflow

### GUI Development

- All GUI components inherit from PyQt6 widgets
- Use signals and slots for component communication
- Test GUI components with pytest-qt
- Set `QT_QPA_PLATFORM=offscreen` for headless testing

### Testing Guidelines

- Write unit tests for all new functionality
- Use pytest fixtures from `conftest.py`
- Follow the Arrange-Act-Assert pattern
- Test both success and error cases
- Backend tests should not require a GUI display

### Commit Messages

Use Conventional Commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(backend): add support for BDF files`
- `fix(gui): resolve crash on file selection`
- `docs(readme): update installation instructions`
- `test(ica): add unit tests for component detection`

## Important Notes

- The project is bilingual (Greek and English) - maintain consistency with existing documentation
- EDF files are the primary supported format
- ICA uses FastICA algorithm via MNE-Python and scikit-learn
- Signal processing pipeline: Raw EEG → Band-pass Filter (1-40 Hz) → ICA → Artifact Removal → Clean EEG
- Detection criteria include variance, kurtosis, range, and EOG correlation thresholds
