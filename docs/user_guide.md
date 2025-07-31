# Katharsis EEG Analysis Platform - User Guide

## Overview

Katharsis is a comprehensive EEG analysis platform inspired by EEGLAB, offering advanced preprocessing, artifact removal, and analysis capabilities with a modern Python/Qt6 interface.

## Complete Workflow Process

### Step 1: Application Startup

When you launch Katharsis, you'll see a loading splash screen while the application initializes all backend components:

![Splash Screen](screenshots/01_splash_screen.png)

The application loads:
- MNE-Python libraries for EEG processing
- ICA algorithms for artifact detection
- Advanced preprocessing modules
- GUI components

### Step 2: File Selection

After startup, you'll see the welcome screen with the main file selection interface:

![Welcome Screen](screenshots/02_welcome_screen.png)

**To select an EEG file:**
1. Click the "🔍 Επιλογή Αρχείου EDF για Ανάλυση" button
2. Navigate to your EEG data file
3. Supported formats: EDF, BDF, FIF, CSV, SET (EEGLAB format)
4. The application will automatically detect the file format

**File Requirements:**
- EEG data with multiple channels
- Proper channel names (preferably 10-20 system)
- Sampling rate information included
- Duration: recommended 2+ minutes for reliable ICA

### Step 3: Channel Selection

Once you've loaded a file, the Channel Selection screen appears:

![Channel Selection](screenshots/03_channel_selection.png)

**Channel Selection Process:**
1. **View Channel Information**: All detected channels are displayed with their properties
2. **Select Channels**: Choose channels for analysis by checking boxes
   - Recommended: Select 19-64 channels for optimal ICA performance
   - Include frontal channels (Fp1, Fp2) for eye artifact detection
   - Include temporal channels (T7, T8) for muscle artifact detection
3. **Preview Data**: View raw data preview to check signal quality
4. **Bad Channel Detection**: The system can automatically detect problematic channels
5. Click "✅ Συνέχεια με τα Επιλεγμένα Κανάλια" to proceed

### Step 4: Advanced Preprocessing

The heart of Katharsis - comprehensive preprocessing pipeline:

![Advanced Preprocessing](screenshots/04_preprocessing_overview.png)

#### Tab 1: 📊 Channel Analysis

![Channel Analysis Tab](screenshots/05_channel_analysis.png)

**Bad Channel Detection:**
- **Automatic Detection**: Uses multiple criteria:
  - Flat channels (no signal variation)
  - High variance channels (excessive noise)
  - Low correlation with neighboring channels
  - Statistical outliers (z-score > 4)
- **Manual Selection**: Click channels to mark/unmark as bad
- **Interpolation**: Spherical spline interpolation for bad channels

**Channel Information:**
- Real-time statistics for each channel
- Variance, mean, correlation metrics
- Channel locations (if available)

#### Tab 2: 🔄 Filtering

![Filtering Tab](screenshots/06_filtering.png)

**Filter Types Available:**
1. **High-pass Filter**: Remove slow drifts (recommended: 1 Hz)
2. **Low-pass Filter**: Remove high-frequency noise (recommended: 40 Hz)
3. **Band-pass Filter**: Isolate specific frequency bands
4. **Band-stop Filter**: Remove specific frequency ranges
5. **Notch Filter**: Remove line noise (50/60 Hz)

**Filter Configuration:**
- **Method**: FIR (linear phase) or IIR (steeper rolloff)
- **Frequency Parameters**: Precise control over cutoff frequencies
- **Quality Factor**: For notch filters
- **Preview**: Real-time filter response visualization

**Recommended Filter Settings:**
- **Clinical EEG**: 1-40 Hz bandpass + 50 Hz notch
- **ERP Studies**: 0.1-30 Hz bandpass
- **Sleep Analysis**: 0.3-35 Hz bandpass
- **Spectral Analysis**: 1-80 Hz bandpass

#### Tab 3: 🔗 Re-referencing

![Re-referencing Tab](screenshots/07_referencing.png)

**Reference Methods:**
1. **Average Reference**: All channels average as reference
   - Best for high-density EEG (64+ channels)
   - Reduces common noise across channels
2. **Common Reference**: Single channel as reference
   - Good for clinical applications
   - Often uses linked mastoids or Cz
3. **Bipolar Reference**: Adjacent channel pairs
   - Useful for localized analysis
   - Common in clinical montages
4. **Linked Ears**: A1 and A2 average
   - Traditional EEG reference
5. **Custom Reference**: User-defined channel combinations

**Preset Montages:**
- 10-20 Standard
- 10-10 Extended  
- Clinical montages
- High-density layouts

#### Tab 4: ⚙️ Pipeline Configuration

![Pipeline Tab](screenshots/08_pipeline.png)

**Preprocessing Presets:**
1. **Clinical Preset**: 
   - 1-40 Hz bandpass
   - 50 Hz notch
   - Average reference
   - Bad channel detection enabled
2. **Research Preset**:
   - 0.5-80 Hz bandpass
   - Multiple notch filters
   - Custom reference options
3. **ERP Preset**:
   - 0.1-30 Hz bandpass
   - No notch filter
   - Average reference
4. **Sleep Preset**:
   - 0.3-35 Hz bandpass
   - Conservative bad channel detection
5. **Minimal Preset**:
   - Basic filtering only

**Pipeline Execution:**
- Shows processing steps in order
- Estimated processing time
- Progress monitoring
- Detailed logging

### Step 5: Running Preprocessing

![Processing Progress](screenshots/09_processing.png)

When you click "🚀 Run Preprocessing":
1. **Validation**: Checks configuration and data compatibility
2. **Sequential Processing**: Applies filters, referencing, interpolation
3. **Progress Updates**: Real-time status and progress bar
4. **Quality Metrics**: Calculates improvement statistics
5. **Results Summary**: Shows before/after comparisons

**Processing Status Messages:**
- "Loading and validating data..."
- "Applying high-pass filter..."
- "Detecting bad channels..."
- "Interpolating bad channels..."
- "Applying reference..."
- "Computing quality metrics..."
- "Preprocessing completed!"

### Step 6: Enhanced ICA Analysis

After preprocessing, you proceed to the enhanced ICA system:

![Enhanced ICA](screenshots/10_enhanced_ica.png)

#### ICA Algorithm Selection

**Available Algorithms:**
1. **FastICA**: Fast, robust, good for most datasets
2. **Extended Infomax**: Handles sub-Gaussian and super-Gaussian sources
3. **Picard**: Preconditioned ICA, fastest convergence
4. **MNE Default**: Reliable, well-tested implementation

#### Automatic Artifact Classification

![Artifact Classification](screenshots/11_artifact_classification.png)

**Detected Artifact Types:**
1. **👁️ Eye Blinks**: Vertical eye movements, frontal distribution
2. **👀 Eye Movements**: Horizontal eye movements, frontal-temporal
3. **💪 Muscle Artifacts**: High-frequency, temporal/occipital
4. **❤️ Heart Artifacts**: Cardiac rhythm, regular pattern
5. **⚡ Line Noise**: 50/60 Hz, power line interference
6. **📊 Channel Issues**: Bad channels, electrode problems
7. **🏃 Movement**: Motion artifacts, irregular patterns
8. **📉 Statistical Outliers**: Unusual statistical properties

**Classification Features:**
- **Confidence Scoring**: Each classification has confidence percentage
- **Automatic Recommendations**: Suggests components for rejection
- **Manual Override**: User can accept/reject recommendations
- **Stability Analysis**: Multi-run validation for robust results

#### Component Visualization

![ICA Components](screenshots/12_ica_components.png)

**For Each Component:**
- **Topographic Map**: Spatial distribution
- **Time Series**: Component activation over time
- **Power Spectrum**: Frequency content analysis
- **Classification Label**: Automatic artifact type detection
- **Confidence Score**: Reliability of classification
- **Recommendation**: Accept/Reject suggestion

### Step 7: Component Selection and Rejection

![Component Selection](screenshots/13_component_selection.png)

**Selection Process:**
1. **Review Classifications**: Check automatic artifact detection
2. **Examine Topographies**: Look for characteristic patterns
3. **Check Time Series**: Identify artifact-like activations
4. **Verify Spectra**: Confirm frequency characteristics
5. **Select for Rejection**: Choose components to remove
6. **Apply Changes**: Remove selected artifacts

**Selection Tips:**
- **Eye Blinks**: Strong frontal, symmetrical pattern
- **Eye Movements**: Frontal-temporal, horizontal pattern  
- **Muscle**: High-frequency, localized to temporal/occipital
- **Heart**: Regular rhythm, may appear in multiple components
- **Line Noise**: Sharp peak at 50/60 Hz

### Step 8: Results Comparison

![Results Comparison](screenshots/14_results_comparison.png)

**Comparison Features:**
1. **Before/After Time Series**: Direct visual comparison
2. **Topographic Maps**: Spatial distribution changes
3. **Power Spectral Density**: Frequency content analysis
4. **Quality Metrics**: Quantitative improvement measures
5. **Artifact Reduction**: Specific artifact suppression stats

**Quality Metrics:**
- **SNR Improvement**: Signal-to-noise ratio enhancement
- **Artifact Reduction**: Percentage decrease in artifact power
- **Data Preservation**: Percentage of clean signal retained
- **Channel Quality**: Per-channel improvement scores

### Step 9: Data Export

![Data Export](screenshots/15_data_export.png)

**Export Options:**
1. **Clean EEG Data**: Processed, artifact-free EEG
2. **Processing Report**: Detailed analysis summary
3. **ICA Components**: All computed components
4. **Quality Metrics**: Before/after statistics
5. **Configuration**: Settings used for reproducibility

**Export Formats:**
- **MNE Raw (.fif)**: Full MNE compatibility
- **EEGLAB Set (.set)**: MATLAB/EEGLAB format
- **European Data Format (.edf)**: Clinical standard
- **Comma Separated Values (.csv)**: Universal format
- **MATLAB (.mat)**: Research analysis

## Advanced Features

### MATLAB/EEGLAB Comparison

Katharsis includes comprehensive comparison testing with EEGLAB:

**Test Data Generation:**
```bash
python tests/test_matlab_comparison.py
```

**MATLAB Validation:**
```matlab
% In MATLAB with EEGLAB
run('tests/matlab_comparison_tests.m')
```

**Comparison Metrics:**
- Signal correlation (>0.99 expected)
- Root mean square error
- Maximum absolute difference
- Signal-to-noise ratio

### Batch Processing

For processing multiple files:

```python
from backend import PreprocessingPipeline, PreprocessingPresets

# Load preset configuration
config = PreprocessingPresets.get_clinical_preset()

# Process multiple files
for file_path in eeg_files:
    raw = mne.io.read_raw(file_path, preload=True)
    pipeline = PreprocessingPipeline()
    processed_raw, results = pipeline.run_pipeline(raw, config)
    # Save results...
```

### Custom Preprocessing Pipelines

Create custom configurations:

```python
from backend import PreprocessingConfig, FilterConfig, ReferenceConfig

# Custom filter chain
filter_configs = [
    FilterConfig('highpass', freq_low=0.5),
    FilterConfig('lowpass', freq_high=45.0),
    FilterConfig('bandstop', freq_low=48.0, freq_high=52.0),  # EU line noise
    FilterConfig('bandstop', freq_low=98.0, freq_high=102.0)  # 2nd harmonic
]

# Custom preprocessing configuration
config = PreprocessingConfig(
    apply_filters=True,
    filter_configs=filter_configs,
    detect_bad_channels=True,
    bad_channel_detection_params={
        'method': 'multi_criteria',
        'flat_threshold': 1e-6,
        'correlation_threshold': 0.3,
        'variance_threshold': 3.0
    },
    interpolate_bad_channels=True,
    apply_reference=True,
    reference_config=ReferenceConfig('average'),
    verbose=True
)
```

## Troubleshooting

### Common Issues

**1. File Loading Problems**
- **Error**: "Unsupported file format"
- **Solution**: Ensure file has proper EEG headers and channel information
- **Formats**: Use EDF, BDF, FIF, or EEGLAB SET files

**2. Channel Detection Issues**  
- **Error**: "No valid EEG channels found"
- **Solution**: Check channel names match standard conventions (Fp1, Fp2, C3, C4, etc.)

**3. ICA Convergence Problems**
- **Error**: "ICA did not converge"
- **Solution**: 
  - Ensure sufficient data length (>2 minutes)
  - Check for flat or extremely noisy channels
  - Try different ICA algorithm (FastICA → Picard)

**4. Memory Issues**
- **Error**: "Out of memory"
- **Solution**:
  - Reduce data length or channel count
  - Close other applications
  - Use 64-bit Python installation

**5. Preprocessing Failures**
- **Error**: "Filter application failed"
- **Solution**:
  - Check filter parameters (frequencies within Nyquist limit)
  - Ensure sufficient data length for filter stability
  - Try FIR instead of IIR filters

### Performance Optimization

**For Large Datasets:**
1. **Downsample**: Reduce sampling rate if appropriate
2. **Segment**: Process data in chunks
3. **Channel Selection**: Use fewer channels for initial analysis
4. **Memory**: Ensure sufficient RAM (>8GB recommended)

**For Slow Processing:**
1. **Algorithm**: Try FastICA or Picard for faster ICA
2. **Filters**: Use IIR filters for computational efficiency
3. **Parallel**: Enable multi-threading in MNE settings

### Getting Help

**Documentation**: Check `/docs` directory for technical details
**Issues**: Report bugs on GitHub repository
**Community**: Join discussions in project forums
**Citation**: Please cite Katharsis in publications

## System Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 1GB disk space
- Any modern OS (Windows, Mac, Linux)

**Recommended:**
- Python 3.11+
- 8GB+ RAM
- SSD storage
- Multi-core processor
- High-resolution display (1920x1080+)

**Dependencies:**
- PyQt6 (GUI framework)
- MNE-Python (EEG processing)
- NumPy, SciPy (numerical computing)
- Matplotlib (visualization)
- scikit-learn (machine learning)

---

*This guide provides comprehensive instructions for using Katharsis EEG Analysis Platform. For technical details, see the API documentation in `/docs/api`.*