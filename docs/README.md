# Katharsis Documentation

This directory contains comprehensive documentation for the Katharsis EEG Analysis Platform.

## Documentation Files

### User Documentation
- **[user_guide.md](user_guide.md)** - Complete user guide with step-by-step instructions
- **[screenshots/](screenshots/)** - GUI screenshots for each workflow step

### Technical Documentation
- **[api/](api/)** - API documentation (auto-generated)
- **[examples/](examples/)** - Usage examples and tutorials

## Quick Start

1. **Installation**: Follow setup instructions in main README
2. **User Guide**: Read [user_guide.md](user_guide.md) for complete workflow
3. **Screenshots**: View [screenshots/](screenshots/) for visual guidance

## Generating Screenshots

To update screenshots after GUI changes:

```bash
# In headless environment (CI)
python docs/generate_screenshots.py

# With display (for actual screenshots)
DISPLAY=:0 python docs/generate_screenshots.py
```

## Documentation Structure

```
docs/
├── user_guide.md              # Main user documentation
├── generate_screenshots.py    # Screenshot generation script  
├── screenshots/               # GUI screenshots
│   ├── 01_splash_screen.png
│   ├── 02_welcome_screen.png
│   ├── 03_channel_selection.png
│   ├── 04_preprocessing_overview.png
│   ├── 05_channel_analysis.png
│   ├── 06_filtering.png
│   ├── 07_referencing.png
│   ├── 08_pipeline.png
│   ├── 09_processing.png
│   ├── 10_enhanced_ica.png
│   ├── 11_artifact_classification.png
│   ├── 12_ica_components.png
│   ├── 13_component_selection.png
│   ├── 14_results_comparison.png
│   └── 15_data_export.png
├── api/                       # API documentation
└── examples/                  # Usage examples
```

## Contributing to Documentation

When updating documentation:

1. **Keep screenshots current** - Regenerate after GUI changes
2. **Update user guide** - Reflect new features and workflows  
3. **Test examples** - Ensure all code examples work
4. **Validate links** - Check all internal and external links

## Documentation Standards

- **Language**: English for main documentation, Greek for UI labels
- **Screenshots**: 800x600 minimum resolution, PNG format
- **Code examples**: Tested and working
- **Structure**: Clear headings, numbered steps, bullet points
- **Links**: Relative paths for internal links

---

For technical support or documentation questions, please open an issue on GitHub.