# Contributing to Katharsis

Welcome to Katharsis! Thank you for your interest in contributing to this project.

## 🎯 Contribution Guide

### Types of Contributions

We accept various types of contributions:

- 🐛 **Bug Reports**: Report issues
- 💡 **Feature Requests**: Suggestions for new features
- 📝 **Documentation**: Improve documentation
- 🔧 **Code Contributions**: Implement improvements
- 🌍 **Translations**: Translations to other languages
- 🧪 **Testing**: Testing and feedback

### Setting Up Your Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/[your-username]/Katharsis.git
   cd Katharsis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv katharsis_env
   source katharsis_env/bin/activate  # Linux/Mac
   # or
   katharsis_env\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Reporting Bugs

When reporting a bug, please include:

- **Problem description**: What happened?
- **Reproduction steps**: How can we reproduce the problem?
- **Expected behavior**: What did you expect to happen?
- **Screenshots**: If GUI-related
- **Environment**:
  - OS (Windows/Mac/Linux)
  - Python version
  - Katharsis version
  - Error logs

### Feature Suggestions

For new features:

- **Describe the functionality**: What do you want it to do?
- **Justify the usefulness**: Why is it useful?
- **Suggest implementation**: How could it be implemented?
- **Consider the implications**: How does it affect existing features?

## 💻 Code Contributions

### Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. **Implement changes**
   - Write clean, readable code
   - Follow existing style conventions
   - Add docstrings to new functions/classes
   - Add or update tests

3. **Testing**
   ```bash
   python -m pytest tests/ -v
   python -m flake8 .
   python -m black --check .
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Standards

#### Style Guide

- **PEP 8** compliance
- **Black** for formatting
- **isort** for import sorting
- **Maximum line length**: 127 characters
- **Docstrings**: Google style

#### Naming Conventions

```python
# Classes: PascalCase
class EEGProcessor:
    pass

# Functions/Variables: snake_case
def process_eeg_data():
    file_path = "data.edf"

# Constants: UPPER_SNAKE_CASE
MAX_COMPONENTS = 10

# Private methods: _leading_underscore
def _internal_method():
    pass
```

#### Documentation

```python
def process_signal(data: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Process EEG signal with filtering.
    
    Args:
        data (np.ndarray): The raw EEG data
        sampling_rate (float): The sampling rate in Hz
        
    Returns:
        np.ndarray: The filtered data
        
    Raises:
        ValueError: If data is empty
        
    Example:
        >>> filtered = process_signal(raw_data, 256.0)
    """
```

### Testing

#### Test Structure

```
tests/
├── test_backend.py
├── test_components.py
├── test_integration.py
└── fixtures/
    └── sample_data.py
```

#### Test Conventions

```python
import pytest
from backend.eeg_backend import EEGBackendCore


class TestEEGBackend:
    """Tests for EEG Backend functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.backend = EEGBackendCore()
    
    def test_load_valid_file(self):
        """Test loading of valid EDF file."""
        # Arrange
        file_path = "tests/fixtures/sample.edf"
        
        # Act
        result = self.backend.load_file(file_path)
        
        # Assert
        assert result['success'] is True
        assert 'channels' in result
    
    def test_load_invalid_file(self):
        """Test handling of invalid file."""
        with pytest.raises(FileNotFoundError):
            self.backend.load_file("nonexistent.edf")
```

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

#### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

#### Examples

```bash
feat(backend): add support for BDF files
fix(gui): resolve crash on file selection
docs(readme): update installation instructions
test(ica): add unit tests for component detection
```

## 📚 Documentation

### README Updates

When adding new features:

- Update the features list
- Add usage examples
- Update dependency installations

### Code Documentation

- **Modules**: Module-level docstring at the top
- **Classes**: Class docstring with description and attributes
- **Methods**: Docstring with Args, Returns, Raises
- **Complex Code**: Inline comments for complex logic

### API Documentation

When adding new API:

```python
def new_api_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Brief description of the function.
    
    Detailed description of what the function does and how.
    Can contain multiple lines.
    
    Args:
        param1 (str): Description of the first parameter
        param2 (int, optional): Description of the second parameter.
                               Default is 10.
    
    Returns:
        Dict[str, Any]: Description of the returned dictionary
                       with keys and value types
    
    Raises:
        ValueError: When param1 is an empty string
        TypeError: When param2 is not an integer
    
    Example:
        >>> result = new_api_function("test", 20)
        >>> print(result['status'])
        'success'
    
    Note:
        This function changes the internal state of the object.
    """
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] I have tested my changes locally
- [ ] I have added/updated tests
- [ ] I have updated the documentation
- [ ] The code passes all quality checks
- [ ] I have checked for merge conflicts

### PR Template

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (change that will break existing functionality)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)

Add screenshots for GUI changes.

## Checklist

- [ ] Code follows the style guide
- [ ] Self-review of the code
- [ ] Code changes generate no new warnings
- [ ] Tests for new functionality
- [ ] Documentation updates
```

### Review Process

1. **Automated Checks**: CI/CD will run the tests
2. **Code Review**: Maintainers will review the code
3. **Feedback**: Changes may be requested
4. **Approval**: After approval, the PR will be merged

## 🌍 Internationalization

### Adding Translations

1. **Create language file**
   ```python
   # locales/en.py
   TRANSLATIONS = {
       "welcome_message": "Welcome to Katharsis",
       "select_file": "Select EDF File",
       "processing": "Processing...",
   }
   ```

2. **Use in code**
   ```python
   from locales import get_translation
   
   label.setText(get_translation("welcome_message"))
   ```

## 🏆 Recognition

### Contributors

All contributors are mentioned in:
- README.md
- Release notes
- Contributors page

### Types of Recognition

- **Code Contributors**: Credit in commits
- **Issue Reporters**: Credit in issue fixes
- **Documentation**: Credit in documentation updates
- **Translators**: Credit for translations
- **Testers**: Credit for extensive testing

## ❓ Questions?

If you have questions:

- **GitHub Issues**: For general questions
- **GitHub Discussions**: For community discussions
- **Email**: [maintainers@katharsis-eeg.org]

## 📄 License

By contributing, you agree that your contributions will be available under the same MIT License that covers the project.

---

Thank you for contributing to Katharsis! 🧠✨
