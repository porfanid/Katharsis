# Contributing to Katharsis

Καλώς ήρθατε στο Katharsis! Ευχαριστούμε που ενδιαφέρεστε να συνεισφέρετε σε αυτό το project.

## 🎯 Οδηγός Συνεισφοράς

### Τύποι Συνεισφορών

Δεχόμαστε διάφορους τύπους συνεισφορών:

- 🐛 **Bug Reports**: Αναφορά προβλημάτων
- 💡 **Feature Requests**: Προτάσεις νέων χαρακτηριστικών
- 📝 **Documentation**: Βελτίωση τεκμηρίωσης
- 🔧 **Code Contributions**: Υλοποίηση βελτιώσεων
- 🌍 **Translations**: Μεταφράσεις σε άλλες γλώσσες
- 🧪 **Testing**: Δοκιμές και feedback

### Προετοιμασία Περιβάλλοντος

1. **Fork το repository**
   ```bash
   git clone https://github.com/[your-username]/Katharsis.git
   cd Katharsis
   ```

2. **Δημιουργία virtual environment**
   ```bash
   python -m venv katharsis_env
   source katharsis_env/bin/activate  # Linux/Mac
   # ή
   katharsis_env\Scripts\activate     # Windows
   ```

3. **Εγκατάσταση dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Αν υπάρχει
   ```

4. **Εγκατάσταση pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Αναφορά Bugs

Όταν αναφέρετε ένα bug, παρακαλούμε συμπεριλάβετε:

- **Περιγραφή του προβλήματος**: Τι συνέβη;
- **Βήματα αναπαραγωγής**: Πώς μπορούμε να αναπαράγουμε το πρόβλημα;
- **Αναμενόμενη συμπεριφορά**: Τι περιμένατε να συμβεί;
- **Screenshots**: Αν είναι GUI-related
- **Περιβάλλον**:
  - OS (Windows/Mac/Linux)
  - Python version
  - Katharsis version
  - Error logs

### Προτάσεις Χαρακτηριστικών

Για νέα χαρακτηριστικά:

- **Περιγράψτε τη λειτουργία**: Τι θέλετε να κάνει;
- **Αιτιολογήστε τη χρησιμότητα**: Γιατί είναι χρήσιμο;
- **Προτείνετε υλοποίηση**: Πώς θα μπορούσε να υλοποιηθεί;
- **Εξετάστε τις επιπτώσεις**: Πώς επηρεάζει υπάρχοντα features;

## 💻 Code Contributions

### Workflow

1. **Δημιουργία branch**
   ```bash
   git checkout -b feature/your-feature-name
   # ή
   git checkout -b bugfix/issue-number
   ```

2. **Υλοποίηση αλλαγών**
   - Γράψτε καθαρό, readable κώδικα
   - Ακολουθήστε τα υπάρχοντα style conventions
   - Προσθέστε docstrings στις νέες functions/classes
   - Προσθέστε ή ενημερώστε tests

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

5. **Push και Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Standards

#### Style Guide

- **PEP 8** compliance
- **Black** για formatting
- **isort** για import sorting
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
    Επεξεργάζεται το EEG σήμα με φιλτράρισμα.
    
    Args:
        data (np.ndarray): Τα raw EEG δεδομένα
        sampling_rate (float): Η συχνότητα δειγματοληψίας σε Hz
        
    Returns:
        np.ndarray: Τα φιλτραρισμένα δεδομένα
        
    Raises:
        ValueError: Αν τα δεδομένα είναι άδεια
        
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
        """Setup για κάθε test method."""
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

Χρησιμοποιούμε [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

#### Types

- `feat`: Νέο χαρακτηριστικό
- `fix`: Bug fix
- `docs`: Αλλαγές documentation
- `style`: Formatting, missing semicolons, κλπ.
- `refactor`: Code refactoring
- `test`: Προσθήκη tests
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

Όταν προσθέτετε νέα features:

- Ενημερώστε τη λίστα χαρακτηριστικών
- Προσθέστε παραδείγματα χρήσης
- Ενημερώστε τις εγκαταστάσεις dependencies

### Code Documentation

- **Modules**: Module-level docstring στην κορυφή
- **Classes**: Class docstring με περιγραφή και attributes
- **Methods**: Docstring με Args, Returns, Raises
- **Complex Code**: Inline comments για περίπλοκη λογική

### API Documentation

Αν προσθέτετε νέα API:

```python
def new_api_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Συνοπτική περιγραφή της λειτουργίας.
    
    Λεπτομερής περιγραφή του τι κάνει η συνάρτηση και πώς.
    Μπορεί να περιέχει πολλές γραμμές.
    
    Args:
        param1 (str): Περιγραφή του πρώτου παραμέτρου
        param2 (int, optional): Περιγραφή του δεύτερου παραμέτρου.
                               Default είναι 10.
    
    Returns:
        Dict[str, Any]: Περιγραφή του επιστρεφόμενου dictionary
                       με τα keys και τους τύπους values
    
    Raises:
        ValueError: Όταν το param1 είναι κενό string
        TypeError: Όταν το param2 δεν είναι integer
    
    Example:
        >>> result = new_api_function("test", 20)
        >>> print(result['status'])
        'success'
    
    Note:
        Αυτή η συνάρτηση αλλάζει την εσωτερική κατάσταση του object.
    """
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Έχω δοκιμάσει τις αλλαγές μου τοπικά
- [ ] Έχω προσθέσει/ενημερώσει tests
- [ ] Έχω ενημερώσει τη documentation
- [ ] Ο κώδικας περνάει όλα τα quality checks
- [ ] Έχω ελέγξει για merge conflicts

### PR Template

```markdown
## Περιγραφή

Σύντομη περιγραφή των αλλαγών.

## Τύπος Αλλαγής

- [ ] Bug fix (non-breaking change που διορθώνει πρόβλημα)
- [ ] New feature (non-breaking change που προσθέτει λειτουργία)
- [ ] Breaking change (αλλαγή που θα σπάσει existing functionality)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (αν εφαρμόζεται)

Προσθέστε screenshots για GUI changes.

## Checklist

- [ ] Ο κώδικας ακολουθεί το style guide
- [ ] Self-review of the code
- [ ] Code changes generate no new warnings
- [ ] Tests για νέα functionality
- [ ] Documentation updates
```

### Review Process

1. **Automated Checks**: CI/CD θα τρέξει τα tests
2. **Code Review**: Maintainers θα ελέγξουν τον κώδικα
3. **Feedback**: Ενδέχεται να ζητηθούν αλλαγές
4. **Approval**: Μετά την έγκριση, το PR θα γίνει merge

## 🌍 Internationalization

### Προσθήκη Μεταφράσεων

1. **Δημιουργία language file**
   ```python
   # locales/en.py
   TRANSLATIONS = {
       "welcome_message": "Welcome to Katharsis",
       "select_file": "Select EDF File",
       "processing": "Processing...",
   }
   ```

2. **Χρήση στον κώδικα**
   ```python
   from locales import get_translation
   
   label.setText(get_translation("welcome_message"))
   ```

## 🏆 Recognition

### Contributors

Όλοι οι contributors αναφέρονται στο:
- README.md
- Release notes
- Contributors page

### Types of Recognition

- **Code Contributors**: Αναφορά σε commits
- **Issue Reporters**: Credit στα issue fixes
- **Documentation**: Αναφορά σε documentation updates
- **Translators**: Credit για translations
- **Testers**: Αναφορά για extensive testing

## ❓ Ερωτήσεις;

Αν έχετε ερωτήσεις:

- **GitHub Issues**: Για γενικές ερωτήσεις
- **GitHub Discussions**: Για συζητήσεις κοινότητας
- **Email**: [maintainers@katharsis-eeg.org]

## 📄 License

Συνεισφέροντας, συμφωνείτε ότι οι συνεισφορές σας θα διαθέτονται υπό την ίδια MIT License που καλύπτει το project.

---

Ευχαριστούμε για τη συνεισφορά σας στο Katharsis! 🧠✨