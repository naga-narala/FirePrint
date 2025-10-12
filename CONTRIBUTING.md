# Contributing to FirePrint v1.0

Thank you for your interest in contributing to FirePrint! This document provides guidelines and instructions for contributing.

## 🌟 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/FirePrint-v1.0/issues)
2. If not, create a new issue using the Bug Report template
3. Include detailed information:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)
   - Error messages and stack traces

### Suggesting Features

1. Check if the feature has been suggested in [Issues](https://github.com/yourusername/FirePrint-v1.0/issues)
2. Create a new issue using the Feature Request template
3. Clearly describe:
   - The problem you're trying to solve
   - Your proposed solution
   - Why this feature would be useful
   - Any alternative approaches considered

### Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   # Click 'Fork' on GitHub, then:
   git clone https://github.com/yourusername/FirePrint-v1.0.git
   cd FirePrint-v1.0
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

#### Development Workflow

1. **Make your changes**
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Comment complex logic

2. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   flake8 src/
   
   # Check type hints
   mypy src/
   ```

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```
   
   Commit message format:
   - `feat: add new feature`
   - `fix: fix bug in component`
   - `docs: update documentation`
   - `refactor: refactor code`
   - `test: add tests`

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill in the PR template
   - Link related issues

#### Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use meaningful variable and function names
- Write docstrings in Google or NumPy format

Example:
```python
def polygon_to_fingerprint(
    geometry: Polygon,
    image_size: int = 224,
    debug: bool = False
) -> np.ndarray:
    """
    Convert fire polygon to 4-channel fingerprint image.
    
    Args:
        geometry: Shapely polygon representing fire boundary
        image_size: Output image dimension (default: 224)
        debug: Whether to show debug visualizations
        
    Returns:
        NumPy array of shape (image_size, image_size, 4)
        
    Raises:
        ValueError: If geometry is invalid or empty
    """
    # Implementation
    pass
```

#### Testing

- Write unit tests for new features
- Maintain or improve test coverage
- Test edge cases and error conditions
- Use pytest fixtures for common test data

```python
import pytest
from src.polygon_converter import polygon_to_fingerprint

def test_polygon_to_fingerprint_basic():
    """Test basic polygon conversion."""
    # Test implementation
    pass

def test_polygon_to_fingerprint_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(ValueError):
        polygon_to_fingerprint(None)
```

### Documentation Contributions

- Update docstrings when changing code
- Update README.md for user-facing changes
- Update docs/DOCUMENTATION.md for technical details
- Add examples for new features
- Fix typos and improve clarity

### Areas We Need Help

1. **Core Features**
   - Additional geometric feature extractors
   - Alternative CNN architectures
   - Enhanced similarity metrics
   - Performance optimizations

2. **Testing**
   - Unit test coverage
   - Integration tests
   - Performance benchmarks
   - Edge case testing

3. **Documentation**
   - API documentation
   - Tutorial notebooks
   - Use case examples
   - Video tutorials

4. **Tools & Infrastructure**
   - CI/CD improvements
   - Docker containerization
   - Web dashboard
   - REST API

5. **Research**
   - New applications
   - Algorithm improvements
   - Comparative studies
   - Integration with other tools

## 📋 Pull Request Process

1. **Before submitting:**
   - Update documentation
   - Add/update tests
   - Ensure all tests pass
   - Update CHANGELOG.md

2. **PR Requirements:**
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if relevant
   - Passes CI checks

3. **Review Process:**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

4. **After merge:**
   - Delete your branch
   - Pull latest changes from main
   - Your contribution will be in the next release!

## 🎯 Coding Standards

### Python Best Practices

- Use virtual environments
- Pin dependency versions
- Handle errors gracefully
- Log important operations
- Avoid hardcoded paths
- Use configuration files

### Performance Considerations

- Profile before optimizing
- Use vectorized operations (NumPy)
- Batch process when possible
- Cache expensive computations
- Consider memory usage

### Security

- Never commit credentials
- Validate user inputs
- Sanitize file paths
- Use secure random generators
- Handle sensitive data carefully

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other unprofessional conduct

## 📞 Getting Help

- **Questions**: Open a [Discussion](https://github.com/yourusername/FirePrint-v1.0/discussions)
- **Bugs**: Create an [Issue](https://github.com/yourusername/FirePrint-v1.0/issues)
- **Chat**: Join our community (link coming soon)
- **Email**: your.email@example.com

## 🎉 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

Thank you for contributing to FirePrint! 🔥

---

*This guide is inspired by successful open-source projects and adapted for FirePrint's needs.*

