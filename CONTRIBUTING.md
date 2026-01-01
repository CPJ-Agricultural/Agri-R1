# Contributing to Agri-R1

Thank you for your interest in contributing to Agri-R1! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment details (OS, GPU, Python version, etc.)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear commit messages
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic
- Use type hints where appropriate

### Testing

- Test your code on sample data before submitting
- Ensure backward compatibility
- Report any performance impacts

### Documentation

- Update README.md if adding new features
- Add examples for new functionality
- Keep documentation clear and concise

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Agri-R1.git
cd Agri-R1

# Create development environment
conda create -n agri-r1-dev python=3.11
conda activate agri-r1-dev

# Install dependencies
pip install -r requirements.txt
cd src/r1-v && pip install -e . && cd ../..

# Install development tools
pip install black flake8 isort pytest
```

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to open an issue for any questions about contributing!
