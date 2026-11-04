# Contributing to MessageShield

Thank you for considering contributing to MessageShield! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots if applicable

### Suggesting Enhancements

We love new ideas! To suggest an enhancement:
1. Check if the enhancement has already been suggested
2. Create an issue with the `enhancement` label
3. Clearly describe the feature and its benefits
4. If possible, provide examples or mockups

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Make your changes** with clear, descriptive commits
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

### Testing

Before submitting:
- Test the web interface (`python -m streamlit run app.py`)
- Test CLI tools (`python src/classify.py "test message"`)
- Ensure the model training script works (`python src/train_improved.py`)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/MessageShield.git
cd MessageShield

# Install dependencies
pip install -r requirements.txt

# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes and test

# Commit your changes
git add .
git commit -m "Add: your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## Questions?

Feel free to open an issue with the `question` label or reach out to [@k-deejah](https://github.com/k-deejah).

Thank you for contributing! 🙏