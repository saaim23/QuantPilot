# Contributing to Quantitative Finance Platform

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Basic understanding of finance and quantitative analysis
- Familiarity with pandas, numpy, and scikit-learn

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/your-username/quantitative-finance-platform.git
cd quantitative-finance-platform
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a development branch:
```bash
git checkout -b feature/your-feature-name
```

## How to Contribute

### 🐛 Reporting Bugs

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

**Example Bug Report:**

```
**Bug Summary**: Option pricing calculation returns negative values for deep ITM calls

**Steps to Reproduce:**
1. Go to Option Pricing module
2. Enter: AAPL, Strike: $100, Current Price: $200, Time to expiry: 30 days
3. Select Black-Scholes model
4. Click Calculate

**Expected**: Positive option value around $100
**Actual**: Returns -$15.50
**Environment**: Python 3.11, Windows 11, latest version
```

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear description** of the enhancement
- **Use case** - why would this be useful?
- **Implementation ideas** - how might this work?

**Example Enhancement:**

```
**Enhancement**: Add cryptocurrency options pricing

**Use Case**: Many traders now work with crypto derivatives, but current platform only supports traditional equities.

**Implementation**: 
- Add crypto data provider (e.g., CoinGecko API)
- Modify volatility models for 24/7 trading
- Add crypto-specific Greeks calculations
```

### 🔧 Code Contributions

#### Code Style Guidelines

**Python Style:**
- Follow PEP 8
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small (under 50 lines when possible)

**Financial Code Standards:**
- Always validate input parameters (no negative prices, etc.)
- Include error handling for financial calculations
- Add unit tests for all financial formulas
- Document assumptions and limitations

**Example Good Code:**
```python
def calculate_black_scholes_call(
    S: float,  # Current stock price
    K: float,  # Strike price
    T: float,  # Time to expiration (years)
    r: float,  # Risk-free rate
    sigma: float  # Volatility
) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current underlying price (must be > 0)
        K: Strike price (must be > 0)
        T: Time to expiration in years (must be > 0)
        r: Risk-free rate (decimal, e.g., 0.05 for 5%)
        sigma: Implied volatility (decimal, must be > 0)
    
    Returns:
        Call option price
        
    Raises:
        ValueError: If any input parameter is invalid
    """
    # Validate inputs
    if S <= 0:
        raise ValueError("Stock price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T <= 0:
        raise ValueError("Time to expiration must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call price
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price
```

#### Commit Messages

Use clear, descriptive commit messages:

**Good:**
```
Add GARCH(1,1) volatility forecasting model

- Implement GARCH parameter estimation
- Add volatility forecasting functionality  
- Include unit tests and documentation
- Fixes #123
```

**Bad:**
```
fixed stuff
update code
more changes
```

### 📊 Contributing New Models

When adding new financial models:

1. **Research First**: Ensure the model is well-established in academic literature
2. **Documentation**: Include references to papers/books
3. **Validation**: Compare results with known benchmarks
4. **Tests**: Add comprehensive unit tests
5. **UI Integration**: Update the Streamlit interface appropriately

**Model Contribution Checklist:**
- [ ] Mathematical implementation is correct
- [ ] Code includes proper error handling
- [ ] Comprehensive docstrings with references
- [ ] Unit tests with known test cases
- [ ] Integration with main app interface
- [ ] Documentation updated

### 🧪 Testing

**Running Tests:**
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_option_pricing.py

# Run with coverage
python -m pytest --cov=models tests/
```

**Writing Tests:**
- Test edge cases (very high/low volatility, near expiration)
- Test error conditions (negative inputs, invalid parameters)
- Test against known benchmarks
- Use realistic market data when possible

### 📝 Documentation

**Code Documentation:**
- All public functions need docstrings
- Include parameter types and descriptions
- Add examples for complex functions
- Document assumptions and limitations

**User Documentation:**
- Update README.md for new features
- Add examples for new functionality
- Update setup instructions if needed

## Pull Request Process

1. **Fork & Branch**: Create a feature branch from `main`
2. **Make Changes**: Implement your feature/fix
3. **Test**: Ensure all tests pass and add new tests
4. **Document**: Update relevant documentation
5. **Commit**: Use clear, descriptive commit messages
6. **Pull Request**: Submit PR with detailed description

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Related Issues
Fixes #(issue number)

## Screenshots (if applicable)
```

## Code Review Process

1. **Automated Checks**: CI/CD will run tests and style checks
2. **Maintainer Review**: Core team reviews code and design
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, we'll merge your PR

**Review Criteria:**
- Code quality and style
- Test coverage
- Documentation completeness
- Financial accuracy
- Performance impact

## Community Guidelines

### Our Values
- **Accuracy**: Financial calculations must be correct
- **Transparency**: Code should be readable and well-documented
- **Education**: Help others learn quantitative finance
- **Collaboration**: Respect different perspectives and approaches

### Behavior Standards

**Examples of encouraged behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment or discriminatory comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Getting Help

**Questions about contributing?**
- Open a GitHub issue with the "question" label
- Join our discussions in the GitHub Discussions tab
- Check existing documentation and issues first

**Financial/Mathematical Questions:**
- Provide references to relevant literature
- Show your work and assumptions
- Be specific about use cases

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Project documentation
- Release notes for significant contributions

**Types of Contributions We Value:**
- Code contributions (features, bug fixes)
- Documentation improvements
- Bug reports and testing
- Financial model validation
- Educational content
- Community support

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

**Thank you for contributing to the Quantitative Finance Platform!** 

Your contributions help make quantitative finance more accessible to everyone. 🚀