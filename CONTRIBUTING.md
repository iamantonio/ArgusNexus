# Contributing to ArgusNexus

Thank you for your interest in contributing to ArgusNexus! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive. We're building something together.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Virtual environment (recommended)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ArgusNexus/ArgusNexus.git
cd ArgusNexus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy example config
cp config.yaml.example config.yaml
cp .env.example .env

# Edit .env with your API keys (for testing)

# Run tests
pytest
```

## Code Style

We follow **PEP 8** with these specifics:

- **Line length**: 100 characters max
- **Imports**: Use `isort` for ordering
- **Type hints**: Required for public functions
- **Docstrings**: Google style, required for public classes/functions

```python
def calculate_position_size(
    capital: Decimal,
    entry_price: Decimal,
    stop_loss: Decimal,
    risk_pct: Decimal = Decimal("0.01")
) -> Decimal:
    """
    Calculate position size based on risk parameters.

    Args:
        capital: Total available capital in USD
        entry_price: Expected entry price
        stop_loss: Stop loss price
        risk_pct: Percentage of capital to risk (default: 1%)

    Returns:
        Position size in base currency units
    """
    ...
```

## Areas for Contribution

We especially welcome contributions to:

### Truth Engine (`src/truth/`)
The decision logging system is core to ArgusNexus. Improvements to:
- Query performance
- New analysis views
- Export formats

### Risk System (`src/risk/`)
The 10-layer risk gate can always be improved:
- New risk checks
- Better correlation detection
- Performance optimization

### Strategy Framework (`src/strategy/`)
While we don't accept specific trading strategies (that's your alpha!), improvements to:
- Strategy base classes
- Signal context enrichment
- Backtesting infrastructure

### Documentation
- Tutorials and guides
- API documentation
- Example configurations

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality
4. **Ensure all tests pass** (`pytest`)
5. **Update documentation** if needed
6. **Commit** with clear messages
7. **Push** to your fork
8. **Open a Pull Request**

### PR Requirements

- [ ] All tests pass
- [ ] Code follows PEP 8
- [ ] New code has type hints
- [ ] Public functions have docstrings
- [ ] CHANGELOG.md updated (if applicable)

## Commit Messages

Use conventional commits:

```
feat: Add new risk check for volatility spikes
fix: Correct chandelier exit calculation for shorts
docs: Update README with new configuration options
test: Add tests for inverse chandelier exit
refactor: Simplify position sizing logic
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_strategy.py -v
```

## Questions?

- Open a GitHub Discussion for questions
- Check existing issues before creating new ones
- Tag issues appropriately (`bug`, `enhancement`, `question`)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
