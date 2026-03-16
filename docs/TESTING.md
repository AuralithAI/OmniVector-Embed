"""Development guidelines and testing instructions."""

# Testing Guidelines for OmniVector-Embed

## Unit Tests (Run Locally, No GPU)

All unit tests should run on CPU without downloading large models.

### Run all unit tests:
```bash
pytest tests/unit/ -v --tb=short
```

### Run specific test file:
```bash
pytest tests/unit/test_backbone.py -v
```

### Run with coverage:
```bash
pytest tests/unit/ --cov=src/omnivector --cov-report=html
```

### Key test files:
- `test_backbone.py`: Bidirectional attention, LoRA, parameters
- `test_latent_attention.py`: EagerMultiheadAttention, LatentAttentionPooling
- `test_data.py`: Schema, preprocessing, validation
- `test_model.py`: OmniVectorModel forward pass, initialization

## Integration Tests (Optional GPU)

Integration tests may require GPU and larger models.

### Run integration tests:
```bash
pytest tests/integration/ -v -m integration
```

### Mark tests:
```python
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_feature():
    pass
```

## Code Quality

### Format code:
```bash
ruff format src/ tests/
```

### Lint:
```bash
ruff check src/ tests/
```

### Type check:
```bash
mypy src/omnivector/ --strict
```

### All checks:
```bash
ruff format --check src/ tests/
ruff check src/ tests/
mypy src/omnivector/ --strict
pytest tests/unit/ -v
```

## Pre-commit Hooks

Install and use pre-commit to automatically check before commits:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Test Coverage

Target minimum coverage: 80% for `src/omnivector/`

```bash
pytest tests/unit/ --cov=src/omnivector --cov-report=term-missing --cov-fail-under=70
```

## Fixture Guidelines

Use `tests/conftest.py` for shared fixtures:
- `device`: Torch device (CPU/GPU)
- `tokenizer`: Mistral tokenizer
- `sample_embedding_pair`: Test data
- `mock_backbone`: Test model component
- `mock_pooling`: Test pooling layer

## Test Organization

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test components working together
- **Mark slow tests**: `@pytest.mark.slow` for tests > 10 seconds
- **Skip if needed**: `pytest.skip("reason")` for conditional skipping

## CI/CD

Tests run automatically on:
1. **Every PR**: Unit tests + lint + type check (must pass)
2. **Manual trigger**: Full integration tests (optional)
3. **Tag release**: All tests + ONNX export validation

## Example Test

```python
import pytest
import torch

class TestMyFeature:
    \"\"\"Test suite for my feature.\"\"\"

    def test_basic_functionality(self):
        \"\"\"Test basic functionality.\"\"\"
        from omnivector.model import MyModel
        
        model = MyModel()
        assert model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_gpu_feature(self):
        \"\"\"Test GPU-specific feature.\"\"\"
        # GPU test here
        pass
```

## Debugging Failed Tests

```bash
# Verbose output
pytest tests/unit/test_backbone.py::TestMistralEmbeddingBackbone::test_forward_pass -vv

# Drop into debugger on failure
pytest --pdb tests/unit/test_backbone.py

# Show local variables
pytest -l tests/unit/test_backbone.py

# Run only failed tests from last run
pytest --lf
```

## Adding New Tests

1. Create test file in `tests/unit/` or `tests/integration/`
2. Follow naming: `test_*.py` for modules, `Test*` for classes, `test_*` for functions
3. Add docstrings and type hints
4. Use fixtures from `conftest.py`
5. Mark with `@pytest.mark.skip()` if not ready
6. Run locally before pushing
7. Ensure CI passes before merge

## Performance Testing

For long-running tests, use `pytest-benchmark`:

```python
def test_performance(benchmark):
    \"\"\"Benchmark model inference.\"\"\"
    @benchmark
    def forward_pass():
        return model(input_ids)
```

Run with: `pytest --benchmark-only`
