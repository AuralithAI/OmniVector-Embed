"""Contributing to OmniVector-Embed

We welcome contributions! This guide explains how to participate.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. Fork the repository
2. Clone locally: `git clone https://github.com/AuralithAI/omnivector-embed.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev,vision]"`
5. Set up pre-commit: `pre-commit install`

## Making Changes

### Before you start:
- Check existing issues/PRs to avoid duplicates
- Open an issue for large changes to discuss first
- Review the architecture docs in `docs/architecture.md`

### Branch naming:
- Feature: `feature/short-description`
- Bug fix: `bugfix/short-description`
- Docs: `docs/short-description`

### Commit messages:
```
<type>: <subject> [(<ticket>)]

<body>

footers as needed
```

Examples:
```
feat: add vision encoder integration (OV-42)

Implement SigLIP-based vision encoder with 1152→4096 projection.
Includes unit tests and documentation.

Fixes #42
```

```
test: increase backbone unit test coverage
```

```
docs: add CONTRIBUTING.md
```

## Code Quality Standards

### 1. Type Hints (Required)
- Add type hints to all function signatures
- Use proper return types

```python
def encode_text(
    self,
    texts: Union[str, List[str]],
    instruction: Optional[str] = None,
) -> torch.Tensor:
    """Encode text to embeddings."""
    pass
```

### 2. Docstrings (Required)
- Module, class, and public function docstrings
- Use Google-style docstrings

```python
class MyClass:
    \"\"\"Brief description.
    
    Longer description if needed.
    
    Attributes:
        attr1: Description
        attr2: Description
    \"\"\"
    
    def my_method(self, arg1: int, arg2: str) -> bool:
        \"\"\"Brief description.
        
        Longer description.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            Boolean result
            
        Raises:
            ValueError: If invalid argument
        \"\"\"
        pass
```

### 3. Logging (Use it!)
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Model loaded: 4096-dim output")
logger.warning("Dimension mismatch, using fallback")
logger.error("Failed to initialize encoder", exc_info=True)
```

### 4. Error Handling
```python
try:
    model = load_model(path)
except FileNotFoundError as e:
    logger.error(f"Model not found at {path}: {e}")
    raise ValueError(f"Invalid model path: {path}") from e
```

### 5. Testing (Required)
- Aim for 80%+ coverage
- Test happy path and error cases
- Use descriptive test names
- See **[docs/testing.md](docs/testing.md)** for the full testing guide
  (markers, multi-GPU, fixtures, CI)

```python
def test_encode_text_with_valid_input(self):
    \"\"\"Test encoding with valid text input.\"\"\"
    # Arrange
    text = "Test query"
    
    # Act
    embedding = model.encode_text(text)
    
    # Assert
    assert embedding.shape == (1, 4096)
    assert torch.allclose(embedding.norm(), torch.ones(1))
```

## Pull Request Process

1. **Before creating PR:**
   - Format code: `ruff format src/ tests/`
   - Lint: `ruff check src/ tests/`
   - Type check: `mypy src/omnivector/ --strict`
   - Test: `pytest tests/unit/ -v --cov=src/omnivector`

2. **Create PR:**
   - Base branch: `develop` (or `main` for critical fixes)
   - Title: `[TYPE] Short description`
   - Description: What changed, why, how to test

3. **PR template:**
   ```markdown
   ## Description
   Brief explanation of changes.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   
   ## Related Issue
   Closes #123
   
   ## Testing
   How to test this change.
   
   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests pass locally
   - [ ] New dependencies added to pyproject.toml
   - [ ] Docstrings updated
   - [ ] No breaking changes
   ```

4. **CI must pass:**
   - Linting (ruff)
   - Type checking (mypy)
   - Unit tests (pytest)
   - Build test (hatch)

5. **Request review:**
   - Assign reviewers
   - Address feedback
   - Squash commits before merge

## Architecture Guidelines

### Model Components
- **Backbone**: Text encoder (MistralEmbeddingBackbone)
- **Pooling**: Latent attention aggregation (LatentAttentionPooling)
- **Vision**: Image encoder (SigLIPVisionEncoder)
- **Video**: Temporal frame aggregation (VideoEncoder)

### Critical Design Decisions
1. **No SDPA in latent attention** — use explicit matmuls for ONNX
2. **LoRA merge pre-export** — eliminates branching  
3. **Bidirectional attention** — override `_update_causal_mask`
4. **MRL at 4 dimensions** — InfoNCE loss weighted sum
5. **Pin transformers==4.44.2** — RoPE stability

See `docs/implementation-plan.md` for full architecture.

## Common Pitfalls

❌ **Don't:**
- Use `nn.MultiheadAttention` in latent attention (PyTorch 2.x routes to SDPA)
- Upgrade `transformers` or `peft` without testing RoPE ops
- Forget `merge_adapter()` before ONNX export
- Use causal masking for retrieval encoder
- Quantize attention scores in int8 (set `MatMulConstBOnly=True`)

✅ **Do:**
- Use `EagerMultiheadAttention` (explicit matmuls)
- Test bidirectionality: token[0] should see token[T-1]
- Run full CI before requesting review
- Add logging at key decision points
- Document non-obvious design choices

## Roadmap

Development is tracked via GitHub Issues and Milestones. See the
[README](README.md) for current feature status and the
[architecture guide](docs/architecture.md) for design context.

Priority areas for contributions:

- **Evaluation**: Additional MTEB task coverage and regression dashboards
- **Data loaders**: New retrieval/NLI dataset integrations
- **Multimodal**: Video temporal encoding improvements, audio augmentation
- **Deployment**: TensorRT, CoreML, and WASM export targets
- **Documentation**: Tutorials, API reference, deployment guides

## Getting Help

- **Technical questions?** Open an issue with `[QUESTION]` prefix
- **Bug report?** Use issue template with reproducible example
- **Feature request?** Describe use case and expected behavior
- **Discussion?** Use GitHub Discussions

## Recognition

Contributors will be:
1. Added to AUTHORS.md
2. Mentioned in release notes
3. Added to GitHub contributors

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for helping make OmniVector-Embed production-grade! 🚀
"""
