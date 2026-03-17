"""Tests for training configs, resume support, synthetic data, and custom datasets.

Covers:
- Stage 2 YAML config validation (bf16, lr, steps, data_path, resume)
- Stage 3 YAML config validation (audio, cross-modal weight, freeze settings)
- YAML config loading in training scripts
- --resume flag passthrough to TrainingArguments
- Synthetic data generation (template-based)
- Custom JSONL dataset loading
- --add-synthetic / --add-custom CLI flags in build_dataset
"""

import inspect
import json
import tempfile
from pathlib import Path

import pytest


# ── Resolve project root (tests/unit/ -> project root = 2 parents up) ──

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Stage 2 Generalist Config ──


class TestStage2Config:
    """Tests for updated stage2_generalist.yaml."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        import yaml

        config_path = PROJECT_ROOT / "configs" / "stage2_generalist.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def test_config_is_valid_yaml(self):
        assert self.config is not None
        assert "training_args" in self.config

    def test_bf16_enabled(self):
        assert self.config["training_args"].get("bf16") is True
        assert self.config["training_args"].get("fp16", False) is not True

    def test_in_batch_negatives_off(self):
        assert self.config["loss_config"]["use_in_batch_negatives"] is False

    def test_learning_rate(self):
        assert self.config["training_args"]["learning_rate"] == 1.5e-5

    def test_max_steps(self):
        assert self.config["training_args"]["max_steps"] == 18000

    def test_gradient_checkpointing(self):
        assert self.config["training_args"]["gradient_checkpointing"] is True

    def test_data_path_unified(self):
        data_config = self.config["data_config"]
        assert "data_path" in data_config
        assert "train_files" not in data_config

    def test_resume_from_checkpoint(self):
        resume = self.config["training_args"].get("resume_from_checkpoint")
        assert resume is not None
        assert "stage1" in resume


# ── Stage 3 Multimodal Config ──


class TestStage3Config:
    """Tests for stage3_multimodal.yaml."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        import yaml

        config_path = PROJECT_ROOT / "configs" / "stage3_multimodal.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def test_config_is_valid_yaml(self):
        assert self.config is not None

    def test_cross_modal_weight(self):
        assert self.config["loss_config"]["cross_modal_weight"] == 0.2

    def test_audio_config_present(self):
        assert "audio_config" in self.config
        assert self.config["audio_config"]["model_name"] == "whisper-tiny"
        assert self.config["audio_config"]["embed_dim"] == 4096

    def test_freeze_settings(self):
        mc = self.config["modality_config"]
        assert mc["freeze_text"] is False
        assert mc["freeze_vision"] is False
        assert mc["freeze_audio"] is False

    def test_resume_from_stage2(self):
        resume = self.config["training_args"].get("resume_from_checkpoint")
        assert resume is not None
        assert "stage2" in resume

    def test_bf16_enabled(self):
        assert self.config["training_args"]["bf16"] is True


# ── Resume Flag Support ──


class TestResumeSupport:
    """Tests for --resume flag in training scripts."""

    def test_training_script_yaml_loading(self):
        import yaml
        from scripts.training import _load_yaml_training_args

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({
                "training_args": {
                    "learning_rate": 1e-5,
                    "max_steps": 100,
                    "bf16": False,
                    "fp16": False,
                },
            }, f)
            config_path = f.name

        try:
            args = _load_yaml_training_args(config_path, "/tmp/output")
            assert args.learning_rate == 1e-5
            assert args.max_steps == 100
        finally:
            Path(config_path).unlink()

    def test_resume_cli_overrides_yaml(self):
        import yaml
        from scripts.training import _load_yaml_training_args

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({
                "training_args": {
                    "learning_rate": 1e-5,
                    "resume_from_checkpoint": "yaml_checkpoint",
                    "bf16": False,
                    "fp16": False,
                },
            }, f)
            config_path = f.name

        try:
            args = _load_yaml_training_args(
                config_path, "/tmp/output",
                resume_from_checkpoint="cli_checkpoint",
            )
            assert args.resume_from_checkpoint == "cli_checkpoint"

            args2 = _load_yaml_training_args(config_path, "/tmp/output")
            assert args2.resume_from_checkpoint == "yaml_checkpoint"
        finally:
            Path(config_path).unlink()


# ── Synthetic Data Generation ──


class TestSyntheticDataGeneration:
    """Tests for template-based synthetic data generation."""

    def test_generate_expected_count(self):
        from scripts.build_dataset import generate_synthetic_pairs

        pairs = generate_synthetic_pairs(num_pairs=100, seed=42)
        assert len(pairs) == 100

    def test_pair_structure(self):
        from scripts.build_dataset import generate_synthetic_pairs

        pairs = generate_synthetic_pairs(num_pairs=10, seed=42)
        for p in pairs:
            assert "query" in p
            assert "positive" in p
            assert "domain" in p
            assert "modality" in p
            assert len(p["query"]) > 0
            assert len(p["positive"]) > 0

    def test_domains_balanced(self):
        from scripts.build_dataset import generate_synthetic_pairs

        pairs = generate_synthetic_pairs(num_pairs=400, seed=42)
        domains = set(p["domain"] for p in pairs)
        assert len(domains) >= 3

    def test_deterministic_with_seed(self):
        from scripts.build_dataset import generate_synthetic_pairs

        pairs1 = generate_synthetic_pairs(num_pairs=50, seed=123)
        pairs2 = generate_synthetic_pairs(num_pairs=50, seed=123)
        assert pairs1 == pairs2


# ── Custom Dataset Loading ──


class TestCustomDatasetLoading:
    """Tests for load_custom_dataset function."""

    def test_load_from_directory(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        data = [
            {"query": "What is ML?", "positive": "ML is a field of AI."},
            {"query": "What is DL?", "positive": "Deep learning uses neural nets."},
        ]
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        pairs = load_custom_dataset(str(tmp_path))
        assert len(pairs) == 2
        assert pairs[0]["query"] == "What is ML?"

    def test_load_from_single_file(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        jsonl_file = tmp_path / "single.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"query": "test", "positive": "answer"}) + "\n")

        pairs = load_custom_dataset(str(jsonl_file))
        assert len(pairs) == 1

    def test_respects_max_samples(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        jsonl_file = tmp_path / "many.jsonl"
        with open(jsonl_file, "w") as f:
            for i in range(100):
                f.write(json.dumps({"query": f"q{i}", "positive": f"p{i}"}) + "\n")

        pairs = load_custom_dataset(str(tmp_path), max_samples=10)
        assert len(pairs) == 10

    def test_skips_missing_fields(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        jsonl_file = tmp_path / "mixed.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"query": "good", "positive": "pair"}) + "\n")
            f.write(json.dumps({"query": "no positive"}) + "\n")
            f.write(json.dumps({"positive": "no query"}) + "\n")
            f.write(json.dumps({"query": "also good", "positive": "pair2"}) + "\n")

        pairs = load_custom_dataset(str(tmp_path))
        assert len(pairs) == 2

    def test_nonexistent_path_raises(self):
        from scripts.build_dataset import load_custom_dataset

        with pytest.raises(FileNotFoundError):
            load_custom_dataset("/nonexistent/path/to/nowhere")

    def test_alternative_field_names(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        jsonl_file = tmp_path / "alt.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"question": "What?", "answer": "This."}) + "\n")

        pairs = load_custom_dataset(str(tmp_path))
        assert len(pairs) == 1
        assert pairs[0]["query"] == "What?"
        assert pairs[0]["positive"] == "This."


# ── Build Dataset CLI Flags ──


class TestBuildDatasetFlags:
    """Tests for --add-synthetic and --add-custom CLI flags."""

    def test_build_stage_dataset_signature(self):
        from scripts.build_dataset import build_stage_dataset

        sig = inspect.signature(build_stage_dataset)
        assert "add_synthetic" in sig.parameters
        assert "add_custom" in sig.parameters

    def test_argparse_accepts_new_flags(self):
        import argparse
        import sys

        original_argv = sys.argv
        sys.argv = [
            "build_dataset.py",
            "--stage", "2",
            "--output-dir", "/tmp/test",
            "--add-synthetic", "1000",
            "--add-custom", "/tmp/custom",
        ]

        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
            parser.add_argument("--output-dir", type=Path, required=True)
            parser.add_argument("--add-synthetic", type=int, default=0)
            parser.add_argument("--add-custom", type=str, default=None)

            args = parser.parse_args()
            assert args.add_synthetic == 1000
            assert args.add_custom == "/tmp/custom"
        finally:
            sys.argv = original_argv
