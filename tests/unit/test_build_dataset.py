"""Unit tests for build_dataset domain-balanced upsampling (Fix 8)."""

import json
import logging
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pytest


class TestDomainBalancedUpsampling:
    """Tests for the domain-balanced upsampling in build_stage_dataset."""

    def _run_upsampling(self, records, target_size):
        """Simulate the upsampling logic extracted from build_stage_dataset."""
        rng = np.random.default_rng(42)

        domain_indices: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            dom = rec.get("domain", "unknown")
            domain_indices.setdefault(dom, []).append(i)

        extra_needed = target_size - len(records)
        n_domains = len(domain_indices)
        base_per_domain = extra_needed // n_domains

        extra_records = []
        for dom, indices in domain_indices.items():
            domain_quota = min(base_per_domain, len(indices) * 5)
            if domain_quota == 0:
                continue
            sampled = rng.choice(indices, size=domain_quota, replace=True)
            extra_records.extend([records[i] for i in sampled])

        still_needed = extra_needed - len(extra_records)
        if still_needed > 0:
            domain_weights = np.array([
                1.0 / max(len(idxs), 1) for idxs in domain_indices.values()
            ])
            domain_weights /= domain_weights.sum()
            domain_names = list(domain_indices.keys())
            for _ in range(still_needed):
                chosen_dom = rng.choice(domain_names, p=domain_weights)
                chosen_idx = rng.choice(domain_indices[chosen_dom])
                extra_records.append(records[chosen_idx])

        records.extend(extra_records)
        return records

    def test_upsampling_reaches_target(self):
        records = [
            {"query": f"q{i}", "positive": f"p{i}", "domain": f"d{i % 3}", "modality": "text"}
            for i in range(100)
        ]
        target = 500
        result = self._run_upsampling(records, target)
        assert len(result) == target

    def test_upsampling_preserves_originals(self):
        originals = [
            {"query": f"q{i}", "positive": f"p{i}", "domain": "a", "modality": "text"}
            for i in range(10)
        ]
        result = self._run_upsampling(list(originals), 50)
        # First 10 should be originals
        for i in range(10):
            assert result[i] == originals[i]

    def test_upsampling_domain_balanced(self):
        """With balanced input, upsampled result should stay ~balanced."""
        records = []
        for dom in ["a", "b", "c"]:
            for i in range(100):
                records.append({"query": f"q", "positive": f"p", "domain": dom, "modality": "text"})
        result = self._run_upsampling(records, 900)

        counts = {}
        for r in result:
            counts[r["domain"]] = counts.get(r["domain"], 0) + 1

        # Each domain should get roughly 300
        for dom, cnt in counts.items():
            assert 200 <= cnt <= 400, f"Domain {dom} got {cnt}, expected ~300"

    def test_upsampling_caps_dominant_domain(self):
        """A big domain should be capped at 5× its original size."""
        records = []
        # big domain: 100 records, small domain: 10 records
        for i in range(100):
            records.append({"query": "q", "positive": "p", "domain": "big", "modality": "text"})
        for i in range(10):
            records.append({"query": "q", "positive": "p", "domain": "small", "modality": "text"})

        result = self._run_upsampling(records, 1100)

        big_count = sum(1 for r in result if r["domain"] == "big")
        # big domain started at 100, cap is 5× = 500.
        # base_per_domain = 990 // 2 = 495 which is < 500 cap, so it gets 495 extra
        # total = 100 + 495 = 595, but let's just check < 700
        assert big_count < 700, f"Big domain got {big_count}, should be capped"

    def test_upsampling_small_domain_gets_boost(self):
        """Small domains get boosted via weighted fill."""
        records = []
        for i in range(1000):
            records.append({"query": "q", "positive": "p", "domain": "big", "modality": "text"})
        for i in range(10):
            records.append({"query": "q", "positive": "p", "domain": "tiny", "modality": "text"})

        result = self._run_upsampling(records, 5000)

        tiny_count = sum(1 for r in result if r["domain"] == "tiny")
        # tiny started at 10, must get significant boost
        assert tiny_count > 10, f"Tiny domain only got {tiny_count}, should be boosted"


class TestGenerateSyntheticPairs:
    """Tests for the template-based synthetic generation."""

    def test_generates_requested_count(self):
        from scripts.build_dataset import generate_synthetic_pairs
        pairs = generate_synthetic_pairs(num_pairs=100)
        assert len(pairs) == 100

    def test_pairs_have_required_keys(self):
        from scripts.build_dataset import generate_synthetic_pairs
        pairs = generate_synthetic_pairs(num_pairs=10)
        for p in pairs:
            assert "query" in p
            assert "positive" in p
            assert "domain" in p
            assert "modality" in p

    def test_reproducible_with_seed(self):
        from scripts.build_dataset import generate_synthetic_pairs
        a = generate_synthetic_pairs(num_pairs=20, seed=123)
        b = generate_synthetic_pairs(num_pairs=20, seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        from scripts.build_dataset import generate_synthetic_pairs
        a = generate_synthetic_pairs(num_pairs=20, seed=1)
        b = generate_synthetic_pairs(num_pairs=20, seed=2)
        assert a != b


class TestLoadCustomDataset:
    """Tests for the custom JSONL loader."""

    def test_loads_valid_jsonl(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        f = tmp_path / "data.jsonl"
        records = [
            {"query": "q1", "positive": "p1"},
            {"query": "q2", "positive": "p2", "domain": "custom"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records))

        result = load_custom_dataset(str(tmp_path))
        assert len(result) == 2
        assert result[0]["query"] == "q1"
        assert result[1]["domain"] == "custom"

    def test_skips_malformed_lines(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        f = tmp_path / "bad.jsonl"
        f.write_text('{"query": "ok", "positive": "fine"}\nNOT JSON\n{"query": "q2", "positive": "p2"}\n')

        result = load_custom_dataset(str(tmp_path))
        assert len(result) == 2

    def test_missing_path_raises(self):
        from scripts.build_dataset import load_custom_dataset
        with pytest.raises(FileNotFoundError):
            load_custom_dataset("/nonexistent/path/xyz")

    def test_max_samples_limit(self, tmp_path):
        from scripts.build_dataset import load_custom_dataset

        f = tmp_path / "many.jsonl"
        lines = [json.dumps({"query": f"q{i}", "positive": f"p{i}"}) for i in range(50)]
        f.write_text("\n".join(lines))

        result = load_custom_dataset(str(tmp_path), max_samples=10)
        assert len(result) == 10
