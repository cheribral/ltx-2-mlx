"""Unit tests for read_hdr_lora_config."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import safetensors.numpy

from ltx_core_mlx.loader.hdr_metadata import HdrLoraConfig, read_hdr_lora_config


def _save_lora_with_metadata(path: Path, metadata: dict[str, str]) -> None:
    """Write a tiny safetensors with the given metadata."""
    tensors = {"dummy.weight": np.zeros((1, 1), dtype=np.float32)}
    safetensors.numpy.save_file(tensors, str(path), metadata=metadata)


class TestReadHdrLoraConfig:
    def test_detects_hdr_transform(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lora.safetensors"
            _save_lora_with_metadata(p, {"hdr_transform": "logc3", "reference_downscale_factor": "2"})
            cfg = read_hdr_lora_config(str(p))
            assert cfg == HdrLoraConfig(hdr_transform="logc3", reference_downscale_factor=2)

    def test_legacy_use_hdr_transform_flag(self):
        """Old metadata format: bool flag without explicit transform name."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lora.safetensors"
            _save_lora_with_metadata(p, {"use_hdr_transform": "true"})
            cfg = read_hdr_lora_config(str(p))
            # Falls back to default 'logc3' when only the flag is set.
            assert cfg is not None
            assert cfg.hdr_transform == "logc3"

    def test_no_hdr_metadata_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lora.safetensors"
            _save_lora_with_metadata(p, {"random_key": "value"})
            assert read_hdr_lora_config(str(p)) is None

    def test_missing_file_returns_none(self):
        assert read_hdr_lora_config("/nonexistent/path/lora.safetensors") is None

    def test_default_downscale_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lora.safetensors"
            _save_lora_with_metadata(p, {"hdr_transform": "logc3"})
            cfg = read_hdr_lora_config(str(p))
            assert cfg is not None
            assert cfg.reference_downscale_factor == 1
