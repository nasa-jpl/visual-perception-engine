"""Tests for vp_engine.doctor."""

from pathlib import Path

from vp_engine.doctor import main

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_valid_default_config():
    config = ROOT / "configs" / "default.json"
    assert config.exists()
    assert main([str(config)]) == 0


def test_doctor_missing_config():
    assert main([str(ROOT / "configs" / "does-not-exist.json")]) == 1
