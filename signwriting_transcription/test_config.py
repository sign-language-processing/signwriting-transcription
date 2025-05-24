# Test to validate config.yaml file

import os
import yaml
import pytest


def test_config_yaml_is_valid():
    """Test that config.yaml is a valid YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as stream:
        try:
            config = yaml.safe_load(stream)
            assert config is not None, "Config file should not be empty"
        except yaml.YAMLError as exc:
            pytest.fail(f"Invalid YAML syntax in config.yaml: {exc}")


def test_config_has_required_sections():
    """Test that config.yaml contains required configuration sections."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        assert section in config, f"Required section '{section}' missing from config.yaml"


def test_config_model_section():
    """Test that model section contains required fields."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    model_config = config.get('model', {})
    required_fields = ['type', 'backbone_type', 'pretrained_backbone', 'feat_dim']

    for field in required_fields:
        assert field in model_config, f"Required field '{field}' missing from model section"
