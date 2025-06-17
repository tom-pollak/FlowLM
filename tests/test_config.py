"""Unit tests for configuration system."""

import pytest
from omegaconf import OmegaConf
from flowlm.config import (
    FlowLMConfig,
    get_llada_config,
    get_flowlm_config,
)


class TestStructuredConfigs:
    """Test structured configuration classes."""
    
    def test_default_config_creation(self):
        """Test creating a default FlowLMConfig."""
        config = FlowLMConfig()
        
        assert config.model.model_id == "answerdotai/ModernBERT-large"
        assert config.masking.strategy == "LLADA_STRATEGY"
        assert config.training.batch_size == 32
        assert config.logging.wandb.project == "flowlm"
    
    def test_config_modification(self):
        """Test modifying config values."""
        config = FlowLMConfig()
        config.masking.strategy = "FLOWLM_STRATEGY"
        config.masking.min_ratio = 0.3
        
        assert config.masking.strategy == "FLOWLM_STRATEGY"
        assert config.masking.min_ratio == 0.3
    
    def test_structured_config_with_omegaconf(self):
        """Test using structured config with OmegaConf."""
        config = FlowLMConfig()
        structured_config = OmegaConf.structured(config)
        
        assert structured_config.model.model_id == "answerdotai/ModernBERT-large"
        assert structured_config.masking.strategy == "LLADA_STRATEGY"
        assert structured_config.training.learning_rate == 2e-5


class TestPresetConfigs:
    """Test preset configuration functions."""
    
    def test_llada_config(self):
        """Test LLaDA preset configuration."""
        config = get_llada_config()
        
        assert config.masking.strategy == "LLADA_STRATEGY"
        assert config.logging.wandb.name == "llada-baseline"
        assert "llada" in config.logging.wandb.tags
        assert config.logging.save_dir == "checkpoints/llada"
    
    def test_flowlm_config(self):
        """Test FlowLM preset configuration."""
        config = get_flowlm_config()
        
        assert config.masking.strategy == "FLOWLM_STRATEGY"
        assert config.logging.wandb.name == "flowlm-experiment"
        assert "flowlm" in config.logging.wandb.tags
        assert config.logging.save_dir == "checkpoints/flowlm"
    
    def test_preset_configs_with_omegaconf(self):
        """Test preset configs work with OmegaConf.structured."""
        llada_config = OmegaConf.structured(get_llada_config())
        flowlm_config = OmegaConf.structured(get_flowlm_config())
        
        assert llada_config.masking.strategy == "LLADA_STRATEGY"
        assert flowlm_config.masking.strategy == "FLOWLM_STRATEGY"


class TestYAMLConfigMerging:
    """Test YAML configuration merging with structured configs."""
    
    def test_yaml_config_merging(self, tmp_path):
        """Test merging YAML config with structured config."""
        # Create a temporary YAML config
        yaml_content = """
masking:
  strategy: "FLOWLM_STRATEGY"
  min_ratio: 0.2
  
logging:
  wandb:
    name: "test-run"
    tags: ["test"]
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        
        # Load and merge with structured config
        yaml_config = OmegaConf.load(yaml_file)
        structured_config = OmegaConf.structured(FlowLMConfig)
        merged_config = OmegaConf.merge(structured_config, yaml_config)
        
        # Check that YAML values override defaults
        assert merged_config.masking.strategy == "FLOWLM_STRATEGY"
        assert merged_config.masking.min_ratio == 0.2
        assert merged_config.logging.wandb.name == "test-run"
        assert merged_config.logging.wandb.tags == ["test"]
        
        # Check that defaults are preserved
        assert merged_config.model.model_id == "answerdotai/ModernBERT-large"
        assert merged_config.training.batch_size == 32
        assert merged_config.masking.max_ratio == 0.99  # Default value
    
    def test_partial_yaml_config(self, tmp_path):
        """Test YAML config with only some values specified."""
        yaml_content = """
training:
  batch_size: 64
  learning_rate: 1e-4
"""
        yaml_file = tmp_path / "partial_config.yaml"
        yaml_file.write_text(yaml_content)
        
        yaml_config = OmegaConf.load(yaml_file)
        structured_config = OmegaConf.structured(FlowLMConfig)
        merged_config = OmegaConf.merge(structured_config, yaml_config)
        
        # Check overridden values
        assert merged_config.training.batch_size == 64
        assert merged_config.training.learning_rate == 1e-4
        
        # Check preserved defaults
        assert merged_config.training.weight_decay == 0.01
        assert merged_config.masking.strategy == "LLADA_STRATEGY"
        assert merged_config.model.model_id == "answerdotai/ModernBERT-large"


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_strategy_name(self):
        """Test handling of invalid strategy names."""
        config = FlowLMConfig()
        config.masking.strategy = "INVALID_STRATEGY"
        
        # This should be handled by the get_masking_strategy function in train.py
        # Here we just test that the config accepts the string
        assert config.masking.strategy == "INVALID_STRATEGY"
    
    def test_config_types(self):
        """Test that config maintains proper types."""
        config = FlowLMConfig()
        
        assert isinstance(config.training.batch_size, int)
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.masking.min_ratio, float)
        assert isinstance(config.logging.wandb.tags, list)
        assert isinstance(config.dataset.max_length, int)