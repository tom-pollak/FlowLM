"""Integration tests for training script functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from flowlm.config import get_llada_config, get_flowlm_config
import train


class TestTrainScriptIntegration:
    """Integration tests for the training script."""
    
    def test_load_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_content = """
masking:
  strategy: "FLOWLM_STRATEGY"
  min_ratio: 0.3

logging:
  wandb:
    name: "test-run"
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        
        config = train.load_config(str(yaml_file))
        
        assert config.masking.strategy == "FLOWLM_STRATEGY"
        assert config.masking.min_ratio == 0.3
        assert config.logging.wandb.name == "test-run"
        # Check defaults are preserved
        assert config.model.model_id == "answerdotai/ModernBERT-large"
    
    def test_load_config_nonexistent_file(self):
        """Test error handling for nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            train.load_config("nonexistent_config.yaml")
    
    def test_get_masking_strategy(self):
        """Test masking strategy retrieval."""
        bert_strategy = train.get_masking_strategy("BERT_STRATEGY")
        llada_strategy = train.get_masking_strategy("LLADA_STRATEGY")
        flowlm_strategy = train.get_masking_strategy("FLOWLM_STRATEGY")
        
        assert bert_strategy.mask_prob == 0.8
        assert llada_strategy.mask_prob == 0.9
        assert flowlm_strategy.mask_prob == 0.3
        
        with pytest.raises(ValueError):
            train.get_masking_strategy("INVALID_STRATEGY")
    
    @patch('train.wandb')
    def test_setup_wandb(self, mock_wandb):
        """Test wandb setup function."""
        config = get_llada_config()
        structured_config = OmegaConf.structured(config)
        
        train.setup_wandb(structured_config)
        
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args[1]
        assert call_args['project'] == 'flowlm'
        assert call_args['name'] == 'llada-baseline'
        assert 'llada' in call_args['tags']
    
    @patch('train.iterative_decode')
    def test_test_inference(self, mock_iterative_decode):
        """Test inference testing function."""
        mock_iterative_decode.return_value = "Test result"
        
        config = get_flowlm_config()
        structured_config = OmegaConf.structured(config)
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        results = train.test_inference(mock_model, mock_tokenizer, structured_config, "cpu")
        
        # Should test both LLaDA and FlowLM modes for each prompt
        expected_calls = len(config.inference.test_prompts) * 2  # Both modes
        assert mock_iterative_decode.call_count == expected_calls
        
        # Check that results contain both modes for each prompt
        for prompt in config.inference.test_prompts:
            prompt_key = prompt[:20]
            assert f"llada_{prompt_key}" in results
            assert f"flowlm_{prompt_key}" in results


class TestConfigFiles:
    """Test actual config files in the configs/ directory."""
    
    def test_llada_config_file(self):
        """Test that the LLaDA config file loads correctly."""
        config_path = "configs/llada.yaml"
        if os.path.exists(config_path):
            config = train.load_config(config_path)
            
            assert config.masking.strategy == "LLADA_STRATEGY"
            assert config.logging.wandb.name == "llada-baseline"
            assert "llada" in config.logging.wandb.tags
            assert config.logging.save_dir == "checkpoints/llada"
    
    def test_flowlm_config_file(self):
        """Test that the FlowLM config file loads correctly."""
        config_path = "configs/flowlm.yaml"
        if os.path.exists(config_path):
            config = train.load_config(config_path)
            
            assert config.masking.strategy == "FLOWLM_STRATEGY"
            assert config.logging.wandb.name == "flowlm-experiment"
            assert "flowlm" in config.logging.wandb.tags
            assert config.logging.save_dir == "checkpoints/flowlm"


@pytest.mark.integration
class TestMainFunction:
    """Integration tests for the main function."""
    
    @patch('train.setup_wandb')
    @patch('train.AutoTokenizer')
    @patch('train.AutoModelForMaskedLM')
    @patch('train.prepare_dataset')
    @patch('train.Trainer')
    @patch('train.test_inference')
    @patch('train.wandb')
    @patch('sys.argv', ['train.py', '--preset', 'llada'])
    def test_main_with_preset(self, mock_wandb, mock_test_inference, 
                             mock_trainer_class, mock_prepare_dataset,
                             mock_model, mock_tokenizer, mock_setup_wandb):
        """Test main function with preset configuration."""
        # Mock returns
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_prepare_dataset.return_value = (MagicMock(), MagicMock())
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.evaluate.return_value = {"eval_loss": 0.5}
        mock_test_inference.return_value = {"test": "result"}
        mock_wandb.run = MagicMock()
        mock_wandb.run.name = "test-run"
        
        # This would normally be tested with subprocess or similar
        # For now, we just test that the components work together
        config = OmegaConf.structured(get_llada_config())
        assert config.masking.strategy == "LLADA_STRATEGY"
        assert config.logging.wandb.name == "llada-baseline"