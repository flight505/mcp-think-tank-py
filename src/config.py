#!/usr/bin/env python3
"""
Configuration for MCP Think Tank
Handles environment variables, constants, and settings
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration object with all settings for MCP Think Tank"""
    # Storage paths
    memory_file_path: str
    
    # Feature flags
    use_embeddings: bool
    enable_reflexion: bool
    
    # API Keys (optional)
    anthropic_api_key: Optional[str] = None
    
    # Paths
    project_path: Optional[str] = None
    
    # Local model configuration
    use_local_model: bool
    local_model_path: Optional[str] = None


def get_config() -> Config:
    """
    Load configuration from environment variables or defaults
    
    Returns:
        Config: Configuration object with all settings
    """
    # Directory for storing data, defaults to ~/.mcp-think-tank
    base_dir = os.environ.get(
        "MEMORY_DIR_PATH", 
        os.path.expanduser("~/.mcp-think-tank")
    )
    
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Memory storage file
    memory_file_path = os.environ.get(
        "MEMORY_FILE_PATH", 
        os.path.join(base_dir, "memory.jsonl")
    )
    
    # Feature flags
    use_embeddings = os.environ.get("USE_EMBEDDINGS", "true").lower() == "true"
    enable_reflexion = os.environ.get("ENABLE_REFLEXION", "true").lower() == "true"
    
    # API Keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Project path for file watchers (if applicable)
    project_path = os.environ.get("PROJECT_PATH")
    
    # Local model configuration
    use_local_model = os.environ.get("USE_LOCAL_MODEL", "true").lower() == "true"
    local_model_path = os.environ.get("LOCAL_MODEL_PATH", os.path.join(base_dir, "models/gemma-3-1b"))
    
    # Create and return the config
    return Config(
        memory_file_path=memory_file_path,
        use_embeddings=use_embeddings,
        enable_reflexion=enable_reflexion,
        anthropic_api_key=anthropic_api_key,
        project_path=project_path,
        use_local_model=use_local_model,
        local_model_path=local_model_path
    )


def validate_config(config: Config) -> None:
    """
    Validate that the configuration is valid and all required values are set
    
    Args:
        config: The config object to validate
        
    Raises:
        ValueError: If any required config is missing or invalid
    """
    # Check that memory_file_path is usable
    memory_dir = os.path.dirname(config.memory_file_path)
    if not os.access(memory_dir, os.W_OK):
        raise ValueError(
            f"Memory directory {memory_dir} is not writable. "
            "Set MEMORY_FILE_PATH to a writable location."
        )
    
    # If embeddings are enabled, we'll need to verify dependencies
    # This will be checked when the embedding is actually initialized
    
    # If API keys are required but not set, warn or fail
    if config.use_embeddings and not config.anthropic_api_key:
        # Not a hard failure, but should warn
        print("[WARNING] ANTHROPIC_API_KEY not set, some features may be limited.") 