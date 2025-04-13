#!/usr/bin/env python3
"""
Setup script for MCP Think Tank with local model installation option
"""
import os
import sys
import argparse
from setuptools import setup, find_packages, Command

# Try to import the model downloading function
try:
    from src.tools.tasks import download_local_model
    CAN_DOWNLOAD_MODEL = True
except ImportError:
    CAN_DOWNLOAD_MODEL = False

class InstallLocalModelCommand(Command):
    """Custom command to install the local LLM model"""
    description = "Download and install the local LLM model for task parsing"
    user_options = [
        ('force', 'f', 'Force re-download even if model exists'),
    ]
    
    def initialize_options(self):
        self.force = False
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Run the command"""
        if not CAN_DOWNLOAD_MODEL:
            print("Error: Cannot download model - required packages not installed.")
            print("Please install the package with extra dependencies first:")
            print("pip install -e .[local-llm]")
            return
            
        print("\n===== MCP Think Tank: Local Model Installation =====")
        print("\nThis will download the Gemma 3 1B model (~900MB) for local task parsing.")
        print("The model will be used as a fallback when Anthropic API is not available.")
        
        if not self.force:
            answer = input("\nProceed with download? [y/N] ").lower()
            if answer not in ('y', 'yes'):
                print("Download cancelled.")
                return
        
        print("\nDownloading model...")
        success = download_local_model(force=self.force)
        
        if success:
            print("\nModel downloaded successfully!")
            print("The local model will be used as a fallback when Anthropic API is not available.")
        else:
            print("\nError downloading model.")
            print("You can try again later with:")
            print("  python setup.py install_local_model")
            print("Or install required dependencies with:")
            print("  pip install -e .[local-llm]")

if __name__ == "__main__":
    setup(
        name="mcp-think-tank",
        version="0.1.0",
        description="MCP Think Tank - Advanced knowledge graph and reasoning tools",
        author="MCP Think Tank Team",
        author_email="example@example.com",
        packages=find_packages(),
        install_requires=[
            "fastmcp>=2.0.0",
            "pydantic>=2.0.0",
            "tqdm",
        ],
        extras_require={
            "anthropic": ["anthropic>=0.5.0"],
            "local-llm": [
                "huggingface_hub",
                "torch",
                "transformers",
                "accelerate",
            ],
            "all": [
                "anthropic>=0.5.0",
                "huggingface_hub",
                "torch",
                "transformers",
                "accelerate",
                "sentence-transformers",
            ],
        },
        cmdclass={
            'install_local_model': InstallLocalModelCommand,
        },
        entry_points={
            'console_scripts': [
                'mcp-think-tank=src.cli:main',
            ],
        },
    ) 